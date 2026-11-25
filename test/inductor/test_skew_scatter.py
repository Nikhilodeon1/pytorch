# Owner(s): ["module: inductor"]

import glob
import itertools
import os
import re
import types
import unittest
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch._inductor.fx_passes import skew_scatter as skew_scatter_pass
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache, run_and_get_code
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


def _index(n, r, hot, seed=0, dtype=torch.int64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.randint(0, r, (n,), generator=g, device="cuda")
    mask = torch.rand(n, generator=g, device="cuda") < hot
    return torch.where(mask, torch.full_like(idx, r // 3), idx).to(dtype)


def _index_add(src, idx, r, out_dtype):
    out = torch.zeros(r, src.shape[1], device=src.device, dtype=out_dtype)
    return out.index_add(0, idx, src.to(out_dtype))


def _index_put(src, idx, r, out_dtype):
    out = src.new_zeros(r, src.shape[1], dtype=out_dtype)
    return out.index_put((idx,), src.to(out_dtype), accumulate=True)


def _scatter_add(src, idx, r, out_dtype):
    out = torch.zeros(r, src.shape[1], device=src.device, dtype=out_dtype)
    index = idx.long().unsqueeze(-1).expand(-1, src.shape[1])
    return out.scatter_add(0, index, src.to(out_dtype))


OPS = {"index_add": _index_add, "index_put": _index_put, "scatter_add": _scatter_add}


def _picked_modes():
    """MODE of the config each tuned ``_skew_scatter_kernel`` launches with."""
    seen = {}
    for mod in PyCodeCache.modules:
        for name, k in vars(mod).items():
            launchers = getattr(k, "launchers", None)
            if name.startswith("_skew_scatter_kernel") and launchers and len(launchers) == 1:
                seen[id(k)] = launchers[0].config.kwargs["MODE"]
    return sorted(seen.values())


def _bound(src, idx, r, base=None):
    """Twice the max error of eager fp32 atomics vs float64 (plus a floor): the
    flat kernel uses the same atomics, so hot rows carry the same error."""
    out = torch.zeros(r, src.shape[1], device="cuda") if base is None else base.clone()
    ref = out.double().index_add(0, idx, src.double())
    err = (out.index_add(0, idx, src.float()).double() - ref).abs().max().item()
    return 2 * err + 1e-4


@unittest.skipIf(not HAS_CUDA_AND_TRITON or torch.version.hip, "needs CUDA")
class SkewScatterTest(TestCase):
    def setUp(self):
        super().setUp()
        patch = config.patch(skew_scatter_enabled=True, skew_scatter_min_index_size=1024)
        patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)
        counters.clear()

    def _errors(self, fn, src, idx, r, out_dtype, base=None):
        truth = fn(src.double(), idx, r, torch.float64)
        if base is not None:
            truth = truth + base.double()

        def run(f):
            out = f(src, idx, r, out_dtype)
            return out + base if base is not None else out

        with config.patch(skew_scatter_enabled=False):
            ref = run(torch.compile(fn, dynamic=False))
        torch._dynamo.reset()
        got, (code,) = run_and_get_code(run, torch.compile(fn, dynamic=False))
        ref_err = (ref.double() - truth).abs()
        got_err = (got.double() - truth).abs()
        return ref_err, got_err, code

    def _check(self, op, n, r, d, hot, src_dtype, out_dtype, idx_dtype=torch.int64):
        idx = _index(n, r, hot, dtype=idx_dtype)
        src = torch.randn(n, d + 8, device="cuda", dtype=src_dtype)[:, :d]
        ref_err, got_err, code = self._errors(OPS[op], src, idx, r, out_dtype)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 1)
        self.assertIn("_scatter_kernel", code)
        self.assertNotIn("skew_scatter_rows", code)
        self.assertNotIn("clone", code)
        tol = 1e-3 if out_dtype == torch.float32 else 3e-2
        bound = max(2 * ref_err.max().item(), tol * max(1.0, hot * n / 100))
        self.assertLessEqual(got_err.max().item(), bound)
        self.assertLessEqual(got_err.mean().item(), max(2 * ref_err.mean().item(), tol))

    @parametrize("op", list(OPS))
    @parametrize("hot", [0.0, 0.5])
    def test_fp32(self, op, hot):
        self._check(op, 200_000, 5000, 64, hot, torch.float32, torch.float32)

    @parametrize("op", list(OPS))
    def test_bf16_src_fp32_out_cast_folded(self, op):
        self._check(op, 300_000, 40_000, 64, 0.5, torch.bfloat16, torch.float32)

    def test_zipf_like_many_hot_rows(self):
        n, r, d = 400_000, 20_000, 32
        g = torch.Generator(device="cuda").manual_seed(0)
        p = torch.arange(1, r + 1, device="cuda", dtype=torch.float64).pow(-1.0)
        idx = torch.multinomial(p, n, replacement=True, generator=g)
        src = torch.randn(n, d, device="cuda")
        ref_err, got_err, _ = self._errors(_index_add, src, idx, r, torch.float32)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 1)
        self.assertLessEqual(got_err.max().item(), max(2 * ref_err.max().item(), 1e-3))

    @unittest.skipIf(not SM90OrLater, "bf16 atomic_add needs sm90")
    def test_bf16_out_large_destination_left_to_inductor(self):
        idx = _index(100_000, 200_000, 0.3)
        src = torch.randn(100_000, 128, device="cuda", dtype=torch.bfloat16)
        torch.compile(_index_add)(src, idx, 200_000, torch.bfloat16)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)

    @unittest.skipIf(not SM90OrLater, "sm80 rewrites fp32 outputs only")
    def test_fp16_out_small_destination(self):
        self._check("index_put", 200_000, 50, 64, 0.3, torch.float16, torch.float16)

    def test_small_destination_partitions(self):
        self._check("index_add", 500_000, 64, 8, 0.2, torch.float32, torch.float32)

    def test_wide_rows_int32_index(self):
        self._check("index_add", 100_000, 3000, 256, 0.4, torch.float32, torch.float32, torch.int32)

    def test_nonzero_base(self):
        def fn(base, src, idx):
            return base.index_add(0, idx, src)

        n, r, d = 200_000, 3000, 64
        idx = _index(n, r, 0.5)
        src = torch.randn(n, d, device="cuda")
        base = torch.randn(r, d, device="cuda")
        base_copy = base.clone()
        truth = base.double().index_add(0, idx, src.double())
        got, (code,) = run_and_get_code(torch.compile(fn), base, src, idx)
        self.assertIn("_scatter_kernel", code)
        self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r, base))
        self.assertEqual(base, base_copy)

    def test_backward_of_index_select(self):
        n, r, d = 200_000, 3000, 64
        idx = _index(n, r, 0.5)
        w = torch.randn(r, d, device="cuda", requires_grad=True)
        g = torch.randn(n, d, device="cuda")

        def fn(w, idx):
            return w.index_select(0, idx)

        torch.compile(fn)(w, idx).backward(g)
        truth = torch.zeros(r, d, device="cuda", dtype=torch.float64).index_add(
            0, idx, g.double()
        )
        self.assertGreaterEqual(counters["inductor"]["skew_scatter_applied"], 1)
        self.assertLessEqual((w.grad.double() - truth).abs().max().item(), _bound(g, idx, r))

    def test_dynamic_shapes_use_the_opaque_op(self):
        fn = torch.compile(_index_add, dynamic=True)
        for n in (100_000, 150_000):
            idx = _index(n, 4000, 0.5)
            src = torch.randn(n, 32, device="cuda")
            got, (code,) = run_and_get_code(fn, src, idx, 4000, torch.float32)
            truth = _index_add(src.double(), idx, 4000, torch.float64)
            self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, 4000))
            self.assertIn("skew_scatter_rows", code)
        self.assertGreaterEqual(counters["inductor"]["skew_scatter_applied"], 1)

    # Without a first call on real data, large destinations take the flat kernel on
    # sm80 and are left to Inductor on sm90+; small destinations are privatized.
    @parametrize("r", [64, 5000])
    @config.patch(cpp_wrapper=True)
    def test_cpp_wrapper(self, r):
        idx = _index(200_000, r, 0.5)
        src = torch.randn(200_000, 64, device="cuda", dtype=torch.bfloat16)
        got = torch.compile(_index_put)(src, idx, r, torch.float32)
        truth = _index_put(src.double(), idx, r, torch.float64)
        self.assertLessEqual((got.double() - truth).abs().max().item(), 1e-2)
        applied = int(r == 64 or not SM90OrLater)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], applied)

    @parametrize("r", [64, 5000])
    def test_aot_inductor(self, r):
        class M(torch.nn.Module):
            def forward(self, src, idx):
                return _index_add(src, idx, r, torch.float32)

        idx = _index(200_000, r, 0.5)
        src = torch.randn(200_000, 64, device="cuda")
        ep = torch.export.export(M(), (src, idx))
        path = torch._inductor.aoti_compile_and_package(ep)
        got = torch._inductor.aoti_load_package(path)(src, idx)
        truth = _index_add(src.double(), idx, r, torch.float64)
        self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r))
        applied = int(r == 64 or not SM90OrLater)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], applied)

    def test_cudagraphs(self):
        fn = torch.compile(_index_add, mode="reduce-overhead", dynamic=False)
        idx = _index(200_000, 5000, 0.5)
        src = torch.randn(200_000, 64, device="cuda")
        truth = _index_add(src.double(), idx, 5000, torch.float64)
        bound = _bound(src, idx, 5000)
        for _ in range(3):
            got = fn(src, idx, 5000, torch.float32)
            self.assertLessEqual((got.double() - truth).abs().max().item(), bound)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 1)

    def test_gates(self):
        idx = _index(2000, 100, 0.5)
        src = torch.randn(2000, 16, device="cuda")
        with config.patch(skew_scatter_min_index_size=4096):
            torch.compile(_index_add)(src, idx, 100, torch.float32)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)
        with config.patch(skew_scatter_enabled=False):
            torch.compile(_index_put)(src, idx, 100, torch.float32)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)
        src3 = torch.randn(2000, 4, 4, device="cuda")
        torch.compile(lambda s, i: s.new_zeros(100, 4, 4).index_add(0, i, s))(src3, idx)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)

    @parametrize("dist", ["uniform", "hot", "sorted", "same", "negative"])
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_tile_kernel(self, dist, dtype):
        # The per-tile duplicate detection kernel on any CUDA arch, called directly.
        from torch._inductor.kernel.skew_scatter import _tile_scatter

        n, r, d = 300_007, 3000, 48
        idx = _index(n, r, 0.5 if dist == "hot" else 0.0)
        if dist == "sorted":
            idx = idx.sort().values
        elif dist == "same":
            idx = torch.full_like(idx, 7)
        elif dist == "negative":
            idx = idx - r
        src = torch.randn(n, d + 8, device="cuda", dtype=dtype)[:, :d]
        out = torch.zeros(r, d, device="cuda", dtype=torch.float32)
        _tile_scatter(out, idx, src)
        truth = torch.zeros(r, d, device="cuda", dtype=torch.float64).index_add(
            0, idx % r, src.double()
        )
        bound = _bound(src, idx % r, r)
        self.assertLessEqual((out.double() - truth).abs().max().item(), bound)

    def test_output_code_does_not_depend_on_index_values(self):
        # The kernel choice is made at run time, on the first call.
        n, r, d = 200_000, 5000, 64
        src = torch.randn(n, d, device="cuda")
        codes = []
        for hot in (0.0, 0.9):
            torch._dynamo.reset()
            idx = _index(n, r, hot)
            with mock.patch.object(skew_scatter_pass, "_sites", itertools.count()):
                got, (code,) = run_and_get_code(
                    torch.compile(_index_add, dynamic=False), src, idx, r, torch.float32
                )
            truth = _index_add(src.double(), idx, r, torch.float64)
            self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r))
            # same code apart from the AOT id and the cache dir
            codes.append(re.sub(r"/tmp[^/'\"]*", "", code.split("\n", 1)[1]))
        self.assertEqual(codes[0], codes[1])

    def test_fused_src_is_left_to_inductor(self):
        # Inductor's scatter computes a pointwise src in place; a user kernel would
        # have to materialize it first.
        idx = _index(200_000, 5000, 0.5)
        src = torch.randn(200_000, 64, device="cuda")
        fn = torch.compile(lambda s, i: _index_add(s * 2, i, 5000, torch.float32))
        got = fn(src, idx)
        truth = _index_add(src.double() * 2, idx, 5000, torch.float64)
        self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src * 2, idx, 5000))
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)

    @unittest.skipIf(not SM90OrLater, "sm90+")
    @config.patch({"triton.autotune_at_compile_time": True})
    def test_sm90_without_first_call_only_small_destinations(self):
        idx = _index(500_000, 5000, 0.5)
        src = torch.randn(500_000, 64, device="cuda")
        torch.compile(_index_add)(src, idx, 5000, torch.float32)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)
        torch.compile(_index_add)(src[:, :8], idx % 64, 64, torch.float32)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 1)

    @unittest.skipIf(
        not hasattr(config, "sorted_segment_scatter_enabled"),
        "needs fx_passes/sorted_segment_scatter.py",
    )
    def test_leaves_small_tables_to_sorted_segment_scatter(self):
        n, r, d = 1_100_000, 1000, 64
        idx = _index(n, r, 0.5)
        src = torch.randn(n, d, device="cuda")
        with config.patch(sorted_segment_scatter_enabled=True):
            got = torch.compile(_index_put)(src, idx, r, torch.float32)
        truth = _index_put(src.double(), idx, r, torch.float64)
        self.assertLessEqual((got.double() - truth).abs().max().item(), 1e-2)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 0)
        self.assertEqual(counters["inductor"]["sorted_segment_scatter_applied"], 1)


    def test_autotune_picks_plain_scatter_for_uniform_indices(self):
        n, r, d = 2_000_000, 50_000, 64
        src = torch.randn(n, d, device="cuda")
        for hot in (0.0, 0.5):
            self._reset()
            idx = _index(n, r, hot)
            got = torch.compile(_index_add, dynamic=False)(src, idx, r, torch.float32)
            truth = _index_add(src.double(), idx, r, torch.float64)
            self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r))
            (mode,) = _picked_modes()
            if hot:
                self.assertNotEqual(mode, 0)
            else:
                self.assertEqual(mode, 0)

    def test_call_sites_are_tuned_separately(self):
        # Two scatters of the same shapes in one graph: one uniform, one hot.
        n, r, d = 2_000_000, 50_000, 64
        src = torch.randn(n, d, device="cuda")
        uniform, hot = _index(n, r, 0.0), _index(n, r, 0.5, seed=1)

        def fn(src, i, j):
            return _index_add(src, i, r, torch.float32), _index_add(src, j, r, torch.float32)

        self._reset()
        a, b = torch.compile(fn, dynamic=False)(src, uniform, hot)
        self.assertEqual(counters["inductor"]["skew_scatter_applied"], 2)
        for got, idx in ((a, uniform), (b, hot)):
            truth = _index_add(src.double(), idx, r, torch.float64)
            self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r))
        modes = _picked_modes()
        self.assertEqual(len(modes), 2)
        self.assertEqual(modes[0], 0)
        self.assertNotEqual(modes[1], 0)

    @parametrize("with_base", [False, True])
    def test_autotune_restores_output_between_trials(self, with_base):
        # The first call times every config on the real output buffer; a trial
        # that is not undone would add the scatter once per benchmark run.
        n, r, d = 1_000_000, 20_000, 32
        idx = _index(n, r, 0.5)
        src = torch.randn(n, d, device="cuda")
        base = torch.randn(r, d, device="cuda")
        self._reset()
        if with_base:
            got = torch.compile(lambda b, s, i: b.index_add(0, i, s))(base, src, idx)
            truth = base.double().index_add(0, idx, src.double())
        else:
            got = torch.compile(_index_add)(src, idx, r, torch.float32)
            truth = _index_add(src.double(), idx, r, torch.float64)
        self.assertEqual(len(_picked_modes()), 1)
        self.assertLessEqual(
            (got.double() - truth).abs().max().item(), _bound(src, idx, r, base if with_base else None)
        )

    @config.patch(
        force_disable_caches=False,
        fx_graph_cache=True,
        fx_graph_remote_cache=False,
        autotune_local_cache=True,
        autotune_remote_cache=False,
    )
    def test_pick_is_not_reused_across_processes(self):
        # A uniform-index run, then a hot-index run of the same graph in a "new
        # process" (in-memory caches cleared, on-disk caches kept): the compiled
        # graph comes from the cache, but the kernel is tuned again on the hot
        # indices instead of reusing the uniform run's pick.
        n, r, d = 2_000_000, 50_000, 64
        src = torch.randn(n, d, device="cuda")
        with fresh_cache():
            for hot, uniform_pick in ((0.0, True), (0.5, False)):
                self._reset()
                idx = _index(n, r, hot)
                got = torch.compile(_index_add, dynamic=False)(src, idx, r, torch.float32)
                truth = _index_add(src.double(), idx, r, torch.float64)
                self.assertLessEqual((got.double() - truth).abs().max().item(), _bound(src, idx, r))
                (mode,) = _picked_modes()
                self.assertEqual(mode == 0, uniform_pick)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            for path in glob.glob(os.path.join(cache_dir(), "**", "*.best_config"), recursive=True):
                with open(path) as f:
                    self.assertNotIn('"MODE"', f.read())

    def test_flat_config_is_the_default(self):
        # The pick keeps the best flat config unless another one is at least 10%
        # faster, and never comes from the autotune caches.
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        n, r, d = 200_000, 5000, 64
        src = torch.randn(n, d, device="cuda")
        _, (code,) = run_and_get_code(torch.compile(_index_add), src, _index(n, r, 0.0), r, torch.float32)
        for meta in ("'autotune_default_margin': 0.1", "'autotune_local_cache': False", "'autotune_remote_cache': False"):
            self.assertIn(meta, code)
        tuner = types.SimpleNamespace(inductor_meta={"autotune_default_margin": 0.1, "autotune_default_configs": 2})
        self.assertEqual(CachingAutotuner._pick_timing(tuner, 1, 1.0), 0.9)
        self.assertEqual(CachingAutotuner._pick_timing(tuner, 2, 0.95), 0.95)

    def _reset(self):
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()


instantiate_parametrized_tests(SkewScatterTest)


if __name__ == "__main__":
    run_tests()
