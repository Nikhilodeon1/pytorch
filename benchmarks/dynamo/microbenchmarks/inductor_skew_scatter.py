"""Benchmark Inductor's row scatter-adds with and without `skew_scatter_enabled`.

Shapes are a synthetic sweep over the index size n, destination rows r and row
width d, from small destinations (r * d <= 4096) to large tables, with fp32 and
bf16 sources and outputs. Each case is run with uniform, Zipf(s) and hot-row (a
fraction h of the indices on one row) index distributions. For every case it
reports the median time of the compiled function with the flag off and on
(`triton.testing.do_bench`, L2 flushed before each call, off/on interleaved over
`--rounds`; each arm is compiled on empty caches, so both autotune on that
distribution's first call), the eager time, and the error of both compiled outputs against a
float64 reference, relative to the reference's magnitude: max |err| / max |ref|
and mean |err| / mean |ref|.

`--startup` instead measures the first call (compile, and for the flag-on arm the
first-call autotuning) of one case per distribution, cold (empty caches) and
warm (compiled graph and kernel binaries cached, as in a new process).

    python benchmarks/dynamo/microbenchmarks/inductor_skew_scatter.py [--cases a,b] [--output-json out.json]
    python benchmarks/dynamo/microbenchmarks/inductor_skew_scatter.py --startup --cases idxadd_n8M_r50K_d64_bf16
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch._dynamo
from torch._dynamo.utils import compilation_time_metrics
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import fresh_cache
from triton.testing import do_bench


UNIFORM = ("uniform", None)
ZIPF08, ZIPF10, ZIPF12 = ("zipf", 0.8), ("zipf", 1.0), ("zipf", 1.2)
HOT05, HOT10, HOT50, HOT70 = ("hot", 0.05), ("hot", 0.1), ("hot", 0.5), ("hot", 0.7)

K, M = 1 << 10, 1 << 20
bf16, fp32 = torch.bfloat16, torch.float32

ALL = [UNIFORM, ZIPF08, ZIPF10, ZIPF12, HOT10, HOT50]

# name, op, n, r, d, src_width (> d: src is a strided slice), src_dtype, out_dtype, distributions
CASES = [
    ("idxadd_n2M_r256_d16", "index_add", 2 * M, 256, 16, 16, fp32, fp32, ALL),
    ("idxput_n8M_r64_d64_bf16", "index_put", 8 * M, 64, 64, 128, bf16, fp32, ALL),
    ("idxadd_n512K_r50K_d128", "index_add", 512 * K, 50 * K, 128, 128, fp32, fp32, ALL),
    ("idxadd_n2M_r50K_d64_bf16", "index_add", 2 * M, 50 * K, 64, 128, bf16, fp32, [*ALL, HOT05]),
    ("idxadd_n8M_r50K_d64_bf16", "index_add", 8 * M, 50 * K, 64, 128, bf16, fp32, ALL),
    ("idxadd_n32M_r50K_d64_bf16", "index_add", 32 * M, 50 * K, 64, 64, bf16, fp32, ALL),
    ("idxput_n2M_r2K_d16", "index_put", 2 * M, 2 * K, 16, 16, fp32, fp32, ALL),
    ("idxput_n8M_r2K_d64_bf16", "index_put", 8 * M, 2 * K, 64, 128, bf16, fp32, ALL),
    ("scatadd_n2M_r2K_d256", "scatter_add", 2 * M, 2 * K, 256, 256, fp32, fp32, [*ALL, HOT70]),
    ("scatadd_n8M_r50K_d256", "scatter_add", 8 * M, 50 * K, 256, 256, fp32, fp32, ALL),
    ("idxadd_n8M_r1M_d128", "index_add", 8 * M, 1 * M, 128, 128, fp32, fp32, ALL),
    ("idxadd_n2M_r1M_d256_bf16", "index_add", 2 * M, 1 * M, 256, 256, bf16, fp32, ALL),
    # bf16 output into a large destination: not rewritten (same code in both arms)
    ("scatadd_n8M_r8M_d128_bf16", "scatter_add", 8 * M, 8 * M, 128, 128, bf16, bf16, [UNIFORM, ZIPF10, HOT50]),
]


def make_index(n: int, r: int, dist: tuple[str, float | None]) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    kind, p = dist
    idx = torch.randint(0, r, (n,), generator=g, device="cuda")
    if kind == "hot":
        hot = torch.rand(n, generator=g, device="cuda") < p
        idx = torch.where(hot, torch.full_like(idx, r - 1), idx)
    elif kind == "zipf":
        w = torch.arange(1, r + 1, device="cuda", dtype=torch.float64).pow(-p)
        cdf = torch.cumsum(w / w.sum(), 0)
        u = torch.rand(n, generator=g, device="cuda", dtype=torch.float64)
        rank = torch.searchsorted(cdf, u).clamp_(max=r - 1)
        idx = torch.randperm(r, generator=g, device="cuda")[rank]
    return idx


def make_fn(op: str) -> Callable[..., torch.Tensor]:
    def fn(src: torch.Tensor, idx: torch.Tensor, r: int, out_dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros(r, src.shape[1], device=src.device, dtype=out_dtype)
        src = src.to(out_dtype)
        if op == "index_add":
            return out.index_add(0, idx, src)
        if op == "index_put":
            return out.index_put((idx,), src, accumulate=True)
        return out.scatter_add(0, idx[:, None].expand(-1, src.shape[1]), src)

    return fn


def fresh(fn: Callable, tag: str) -> Callable:
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return type(fn)(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


def reference(src: torch.Tensor, idx: torch.Tensor, r: int) -> torch.Tensor:
    out = torch.zeros(r, src.shape[1], device=src.device, dtype=torch.float64)
    step = 1 << 22
    for i in range(0, idx.numel(), step):
        out.index_add_(0, idx[i : i + step], src[i : i + step].double())
    return out


def errors(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """max |err| / max |ref| and mean |err| / mean |ref|."""
    e = (x.double() - ref).abs()
    a = ref.abs()
    return (e.max() / a.max().clamp_min(1e-30)).item(), (e.mean() / a.mean().clamp_min(1e-30)).item()


NO_REMOTE_CACHES = {"fx_graph_remote_cache": False, "autotune_remote_cache": False}


def compile_arm(op: str, on: bool, **options) -> Callable:
    return torch.compile(fresh(make_fn(op), "on" if on else "off"), dynamic=False, options={"skew_scatter_enabled": on, **options})


def picked_configs() -> list[str]:
    """Config each first-call-tuned skew scatter kernel of the loaded modules runs."""
    seen = {}
    for mod in PyCodeCache.modules:
        for name, k in vars(mod).items():
            launchers = getattr(k, "launchers", None)
            if name.startswith("_skew_scatter_kernel") and launchers and len(launchers) == 1:
                kw = launchers[0].config.kwargs
                seen[id(k)] = f"MODE={kw['MODE']} BLOCK={kw['BLOCK']} RB={kw['RB']} warps={launchers[0].config.num_warps}"
    return sorted(seen.values())


def run_case(case: tuple, rounds: int) -> list[dict]:
    name, op, n, r, d, width, src_dtype, out_dtype, dists = case
    torch.manual_seed(0)
    src = torch.randn(n, width, device="cuda", dtype=src_dtype)[:, :d]
    eager = make_fn(op)
    rows = []
    for dist in dists:
        # Compile per distribution on empty caches, so that both arms autotune on
        # this distribution's first call.
        torch._dynamo.reset()
        PyCodeCache.cache_clear()
        with fresh_cache():
            off = compile_arm(op, False, **NO_REMOTE_CACHES)
            on = compile_arm(op, True, **NO_REMOTE_CACHES)
            idx = make_index(n, r, dist)
            off(src, idx, r, out_dtype)
            on(src, idx, r, out_dtype)
            picked = picked_configs()
        args = (src, idx, r, out_dtype)
        ref = reference(src, idx, r)
        off_err, on_err = errors(off(*args), ref), errors(on(*args), ref)
        del ref
        t_off, t_on = [], []
        for _ in range(rounds):
            t_off.append(do_bench(lambda: off(*args), return_mode="median"))
            t_on.append(do_bench(lambda: on(*args), return_mode="median"))
        t_eager = do_bench(lambda: eager(*args), return_mode="median")
        dist_name = dist[0] if dist[1] is None else f"{dist[0]} {dist[1]}"
        rows.append(
            {
                "case": name,
                "op": op,
                "shape": f"n={n} r={r} d={d} {str(src_dtype)[6:]}->{str(out_dtype)[6:]}",
                "dist": dist_name,
                "eager_ms": t_eager,
                "off_ms": statistics.median(t_off),
                "on_ms": statistics.median(t_on),
                "speedup": statistics.median(t_off) / statistics.median(t_on),
                "off_err": off_err,
                "on_err": on_err,
                "picked": picked,
            }
        )
        print(
            f"| {name} | {dist_name} | {t_eager:.3f} | {rows[-1]['off_ms']:.3f} | {rows[-1]['on_ms']:.3f} "
            f"| {rows[-1]['speedup']:.2f}x | {off_err[0]:.2e} / {off_err[1]:.2e} | {on_err[0]:.2e} / {on_err[1]:.2e} | {', '.join(picked)} |",
            flush=True,
        )
    return rows


def first_call_s(fn: Callable, args: tuple) -> tuple[float, float]:
    """Wall time of the first call, and the part of it spent autotuning."""
    tuning = compilation_time_metrics.setdefault("CachingAutotuner.benchmark_all_configs", [])
    before = sum(tuning)
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t, sum(tuning) - before


def run_startup(case: tuple) -> list[dict]:
    """First-call time of each arm: cold (fresh caches) and warm (a second
    compile on the same caches, i.e. the compiled graph and kernel binaries come
    from the caches, as in a new process; the flag-on arm still autotunes)."""
    name, op, n, r, d, width, src_dtype, out_dtype, dists = case
    src = torch.randn(n, width, device="cuda", dtype=src_dtype)[:, :d]
    rows = []
    with fresh_cache():
        compile_arm(op, False)(src, make_index(n, r, UNIFORM), r, out_dtype)  # process warm-up
    for dist in dists:
        idx = make_index(n, r, dist)
        args = (src, idx, r, out_dtype)
        row = {"case": name, "dist": dist[0] if dist[1] is None else f"{dist[0]} {dist[1]}"}
        for on in (False, True):
            arm = "on" if on else "off"
            with fresh_cache():
                for state in ("cold", "warm"):
                    torch._dynamo.reset()
                    PyCodeCache.cache_clear(purge=True)
                    fn = compile_arm(op, on, fx_graph_cache=True, autotune_local_cache=True, **NO_REMOTE_CACHES)
                    row[f"{arm}_{state}_s"], row[f"{arm}_{state}_tune_s"] = first_call_s(fn, args)
        rows.append(row)
        print(
            f"| {name} | {row['dist']} | {row['off_cold_s']:.2f} | {row['on_cold_s']:.2f} | {row['on_cold_s'] - row['off_cold_s']:+.2f} "
            f"| {row['off_warm_s']:.2f} | {row['on_warm_s']:.2f} | {row['on_warm_s'] - row['off_warm_s']:+.2f} "
            f"| {row['on_warm_tune_s'] - row['off_warm_tune_s']:+.2f} |",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    # optional leading action, for benchmark harnesses that pass one
    parser.add_argument("action", nargs="?", default="perf", choices=["perf"])
    parser.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--startup", action="store_true", help="measure first-call (compile + autotune) time")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    names = set(args.cases.split(",")) if args.cases else None
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    if args.startup:
        print("| case | index dist | cold off s | cold on s | added | warm off s | warm on s | added | of which autotuning |")
        print("|---|---|---|---|---|---|---|---|---|")
    else:
        print("| case | index dist | eager ms | flag off ms | flag on ms | off/on | off rel err max / mean | on rel err max / mean | picked |")
        print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for case in CASES:
        if names is None or case[0] in names:
            rows += run_startup(case) if args.startup else run_case(case, args.rounds)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "device": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
