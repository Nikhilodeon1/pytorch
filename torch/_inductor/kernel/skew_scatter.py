# mypy: allow-untyped-defs
"""
Skew-aware row scatter-add: ``out.index_add_(0, index, src)`` for 2-D ``out``
without hot-row atomic serialization.

A flat atomic scatter serializes on rows that receive a large share of the
indices (padding, clamped or popular ids). ``_skew_scatter_kernel`` is one
autotuned launch with three kinds of configs:

- ``MODE=0`` (the default): the flat [N * D] atomic scatter Inductor itself
  emits for these ops, with Inductor's two pointwise configs on sm90+ and one
  element per thread on sm80 (A100 issues fp32 atomics at about half the rate when a thread
  has several, which is what Inductor's pointwise configs give). Nothing else is
  launched, so a call that picks it costs what today's lowering costs.
- ``MODE=1``: a persistent kernel in which every program first samples the index
  (the first 64 of 512 strided samples are counted against all of them) and
  takes up to K rows sampled at least ``max(MIN_COUNT, 8x the uniform
  expectation)`` times as hot. Hot elements are summed per program in fp32
  registers and added once per program at the end; the rest are atomics.
- ``MODE=2``: per-tile exact duplicate detection. Each program sorts the
  (row, position) pairs of ``TILE`` consecutive indices in registers. A tile
  without a repeated row adds every element with an atomic; a tile with repeated
  rows sums each row in fp32 (segmented scan over the sorted order) and adds each
  distinct row once.

The choice is made by autotuning, i.e. by timing the configs on the call's real
arguments the first time each call site runs (``out`` is restored between the
trials). The best flat config is kept unless another one is at least 10% faster, so
uniform indices keep ``MODE=0`` and skewed ones pick ``MODE=1`` or ``2``. The
pick depends on the index values, which no cache key covers, so it is never read
from or written to Inductor's autotune caches: every process tunes on its own
first call (compiled binaries are cached as usual). That first call
synchronizes, as all Inductor autotuning does; later calls do not.

Small destinations (R * D <= 4096, >= 16 adds per row) instead privatize every
row into up to 4096 fp32 copies of ``out``, summed once. Where no first call on
real data is available (AOT compilation, cpp_wrapper, autotuning at compile
time) the kernel is chosen from metadata only: ``_flat_scatter_kernel`` on sm80,
nothing on sm90+.
"""

from __future__ import annotations

import torch
from torch._inductor.runtime.triton_compat import tl, triton
from torch.library import wrap_triton


_SAMPLE = 512
_CAND = 64
_K = 16
_MIN_COUNT = 6


@triton.jit
def _hot_list(idx_ptr, N, R, sample_stride, S: tl.constexpr, C: tl.constexpr, K: tl.constexpr, MIN_COUNT: tl.constexpr):
    s = tl.arange(0, S).to(tl.int64) * sample_stride
    smp = tl.load(idx_ptr + s, mask=s < N, other=0).to(tl.int64)
    smp = tl.where(s < N, tl.where(smp < 0, smp + R, smp), -1)
    c = tl.arange(0, C)
    cp = c.to(tl.int64) * sample_stride
    cand = tl.load(idx_ptr + cp, mask=cp < N, other=0).to(tl.int64)
    cand = tl.where(cp < N, tl.where(cand < 0, cand + R, cand), -2)
    cnt = tl.sum((cand[:, None] == smp[None, :]).to(tl.int32), 1)
    earlier = tl.sum(((cand[:, None] == cand[None, :]) & (c[None, :] < c[:, None])).to(tl.int32), 1)
    n_s = tl.minimum(N, S)
    t = tl.maximum(MIN_COUNT, 8 * n_s // tl.maximum(R, 1))
    sel = (earlier == 0) & (cnt >= t) & (cand >= 0)
    slot = tl.cumsum(sel.to(tl.int32), 0) - 1
    k = tl.arange(0, K)
    hit = (slot[None, :] == k[:, None]) & sel[None, :]
    return tl.sum(tl.where(hit, cand[None, :] + 1, 0), 1) - 1


@triton.jit
def _seg_sum(k1, v1, k2, v2):
    return k2, v2 + tl.where(k1 == k2, v1, 0)


def _configs(flat: list[tuple[int, int, int, int]]) -> list[triton.Config]:
    """The flat scatter configs (elements per program, warps, evict-first src,
    stages) first: the defaults the autotuner keeps unless another config is
    clearly faster. Then the hot-row and tile kernels."""
    hot = [(16, 2, 8), (64, 4, 8)]
    extra = {"EVICT": 0, "BLOCK": 512}
    return (
        [
            triton.Config({"MODE": 0, "BLOCK": b, "EVICT": e, "RB": 16, "WAVES": 1, "TILE": 32}, num_warps=w, num_stages=st)
            for b, w, e, st in flat
        ]
        + [triton.Config({"MODE": 1, **extra, "RB": rb, "WAVES": v, "TILE": 32}, num_warps=w) for rb, w, v in hot]
        + [triton.Config({"MODE": 2, **extra, "RB": 16, "WAVES": 1, "TILE": 32}, num_warps=8)]
    )


@triton.jit
def _flat_part(
    idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out,
    D: tl.constexpr, BLOCK: tl.constexpr, INT64: tl.constexpr,
    EVEN: tl.constexpr, SRC_CONTIG: tl.constexpr, OUT_CONTIG: tl.constexpr, EVICT: tl.constexpr,
):
    # With EVEN (BLOCK divides N * D) and contiguous src and out, and without
    # EVICT, this is the code Inductor generates for these scatters.
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // D
    col = offs % D
    src = src_ptr + offs if SRC_CONTIG else src_ptr + row * stride_src + col
    if EVEN:
        m = None
        idx = tl.load(idx_ptr + row, eviction_policy="evict_last").to(tl.int64)
        if EVICT:
            v = tl.load(src, eviction_policy="evict_first")
        else:
            v = tl.load(src)
    else:
        m = row < N
        idx = tl.load(idx_ptr + row, mask=m, other=0, eviction_policy="evict_last").to(tl.int64)
        if EVICT:
            v = tl.load(src, mask=m, other=0.0, eviction_policy="evict_first")
        else:
            v = tl.load(src, mask=m, other=0.0)
    idx = tl.where(idx < 0, idx + R, idx)
    dst = out_ptr + idx * D + col if OUT_CONTIG else out_ptr + idx * stride_out + col
    tl.atomic_add(dst, v.to(out_ptr.dtype.element_ty), mask=m, sem="relaxed")


@triton.jit
def _hot_part(
    idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out, sample_stride,
    D: tl.constexpr, DB: tl.constexpr, RB: tl.constexpr,
    S: tl.constexpr, C: tl.constexpr, K: tl.constexpr, MIN_COUNT: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.program_id(1) * DB + tl.arange(0, DB)
    cm = cols < D
    hot = _hot_list(idx_ptr, N, R, sample_stride, S, C, K, MIN_COUNT)
    n_hot = tl.sum((hot >= 0).to(tl.int32), 0)
    kk = tl.arange(0, K)
    acc = tl.zeros([K, DB], dtype=tl.float32)
    for t in range(pid, tl.cdiv(N, RB), nprog):
        rows = t.to(tl.int64) * RB + tl.arange(0, RB)
        m = rows < N
        idx = tl.load(idx_ptr + rows, mask=m, other=0).to(tl.int64)
        idx = tl.where(idx < 0, idx + R, idx)
        v = tl.load(
            src_ptr + rows[:, None] * stride_src + cols[None, :],
            mask=m[:, None] & cm[None, :],
            other=0.0,
        ).to(tl.float32)
        cold = m
        if n_hot > 0:
            cold = m & (tl.sum((idx[:, None] == hot[None, :]).to(tl.int32), 1) == 0)
            for k in range(0, n_hot):
                hk = tl.sum(tl.where(kk == k, hot, 0), 0)
                part = tl.sum(tl.where(((idx == hk) & m)[:, None], v, 0.0), 0)
                acc += tl.where(kk[:, None] == k, part[None, :], 0.0)
        tl.atomic_add(
            out_ptr + idx[:, None] * stride_out + cols[None, :],
            v.to(out_ptr.dtype.element_ty),
            mask=cold[:, None] & cm[None, :],
            sem="relaxed",
        )
    if n_hot > 0:
        tl.atomic_add(
            out_ptr + hot[:, None] * stride_out + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=(hot >= 0)[:, None] & cm[None, :],
            sem="relaxed",
        )


@triton.jit
def _tile_part(
    idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out,
    D: tl.constexpr, DB: tl.constexpr, TILE: tl.constexpr, KEY64: tl.constexpr,
):
    tile = tl.program_id(0)
    pos = tl.arange(0, TILE)
    rows = tile.to(tl.int64) * TILE + pos
    m = rows < N
    idx = tl.load(idx_ptr + rows, mask=m, other=0).to(tl.int64)
    idx = tl.where(idx < 0, idx + R, idx)
    # (row, position) pairs sorted by row; positions past N sort last (row R)
    key = tl.where(m, idx, R) * TILE + pos
    if not KEY64:
        key = key.to(tl.int32)
    s = tl.sort(key)
    srow = (s // TILE).to(tl.int64)
    valid = srow < R
    # occurrences of each position's row at this or later positions: 1 = last one
    _keys, left = tl.associative_scan(
        (tl.flip(srow, 0), tl.full([TILE], 1, tl.int32)), 0, _seg_sum
    )
    tail = valid & (tl.flip(left, 0) == 1)
    repeats = tl.sum(tail.to(tl.int32), 0) < tl.sum(m.to(tl.int32), 0)
    spos = tile.to(tl.int64) * TILE + (s % TILE).to(tl.int64)
    for c0 in range(0, D, DB):
        cols = c0 + tl.arange(0, DB)
        cm = cols < D
        if repeats:
            vs = tl.load(
                src_ptr + spos[:, None] * stride_src + cols[None, :],
                mask=valid[:, None] & cm[None, :],
                other=0.0,
            ).to(tl.float32)
            _rows, acc = tl.associative_scan(
                (tl.broadcast_to(srow[:, None], (TILE, DB)), vs), 0, _seg_sum
            )
            tl.atomic_add(
                out_ptr + srow[:, None] * stride_out + cols[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=tail[:, None] & cm[None, :],
                sem="relaxed",
            )
        else:
            v = tl.load(
                src_ptr + rows[:, None] * stride_src + cols[None, :],
                mask=m[:, None] & cm[None, :],
                other=0.0,
            )
            tl.atomic_add(
                out_ptr + idx[:, None] * stride_out + cols[None, :],
                v.to(out_ptr.dtype.element_ty),
                mask=m[:, None] & cm[None, :],
                sem="relaxed",
            )


def _make_skew_scatter_kernel(flat: list[tuple[int, int, int, int]]):
    """The autotuned kernel, with flat scatter configs ``flat`` as its defaults."""

    # ``site`` is unused by the kernel: one value per rewritten call site, so that
    # call sites with the same shapes are tuned separately. Values are multiples of
    # 16, so every site shares one compiled binary per config.
    @triton.autotune(configs=_configs(flat), key=["N", "D", "site"], restore_value=["out_ptr"])
    @triton.jit
    def _skew_scatter_kernel(
        idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out, sample_stride, site,
        D: tl.constexpr, DB: tl.constexpr, TDB: tl.constexpr, INT64: tl.constexpr, KEY64: tl.constexpr,
        EVEN: tl.constexpr, SRC_CONTIG: tl.constexpr, OUT_CONTIG: tl.constexpr, EVICT: tl.constexpr,
        MODE: tl.constexpr, BLOCK: tl.constexpr, RB: tl.constexpr, WAVES: tl.constexpr, TILE: tl.constexpr,
        S: tl.constexpr, C: tl.constexpr, K: tl.constexpr, MIN_COUNT: tl.constexpr,
    ):
        if MODE == 0:
            _flat_part(
                idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out,
                D, BLOCK, INT64, EVEN, SRC_CONTIG, OUT_CONTIG, EVICT,
            )
        elif MODE == 1:
            _hot_part(idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out, sample_stride, D, DB, RB, S, C, K, MIN_COUNT)
        else:
            _tile_part(idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out, D, TDB, TILE, KEY64)

    # Read by Inductor: keep the pick out of its local and remote autotune caches,
    # and keep the best flat config unless another one is >= 10% faster.
    _skew_scatter_kernel.fn.autotune_per_process = True
    _skew_scatter_kernel.fn.autotune_default_configs = len(flat)
    _skew_scatter_kernel.fn.autotune_default_margin = 0.1
    return _skew_scatter_kernel


# Default flat configs: on sm90+, Inductor's own kernel with the two configs it
# autotunes for these 1-D pointwise kernels (one pipeline stage, as Inductor's
# pointwise configs); on sm80 one element per thread, with src read evict-first,
# since fp32 atomics issued several per thread run at about half the rate there
# (Inductor's configs issue 4-8).
_skew_scatter_kernel = _make_skew_scatter_kernel([(1024, 4, 0, 1), (512, 8, 0, 1)])
_skew_scatter_kernel_sm80 = _make_skew_scatter_kernel([(256, 8, 1, 3)])


@triton.jit
def _flat_scatter_kernel(
    idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out,
    D: tl.constexpr, BLOCK: tl.constexpr, INT64: tl.constexpr,
):
    _flat_part(idx_ptr, src_ptr, out_ptr, N, R, stride_src, stride_out, D, BLOCK, INT64, False, False, False, True)


@triton.jit
def _parts_scatter_kernel(
    idx_ptr, src_ptr, out_ptr, N, R, stride_src, part_stride,
    D: tl.constexpr, BLOCK: tl.constexpr, PARTS: tl.constexpr, INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // D
    col = offs % D
    m = row < N
    idx = tl.load(idx_ptr + row, mask=m, other=0)
    idx = tl.where(idx < 0, idx + R, idx)
    m = m & (idx >= 0) & (idx < R)
    v = tl.load(src_ptr + row * stride_src + col, mask=m, other=0.0).to(tl.float32)
    po = out_ptr + (pid % PARTS) * part_stride + idx * D + col
    tl.atomic_add(po, v, mask=m, sem="relaxed")


def _num_parts(n: int, r: int, d: int) -> int:
    """Private copies of ``out`` for small destinations with many adds per row."""
    if r * d > (1 << 12) or n < 16 * r:
        return 1
    p = (1 << 21) // max(1, r * d)
    return 1 << (min(4096, p).bit_length() - 1)


def _parts_scatter(index: torch.Tensor, src: torch.Tensor, r: int, parts: int) -> torch.Tensor:
    """fp32 [r, D] sum of the scatter, via ``parts`` privatized copies."""
    n, d = index.numel(), src.shape[1]
    target = torch.zeros((parts, r, d), dtype=torch.float32, device=src.device)
    block = 512
    wrap_triton(_parts_scatter_kernel)[(triton.cdiv(n * d, block),)](
        index, src, target, n, r, src.stride(0), r * d,
        D=d, BLOCK=block, PARTS=parts, INT64=max(n * max(src.stride(0), d), parts * r * d) >= 2**31 - block,
        num_warps=4,
    )
    return target.sum(0)


def _int64(n: int, r: int, d: int, src: torch.Tensor, out: torch.Tensor) -> bool:
    return max(n * max(src.stride(0), d), r * out.stride(0)) >= 2**31 - 4096


def _flat_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    n = index.numel()
    r, d = out.shape
    # one element per thread: on A100, fp32 atomics issued several per thread run
    # at about half the rate (Inductor's pointwise configs use 4-16 per thread)
    block = 256
    wrap_triton(_flat_scatter_kernel)[(triton.cdiv(n * d, block),)](
        index, src, out, n, r, src.stride(0), out.stride(0),
        D=d, BLOCK=block, INT64=_int64(n, r, d, src, out), num_warps=8,
    )


def _skew_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int) -> None:
    n = index.numel()
    r, d = out.shape
    db = max(16, min(triton.next_power_of_2(d), 128))
    nd = triton.cdiv(d, db)
    sms = torch.cuda.get_device_properties(out.device).multi_processor_count

    def grid(meta):
        if meta["MODE"] == 0:
            return (triton.cdiv(n * d, meta["BLOCK"]), 1)
        if meta["MODE"] == 1:
            return (max(1, min(triton.cdiv(n, meta["RB"]), sms * meta["WAVES"] // nd)), nd)
        return (triton.cdiv(n, meta["TILE"]), 1)

    sm80 = torch.cuda.get_device_capability(out.device)[0] == 8
    kernel = _skew_scatter_kernel_sm80 if sm80 else _skew_scatter_kernel
    wrap_triton(kernel)[grid](
        index, src, out, n, r, src.stride(0), out.stride(0), max(1, n // _SAMPLE), 16 * (site + 1),
        D=d, DB=db, TDB=min(triton.next_power_of_2(d), 8), INT64=_int64(n, r, d, src, out),
        KEY64=(r + 1) * 64 >= 2**31, EVEN=not sm80 and (n * d) % 1024 == 0,
        SRC_CONTIG=not sm80 and src.stride(0) == d, OUT_CONTIG=not sm80 and out.stride(0) == d, S=_SAMPLE, C=_CAND, K=_K, MIN_COUNT=_MIN_COUNT,
    )


def _tile_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    """The ``MODE=2`` path alone (tests)."""
    n = index.numel()
    r, d = out.shape
    _skew_scatter_kernel.fn[(triton.cdiv(n, 32), 1)](
        index, src, out, n, r, src.stride(0), out.stride(0), 1, 16,
        D=d, DB=16, TDB=min(triton.next_power_of_2(d), 8), INT64=False, KEY64=(r + 1) * 32 >= 2**31,
        EVEN=False, SRC_CONTIG=False, OUT_CONTIG=False, EVICT=0, MODE=2, BLOCK=512, RB=16, WAVES=1, TILE=32, S=_SAMPLE, C=_CAND, K=_K, MIN_COUNT=_MIN_COUNT,
        num_warps=8,
    )


def rewrite_supported(
    device: torch.device, out_dtype: torch.dtype, rows: int, width: int, updates: int, autotune: bool
) -> bool:
    """Whether the pass rewrites a row scatter of this shape.

    - Small destinations: any CUDA arch (fp32 outputs only on sm80, where 16-bit
      outputs through the fp32 buffer are slower than Inductor's scatter).
    - Otherwise fp32 outputs only (16-bit atomics on repeated rows serialize):
      with ``autotune``, the autotuned kernel on sm80+; without, the flat kernel
      on sm80 and nothing on sm90+, where it is slower than Inductor's scatter.
    """
    cc = torch.cuda.get_device_capability(device)
    if _num_parts(updates, rows, width) > 1:
        return out_dtype == torch.float32 or cc >= (9, 0)
    if out_dtype != torch.float32 or cc < (8, 0):
        return False
    return autotune or cc[0] == 8


def _row_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int) -> None:
    """``site`` < 0: no first-call autotuning (the flat kernel)."""
    if site >= 0:
        _skew_scatter(out, index, src, site)
    else:
        _flat_scatter(out, index, src)


def _supported(out_dtype: torch.dtype, index: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        src.is_cuda
        and torch.version.hip is None
        and src.dim() == 2
        and src.stride(1) == 1
        and index.numel() > 0
        and src.is_floating_point()
        and out_dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


def _accumulate_fp32(out_dtype: torch.dtype, rows: int, src: torch.Tensor) -> bool:
    """16-bit outputs that are small next to ``src`` accumulate in an fp32 buffer
    rounded once."""
    return out_dtype != torch.float32 and 2 * rows * src.shape[1] <= src.numel()


def scatter_rows(
    index: torch.Tensor, src: torch.Tensor, num_rows: int, out_dtype: torch.dtype, site: int = -1
) -> torch.Tensor:
    """``zeros(num_rows, D, dtype=out_dtype).index_add_(0, index, src)``."""
    d = src.shape[-1]
    if not _supported(out_dtype, index, src):
        out = torch.zeros((num_rows, d), dtype=out_dtype, device=src.device)
        return out.index_add_(0, index.reshape(-1), src.to(out_dtype))
    index = index.reshape(-1)
    parts = _num_parts(index.numel(), num_rows, d)
    if parts > 1:
        return _parts_scatter(index, src, num_rows, parts).to(out_dtype)
    acc_fp32 = _accumulate_fp32(out_dtype, num_rows, src)
    out = torch.zeros((num_rows, d), dtype=torch.float32 if acc_fp32 else out_dtype, device=src.device)
    _row_scatter(out, index, src, site)
    return out.to(out_dtype) if acc_fp32 else out


def skew_scatter_add_(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int = -1
) -> torch.Tensor:
    """
    In-place ``out.index_add_(0, index, src)`` for 2-D ``out`` and ``src`` (dim 0,
    alpha 1); ``src`` may be a narrower float dtype than ``out`` and is read
    directly. bf16/fp16 ``out`` accumulates into an fp32 buffer rounded once when
    ``out`` is small next to ``src``, else uses atomics in ``out``'s dtype.
    ``site`` >= 0 names the call site for first-call autotuning.
    """
    if not _supported(out.dtype, index, src) or out.dim() != 2 or out.stride(1) != 1:
        return out.index_add_(0, index, src.to(out.dtype))
    index = index.reshape(-1)
    r, d = out.shape
    parts = _num_parts(index.numel(), r, d)
    if parts > 1:
        return out.add_(_parts_scatter(index, src, r, parts).to(out.dtype))
    if _accumulate_fp32(out.dtype, r, src):
        acc = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
        _row_scatter(acc, index, src, site)
        return out.add_(acc)
    _row_scatter(out, index, src, site)
    return out
