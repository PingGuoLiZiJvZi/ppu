#!/usr/bin/env python3
"""Probe PPU-ZW810E bandwidth / GEMV / launch-overhead limits for decode optimization.

No model load: synthetic tensors only, exact decode shapes of Qwen3.5-2B.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen35_fused.kernels import lm_head_argmax, ppu_swiglu_gemv


def time_ms(fn, repetitions: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repetitions)


# ---------------------------------------------------------------- bandwidth


def bandwidth_probe() -> dict:
    rows = {}
    n = 2 * 1024**3 // 2  # 2 GiB of BF16
    src = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    ms = time_ms(lambda: dst.copy_(src), repetitions=10)
    bytes_moved = 2 * n * 2
    rows["d2d_copy_GBps"] = bytes_moved / (ms / 1e3) / 1e9

    # Pure-read reduction kernel: measures achievable read bandwidth.
    @triton.jit
    def _read_reduce_kernel(x_ptr, out_ptr, numel, block: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * block + tl.arange(0, block)
        vals = tl.load(x_ptr + offsets, mask=offsets < numel, other=0.0).to(tl.float32)
        tl.atomic_add(out_ptr, tl.sum(vals, axis=0))

    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    for block, warps in ((1024, 4), (2048, 8), (4096, 8)):
        grid = (triton.cdiv(n, block),)
        ms = time_ms(lambda: _read_reduce_kernel[grid](src, out, n, block, num_warps=warps), repetitions=10)
        rows[f"read_reduce_b{block}_w{warps}_GBps"] = n * 2 / (ms / 1e3) / 1e9

    del src, dst
    torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------- GEMV


DECODE_SHAPES = [
    ("delta_in_proj", 8320, 2048, 18),
    ("delta_out_proj", 2048, 2048, 18),
    ("attn_qgkv", 5120, 2048, 6),
    ("attn_o_proj", 2048, 2048, 6),
    ("mlp_gate_up", 12288, 2048, 24),
    ("mlp_down", 6144, 6144, 24),
]

# lm_head separate: 248320 x 2048, count 1


def gemv_probe() -> tuple[dict, float]:
    rows = {}
    total_linear_ms = 0.0
    for name, n_out, k_in, count in DECODE_SHAPES:
        weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, 1, k_in, device="cuda", dtype=torch.bfloat16)
        ms = time_ms(lambda: F.linear(x, weight))
        gbps = (n_out * k_in * 2) / (ms / 1e3) / 1e9
        total_linear_ms += ms * count
        rows[name] = {
            "shape": [n_out, k_in],
            "count_per_token": count,
            "flinear_ms": round(ms, 4),
            "flinear_GBps": round(gbps, 1),
            "total_ms_per_token": round(ms * count, 4),
        }
        del weight, x
    torch.cuda.empty_cache()

    # Existing fused SwiGLU GEMV on the MLP gate/up shape.
    gate_up = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, 1, 2048, device="cuda", dtype=torch.bfloat16)
    ms = time_ms(lambda: ppu_swiglu_gemv(x, gate_up))
    rows["mlp_gate_up"]["ppu_swiglu_gemv_ms"] = round(ms, 4)
    rows["mlp_gate_up"]["ppu_swiglu_gemv_GBps"] = round(
        (12288 * 2048 * 2) / (ms / 1e3) / 1e9, 1
    )
    del gate_up, x
    torch.cuda.empty_cache()
    return rows, total_linear_ms


# --------------------------------------------------- custom GEMV variants


@triton.jit
def _gemv_splitk_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    splits: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * block_n + tl.arange(0, block_n)
    k = pid_k * (K // splits) + tl.arange(0, block_k)
    acc = tl.zeros((block_n,), tl.float32)
    for kk in range(0, K // splits, block_k):
        idx = pid_k * (K // splits) + kk + tl.arange(0, block_k)
        xv = tl.load(x_ptr + idx).to(tl.float32)
        w = tl.load(w_ptr + n[:, None] * K + idx[None, :]).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.atomic_add(y_ptr + n, acc)


def custom_gemv_probe() -> dict:
    """Try a split-K / wider tile GEMV on the biggest non-LM-head shape."""
    n_out, k_in = 12288, 2048
    weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(k_in, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros(n_out, device="cuda", dtype=torch.float32)
    rows = {}
    for block_n, block_k, splits, warps in (
        (16, 128, 1, 4),
        (32, 128, 1, 4),
        (32, 128, 1, 8),
        (64, 128, 1, 8),
        (16, 64, 4, 4),
        (32, 64, 4, 4),
        (32, 128, 2, 4),
        (64, 128, 2, 8),
    ):
        grid = (triton.cdiv(n_out, block_n), splits)
        try:
            def run():
                y.zero_()
                _gemv_splitk_kernel[grid](
                    x, weight, y, n_out, k_in, block_n, block_k, splits, num_warps=warps
                )
            ms = time_ms(run)
            rows[f"bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = {
                "ms": round(ms, 4),
                "GBps": round((n_out * k_in * 2) / (ms / 1e3) / 1e9, 1),
            }
        except Exception as exc:  # noqa: BLE001
            rows[f"bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = {"error": str(exc)[:80]}
    del weight, x, y
    torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------- launch overhead


def launch_probe() -> dict:
    x = torch.randn(256, device="cuda", dtype=torch.bfloat16)

    @triton.jit
    def _tiny_kernel(x_ptr, out_ptr, block: tl.constexpr):
        offs = tl.arange(0, block)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1.0)

    out = torch.empty_like(x)
    _tiny_kernel[(1,)](x, out, 256, num_warps=1)
    ms = time_ms(lambda: _tiny_kernel[(1,)](x, out, 256, num_warps=1), repetitions=200)
    return {
        "triton_tiny_kernel_ms_eager": round(ms, 5),
        "estimated_us_per_launch": round(ms * 1000, 2),
    }


def lm_head_probe() -> dict:
    vocab, hidden = 248320, 2048
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    h = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
    rows = {}
    for block_n in (8, 16, 32, 64):
        try:
            ms = time_ms(lambda: lm_head_argmax(h, weight, block_n=block_n))
            rows[f"block_n{block_n}"] = {
                "ms": round(ms, 4),
                "GBps": round((vocab * hidden * 2) / (ms / 1e3) / 1e9, 1),
            }
        except Exception as exc:  # noqa: BLE001
            rows[f"block_n{block_n}"] = {"error": str(exc)[:80]}
    rows["eager_linear_argmax_ms"] = round(
        time_ms(lambda: F.linear(h, weight).argmax(dim=-1, keepdim=True)), 4
    )
    del weight, h
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    torch.manual_seed(20260826)
    result = {
        "bandwidth": bandwidth_probe(),
        "decode_gemv": None,
        "custom_gemv_12288x2048": custom_gemv_probe(),
        "launch": launch_probe(),
        "lm_head": lm_head_probe(),
    }
    gemv_rows, total_ms = gemv_probe()
    result["decode_gemv"] = gemv_rows
    result["decode_gemv_total_flinear_ms_per_token"] = round(total_ms, 4)
    out_path = ROOT / "artifacts" / "probe_ppu_limits.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
