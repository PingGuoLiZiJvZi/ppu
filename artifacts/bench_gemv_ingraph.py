#!/usr/bin/env python3
"""In-graph, cold-weight GEMV benchmark: the only trustworthy comparison regime.

24 distinct weight tensors per shape (no L2 reuse), captured in a CUDA graph.
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


@triton.jit
def _stream_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    splits,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * block_n + tl.arange(0, block_n)
    n_mask = n < N
    acc = tl.zeros((block_n,), tl.float32)
    per_split = K // splits
    for kk in range(0, per_split, block_k):
        idx = pid_k * per_split + kk + tl.arange(0, block_k)
        k_mask = idx < K
        xv = tl.load(x_ptr + idx, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + n[:, None] * K + idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.atomic_add(y_ptr + n, acc, mask=n_mask)


@triton.jit
def _cast_kernel(y_ptr, out_ptr, numel: tl.constexpr, block: tl.constexpr):
    offs = tl.program_id(0) * block + tl.arange(0, block)
    mask = offs < numel
    tl.store(out_ptr + offs, tl.load(y_ptr + offs, mask=mask).to(tl.bfloat16), mask=mask)


def bench_in_graph(fn, *, warmup=3, reps=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(g):
        tmp = fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / reps)


def main() -> None:
    torch.manual_seed(11)
    shapes = [
        ("delta_out/o_proj", 2048, 2048, 24),
        ("mlp_down", 2048, 6144, 24),
        ("delta_in_proj", 8320, 2048, 18),
        ("attn_qgkv", 5120, 2048, 6),
        ("mlp_gate_up", 12288, 2048, 24),
    ]
    rows = {}
    for name, n_out, k_in, count in shapes:
        weights = [
            (torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02)
            for _ in range(count)
        ]
        xs = [torch.randn(k_in, device="cuda", dtype=torch.bfloat16) for _ in range(count)]
        # F.linear path inside a graph.
        def linear_path():
            total = torch.zeros(1, device="cuda", dtype=torch.float32)
            for w, x in zip(weights, xs):
                total = total + F.linear(x.view(1, 1, -1), w).float().sum()
            return total

        ms = bench_in_graph(linear_path)
        rows[name] = {
            "count": count,
            "linear_graph_ms": round(ms, 4),
            "linear_us_per_launch": round(ms * 1000 / count, 2),
            "linear_GBps": round(n_out * k_in * 2 * count / (ms / 1e3) / 1e9, 1),
        }

        # Custom split-K streaming GEMV + cast, per config.
        best = None
        for block_n, block_k, splits, warps in (
            (16, 128, 1, 4),
            (32, 128, 1, 4),
            (16, 128, 2, 4),
            (16, 128, 4, 4),
            (32, 256, 1, 8),
            (16, 256, 1, 4),
        ):
            if k_in % (block_k * splits):
                continue
            outs_f32 = [torch.zeros(n_out, device="cuda", dtype=torch.float32) for _ in range(count)]
            outs_bf16 = [torch.empty(n_out, device="cuda", dtype=torch.bfloat16) for _ in range(count)]

            def custom_path(bn=block_n, bk=block_k, s=splits, w_=warps):
                total = torch.zeros(1, device="cuda", dtype=torch.float32)
                for i, (w, x) in enumerate(zip(weights, xs)):
                    y = outs_f32[i]
                    grid = (triton.cdiv(n_out, bn), s)
                    _stream_gemv_kernel[grid](
                        x, w, y, n_out, k_in, bn, bk, s, num_warps=w_, num_stages=1,
                    )
                    _cast_kernel[(triton.cdiv(n_out, 1024),)](y, outs_bf16[i], n_out, 1024, num_warps=4)
                    total = total + outs_bf16[i].float().sum()
                return total

            try:
                ms = bench_in_graph(custom_path)
            except Exception as exc:  # noqa: BLE001
                rows[name][f"custom_bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = str(exc)[:40]
                continue
            per_launch = ms * 1000 / count
            rows[name][f"custom_bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = {
                "graph_ms": round(ms, 4),
                "us_per_launch_pair": round(per_launch, 2),
                "GBps": round(n_out * k_in * 2 * count / (ms / 1e3) / 1e9, 1),
            }
            if best is None or per_launch < best[0]:
                best = (per_launch, f"bn{block_n}_bk{block_k}_s{splits}_w{warps}")
        if best:
            rows[name]["best_custom"] = {"us_per_launch": round(best[0], 2), "config": best[1]}
        del weights, xs
        torch.cuda.empty_cache()

    out = ROOT / "artifacts" / "bench_gemv_ingraph.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
