#!/usr/bin/env python3
"""Probe 2: validate decode-attention / delta-scan / small-GEMV optimization hypotheses on PPU."""

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


# ---------------------------------------------------------- GQA decode attn


@triton.jit
def _gqa_decode_attn_splitkv(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    l_ptr,
    seq_len,
    kv_heads: tl.constexpr,
    group: tl.constexpr,
    head_dim: tl.constexpr,
    scale,
    block_n: tl.constexpr,
    splits: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // group
    d = tl.arange(0, head_dim)

    q = tl.load(q_ptr + head * head_dim + d).to(tl.float32)

    per_split = tl.cdiv(seq_len, splits)
    n_start = split * per_split
    n_end = tl.minimum(n_start + per_split, seq_len)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((head_dim,), tl.float32)

    for start in range(n_start, n_end, block_n):
        n = start + tl.arange(0, block_n)
        mask = n < n_end
        k = tl.load(
            k_ptr + kv_head * seq_len * head_dim + n[:, None] * head_dim + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr + kv_head * seq_len * head_dim + n[:, None] * head_dim + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    base = (head * splits + split) * (head_dim + 2)
    tl.store(o_ptr + head * splits * head_dim + split * head_dim + d, acc)
    tl.store(m_ptr + base, m_i)
    tl.store(l_ptr + base + 1, l_i)


@triton.jit
def _gqa_decode_attn_combine(
    acc_ptr,
    m_ptr,
    o_ptr,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    splits: tl.constexpr,
):
    head = tl.program_id(0)
    d = tl.arange(0, head_dim)
    m_max = -float("inf")
    for s in range(splits):
        m_max = tl.maximum(m_max, tl.load(m_ptr + (head * splits + s) * (head_dim + 2)))
    acc = tl.zeros((head_dim,), tl.float32)
    l_total = 0.0
    for s in range(splits):
        base = (head * splits + s) * (head_dim + 2)
        m_s = tl.load(m_ptr + base)
        l_s = tl.load(m_ptr + base + 1)
        weight = tl.exp(m_s - m_max)
        acc += tl.load(acc_ptr + head * splits * head_dim + s * head_dim + d) * weight
        l_total += l_s * weight
    tl.store(o_ptr + head * head_dim + d, acc / l_total)


def triton_gqa_decode(q, k, v, scale, splits=4, block_n=64, warps=4):
    q_heads, head_dim = q.shape
    kv_heads, seq_len, _ = k.shape
    group = q_heads // kv_heads
    o_part = torch.empty(q_heads, splits, head_dim, device=q.device, dtype=torch.float32)
    ml = torch.empty(q_heads, splits, head_dim + 2, device=q.device, dtype=torch.float32)
    _gqa_decode_attn_splitkv[(q_heads, splits)](
        q, k, v, o_part, ml, ml, seq_len, kv_heads, group, head_dim, scale, block_n, splits,
        num_warps=warps,
    )
    out = torch.empty(q_heads, head_dim, device=q.device, dtype=torch.float32)
    _gqa_decode_attn_combine[(q_heads,)](o_part, ml, out, q_heads, head_dim, splits, num_warps=4)
    return out


def attention_probe() -> dict:
    rows = {}
    for seq_len in (512, 1024):
        torch.manual_seed(7)
        q = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, seq_len, 256, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, seq_len, 256, device="cuda", dtype=torch.bfloat16)
        scale = 1.0 / 16.0

        # Reference: SDPA as the current HF attention interface would call it.
        q4 = q.view(1, 8, 1, 256)
        k4 = k.view(1, 2, seq_len, 256)
        v4 = v.view(1, 2, seq_len, 256)
        ref = F.scaled_dot_product_attention(q4, k4, v4, scale=scale, enable_gqa=True)

        rows[f"sdpa_L{seq_len}_ms"] = time_ms(
            lambda: F.scaled_dot_product_attention(q4, k4, v4, scale=scale, enable_gqa=True)
        )

        k_full = k4.repeat_interleave(4, dim=1).float()
        v_full = v4.repeat_interleave(4, dim=1).float()

        def manual():
            scores = (q4.float() * scale) @ k_full.transpose(-1, -2)
            return torch.softmax(scores, dim=-1) @ v_full
        rows[f"manual_torch_L{seq_len}_ms"] = time_ms(manual)

        for splits in (2, 4, 8):
            for block_n in (32, 64, 128):
                for warps in (2, 4, 8):
                    key = f"triton_s{splits}_b{block_n}_w{warps}_L{seq_len}"
                    try:
                        out = triton_gqa_decode(q, k, v, scale, splits, block_n, warps)
                        diff = (out.view(1, 8, 1, 256).to(torch.bfloat16).float() - ref.float()).abs().max().item()
                        ms = time_ms(lambda: triton_gqa_decode(q, k, v, scale, splits, block_n, warps))
                        rows[key] = {"ms": round(ms, 5), "max_diff_vs_sdpa": round(diff, 4)}
                    except Exception as exc:  # noqa: BLE001
                        rows[key] = {"error": str(exc)[:60]}
        del q, k, v, q4, k4, v4
        torch.cuda.empty_cache()
    return rows


# -------------------------------------------------- delta recurrent pipelining


@triton.jit(do_not_specialize_on_alignment=["seq_len"])
def _delta_recurrent_stages_kernel(
    qkv_ptr,
    packed_ptr,
    a_log_ptr,
    dt_bias_ptr,
    state_ptr,
    factor_ptr,
    out_ptr,
    batch: tl.constexpr,
    seq_len,
    heads: tl.constexpr,
    dim: tl.constexpr,
    qkv_stride: tl.constexpr,
    packed_stride: tl.constexpr,
    z_offset: tl.constexpr,
    a_offset: tl.constexpr,
    b_offset: tl.constexpr,
    block_v: tl.constexpr,
    block_k: tl.constexpr,
    use_precomputed_factors: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_head = tl.cdiv(dim, block_v)
    tile_v = pid % tiles_per_head
    head_flat = pid // tiles_per_head
    head = head_flat % heads
    batch_idx = head_flat // heads

    v_idx = tile_v * block_v + tl.arange(0, block_v)
    k_idx = tl.arange(0, block_k)
    v_mask = v_idx < dim
    k_mask = k_idx < dim
    matrix_mask = v_mask[:, None] & k_mask[None, :]

    state_offsets = ((batch_idx * heads + head) * dim + k_idx[None, :]) * dim + v_idx[:, None]
    state = tl.load(state_ptr + state_offsets, mask=matrix_mask, other=0.0).to(tl.float32)

    for token in range(0, seq_len):
        qkv_base = (batch_idx * seq_len + token) * qkv_stride
        q = tl.load(qkv_ptr + qkv_base + head * dim + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k = tl.load(qkv_ptr + qkv_base + heads * dim + head * dim + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        value = tl.load(qkv_ptr + qkv_base + 2 * heads * dim + head * dim + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        factor_base = ((batch_idx * seq_len + token) * heads + head) * 4
        q_inv = tl.load(factor_ptr + factor_base)
        k_inv = tl.load(factor_ptr + factor_base + 1)
        alpha = tl.load(factor_ptr + factor_base + 2)
        beta = tl.load(factor_ptr + factor_base + 3)
        q_hat = q * q_inv * (1.0 / 1.1285969206630595)
        k_hat = k * k_inv
        state *= alpha
        memory_k = tl.sum(state * k_hat[None, :], axis=1)
        delta = beta * (value - memory_k)
        state += delta[:, None] * k_hat[None, :]
        output = tl.sum(state * q_hat[None, :], axis=1)
        output_offsets = ((batch_idx * seq_len + token) * heads + head) * dim + v_idx
        tl.store(out_ptr + output_offsets, output, mask=v_mask)

    tl.store(state_ptr + state_offsets, state, mask=matrix_mask)


def delta_scan_probe() -> dict:
    from qwen35_fused.kernels import delta_recurrent_fused

    rows = {}
    for seq_len in (340, 700):
        qkv = torch.randn(1, seq_len, 6144, device="cuda", dtype=torch.bfloat16)
        packed = torch.randn(1, seq_len, 8224, device="cuda", dtype=torch.bfloat16)
        a_log = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        dt_bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(1, 16, 128, 128, device="cuda", dtype=torch.float32)
        out = delta_recurrent_fused(qkv, packed, a_log, dt_bias, state, precompute_factors=True)
        base_ms = time_ms(
            lambda: delta_recurrent_fused(qkv, packed, a_log, dt_bias, state, precompute_factors=True),
            repetitions=10,
        )
        rows[f"current_stages1_L{seq_len}_ms"] = round(base_ms, 4)
        # per-layer estimate x18
        rows[f"current_stages1_L{seq_len}_x18_ms"] = round(base_ms * 18, 3)

        # num_stages sweep on the copied kernel with precomputed factors
        factors = torch.empty(1, seq_len, 16, 4, device="cuda", dtype=torch.float32).normal_()
        grid = (16 * (128 // 8),)
        for stages in (2, 3, 4):
            def run():
                _delta_recurrent_stages_kernel[grid](
                    qkv, packed, a_log, dt_bias, state, factors,
                    torch.empty(1, seq_len, 16, 128, device="cuda", dtype=torch.bfloat16),
                    1, seq_len, 16, 128, 6144, 8224, 6144, 6144 + 2048, 6144 + 2048 + 16,
                    8, 128, True, num_warps=4, num_stages=stages,
                )
            try:
                ms = time_ms(run, repetitions=10)
                rows[f"stages{stages}_L{seq_len}_ms"] = round(ms, 4)
            except Exception as exc:  # noqa: BLE001
                rows[f"stages{stages}_L{seq_len}_ms"] = {"error": str(exc)[:60]}
        del qkv, packed, state, out, factors
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------- small-N GEMV split-K

SMALL_SHAPES = [(2048, 2048), (6144, 6144), (248320, 2048)]


def small_gemv_probe() -> dict:
    rows = {}
    for n_out, k_in in SMALL_SHAPES:
        weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, 1, k_in, device="cuda", dtype=torch.bfloat16)
        ms = time_ms(lambda: F.linear(x, weight), repetitions=20)
        rows[f"flinear_{n_out}x{k_in}_ms"] = round(ms, 5)
        rows[f"flinear_{n_out}x{k_in}_GBps"] = round(n_out * k_in * 2 / (ms / 1e3) / 1e9, 1)
        del weight, x
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    result = {
        "gqa_decode_attention": attention_probe(),
        "delta_scan": delta_scan_probe(),
        "small_gemv": small_gemv_probe(),
    }
    out_path = ROOT / "artifacts" / "probe_decode_hypotheses.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
