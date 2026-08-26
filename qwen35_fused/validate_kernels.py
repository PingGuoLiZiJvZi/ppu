#!/usr/bin/env python3
"""Correctness and latency checks for the standalone fused kernels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen35_fused.kernels import (
    attention_gate_mul,
    causal_conv1d_fused,
    delta_recurrent_fused,
    gated_rms_norm,
    gqa_decode_attention,
    layer_norm,
    lm_head_argmax,
    position_embed_add,
    ppu_swiglu_gemv,
    qgkv_norm_rope,
    residual_add_rms_norm,
    residual_add_layer_norm,
    rms_norm,
    sigmoid_mul,
    silu_and_mul,
    vision_qkv_rope,
)


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def time_ms(fn: Callable[[], object], repetitions: int = 100) -> float:
    for _ in range(10):
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


def reference_rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (out * (1.0 + weight.float())).type_as(x)


def reference_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    out = weight * out.to(x.dtype)
    out = out * F.silu(gate.float())
    return out.to(x.dtype)


def reference_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    batch, seq_len, channels = x.shape
    kernel_size = weight.shape[-1]
    if seq_len == 1:
        merged = torch.cat([state, x.transpose(1, 2)], dim=-1).to(weight.dtype)
        state.copy_(merged[:, :, -kernel_size:])
        out = F.conv1d(merged, weight.unsqueeze(1), padding=0, groups=channels)[:, :, -1:]
        return F.silu(out).transpose(1, 2).to(x.dtype)
    state.copy_(x.transpose(1, 2)[..., -kernel_size:])
    out = F.conv1d(
        x.transpose(1, 2).to(weight.dtype),
        weight.unsqueeze(1),
        padding=kernel_size - 1,
        groups=channels,
    )[:, :, :seq_len]
    return F.silu(out).transpose(1, 2).contiguous().to(x.dtype)


def reference_delta(
    qkv: torch.Tensor,
    packed: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, width = qkv.shape
    heads = a_log.numel()
    dim = width // (3 * heads)
    q, k, v = qkv.float().reshape(batch, seq_len, 3, heads, dim).unbind(2)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / dim**0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    a_offset = width + heads * dim
    b_offset = a_offset + heads
    a = packed[..., a_offset:b_offset].float()
    b = packed[..., b_offset : b_offset + heads].float()
    g = -a_log.float().exp() * F.softplus(a + dt_bias.float())
    beta = b.sigmoid()
    outputs = []
    for token in range(seq_len):
        state.mul_(g[:, token].exp()[..., None, None])
        memory = (state * k[:, token].unsqueeze(-1)).sum(-2)
        delta = (v[:, token] - memory) * beta[:, token].unsqueeze(-1)
        state.add_(k[:, token].unsqueeze(-1) * delta.unsqueeze(-2))
        outputs.append((state * q[:, token].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1).to(qkv.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="optional path for the machine-readable result")
    args = parser.parse_args()
    torch.manual_seed(20260819)
    device = "cuda"
    dtype = torch.bfloat16
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    for width in (128, 256, 2048):
        x = torch.randn(7, width, device=device, dtype=dtype)
        weight = torch.randn(width, device=device, dtype=dtype)
        expected = reference_rms(x, weight)
        actual = rms_norm(x, weight)
        results[f"rms_{width}"] = error(actual, expected)

    residual = torch.randn(7, 2048, device=device, dtype=dtype)
    update = torch.randn_like(residual)
    weight = torch.randn(2048, device=device, dtype=dtype)
    expected_residual = residual + update
    expected_norm = reference_rms(expected_residual, weight)
    actual_residual, actual_norm = residual_add_rms_norm(residual, update, weight)
    results["residual_add"] = error(actual_residual, expected_residual)
    results["residual_add_rms"] = error(actual_norm, expected_norm)

    vision_x = torch.randn(13, 1024, device=device, dtype=dtype)
    vision_update = torch.randn_like(vision_x)
    vision_weight = torch.randn(1024, device=device, dtype=dtype)
    vision_bias = torch.randn(1024, device=device, dtype=dtype)
    expected_layer = F.layer_norm(vision_x, (1024,), vision_weight, vision_bias, 1e-6)
    results["layer_norm"] = error(
        layer_norm(vision_x, vision_weight, vision_bias), expected_layer
    )
    expected_vision_residual = vision_x + vision_update
    expected_vision_norm = F.layer_norm(
        expected_vision_residual, (1024,), vision_weight, vision_bias, 1e-6
    )
    actual_vision_residual, actual_vision_norm = residual_add_layer_norm(
        vision_x, vision_update, vision_weight, vision_bias
    )
    results["vision_residual_add"] = error(actual_vision_residual, expected_vision_residual)
    results["vision_residual_norm"] = error(actual_vision_norm, expected_vision_norm)

    table = torch.randn(64, 1024, device=device, dtype=dtype)
    indices = torch.randint(0, 64, (13, 4), device=device)
    interpolation = torch.rand(13, 4, device=device)
    expected_position = vision_x + (
        F.embedding(indices, table) * interpolation[:, :, None]
    ).sum(1).to(dtype)
    results["position_embed_add"] = error(
        position_embed_add(vision_x, table, indices, interpolation), expected_position
    )

    vision_tokens, vision_heads, vision_dim = 13, 4, 64
    vision_packed = torch.randn(
        vision_tokens,
        3 * vision_heads * vision_dim,
        device=device,
        dtype=dtype,
    )
    vision_cos = torch.randn(vision_tokens, vision_dim, device=device, dtype=torch.float32)
    vision_sin = torch.randn_like(vision_cos)
    vision_q, vision_k, vision_v = vision_packed.view(
        vision_tokens, 3, vision_heads, vision_dim
    ).permute(1, 0, 2, 3).unbind(0)

    def vision_rope_reference(x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rotated = torch.cat((-x32[..., vision_dim // 2 :], x32[..., : vision_dim // 2]), dim=-1)
        return (x32 * vision_cos[:, None, :] + rotated * vision_sin[:, None, :]).to(dtype)

    expected_vision_q = vision_rope_reference(vision_q).transpose(0, 1).unsqueeze(0)
    expected_vision_k = vision_rope_reference(vision_k).transpose(0, 1).unsqueeze(0)
    expected_vision_v = vision_v.transpose(0, 1).unsqueeze(0)
    actual_vision_q, actual_vision_k, actual_vision_v = vision_qkv_rope(
        vision_packed,
        vision_cos,
        vision_sin,
        heads=vision_heads,
        dim=vision_dim,
    )
    results["vision_qkv_rope_q"] = error(actual_vision_q, expected_vision_q)
    results["vision_qkv_rope_k"] = error(actual_vision_k, expected_vision_k)
    results["vision_qkv_rope_v"] = error(actual_vision_v, expected_vision_v)

    x = torch.randn(32, 128, device=device, dtype=dtype)
    gate = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=dtype)
    results["gated_rms"] = error(gated_rms_norm(x, gate, weight), reference_gated(x, gate, weight))

    packed_mlp = torch.randn(7, 12288, device=device, dtype=dtype)
    expected_mlp = F.silu(packed_mlp[:, :6144]) * packed_mlp[:, 6144:]
    results["silu_mul"] = error(silu_and_mul(packed_mlp), expected_mlp)

    swiglu_x = torch.randn(1, 1, 2048, device=device, dtype=dtype)
    swiglu_weight = torch.randn(12288, 2048, device=device, dtype=dtype)
    swiglu_expected = silu_and_mul(F.linear(swiglu_x, swiglu_weight))
    results["ppu_swiglu_gemv"] = error(
        ppu_swiglu_gemv(swiglu_x, swiglu_weight),
        swiglu_expected,
    )

    gate = torch.randn(7, 2048, device=device, dtype=dtype)
    value = torch.randn_like(gate)
    results["sigmoid_mul"] = error(sigmoid_mul(value, gate), value * gate.sigmoid())

    # Decode GQA attention vs maskless SDPA over the valid prefix. The kernel
    # reassociates the FP32 online softmax, so expect BF16-level (not exact)
    # agreement.
    decode_q = torch.randn(1, 8, 1, 256, device=device, dtype=dtype)
    decode_keys = torch.randn(1, 2, 512, 256, device=device, dtype=dtype)
    decode_values = torch.randn_like(decode_keys)
    for n_valid in (1, 7, 370, 512):
        length = torch.tensor([n_valid], device=device, dtype=torch.int64)
        expected_decode = F.scaled_dot_product_attention(
            decode_q,
            decode_keys[:, :, :n_valid],
            decode_values[:, :, :n_valid],
            scale=0.0625,
            enable_gqa=True,
        ).view(1, 1, -1)
        actual_decode = gqa_decode_attention(
            decode_q, decode_keys, decode_values, length, 0.0625
        )
        results[f"gqa_decode_attn_{n_valid}"] = error(actual_decode, expected_decode)

    batch, seq_len, q_heads, kv_heads, head_dim, rotary_dim = 1, 5, 8, 2, 256, 64
    packed_width = 2 * q_heads * head_dim + 2 * kv_heads * head_dim
    packed_qgkv = torch.randn(batch, seq_len, packed_width, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    cos = torch.randn(batch, seq_len, rotary_dim, device=device, dtype=dtype)
    sin = torch.randn_like(cos)
    q_proj_raw = packed_qgkv[..., : 2 * q_heads * head_dim].view(
        batch, seq_len, q_heads, 2 * head_dim
    )
    q_raw = q_proj_raw[..., :head_dim]
    k_start = 2 * q_heads * head_dim
    k_raw = packed_qgkv[..., k_start : k_start + kv_heads * head_dim].view(
        batch, seq_len, kv_heads, head_dim
    )
    v_raw = packed_qgkv[..., k_start + kv_heads * head_dim :].view(
        batch, seq_len, kv_heads, head_dim
    )
    q_norm = reference_rms(q_raw, q_weight).transpose(1, 2)
    k_norm = reference_rms(k_raw, k_weight).transpose(1, 2)

    def reference_rope(x: torch.Tensor) -> torch.Tensor:
        rotated = torch.cat((-x[..., rotary_dim // 2 : rotary_dim], x[..., : rotary_dim // 2]), dim=-1)
        front = x[..., :rotary_dim] * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
        return torch.cat((front, x[..., rotary_dim:]), dim=-1)

    expected_q = reference_rope(q_norm)
    expected_k = reference_rope(k_norm)
    expected_v = v_raw.transpose(1, 2)
    actual_q, actual_k, actual_v = qgkv_norm_rope(
        packed_qgkv,
        q_weight,
        k_weight,
        cos,
        sin,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim=head_dim,
        rotary_dim=rotary_dim,
    )
    results["qgkv_q"] = error(actual_q, expected_q)
    results["qgkv_k"] = error(actual_k, expected_k)
    results["qgkv_v"] = error(actual_v, expected_v)

    cache_len = 13
    cache_start = torch.tensor(3, device=device, dtype=torch.int64)
    key_cache = torch.zeros(
        batch, kv_heads, cache_len, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros_like(key_cache)
    cached_q, cached_k, cached_v = qgkv_norm_rope(
        packed_qgkv,
        q_weight,
        k_weight,
        cos,
        sin,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim=head_dim,
        rotary_dim=rotary_dim,
        cache_key=key_cache,
        cache_value=value_cache,
        cache_start=cache_start,
    )
    results["qgkv_cache_q"] = error(cached_q, expected_q)
    results["qgkv_cache_k"] = error(cached_k[:, :, 3 : 3 + seq_len], expected_k)
    results["qgkv_cache_v"] = error(cached_v[:, :, 3 : 3 + seq_len], expected_v)
    results["qgkv_cache_prefix"] = error(
        cached_k[:, :, :3], torch.zeros_like(cached_k[:, :, :3])
    )
    results["qgkv_cache_suffix"] = error(
        cached_v[:, :, 3 + seq_len :], torch.zeros_like(cached_v[:, :, 3 + seq_len :])
    )
    attn_out = torch.randn(batch, seq_len, q_heads * head_dim, device=device, dtype=dtype)
    gate_raw = q_proj_raw[..., head_dim:].reshape(batch, seq_len, q_heads * head_dim)
    expected_gate = attn_out * gate_raw.sigmoid()
    results["attention_gate"] = error(
        attention_gate_mul(attn_out, packed_qgkv, head_dim), expected_gate
    )

    lm_hidden = torch.randn(1, 128, device=device, dtype=dtype)
    lm_weight = torch.randn(257, 128, device=device, dtype=dtype)
    expected_top1 = F.linear(lm_hidden, lm_weight).argmax(dim=-1, keepdim=True)
    actual_top1 = lm_head_argmax(lm_hidden, lm_weight)
    results["lm_head_top1"] = {
        "exact": bool(torch.equal(actual_top1, expected_top1)),
        "expected": int(expected_top1.item()),
        "actual": int(actual_top1.item()),
    }
    tie_hidden = torch.zeros(1, 128, device=device, dtype=dtype)
    tie_weight = torch.zeros(257, 128, device=device, dtype=dtype)
    tie_top1 = lm_head_argmax(tie_hidden, tie_weight)
    results["lm_head_lowest_index_tie"] = {
        "exact": int(tie_top1.item()) == 0,
        "actual": int(tie_top1.item()),
    }

    for seq_len in (1, 17):
        channels = 64
        x = torch.randn(1, seq_len, channels, device=device, dtype=dtype)
        conv_weight = torch.randn(channels, 4, device=device, dtype=dtype)
        initial = torch.randn(1, channels, 4, device=device, dtype=dtype)
        reference_state = initial.clone()
        actual_state = initial.clone()
        expected = reference_conv(x, conv_weight, reference_state)
        actual = causal_conv1d_fused(x, conv_weight, actual_state)
        results[f"conv_{seq_len}"] = error(actual, expected)
        results[f"conv_state_{seq_len}"] = error(actual_state, reference_state)

        packed_x = torch.randn(1, seq_len, channels + 32, device=device, dtype=dtype)
        reference_state = initial.clone()
        actual_state = initial.clone()
        expected = reference_conv(packed_x[..., :channels].contiguous(), conv_weight, reference_state)
        actual = causal_conv1d_fused(packed_x, conv_weight, actual_state)
        results[f"packed_conv_{seq_len}"] = error(actual, expected)
        results[f"packed_conv_state_{seq_len}"] = error(actual_state, reference_state)

    batch, seq_len, heads, dim = 1, 3, 16, 128
    qkv_width = 3 * heads * dim
    packed_width = qkv_width + heads * dim + 2 * heads
    qkv = torch.randn(batch, seq_len, qkv_width, device=device, dtype=dtype)
    packed_delta = torch.randn(batch, seq_len, packed_width, device=device, dtype=dtype)
    packed_delta[..., :qkv_width].copy_(qkv)
    a_log = torch.randn(heads, device=device, dtype=dtype)
    dt_bias = torch.randn(heads, device=device, dtype=dtype)
    initial_state = torch.randn(batch, heads, dim, dim, device=device, dtype=torch.float32) * 0.01
    reference_state = initial_state.clone()
    actual_state = initial_state.clone()
    expected = reference_delta(qkv, packed_delta, a_log, dt_bias, reference_state)
    actual = delta_recurrent_fused(qkv, packed_delta, a_log, dt_bias, actual_state, block_v=8)
    results["delta_output"] = error(actual, expected)
    results["delta_state"] = error(actual_state, reference_state)
    precomputed_state = initial_state.clone()
    precomputed = delta_recurrent_fused(
        qkv,
        packed_delta,
        a_log,
        dt_bias,
        precomputed_state,
        block_v=8,
        precompute_factors=True,
    )
    results["delta_precomputed_output"] = error(precomputed, expected)
    results["delta_precomputed_state"] = error(
        precomputed_state,
        reference_state,
    )

    timing_x = torch.randn(1, 2048, device=device, dtype=dtype)
    timing_w = torch.randn(2048, device=device, dtype=dtype)
    results["latency_ms"] = {
        "rms_eager": time_ms(lambda: reference_rms(timing_x, timing_w)),
        "rms_fused": time_ms(lambda: rms_norm(timing_x, timing_w)),
        "residual_norm_eager": time_ms(
            lambda: reference_rms(timing_x + timing_x, timing_w)
        ),
        "residual_norm_fused": time_ms(
            lambda: residual_add_rms_norm(timing_x, timing_x, timing_w)
        ),
    }

    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
