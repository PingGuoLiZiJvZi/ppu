#!/usr/bin/env python3
"""Ahead-of-time compile every production Triton specialization into one cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("/root/ppu/triton"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    import torch

    from qwen35_fused.kernels import (
        attention_gate_mul,
        causal_conv1d_fused,
        delta_recurrent_fused,
        gated_rms_norm,
        layer_norm,
        lm_head_argmax,
        position_embed_add,
        ppu_swiglu_gemv,
        qgkv_norm_rope,
        residual_add_layer_norm,
        residual_add_rms_norm,
        rms_norm,
        sigmoid_mul,
        silu_and_mul,
        vision_qkv_rope,
    )

    if "PPU-ZW810E" not in torch.cuda.get_device_name(0):
        raise RuntimeError("precompile_triton.py requires PPU-ZW810E")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    text_width = 2048
    intermediate_width = 6144
    delta_heads = 16
    delta_dim = 128
    delta_qkv_width = 3 * delta_heads * delta_dim
    delta_packed_width = 8320
    attention_heads = 8
    attention_kv_heads = 2
    attention_dim = 256
    attention_rotary_dim = 64
    attention_packed_width = 5120
    vision_width = 1024
    vision_heads = 16
    vision_dim = 64

    text_norm_weight = torch.zeros(text_width, device=device, dtype=dtype)
    delta_norm_weight = torch.zeros(delta_dim, device=device, dtype=dtype)
    vision_norm_weight = torch.ones(vision_width, device=device, dtype=dtype)
    vision_norm_bias = torch.zeros(vision_width, device=device, dtype=dtype)
    attention_q_weight = torch.zeros(attention_dim, device=device, dtype=dtype)
    attention_k_weight = torch.zeros(attention_dim, device=device, dtype=dtype)
    delta_a_log = torch.zeros(delta_heads, device=device, dtype=dtype)
    delta_dt_bias = torch.zeros(delta_heads, device=device, dtype=dtype)
    conv_weight = torch.zeros(
        delta_qkv_width, 4, device=device, dtype=dtype
    )
    key_cache = torch.empty(
        (1, attention_kv_heads, 512, attention_dim),
        device=device,
        dtype=dtype,
    )
    value_cache = torch.empty_like(key_cache)
    cache_start = torch.zeros((), device=device, dtype=torch.int64)

    # Triton 3.5 specializes runtime integers into exactly three relevant
    # classes: value 1, values divisible by 16, and all remaining values.
    for seq_len in (16, 17, 1):
        hidden = torch.zeros((1, seq_len, text_width), device=device, dtype=dtype)
        update = torch.zeros_like(hidden)
        rms_norm(hidden, text_norm_weight)
        residual_add_rms_norm(hidden, update, text_norm_weight)
        silu_and_mul(
            torch.zeros(
                (1, seq_len, 2 * intermediate_width),
                device=device,
                dtype=dtype,
            )
        )
        sigmoid_mul(hidden, update)

        delta_packed = torch.zeros(
            (1, seq_len, delta_packed_width), device=device, dtype=dtype
        )
        delta_qkv = torch.zeros(
            (1, seq_len, delta_qkv_width), device=device, dtype=dtype
        )
        conv_state = torch.zeros(
            (1, delta_qkv_width, 4), device=device, dtype=dtype
        )
        causal_conv1d_fused(delta_packed, conv_weight, conv_state)
        recurrent_state = torch.zeros(
            (1, delta_heads, delta_dim, delta_dim),
            device=device,
            dtype=torch.float32,
        )
        delta_recurrent_fused(
            delta_qkv,
            delta_packed,
            delta_a_log,
            delta_dt_bias,
            recurrent_state,
            block_v=8,
            precompute_factors=seq_len > 1,
        )
        gated_rms_norm(
            torch.zeros(
                (seq_len * delta_heads, delta_dim), device=device, dtype=dtype
            ),
            torch.zeros(
                (seq_len * delta_heads, delta_dim), device=device, dtype=dtype
            ),
            delta_norm_weight,
        )

        packed_attention = torch.zeros(
            (1, seq_len, attention_packed_width), device=device, dtype=dtype
        )
        cos = torch.zeros(
            (1, seq_len, attention_rotary_dim), device=device, dtype=dtype
        )
        sin = torch.zeros_like(cos)
        qgkv_norm_rope(
            packed_attention,
            attention_q_weight,
            attention_k_weight,
            cos,
            sin,
            q_heads=attention_heads,
            kv_heads=attention_kv_heads,
            dim=attention_dim,
            rotary_dim=attention_rotary_dim,
        )
        qgkv_norm_rope(
            packed_attention,
            attention_q_weight,
            attention_k_weight,
            cos,
            sin,
            q_heads=attention_heads,
            kv_heads=attention_kv_heads,
            dim=attention_dim,
            rotary_dim=attention_rotary_dim,
            cache_key=key_cache,
            cache_value=value_cache,
            cache_start=cache_start,
        )
        attention_gate_mul(hidden, packed_attention, attention_dim)

        vision_hidden = torch.zeros(
            (seq_len, vision_width), device=device, dtype=dtype
        )
        layer_norm(
            vision_hidden, vision_norm_weight, vision_norm_bias
        )
        residual_add_layer_norm(
            vision_hidden,
            torch.zeros_like(vision_hidden),
            vision_norm_weight,
            vision_norm_bias,
        )
        position_embed_add(
            vision_hidden,
            torch.zeros((2304, vision_width), device=device, dtype=dtype),
            torch.zeros((seq_len, 4), device=device, dtype=torch.int64),
            torch.zeros((seq_len, 4), device=device, dtype=torch.float32),
        )
        vision_qkv_rope(
            torch.zeros(
                (seq_len, 3 * vision_heads * vision_dim),
                device=device,
                dtype=dtype,
            ),
            torch.zeros((seq_len, vision_dim), device=device, dtype=torch.float32),
            torch.zeros((seq_len, vision_dim), device=device, dtype=torch.float32),
            heads=vision_heads,
            dim=vision_dim,
        )

    ppu_swiglu_gemv(
        torch.zeros((1, 1, text_width), device=device, dtype=dtype),
        torch.zeros(
            (2 * intermediate_width, text_width), device=device, dtype=dtype
        ),
        block_n=8,
    )
    lm_head_argmax(
        torch.zeros((1, text_width), device=device, dtype=dtype),
        torch.zeros((248320, text_width), device=device, dtype=dtype),
    )
    torch.cuda.synchronize()

    files = sum(1 for path in cache_dir.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "cache_dir": str(cache_dir),
                "specialization_values": [16, 17, 1],
                "cache_files": files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
