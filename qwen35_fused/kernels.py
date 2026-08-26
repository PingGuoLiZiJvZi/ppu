"""Triton kernels used by the Qwen3.5 fused inference path.

The wrappers intentionally accept ordinary torch tensors so every optimized
module can fall back without changing the Transformers public interface.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _rows_and_width(x: torch.Tensor) -> tuple[int, int]:
    if not x.is_cuda:
        raise ValueError("fused kernels require a CUDA-compatible PPU tensor")
    if not x.is_contiguous():
        raise ValueError("fused kernels require contiguous input")
    return x.numel() // x.shape[-1], x.shape[-1]


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    norm = x * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = norm * (1.0 + weight)
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 RMSNorm: FP32 reduction and direct BF16 output."""

    rows, width = _rows_and_width(x)
    if weight.numel() != width:
        raise ValueError(f"weight has {weight.numel()} values, expected {width}")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _rms_norm_kernel[(rows,)](x, weight, out, width, eps, block, num_warps=8 if block >= 2048 else 4)
    return out


@triton.jit
def _residual_add_rms_norm_kernel(
    residual_ptr,
    update_ptr,
    weight_ptr,
    residual_out_ptr,
    norm_out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Eager BF16 residual addition rounds before the following RMSNorm.
    added_bf16 = (residual + update).to(tl.bfloat16)
    tl.store(residual_out_ptr + offsets, added_bf16, mask=mask)

    added = added_bf16.to(tl.float32)
    variance = tl.sum(added * added, axis=0) / n_cols
    norm = added * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(norm_out_ptr + offsets, norm * (1.0 + weight), mask=mask)


def residual_add_rms_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + update, RMSNorm(residual + update))`` in one launch."""

    if residual.shape != update.shape or residual.dtype != update.dtype:
        raise ValueError("residual and update must have identical shape and dtype")
    rows, width = _rows_and_width(residual)
    if not update.is_contiguous():
        raise ValueError("update must be contiguous")
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    block = triton.next_power_of_2(width)
    _residual_add_rms_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        residual_out,
        norm_out,
        width,
        eps,
        block,
        num_warps=8 if block >= 2048 else 4,
    )
    return residual_out, norm_out


@triton.jit
def _gated_rms_norm_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    norm_bf16 = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    weighted_bf16 = (norm_bf16 * weight).to(tl.bfloat16)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, weighted_bf16.to(tl.float32) * silu, mask=mask)


def gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Qwen3.5 gated RMSNorm with eager-compatible intermediate rounding."""

    if x.shape != gate.shape:
        raise ValueError("x and gate must have identical shape")
    rows, width = _rows_and_width(x)
    if not gate.is_contiguous():
        raise ValueError("gate must be contiguous")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _gated_rms_norm_kernel[(rows,)](x, gate, weight, out, width, eps, block, num_warps=4)
    return out


@triton.jit
def _silu_and_mul_kernel(packed_ptr, out_ptr, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < width
    row_base = row * width * 2
    gate = tl.load(packed_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(packed_ptr + row_base + width + cols, mask=mask, other=0.0).to(tl.bfloat16)
    silu_bf16 = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    tl.store(out_ptr + row * width + cols, silu_bf16 * up, mask=mask)


def silu_and_mul(packed: torch.Tensor) -> torch.Tensor:
    """Apply SiLU to the first half and multiply the second half."""

    rows, packed_width = _rows_and_width(packed)
    if packed_width % 2:
        raise ValueError("packed last dimension must be even")
    width = packed_width // 2
    out = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    block = triton.next_power_of_2(width)
    _silu_and_mul_kernel[(rows,)](packed, out, width, block, num_warps=8)
    return out


@triton.jit
def _ppu_swiglu_gemv_kernel(
    x_ptr,
    packed_weight_ptr,
    out_ptr,
    output_width: tl.constexpr,
    input_width: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """M=1 BF16 GEMV with the Qwen SwiGLU epilogue kept in registers."""

    block_id = tl.program_id(0)
    output_indices = block_id * block_n + tl.arange(0, block_n)
    output_mask = output_indices < output_width
    gate_accumulator = tl.zeros((block_n,), tl.float32)
    up_accumulator = tl.zeros((block_n,), tl.float32)
    for k_start in range(0, input_width, block_k):
        k = k_start + tl.arange(0, block_k)
        k_mask = k < input_width
        x = tl.load(x_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        gate_weight = tl.load(
            packed_weight_ptr + output_indices[:, None] * input_width + k[None, :],
            mask=output_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        up_weight = tl.load(
            packed_weight_ptr
            + (output_width + output_indices[:, None]) * input_width
            + k[None, :],
            mask=output_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_accumulator += tl.sum(gate_weight * x[None, :], axis=1)
        up_accumulator += tl.sum(up_weight * x[None, :], axis=1)

    # Match F.linear(BF16) followed by the existing eager-compatible fused
    # SiLU kernel: both projections and SiLU are rounded before multiplication.
    gate_bf16 = gate_accumulator.to(tl.bfloat16)
    up_bf16 = up_accumulator.to(tl.bfloat16)
    gate = gate_bf16.to(tl.float32)
    silu_bf16 = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    tl.store(
        out_ptr + output_indices,
        silu_bf16 * up_bf16,
        mask=output_mask,
    )


def ppu_swiglu_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    block_n: int = 8,
) -> torch.Tensor:
    """PPU decode-only packed gate/up projection with fused SwiGLU."""

    if x.ndim != 3 or x.shape[:2] != (1, 1) or x.dtype != torch.bfloat16:
        raise ValueError("PPU SwiGLU GEMV expects BF16 x[1,1,K]")
    if packed_weight.ndim != 2 or packed_weight.shape[0] % 2:
        raise ValueError("packed SwiGLU weight must have shape [2N,K]")
    if packed_weight.shape[1] != x.shape[-1] or packed_weight.dtype != x.dtype:
        raise ValueError("PPU SwiGLU GEMV input and weight dimensions do not match")
    if not x.is_contiguous() or not packed_weight.is_contiguous():
        raise ValueError("PPU SwiGLU GEMV tensors must be contiguous")
    if block_n not in (2, 4, 8):
        raise ValueError("PPU SwiGLU GEMV block_n must be 2, 4 or 8")
    output_width = packed_weight.shape[0] // 2
    output = torch.empty((1, 1, output_width), device=x.device, dtype=x.dtype)
    _ppu_swiglu_gemv_kernel[(triton.cdiv(output_width, block_n),)](
        x,
        packed_weight,
        output,
        output_width,
        x.shape[-1],
        block_n,
        128,
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _sigmoid_mul_kernel(x_ptr, gate_ptr, out_ptr, n_elements: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    tl.store(out_ptr + offsets, x.to(tl.bfloat16) * sigmoid_bf16, mask=mask)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if x.shape != gate.shape or not x.is_contiguous() or not gate.is_contiguous():
        raise ValueError("x and gate must be contiguous tensors with identical shape")
    out = torch.empty_like(x)
    block = 256
    _sigmoid_mul_kernel[(triton.cdiv(x.numel(), block),)](x, gate, out, x.numel(), block, num_warps=4)
    return out


@triton.jit
def _attention_gate_mul_kernel(
    x_ptr,
    packed_ptr,
    out_ptr,
    n_elements,
    hidden: tl.constexpr,
    packed_stride: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    token = offsets // hidden
    col = offsets % hidden
    head = col // head_dim
    head_col = col % head_dim
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    gate = tl.load(
        packed_ptr + token * packed_stride + head * 2 * head_dim + head_dim + head_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    tl.store(out_ptr + offsets, x * sigmoid_bf16, mask=mask)


def attention_gate_mul(x: torch.Tensor, packed_qgkv: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Multiply attention output by a gate stored inside packed QGKV output."""

    if not x.is_contiguous() or not packed_qgkv.is_contiguous():
        raise ValueError("attention gate tensors must be contiguous")
    hidden = x.shape[-1]
    if packed_qgkv.shape[:-1] != x.shape[:-1]:
        raise ValueError("attention output and packed projection rows do not match")
    out = torch.empty_like(x)
    block = 256
    _attention_gate_mul_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        packed_qgkv,
        out,
        x.numel(),
        hidden,
        packed_qgkv.shape[-1],
        head_dim,
        block,
        num_warps=4,
    )
    return out


@triton.jit
def _vision_qkv_rope_kernel(
    packed_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    tokens,
    heads: tl.constexpr,
    dim: tl.constexpr,
    packed_stride: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % heads
    token = pid // heads
    cols = tl.arange(0, block)
    mask = (token < tokens) & (cols < dim)
    half = dim // 2
    partner = tl.where(cols < half, cols + half, cols - half)
    sign = tl.where(cols < half, -1.0, 1.0)

    token_base = token * packed_stride
    q_base = token_base + head * dim
    k_base = token_base + heads * dim + head * dim
    v_base = token_base + 2 * heads * dim + head * dim
    q = tl.load(packed_ptr + q_base + cols, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(packed_ptr + k_base + cols, mask=mask, other=0.0).to(tl.float32)
    q_partner = tl.load(packed_ptr + q_base + partner, mask=mask, other=0.0).to(tl.float32)
    k_partner = tl.load(packed_ptr + k_base + partner, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + token * dim + cols, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * dim + cols, mask=mask, other=0.0).to(tl.float32)

    output_offset = (head * tokens + token) * dim + cols
    tl.store(q_out_ptr + output_offset, q * cos + q_partner * sign * sin, mask=mask)
    tl.store(k_out_ptr + output_offset, k * cos + k_partner * sign * sin, mask=mask)
    value = tl.load(packed_ptr + v_base + cols, mask=mask, other=0.0)
    tl.store(v_out_ptr + output_offset, value, mask=mask)


def vision_qkv_rope(
    packed_qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    heads: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed Vision QKV to BHLD and apply RoPE to Q/K in one launch."""

    if packed_qkv.ndim != 2 or not packed_qkv.is_contiguous():
        raise ValueError("packed Vision QKV must be contiguous [tokens,3*hidden]")
    tokens, packed_width = packed_qkv.shape
    if packed_width != 3 * heads * dim:
        raise ValueError(f"packed Vision QKV width {packed_width}, expected {3 * heads * dim}")
    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.shape != (tokens, dim) or sin.shape != cos.shape:
        raise ValueError(f"Vision RoPE shape {tuple(cos.shape)}, expected {(tokens, dim)}")
    output_shape = (1, heads, tokens, dim)
    q = torch.empty(output_shape, device=packed_qkv.device, dtype=packed_qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    block = triton.next_power_of_2(dim)
    _vision_qkv_rope_kernel[(tokens * heads,)](
        packed_qkv,
        cos,
        sin,
        q,
        k,
        v,
        tokens,
        heads,
        dim,
        packed_width,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k, v


@triton.jit
def _qgkv_norm_rope_kernel(
    packed_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    cache_start_ptr,
    seq_len,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    packed_stride: tl.constexpr,
    cache_len,
    write_cache: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % q_heads
    token_flat = pid // q_heads
    token = token_flat % seq_len
    batch_idx = token_flat // seq_len
    cols = tl.arange(0, block)
    mask = cols < dim
    base = token_flat * packed_stride

    q_head_base = base + head * 2 * dim
    q = tl.load(packed_ptr + q_head_base + cols, mask=mask, other=0.0).to(tl.float32)
    q_variance = tl.sum(q * q, axis=0) / dim
    q_weight = tl.load(q_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    q_norm = (q * tl.rsqrt(q_variance + 1.0e-6) * (1.0 + q_weight)).to(tl.bfloat16)

    half_rotary = rotary_dim // 2
    partner_cols = tl.where(cols < half_rotary, cols + half_rotary, cols - half_rotary)
    partner_mask = cols < rotary_dim
    q_partner = tl.load(
        packed_ptr + q_head_base + partner_cols, mask=partner_mask, other=0.0
    ).to(tl.float32)
    partner_weight = tl.load(q_weight_ptr + partner_cols, mask=partner_mask, other=0.0).to(tl.float32)
    q_partner_norm = (
        q_partner * tl.rsqrt(q_variance + 1.0e-6) * (1.0 + partner_weight)
    ).to(tl.bfloat16)
    rotate_sign = tl.where(cols < half_rotary, -1.0, 1.0)
    cos = tl.load(cos_ptr + token_flat * rotary_dim + cols, mask=partner_mask, other=1.0).to(tl.bfloat16)
    sin = tl.load(sin_ptr + token_flat * rotary_dim + cols, mask=partner_mask, other=0.0).to(tl.bfloat16)
    q_cos = (q_norm * cos).to(tl.bfloat16)
    q_rot_half = (q_partner_norm * rotate_sign).to(tl.bfloat16)
    q_sin = (q_rot_half * sin).to(tl.bfloat16)
    q_rotated = (q_cos + q_sin).to(tl.bfloat16)
    q_final = tl.where(cols < rotary_dim, q_rotated, q_norm)
    q_offset = ((batch_idx * q_heads + head) * seq_len + token) * dim + cols
    tl.store(q_out_ptr + q_offset, q_final, mask=mask)

    if head < kv_heads:
        k_base = base + 2 * q_heads * dim + head * dim
        v_base = base + 2 * q_heads * dim + kv_heads * dim + head * dim
        k = tl.load(packed_ptr + k_base + cols, mask=mask, other=0.0).to(tl.float32)
        k_variance = tl.sum(k * k, axis=0) / dim
        k_weight = tl.load(k_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_norm = (k * tl.rsqrt(k_variance + 1.0e-6) * (1.0 + k_weight)).to(tl.bfloat16)
        k_partner = tl.load(packed_ptr + k_base + partner_cols, mask=partner_mask, other=0.0).to(tl.float32)
        k_partner_weight = tl.load(
            k_weight_ptr + partner_cols, mask=partner_mask, other=0.0
        ).to(tl.float32)
        k_partner_norm = (
            k_partner * tl.rsqrt(k_variance + 1.0e-6) * (1.0 + k_partner_weight)
        ).to(tl.bfloat16)
        k_cos = (k_norm * cos).to(tl.bfloat16)
        k_rot_half = (k_partner_norm * rotate_sign).to(tl.bfloat16)
        k_sin = (k_rot_half * sin).to(tl.bfloat16)
        k_rotated = (k_cos + k_sin).to(tl.bfloat16)
        k_final = tl.where(cols < rotary_dim, k_rotated, k_norm)
        if write_cache:
            cache_start = tl.load(cache_start_ptr)
            kv_offset = (
                (batch_idx * kv_heads + head) * cache_len + cache_start + token
            ) * dim + cols
        else:
            kv_offset = ((batch_idx * kv_heads + head) * seq_len + token) * dim + cols
        tl.store(k_out_ptr + kv_offset, k_final, mask=mask)
        value = tl.load(packed_ptr + v_base + cols, mask=mask, other=0.0)
        tl.store(v_out_ptr + kv_offset, value, mask=mask)


def qgkv_norm_rope(
    packed_qgkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    dim: int,
    rotary_dim: int,
    cache_key: torch.Tensor | None = None,
    cache_value: torch.Tensor | None = None,
    cache_start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layout packed Q+gate/K/V and fuse Q/K RMSNorm with partial RoPE."""

    if not packed_qgkv.is_contiguous():
        raise ValueError("packed QGKV must be contiguous")
    batch, seq_len, packed_width = packed_qgkv.shape
    expected = 2 * q_heads * dim + 2 * kv_heads * dim
    if packed_width != expected:
        raise ValueError(f"packed QGKV width {packed_width}, expected {expected}")
    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.shape != (batch, seq_len, rotary_dim) or sin.shape != cos.shape:
        raise ValueError(f"RoPE shape {tuple(cos.shape)}, expected {(batch, seq_len, rotary_dim)}")
    q = torch.empty((batch, q_heads, seq_len, dim), device=packed_qgkv.device, dtype=packed_qgkv.dtype)
    write_cache = cache_key is not None
    if write_cache:
        if cache_value is None or cache_start is None:
            raise ValueError("cache key, value and start must be provided together")
        if cache_key.shape[:2] != (batch, kv_heads) or cache_key.shape[-1] != dim:
            raise ValueError("static K cache shape does not match QGKV")
        if cache_value.shape != cache_key.shape:
            raise ValueError("static K/V cache shapes must match")
        k = cache_key
        v = cache_value
        cache_len = cache_key.shape[-2]
    else:
        k = torch.empty((batch, kv_heads, seq_len, dim), device=packed_qgkv.device, dtype=packed_qgkv.dtype)
        v = torch.empty_like(k)
        cache_start = torch.zeros((), device=packed_qgkv.device, dtype=torch.int64)
        cache_len = seq_len
    block = triton.next_power_of_2(dim)
    _qgkv_norm_rope_kernel[(batch * seq_len * q_heads,)](
        packed_qgkv,
        q_weight,
        k_weight,
        cos,
        sin,
        q,
        k,
        v,
        cache_start,
        seq_len,
        q_heads,
        kv_heads,
        dim,
        rotary_dim,
        packed_width,
        cache_len,
        write_cache,
        block,
        num_warps=4,
    )
    return q, k, v


@triton.jit
def _causal_conv1d_kernel(
    x_ptr,
    weight_ptr,
    state_ptr,
    out_ptr,
    batch: tl.constexpr,
    seq_len,
    channels: tl.constexpr,
    input_stride: tl.constexpr,
    kernel_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    total = batch * seq_len * channels
    mask = offsets < total
    channel = offsets % channels
    token_flat = offsets // channels
    token = token_flat % seq_len
    batch_idx = token_flat // seq_len

    acc = tl.zeros((block,), tl.float32)
    for tap in tl.static_range(0, kernel_size):
        source_token = token - (kernel_size - 1 - tap)
        source_mask = mask & (source_token >= 0)
        source_offset = (batch_idx * seq_len + source_token) * input_stride + channel
        value = tl.load(x_ptr + source_offset, mask=source_mask, other=0.0).to(tl.float32)
        conv_weight = tl.load(weight_ptr + channel * kernel_size + tap, mask=mask, other=0.0).to(tl.float32)
        acc += value * conv_weight

    # conv1d rounds to BF16 before the separate SiLU in the eager path.
    conv_bf16 = acc.to(tl.bfloat16)
    conv = conv_bf16.to(tl.float32)
    tl.store(out_ptr + offsets, conv / (1.0 + tl.exp(-conv)), mask=mask)

    state_pos = token - (seq_len - kernel_size)
    state_mask = mask & (state_pos >= 0) & (state_pos < kernel_size)
    state_offset = (batch_idx * channels + channel) * kernel_size + state_pos
    current_input_offset = (batch_idx * seq_len + token) * input_stride + channel
    tl.store(
        state_ptr + state_offset,
        tl.load(x_ptr + current_input_offset, mask=state_mask, other=0.0),
        mask=state_mask,
    )


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,
    weight_ptr,
    state_ptr,
    out_ptr,
    total_elements,
    channels: tl.constexpr,
    input_stride: tl.constexpr,
    kernel_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    channel_mask = offsets < total_elements
    channel = offsets % channels
    batch_idx = offsets // channels
    valid = channel_mask & (channel < channels)
    state_base = (batch_idx * channels + channel) * kernel_size

    acc = tl.zeros((block,), tl.float32)
    for tap in tl.static_range(0, kernel_size - 1):
        value = tl.load(state_ptr + state_base + tap + 1, mask=valid, other=0.0)
        tl.store(state_ptr + state_base + tap, value, mask=valid)
        weight = tl.load(weight_ptr + channel * kernel_size + tap, mask=valid, other=0.0).to(tl.float32)
        acc += value.to(tl.float32) * weight

    x = tl.load(x_ptr + batch_idx * input_stride + channel, mask=valid, other=0.0)
    tl.store(state_ptr + state_base + kernel_size - 1, x, mask=valid)
    last_weight = tl.load(
        weight_ptr + channel * kernel_size + kernel_size - 1, mask=valid, other=0.0
    ).to(tl.float32)
    acc += x.to(tl.float32) * last_weight
    conv_bf16 = acc.to(tl.bfloat16)
    conv = conv_bf16.to(tl.float32)
    tl.store(out_ptr + batch_idx * channels + channel, conv / (1.0 + tl.exp(-conv)), mask=valid)


def causal_conv1d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Causal depthwise conv + SiLU, updating ``state`` in place.

    ``x`` is ``[B, L, C]``, ``weight`` is ``[C, K]`` and ``state`` is
    ``[B, C, K]``. The decode path uses a dedicated in-place update kernel.
    """

    if x.ndim != 3 or weight.ndim != 2 or state.ndim != 3:
        raise ValueError("expected x[B,L,C], weight[C,K], state[B,C,K]")
    if not x.is_contiguous() or not weight.is_contiguous() or not state.is_contiguous():
        raise ValueError("causal conv tensors must be contiguous")
    batch, seq_len, input_stride = x.shape
    channels = weight.shape[0]
    kernel_size = weight.shape[1]
    if input_stride < channels or state.shape != (batch, channels, kernel_size):
        raise ValueError("causal conv shapes do not match")
    out = torch.empty((batch, seq_len, channels), device=x.device, dtype=x.dtype)
    block = 256
    if seq_len == 1:
        total = batch * channels
        _causal_conv1d_update_kernel[(triton.cdiv(total, block),)](
            x,
            weight,
            state,
            out,
            total,
            channels,
            input_stride,
            kernel_size,
            block,
            num_warps=4,
        )
    else:
        total = batch * seq_len * channels
        _causal_conv1d_kernel[(triton.cdiv(total, block),)](
            x,
            weight,
            state,
            out,
            batch,
            seq_len,
            channels,
            input_stride,
            kernel_size,
            block,
            num_warps=4,
        )
    return out


@triton.jit(do_not_specialize_on_alignment=["seq_len"])
def _delta_prepare_factors_kernel(
    qkv_ptr,
    packed_ptr,
    a_log_ptr,
    dt_bias_ptr,
    factor_ptr,
    seq_len,
    heads: tl.constexpr,
    dim: tl.constexpr,
    qkv_stride: tl.constexpr,
    packed_stride: tl.constexpr,
    a_offset: tl.constexpr,
    b_offset: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute tile-invariant DeltaNet scalars once per token and head."""

    pid = tl.program_id(0)
    head = pid % heads
    token_flat = pid // heads
    token = token_flat % seq_len
    qkv_base = token_flat * qkv_stride
    packed_base = token_flat * packed_stride
    k_idx = tl.arange(0, block_k)
    k_mask = k_idx < dim
    q = tl.load(
        qkv_ptr + qkv_base + head * dim + k_idx,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        qkv_ptr + qkv_base + heads * dim + head * dim + k_idx,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    # Preserve the v2 arithmetic order in the recurrent kernel. Folding the
    # scale into q_inv here changes FP32 association and can flip close logits.
    q_inv = tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
    k_inv = tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)
    a = (
        tl.load(packed_ptr + packed_base + a_offset + head).to(tl.float32)
        + tl.load(dt_bias_ptr + head).to(tl.float32)
    )
    softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    alpha = tl.exp(-tl.exp(tl.load(a_log_ptr + head).to(tl.float32)) * softplus)
    b = tl.load(packed_ptr + packed_base + b_offset + head).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-b))
    factor_base = pid * 4
    tl.store(factor_ptr + factor_base, q_inv)
    tl.store(factor_ptr + factor_base + 1, k_inv)
    tl.store(factor_ptr + factor_base + 2, alpha)
    tl.store(factor_ptr + factor_base + 3, beta)


@triton.jit(do_not_specialize_on_alignment=["seq_len"])
def _delta_recurrent_kernel(
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
    if not use_precomputed_factors:
        a_log = tl.load(a_log_ptr + head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)

    for token in range(0, seq_len):
        qkv_base = (batch_idx * seq_len + token) * qkv_stride
        packed_base = (batch_idx * seq_len + token) * packed_stride
        q = tl.load(qkv_ptr + qkv_base + head * dim + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k = tl.load(
            qkv_ptr + qkv_base + heads * dim + head * dim + k_idx, mask=k_mask, other=0.0
        ).to(tl.float32)
        value = tl.load(
            qkv_ptr + qkv_base + 2 * heads * dim + head * dim + v_idx, mask=v_mask, other=0.0
        ).to(tl.float32)

        if use_precomputed_factors:
            factor_base = ((batch_idx * seq_len + token) * heads + head) * 4
            q_inv = tl.load(factor_ptr + factor_base)
            k_inv = tl.load(factor_ptr + factor_base + 1)
            alpha = tl.load(factor_ptr + factor_base + 2)
            beta = tl.load(factor_ptr + factor_base + 3)
            q_hat = q * q_inv * (1.0 / math.sqrt(dim))
            k_hat = k * k_inv
        else:
            q_inv = tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
            k_inv = tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)
            q_hat = q * q_inv * (1.0 / math.sqrt(dim))
            k_hat = k * k_inv
            a = (
                tl.load(packed_ptr + packed_base + a_offset + head).to(tl.float32)
                + dt_bias
            )
            softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
            g = -tl.exp(a_log) * softplus
            alpha = tl.exp(g)
            b = tl.load(packed_ptr + packed_base + b_offset + head).to(tl.float32)
            beta = 1.0 / (1.0 + tl.exp(-b))

        state *= alpha
        memory_k = tl.sum(state * k_hat[None, :], axis=1)
        delta = beta * (value - memory_k)
        state += delta[:, None] * k_hat[None, :]
        output = tl.sum(state * q_hat[None, :], axis=1)
        output_offsets = ((batch_idx * seq_len + token) * heads + head) * dim + v_idx
        tl.store(out_ptr + output_offsets, output, mask=v_mask)

    tl.store(state_ptr + state_offsets, state, mask=matrix_mask)


def delta_recurrent_fused(
    qkv: torch.Tensor,
    packed_projection: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    block_v: int = 8,
    precompute_factors: bool = False,
) -> torch.Tensor:
    """Sequential gated-delta recurrence for prefill and single-token decode.

    The state is updated in place. ``qkv`` has shape ``[B,L,3*H*D]`` and
    ``packed_projection`` stores ``[qkv,z,a,b]`` in its last dimension.
    """

    if qkv.ndim != 3 or packed_projection.ndim != 3 or state.ndim != 4:
        raise ValueError("invalid delta recurrent tensor rank")
    if not qkv.is_contiguous() or not packed_projection.is_contiguous() or not state.is_contiguous():
        raise ValueError("delta recurrent tensors must be contiguous")
    batch, seq_len, qkv_width = qkv.shape
    _, heads, dim, value_dim = state.shape
    if dim != value_dim or qkv_width != 3 * heads * dim:
        raise ValueError("delta recurrent dimensions do not match")
    packed_width = packed_projection.shape[-1]
    z_offset = qkv_width
    a_offset = z_offset + heads * dim
    b_offset = a_offset + heads
    if packed_width < b_offset + heads:
        raise ValueError(f"packed projection width {packed_width} is smaller than {b_offset + heads}")
    out = torch.empty((batch, seq_len, heads, dim), device=qkv.device, dtype=qkv.dtype)
    use_precomputed_factors = bool(precompute_factors and seq_len > 1)
    if use_precomputed_factors:
        factors = torch.empty(
            (batch, seq_len, heads, 4),
            device=qkv.device,
            dtype=torch.float32,
        )
        _delta_prepare_factors_kernel[(batch * seq_len * heads,)](
            qkv,
            packed_projection,
            a_log,
            dt_bias,
            factors,
            seq_len,
            heads,
            dim,
            qkv_width,
            packed_width,
            a_offset,
            b_offset,
            triton.next_power_of_2(dim),
            num_warps=4,
            num_stages=1,
        )
    else:
        # Triton pointer arguments cannot be None. The constexpr branch makes
        # this alias unreachable in the original v2 kernel specialization.
        factors = qkv
    grid = (batch * heads * triton.cdiv(dim, block_v),)
    _delta_recurrent_kernel[grid](
        qkv,
        packed_projection,
        a_log,
        dt_bias,
        state,
        factors,
        out,
        batch,
        seq_len,
        heads,
        dim,
        qkv_width,
        packed_width,
        z_offset,
        a_offset,
        b_offset,
        block_v,
        triton.next_power_of_2(dim),
        use_precomputed_factors,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _gqa_decode_attn_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    len_ptr,
    acc_ptr,
    ml_ptr,
    cache_len,
    scale,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    splits: tl.constexpr,
    block_n: tl.constexpr,
):
    """Maskless split-KV GQA attention for one decode step over a static cache.

    The valid prefix length is read from the device scalar ``len_ptr`` so the
    kernel keeps a fixed grid and loop structure inside a captured graph.
    """

    head = tl.program_id(0)
    split = tl.program_id(1)
    group: tl.constexpr = q_heads // kv_heads
    kv_head = head // group
    d = tl.arange(0, head_dim)

    q = tl.load(q_ptr + head * head_dim + d).to(tl.float32) * scale
    n_valid = tl.load(len_ptr).to(tl.int32)
    chunk = tl.cdiv(cache_len, splits)
    n_start = split * chunk
    n_end = tl.minimum(n_start + chunk, n_valid)

    kv_base = kv_head * cache_len * head_dim
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((head_dim,), tl.float32)

    for start in range(n_start, n_end, block_n):
        n = start + tl.arange(0, block_n)
        mask = n < n_end
        k = tl.load(
            k_ptr + kv_base + n[:, None] * head_dim + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        v = tl.load(
            v_ptr + kv_base + n[:, None] * head_dim + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    slot = head * splits + split
    tl.store(acc_ptr + slot * head_dim + d, acc)
    tl.store(ml_ptr + slot * 2, m_i)
    tl.store(ml_ptr + slot * 2 + 1, l_i)


@triton.jit
def _gqa_decode_attn_combine_kernel(
    acc_ptr,
    ml_ptr,
    out_ptr,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    splits: tl.constexpr,
):
    head = tl.program_id(0)
    d = tl.arange(0, head_dim)
    m_max = -float("inf")
    for split in range(splits):
        m_max = tl.maximum(m_max, tl.load(ml_ptr + (head * splits + split) * 2))
    total = tl.zeros((head_dim,), tl.float32)
    l_total = 0.0
    for split in range(splits):
        slot = head * splits + split
        weight = tl.exp(tl.load(ml_ptr + slot * 2) - m_max)
        total += tl.load(acc_ptr + slot * head_dim + d) * weight
        l_total += tl.load(ml_ptr + slot * 2 + 1) * weight
    tl.store(out_ptr + head * head_dim + d, (total / l_total).to(tl.bfloat16))


def gqa_decode_attention(
    q: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    valid_length: torch.Tensor,
    scale: float,
    *,
    splits: int = 4,
    block_n: int = 64,
) -> torch.Tensor:
    """Compute one greedy decode attention step without a mask or GQA expansion.

    ``q`` is ``[1, H, 1, D]`` BF16; ``cache_keys``/``cache_values`` are static
    cache tensors ``[1, KVH, L, D]``; ``valid_length`` is a device scalar that
    already includes the current token. Returns ``[1, 1, H*D]`` BF16.
    """

    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError("GQA decode attention expects q[1,H,1,D]")
    if cache_keys.ndim != 4 or cache_keys.shape[0] != 1:
        raise ValueError("static K cache must be [1,KVH,L,D]")
    if cache_values.shape != cache_keys.shape:
        raise ValueError("static K/V cache shapes must match")
    if not q.is_contiguous() or not cache_keys.is_contiguous() or not cache_values.is_contiguous():
        raise ValueError("GQA decode attention tensors must be contiguous")
    heads, head_dim = q.shape[1], q.shape[3]
    kv_heads, cache_len = cache_keys.shape[1], cache_keys.shape[2]
    if head_dim != cache_keys.shape[3] or heads % kv_heads:
        raise ValueError("GQA decode attention head layout does not match cache")
    slots = heads * splits
    acc = torch.empty((slots, head_dim), device=q.device, dtype=torch.float32)
    ml = torch.empty((slots, 2), device=q.device, dtype=torch.float32)
    out = torch.empty((1, 1, heads * head_dim), device=q.device, dtype=q.dtype)
    _gqa_decode_attn_splitkv_kernel[(heads, splits)](
        q,
        cache_keys,
        cache_values,
        valid_length,
        acc,
        ml,
        cache_len,
        scale,
        heads,
        kv_heads,
        head_dim,
        splits,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    _gqa_decode_attn_combine_kernel[(heads,)](
        acc,
        ml,
        out,
        heads,
        head_dim,
        splits,
        num_warps=4,
    )
    return out


@triton.jit
def _lm_head_block_top1_kernel(
    hidden_ptr,
    weight_ptr,
    candidate_value_ptr,
    candidate_index_ptr,
    vocab_size,
    hidden_size: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    block_id = tl.program_id(0)
    vocab = block_id * block_n + tl.arange(0, block_n)
    vocab_mask = vocab < vocab_size
    accumulator = tl.zeros((block_n,), tl.float32)
    for k_start in tl.static_range(0, hidden_size, block_k):
        k = k_start + tl.arange(0, block_k)
        k_mask = k < hidden_size
        hidden = tl.load(hidden_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        weight = tl.load(
            weight_ptr + vocab[:, None] * hidden_size + k[None, :],
            mask=vocab_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(weight * hidden[None, :], axis=1)

    # Eager materializes BF16 logits before argmax.
    rounded = accumulator.to(tl.bfloat16)
    comparable = tl.where(vocab_mask, rounded.to(tl.float32), -float("inf"))
    max_value = tl.max(comparable, axis=0)
    tied_index = tl.where(comparable == max_value, vocab, 0x7FFFFFFF)
    min_index = tl.min(tied_index, axis=0)
    tl.store(candidate_value_ptr + block_id, max_value)
    tl.store(candidate_index_ptr + block_id, min_index)


@triton.jit
def _lm_head_final_top1_kernel(
    candidate_value_ptr,
    candidate_index_ptr,
    output_ptr,
    n_candidates: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.arange(0, block)
    mask = offsets < n_candidates
    values = tl.load(candidate_value_ptr + offsets, mask=mask, other=-float("inf"))
    indices = tl.load(candidate_index_ptr + offsets, mask=mask, other=0x7FFFFFFF)
    max_value = tl.max(values, axis=0)
    tied_index = tl.where(values == max_value, indices, 0x7FFFFFFF)
    tl.store(output_ptr, tl.min(tied_index, axis=0))


def lm_head_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 128,
) -> torch.Tensor:
    """Compute greedy token id without materializing the BF16 logits tensor."""

    if hidden.shape != (1, weight.shape[1]) or hidden.dtype != torch.bfloat16:
        raise ValueError("lm_head_argmax supports BF16 batch=1 hidden vectors")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("LM head tensors must be contiguous")
    vocab_size, hidden_size = weight.shape
    candidates = triton.cdiv(vocab_size, block_n)
    candidate_values = torch.empty(candidates, device=hidden.device, dtype=torch.float32)
    candidate_indices = torch.empty(candidates, device=hidden.device, dtype=torch.int32)
    output = torch.empty((1, 1), device=hidden.device, dtype=torch.int64)
    _lm_head_block_top1_kernel[(candidates,)](
        hidden,
        weight,
        candidate_values,
        candidate_indices,
        vocab_size,
        hidden_size,
        block_n,
        block_k,
        num_warps=4,
        num_stages=1,
    )
    final_block = triton.next_power_of_2(candidates)
    _lm_head_final_top1_kernel[(1,)](
        candidate_values,
        candidate_indices,
        output,
        candidates,
        final_block,
        num_warps=8,
    )
    return output


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = centered * tl.rsqrt(variance + eps) * weight + bias
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    rows, width = _rows_and_width(x)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _layer_norm_kernel[(rows,)](x, weight, bias, out, width, eps, block, num_warps=8)
    return out


@triton.jit
def _residual_add_layer_norm_kernel(
    residual_ptr,
    update_ptr,
    weight_ptr,
    bias_ptr,
    residual_out_ptr,
    norm_out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    added_bf16 = (residual + update).to(tl.bfloat16)
    tl.store(residual_out_ptr + offsets, added_bf16, mask=mask)
    added = added_bf16.to(tl.float32)
    mean = tl.sum(added, axis=0) / n_cols
    centered = added - mean
    variance = tl.sum(centered * centered, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    norm = centered * tl.rsqrt(variance + eps) * weight + bias
    tl.store(norm_out_ptr + offsets, norm, mask=mask)


def residual_add_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != update.shape:
        raise ValueError("vision residual tensors must have identical shape")
    rows, width = _rows_and_width(residual)
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    block = triton.next_power_of_2(width)
    _residual_add_layer_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        bias,
        residual_out,
        norm_out,
        width,
        eps,
        block,
        num_warps=8,
    )
    return residual_out, norm_out


@triton.jit
def _position_embed_add_kernel(
    hidden_ptr,
    table_ptr,
    indices_ptr,
    interpolation_weight_ptr,
    out_ptr,
    rows,
    width: tl.constexpr,
    neighbors: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < width)
    position = tl.zeros((block,), tl.float32)
    for neighbor in tl.static_range(0, neighbors):
        index = tl.load(indices_ptr + row * neighbors + neighbor)
        interpolation_weight = tl.load(
            interpolation_weight_ptr + row * neighbors + neighbor
        ).to(tl.float32)
        embedding = tl.load(table_ptr + index * width + cols, mask=mask, other=0.0).to(tl.float32)
        position += embedding * interpolation_weight
    position_bf16 = position.to(tl.bfloat16)
    hidden = tl.load(hidden_ptr + row * width + cols, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + row * width + cols, hidden + position_bf16, mask=mask)


def position_embed_add(
    hidden: torch.Tensor,
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    interpolation_weight: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 2 or not hidden.is_contiguous():
        raise ValueError("vision hidden tensor must be contiguous [rows,width]")
    rows, width = hidden.shape
    neighbors = indices.shape[-1]
    if indices.shape != interpolation_weight.shape or indices.shape[0] != rows:
        raise ValueError("position interpolation tensors do not match hidden rows")
    out = torch.empty_like(hidden)
    block = triton.next_power_of_2(width)
    _position_embed_add_kernel[(rows,)](
        hidden,
        embedding_table,
        indices,
        interpolation_weight,
        out,
        rows,
        width,
        neighbors,
        block,
        num_warps=8,
    )
    return out
