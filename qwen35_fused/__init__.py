"""PPU-oriented fused inference path for the local Qwen3.5-2B model."""

from .integration import precompute_vision_kwargs

from .kernels import (
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

__all__ = [
    "attention_gate_mul",
    "causal_conv1d_fused",
    "delta_recurrent_fused",
    "gated_rms_norm",
    "layer_norm",
    "lm_head_argmax",
    "position_embed_add",
    "ppu_swiglu_gemv",
    "qgkv_norm_rope",
    "residual_add_layer_norm",
    "residual_add_rms_norm",
    "rms_norm",
    "sigmoid_mul",
    "silu_and_mul",
    "vision_qkv_rope",
    "precompute_vision_kwargs",
]
