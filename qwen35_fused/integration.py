"""Install the fused Qwen3.5 inference path on a loaded Transformers model."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_impl

from .kernels import (
    attention_gate_mul,
    causal_conv1d_fused,
    delta_recurrent_fused,
    gated_rms_norm,
    layer_norm,
    position_embed_add,
    qgkv_norm_rope,
    residual_add_layer_norm,
    residual_add_rms_norm,
    rms_norm,
    silu_and_mul,
    vision_qkv_rope,
)


@dataclass(frozen=True)
class FusionConfig:
    delta: bool = True
    rms_norm: bool = True
    residual_norm: bool = True
    swiglu: bool = True
    attention: bool = True
    vision: bool = True
    delta_projection_padding: int = 128
    delta_block_v: int = 8


def _is_fast_tensor(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype == torch.bfloat16


def _register_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if hasattr(module, name):
        setattr(module, name, value)
    else:
        module.register_buffer(name, value, persistent=False)


def _pack_linear_weights(model: torch.nn.Module, config: FusionConfig) -> dict[str, int]:
    text_model = model.model.language_model
    stats = {"delta": 0, "mlp": 0, "attention": 0, "vision": 0}
    for layer in text_model.layers:
        if hasattr(layer, "linear_attn") and config.delta:
            delta = layer.linear_attn
            packed = torch.cat(
                (
                    delta.in_proj_qkv.weight,
                    delta.in_proj_z.weight,
                    delta.in_proj_a.weight,
                    delta.in_proj_b.weight,
                ),
                dim=0,
            ).contiguous()
            if config.delta_projection_padding:
                pad_rows = (-packed.shape[0]) % config.delta_projection_padding
                if pad_rows:
                    packed = F.pad(packed, (0, 0, 0, pad_rows))
            _register_buffer(delta, "_fused_in_proj_weight", packed)
            stats["delta"] += 1

        if config.swiglu:
            mlp = layer.mlp
            gate_up = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0).contiguous()
            _register_buffer(mlp, "_fused_gate_up_weight", gate_up)
            stats["mlp"] += 1

        if hasattr(layer, "self_attn") and config.attention:
            attention = layer.self_attn
            qgkv = torch.cat(
                (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight), dim=0
            ).contiguous()
            _register_buffer(attention, "_fused_qgkv_weight", qgkv)
            biases = (attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias)
            if any(bias is not None for bias in biases):
                if not all(bias is not None for bias in biases):
                    raise ValueError("Qwen3.5 attention projection biases must be all present or all absent")
                _register_buffer(attention, "_fused_qgkv_bias", torch.cat(biases).contiguous())
            stats["attention"] += 1
    return stats


def _patch_norm_module(module: torch.nn.Module) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(self, x: torch.Tensor):
        if _is_fast_tensor(x) and x.is_contiguous():
            return rms_norm(x, self.weight, self.eps)
        return self._fused_original_forward(x)

    module.forward = types.MethodType(forward, module)


def _patch_gated_norm_module(module: torch.nn.Module) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None):
        if gate is not None and _is_fast_tensor(hidden_states):
            return gated_rms_norm(hidden_states.contiguous(), gate.contiguous(), self.weight, self.variance_epsilon)
        return self._fused_original_forward(hidden_states, gate)

    module.forward = types.MethodType(forward, module)


def _patch_mlp(module: torch.nn.Module) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(self, x: torch.Tensor):
        if _is_fast_tensor(x):
            packed = F.linear(x, self._fused_gate_up_weight)
            activated = silu_and_mul(packed)
            return self.down_proj(activated)
        return self._fused_original_forward(x)

    module.forward = types.MethodType(forward, module)


def _delta_cache_tensors(
    module: torch.nn.Module,
    cache_params: Any,
    batch: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    conv_shape = (batch, module.conv_dim, module.conv_kernel_size)
    recurrent_shape = (batch, module.num_v_heads, module.head_k_dim, module.head_v_dim)
    if cache_params is None:
        return (
            torch.zeros(conv_shape, device=device, dtype=dtype),
            torch.zeros(recurrent_shape, device=device, dtype=torch.float32),
        )

    cache_layer = cache_params.layers[module.layer_idx]
    if cache_layer.record_past:
        raise RuntimeError("fused DeltaNet does not support rollback cache recording")
    if not cache_layer.is_conv_states_initialized[0]:
        example = torch.empty((batch, module.conv_dim, seq_len), device=device, dtype=dtype)
        cache_layer.lazy_initialization(conv_states=example, conv_kernel_size=module.conv_kernel_size)
    if not cache_layer.is_recurrent_states_initialized[0]:
        example = torch.empty(recurrent_shape, device=device, dtype=torch.float32)
        cache_layer.lazy_initialization(recurrent_states=example)
    cache_layer.has_previous_state[0] = True
    return cache_layer.conv_states[0], cache_layer.recurrent_states[0]


def _patch_delta(module: torch.nn.Module, config: FusionConfig) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if not _is_fast_tensor(hidden_states):
            return self._fused_original_forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        if attention_mask is not None:
            hidden_states = qwen_impl.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, seq_len, _ = hidden_states.shape
        packed = F.linear(hidden_states, self._fused_in_proj_weight)
        conv_state, recurrent_state = _delta_cache_tensors(
            self,
            cache_params,
            batch,
            seq_len,
            hidden_states.device,
            hidden_states.dtype,
        )
        qkv = causal_conv1d_fused(packed, self.conv1d.weight.squeeze(1).contiguous(), conv_state)
        core = delta_recurrent_fused(
            qkv,
            packed,
            self.A_log,
            self.dt_bias,
            recurrent_state,
            block_v=config.delta_block_v,
        )
        z_start = self.conv_dim
        z = packed[..., z_start : z_start + self.value_dim].reshape(-1, self.head_v_dim).contiguous()
        core = gated_rms_norm(
            core.reshape(-1, self.head_v_dim),
            z,
            self.norm.weight,
            self.layer_norm_epsilon,
        )
        return self.out_proj(core.reshape(batch, seq_len, self.value_dim))

    module.forward = types.MethodType(forward, module)


def _patch_attention(module: torch.nn.Module) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        if not _is_fast_tensor(hidden_states):
            return self._fused_original_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        packed = F.linear(
            hidden_states,
            self._fused_qgkv_weight,
            getattr(self, "_fused_qgkv_bias", None),
        )
        cos, sin = position_embeddings
        cache_layer = None
        if past_key_values is not None:
            candidate = past_key_values.layers[self.layer_idx]
            if hasattr(candidate, "max_cache_len"):
                cache_layer = candidate
                if not cache_layer.is_initialized:
                    batch, seq_len = hidden_states.shape[:2]
                    cache_shape = (
                        batch,
                        self.config.num_key_value_heads,
                        seq_len,
                        self.head_dim,
                    )
                    initial_key = torch.empty(
                        cache_shape,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    cache_layer.lazy_initialization(initial_key, torch.empty_like(initial_key))
        query_states, key_states, value_states = qgkv_norm_rope(
            packed,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            q_heads=self.config.num_attention_heads,
            kv_heads=self.config.num_key_value_heads,
            dim=self.head_dim,
            rotary_dim=cos.shape[-1],
            cache_key=None if cache_layer is None else cache_layer.keys,
            cache_value=None if cache_layer is None else cache_layer.values,
            cache_start=None if cache_layer is None else cache_layer.cumulative_length,
        )
        if cache_layer is not None:
            cache_layer.cumulative_length.add_(hidden_states.shape[1])
        elif past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        attention_interface = qwen_impl.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen_impl.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        input_shape = hidden_states.shape[:-1]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attention_gate_mul(
            attn_output,
            packed,
            self.head_dim,
        )
        return self.o_proj(attn_output), attn_weights

    module.forward = types.MethodType(forward, module)


def _patch_text_model(text_model: torch.nn.Module, config: FusionConfig) -> None:
    if hasattr(text_model, "_fused_original_forward"):
        return
    text_model._fused_original_forward = text_model.forward

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if not _is_fast_tensor(inputs_embeds):
            return self._fused_original_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rotary_position_ids = position_ids

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": qwen_impl.create_causal_mask(**mask_kwargs),
                "linear_attention": qwen_impl.create_recurrent_attention_mask(**mask_kwargs),
            }

        position_embeddings = self.rotary_emb(inputs_embeds, rotary_position_ids)
        residual = inputs_embeds.contiguous()
        normalized = rms_norm(
            residual,
            self.layers[0].input_layernorm.weight,
            self.layers[0].input_layernorm.eps,
        )
        for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if layer.block_type == "linear_attention":
                mixer_update = layer.linear_attn(
                    hidden_states=normalized,
                    cache_params=past_key_values,
                    attention_mask=causal_mask_mapping["linear_attention"],
                    **kwargs,
                )
            else:
                mixer_update, _ = layer.self_attn(
                    hidden_states=normalized,
                    attention_mask=causal_mask_mapping["full_attention"],
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            residual, mlp_input = residual_add_rms_norm(
                residual,
                mixer_update.contiguous(),
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            mlp_update = layer.mlp(mlp_input)
            if index + 1 < self.config.num_hidden_layers:
                next_norm = self.layers[index + 1].input_layernorm
                residual, normalized = residual_add_rms_norm(
                    residual,
                    mlp_update.contiguous(),
                    next_norm.weight,
                    next_norm.eps,
                )
            else:
                _, normalized = residual_add_rms_norm(
                    residual,
                    mlp_update.contiguous(),
                    self.norm.weight,
                    self.norm.eps,
                )

        return qwen_impl.Qwen3_5ModelOutputWithPast(
            last_hidden_state=normalized,
            past_key_values=past_key_values,
        )

    text_model.forward = types.MethodType(forward, text_model)


def _patch_vision_model(vision_model: torch.nn.Module) -> None:
    if hasattr(vision_model, "_fused_original_forward"):
        return
    vision_model._fused_original_forward = vision_model.forward

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        if not _is_fast_tensor(hidden_states):
            return self._fused_original_forward(hidden_states, grid_thw, **kwargs)
        interp_indices, interp_weights = qwen_impl.get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = qwen_impl.get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens, max_seqlen = qwen_impl.get_vision_attention_seqlens(
            grid_thw, self.config, kwargs=kwargs
        )
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = position_embed_add(
            hidden_states.contiguous(),
            self.pos_embed.weight,
            interp_indices.contiguous(),
            interp_weights.contiguous(),
        )
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len, _ = hidden_states.shape
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        residual = hidden_states.contiguous()
        first_norm = self.blocks[0].norm1
        normalized = layer_norm(residual, first_norm.weight, first_norm.bias, first_norm.eps)
        for index, block in enumerate(self.blocks):
            attention_update = block.attn(
                normalized,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            residual, mlp_input = residual_add_layer_norm(
                residual,
                attention_update.contiguous(),
                block.norm2.weight,
                block.norm2.bias,
                block.norm2.eps,
            )
            mlp_update = block.mlp(mlp_input)
            if index + 1 < len(self.blocks):
                next_norm = self.blocks[index + 1].norm1
                residual, normalized = residual_add_layer_norm(
                    residual,
                    mlp_update.contiguous(),
                    next_norm.weight,
                    next_norm.bias,
                    next_norm.eps,
                )
            else:
                hidden_states = residual + mlp_update

        merged_hidden_states = self.merger(hidden_states)
        return qwen_impl.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    vision_model.forward = types.MethodType(forward, vision_model)


def _patch_vision_attention(module: torch.nn.Module) -> None:
    if hasattr(module, "_fused_original_forward"):
        return
    module._fused_original_forward = module.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ):
        if not _is_fast_tensor(hidden_states) or position_embeddings is None:
            return self._fused_original_forward(
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
                **kwargs,
            )
        packed = self.qkv(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states, value_states = vision_qkv_rope(
            packed.contiguous(),
            cos,
            sin,
            heads=self.num_heads,
            dim=self.dim // self.num_heads,
        )
        attention_interface = qwen_impl.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen_impl.eager_attention_forward
        )
        if qwen_impl.is_flash_attention_requested(self.config):
            max_seqlen = qwen_impl.get_max_seqlen(
                cu_seqlens,
                self.config,
                kwargs={"max_seqlen": max_seqlen},
            )
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            if cu_seqlens.numel() == 2:
                attn_output, _ = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                outputs = [
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(outputs, dim=1)
        seq_length = hidden_states.shape[0]
        return self.proj(attn_output.reshape(seq_length, -1).contiguous())

    module.forward = types.MethodType(forward, module)


def precompute_vision_kwargs(model: torch.nn.Module, inputs: Any) -> Any:
    """Attach Transformers' precomputed Vision tensors to a model input mapping.

    Run this on the processor's CPU output before ``BatchFeature.to(device)``.
    The modality-prefixed names are moved with the remaining inputs, translated
    by ``get_image_features`` and consumed by the Vision encoder.
    """

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return inputs
    visual = model.model.visual
    helper_kwargs: dict[str, Any] = {}
    interp_indices, interp_weights = qwen_impl.get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
        kwargs=helper_kwargs,
    )
    position_ids = qwen_impl.get_vision_position_ids(
        grid_thw,
        visual.spatial_merge_size,
        kwargs=helper_kwargs,
    )
    cu_seqlens, max_seqlen = qwen_impl.get_vision_attention_seqlens(
        grid_thw,
        visual.config,
        kwargs=helper_kwargs,
    )
    inputs["image_interp_indices"] = interp_indices
    inputs["image_interp_weights"] = interp_weights
    inputs["image_position_ids"] = position_ids
    inputs["image_cu_seqlens"] = cu_seqlens
    if max_seqlen is not None:
        inputs["image_max_seqlen"] = max_seqlen
    return inputs


def apply_fusions(model: torch.nn.Module, config: FusionConfig | None = None) -> dict[str, int]:
    """Pack weights and install all enabled inference-only fast paths."""

    config = config or FusionConfig()
    if model.training:
        raise ValueError("apply_fusions requires model.eval()")
    stats = _pack_linear_weights(model, config)
    text_model = model.model.language_model

    for layer in text_model.layers:
        if config.rms_norm:
            _patch_norm_module(layer.input_layernorm)
            _patch_norm_module(layer.post_attention_layernorm)
        if config.swiglu:
            _patch_mlp(layer.mlp)
        if hasattr(layer, "linear_attn") and config.delta:
            _patch_gated_norm_module(layer.linear_attn.norm)
            _patch_delta(layer.linear_attn, config)
        if hasattr(layer, "self_attn") and config.attention:
            _patch_norm_module(layer.self_attn.q_norm)
            _patch_norm_module(layer.self_attn.k_norm)
            _patch_attention(layer.self_attn)
    if config.rms_norm:
        _patch_norm_module(text_model.norm)
    if config.residual_norm:
        _patch_text_model(text_model, config)
    if config.vision:
        for block in model.model.visual.blocks:
            _patch_vision_attention(block.attn)
        _patch_vision_model(model.model.visual)
        stats["vision"] = len(model.model.visual.blocks)
    model._qwen35_fusion_config = config
    return stats
