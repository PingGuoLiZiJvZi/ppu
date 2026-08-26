#!/usr/bin/env python3
"""Profile actual BF16 kernels used by the local Qwen3.5-2B inference path."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from transformers.cache_utils import StaticCache
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv
from evaluation_wrapper import VLMModel
from qwen35_fused.graph import GreedyDecodeGraph
from qwen35_fused.integration import precompute_vision_kwargs


FUSIONS_ENABLED = os.environ.get("QWEN35_FUSIONS", "1") != "0"
OUT = ROOT / "artifacts" / (
    "qwen3_5_fused_kernel_profile.json" if FUSIONS_ENABLED else "qwen3_5_bf16_kernel_profile.json"
)
DECODE_STEPS = 8


def kernel_category(name: str) -> str:
    lower = name.lower()
    if "gemm" in lower or "gemv" in lower or "blas" in lower:
        return "gemm_gemv"
    if any(token in lower for token in ("flash", "fmha", "attention", "sdpa")):
        return "attention"
    if "conv" in lower:
        return "convolution"
    if any(token in lower for token in ("softmax", "reduce", "norm", "layer_norm", "rms")):
        return "reduction_norm_softmax"
    if any(token in lower for token in ("elementwise", "pointwise", "vectorized", "foreach")):
        return "elementwise"
    if any(token in lower for token in ("copy", "scatter", "index", "cat", "slice", "pad")):
        return "layout_copy_index"
    return "other"


def dtype_tag(name: str) -> str:
    lower = name.lower()
    tags = []
    for needles, label in (
        (("bf16", "bfloat16"), "BF16"),
        (("fp32", "float"), "FP32"),
        (("fp16", "half"), "FP16"),
        (("int8",), "INT8"),
    ):
        if any(needle in lower for needle in needles):
            tags.append(label)
    return "+".join(tags) if tags else "unspecified"


def short_kernel_name(name: str) -> str:
    # Keep the dispatch-defining fields while making PPU GEMM symbols readable.
    if name.startswith("gemm_"):
        fields = [
            re.search(r"dtype[^_]+", name),
            re.search(r"tile[^_]+", name),
            re.search(r"layout[^_]+", name),
            re.search(r"fusion[^_]+", name),
        ]
        selected = [match.group(0) for match in fields if match]
        return "gemm_" + "_".join(selected)
    return name


def aggregate_profile(prof, repetitions: int, event_ms: float) -> dict[str, Any]:
    device_events = [event for event in prof.events() if str(event.device_type) == "DeviceType.CUDA"]
    kernels: dict[str, dict[str, float | int | str]] = {}
    for event in device_events:
        item = kernels.setdefault(
            event.name,
            {
                "name": event.name,
                "short_name": short_kernel_name(event.name),
                "category": kernel_category(event.name),
                "dtype_tag": dtype_tag(event.name),
                "count": 0,
                "device_time_us": 0.0,
            },
        )
        item["count"] = int(item["count"]) + 1
        item["device_time_us"] = float(item["device_time_us"]) + float(event.self_device_time_total)

    ordered_kernels = sorted(kernels.values(), key=lambda item: float(item["device_time_us"]), reverse=True)
    total_kernel_us = sum(float(item["device_time_us"]) for item in ordered_kernels)
    category_totals: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "device_time_us": 0.0})
    dtype_totals: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "device_time_us": 0.0})
    for item in ordered_kernels:
        for bucket, key in ((category_totals, str(item["category"])), (dtype_totals, str(item["dtype_tag"]))):
            bucket[key]["count"] = int(bucket[key]["count"]) + int(item["count"])
            bucket[key]["device_time_us"] = float(bucket[key]["device_time_us"]) + float(item["device_time_us"])

    launch_apis = Counter()
    for event in prof.events():
        if str(event.device_type) != "DeviceType.CPU":
            continue
        if "LaunchKernel" in event.name or "GraphLaunch" in event.name or event.name in {
            "cuLaunchKernel",
            "cuLaunchKernelEx",
            "cudaGraphLaunch",
        }:
            launch_apis[event.name] += 1

    operator_rows = []
    for event in prof.key_averages():
        if not event.key.startswith("aten::") or event.device_time_total <= 0:
            continue
        operator_rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "device_time_us": float(event.device_time_total),
                "self_device_time_us": float(event.self_device_time_total),
            }
        )
    operator_rows.sort(key=lambda item: item["device_time_us"], reverse=True)

    return {
        "repetitions": repetitions,
        "cuda_event_total_ms": event_ms,
        "cuda_event_ms_per_repetition": event_ms / repetitions,
        "device_kernel_launches": len(device_events),
        "device_kernel_launches_per_repetition": len(device_events) / repetitions,
        "unique_kernel_symbols": len(ordered_kernels),
        "summed_kernel_device_time_us": total_kernel_us,
        "launch_api_calls": dict(launch_apis),
        "categories": dict(category_totals),
        "dtype_tags": dict(dtype_totals),
        "top_kernels": ordered_kernels[:80],
        "top_aten_operators": operator_rows[:60],
    }


def run_profile(label: str, fn: Callable[[], Any], repetitions: int = 1) -> tuple[Any, dict[str, Any]]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        start.record()
        result = None
        for _ in range(repetitions):
            result = fn()
        end.record()
        torch.cuda.synchronize()
    summary = aggregate_profile(prof, repetitions, start.elapsed_time(end))
    summary["label"] = label
    return result, summary


def time_cuda(fn: Callable[[], Any], repetitions: int) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for _ in range(repetitions):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / repetitions


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU device is unavailable")

    torch.manual_seed(20260625)
    sample = load_mmbench_tsv(ROOT / "datasets/mmbench/mmbench_dev_en.tsv", limit=1)[0]
    image = decode_image(sample.image_b64)
    prompt = build_prompt(sample)

    load_started = time.perf_counter()
    wrapper = VLMModel(str(ROOT / "Qwen3.5-2B"), backend="transformers", device="auto")
    model = wrapper._model
    processor = wrapper._processor
    load_seconds = time.perf_counter() - load_started

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    precompute_vision_kwargs(model, inputs)
    inputs = inputs.to(model.device)

    # Vision warmup and actual profile.
    with torch.inference_mode():
        vision_kwargs = {
            key: value
            for key, value in inputs.items()
            if key.startswith("image_") and key != "image_grid_thw"
        }
        _ = model.model.get_image_features(
            inputs.pixel_values,
            inputs.image_grid_thw,
            return_dict=True,
            **vision_kwargs,
        )
        torch.cuda.synchronize()
        _, vision_unprofiled_ms = time_cuda(
            lambda: model.model.get_image_features(
                inputs.pixel_values,
                inputs.image_grid_thw,
                return_dict=True,
                **vision_kwargs,
            ),
            repetitions=5,
        )
        vision_output, vision_profile = run_profile(
            "vision_encoder",
            lambda: model.model.get_image_features(
                inputs.pixel_values,
                inputs.image_grid_thw,
                return_dict=True,
                **vision_kwargs,
            ),
        )
        vision_profile["unprofiled_cuda_event_ms_per_repetition"] = vision_unprofiled_ms

    image_embeds = torch.cat(vision_output.pooler_output, dim=0).to(model.device, model.dtype)

    def fusion_prefill():
        model.model.rope_deltas = None
        input_embeds = model.get_input_embeddings()(inputs.input_ids)
        image_mask, _ = model.model.get_placeholder_mask(
            inputs.input_ids,
            inputs_embeds=input_embeds,
            image_features=image_embeds,
        )
        input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = model.model.compute_3d_position_ids(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=input_embeds,
            attention_mask=inputs.attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs.mm_token_type_ids,
        )
        outputs = model.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=None,
            inputs_embeds=input_embeds,
            use_cache=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
        return outputs, logits, input_embeds

    with torch.inference_mode():
        _ = fusion_prefill()
        torch.cuda.synchronize()
        _, prefill_unprofiled_ms = time_cuda(fusion_prefill, repetitions=3)
        prefill_result, prefill_profile = run_profile("fusion_language_prefill", fusion_prefill)
        prefill_profile["unprofiled_cuda_event_ms_per_repetition"] = prefill_unprofiled_ms
        prefill_outputs, prefill_logits, fused_embeds = prefill_result

        state: dict[str, Any] = {
            "cache": prefill_outputs.past_key_values,
            "token": prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True),
            "attention_mask": inputs.attention_mask,
            "last_hidden": prefill_outputs.last_hidden_state,
            "last_logits": prefill_logits,
        }

        def decode_step():
            state["attention_mask"] = torch.cat(
                [state["attention_mask"], torch.ones_like(state["token"], device=model.device)], dim=-1
            )
            # Match Qwen3_5ForConditionalGeneration._prepare_position_ids_for_generation:
            # during cached decode, use the sliced 1-D text position plus the cached multimodal rope delta.
            text_position = state["attention_mask"].long().sum(dim=-1, keepdim=True) - 1
            position_ids = text_position.unsqueeze(0) + model.model.rope_deltas.unsqueeze(0)
            outputs = model(
                input_ids=state["token"],
                attention_mask=state["attention_mask"],
                position_ids=position_ids,
                past_key_values=state["cache"],
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            state["cache"] = outputs.past_key_values
            state["token"] = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            state["last_hidden"] = outputs.hidden_states
            state["last_logits"] = outputs.logits
            return outputs

        # Two decode warmups leave all lazy kernels/cache buffers initialized.
        decode_step()
        decode_step()
        torch.cuda.synchronize()
        _, decode_unprofiled_ms = time_cuda(decode_step, repetitions=20)
        decode_output, decode_profile = run_profile("steady_decode", decode_step, repetitions=DECODE_STEPS)
        decode_profile["unprofiled_cuda_event_ms_per_repetition"] = decode_unprofiled_ms

        static_cache = StaticCache(config=model.config, max_cache_len=512)
        static_position_ids = model.model.compute_3d_position_ids(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=fused_embeds,
            attention_mask=inputs.attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs.mm_token_type_ids,
        )
        static_outputs = model.model.language_model(
            input_ids=None,
            position_ids=static_position_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=static_cache,
            inputs_embeds=fused_embeds,
            use_cache=True,
        )
        static_token = model.lm_head(static_outputs.last_hidden_state[:, -1:, :]).argmax(
            dim=-1
        )
        first_decode_position = (
            inputs.attention_mask.long().sum(dim=-1, keepdim=True).unsqueeze(0)
            + model.model.rope_deltas.unsqueeze(0)
        )
        graph_runner = GreedyDecodeGraph(
            model,
            static_cache,
            static_token,
            first_decode_position,
            fused_lm_head=True,
        )
        _, graph_unprofiled_ms = time_cuda(graph_runner.replay, repetitions=20)
        _, graph_profile = run_profile(
            "steady_decode_graph", graph_runner.replay, repetitions=DECODE_STEPS
        )
        graph_profile["unprofiled_cuda_event_ms_per_repetition"] = graph_unprofiled_ms

    linear_layer = model.model.language_model.layers[0].linear_attn
    full_layer = model.model.language_model.layers[3].self_attn
    cache_linear = state["cache"].layers[0]
    cache_full = state["cache"].layers[3]
    runtime = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "model_class": model.__class__.__name__,
        "model_dtype": str(model.dtype),
        "text_attention_backend": model.config.text_config._attn_implementation,
        "vision_attention_backend": model.config.vision_config._attn_implementation,
        "torch_compile_enabled": False,
        "fla_installed": False,
        "causal_conv1d_installed": False,
        "load_seconds": load_seconds,
        "sample_id": sample.sample_id,
        "input_tokens": int(inputs.input_ids.shape[1]),
        "image_tokens": int((inputs.input_ids == model.config.image_token_id).sum().item()),
        "pixel_values_shape": list(inputs.pixel_values.shape),
        "weights": {
            "vision_patch_embed": str(model.model.visual.patch_embed.proj.weight.dtype),
            "token_embedding": str(model.get_input_embeddings().weight.dtype),
            "linear_qkv": str(getattr(linear_layer.in_proj_qkv.weight, "dtype", None)),
            "linear_A_log": str(linear_layer.A_log.dtype),
            "linear_dt_bias": str(linear_layer.dt_bias.dtype),
            "full_q_proj": str(getattr(full_layer.q_proj.weight, "dtype", None)),
            "lm_head": str(model.lm_head.weight.dtype),
        },
        "activations": {
            "vision_pooler": str(image_embeds.dtype),
            "fused_embeddings": str(fused_embeds.dtype),
            "language_last_hidden": str(prefill_outputs.last_hidden_state.dtype),
            "prefill_logits": str(prefill_logits.dtype),
            "decode_logits": str(decode_output.logits.dtype),
        },
        "cache": {
            "linear_conv": str(cache_linear.conv_states[0].dtype),
            "linear_conv_shape": list(cache_linear.conv_states[0].shape),
            "linear_recurrent": str(cache_linear.recurrent_states[0].dtype),
            "linear_recurrent_shape": list(cache_linear.recurrent_states[0].shape),
            "full_key": str(cache_full.keys.dtype),
            "full_key_shape": list(cache_full.keys.shape),
            "full_value": str(cache_full.values.dtype),
            "full_value_shape": list(cache_full.values.shape),
        },
    }

    payload = {
        "runtime": runtime,
        "profiles": {
            "vision_encoder": vision_profile,
            "fusion_language_prefill": prefill_profile,
            "steady_decode": decode_profile,
            "steady_decode_graph": graph_profile,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)
    print(json.dumps(runtime, indent=2, ensure_ascii=False))
    for key, item in payload["profiles"].items():
        print(
            key,
            "unprofiled_ms=", round(item["unprofiled_cuda_event_ms_per_repetition"], 3),
            "profiled_ms=", round(item["cuda_event_ms_per_repetition"], 3),
            "launches/rep=", round(item["device_kernel_launches_per_repetition"], 1),
            "unique=", item["unique_kernel_symbols"],
        )


if __name__ == "__main__":
    main()
