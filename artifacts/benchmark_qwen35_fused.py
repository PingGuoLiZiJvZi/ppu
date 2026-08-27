#!/usr/bin/env python3
"""Compare baseline and fused Qwen3.5 language-model phases on PPU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import StaticCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv
from qwen35_fused.integration import FusionConfig, apply_fusions
from qwen35_fused.graph import GreedyDecodeGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("baseline", "norm_mlp", "delta", "attention", "all"),
        default="all",
    )
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--static-cache", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


def variant_config(name: str) -> FusionConfig | None:
    if name == "baseline":
        return None
    if name == "norm_mlp":
        return FusionConfig(delta=False, rms_norm=True, residual_norm=True, swiglu=True, attention=False, vision=False)
    if name == "delta":
        return FusionConfig(delta=True, rms_norm=False, residual_norm=False, swiglu=False, attention=False, vision=False)
    if name == "attention":
        return FusionConfig(delta=False, rms_norm=False, residual_norm=False, swiglu=False, attention=True, vision=False)
    return FusionConfig()


def event_time(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260819)
    sample = load_mmbench_tsv(ROOT / "datasets/mmbench/mmbench_dev_en.tsv", limit=1)[0]
    image = decode_image(sample.image_b64)
    prompt = build_prompt(sample)

    load_start = time.perf_counter()
    # This benchmark applies exactly the requested variant below.
    model_path = str(ROOT / "Qwen3.5-2B")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    memory_before_fusion = int(torch.cuda.memory_allocated())
    fusion_config = variant_config(args.variant)
    fusion_stats = apply_fusions(model, fusion_config) if fusion_config is not None else {}
    memory_after_fusion = int(torch.cuda.memory_allocated())
    load_seconds = time.perf_counter() - load_start

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        vision_output = model.model.get_image_features(
            inputs.pixel_values, inputs.image_grid_thw, return_dict=True
        )
        image_embeds = torch.cat(vision_output.pooler_output, dim=0).to(model.device, model.dtype)
        input_embeds = model.get_input_embeddings()(inputs.input_ids)
        image_mask, _ = model.model.get_placeholder_mask(
            inputs.input_ids,
            inputs_embeds=input_embeds,
            image_features=image_embeds,
        )
        input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
        model.model.rope_deltas = None
        position_ids = model.model.compute_3d_position_ids(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=input_embeds,
            attention_mask=inputs.attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs.mm_token_type_ids,
        )

    def rollout() -> dict[str, Any]:
        state: dict[str, Any] = {}

        def prefill():
            cache = (
                StaticCache(
                    config=model.config,
                    max_cache_len=int(inputs.input_ids.shape[-1]) + args.decode_steps + 4,
                )
                if args.static_cache
                else None
            )
            outputs = model.model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=inputs.attention_mask,
                past_key_values=cache,
                inputs_embeds=input_embeds,
                use_cache=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
            return outputs, logits

        (prefill_outputs, prefill_logits), prefill_ms = event_time(prefill)
        state["cache"] = prefill_outputs.past_key_values
        state["token"] = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        state["attention_mask"] = inputs.attention_mask
        tokens = [int(state["token"].item())]
        decode_times = []

        if args.cuda_graph:
            if not args.static_cache:
                raise ValueError("--cuda-graph requires --static-cache")
            first_position = (
                state["attention_mask"].long().sum(dim=-1, keepdim=True).unsqueeze(0)
                + model.model.rope_deltas.unsqueeze(0)
            )
            runner = GreedyDecodeGraph(
                model,
                state["cache"],
                state["token"],
                first_position,
            )
            for _ in range(args.decode_steps):
                token, decode_ms = event_time(runner.replay)
                decode_times.append(decode_ms)
                tokens.append(int(token.item()))
            return {"prefill_ms": prefill_ms, "decode_ms": decode_times, "tokens": tokens}

        for _ in range(args.decode_steps):
            def decode():
                state["attention_mask"] = torch.cat(
                    [state["attention_mask"], torch.ones_like(state["token"])], dim=-1
                )
                text_position = state["attention_mask"].long().sum(dim=-1, keepdim=True) - 1
                decode_position_ids = text_position.unsqueeze(0) + model.model.rope_deltas.unsqueeze(0)
                outputs = model(
                    input_ids=state["token"],
                    attention_mask=state["attention_mask"],
                    position_ids=decode_position_ids,
                    past_key_values=state["cache"],
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                state["cache"] = outputs.past_key_values
                state["token"] = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                return outputs

            _, decode_ms = event_time(decode)
            decode_times.append(decode_ms)
            tokens.append(int(state["token"].item()))
        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_times,
            "tokens": tokens,
        }

    with torch.inference_mode():
        warmup = rollout()
        measured = [rollout() for _ in range(args.repetitions)]

    payload = {
        "variant": args.variant,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "load_seconds": load_seconds,
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "fusion_stats": fusion_stats,
        "memory": {
            "before_fusion_allocated_bytes": memory_before_fusion,
            "after_fusion_allocated_bytes": memory_after_fusion,
            "fusion_allocated_delta_bytes": memory_after_fusion - memory_before_fusion,
        },
        "static_cache": args.static_cache,
        "cuda_graph": args.cuda_graph,
        "warmup": warmup,
        "tokens": measured[-1]["tokens"],
        "decoded": processor.tokenizer.decode(measured[-1]["tokens"]),
        "prefill_ms": {
            "samples": [item["prefill_ms"] for item in measured],
            "median": statistics.median(item["prefill_ms"] for item in measured),
        },
        "decode_ms": {
            "samples": [value for item in measured for value in item["decode_ms"]],
            "median": statistics.median(value for item in measured for value in item["decode_ms"]),
            "mean": statistics.fmean(value for item in measured for value in item["decode_ms"]),
        },
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
