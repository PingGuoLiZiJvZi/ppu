#!/usr/bin/env python3
"""Probe 3: capture the REAL full-attention decode call from the fused path and time alternatives."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv
from evaluation_wrapper import VLMModel
from qwen35_fused.integration import precompute_vision_kwargs

sys.path.insert(0, str(ROOT / "artifacts"))
from probe_decode_attention import triton_gqa_decode  # noqa: E402


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


def main() -> None:
    torch.manual_seed(20260625)
    sample = load_mmbench_tsv(ROOT / "datasets/mmbench/mmbench_dev_en.tsv", limit=1)[0]
    wrapper = VLMModel(str(ROOT / "Qwen3.5-2B"), backend="transformers", device="auto")
    model = wrapper._model
    processor = wrapper._processor

    messages = [{"role": "user", "content": [
        {"type": "image", "image": decode_image(sample.image_b64)},
        {"type": "text", "text": build_prompt(sample)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    precompute_vision_kwargs(model, inputs)
    inputs = inputs.to(model.device)
    input_len = int(inputs.input_ids.shape[-1])

    from transformers.cache_utils import StaticCache
    cache = StaticCache(config=model.config, max_cache_len=512)

    recorded = []
    real_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(q, k, v, attn_mask=None, **kw):
        recorded.append({
            "q": q.detach().clone(), "k": k.detach().clone(), "v": v.detach().clone(),
            "attn_mask": None if attn_mask is None else attn_mask.detach().clone(),
            "kw": {key: value for key, value in kw.items() if isinstance(value, (int, float, bool, str))},
            "q_stride": list(q.stride()), "k_stride": list(k.stride()),
        })
        return real_sdpa(q, k, v, attn_mask=attn_mask, **kw)

    with torch.inference_mode():
        prefill = model.model(**inputs, past_key_values=cache, use_cache=True, return_dict=True)
        first_position = (
            inputs.attention_mask.long().sum(dim=-1, keepdim=True).unsqueeze(0)
            + model.model.rope_deltas.unsqueeze(0)
        )
        token = prefill.last_hidden_state[:, -1:, :].contiguous().float()
        token = torch.randint(0, 1000, (1, 1), device=model.device)

        F_scaled = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = recording_sdpa
        try:
            _ = model.model(
                input_ids=token, attention_mask=None, position_ids=first_position,
                past_key_values=cache, use_cache=True, return_dict=True,
            )
        finally:
            torch.nn.functional.scaled_dot_product_attention = F_scaled

    print(f"recorded {len(recorded)} sdpa calls in one decode step")
    if not recorded:
        print("no sdpa calls recorded - attention uses another backend")
        return

    rows = {}
    call = recorded[0]
    q, k, v, mask = call["q"], call["k"], call["v"], call["attn_mask"]
    rows["shapes"] = {
        "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
        "mask": None if mask is None else list(mask.shape),
        "q_stride": call["q_stride"], "k_stride": call["k_stride"],
        "kw": call["kw"],
    }
    scale = call["kw"].get("scale", 1.0)

    def as_recorded():
        return real_sdpa(q, k, v, attn_mask=mask, **{kk: vv for kk, vv in call["kw"].items()})

    rows["as_recorded_ms"] = round(time_ms(as_recorded), 5)
    if mask is not None:
        rows["no_mask_ms"] = round(time_ms(lambda: real_sdpa(q, k, v, **call["kw"])), 5)

    # Sliced-KV variant: attend only to the first `pos` entries, no mask.
    pos = int(call["kw"].get("cache_position", [None])[-1]) if "cache_position" in call["kw"] else None
    kv_used = k.shape[-2]
    rows["kv_len_full"] = kv_used
    for cut in (k.shape[-2],):
        ks = k[:, :, :cut].contiguous()
        vs = v[:, :, :cut].contiguous()
        rows[f"contig_kv{cut}_ms"] = round(
            time_ms(lambda: real_sdpa(q, ks, vs, **call["kw"])), 5
        )

    # Triton split-KV kernel on the same data (q reshaped to [H, D]).
    q2 = q.reshape(-1, q.shape[-1]).contiguous()
    k2 = k.permute(0, 1, 3, 2).reshape(k.shape[1], k.shape[-1], -1) if False else k[0].transpose(1, 2).contiguous()  # [kvH, D, L]? keep simple below
    # k real layout: [B, kvH, L, D] -> triton kernel wants [kvH, L, D]
    k_t = k[0].contiguous()
    v_t = v[0].contiguous()
    for splits in (4, 8):
        out = triton_gqa_decode(q2, k_t, v_t, scale, splits=splits)
        rows[f"triton_s{splits}_ms"] = round(
            time_ms(lambda: triton_gqa_decode(q2, k_t, v_t, scale, splits=splits)), 5
        )
        ref = as_recorded().reshape(-1, q.shape[-1]).float()
        rows[f"triton_s{splits}_maxdiff"] = round((out - ref).abs().max().item(), 4)

    out_path = ROOT / "artifacts" / "probe_real_decode_attn.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
