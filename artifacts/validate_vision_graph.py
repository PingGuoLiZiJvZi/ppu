#!/usr/bin/env python3
"""Validate Stage 5 vision CUDA graph: exactness (eager vs replay) and latency."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv  # noqa: E402
from evaluation_wrapper import GenerationConfig, VLMModel  # noqa: E402
from qwen35_fused.integration import precompute_vision_kwargs  # noqa: E402


def timed_generate(wrapper, image, prompt):
    return wrapper.generate_with_metrics(
        image=image, prompt=prompt, choices={},
        generation_config=GenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
        sample_id="probe",
    )


def main() -> None:
    torch.manual_seed(0)
    samples = load_mmbench_tsv(ROOT / "datasets/mmbench/mmbench_dev_en.tsv", limit=6)
    wrapper = VLMModel(str(ROOT / "Qwen3.5-2B"), backend="transformers", device="auto")
    model = wrapper._model
    processor = wrapper._processor

    ok = True
    for index, sample in enumerate(samples):
        image = decode_image(sample.image_b64)
        prompt = build_prompt(sample)
        first = timed_generate(wrapper, image, prompt)   # eager + capture
        second = timed_generate(wrapper, image, prompt)  # graph replay
        third = timed_generate(wrapper, image, prompt)
        same = first.text == second.text == third.text
        ok &= same
        print(
            f"sample {index}: identical={same} tokens={first.token_count}/{second.token_count} "
            f"ttft1={first.ttft_seconds*1000:.1f}ms ttft3={third.ttft_seconds*1000:.1f}ms "
            f"tp1={round((first.token_count-1)/max(first.elapsed_seconds-first.ttft_seconds,1e-6))} "
            f"tp3={round((third.token_count-1)/max(third.elapsed_seconds-third.ttft_seconds,1e-6))}"
        )

    graphs = model.model.visual._vision_graphs
    print(f"captured vision graphs: {len(graphs)} (patch counts: {sorted(graphs)})")

    # Vision-only latency: eager (uncaptured shape) vs replay.
    sample = samples[0]
    image = decode_image(sample.image_b64)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "x"}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    precompute_vision_kwargs(model, inputs)
    inputs = inputs.to(model.device)
    vision_kwargs = {k: v for k, v in inputs.items() if k.startswith("image_") and k != "image_grid_thw"}

    def run_vision():
        return model.model.get_image_features(
            inputs.pixel_values, inputs.image_grid_thw, return_dict=True, **vision_kwargs
        )

    with torch.inference_mode():
        run_vision()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run_vision()
        end.record()
        torch.cuda.synchronize()
        print(f"vision replay path: {start.elapsed_time(end)/20:.3f} ms/image")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
