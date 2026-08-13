#!/usr/bin/env python3
"""Run the complete English and Chinese public baselines with live progress."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm
from transformers import TextIteratorStreamer

from benchmark_public import (
    build_prompt,
    compute_throughput,
    decode_image,
    extract_answer,
    fixed_generation_config,
    load_mmbench_tsv,
    settle_runtime,
    validate_public_result,
)
from evaluation_wrapper import VLMModel


DATASETS = (
    ("English", Path("datasets/mmbench/mmbench_dev_en.tsv"), Path("baseline_full_en.json")),
    ("Chinese", Path("datasets/mmbench/mmbench_dev_cn.tsv"), Path("baseline_full_cn.json")),
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class ProfiledVLMModel(VLMModel):
    """Baseline model with measurement only; inference semantics are unchanged."""

    def generate_profiled(self, *, image, prompt, choices, generation_config, sample_id):
        import torch

        if self.backend_name != "transformers":
            raise RuntimeError("Profiling requires the transformers backend")

        cpu_start = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        processor_seconds = time.perf_counter() - cpu_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_start = time.perf_counter()
        inputs = inputs.to(self._model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_seconds = time.perf_counter() - transfer_start

        input_len = inputs.input_ids.shape[1]
        image_tokens = int((inputs.input_ids == self._model.config.image_token_id).sum().item())
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.temperature > 0,
            "use_cache": True,
            "streamer": streamer,
        }

        event_pairs: dict[str, list[tuple[object, object]]] = {
            "vision": [],
            "language": [],
        }
        cpu_forward_ranges: dict[str, list[tuple[float, float]]] = {
            "vision": [],
            "language": [],
        }
        active_cpu: dict[str, list[float]] = {"vision": [], "language": []}
        active_events: dict[str, list[object]] = {"vision": [], "language": []}

        def pre_hook(name):
            def hook(_module, _args):
                active_cpu[name].append(time.perf_counter())
                if torch.cuda.is_available():
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    active_events[name].append(event)
            return hook

        def post_hook(name):
            def hook(_module, _args, output):
                end_cpu = time.perf_counter()
                start_cpu = active_cpu[name].pop()
                cpu_forward_ranges[name].append((start_cpu, end_cpu))
                if torch.cuda.is_available():
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    event_pairs[name].append((active_events[name].pop(), end_event))
            return hook

        handles = [
            self._model.model.visual.register_forward_pre_hook(pre_hook("vision")),
            self._model.model.visual.register_forward_hook(post_hook("vision")),
            self._model.model.language_model.register_forward_pre_hook(pre_hook("language")),
            self._model.model.language_model.register_forward_hook(post_hook("language")),
        ]

        output_holder: dict[str, object] = {}
        worker_error: dict[str, BaseException] = {}

        def run_generate() -> None:
            try:
                with torch.inference_mode():
                    output_holder["output_ids"] = self._model.generate(**generation_kwargs)
            except BaseException as exc:
                worker_error["error"] = exc

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        memory_before_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

        chunks: list[str] = []
        chunk_arrivals: list[float] = []
        generation_start = time.perf_counter()
        worker = threading.Thread(target=run_generate, daemon=True)
        worker.start()
        for chunk in streamer:
            chunk_received = time.perf_counter()
            chunks.append(chunk)
            chunk_arrivals.append(chunk_received)
        worker.join()
        generation_end = time.perf_counter()

        for handle in handles:
            handle.remove()
        if worker_error:
            raise worker_error["error"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        def elapsed_events(name: str) -> list[float]:
            return [start.elapsed_time(end) for start, end in event_pairs[name]]

        vision_gpu_ms = elapsed_events("vision")
        language_gpu_ms = elapsed_events("language")
        prefill_gpu_ms = language_gpu_ms[0] if language_gpu_ms else 0.0
        decode_gpu_ms = language_gpu_ms[1:] if len(language_gpu_ms) > 1 else []

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        first_chunk_at = next(
            (timestamp for chunk, timestamp in zip(chunks, chunk_arrivals) if chunk),
            generation_end,
        )
        generation_seconds = generation_end - generation_start
        decode_total_gpu_ms = sum(decode_gpu_ms)

        profile = {
            "processor_chat_template_ms": processor_seconds * 1000.0,
            "cpu_to_device_ms": transfer_seconds * 1000.0,
            "vision_encoder_gpu_ms": sum(vision_gpu_ms),
            "language_prefill_gpu_ms": prefill_gpu_ms,
            "decode_total_gpu_ms": decode_total_gpu_ms,
            "decode_gpu_ms_per_token": (
                sum(decode_gpu_ms) / len(decode_gpu_ms) if decode_gpu_ms else 0.0
            ),
            "input_tokens": int(input_len),
            "image_tokens": image_tokens,
            "generated_tokens": int(generated_ids.shape[0]),
            "cuda_memory_allocated_before_bytes": memory_before_allocated,
            "cuda_memory_reserved_before_bytes": memory_before_reserved,
            "cuda_peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "cuda_peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
        }

        return {
            "text": text,
            "token_count": int(generated_ids.shape[0]),
            "ttft_seconds": first_chunk_at - generation_start,
            "elapsed_seconds": generation_seconds,
            "profile": profile,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all English and Chinese public baseline samples with progress"
    )
    parser.add_argument("--model-path", default="./Qwen3.5-2B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--warmup-samples", type=int, default=2)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "avg": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def run_one(
    *,
    label: str,
    dataset_path: Path,
    output_path: Path,
    model_path: str,
    device: str,
    seed: int,
    warmup_samples: int,
) -> dict:
    benchmark_start = time.perf_counter()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    dataset_path = dataset_path.resolve()
    output_path = output_path.resolve()
    samples = load_mmbench_tsv(dataset_path, limit=None)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    print(f"\n[{label}] Loading model: {model_path}", flush=True)
    model = ProfiledVLMModel(model_path, backend="transformers", device=device)

    warmup_count = min(warmup_samples, len(samples))
    if warmup_count:
        print(f"[{label}] Warmup: {warmup_count} samples", flush=True)
    for sample in tqdm(
        samples[:warmup_count],
        desc=f"{label} warmup",
        unit="sample",
        dynamic_ncols=True,
    ):
        settle_runtime(model)
        model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=fixed_generation_config(),
            sample_id=sample.sample_id,
        )
        settle_runtime(model)

    records: list[dict] = []
    ttfts_ms: list[float] = []
    throughputs: list[float] = []
    correct = 0
    validation_errors = 0
    stage_values: dict[str, list[float]] = {
        "image_decode_pillow_ms": [],
        "processor_chat_template_ms": [],
        "cpu_to_device_ms": [],
        "vision_encoder_gpu_ms": [],
        "language_prefill_gpu_ms": [],
        "decode_total_gpu_ms": [],
        "decode_gpu_ms_per_token": [],
    }
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0

    progress = tqdm(
        samples,
        total=len(samples),
        desc=f"{label} baseline",
        unit="sample",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for index, sample in enumerate(progress, start=1):
        settle_runtime(model)
        config = fixed_generation_config()
        image_start = time.perf_counter()
        image = decode_image(sample.image_b64)
        image_decode_ms = (time.perf_counter() - image_start) * 1000.0
        result = model.generate_profiled(
            image=image,
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        parsed_answer = extract_answer(result["text"])
        errors = validate_public_result(
            result["text"],
            parsed_answer,
            result["token_count"],
            config.max_new_tokens,
        )
        validation_errors += int(bool(errors))
        is_correct = parsed_answer == sample.answer
        correct += int(is_correct)

        ttft_ms = result["ttft_seconds"] * 1000.0
        throughput = compute_throughput(
            result["token_count"],
            result["ttft_seconds"],
            result["elapsed_seconds"],
        )
        if math.isfinite(ttft_ms) and ttft_ms > 0:
            ttfts_ms.append(ttft_ms)
        if math.isfinite(throughput) and throughput > 0:
            throughputs.append(throughput)

        sample_profile = result["profile"]
        sample_profile["image_decode_pillow_ms"] = image_decode_ms
        stage_values["image_decode_pillow_ms"].append(image_decode_ms)
        for key in (
            "processor_chat_template_ms",
            "cpu_to_device_ms",
            "vision_encoder_gpu_ms",
            "language_prefill_gpu_ms",
            "decode_total_gpu_ms",
            "decode_gpu_ms_per_token",
        ):
            stage_values[key].append(float(sample_profile[key]))
        peak_allocated_bytes = max(
            peak_allocated_bytes,
            int(sample_profile["cuda_peak_memory_allocated_bytes"]),
        )
        peak_reserved_bytes = max(
            peak_reserved_bytes,
            int(sample_profile["cuda_peak_memory_reserved_bytes"]),
        )

        records.append(
            {
                "question_id": sample.sample_id,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "ttft_ms": round(ttft_ms, 3),
                "throughput_tokens_per_sec": round(throughput, 3),
                "token_count": result["token_count"],
                "validation_errors": errors,
                "meta": {"backend": "transformers"},
                "profile": sample_profile,
            }
        )
        settle_runtime(model)

        elapsed = time.perf_counter() - benchmark_start
        eta = elapsed / index * (len(samples) - index)
        avg_ttft = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0.0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0
        progress.set_postfix_str(
            f"acc={correct / index:.2%} "
            f"ttft={avg_ttft:.1f}ms "
            f"tok/s={avg_throughput:.1f} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)}"
        )

    elapsed = time.perf_counter() - benchmark_start
    payload = {
        "benchmark_version": "dndx_public_self_test",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "seed": seed,
        "backend": model.backend_name,
        "performance": {
            "avg_ttft_ms": round(sum(ttfts_ms) / len(ttfts_ms), 3) if ttfts_ms else None,
            "avg_throughput_tokens_per_sec": (
                round(sum(throughputs) / len(throughputs), 3) if throughputs else 0.0
            ),
        },
        "profiling": {
            "stage_ms": {key: summarize(values) for key, values in stage_values.items()},
            "cuda_peak_memory_allocated_bytes": peak_allocated_bytes,
            "cuda_peak_memory_reserved_bytes": peak_reserved_bytes,
            "measurement_notes": {
                "processor_chat_template": "Combined apply_chat_template, tokenization and image preprocessing time.",
                "vision_and_language": "CUDA event time measured by module hooks.",
                "decode": "Total and average CUDA time of language-model decode forward calls.",
            },
        },
        "timing": {
            "benchmark_elapsed_seconds": round(elapsed, 3),
            "benchmark_elapsed_minutes": round(elapsed / 60.0, 3),
            "avg_seconds_per_sample": round(elapsed / len(samples), 3),
        },
        "accuracy": {
            "score": round(correct / len(samples), 6),
            "correct": correct,
            "total": len(samples),
        },
        "public_validation": {
            "passed": validation_errors == 0,
            "failed_samples": validation_errors,
        },
        "answers": records,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[{label}] Complete: "
        f"accuracy={payload['accuracy']['score']:.2%}, "
        f"avg_ttft={payload['performance']['avg_ttft_ms']} ms, "
        f"throughput={payload['performance']['avg_throughput_tokens_per_sec']} tokens/s, "
        f"output={output_path}",
        flush=True,
    )

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return payload


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    summaries = []
    for label, relative_dataset, relative_output in DATASETS:
        payload = run_one(
            label=label,
            dataset_path=script_dir / relative_dataset,
            output_path=script_dir / relative_output,
            model_path=str((script_dir / args.model_path).resolve()),
            device=args.device,
            seed=args.seed,
            warmup_samples=args.warmup_samples,
        )
        summaries.append(
            {
                "language": label,
                "sample_count": payload["sample_count"],
                "accuracy": payload["accuracy"]["score"],
                "avg_ttft_ms": payload["performance"]["avg_ttft_ms"],
                "avg_throughput_tokens_per_sec": payload["performance"][
                    "avg_throughput_tokens_per_sec"
                ],
                "public_validation_passed": payload["public_validation"]["passed"],
            }
        )

    print("\nAll baselines complete:", flush=True)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
