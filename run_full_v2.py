#!/usr/bin/env python3
"""Run the complete English and Chinese datasets with the latest retained optimizations."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

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
    ("English", Path("datasets/mmbench/mmbench_dev_en.tsv"), Path("fused_full_v2_en.json")),
    ("Chinese", Path("datasets/mmbench/mmbench_dev_cn.tsv"), Path("fused_full_v2_cn.json")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all English and Chinese public samples with kernel fusion enabled"
    )
    parser.add_argument("--model-path", default="./Qwen3.5-2B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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

    print(f"\n[{label}] Loading fused model: {model_path}", flush=True)
    model = VLMModel(model_path, backend="transformers", device=device)
    fusion_stats = getattr(model, "_fusion_stats", {})
    if not fusion_stats:
        raise RuntimeError("Kernel fusion was not applied")

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

    progress = tqdm(
        samples,
        total=len(samples),
        desc=f"{label} fused",
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
        result = model.generate_with_metrics(
            image=image,
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        parsed_answer = extract_answer(result.text)
        errors = validate_public_result(
            result.text,
            parsed_answer,
            result.token_count,
            config.max_new_tokens,
        )
        validation_errors += int(bool(errors))
        is_correct = parsed_answer == sample.answer
        correct += int(is_correct)

        ttft_ms = result.ttft_seconds * 1000.0
        throughput = compute_throughput(
            result.token_count,
            result.ttft_seconds,
            result.elapsed_seconds,
        )
        if math.isfinite(ttft_ms) and ttft_ms > 0:
            ttfts_ms.append(ttft_ms)
        if math.isfinite(throughput) and throughput > 0:
            throughputs.append(throughput)

        records.append(
            {
                "question_id": sample.sample_id,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "ttft_ms": round(ttft_ms, 3),
                "throughput_tokens_per_sec": round(throughput, 3),
                "token_count": result.token_count,
                "image_decode_pillow_ms": round(image_decode_ms, 3),
                "validation_errors": errors,
                "meta": result.meta,
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
        "benchmark_version": "dndx_public_self_test_fused",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "seed": seed,
        "backend": model.backend_name,
        "fusion_stats": fusion_stats,
        "performance": {
            "avg_ttft_ms": round(sum(ttfts_ms) / len(ttfts_ms), 3) if ttfts_ms else None,
            "avg_throughput_tokens_per_sec": (
                round(sum(throughputs) / len(throughputs), 3) if throughputs else 0.0
            ),
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
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for label, relative_dataset, relative_output in DATASETS:
        payload = run_one(
            label=label,
            dataset_path=script_dir / relative_dataset,
            output_path=output_dir / relative_output,
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

    print("\nAll fused runs complete:", flush=True)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
