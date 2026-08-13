#!/usr/bin/env python3
"""Explore the 15 TorchAO W8A16 M/F/L/V combinations with live progress."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from benchmark_public import (
    Sample,
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


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Qwen3.5-2B"
DATASET_EN = ROOT / "datasets/mmbench/mmbench_dev_en.tsv"
DATASET_CN = ROOT / "datasets/mmbench/mmbench_dev_cn.tsv"
BASELINE_EN = ROOT / "baseline_full_en.json"
BASELINE_CN = ROOT / "baseline_full_cn.json"
OUTPUT_DIR = ROOT / "quantization_exploration"
DEVICE = "auto"
SEED = 20260625
WARMUP_SAMPLES = 2
SCREEN64_REPEATS = 3
SCREEN256_REPEATS = 2
FULL_REPEATS = 1
MICROBENCH_WARMUP = 20
MICROBENCH_REPEATS = 100
GROUP_ORDER = "MFLV"
COMBINATIONS = (
    "Q_1000", "Q_0100", "Q_0010", "Q_0001",
    "Q_1100", "Q_1010", "Q_1001", "Q_0110", "Q_0101", "Q_0011",
    "Q_1110", "Q_1101", "Q_1011", "Q_0111",
    "Q_1111",
)

MODULE_PATTERNS = {
    "M": re.compile(
        r"^model\.language_model\.layers\.\d+\.mlp\."
        r"(?:gate_proj|up_proj|down_proj)$"
    ),
    "F": re.compile(
        r"^model\.language_model\.layers\.\d+\.self_attn\."
        r"(?:q_proj|k_proj|v_proj|o_proj)$"
    ),
    "L": re.compile(
        r"^model\.language_model\.layers\.\d+\.linear_attn\."
        r"(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"
    ),
    "V": re.compile(
        r"^model\.visual\.blocks\.\d+\."
        r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
    ),
}
EXPECTED_MODULE_COUNTS = {"M": 72, "F": 24, "L": 90, "V": 96}


@dataclass(frozen=True)
class ManifestEntry:
    language: str
    sample_id: str


@dataclass
class SamplingRow:
    sample_id: str
    category: str
    subcategory: str
    prompt_chars: int
    image_area: int
    prompt_bin: int = 0
    image_bin: int = 0

    @property
    def stratum(self) -> tuple[str, str, int, int]:
        return self.category, self.subcategory, self.prompt_bin, self.image_bin


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def combo_groups(combo: str) -> set[str]:
    bits = combo.removeprefix("Q_")
    return {
        group
        for group, enabled in zip(GROUP_ORDER, bits)
        if enabled == "1"
    }


def classify_fqn(fqn: str) -> str | None:
    for group, pattern in MODULE_PATTERNS.items():
        if pattern.fullmatch(fqn):
            return group
    return None


def parse_manifest(path: Path) -> list[ManifestEntry]:
    return [ManifestEntry(**row) for row in load_json(path)["entries"]]


def prompt_chars(en_sample: Sample, cn_sample: Sample) -> int:
    def count(sample: Sample) -> int:
        return len(sample.question) + len(sample.hint) + sum(
            len(choice) for choice in sample.choices.values()
        )

    return count(en_sample) + count(cn_sample)


def assign_quantile_bins(rows: list[SamplingRow], source: str, target: str) -> None:
    values = sorted(getattr(row, source) for row in rows)
    edges = [values[int((len(values) - 1) * q)] for q in (0.25, 0.50, 0.75)]
    for row in rows:
        setattr(row, target, sum(getattr(row, source) > edge for edge in edges))


def proportional_stratified_ids(
    rows: list[SamplingRow], count: int, seed: int
) -> list[str]:
    groups: dict[tuple[str, str, int, int], list[SamplingRow]] = defaultdict(list)
    for row in rows:
        groups[row.stratum].append(row)

    quotas: dict[tuple[str, str, int, int], int] = {}
    fractions: list[tuple[float, tuple[str, str, int, int]]] = []
    allocated = 0
    for key, group_rows in groups.items():
        exact = count * len(group_rows) / len(rows)
        quotas[key] = math.floor(exact)
        allocated += quotas[key]
        fractions.append((exact - quotas[key], key))

    fractions.sort(key=lambda item: (item[0], repr(item[1])), reverse=True)
    for _, key in fractions[: count - allocated]:
        quotas[key] += 1

    selected: list[SamplingRow] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        local_rows = list(group_rows)
        random.Random(f"{seed}:{key!r}").shuffle(local_rows)
        selected.extend(local_rows[: quotas[key]])
    random.Random(seed).shuffle(selected)
    return [row.sample_id for row in selected]


def build_manifests() -> dict[str, Path]:
    manifest_dir = OUTPUT_DIR / "manifests"
    paths = {
        "screen64": manifest_dir / "bilingual_64.json",
        "screen256": manifest_dir / "bilingual_256.json",
        "full": manifest_dir / "bilingual_full.json",
    }
    tqdm.write("[manifest] 读取中英文数据并生成固定分层样本……")
    en_samples = load_mmbench_tsv(DATASET_EN)
    cn_samples = load_mmbench_tsv(DATASET_CN)
    en_by_id = {sample.sample_id: sample for sample in en_samples}
    cn_by_id = {sample.sample_id: sample for sample in cn_samples}
    common_ids = [sample.sample_id for sample in en_samples if sample.sample_id in cn_by_id]

    rows: list[SamplingRow] = []
    for sample_id in tqdm(common_ids, desc="分析题型/图片/prompt 分层", unit="题"):
        en_sample = en_by_id[sample_id]
        cn_sample = cn_by_id[sample_id]
        image = decode_image(en_sample.image_b64)
        width, height = image.size
        image.close()
        rows.append(
            SamplingRow(
                sample_id=sample_id,
                category=en_sample.category or cn_sample.category or "unknown",
                subcategory=en_sample.subcategory or cn_sample.subcategory or "unknown",
                prompt_chars=prompt_chars(en_sample, cn_sample),
                image_area=width * height,
            )
        )

    assign_quantile_bins(rows, "prompt_chars", "prompt_bin")
    assign_quantile_bins(rows, "image_area", "image_bin")
    ids256 = proportional_stratified_ids(rows, 128, SEED)
    ids256_set = set(ids256)
    rows256 = [row for row in rows if row.sample_id in ids256_set]
    ids64 = proportional_stratified_ids(rows256, 32, SEED + 64)

    def paired(ids: Iterable[str]) -> list[ManifestEntry]:
        return [
            entry
            for sample_id in ids
            for entry in (
                ManifestEntry("en", sample_id),
                ManifestEntry("cn", sample_id),
            )
        ]

    manifests = {
        "screen64": paired(ids64),
        "screen256": paired(ids256),
        "full": (
            [ManifestEntry("en", sample.sample_id) for sample in en_samples]
            + [ManifestEntry("cn", sample.sample_id) for sample in cn_samples]
        ),
    }
    for stage, entries in manifests.items():
        save_json(
            paths[stage],
            {
                "stage": stage,
                "seed": SEED,
                "sample_count": len(entries),
                "entries": [asdict(entry) for entry in entries],
            },
        )
    return paths


def load_samples_for_manifest(
    manifest: list[ManifestEntry], dataset_en: Path, dataset_cn: Path
) -> list[Sample]:
    sources = {
        "en": {sample.sample_id: sample for sample in load_mmbench_tsv(dataset_en)},
        "cn": {sample.sample_id: sample for sample in load_mmbench_tsv(dataset_cn)},
    }
    return [sources[row.language][row.sample_id] for row in manifest]


def baseline_accuracy_for_manifest(
    manifest: list[ManifestEntry], baseline_en: Path, baseline_cn: Path
) -> float:
    payloads = {"en": load_json(baseline_en), "cn": load_json(baseline_cn)}
    correct = {
        language: {
            str(answer["question_id"]): bool(answer["correct"])
            for answer in payload["answers"]
        }
        for language, payload in payloads.items()
    }
    return sum(correct[row.language][row.sample_id] for row in manifest) / len(manifest)


def torchao_api() -> tuple[Any, Any, str]:
    import torchao
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    return quantize_, Int8WeightOnlyConfig, torchao.__version__


def apply_torchao_quantization(model: Any, combo: str) -> dict[str, Any]:
    import torch

    quantize_, Int8WeightOnlyConfig, torchao_version = torchao_api()
    active_groups = combo_groups(combo)
    selected_names: set[str] = set()
    counts = {group: 0 for group in GROUP_ORDER}
    selected_modules: list[Any] = []
    for name, module in model.named_modules():
        group = classify_fqn(name)
        if group in active_groups and isinstance(module, torch.nn.Linear):
            selected_names.add(name)
            selected_modules.append(module)
            counts[group] += 1

    for group in active_groups:
        if counts[group] != EXPECTED_MODULE_COUNTS[group]:
            raise RuntimeError(
                f"{group} 匹配到 {counts[group]} 个 Linear，"
                f"预期 {EXPECTED_MODULE_COUNTS[group]} 个"
            )

    def filter_fn(module: Any, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear) and fqn in selected_names

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    quantize_(model, Int8WeightOnlyConfig(), filter_fn=filter_fn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    weight_types: dict[str, int] = defaultdict(int)
    for module in selected_modules:
        name = f"{type(module.weight).__module__}.{type(module.weight).__name__}"
        weight_types[name] += 1

    return {
        "backend": "torchao.Int8WeightOnlyConfig",
        "torchao_version": torchao_version,
        "active_groups": sorted(active_groups),
        "module_count": len(selected_names),
        "module_count_by_group": counts,
        "weight_types": dict(weight_types),
        "quantization_seconds": time.perf_counter() - started,
    }


def cpu_peak_rss_bytes() -> int:
    if os.name != "nt":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize)


def run_combo(
    stage: str,
    combo: str,
    manifest_path: Path,
    repeats: int,
    result_path: Path,
) -> None:
    import torch
    import transformers

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    manifest = parse_manifest(manifest_path)
    samples = load_samples_for_manifest(manifest, DATASET_EN, DATASET_CN)
    tqdm.write(
        f"[{stage}] {combo}: 加载模型，{len(samples)} 条 × {repeats} 次"
    )

    started = time.perf_counter()
    model = VLMModel(str(MODEL_PATH), backend="transformers", device=DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    quantization = apply_torchao_quantization(model._model, combo)
    initialization_peak_gpu = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )

    for sample in tqdm(
        samples[:WARMUP_SAMPLES],
        desc=f"{combo} 预热",
        unit="条",
        leave=False,
    ):
        settle_runtime(model)
        model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=fixed_generation_config(),
            sample_id=sample.sample_id,
        )

    runs: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        correct = 0
        failed = 0
        ttfts: list[float] = []
        throughputs: list[float] = []
        answers: list[dict[str, Any]] = []
        repeat_started = time.perf_counter()
        progress = tqdm(
            samples,
            desc=(
                f"{stage} {combo} 第 {repeat}/{repeats} 次"
            ),
            unit="条",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for index, sample in enumerate(progress, start=1):
            settle_runtime(model)
            result = model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=fixed_generation_config(),
                sample_id=sample.sample_id,
            )
            parsed = extract_answer(result.text)
            errors = validate_public_result(
                result.text,
                parsed,
                result.token_count,
                fixed_generation_config().max_new_tokens,
            )
            is_correct = parsed == sample.answer
            correct += int(is_correct)
            failed += int(bool(errors))
            ttft_ms = result.ttft_seconds * 1000.0
            throughput = compute_throughput(
                result.token_count,
                result.ttft_seconds,
                result.elapsed_seconds,
            )
            ttfts.append(ttft_ms)
            throughputs.append(throughput)
            answers.append(
                {
                    "question_id": sample.sample_id,
                    "language": sample.language,
                    "parsed_answer": parsed,
                    "correct": is_correct,
                    "ttft_ms": round(ttft_ms, 3),
                    "throughput_tokens_per_sec": round(throughput, 3),
                    "token_count": result.token_count,
                    "validation_errors": errors,
                }
            )

            elapsed = time.perf_counter() - repeat_started
            eta = elapsed / index * (len(samples) - index)
            progress.set_postfix_str(
                f"acc={correct / index:.2%} "
                f"ttft={statistics.fmean(ttfts):.1f}ms "
                f"tok/s={statistics.fmean(throughputs):.1f} "
                f"ETA={eta / 60:.1f}min"
            )

        runs.append(
            {
                "repeat": repeat,
                "accuracy": correct / len(samples),
                "avg_ttft_ms": statistics.fmean(ttfts),
                "avg_throughput_tokens_per_sec": statistics.fmean(throughputs),
                "validation_failed_samples": failed,
                "elapsed_seconds": time.perf_counter() - repeat_started,
                "answers": answers,
            }
        )

    aggregate = {
        "accuracy": statistics.median([run["accuracy"] for run in runs]),
        "avg_ttft_ms": statistics.median([run["avg_ttft_ms"] for run in runs]),
        "avg_throughput_tokens_per_sec": statistics.median(
            [run["avg_throughput_tokens_per_sec"] for run in runs]
        ),
    }
    payload = {
        "stage": stage,
        "combo": combo,
        "sample_count": len(samples),
        "repeats": repeats,
        "aggregate": aggregate,
        "runs": runs,
        "quantization": quantization,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "quantization_seconds": quantization["quantization_seconds"],
        },
        "memory": {
            "initialization_peak_gpu_allocated_bytes": initialization_peak_gpu,
            "total_peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "peak_cpu_rss_bytes": cpu_peak_rss_bytes(),
        },
    }
    save_json(result_path, payload)
    tqdm.write(
        f"[{stage}] {combo} 完成："
        f"accuracy={aggregate['accuracy']:.4f}, "
        f"TTFT={aggregate['avg_ttft_ms']:.2f}ms, "
        f"throughput={aggregate['avg_throughput_tokens_per_sec']:.2f} tok/s"
    )
    del model, samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rank_stage(
    stage: str,
    combos: list[str],
    manifest_path: Path,
    use_accuracy_gate: bool,
) -> dict[str, Any]:
    manifest = parse_manifest(manifest_path)
    baseline_accuracy = (
        baseline_accuracy_for_manifest(manifest, BASELINE_EN, BASELINE_CN)
        if use_accuracy_gate
        else None
    )
    rows: list[dict[str, Any]] = []
    for combo in combos:
        aggregate = load_json(OUTPUT_DIR / stage / f"{combo}.json")["aggregate"]
        gate_passed = (
            baseline_accuracy is None
            or aggregate["accuracy"] >= baseline_accuracy - 0.02
        )
        rows.append(
            {
                "combo": combo,
                **aggregate,
                "accuracy_gate_passed": gate_passed,
            }
        )

    eligible = [row for row in rows if row["accuracy_gate_passed"]]
    max_accuracy = max(row["accuracy"] for row in eligible)
    min_accuracy = min(row["accuracy"] for row in eligible)
    min_ttft = min(row["avg_ttft_ms"] for row in eligible)
    max_throughput = max(row["avg_throughput_tokens_per_sec"] for row in eligible)
    for row in rows:
        row["final_score_max_ref"] = (
            (row["accuracy"] - min_accuracy) / (max_accuracy - min_accuracy) * 4
            - row["avg_ttft_ms"] / min_ttft * 3
            + row["avg_throughput_tokens_per_sec"] / max_throughput * 3
            if row["accuracy_gate_passed"]
            else None
        )

    rows.sort(
        key=lambda row: (
            row["accuracy_gate_passed"],
            row["final_score_max_ref"] or -math.inf,
            row["accuracy"],
            -row["avg_ttft_ms"],
            row["avg_throughput_tokens_per_sec"],
        ),
        reverse=True,
    )
    rank = 0
    for row in rows:
        if row["accuracy_gate_passed"]:
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = None

    ranking = {
        "stage": stage,
        "sample_count": len(manifest),
        "baseline_accuracy": baseline_accuracy,
        "normalization": {
            "max_accuracy": max_accuracy,
            "min_accuracy": min_accuracy,
            "min_avg_ttft_ms": min_ttft,
            "max_avg_throughput_tokens_per_sec": max_throughput,
            "baseline_included": False,
        },
        "ranking": rows,
    }
    save_json(OUTPUT_DIR / f"{stage}_ranking.json", ranking)
    print_ranking(ranking)
    return ranking


def print_ranking(ranking: dict[str, Any]) -> None:
    print(f"\n[{ranking['stage']}] 双语综合排名")
    print(
        f"{'Rank':>4}  {'Combo':<7} {'Gate':<4} {'Accuracy':>9} "
        f"{'TTFT(ms)':>10} {'Tok/s':>9} {'MaxRef':>9}"
    )
    for row in ranking["ranking"]:
        score = (
            f"{row['final_score_max_ref']:.5f}"
            if row["final_score_max_ref"] is not None
            else "-"
        )
        print(
            f"{str(row['rank'] or '-'):>4}  {row['combo']:<7} "
            f"{('yes' if row['accuracy_gate_passed'] else 'no'):<4} "
            f"{row['accuracy']:>9.5f} {row['avg_ttft_ms']:>10.2f} "
            f"{row['avg_throughput_tokens_per_sec']:>9.2f} {score:>9}"
        )
    print()


def promoted_combos(ranking: dict[str, Any], count: int) -> list[str]:
    return [
        row["combo"]
        for row in ranking["ranking"]
        if row["accuracy_gate_passed"]
    ][:count]


def run_stage(
    stage: str,
    combos: list[str],
    manifest_path: Path,
    repeats: int,
    use_accuracy_gate: bool,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIR / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(parse_manifest(manifest_path))
    progress = tqdm(combos, desc=f"{stage} 组合", unit="组")
    for combo in progress:
        result_path = output_dir / f"{combo}.json"
        progress.set_postfix_str(combo)
        tqdm.write(
            f"\n[{stage}] 开始 {combo}：{sample_count} 条 × {repeats} 次"
        )
        run_combo(stage, combo, manifest_path, repeats, result_path)
    return rank_stage(stage, combos, manifest_path, use_accuracy_gate)


def discover_linear_shapes(model_path: Path) -> list[dict[str, Any]]:
    from safetensors import safe_open

    index = load_json(model_path / "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for weight_name, shard in index["weight_map"].items():
        if weight_name.endswith(".weight") and classify_fqn(weight_name[:-7]):
            by_shard[shard].append(weight_name)

    shapes: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for shard, weight_names in by_shard.items():
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as handle:
            for weight_name in weight_names:
                out_features, in_features = handle.get_slice(weight_name).get_shape()
                group = classify_fqn(weight_name[:-7])
                shapes[(group, in_features, out_features)].append(weight_name)

    return [
        {
            "group": group,
            "in_features": in_features,
            "out_features": out_features,
            "module_count": len(names),
            "example": names[0],
        }
        for (group, in_features, out_features), names in sorted(shapes.items())
    ]


def cuda_latencies_ms(module: Any, inputs: Any, warmup: int, repeats: int) -> list[float]:
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            module(inputs)
        torch.cuda.synchronize()
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(inputs)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def run_microbenchmark() -> None:
    import torch

    quantize_, Int8WeightOnlyConfig, torchao_version = torchao_api()
    shapes = discover_linear_shapes(MODEL_PATH)
    cases = [
        (shape, tokens)
        for shape in shapes
        for tokens in ((256, 1024) if shape["group"] == "V" else (1, 128))
    ]

    results = []
    for shape, tokens in tqdm(cases, desc="W8A16 等形状微基准", unit="组"):
        dense = torch.nn.Sequential(
            torch.nn.Linear(
                shape["in_features"],
                shape["out_features"],
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
        ).eval()
        quantized = torch.nn.Sequential(
            torch.nn.Linear(
                shape["in_features"],
                shape["out_features"],
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
        ).eval()
        quantized[0].weight.data.copy_(dense[0].weight.data)
        quantize_(quantized, Int8WeightOnlyConfig())
        inputs = torch.randn(
            tokens,
            shape["in_features"],
            device="cuda",
            dtype=torch.bfloat16,
        )

        dense_ms = cuda_latencies_ms(
            dense, inputs, MICROBENCH_WARMUP, MICROBENCH_REPEATS
        )
        quantized_ms = cuda_latencies_ms(
            quantized, inputs, MICROBENCH_WARMUP, MICROBENCH_REPEATS
        )
        dense_median = statistics.median(dense_ms)
        quantized_median = statistics.median(quantized_ms)
        results.append(
            {
                **shape,
                "tokens": tokens,
                "dense_median_ms": dense_median,
                "dense_p90_ms": percentile(dense_ms, 0.90),
                "w8a16_median_ms": quantized_median,
                "w8a16_p90_ms": percentile(quantized_ms, 0.90),
                "median_speedup": dense_median / quantized_median,
                "quantized_weight_type": (
                    f"{type(quantized[0].weight).__module__}."
                    f"{type(quantized[0].weight).__name__}"
                ),
            }
        )
        del dense, quantized, inputs
        torch.cuda.empty_cache()

    save_json(
        OUTPUT_DIR / "microbenchmark.json",
        {
            "backend": "torchao.Int8WeightOnlyConfig",
            "torchao_version": torchao_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "warmup": MICROBENCH_WARMUP,
            "repeats": MICROBENCH_REPEATS,
            "results": results,
        },
    )
    tqdm.write(
        f"[microbenchmark] 完成 {len(results)} 组，speedup="
        f"{min(row['median_speedup'] for row in results):.3f}x–"
        f"{max(row['median_speedup'] for row in results):.3f}x"
    )


def main() -> None:
    manifests = build_manifests()
    torchao_api()
    run_microbenchmark()

    ranking64 = run_stage(
        "screen64",
        list(COMBINATIONS),
        manifests["screen64"],
        SCREEN64_REPEATS,
        use_accuracy_gate=False,
    )

    top6 = promoted_combos(ranking64, 6)
    ranking256 = run_stage(
        "screen256",
        top6,
        manifests["screen256"],
        SCREEN256_REPEATS,
        use_accuracy_gate=True,
    )

    top3 = promoted_combos(ranking256, 3)
    final_ranking = run_stage(
        "full",
        top3,
        manifests["full"],
        FULL_REPEATS,
        use_accuracy_gate=True,
    )
    save_json(
        OUTPUT_DIR / "final_summary.json",
        {
            "screen64_promoted": top6,
            "screen256_promoted": top3,
            "final_ranking": final_ranking,
        },
    )


if __name__ == "__main__":
    main()
