#!/usr/bin/env python3
"""Explore 16 PPU-native A8W8 M/F/L/V candidates with live progress."""

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

import torch
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
DEVICE = "cuda:0"
SEED = 20260625
WARMUP_SAMPLES = 2
SCREEN64_REPEATS = 3
SCREEN256_REPEATS = 2
FULL_REPEATS = 1
BASELINE_POLL_SECONDS = 30
MICROBENCH_WARMUP = 20
MICROBENCH_REPEATS = 100
COMPILE_MODE = "max-autotune"
COMPILE_DYNAMIC = True
GROUP_ORDER = "MFLV"
QUANTIZATION_SCHEMES = ("PPU_A8W8",)
COMBINATIONS = (
    "Q_0000",
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


@dataclass(frozen=True)
class Candidate:
    scheme: str
    combo: str

    @property
    def name(self) -> str:
        return f"{self.scheme}_{self.combo}"


CANDIDATES = tuple(
    Candidate(scheme, combo)
    for scheme in QUANTIZATION_SCHEMES
    for combo in COMBINATIONS
)


_PPU_SCALED_INT8_QUANT: Any | None = None
_PPU_INT8_GEMM: Any | None = None


def ppu_a8w8_api() -> tuple[Any, Any, str]:
    global _PPU_SCALED_INT8_QUANT, _PPU_INT8_GEMM

    import acext
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.kernels.scaled_mm import cutlass

    if acext.int8_gemm is None:
        raise RuntimeError("acext.int8_gemm 未注册，无法执行 PPU 原生 A8W8")
    _PPU_SCALED_INT8_QUANT = vllm_ops.scaled_int8_quant
    _PPU_INT8_GEMM = torch.ops.vllm.w8a8_int8_matmul_acext
    version = str(acext.get_version()) if acext.get_version is not None else "unknown"
    return _PPU_SCALED_INT8_QUANT, _PPU_INT8_GEMM, version


class PPUA8W8Linear(torch.nn.Module):
    """Dynamic per-token activation and per-output-channel weight A8W8 Linear."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError(
                "acext.int8_gemm 要求 in_features/out_features 均为 16 的倍数，"
                f"得到 {linear.in_features}->{linear.out_features}"
            )

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        with torch.no_grad():
            weight = linear.weight.detach().to(torch.float32)
            absmax = weight.abs().amax(dim=1, keepdim=True)
            weight_scale = torch.where(
                absmax > 0,
                absmax / 127.0,
                torch.ones_like(absmax),
            )
            quantized_weight = (
                torch.round(weight / weight_scale)
                .clamp_(-127, 127)
                .to(torch.int8)
                .contiguous()
            )

        self.register_buffer("weight", quantized_weight)
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_parameter("bias", linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if _PPU_SCALED_INT8_QUANT is None or _PPU_INT8_GEMM is None:
            raise RuntimeError("PPU A8W8 API 尚未初始化")

        original_shape = inputs.shape
        inputs_2d = inputs.reshape(-1, self.in_features).contiguous()
        quantized_inputs, input_scale, _ = _PPU_SCALED_INT8_QUANT(
            inputs_2d,
            scale=None,
            azp=None,
            symmetric=True,
        )
        outputs = _PPU_INT8_GEMM(
            quantized_inputs,
            self.weight,
            input_scale,
            self.weight_scale,
            inputs.dtype,
            self.bias,
        )
        return outputs.reshape(*original_shape[:-1], self.out_features)


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


def baseline_metrics_for_manifest(
    manifest: list[ManifestEntry], baseline_en: Path, baseline_cn: Path
) -> dict[str, float]:
    while missing := [
        path for path in (baseline_en, baseline_cn) if not path.exists()
    ]:
        tqdm.write(
            "[baseline] 等待文件："
            + ", ".join(str(path) for path in missing)
            + f"；{BASELINE_POLL_SECONDS} 秒后重试"
        )
        time.sleep(BASELINE_POLL_SECONDS)

    payloads = {"en": load_json(baseline_en), "cn": load_json(baseline_cn)}
    answers = {
        language: {
            str(answer["question_id"]): answer
            for answer in payload["answers"]
        }
        for language, payload in payloads.items()
    }
    selected = [answers[row.language][row.sample_id] for row in manifest]
    return {
        "accuracy": statistics.fmean(bool(answer["correct"]) for answer in selected),
        "avg_ttft_ms": statistics.fmean(float(answer["ttft_ms"]) for answer in selected),
        "avg_throughput_tokens_per_sec": statistics.fmean(
            float(answer["throughput_tokens_per_sec"]) for answer in selected
        ),
    }


def apply_ppu_a8w8_quantization(
    model: Any, candidate: Candidate
) -> dict[str, Any]:
    if candidate.scheme != "PPU_A8W8":
        raise ValueError(f"不支持的量化方案：{candidate.scheme}")

    _, _, acext_version = ppu_a8w8_api()
    active_groups = combo_groups(candidate.combo)
    selected_names: set[str] = set()
    counts = {group: 0 for group in GROUP_ORDER}
    selected_modules: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        group = classify_fqn(name)
        if group in active_groups and isinstance(module, torch.nn.Linear):
            selected_names.add(name)
            selected_modules.append((name, module))
            counts[group] += 1

    for group in active_groups:
        if counts[group] != EXPECTED_MODULE_COUNTS[group]:
            raise RuntimeError(
                f"{group} 匹配到 {counts[group]} 个 Linear，"
                f"预期 {EXPECTED_MODULE_COUNTS[group]} 个"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    quantized_modules: list[PPUA8W8Linear] = []
    for name, module in selected_modules:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        quantized = PPUA8W8Linear(module)
        setattr(parent, child_name, quantized)
        quantized_modules.append(quantized)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    weight_types: dict[str, int] = defaultdict(int)
    for module in quantized_modules:
        weight_types[str(module.weight.dtype)] += 1

    return {
        "scheme": candidate.scheme,
        "backend": (
            "bf16_control"
            if not active_groups
            else "vllm.scaled_int8_quant+acext.int8_gemm"
        ),
        "acext_version": acext_version,
        "activation_quantization": "dynamic_symmetric_per_token_int8",
        "weight_quantization": "symmetric_per_output_channel_int8",
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


def aggregate_runs(
    stage: str, runs: list[dict[str, Any]]
) -> tuple[str, dict[str, float]]:
    values = {
        key: [float(run[key]) for run in runs]
        for key in (
            "accuracy",
            "avg_ttft_ms",
            "avg_throughput_tokens_per_sec",
        )
    }
    if stage == "screen64":
        method = "median"
        reducer = statistics.median
    else:
        method = "mean" if stage == "screen256" else "single_run"
        reducer = statistics.fmean
    return method, {key: reducer(series) for key, series in values.items()}


def run_combo(
    stage: str,
    candidate: Candidate,
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
        f"[{stage}] {candidate.name}: 加载模型，{len(samples)} 条 × {repeats} 次"
    )

    started = time.perf_counter()
    model = VLMModel(str(MODEL_PATH), backend="transformers", device=DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    quantization = apply_ppu_a8w8_quantization(model._model, candidate)
    model._model.compile(mode=COMPILE_MODE, dynamic=COMPILE_DYNAMIC)
    initialization_peak_gpu = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )

    compile_warmup_started = time.perf_counter()
    for sample in tqdm(
        samples[:WARMUP_SAMPLES],
        desc=f"{candidate.name} 预热",
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compile_warmup_seconds = time.perf_counter() - compile_warmup_started

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
                f"{stage} {candidate.name} 第 {repeat}/{repeats} 次"
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

    aggregate_method, aggregate = aggregate_runs(stage, runs)
    payload = {
        "stage": stage,
        "candidate": candidate.name,
        "scheme": candidate.scheme,
        "combo": candidate.combo,
        "sample_count": len(samples),
        "repeats": repeats,
        "aggregate_method": aggregate_method,
        "aggregate": aggregate,
        "runs": runs,
        "quantization": quantization,
        "compilation": {
            "backend": "inductor",
            "mode": COMPILE_MODE,
            "dynamic": COMPILE_DYNAMIC,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "quantization_seconds": quantization["quantization_seconds"],
            "compile_warmup_seconds": compile_warmup_seconds,
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
        f"[{stage}] {candidate.name} 完成："
        f"accuracy={aggregate['accuracy']:.4f}, "
        f"TTFT={aggregate['avg_ttft_ms']:.2f}ms, "
        f"throughput={aggregate['avg_throughput_tokens_per_sec']:.2f} tok/s"
    )
    del model, samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def final_score_max_ref(
    metrics: dict[str, Any], normalization: dict[str, float]
) -> float:
    return (
        (metrics["accuracy"] - normalization["min_accuracy"])
        / (normalization["max_accuracy"] - normalization["min_accuracy"])
        * 4
        - metrics["avg_ttft_ms"] / normalization["min_avg_ttft_ms"] * 3
        + metrics["avg_throughput_tokens_per_sec"]
        / normalization["max_avg_throughput_tokens_per_sec"]
        * 3
    )


def rank_stage(
    stage: str,
    candidates: list[Candidate],
    manifest_path: Path,
    accuracy_gate_tolerance: float | None,
) -> dict[str, Any]:
    manifest = parse_manifest(manifest_path)
    baseline = (
        baseline_metrics_for_manifest(manifest, BASELINE_EN, BASELINE_CN)
        if accuracy_gate_tolerance is not None
        else None
    )
    gate_threshold = (
        baseline["accuracy"] - accuracy_gate_tolerance if baseline else None
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        aggregate = load_json(
            OUTPUT_DIR / stage / f"{candidate.name}.json"
        )["aggregate"]
        gate_passed = (
            gate_threshold is None
            or aggregate["accuracy"] >= gate_threshold
        )
        rows.append(
            {
                "candidate": candidate.name,
                "scheme": candidate.scheme,
                "combo": candidate.combo,
                **aggregate,
                "accuracy_gate_passed": gate_passed,
            }
        )

    eligible = [row for row in rows if row["accuracy_gate_passed"]]
    max_accuracy = max(row["accuracy"] for row in eligible)
    min_accuracy = min(row["accuracy"] for row in eligible)
    min_ttft = min(row["avg_ttft_ms"] for row in eligible)
    max_throughput = max(row["avg_throughput_tokens_per_sec"] for row in eligible)
    normalization = {
        "max_accuracy": max_accuracy,
        "min_accuracy": min_accuracy,
        "min_avg_ttft_ms": min_ttft,
        "max_avg_throughput_tokens_per_sec": max_throughput,
    }
    for row in rows:
        row["final_score_max_ref"] = (
            final_score_max_ref(row, normalization)
            if row["accuracy_gate_passed"]
            else None
        )

    baseline_reference = (
        {
            **baseline,
            "final_score_max_ref": final_score_max_ref(baseline, normalization),
            "included_in_normalization": False,
            "included_in_ranking": False,
        }
        if baseline
        else None
    )

    rows.sort(
        key=lambda row: (
            row["accuracy_gate_passed"],
            (
                row["final_score_max_ref"]
                if row["final_score_max_ref"] is not None
                else -math.inf
            ),
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
        "accuracy_gate_tolerance": accuracy_gate_tolerance,
        "accuracy_gate_threshold": gate_threshold,
        "baseline_reference": baseline_reference,
        "normalization": {
            **normalization,
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
        f"{'Rank':>4}  {'Candidate':<18} {'Gate':<4} {'Accuracy':>9} "
        f"{'TTFT(ms)':>10} {'Tok/s':>9} {'MaxRef':>9}"
    )
    for row in ranking["ranking"]:
        score = (
            f"{row['final_score_max_ref']:.5f}"
            if row["final_score_max_ref"] is not None
            else "-"
        )
        print(
            f"{str(row['rank'] or '-'):>4}  {row['candidate']:<18} "
            f"{('yes' if row['accuracy_gate_passed'] else 'no'):<4} "
            f"{row['accuracy']:>9.5f} {row['avg_ttft_ms']:>10.2f} "
            f"{row['avg_throughput_tokens_per_sec']:>9.2f} {score:>9}"
        )
    print()


def promoted_candidates(ranking: dict[str, Any], count: int) -> list[Candidate]:
    return [
        Candidate(row["scheme"], row["combo"])
        for row in ranking["ranking"]
        if row["accuracy_gate_passed"]
    ][:count]


def run_stage(
    stage: str,
    candidates: list[Candidate],
    manifest_path: Path,
    repeats: int,
    accuracy_gate_tolerance: float | None,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIR / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(parse_manifest(manifest_path))
    progress = tqdm(candidates, desc=f"{stage} 组合", unit="组")
    for candidate in progress:
        result_path = output_dir / f"{candidate.name}.json"
        progress.set_postfix_str(candidate.name)
        tqdm.write(
            f"\n[{stage}] 开始 {candidate.name}：{sample_count} 条 × {repeats} 次"
        )
        run_combo(stage, candidate, manifest_path, repeats, result_path)
    return rank_stage(
        stage, candidates, manifest_path, accuracy_gate_tolerance
    )


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

    with torch.cuda.device(inputs.device), torch.inference_mode():
        for _ in range(warmup):
            module(inputs)
        torch.cuda.synchronize(inputs.device)
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(inputs)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(inputs.device)
    return [start.elapsed_time(end) for start, end in events]


def run_microbenchmark() -> None:
    _, _, acext_version = ppu_a8w8_api()
    device = DEVICE if DEVICE != "auto" else "cuda"
    shapes = discover_linear_shapes(MODEL_PATH)
    cases = [
        (scheme, shape, tokens)
        for scheme in QUANTIZATION_SCHEMES
        for shape in shapes
        for tokens in ((256, 1024) if shape["group"] == "V" else (1, 128))
    ]

    results = []
    for scheme, shape, tokens in tqdm(
        cases, desc="PPU A8W8 等形状微基准", unit="组"
    ):
        dense = torch.nn.Sequential(
            torch.nn.Linear(
                shape["in_features"],
                shape["out_features"],
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
        ).eval()
        source = torch.nn.Linear(
            shape["in_features"],
            shape["out_features"],
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        source.weight.data.copy_(dense[0].weight.data)
        quantized = torch.nn.Sequential(PPUA8W8Linear(source)).eval()
        path = "vllm_dynamic_per_token_int8_quant+acext_int8_gemm"
        weight_type = str(quantized[0].weight.dtype)
        torch.compiler.reset()
        quantized = torch.compile(
            quantized,
            mode=COMPILE_MODE,
            dynamic=COMPILE_DYNAMIC,
        )
        inputs = torch.randn(
            tokens,
            shape["in_features"],
            device=device,
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
                "scheme": scheme,
                "backend": "vllm.scaled_int8_quant+acext.int8_gemm",
                "tokens": tokens,
                "dense_median_ms": dense_median,
                "dense_p90_ms": percentile(dense_ms, 0.90),
                "quantized_median_ms": quantized_median,
                "quantized_p90_ms": percentile(quantized_ms, 0.90),
                "median_speedup": dense_median / quantized_median,
                "execution_path": path,
                "quantized_weight_type": weight_type,
            }
        )
        del dense, source, quantized, inputs
        torch.cuda.empty_cache()

    torch.compiler.reset()
    save_json(
        OUTPUT_DIR / "microbenchmark.json",
        {
            "schemes": list(QUANTIZATION_SCHEMES),
            "acext_version": acext_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "warmup": MICROBENCH_WARMUP,
            "repeats": MICROBENCH_REPEATS,
            "quantized_compile": {
                "backend": "inductor",
                "mode": COMPILE_MODE,
                "dynamic": COMPILE_DYNAMIC,
            },
            "results": results,
        },
    )
    for scheme in QUANTIZATION_SCHEMES:
        speedups = [
            row["median_speedup"] for row in results if row["scheme"] == scheme
        ]
        tqdm.write(
            f"[microbenchmark] {scheme} 完成 {len(speedups)} 组，speedup="
            f"{min(speedups):.3f}x–{max(speedups):.3f}x"
        )


def main() -> None:
    import torch

    if DEVICE != "auto":
        torch.cuda.set_device(DEVICE)
    manifests = build_manifests()
    ppu_a8w8_api()
    run_microbenchmark()

    ranking64 = run_stage(
        "screen64",
        list(CANDIDATES),
        manifests["screen64"],
        SCREEN64_REPEATS,
        accuracy_gate_tolerance=0.10,
    )

    top6 = promoted_candidates(ranking64, 6)
    ranking256 = run_stage(
        "screen256",
        top6,
        manifests["screen256"],
        SCREEN256_REPEATS,
        accuracy_gate_tolerance=0.02,
    )

    top3 = promoted_candidates(ranking256, 3)
    final_ranking = run_stage(
        "full",
        top3,
        manifests["full"],
        FULL_REPEATS,
        accuracy_gate_tolerance=0.02,
    )
    save_json(
        OUTPUT_DIR / "final_summary.json",
        {
            "screen64_promoted": [asdict(candidate) for candidate in top6],
            "screen256_promoted": [asdict(candidate) for candidate in top3],
            "final_ranking": final_ranking,
        },
    )


if __name__ == "__main__":
    main()
