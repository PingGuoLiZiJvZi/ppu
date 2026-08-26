#!/usr/bin/env python3
"""Probe 4: small-N GEMV tuning on [2048,2048] + vision grid shape census."""

from __future__ import annotations

import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def time_ms(fn, repetitions: int = 100, warmup: int = 20) -> float:
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


@triton.jit
def _gemv_splitk_kernel(
    x_ptr, w_ptr, y_ptr, N: tl.constexpr, K: tl.constexpr,
    block_n: tl.constexpr, block_k: tl.constexpr, splits: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_n,), tl.float32)
    for kk in range(0, K // splits, block_k):
        idx = pid_k * (K // splits) + kk + tl.arange(0, block_k)
        xv = tl.load(x_ptr + idx).to(tl.float32)
        w = tl.load(w_ptr + n[:, None] * K + idx[None, :]).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.atomic_add(y_ptr + n, acc)


def gemv_2048() -> dict:
    rows = {}
    n_out, k_in = 2048, 2048
    weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02
    x3 = torch.randn(1, 1, k_in, device="cuda", dtype=torch.bfloat16)
    x1 = x3.view(-1)
    rows["flinear_ms"] = round(time_ms(lambda: F.linear(x3, weight)), 5)
    rows["mv_ms"] = round(time_ms(lambda: torch.mv(weight, x1)), 5)

    y = torch.zeros(n_out, device="cuda", dtype=torch.float32)
    for block_n, block_k, splits, warps in (
        (16, 128, 1, 4), (32, 128, 1, 4), (16, 64, 4, 4), (32, 64, 4, 4),
        (16, 64, 8, 4), (8, 128, 8, 4), (16, 128, 4, 4), (32, 128, 4, 8),
        (64, 64, 4, 4),
    ):
        grid = (triton.cdiv(n_out, block_n), splits)
        def run(bn=block_n, bk=block_k, s=splits, w=warps, g=grid):
            y.zero_()
            _gemv_splitk_kernel[g](x1, weight, y, n_out, k_in, bn, bk, s, num_warps=w)
        try:
            rows[f"bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = round(time_ms(run), 5)
        except Exception as exc:  # noqa: BLE001
            rows[f"bn{block_n}_bk{block_k}_s{splits}_w{warps}"] = str(exc)[:50]
    return rows


def vision_shape_census(limit: int = 300) -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        ROOT / "Qwen3.5-2B", local_files_only=True, trust_remote_code=True
    )
    image_processor = processor.image_processor
    cfg = {
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "patch_size": getattr(image_processor, "patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
        "size": getattr(image_processor, "size", None),
    }
    grids = Counter()
    with (ROOT / "datasets/mmbench/mmbench_dev_en.tsv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if index >= limit:
                break
            image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
            w, h = image.size
            factor = getattr(image_processor, "merge_size", 2) or 2
            patch = getattr(image_processor, "patch_size", 16) or 16
            # Qwen smart_resize: clamp to min/max pixels, keep aspect, round to factor*patch multiples.
            max_pixels = cfg["max_pixels"] or (768 * 768)
            min_pixels = cfg["min_pixels"] or (16 * 16)
            if h * w > max_pixels:
                shrink = (max_pixels / (h * w)) ** 0.5
                h, w = int(h * shrink), int(w * shrink)
            elif h * w < min_pixels:
                grow = (min_pixels / (h * w)) ** 0.5
                h, w = int(h * grow), int(w * grow)
            h = max(factor, round(h / (factor * patch)) * factor) * patch // 1
            # emulate: round each side to multiple of patch*merge
            def smart(x):
                x = round(x / (factor * patch)) * factor * patch
                return max(factor * patch, x)
            h_t, w_t = smart(h), smart(w)
            grids[(h_t // patch, w_t // patch, 1)] += 1
    return {"processor_cfg": {k: str(v) for k, v in cfg.items()},
            "distinct_patch_grids_in_first": len(grids),
            "top_grids": [[list(k), v] for k, v in grids.most_common(15)]}


def main() -> None:
    result = {"gemv_2048x2048": gemv_2048(), "vision_census": vision_shape_census()}
    out = ROOT / "artifacts" / "probe_gemv_vision.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
