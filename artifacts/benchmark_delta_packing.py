#!/usr/bin/env python3
"""Benchmark the three DeltaNet input-projection packing candidates."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]


def time_ms(fn, repetitions: int) -> float:
    for _ in range(5):
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
    model = AutoModelForImageTextToText.from_pretrained(
        ROOT / "Qwen3.5-2B",
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    delta = model.model.language_model.layers[0].linear_attn
    exact = torch.cat(
        (
            delta.in_proj_qkv.weight,
            delta.in_proj_z.weight,
            delta.in_proj_a.weight,
            delta.in_proj_b.weight,
        )
    ).contiguous()
    main = exact[:8192].contiguous()
    tiny = exact[8192:].contiguous()
    weights = {
        "8224": exact,
        "8256": F.pad(exact, (0, 0, 0, 32)),
        "8320": F.pad(exact, (0, 0, 0, 96)),
    }
    rows = {}
    for seq_len, repetitions in ((1, 100), (340, 20)):
        x = torch.randn(1, seq_len, 2048, device=exact.device, dtype=exact.dtype)
        case = {
            name: time_ms(lambda weight=weight: F.linear(x, weight), repetitions)
            for name, weight in weights.items()
        }
        case["8192_plus_32"] = time_ms(
            lambda: (F.linear(x, main), F.linear(x, tiny)), repetitions
        )
        rows[str(seq_len)] = case
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
