#!/usr/bin/env python3
"""Benchmark BF16-exact fused LM-head top-1 against the PPU GEMV path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen35_fused.kernels import lm_head_argmax


def time_ms(fn, repetitions: int = 20) -> float:
    for _ in range(3):
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
    torch.manual_seed(20260819)
    model = AutoModelForImageTextToText.from_pretrained(
        ROOT / "Qwen3.5-2B",
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    weight = model.lm_head.weight
    hidden = torch.randn(1, weight.shape[1], device=weight.device, dtype=torch.bfloat16)
    reference = F.linear(hidden, weight).argmax(dim=-1, keepdim=True)
    rows = {}
    for block_n in (8, 16, 32):
        actual = lm_head_argmax(hidden, weight, block_n=block_n)
        rows[str(block_n)] = {
            "token": int(actual.item()),
            "exact": bool(torch.equal(actual, reference)),
            "ms": time_ms(lambda block_n=block_n: lm_head_argmax(hidden, weight, block_n=block_n)),
        }
    result = {
        "reference_token": int(reference.item()),
        "eager_ms": time_ms(lambda: F.linear(hidden, weight).argmax(dim=-1, keepdim=True)),
        "fused": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
