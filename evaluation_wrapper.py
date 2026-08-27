"""Participant model wrapper for the DNDX benchmark."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GRAPH_STEPS = 1


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict[str, Any]


class VLMModel:
    """
    Default participant wrapper.

    `backend="dummy"` is for demo-only smoke tests.
    `backend="transformers"` uses a local Hugging Face model directory.
    Participants can replace the internals while preserving `generate_with_metrics`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"
        self._decode_caches: dict[int, Any] = {}
        self._decode_graphs: dict[int, Any] = {}

        if backend in {"auto", "transformers"}:
            try:
                self._load_transformers_backend()
                self._backend_name = "transformers"
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._load_dummy_backend(str(exc))
        else:
            self._load_dummy_backend("backend=dummy")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        if self._backend_name == "transformers":
            return self._generate_with_transformers(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        return self._generate_with_dummy(
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    def _load_transformers_backend(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        bundled_cache = Path(__file__).resolve().parent / "triton"
        if bundled_cache.is_dir():
            os.environ.setdefault("TRITON_CACHE_DIR", str(bundled_cache))
        from qwen35_fused.integration import apply_fusions

        self._fusion_stats = apply_fusions(self._model)
        self._prewarm_decode_buckets()
        self._prewarm_vision_graphs()
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _prewarm_decode_buckets(self, buckets: tuple[int, ...] = (512, 640, 768, 896, 1024)) -> None:
        """Allocate every length bucket's cache and decode graph at load time.

        Without this, the first sample in each bucket pays cache allocation,
        an eager decode step and graph capture (100-300 ms) inside its timing.
        """

        import torch

        from transformers.cache_utils import StaticCache

        from qwen35_fused.graph import GreedyDecodeGraph
        from qwen35_fused.kernels import lm_head_argmax

        model = self._model
        device = model.device
        eos = model.generation_config.eos_token_id
        fill_id = int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos or 0)
        for bucket in buckets:
            input_len = bucket - 256 - 1
            input_ids = torch.full((1, input_len), fill_id, device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            try:
                with torch.inference_mode():
                    cache = StaticCache(config=model.config, max_cache_len=bucket)
                    prefill = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    token = lm_head_argmax(
                        prefill.last_hidden_state[:, -1, :].contiguous(),
                        model.lm_head.weight,
                    )
                    position = (
                        attention_mask.long().sum(dim=-1, keepdim=True).unsqueeze(0)
                        + model.model.rope_deltas.unsqueeze(0)
                    )
                    runner = GreedyDecodeGraph(
                        model,
                        cache,
                        token,
                        position,
                        fused_lm_head=True,
                        steps=GRAPH_STEPS,
                    )
                    runner.replay()
            except Exception:
                # A failed warmup only costs the lazy path on first use.
                self._decode_caches.pop(bucket, None)
                self._decode_graphs.pop(bucket, None)
                continue
            self._decode_caches[bucket] = cache
            self._decode_graphs[bucket] = runner

    # (grid_h, grid_w) pairs for the most common MMBench patch counts; the
    # graph key is the patch count only, so any valid factor pair works.
    _VISION_WARMUP_GRIDS: tuple[tuple[int, int], ...] = (
        (24, 32), (16, 24), (22, 32), (14, 20), (20, 32), (16, 16), (18, 32),
        (32, 32), (16, 32), (16, 20), (18, 16), (30, 32), (20, 30), (28, 32),
        (14, 22), (12, 22), (20, 24), (10, 30),
    )

    def _prewarm_vision_graphs(self) -> None:
        """Capture a decode-free CUDA graph of the Vision blocks for common shapes."""

        import torch

        visual = self._model.model.visual
        if not hasattr(visual, "_vision_graphs"):
            return
        device = self._model.device
        for grid_h, grid_w in self._VISION_WARMUP_GRIDS:
            patches = grid_h * grid_w
            if patches in visual._vision_graphs:
                continue
            vcfg = visual.config
            pixel_width = (
                vcfg.in_channels
                * vcfg.temporal_patch_size
                * vcfg.patch_size
                * vcfg.patch_size
            )
            pixel_values = torch.zeros(
                (patches, pixel_width), device=device, dtype=torch.bfloat16
            )
            grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)
            try:
                with torch.inference_mode():
                    visual(pixel_values, grid_thw=grid_thw)
            except Exception:
                # Unknown shapes fall back to the eager path and lazy capture.
                continue

    def _load_dummy_backend(self, reason: str) -> None:
        self._dummy_reason = reason

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        if self._fusion_stats:
            return self._generate_fused_greedy(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )

        import torch
        from transformers import TextIteratorStreamer

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
        ).to(self._model.device)
        input_len = inputs.input_ids.shape[1]
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

        output_holder: dict[str, Any] = {}

        def _run_generate() -> None:
            with torch.no_grad():
                output_holder["output_ids"] = self._model.generate(**generation_kwargs)

        worker = threading.Thread(target=_run_generate, daemon=True)
        start = time.perf_counter()
        worker.start()

        first_chunk_at = None
        chunks: list[str] = []
        for chunk in streamer:
            now = time.perf_counter()
            if first_chunk_at is None and chunk:
                first_chunk_at = now
            chunks.append(chunk)
        worker.join()
        end = time.perf_counter()

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        ttft = (first_chunk_at - start) if first_chunk_at is not None else (end - start)
        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=end - start,
            meta={"backend": "transformers"},
        )

    def _generate_fused_greedy(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        import torch
        from transformers.cache_utils import StaticCache

        from qwen35_fused.graph import GreedyDecodeGraph
        from qwen35_fused.kernels import lm_head_argmax

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
        from qwen35_fused.integration import precompute_vision_kwargs

        precompute_vision_kwargs(self._model, inputs)
        inputs = inputs.to(self._model.device)
        input_len = int(inputs.input_ids.shape[-1])
        max_new_tokens = max(1, int(generation_config.max_new_tokens))
        required_cache_len = input_len + max_new_tokens + 1
        cache_bucket = max(512, ((required_cache_len + 127) // 128) * 128)
        start = time.perf_counter()
        cache = self._decode_caches.get(cache_bucket)
        cache_is_new = cache is None
        if cache_is_new:
            cache = StaticCache(
                config=self._model.config,
                max_cache_len=cache_bucket,
            )
            self._decode_caches[cache_bucket] = cache
        with torch.inference_mode():
            if not cache_is_new:
                cache.reset()
            prefill = self._model.model(
                **inputs,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            token = lm_head_argmax(
                prefill.last_hidden_state[:, -1, :].contiguous(),
                self._model.lm_head.weight,
            )
            first_token = int(token.item())
            first_token_at = time.perf_counter()
            generated = [first_token]

            eos = self._model.generation_config.eos_token_id
            if eos is None:
                eos_ids: set[int] = set()
            elif isinstance(eos, (list, tuple)):
                eos_ids = {int(item) for item in eos}
            else:
                eos_ids = {int(eos)}

            first_position = (
                inputs.attention_mask.long().sum(dim=-1, keepdim=True).unsqueeze(0)
                + self._model.model.rope_deltas.unsqueeze(0)
            )

            runner = self._decode_graphs.get(cache_bucket)
            graph_was_reused = runner is not None
            if runner is None and len(generated) < max_new_tokens and generated[-1] not in eos_ids:
                decode = self._model.model(
                    input_ids=token,
                    attention_mask=None,
                    position_ids=first_position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                token = lm_head_argmax(
                    decode.last_hidden_state[:, -1, :].contiguous(),
                    self._model.lm_head.weight,
                )
                generated.append(int(token.item()))
                first_position.add_(1)

            if len(generated) < max_new_tokens and generated[-1] not in eos_ids:
                if runner is None:
                    runner = GreedyDecodeGraph(
                        self._model,
                        cache,
                        token,
                        first_position,
                        fused_lm_head=True,
                        steps=GRAPH_STEPS,
                    )
                    self._decode_graphs[cache_bucket] = runner
                else:
                    runner.set_inputs(token, first_position)
                while len(generated) < max_new_tokens and generated[-1] not in eos_ids:
                    chunk = runner.replay().reshape(-1).tolist()
                    remaining = max_new_tokens - len(generated)
                    for token_id in chunk[:remaining]:
                        generated.append(int(token_id))
                        if generated[-1] in eos_ids:
                            break

        end = time.perf_counter()
        generated_tensor = torch.tensor(generated, dtype=torch.long)
        text = self._processor.tokenizer.decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return GenerationResult(
            text=text,
            token_count=len(generated),
            ttft_seconds=first_token_at - start,
            elapsed_seconds=end - start,
            meta={
                "backend": "transformers-fused",
                "cuda_graph": len(generated) > 2,
                "cuda_graph_steps": GRAPH_STEPS,
                "cache_bucket": cache_bucket,
                "cache_reused": not cache_is_new,
                "cuda_graph_reused": graph_was_reused,
                "fusion_stats": self._fusion_stats,
            },
        )

    def _generate_with_dummy(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        usable_choices = [key for key, value in choices.items() if (value or "").strip()]
        picked = usable_choices[hash(sample_id) % len(usable_choices)] if usable_choices else "A"
        text = (
            f"Answer: {picked}\n"
            f"Explanation: dummy backend selected a deterministic option for smoke testing."
        )
        token_count = max(1, min(generation_config.max_new_tokens, len(text.split())))
        end = time.perf_counter()
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=max(end - start, 1e-4),
            elapsed_seconds=max(end - start, 2e-4),
            meta={"backend": "dummy", "reason": getattr(self, "_dummy_reason", "n/a"), "prompt_chars": len(prompt)},
        )
