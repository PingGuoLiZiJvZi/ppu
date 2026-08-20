"""CUDA Graph runner for a single-token greedy Qwen3.5 decode step."""

from __future__ import annotations

import torch

from .kernels import lm_head_argmax


class GreedyDecodeGraph:
    """Capture model decode, LM head argmax and autoregressive token feedback.

    The supplied cache must already contain the prefill state and must use
    static full-attention layers. ``position_ids`` is the first decode token's
    multimodal position tensor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache,
        token: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        fused_lm_head: bool = True,
        steps: int = 1,
    ) -> None:
        if token.shape[-1] != 1 or token.dtype != torch.long or not token.is_cuda:
            raise ValueError("token must be a CUDA int64 tensor with sequence length 1")
        if not position_ids.is_cuda:
            raise ValueError("position_ids must be on the PPU")
        if steps < 1:
            raise ValueError("steps must be positive")
        self.model = model
        self.cache = cache
        self.token = token.clone()
        self.position_ids = position_ids.clone()
        self.graph = torch.cuda.CUDAGraph()
        self.logits = None
        self.fused_lm_head = fused_lm_head
        self.steps = steps
        self.output_tokens = None
        if steps > 1:
            self.output_tokens = torch.empty(
                (steps, *self.token.shape),
                device=self.token.device,
                dtype=self.token.dtype,
            )

        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            for step in range(self.steps):
                if self.fused_lm_head:
                    outputs = self.model.model(
                        input_ids=self.token,
                        attention_mask=None,
                        position_ids=self.position_ids,
                        past_key_values=self.cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    hidden = outputs.last_hidden_state[:, -1, :].contiguous()
                    next_token = lm_head_argmax(hidden, self.model.lm_head.weight)
                else:
                    outputs = self.model(
                        input_ids=self.token,
                        attention_mask=None,
                        position_ids=self.position_ids,
                        past_key_values=self.cache,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    self.logits = outputs.logits
                    next_token = self.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if self.output_tokens is not None:
                    self.output_tokens[step].copy_(next_token)
                self.token.copy_(next_token)
                self.position_ids.add_(1)

    def set_inputs(self, token: torch.Tensor, position_ids: torch.Tensor) -> None:
        """Set the first token and position for the next graph replay."""

        if token.shape != self.token.shape or position_ids.shape != self.position_ids.shape:
            raise ValueError("graph input shapes do not match the captured buffers")
        self.token.copy_(token)
        self.position_ids.copy_(position_ids)

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        if self.steps == 1:
            return self.token
        assert self.output_tokens is not None
        return self.output_tokens
