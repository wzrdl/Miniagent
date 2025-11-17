"""
Policy abstraction that wraps minimind language models for RL training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Sequence

import torch
from torch import nn


class PolicyOutput(NamedTuple):
    """
    Container produced by :class:`BasePolicy.forward`.
    """

    logprobs: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor | None = None
    metadata: Dict[str, Any] | None = None


class BasePolicy(nn.Module, ABC):
    """
    Generic interface for minimind-based policies.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> PolicyOutput:
        """
        Return token-level logits/logprobs/values needed by RL trainers.
        """

    @torch.no_grad()
    def generate_action(
        self,
        messages: Sequence[Dict[str, Any]],
        max_new_tokens: int = 128,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Turn chat messages into an auto-regressive action.

        Concrete subclasses should:
            1. Convert messages via the minimind chat template.
            2. Call the underlying model.generate.
            3. Return both the decoded response and supporting metadata
               (e.g., tool calls, logprobs of generated tokens).
        """

        raise NotImplementedError(
            "generate_action must be implemented by concrete policy wrappers."
        )


__all__ = ["PolicyOutput", "BasePolicy"]

