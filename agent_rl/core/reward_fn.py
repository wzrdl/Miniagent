"""
Reusable reward functions for different tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class RewardFunction(ABC):
    name: str = "base"

    @abstractmethod
    def __call__(self, episode_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Return (reward, info) for a completed episode.
        """


class RewardRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, RewardFunction] = {}

    def register(self, reward_fn: RewardFunction) -> None:
        self._registry[reward_fn.name] = reward_fn

    def get(self, name: str) -> RewardFunction:
        return self._registry[name]


__all__ = ["RewardFunction", "RewardRegistry"]

