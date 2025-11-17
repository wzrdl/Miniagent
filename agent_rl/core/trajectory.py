"""
Trajectory data structures shared between rollout workers and trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import random

import torch


@dataclass
class Step:
    obs: Dict[str, Any]
    action: Dict[str, Any]
    logprob: torch.Tensor | None
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    steps: List[Step]
    episode_id: str
    task_id: str
    final_reward: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_reward(self) -> float:
        if self.final_reward is not None:
            return self.final_reward
        return sum(step.reward for step in self.steps)


class TrajectoryBuffer:
    """
    Minimal experience buffer that stores complete episodes.
    """

    def __init__(self) -> None:
        self._episodes: List[Episode] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def extend(self, episodes: Iterable[Episode]) -> None:
        for ep in episodes:
            self.add(ep)

    def pop_all(self) -> List[Episode]:
        episodes = self._episodes
        self._episodes = []
        return episodes

    def sample_batch(self, batch_size: int) -> List[Episode]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._episodes):
            raise ValueError("Not enough episodes to sample from buffer.")
        return random.sample(self._episodes, batch_size)


__all__ = ["Step", "Episode", "TrajectoryBuffer"]

