"""
Base class for agent RL environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentEnv(ABC):
    """
    Env API inspired by OpenAI Gym but tailored for tool-augmented agents.
    """

    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: Any | None = None,
    ) -> None:
        self.task_sample = task_sample
        self.config = config or {}
        self.tool_manager = tool_manager

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Initialize state and return the first observation dict.
        """

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply the agent action and return (next_obs, reward, done, info).
        """

    def compute_final_reward(self) -> float | None:
        """
        Optional hook that can be overridden when rewards are only computed
        at the end of an episode.
        """

        return None


__all__ = ["AgentEnv"]

