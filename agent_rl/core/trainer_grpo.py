"""
GRPO trainer wrapper compatible with agent trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from torch.optim import Optimizer

from agent_rl.core.policy_base import BasePolicy
from agent_rl.core.trajectory import Episode


@dataclass
class GRPOTrainerConfig:
    learning_rate: float = 1e-5
    grad_clip: float | None = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class GRPOTrainer:
    """
    Thin wrapper that will call into minimind's GRPO utilities.
    """

    def __init__(
        self,
        policy: BasePolicy,
        optimizer: Optimizer,
        config: GRPOTrainerConfig | Dict[str, Any],
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        if isinstance(config, GRPOTrainerConfig):
            self.config = config
        else:
            self.config = GRPOTrainerConfig(**config)

    def update(self, episodes: Iterable[Episode]) -> Dict[str, Any]:
        """
        Convert episode trajectories into token-level training data and
        invoke GRPO loss. This method currently serves as a placeholder
        until the integration with minimind's trainer utilities is wired.
        """

        del episodes
        raise NotImplementedError(
            "GRPOTrainer.update must integrate with minimind trainer stack."
        )


__all__ = ["GRPOTrainer", "GRPOTrainerConfig"]

