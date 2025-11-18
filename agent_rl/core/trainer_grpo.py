"""
GRPO trainer wrapper compatible with agent trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from collections import defaultdict

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
    norm_adv_by_std: bool = True


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
        self._device = next(self.policy.parameters()).device

    def update(self, episodes: Iterable[Episode]) -> Dict[str, Any]:
        """
        Convert episode trajectories into token-level training data and
        invoke GRPO loss. This method currently serves as a placeholder
        until the integration with minimind's trainer utilities is wired.
        """

        batch = list(episodes)
        if not batch:
            return {"loss": 0.0, "mean_reward": 0.0}

        rewards = torch.tensor([self._episode_reward(ep) for ep in batch], device=self._device)
        task_ids = [ep.task_id for ep in batch]
        advantages = self._compute_advantages(rewards, task_ids)

        logprob_sums = []
        for episode in batch:
            policy_meta = episode.steps[-1].info.get("policy", {})
            prompt_text = policy_meta.get("prompt_text")
            response_ids = policy_meta.get("response_ids")
            if prompt_text is None or response_ids is None:
                raise ValueError(
                    "Episode is missing policy metadata required for GRPO update. "
                    "Ensure MiniMindPolicy.generate_action returns response ids."
                )
            response_tensor = torch.as_tensor(response_ids, dtype=torch.long, device=self._device)
            token_logprobs = self.policy.compute_sequence_logprob(prompt_text, response_tensor)
            logprob_sums.append(token_logprobs.sum())

        logprob_tensor = torch.stack(logprob_sums)
        loss = -(advantages * logprob_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "mean_advantage": float(advantages.mean().detach().cpu()),
        }

    def _episode_reward(self, episode: Episode) -> float:
        if episode.final_reward is not None:
            return episode.final_reward
        return episode.total_reward()

    def _compute_advantages(self, rewards: torch.Tensor, task_ids: List[str]) -> torch.Tensor:
        """
        Compute GRPO-style outcome advantages grouped by task id.
        """

        grouped: Dict[str, List[float]] = defaultdict(list)
        for reward, task_id in zip(rewards.tolist(), task_ids):
            grouped[task_id].append(reward)

        advantages = torch.zeros_like(rewards)
        for idx, (reward, task_id) in enumerate(zip(rewards, task_ids)):
            group = torch.tensor(grouped[task_id], device=self._device)
            mean = group.mean()
            if self.config.norm_adv_by_std and group.numel() > 1:
                std = group.std(unbiased=False)
                advantages[idx] = (reward - mean) / (std + 1e-6)
            else:
                advantages[idx] = reward - mean
        return advantages


__all__ = ["GRPOTrainer", "GRPOTrainerConfig"]

