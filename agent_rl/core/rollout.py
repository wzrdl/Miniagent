"""
Synchronous rollout workers that collect trajectories from environments.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Type

from agent_rl.core.policy_base import BasePolicy
from agent_rl.core.trajectory import Episode, Step
from agent_rl.envs.base_env import AgentEnv


@dataclass
class RolloutStats:
    episode_id: str
    task_id: str
    num_steps: int
    latency_s: float
    final_reward: float | None


class RolloutWorker:
    """
    Simple synchronous rollout worker.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env_cls: Type[AgentEnv],
        env_config: Dict[str, Any] | None,
        tool_manager: Any,
        max_steps: int = 8,
    ) -> None:
        self.policy = policy
        self.env_cls = env_cls
        self.env_config = env_config or {}
        self.tool_manager = tool_manager
        self.max_steps = max_steps

    def run_episode(self, task_sample: Dict[str, Any]) -> tuple[Episode, RolloutStats]:
        env = self.env_cls(task_sample, self.env_config, self.tool_manager)
        obs = env.reset()
        steps: list[Step] = []
        episode_id = str(uuid.uuid4())
        start = time.time()

        for _ in range(self.max_steps):
            action = self._sample_action(obs)
            next_obs, reward, done, env_info = env.step(action)
            steps.append(
                Step(
                    obs=obs,
                    action=action,
                    logprob=action.get("metadata", {}).get("response_logprobs"),
                    reward=reward,
                    done=done,
                    info={"env": env_info, "policy": action.get("metadata", {})},
                )
            )
            obs = next_obs
            if done:
                break

        episode = Episode(
            steps=steps,
            episode_id=episode_id,
            task_id=str(task_sample.get("task_id", "unknown")),
        )
        episode.final_reward = env.compute_final_reward()
        latency = time.time() - start

        stats = RolloutStats(
            episode_id=episode_id,
            task_id=episode.task_id,
            num_steps=len(steps),
            latency_s=latency,
            final_reward=episode.final_reward,
        )
        return episode, stats

    def _sample_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default action sampler that delegates to ``policy.generate_action``.
        """

        messages = obs.get("messages")
        if messages is None:
            raise ValueError(
                "Observation must include 'messages' when using default sampler."
            )
        return self.policy.generate_action(messages=messages)


__all__ = ["RolloutWorker", "RolloutStats"]

