"""
Simple placeholder environment for search-based QA tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.messages import Message
from agent_rl.envs.base_env import AgentEnv


class SearchEnv(AgentEnv):
    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: Any | None,
    ) -> None:
        super().__init__(task_sample, config, tool_manager)
        self.messages: List[Message] = []

    def reset(self) -> Dict[str, Any]:
        question = self.task_sample.get("question", "What is the answer?")
        system_prompt = self.config.get(
            "system_prompt",
            "You answer questions and may use search tools.",
        )
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=question),
        ]
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.messages.append(Message(role="assistant", content=action.get("content", "")))
        done = True
        reward = 1.0 if self._answer_matches() else 0.0
        return {"messages": self.messages}, reward, done, {}

    def _answer_matches(self) -> bool:
        target = self.task_sample.get("answer")
        if not target:
            return False
        return target.lower() in self.messages[-1].content.lower()


__all__ = ["SearchEnv"]

