"""
Environment for code-execution tasks that leverage a sandbox tool.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.envs.base_env import AgentEnv
from agent_rl.tools.registry import ToolRegistry


class CodeExecEnv(AgentEnv):
    """
    Multi-turn environment where the agent can invoke an ``exec_code`` tool.
    """

    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: ToolRegistry | None,
    ) -> None:
        super().__init__(task_sample, config, tool_manager)
        self.messages: List[Message] = []
        self.tool_parser = ToolParser()
        self.max_turns = self.config.get("max_turns", 6)

    def reset(self) -> Dict[str, Any]:
        system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful coding assistant that can execute Python code via tools.",
        )
        user_prompt = self.task_sample.get("prompt", "Provide a coding task.")
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assistant_content = action.get("content", "")
        tool_call_str = action.get("tool_call")

        self.messages.append(Message(role="assistant", content=assistant_content))

        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        if tool_call_str:
            if not self.tool_manager:
                raise RuntimeError("Tool manager is required for code execution env.")
            tool_call = self.tool_parser.parse(tool_call_str)
            tool = self.tool_manager.get(tool_call.name)
            tool_result = tool(**tool_call.arguments)
            self.messages.append(
                Message(role="tool", content=tool_result, tool_name=tool_call.name)
            )
            info["tool_result"] = tool_result

        if self._task_solved():
            reward = 1.0
            done = True
        elif len(self.messages) // 2 >= self.max_turns:
            done = True
            reward = self.config.get("timeout_penalty", 0.0)

        return {"messages": self.messages}, reward, done, info

    def compute_final_reward(self) -> float | None:
        if self._task_solved():
            return 1.0
        return 0.0

    def _task_solved(self) -> bool:
        reference = self.task_sample.get("reference_solution")
        if not reference:
            return False
        return reference in self._last_tool_output()

    def _last_tool_output(self) -> str:
        for message in reversed(self.messages):
            if message.role == "tool":
                return message.content
        return ""


__all__ = ["CodeExecEnv"]

