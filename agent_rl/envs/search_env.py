"""
搜索问答环境，支持工具调用与多轮推理。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.envs.base_env import AgentEnv
from agent_rl.tools.registry import ToolRegistry


class SearchEnv(AgentEnv):
    """
    允许代理调用 ``search`` 工具并多轮推理的问答环境。
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
        self.max_turns = self.config.get("max_turns", 4)
        self.answer = str(self.task_sample.get("answer", "")).strip()
        self._corpus = self.task_sample.get("documents", [])
        self._latest_tool_info: Dict[str, Any] | None = None
        self.search_history: List[Dict[str, Any]] = []
        self._turn_count = 0

    def reset(self) -> Dict[str, Any]:
        question = self.task_sample.get("question", "Please answer the question.")
        system_prompt = self.config.get(
            "system_prompt",
            "You are a research assistant. Use the search tool when needed before finalizing an answer.",
        )
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=question),
        ]
        self._latest_tool_info = None
        self.search_history = []
        self._turn_count = 0
        self._configure_search_tool()
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:

        assistant_content = action.get("content", "")
        tool_call_str = action.get("tool_call")
        self.messages.append(Message(role="assistant", content=assistant_content))
        self._turn_count += 1

        reward = 0.0
        done = False
        info: Dict[str, Any] = {"search_history": self.search_history}

        if tool_call_str:
            if not self.tool_manager:
                raise RuntimeError("Tool manager is required for search env when using tool calls.")
            tool_call = self.tool_parser.parse(tool_call_str)
            tool = self.tool_manager.get(tool_call.name)
            tool_result = tool(**tool_call.arguments)
            tool_content = tool_result if isinstance(tool_result, str) else str(tool_result)
            self.messages.append(Message(role="tool", content=tool_content, tool_name=tool_call.name))
            tool_entry = {"tool": tool_call.name, "arguments": tool_call.arguments, "content": tool_content}
            self.search_history.append(tool_entry)
            info["tool_result"] = tool_entry
            self._latest_tool_info = tool_entry
        else:
            if self._answer_matches(assistant_content):
                reward = self._success_reward()
                done = True

        if not done and self._turn_count >= self.max_turns:
            done = True
            reward = reward or self.config.get("timeout_penalty", 0.0)
            info["terminated_reason"] = "max_turns"

        return {"messages": self.messages}, reward, done, info

    def compute_final_reward(self) -> float | None:
        last_response = self._last_assistant_message()
        if last_response and self._answer_matches(last_response):
            return self._success_reward()
        return 0.0

    def _configure_search_tool(self) -> None:
        if not self.tool_manager:
            return
        try:
            search_tool = self.tool_manager.get("search")
        except KeyError:
            return
        if hasattr(search_tool, "set_corpus"):
            search_tool.set_corpus(self._corpus or self.config.get("default_corpus", []))

    def _answer_matches(self, text: str) -> bool:
        if not self.answer:
            return False
        return self.answer.lower() in text.lower()

    def _success_reward(self) -> float:
        return self.config.get("success_reward", 1.0)

    def _last_assistant_message(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.content
        return None


__all__ = ["SearchEnv"]

