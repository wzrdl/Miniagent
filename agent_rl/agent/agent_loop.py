"""
Main agent loop that orchestrates message passing and tool calls.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.chat_template import ChatTemplate
from agent_rl.agent.messages import Message, ToolCall
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.core.policy_base import BasePolicy
from agent_rl.tools.registry import ToolRegistry


class ToolCallingAgent:
    def __init__(
        self,
        policy: BasePolicy,
        tool_registry: ToolRegistry,
        max_turns: int = 6,
    ) -> None:
        self.policy = policy
        self.tool_registry = tool_registry
        self.max_turns = max_turns
        self.chat_template = ChatTemplate()
        self.tool_parser = ToolParser()

    def run_episode(self, messages: List[Message]) -> Dict[str, Any]:
        tool_calls: List[ToolCall] = []
        for _ in range(self.max_turns):
            action = self._one_turn(messages)
            messages.append(Message(role="assistant", content=action["content"]))

            tool_call_text = action.get("tool_call")
            if tool_call_text:
                tool_call = self.tool_parser.parse(tool_call_text)
                tool_calls.append(tool_call)
                tool = self.tool_registry.get(tool_call.name)
                tool_output = tool(**tool_call.arguments)
                messages.append(
                    Message(role="tool", content=tool_output, tool_name=tool_call.name)
                )
            if action.get("done"):
                break

        return {"messages": messages, "tool_calls": tool_calls}

    def _one_turn(self, messages: List[Message]) -> Dict[str, Any]:
        serialized = self.chat_template.encode(messages)
        structured_messages = [
            {"role": m.role, "content": m.content, "tool_name": m.tool_name}
            for m in messages
        ]
        action = self.policy.generate_action(
            messages=structured_messages, prompt=serialized
        )
        return action


__all__ = ["ToolCallingAgent"]

