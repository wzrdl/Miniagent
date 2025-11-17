"""
Lightweight wrapper around minimind chat templates with tool-call support.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from agent_rl.agent.messages import Message


class ChatTemplate:
    """
    Serializes chat messages into the text format expected by minimind models.
    """

    def __init__(self, tool_tag: str = "tool_call") -> None:
        self.tool_tag = tool_tag

    def encode(self, messages: Iterable[Message]) -> str:
        parts: List[str] = []
        for message in messages:
            if message.role == "tool":
                parts.append(f"<tool name='{message.tool_name}'>{message.content}</tool>")
            else:
                parts.append(f"<{message.role}>{message.content}</{message.role}>")
        return "\n".join(parts)

    def build_tool_call(self, name: str, arguments: dict) -> str:
        return f"<{self.tool_tag}>{name}:{arguments}</{self.tool_tag}>"


__all__ = ["ChatTemplate"]

