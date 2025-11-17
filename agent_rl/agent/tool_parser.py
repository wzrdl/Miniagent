"""
Parse tool-call markup emitted by the policy.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_rl.agent.messages import ToolCall


class ToolParser:
    TOOL_PATTERN = re.compile(r"<tool_call>(?P<payload>.+?)</tool_call>", re.DOTALL)

    def parse(self, text: str) -> ToolCall:
        match = self.TOOL_PATTERN.search(text)
        if not match:
            raise ValueError("No <tool_call> payload found in action output.")
        payload = match.group("payload")
        data = json.loads(payload)
        return ToolCall(name=data["name"], arguments=data.get("arguments", {}))


__all__ = ["ToolParser"]

