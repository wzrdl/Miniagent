"""
Registry for tool instances.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from agent_rl.tools.base_tool import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def extend(self, tools: Iterable[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)


__all__ = ["ToolRegistry"]

