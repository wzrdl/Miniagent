"""
Placeholder search tool that emulates document retrieval.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent_rl.tools.base_tool import BaseTool


class SearchTool(BaseTool):
    name = "search"
    description = "Retrieve snippets from a provided corpus."

    def __init__(self, corpus: Iterable[str] | None = None) -> None:
        self.corpus = list(corpus or [])

    def __call__(self, query: str, top_k: int = 3, **_: Any) -> str:
        matches = [doc for doc in self.corpus if query.lower() in doc.lower()]
        if not matches:
            return "No documents matched."
        return "\n---\n".join(matches[:top_k])


__all__ = ["SearchTool"]

