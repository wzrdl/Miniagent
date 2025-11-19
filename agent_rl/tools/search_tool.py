"""
Offline fallback search tool that emulates document retrieval.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from agent_rl.tools.base_tool import BaseTool


class SearchTool(BaseTool):
    """
    Simple in-memory search tool that scans a corpus for case-insensitive matches.
    """

    name = "search"
    description = "Retrieve snippets from a provided corpus."

    def __init__(self, corpus: Iterable[str] | None = None) -> None:
        self.corpus: List[str] = list(corpus or [])

    def set_corpus(self, corpus: Iterable[str]) -> None:
        """
        Update the searchable corpus at runtime.
        """

        self.corpus = list(corpus)

    def __call__(self, query: str, top_k: int = 3, **_: Any) -> str:
        """
        Return up to ``top_k`` documents that contain the query substring.
        """

        if not query:
            return "Empty query provided."
        matches = [doc for doc in self.corpus if query.lower() in doc.lower()]
        if not matches:
            return "No documents matched."
        return "\n---\n".join(matches[:top_k])


__all__ = ["SearchTool"]

