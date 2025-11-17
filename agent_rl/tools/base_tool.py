"""
Base tool definition used by agents during rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base tool."

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the tool and return a text response inserted into the chat.
        """


__all__ = ["BaseTool"]

