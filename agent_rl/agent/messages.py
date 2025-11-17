"""
Common message dataclasses used across agent modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_name: Optional[str] = None


@dataclass
class ToolCall:
    name: str
    arguments: dict


__all__ = ["Message", "ToolCall", "Role"]

