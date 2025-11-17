"""
Agent reinforcement learning subsystem built on top of minimind.

This package follows the blueprint described in ``prompt.md`` by combining:

* minimind's model + GRPO training utilities
* veRL-style trajectory / rollout / trainer abstractions
* VerlTool-inspired tool-as-environment design
"""

from __future__ import annotations

__all__ = [
    "core",
    "agent",
    "envs",
    "tools",
    "server",
    "configs",
    "scripts",
]

