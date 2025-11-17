"""
Training entry point for the search QA agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trajectory import TrajectoryBuffer
from agent_rl.core.trainer_grpo import GRPOTrainer
from agent_rl.envs.search_env import SearchEnv
from agent_rl.tools.registry import ToolRegistry
from agent_rl.tools.search_tool import SearchTool


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_tool_registry(config: Dict[str, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    corpus = config.get("tools", {}).get("search", {}).get("corpus", [])
    registry.register(SearchTool(corpus))
    return registry


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    policy = None  # TODO: load minimind model.
    optimizer = None
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, config=cfg["train"])
    buffer = TrajectoryBuffer()
    tools = build_tool_registry(cfg)
    worker = RolloutWorker(
        policy=policy,
        env_cls=SearchEnv,
        env_config=cfg.get("env"),
        tool_manager=tools,
        max_steps=cfg.get("env", {}).get("max_turns", 4),
    )
    del trainer, buffer, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train search QA agent.")
    parser.add_argument("--config", default="agent_rl/configs/search_qa_grpo.yaml")
    args = parser.parse_args()
    main(args.config)

