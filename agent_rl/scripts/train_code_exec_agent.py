"""
Training entry point for the code execution agent using GRPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trajectory import TrajectoryBuffer
from agent_rl.core.trainer_grpo import GRPOTrainer
from agent_rl.envs.code_exec_env import CodeExecEnv
from agent_rl.tools.code_exec_tool import CodeExecTool
from agent_rl.tools.registry import ToolRegistry


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_tool_registry(config: Dict[str, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    tool_cfg = config.get("tools", {}).get("code_exec", {})
    registry.register(
        CodeExecTool(
            python_executable=tool_cfg.get("python_executable", "python"),
            workdir=tool_cfg.get("workdir"),
        )
    )
    return registry


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    # TODO: integrate with minimind model loader.
    policy = None  # Placeholder
    optimizer = None  # Placeholder
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, config=cfg["train"])
    buffer = TrajectoryBuffer()
    tools = build_tool_registry(cfg)
    worker = RolloutWorker(
        policy=policy,
        env_cls=CodeExecEnv,
        env_config=cfg.get("env"),
        tool_manager=tools,
        max_steps=cfg.get("env", {}).get("max_turns", 6),
    )
    del trainer, buffer, worker  # Placeholder to avoid unused variable warnings.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train code exec agent with GRPO.")
    parser.add_argument("--config", default="agent_rl/configs/code_exec_grpo.yaml")
    args = parser.parse_args()
    main(args.config)

