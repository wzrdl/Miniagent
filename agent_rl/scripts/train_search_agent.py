"""
Training entry point for the search QA agent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from dataset import SearchTaskDataset
from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trainer_grpo import GRPOTrainer
from agent_rl.data import CodeTaskSampler
from agent_rl.envs.search_env import SearchEnv
from agent_rl.tools.registry import ToolRegistry
from agent_rl.tools.search_tool import SearchTool
from agent_rl.tools.web_search_tool import WebSearchTool
from agent_rl.core.async_rollout import AsyncRolloutExecutor
from agent_rl.scripts.train_code_exec_agent import (
    build_policy,
    log_preview_tasks,
    collect_episodes,
    run_training_loop,
)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_tool_registry(config: Dict[str, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    search_cfg = config.get("tools", {}).get("search", {})
    mode = search_cfg.get("mode", "web").lower()

    if mode == "corpus":
        corpus = search_cfg.get("corpus", [])
        registry.register(SearchTool(corpus))
        return registry

    api_key = search_cfg.get("api_key")
    api_key_env = search_cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(api_key_env)

    try:
        tool = WebSearchTool(
            provider=search_cfg.get("provider", "tavily"),
            endpoint=search_cfg.get("endpoint"),
            api_key=api_key,
            default_params=search_cfg.get("default_params"),
            timeout=search_cfg.get("timeout", 15.0),
            max_retries=search_cfg.get("max_retries", 2),
            backoff_factor=search_cfg.get("backoff_factor", 0.8),
            rate_limit_per_min=search_cfg.get("rate_limit_per_min"),
            max_snippet_chars=search_cfg.get("max_snippet_chars", 480),
        )
    except ValueError as exc:
        raise RuntimeError(
            "Failed to initialize WebSearchTool. "
            "Double-check `config.tools.search` settings."
        ) from exc
    registry.register(tool)
    return registry


def build_task_samplers(
    dataset_dir: str,
    train_split: str,
    eval_split: str,
    seed: int,
) -> Tuple[CodeTaskSampler, CodeTaskSampler]:
    train_ds = SearchTaskDataset.from_path(Path(dataset_dir) / f"{train_split}.jsonl")
    eval_ds = SearchTaskDataset.from_path(Path(dataset_dir) / f"{eval_split}.jsonl")
    train_sampler = CodeTaskSampler(dataset=train_ds, seed=seed, shuffle=True)
    eval_sampler = CodeTaskSampler(dataset=eval_ds, seed=seed, shuffle=False)
    return train_sampler, eval_sampler


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    policy = build_policy(cfg, args)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["lr"])
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, config=cfg["train"])
    tools = build_tool_registry(cfg)
    train_sampler, _ = build_task_samplers(
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
    )
    if args.preview_tasks > 0:
        log_preview_tasks(train_sampler, args.preview_tasks)

    worker_kwargs = {
        "policy": policy,
        "env_cls": SearchEnv,
        "env_config": cfg.get("env"),
        "tool_manager": tools,
        "max_steps": cfg.get("env", {}).get("max_turns", 4),
    }
    worker = RolloutWorker(**worker_kwargs)
    async_executor = None
    if args.async_rollout_workers > 1:
        worker_factory = lambda: RolloutWorker(**worker_kwargs)
        async_executor = AsyncRolloutExecutor(worker_factory=worker_factory, max_workers=args.async_rollout_workers)

    try:
        run_training_loop(
            trainer=trainer,
            worker=worker,
            task_sampler=train_sampler,
            rollout_cfg=cfg.get("rollout", {}),
            train_cfg=cfg.get("train", {}),
            async_executor=async_executor,
        )
    finally:
        if async_executor:
            async_executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train search QA agent.")
    parser.add_argument("--config", default="agent_rl/configs/search_qa_grpo.yaml")
    parser.add_argument("--dataset-dir", default="dataset/search_tasks")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy-type",
        choices=["minimind", "huggingface"],
        default="minimind",
    )
    parser.add_argument("--hf-model-name", default=None)
    parser.add_argument("--preview-tasks", type=int, default=2)
    parser.add_argument("--async-rollout-workers", type=int, default=1)
    cli_args = parser.parse_args()
    main(cli_args)
