"""
Training entry point for the code execution agent using GRPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import yaml

from agent_rl.core.minimind_policy import MiniMindPolicy, MiniMindPolicyConfig
from agent_rl.data import CodeTaskSampler, load_code_task_dataset
from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trajectory import Episode
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


def build_task_samplers(
    dataset_dir: str,
    train_split: str,
    eval_split: str,
    seed: int,
) -> Tuple[CodeTaskSampler, CodeTaskSampler]:
    """
    Initialize deterministic task samplers for train/eval splits.
    """

    datasets = load_code_task_dataset(
        dataset_dir,
        train_split=train_split,
        eval_split=eval_split,
        seed=seed,
    )
    # Training sampler uses shuffling for diverse batches; evaluation sampler stays ordered.
    train_sampler = CodeTaskSampler(dataset=datasets[train_split], seed=seed, shuffle=True)
    eval_sampler = CodeTaskSampler(dataset=datasets[eval_split], seed=seed, shuffle=False)
    return train_sampler, eval_sampler


def log_preview_tasks(sampler: CodeTaskSampler, num_tasks: int) -> None:
    """
    Print a small sample of task ids/prompts to verify dataset wiring.
    """

    batch = sampler.next(batch_size=max(1, num_tasks))
    for task in batch:
        task_id = task.get("task_id", "unknown")
        prompt_text = str(task.get("prompt", ""))
        first_line = prompt_text.splitlines()[0] if prompt_text else ""
        print(f"[dataset] task_id={task_id} prompt='{first_line[:80]}...'")


def build_policy(cfg: Dict[str, Any], args: argparse.Namespace) -> MiniMindPolicy:
    """
    Instantiate MiniMind along with its tokenizer/generation parameters.
    """

    model_cfg = cfg.get("model", {})
    policy_cfg = cfg.get("policy", {})
    dtype = _parse_dtype(model_cfg.get("dtype", "float32"))
    policy_config = MiniMindPolicyConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        max_new_tokens=policy_cfg.get("max_new_tokens", 128),
        temperature=policy_cfg.get("temperature", 0.7),
        top_p=policy_cfg.get("top_p", 0.95),
        do_sample=policy_cfg.get("do_sample", True),
    )
    return MiniMindPolicy(policy_config)


def _parse_dtype(name: str) -> torch.dtype:
    """
    Map user-provided dtype strings to ``torch.dtype`` objects.
    """

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return mapping.get(name.lower(), torch.float32)


def main(args: argparse.Namespace) -> None:
    # Load YAML configuration containing model/env/training hyper-parameters.
    cfg = load_config(args.config)
    policy = build_policy(cfg, args)
    # Simple optimizer placeholder; GRPOTrainer will consume it once update() is implemented.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["lr"])
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, config=cfg["train"])
    tools = build_tool_registry(cfg)
    train_sampler, eval_sampler = build_task_samplers(
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
    )
    if args.preview_tasks > 0:
        # Provide a quick sanity check that dataset wiring is correct.
        log_preview_tasks(train_sampler, args.preview_tasks)
    worker = RolloutWorker(
        policy=policy,
        env_cls=CodeExecEnv,
        env_config=cfg.get("env"),
        tool_manager=tools,
        max_steps=cfg.get("env", {}).get("max_turns", 6),
    )
    run_training_loop(
        trainer=trainer,
        worker=worker,
        task_sampler=train_sampler,
        rollout_cfg=cfg.get("rollout", {}),
        train_cfg=cfg.get("train", {}),
    )
def collect_episodes(worker: RolloutWorker, task_batch: List[Dict[str, Any]]) -> List[Episode]:
    """
    Run rollouts for the provided batch of tasks and return finished episodes.
    """

    episodes = []
    for task in task_batch:
        episode, _ = worker.run_episode(task)
        episodes.append(episode)
    return episodes


def run_training_loop(
    trainer: GRPOTrainer,
    worker: RolloutWorker,
    task_sampler: CodeTaskSampler,
    rollout_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    """
    Minimal synchronous training loop: sample tasks → collect episodes → update.
    """

    episodes_per_iter = rollout_cfg.get("num_episodes_per_iter", 4)
    num_iters = train_cfg.get("num_iters", 1)
    for iteration in range(num_iters):
        task_batch = task_sampler.next(batch_size=episodes_per_iter)
        episodes = collect_episodes(worker, task_batch)
        metrics = trainer.update(episodes)
        print(
            f"[train] iter={iteration} "
            f"loss={metrics['loss']:.4f} reward={metrics['mean_reward']:.4f} "
            f"adv={metrics['mean_advantage']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train code exec agent with GRPO.")
    parser.add_argument("--config", default="agent_rl/configs/code_exec_grpo.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--dataset-dir",
        default="dataset/code_tasks",
        help="Directory containing train/eval JSONL splits.",
    )
    parser.add_argument("--train-split", default="train", help="Training split file stem (without extension).")
    parser.add_argument("--eval-split", default="eval", help="Evaluation split file stem.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    parser.add_argument("--model-path", default="model", help="Directory containing MiniMind weights/tokenizer.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy execution (e.g., cpu or cuda:0).")
    parser.add_argument(
        "--preview-tasks",
        type=int,
        default=3,
        help="Print this many sample tasks at startup to confirm dataset wiring.",
    )
    cli_args = parser.parse_args()
    main(cli_args)

