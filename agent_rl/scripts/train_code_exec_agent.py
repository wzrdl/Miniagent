"""
使用 GRPO (Group Relative Policy Optimization) 训练代码执行代理的训练入口脚本。

该脚本实现了完整的训练流程：
1. 加载配置文件（模型、环境、训练超参数等）
2. 构建策略网络（MiniMind 模型）
3. 初始化工具注册表（代码执行工具）
4. 构建任务采样器（训练集和验证集）
5. 创建训练器和环境工作器
6. 运行训练循环（采样任务 → 收集轨迹 → 更新策略）

主要组件：
- MiniMindPolicy: 基于 MiniMind 模型的策略网络
- CodeExecEnv: 代码执行环境，用于模拟代码执行任务
- GRPOTrainer: GRPO 训练器，负责策略优化
- RolloutWorker: 轨迹收集工作器，用于在环境中运行策略并收集经验
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import yaml

from agent_rl.core.policy_base import BasePolicy
from agent_rl.core.minimind_policy import MiniMindPolicy, MiniMindPolicyConfig
from agent_rl.core.hf_policy import HuggingFaceCausalPolicy, HFCausalPolicyConfig
from agent_rl.core.async_rollout import AsyncRolloutExecutor
from agent_rl.data import CodeTaskSampler, load_code_task_dataset
from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trajectory import Episode
from agent_rl.core.trainer_grpo import GRPOTrainer
from agent_rl.envs.code_exec_env import CodeExecEnv
from agent_rl.tools.code_exec_tool import CodeExecTool
from agent_rl.tools.registry import ToolRegistry


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    从 YAML 配置文件中加载配置信息。
    
    配置文件通常包含以下部分：
    - model: 模型相关配置（数据类型、路径等）
    - policy: 策略生成参数（最大token数、温度、top_p等）
    - env: 环境配置（最大回合数等）
    - train: 训练超参数（学习率、迭代次数等）
    - rollout: 轨迹收集配置（每次迭代的episode数量等）
    - tools: 工具配置（代码执行工具的配置）
    
    Args:
        path: YAML 配置文件的路径（字符串或 Path 对象）
    
    Returns:
        包含所有配置信息的字典
    
    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果 YAML 文件格式错误
    """
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_tool_registry(config: Dict[str, Any]) -> ToolRegistry:
    """
    根据配置构建工具注册表，注册代码执行工具。
    
    工具注册表用于管理代理可以使用的各种工具。当前实现中主要注册了
    CodeExecTool，该工具允许代理执行 Python 代码并获取执行结果。
    
    Args:
        config: 配置字典，应包含 "tools" -> "code_exec" 的配置项
            - python_executable: Python 解释器路径（默认: "python"）
            - workdir: 代码执行的工作目录
    
    Returns:
        已注册代码执行工具的 ToolRegistry 实例
    
    Note:
        未来可以扩展以支持更多工具类型（如文件操作、网络请求等）
    """
    registry = ToolRegistry()
    # 从配置中获取代码执行工具的配置，如果不存在则使用空字典
    tool_cfg = config.get("tools", {}).get("code_exec", {})
    # 创建代码执行工具并注册到注册表中
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
    为训练集和验证集构建确定性的任务采样器。
    
    该函数从指定目录加载代码任务数据集，并创建两个采样器：
    - 训练采样器：使用随机打乱，确保每个批次的任务多样性
    - 验证采样器：保持顺序，确保评估结果的可重复性
    
    Args:
        dataset_dir: 数据集目录路径，应包含训练和验证的 JSONL 文件
        train_split: 训练集文件名（不含扩展名），例如 "train"
        eval_split: 验证集文件名（不含扩展名），例如 "eval"
        seed: 随机种子，用于确保数据加载和采样的可重复性
    
    Returns:
        包含两个 CodeTaskSampler 的元组：
        - 第一个是训练采样器（shuffle=True）
        - 第二个是验证采样器（shuffle=False）
    
    Note:
        数据集文件应为 JSONL 格式，每行包含一个任务，至少包含 "task_id" 和 "prompt" 字段
    """
    # 从数据集目录加载训练集和验证集
    datasets = load_code_task_dataset(
        dataset_dir,
        train_split=train_split,  
        eval_split=eval_split,
        seed=seed,
    )
    # 训练采样器使用随机打乱，增加批次多样性，有助于模型泛化
    train_sampler = CodeTaskSampler(dataset=datasets[train_split], seed=seed, shuffle=True)
    # 验证采样器保持顺序，确保评估结果可重复且一致
    eval_sampler = CodeTaskSampler(dataset=datasets[eval_split], seed=seed, shuffle=False)
    return train_sampler, eval_sampler


def log_preview_tasks(sampler: CodeTaskSampler, num_tasks: int) -> None:
    """
    打印少量任务样本，用于验证数据集连接是否正确。
    
    该函数主要用于调试和验证目的，在训练开始前快速检查：
    - 数据集是否正确加载
    - 任务格式是否符合预期
    - 采样器是否正常工作
    
    Args:
        sampler: 任务采样器，用于获取任务批次
        num_tasks: 要预览的任务数量（至少为 1）
    
    Note:
        输出格式：`[dataset] task_id=<id> prompt='<prompt前80字符>...'`
        如果任务没有 task_id 或 prompt 字段，将使用默认值
    """
    # 从采样器获取指定数量的任务（至少1个）
    batch = sampler.next(batch_size=max(1, num_tasks))
    # 遍历每个任务，打印其 ID 和提示的前80个字符
    for task in batch:
        task_id = task.get("task_id", "unknown")
        prompt_text = str(task.get("prompt", ""))
        # 提取提示的第一行，如果为空则使用空字符串
        first_line = prompt_text.splitlines()[0] if prompt_text else ""
        # 打印任务信息，限制提示长度为80字符以便阅读
        print(f"[dataset] task_id={task_id} prompt='{first_line[:80]}...'")


def build_policy(cfg: Dict[str, Any], args: argparse.Namespace) -> BasePolicy:
    """
    根据配置和命令行参数构建策略网络（MiniMind 或 HuggingFace）。
    """

    model_cfg = cfg.get("model", {})
    policy_cfg = cfg.get("policy", {})
    dtype = _parse_dtype(model_cfg.get("dtype", "float32"))

    if args.policy_type == "huggingface":
        model_name = args.hf_model_name or model_cfg.get("name")
        if not model_name:
            raise ValueError("使用 HuggingFace 策略时需要指定 --hf-model-name 或在配置中设置 model.name。")
        hf_config = HFCausalPolicyConfig(
            model_name_or_path=model_name,
            device=args.device,
            dtype=dtype,
            max_new_tokens=policy_cfg.get("max_new_tokens", 128),
            temperature=policy_cfg.get("temperature", 0.7),
            top_p=policy_cfg.get("top_p", 0.95),
            do_sample=policy_cfg.get("do_sample", True),
        )
        return HuggingFaceCausalPolicy(hf_config)

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
    将用户提供的数据类型字符串映射为 torch.dtype 对象。
    
    该函数支持多种常见的数据类型表示方式，包括完整名称和缩写形式。
    如果提供的字符串不在映射表中，则默认返回 float32。
    
    Args:
        name: 数据类型字符串，支持以下值（不区分大小写）：
            - "float32" 或 "fp32": 32位浮点数
            - "bfloat16" 或 "bf16": 16位脑浮点数（常用于训练）
            - "float16" 或 "fp16": 16位半精度浮点数
    
    Returns:
        对应的 torch.dtype 对象，如果未匹配则返回 torch.float32
    
    Examples:
        >>> _parse_dtype("bf16")
        torch.bfloat16
        >>> _parse_dtype("FP32")
        torch.float32
        >>> _parse_dtype("unknown")
        torch.float32  # 默认值
    """
    # 数据类型字符串到 torch.dtype 的映射表
    mapping = {
        "float32": torch.float32,  # 标准32位浮点数
        "fp32": torch.float32,  # float32 的缩写
        "bfloat16": torch.bfloat16,  # 16位脑浮点数，常用于现代GPU训练
        "bf16": torch.bfloat16,  # bfloat16 的缩写
        "float16": torch.float16,  # 标准16位半精度浮点数
        "fp16": torch.float16,  # float16 的缩写
    }
    # 将输入转换为小写后查找，如果未找到则返回默认值 float32
    return mapping.get(name.lower(), torch.float32)


def main(args: argparse.Namespace) -> None:
    """
    主训练函数，负责初始化所有组件并启动训练流程。
    
    该函数按照以下顺序执行：
    1. 加载配置文件
    2. 构建策略网络
    3. 创建优化器和训练器
    4. 构建工具注册表
    5. 构建任务采样器（训练集和验证集）
    6. 可选：预览任务样本以验证数据集
    7. 创建轨迹收集工作器
    8. 运行训练循环
    
    Args:
        args: 命令行参数命名空间，包含以下字段：
            - config: YAML 配置文件路径
            - dataset_dir: 数据集目录路径
            - train_split: 训练集文件名
            - eval_split: 验证集文件名
            - seed: 随机种子
            - model_path: 模型路径
            - device: 计算设备
            - preview_tasks: 预览任务数量
    
    Note:
        该函数会修改全局状态（如模型权重），但不返回任何值
    """
    # 步骤1: 从 YAML 文件加载配置，包含模型/环境/训练超参数
    cfg = load_config(args.config)
    
    # 步骤2: 根据配置和命令行参数构建策略网络（MiniMind 模型）
    policy = build_policy(cfg, args)
    
    # 步骤3: 创建优化器（AdamW）和 GRPO 训练器
    # 优化器用于更新策略网络参数，学习率从配置文件中读取
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["lr"])
    # GRPOTrainer 负责执行 GRPO 算法，包括优势计算和策略更新
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, config=cfg["train"])
    
    # 步骤4: 构建工具注册表，注册代码执行工具等
    tools = build_tool_registry(cfg)
    
    # 步骤5: 构建训练集和验证集的任务采样器
    train_sampler, eval_sampler = build_task_samplers(
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
    )
    
    # 步骤6: 如果指定了预览任务数量，打印样本任务以验证数据集连接
    if args.preview_tasks > 0:
        # 快速检查数据集连接是否正确，帮助调试
        log_preview_tasks(train_sampler, args.preview_tasks)
    
    worker_kwargs = {
        "policy": policy,
        "env_cls": CodeExecEnv,
        "env_config": cfg.get("env"),
        "tool_manager": tools,
        "max_steps": cfg.get("env", {}).get("max_turns", 6),
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
def collect_episodes(worker: RolloutWorker, task_batch: List[Dict[str, Any]]) -> List[Episode]:
    """
    为提供的任务批次运行轨迹收集，返回完成的 episode 列表。
    
    该函数遍历任务批次，对每个任务运行一个完整的 episode，收集策略在环境中的
    交互轨迹。每个 episode 包含状态、动作、奖励等完整信息，用于后续的策略更新。
    
    Args:
        worker: 轨迹收集工作器，负责在环境中运行策略
        task_batch: 任务批次列表，每个任务是一个字典，至少包含 "task_id" 和 "prompt"
    
    Returns:
        完成的 episode 列表，每个 Episode 对象包含：
        - 状态序列（观察）
        - 动作序列（策略输出）
        - 奖励序列
        - 其他元数据（如是否完成、步数等）
    
    Note:
        - worker.run_episode() 返回 (episode, info) 元组，这里只使用 episode
        - 每个任务对应一个独立的 episode
        - episode 会在达到最大步数或任务完成时结束
    """
    episodes = []
    # 遍历任务批次中的每个任务
    for task in task_batch:
        # 运行一个完整的 episode，收集策略与环境的交互轨迹
        # run_episode 返回 (episode, info)，这里只使用 episode
        episode, _ = worker.run_episode(task)
        episodes.append(episode)
    return episodes


def run_training_loop(
    trainer: GRPOTrainer,
    worker: RolloutWorker,
    task_sampler: CodeTaskSampler,
    rollout_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    async_executor: AsyncRolloutExecutor | None = None,
) -> None:
    """
    运行最小化的同步训练循环：采样任务 → 收集轨迹 → 更新策略。
    
    这是标准的强化学习训练循环，每个迭代包含以下步骤：
    1. 从任务采样器中采样一批任务
    2. 使用工作器在环境中运行策略，收集 episode 轨迹
    3. 使用训练器根据收集的轨迹更新策略参数
    4. 打印训练指标（损失、平均奖励、平均优势等）
    
    Args:
        trainer: GRPO 训练器，负责策略更新和优势计算
        worker: 轨迹收集工作器，用于在环境中运行策略
        task_sampler: 任务采样器，用于采样训练任务
        rollout_cfg: 轨迹收集配置字典
            - num_episodes_per_iter: 每次迭代收集的 episode 数量（默认: 4）
        train_cfg: 训练配置字典
            - num_iters: 训练迭代次数（默认: 1）
    
    Note:
        - 这是同步训练循环，即收集完所有轨迹后再更新策略
        - 每个迭代会打印训练指标，包括损失、平均奖励和平均优势
        - 优势（advantage）是 GRPO 算法的核心，用于衡量动作的相对好坏
        - 训练循环会持续 num_iters 次迭代
    """
    # 从配置中获取每次迭代收集的 episode 数量，默认值为 4
    episodes_per_iter = rollout_cfg.get("num_episodes_per_iter", 4)
    # 从配置中获取训练迭代次数，默认值为 1
    num_iters = train_cfg.get("num_iters", 1)
    
    # 训练循环：重复执行采样 → 收集 → 更新的过程
    for iteration in range(num_iters):
        # 步骤1: 从任务采样器中采样一批任务
        task_batch = task_sampler.next(batch_size=episodes_per_iter)
        
        # 步骤2: 使用同步或异步方式收集轨迹
        if async_executor:
            episodes, _ = async_executor.run_batch(task_batch)
        else:
            episodes = collect_episodes(worker, task_batch)
        
        # 步骤3: 使用训练器根据收集的轨迹更新策略参数
        # update 方法会计算优势、损失，并执行反向传播和参数更新
        metrics = trainer.update(episodes)
        
        # 步骤4: 打印当前迭代的训练指标
        print(
            f"[train] iter={iteration} "  # 迭代编号
            f"loss={metrics['loss']:.4f} "  # 策略损失（用于优化）
            f"reward={metrics['mean_reward']:.4f} "  # 平均奖励（性能指标）
            f"adv={metrics['mean_advantage']:.4f}"  # 平均优势（GRPO核心指标）
        )


if __name__ == "__main__":
    """
    脚本入口点：解析命令行参数并启动训练流程。
    
    该脚本支持通过命令行参数自定义训练配置，包括：
    - 配置文件路径
    - 数据集路径和分割
    - 模型路径和设备
    - 随机种子
    - 任务预览数量
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Train code exec agent with GRPO.")
    
    # 配置文件路径：YAML 格式，包含模型/环境/训练等所有配置
    parser.add_argument(
        "--config",
        default="agent_rl/configs/code_exec_grpo.yaml",
        help="Path to YAML config file containing model/env/training hyperparameters."
    )
    
    # 数据集目录：包含训练集和验证集的 JSONL 文件
    parser.add_argument(
        "--dataset-dir",
        default="dataset/code_tasks",
        help="Directory containing train/eval JSONL split files."
    )
    
    # 训练集文件名（不含扩展名）：例如 "train" 对应 "train.jsonl"
    parser.add_argument(
        "--train-split",
        default="train",
        help="Training split file stem (without extension), e.g., 'train' for 'train.jsonl'."
    )
    
    # 验证集文件名（不含扩展名）：例如 "eval" 对应 "eval.jsonl"
    parser.add_argument(
        "--eval-split",
        default="eval",
        help="Evaluation split file stem (without extension), e.g., 'eval' for 'eval.jsonl'."
    )
    
    # 随机种子：用于确保数据加载和采样的可重复性
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling and reproducibility."
    )
    
    # 模型路径：包含 MiniMind 模型权重和分词器的目录
    parser.add_argument(
        "--model-path",
        default="model",
        help="Directory containing MiniMind model weights and tokenizer files."
    )
    
    # 计算设备：指定运行策略的设备（CPU 或 GPU）
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for policy execution, e.g., 'cpu' or 'cuda:0' for GPU."
    )
    parser.add_argument(
        "--policy-type",
        choices=["minimind", "huggingface"],
        default="minimind",
        help="Choose which policy backend to instantiate.",
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="HuggingFace model identifier when --policy-type=huggingface.",
    )
    
    # 预览任务数量：在训练开始前打印样本任务，用于验证数据集连接
    parser.add_argument(
        "--preview-tasks",
        type=int,
        default=3,
        help="Number of sample tasks to print at startup for dataset verification (0 to disable)."
    )
    parser.add_argument(
        "--async-rollout-workers",
        type=int,
        default=1,
        help="Number of parallel rollout workers for asynchronous data collection.",
    )
    
    # 解析命令行参数
    cli_args = parser.parse_args()
    
    # 调用主函数启动训练流程
    main(cli_args)

