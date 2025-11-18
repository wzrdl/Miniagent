"""
GRPO（Group Relative Policy Optimization）训练器包装器，兼容代理轨迹。

该模块实现了 GRPO 训练算法，用于训练语言模型策略。GRPO 是一种
相对策略优化方法，通过比较同一任务组内的不同 episode 来计算优势函数。

主要组件：
- GRPOTrainerConfig: GRPO 训练配置
- GRPOTrainer: GRPO 训练器，执行策略更新

GRPO 算法特点：
- 使用组内相对比较计算优势（而非绝对奖励）
- 支持按任务 ID 分组
- 可以标准化优势（除以标准差）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from collections import defaultdict

import torch
from torch.optim import Optimizer

from agent_rl.core.policy_base import BasePolicy
from agent_rl.core.trajectory import Episode


@dataclass
class GRPOTrainerConfig:
    """
    GRPO 训练器配置数据类。
    
    该数据类包含 GRPO 训练所需的所有超参数。
    
    Attributes:
        learning_rate: float, default=1e-5
            学习率，控制参数更新的步长
            较小的值使训练更稳定但更慢
        
        grad_clip: float | None, default=1.0
            梯度裁剪阈值，防止梯度爆炸
            如果为 None，则不进行梯度裁剪
            使用梯度范数裁剪（gradient norm clipping）
        
        gamma: float, default=0.99
            折扣因子，用于计算未来奖励的现值
            当前实现中可能未使用，保留用于未来扩展
        
        lam: float, default=0.95
            GAE（Generalized Advantage Estimation）的 lambda 参数
            当前实现中可能未使用，保留用于未来扩展
        
        value_coef: float, default=0.5
            价值函数损失的系数
            当前实现中可能未使用，保留用于未来扩展
        
        entropy_coef: float, default=0.01
            熵正则化系数，鼓励探索
            当前实现中可能未使用，保留用于未来扩展
        
        norm_adv_by_std: bool, default=True
            是否通过标准差标准化优势
            True: 优势 = (奖励 - 均值) / 标准差
            False: 优势 = 奖励 - 均值
    
    Examples:
        >>> config = GRPOTrainerConfig(
        ...     learning_rate=2e-5,
        ...     grad_clip=1.0,
        ...     norm_adv_by_std=True
        ... )
    
    Note:
        - 使用 dataclass 可以方便地创建和传递配置
        - 某些参数（如 gamma、lam）可能在当前实现中未使用
    """
    # 学习率
    learning_rate: float = 1e-5
    # 梯度裁剪阈值（None 表示不裁剪）
    grad_clip: float | None = 1.0
    # 折扣因子（可能未使用）
    gamma: float = 0.99
    # GAE lambda 参数（可能未使用）
    lam: float = 0.95
    # 价值函数损失系数（可能未使用）
    value_coef: float = 0.5
    # 熵正则化系数（可能未使用）
    entropy_coef: float = 0.01
    # 是否通过标准差标准化优势
    norm_adv_by_std: bool = True


class GRPOTrainer:
    """
    薄包装器，将调用 MiniMind 的 GRPO 工具。
    
    该训练器实现了 GRPO 算法的核心逻辑：
    1. 从 episode 轨迹中提取奖励和元数据
    2. 按任务 ID 分组计算优势函数
    3. 计算策略损失并更新参数
    
    当前实现是一个占位符，未来将集成 MiniMind 的训练工具。
    
    Attributes:
        policy: BasePolicy
            要训练的策略网络
        
        optimizer: Optimizer
            优化器，用于更新策略参数
        
        config: GRPOTrainerConfig
            训练配置对象
        
        _device: torch.device
            计算设备（CPU 或 GPU），从策略参数中自动检测
    
    Examples:
        >>> trainer = GRPOTrainer(
        ...     policy=policy,
        ...     optimizer=optimizer,
        ...     config={"learning_rate": 1e-5}
        ... )
        >>> metrics = trainer.update(episodes)
    
    Note:
        - 当前实现是简化版本，未来将集成完整的 GRPO 实现
        - 需要 episode 包含策略元数据（prompt_text、response_ids）
    """

    def __init__(
        self,
        policy: BasePolicy,
        optimizer: Optimizer,
        config: GRPOTrainerConfig | Dict[str, Any],
    ) -> None:
        """
        初始化 GRPO 训练器。
        
        Args:
            policy: 要训练的策略网络
            optimizer: 优化器，用于更新参数
            config: 训练配置，可以是 GRPOTrainerConfig 对象或字典
                如果是字典，会自动转换为 GRPOTrainerConfig 对象
        """
        # 保存策略网络
        self.policy = policy
        # 保存优化器
        self.optimizer = optimizer
        # 处理配置：如果是字典则转换为配置对象
        if isinstance(config, GRPOTrainerConfig):
            self.config = config
        else:
            self.config = GRPOTrainerConfig(**config)
        # 从策略参数中自动检测计算设备
        self._device = next(self.policy.parameters()).device

    def update(self, episodes: Iterable[Episode]) -> Dict[str, Any]:
        """
        将 episode 轨迹转换为 token 级别的训练数据并调用 GRPO 损失。
        
        该方法当前作为占位符，直到与 MiniMind 的训练工具集成完成。
        
        处理流程：
        1. 收集所有 episode 的奖励
        2. 按任务 ID 分组计算优势函数
        3. 计算每个 episode 的对数概率
        4. 计算策略损失并更新参数
        
        Args:
            episodes: episode 的可迭代对象
                每个 episode 应该包含策略元数据（prompt_text、response_ids）
        
        Returns:
            Dict[str, Any]: 包含训练指标的字典：
                - "loss": 策略损失值
                - "mean_reward": 平均奖励值
                - "mean_advantage": 平均优势值
        
        Raises:
            ValueError: 如果 episode 缺少必需的策略元数据
        
        Examples:
            >>> episodes = [episode1, episode2, episode3]
            >>> metrics = trainer.update(episodes)
            >>> print(f"Loss: {metrics['loss']}, Reward: {metrics['mean_reward']}")
        
        Note:
            - 如果批次为空，返回零损失和零奖励
            - 需要 episode 的最后一步包含策略元数据
            - 使用策略的 compute_sequence_logprob 重新计算对数概率
        """
        # 步骤1: 将可迭代对象转换为列表
        batch = list(episodes)
        
        # 步骤2: 如果批次为空，返回零指标
        if not batch:
            return {"loss": 0.0, "mean_reward": 0.0}
        
        # 步骤3: 收集所有 episode 的奖励并转换为张量
        rewards = torch.tensor(
            [self._episode_reward(ep) for ep in batch],
            device=self._device
        )
        
        # 步骤4: 收集任务 ID 列表
        task_ids = [ep.task_id for ep in batch]
        
        # 步骤5: 按任务 ID 分组计算优势函数
        advantages = self._compute_advantages(rewards, task_ids)
        
        # 步骤6: 计算每个 episode 的对数概率总和
        logprob_sums = []
        for episode in batch:
            # 6.1: 从最后一步的元数据中提取策略信息
            policy_meta = episode.steps[-1].info.get("policy", {})
            prompt_text = policy_meta.get("prompt_text")
            response_ids = policy_meta.get("response_ids")
            
            # 6.2: 验证必需的元数据是否存在
            if prompt_text is None or response_ids is None:
                raise ValueError(
                    "Episode is missing policy metadata required for GRPO update. "
                    "Ensure MiniMindPolicy.generate_action returns response ids."
                )
            
            # 6.3: 将响应 IDs 转换为张量
            response_tensor = torch.as_tensor(
                response_ids,
                dtype=torch.long,
                device=self._device
            )
            
            # 6.4: 使用策略重新计算响应序列的对数概率
            token_logprobs = self.policy.compute_sequence_logprob(
                prompt_text,
                response_tensor
            )
            
            # 6.5: 将对数概率求和（整个响应的总对数概率）
            logprob_sums.append(token_logprobs.sum())
        
        # 步骤7: 堆叠所有对数概率总和
        logprob_tensor = torch.stack(logprob_sums)
        
        # 步骤8: 计算策略损失
        # GRPO 损失 = -优势 * 对数概率（最大化优势高的动作的概率）
        loss = -(advantages * logprob_tensor).mean()
        
        # 步骤9: 执行反向传播和参数更新
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        # 梯度裁剪（如果配置了）
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.grad_clip
            )
        self.optimizer.step()  # 更新参数
        
        # 步骤10: 返回训练指标（转换为 Python 标量）
        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "mean_advantage": float(advantages.mean().detach().cpu()),
        }

    def _episode_reward(self, episode: Episode) -> float:
        """
        获取 episode 的奖励值。
        
        优先返回 final_reward，如果不存在则返回总奖励。
        
        Args:
            episode: Episode 对象
        
        Returns:
            float: episode 的奖励值
        
        Note:
            - final_reward 优先于步骤奖励总和
        """
        # 如果设置了最终奖励，直接返回
        if episode.final_reward is not None:
            return episode.final_reward
        # 否则计算总奖励
        return episode.total_reward()

    def _compute_advantages(self, rewards: torch.Tensor, task_ids: List[str]) -> torch.Tensor:
        """
        计算 GRPO 风格的结果优势，按任务 ID 分组。
        
        GRPO 的核心思想是：优势 = 奖励 - 组内平均奖励
        这样可以比较同一任务组内的不同 episode，而不是使用绝对奖励。
        
        如果启用了标准化，优势会被除以组内标准差，使其具有单位方差。
        
        Args:
            rewards: 奖励张量，形状为 (batch_size,)
                每个元素对应一个 episode 的奖励
            task_ids: 任务 ID 列表，长度与 rewards 相同
                用于将 episode 分组
        
        Returns:
            torch.Tensor: 优势张量，形状为 (batch_size,)
                每个元素是对应 episode 的优势值
        
        Examples:
            >>> rewards = torch.tensor([1.0, 0.5, 1.0, 0.5])
            >>> task_ids = ["task_1", "task_1", "task_2", "task_2"]
            >>> advantages = trainer._compute_advantages(rewards, task_ids)
            >>> # task_1 组: 均值 0.75, 优势 = [0.25, -0.25]
            >>> # task_2 组: 均值 0.75, 优势 = [0.25, -0.25]
        
        Note:
            - 优势是相对于组内平均值的
            - 如果组内只有一个样本，无法计算标准差，使用均值中心化
            - 标准化可以稳定训练，但可能不是必需的
        """
        # 步骤1: 按任务 ID 分组收集奖励
        grouped: Dict[str, List[float]] = defaultdict(list)
        for reward, task_id in zip(rewards.tolist(), task_ids):
            grouped[task_id].append(reward)
        
        # 步骤2: 初始化优势张量
        advantages = torch.zeros_like(rewards)
        
        # 步骤3: 为每个 episode 计算优势
        for idx, (reward, task_id) in enumerate(zip(rewards, task_ids)):
            # 3.1: 获取该任务组的所有奖励
            group = torch.tensor(grouped[task_id], device=self._device)
            
            # 3.2: 计算组内平均奖励
            mean = group.mean()
            
            # 3.3: 计算优势（根据配置决定是否标准化）
            if self.config.norm_adv_by_std and group.numel() > 1:
                # 如果启用标准化且组内有多个样本，除以标准差
                std = group.std(unbiased=False)  # 使用有偏标准差（除以 n）
                advantages[idx] = (reward - mean) / (std + 1e-6)  # 添加小值防止除零
            else:
                # 否则只进行均值中心化
                advantages[idx] = reward - mean
        
        # 步骤4: 返回优势张量
        return advantages


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["GRPOTrainer", "GRPOTrainerConfig"]

