"""
轨迹数据结构，在轨迹收集工作器和训练器之间共享。

该模块定义了强化学习中用于存储和传递轨迹数据的数据结构：
- Step: 单个时间步的数据（状态、动作、奖励等）
- Episode: 完整的 episode，包含多个步骤
- TrajectoryBuffer: 经验缓冲区，用于存储和管理多个 episode

这些数据结构是强化学习训练流程的核心，用于：
1. 收集策略在环境中的交互轨迹
2. 存储经验数据供训练器使用
3. 支持批处理和采样操作
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import random

import torch


@dataclass
class Step:
    """
    单个时间步的数据结构，表示策略在环境中的一个交互步骤。
    
    该数据类使用 @dataclass 装饰器，自动生成 __init__、__repr__ 等方法。
    每个 Step 包含一个完整的 (状态, 动作, 奖励) 三元组。
    
    Attributes:
        obs: Dict[str, Any]
            观察（状态），包含环境返回的所有信息
            通常包括 "messages" 字段，包含对话历史
            形状和内容取决于具体环境
    
        action: Dict[str, Any]
            动作，策略生成的响应
            通常包括 "content"（文本内容）和 "metadata"（元数据）
            可能包含 "tool_call"（工具调用信息）
    
        logprob: torch.Tensor | None, optional
            动作的对数概率，用于策略梯度计算
            形状通常为 (action_length,)，表示每个 token 的对数概率
            如果为 None，表示未提供对数概率信息
    
        reward: float
            该步骤获得的奖励值
            可以是任意实数，正数表示好的表现，负数表示差的表现
    
        done: bool
            是否结束标志，表示该步骤后 episode 是否结束
            True 表示 episode 已完成，False 表示继续
    
        info: Dict[str, Any], default={}
            额外的信息字典，可以包含：
            - "env": 环境返回的额外信息
            - "policy": 策略返回的元数据（如 prompt_text、response_ids 等）
            - 其他调试或分析信息
    
    Examples:
        >>> step = Step(
        ...     obs={"messages": [...]},
        ...     action={"content": "Hello", "metadata": {...}},
        ...     logprob=torch.tensor([-0.5, -0.3]),
        ...     reward=0.1,
        ...     done=False,
        ...     info={"env": {}, "policy": {}}
        ... )
    
    Note:
        - 使用 dataclass 可以方便地创建和访问字段
        - logprob 和 info 是可选的，有默认值
        - 所有字段都可以在创建后修改
    """
    # 观察（状态），包含环境的完整信息
    obs: Dict[str, Any]
    # 动作，策略生成的响应
    action: Dict[str, Any]
    # 动作的对数概率（可选），用于策略梯度计算
    logprob: torch.Tensor | None
    # 该步骤获得的奖励值
    reward: float
    # 是否结束标志
    done: bool
    # 额外的信息字典（可选）
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """
    完整的 episode 数据结构，包含一个任务的所有交互步骤。
    
    该数据类表示从任务开始到结束的完整轨迹，包含所有步骤的序列。
    用于存储和传递完整的交互历史，供训练器使用。
    
    Attributes:
        steps: List[Step]
            步骤列表，按时间顺序排列
            第一个步骤是初始状态，最后一个步骤通常是结束状态
            列表长度表示 episode 的步数
    
        episode_id: str
            Episode 的唯一标识符
            通常使用 UUID 生成，用于追踪和调试
    
        task_id: str
            任务 ID，标识该 episode 对应的任务
            用于将多个 episode 分组（同一任务可能有多个 episode）
    
        final_reward: float | None, optional, default=None
            最终奖励值，由环境在 episode 结束时计算
            如果为 None，则使用所有步骤奖励的总和
            通常用于任务完成时的最终评估
    
        metadata: Dict[str, Any], default={}
            额外的元数据字典，可以包含：
            - 任务特定的信息
            - 调试信息
            - 性能指标
    
    Methods:
        total_reward() -> float:
            计算 episode 的总奖励
            如果 final_reward 不为 None，返回 final_reward
            否则返回所有步骤奖励的总和
    
    Examples:
        >>> episode = Episode(
        ...     steps=[step1, step2, step3],
        ...     episode_id="ep_123",
        ...     task_id="task_1",
        ...     final_reward=1.0
        ... )
        >>> total = episode.total_reward()  # 1.0
    
    Note:
        - steps 列表应该按时间顺序排列
        - episode_id 应该是唯一的
        - final_reward 优先于步骤奖励总和
    """
    # 步骤列表，按时间顺序排列
    steps: List[Step]
    # Episode 的唯一标识符
    episode_id: str
    # 任务 ID，标识对应的任务
    task_id: str
    # 最终奖励值（可选）
    final_reward: float | None = None
    # 额外的元数据字典（可选）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_reward(self) -> float:
        """
        计算 episode 的总奖励。
        
        该方法返回 episode 的累计奖励。如果设置了 final_reward，
        则优先返回 final_reward；否则计算所有步骤奖励的总和。
        
        Returns:
            float: episode 的总奖励值
        
        Examples:
            >>> episode.final_reward = 1.0
            >>> episode.total_reward()  # 1.0
            >>> episode.final_reward = None
            >>> episode.total_reward()  # sum(step.reward for step in episode.steps)
        
        Note:
            - final_reward 优先于步骤奖励总和
            - 如果所有步骤的奖励都是 0，总奖励也是 0
        """
        # 如果设置了最终奖励，直接返回
        if self.final_reward is not None:
            return self.final_reward
        # 否则计算所有步骤奖励的总和
        return sum(step.reward for step in self.steps)


class TrajectoryBuffer:
    """
    最小化的经验缓冲区，用于存储完整的 episode。
    
    该缓冲区提供基本的存储和采样功能，用于管理收集到的经验数据。
    支持添加单个或批量 episode，以及随机采样批次。
    
    Attributes:
        _episodes: List[Episode]
            内部列表，存储所有 episode
            按添加顺序排列，支持随机采样
    
    Examples:
        >>> buffer = TrajectoryBuffer()
        >>> buffer.add(episode1)
        >>> buffer.extend([episode2, episode3])
        >>> batch = buffer.sample_batch(2)
        >>> all_episodes = buffer.pop_all()
    
    Note:
        - 缓冲区是先进先出（FIFO）的，但采样是随机的
        - pop_all 会清空缓冲区
        - 不支持删除单个 episode
    """

    def __init__(self) -> None:
        """
        初始化经验缓冲区。
        
        创建一个空的缓冲区，准备存储 episode。
        """
        # 内部列表，存储所有 episode
        self._episodes: List[Episode] = []

    def __len__(self) -> int:
        """
        返回缓冲区中 episode 的数量。
        
        Returns:
            int: episode 的数量
        
        Examples:
            >>> buffer = TrajectoryBuffer()
            >>> len(buffer)  # 0
            >>> buffer.add(episode)
            >>> len(buffer)  # 1
        """
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        """
        添加一个 episode 到缓冲区。
        
        Args:
            episode: 要添加的 Episode 对象
        
        Examples:
            >>> buffer = TrajectoryBuffer()
            >>> buffer.add(episode1)
            >>> len(buffer)  # 1
        
        Note:
            - episode 会被添加到列表末尾
            - 不会检查 episode 是否已存在
        """
        # 将 episode 添加到列表末尾
        self._episodes.append(episode)

    def extend(self, episodes: Iterable[Episode]) -> None:
        """
        批量添加多个 episode 到缓冲区。
        
        Args:
            episodes: episode 的可迭代对象（如列表、元组等）
        
        Examples:
            >>> buffer = TrajectoryBuffer()
            >>> buffer.extend([episode1, episode2, episode3])
            >>> len(buffer)  # 3
        
        Note:
            - 内部调用 add 方法逐个添加
            - 保持原有顺序
        """
        # 遍历可迭代对象，逐个添加 episode
        for ep in episodes:
            self.add(ep)

    def pop_all(self) -> List[Episode]:
        """
        弹出缓冲区中的所有 episode 并清空缓冲区。
        
        该方法返回所有 episode 的列表，并清空内部缓冲区。
        用于一次性获取所有经验数据。
        
        Returns:
            List[Episode]: 所有 episode 的列表
        
        Examples:
            >>> buffer = TrajectoryBuffer()
            >>> buffer.add(episode1)
            >>> buffer.add(episode2)
            >>> episodes = buffer.pop_all()
            >>> len(episodes)  # 2
            >>> len(buffer)  # 0
        
        Note:
            - 调用后缓冲区会被清空
            - 返回的列表是原始列表的引用，不是副本
        """
        # 保存当前所有 episode
        episodes = self._episodes
        # 清空缓冲区
        self._episodes = []
        # 返回所有 episode
        return episodes

    def sample_batch(self, batch_size: int) -> List[Episode]:
        """
        从缓冲区中随机采样指定数量的 episode。
        
        使用随机采样，不重复选择。如果请求的数量大于缓冲区大小，
        会抛出 ValueError。
        
        Args:
            batch_size: 要采样的 episode 数量
                必须为正整数，且不能大于缓冲区大小
        
        Returns:
            List[Episode]: 随机采样的 episode 列表
        
        Raises:
            ValueError: 如果 batch_size <= 0 或大于缓冲区大小
        
        Examples:
            >>> buffer = TrajectoryBuffer()
            >>> buffer.extend([ep1, ep2, ep3, ep4, ep5])
            >>> batch = buffer.sample_batch(3)
            >>> len(batch)  # 3
            >>> all(ep in buffer._episodes for ep in batch)  # True
        
        Note:
            - 使用 random.sample 进行无重复随机采样
            - 采样不会从缓冲区中删除 episode
            - 每次调用的结果可能不同
        """
        # 验证 batch_size 是否为正数
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # 验证是否有足够的 episode 可以采样
        if batch_size > len(self._episodes):
            raise ValueError("Not enough episodes to sample from buffer.")
        # 使用 random.sample 进行无重复随机采样
        return random.sample(self._episodes, batch_size)


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["Step", "Episode", "TrajectoryBuffer"]

