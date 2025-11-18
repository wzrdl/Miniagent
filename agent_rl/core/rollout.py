"""
同步轨迹收集工作器，用于从环境中收集轨迹。

该模块实现了同步的轨迹收集工作器，负责：
1. 在环境中运行策略
2. 收集交互轨迹（状态、动作、奖励等）
3. 记录性能统计信息

主要组件：
- RolloutStats: 轨迹收集统计信息
- RolloutWorker: 轨迹收集工作器，执行完整的 episode
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Type

from agent_rl.core.policy_base import BasePolicy
from agent_rl.core.trajectory import Episode, Step
from agent_rl.envs.base_env import AgentEnv


@dataclass
class RolloutStats:
    """
    轨迹收集统计信息数据类。
    
    该数据类用于记录一个 episode 的收集统计信息，包括：
    - episode 标识信息
    - 性能指标（步数、延迟等）
    - 最终奖励
    
    Attributes:
        episode_id: str
            Episode 的唯一标识符
            与 Episode.episode_id 一致
        
        task_id: str
            任务 ID，标识对应的任务
            与 Episode.task_id 一致
        
        num_steps: int
            Episode 的步数
            表示从开始到结束的交互次数
        
        latency_s: float
            轨迹收集的延迟时间（秒）
            从 episode 开始到结束的总时间
        
        final_reward: float | None
            最终奖励值
            与 Episode.final_reward 一致
    
    Examples:
        >>> stats = RolloutStats(
        ...     episode_id="ep_123",
        ...     task_id="task_1",
        ...     num_steps=5,
        ...     latency_s=2.3,
        ...     final_reward=1.0
        ... )
    
    Note:
        - 使用 dataclass 可以方便地创建和访问字段
        - 所有字段都是必需的，没有默认值
    """
    # Episode 的唯一标识符
    episode_id: str
    # 任务 ID
    task_id: str
    # Episode 的步数
    num_steps: int
    # 轨迹收集的延迟时间（秒）
    latency_s: float
    # 最终奖励值
    final_reward: float | None


class RolloutWorker:
    """
    简单的同步轨迹收集工作器。
    
    该工作器负责在环境中运行策略并收集完整的 episode 轨迹。
    工作流程：
    1. 创建环境实例
    2. 重置环境获取初始观察
    3. 循环执行：采样动作 → 环境步进 → 记录步骤
    4. 直到 episode 结束或达到最大步数
    5. 计算最终奖励并返回轨迹和统计信息
    
    Attributes:
        policy: BasePolicy
            策略网络，用于生成动作
        
        env_cls: Type[AgentEnv]
            环境类，用于创建环境实例
        
        env_config: Dict[str, Any]
            环境配置字典，传递给环境构造函数
        
        tool_manager: Any
            工具管理器，提供给环境使用
        
        max_steps: int, default=8
            每个 episode 的最大步数
            防止无限循环
    
    Examples:
        >>> worker = RolloutWorker(
        ...     policy=policy,
        ...     env_cls=CodeExecEnv,
        ...     env_config={},
        ...     tool_manager=tools,
        ...     max_steps=10
        ... )
        >>> episode, stats = worker.run_episode(task_sample)
    
    Note:
        - 这是同步工作器，一次只能运行一个 episode
        - 如果达到最大步数仍未结束，会强制结束
        - 所有步骤都会被记录，即使提前结束
    """

    def __init__(
        self,
        policy: BasePolicy,
        env_cls: Type[AgentEnv],
        env_config: Dict[str, Any] | None,
        tool_manager: Any,
        max_steps: int = 8,
    ) -> None:
        """
        初始化轨迹收集工作器。
        
        Args:
            policy: 策略网络，用于生成动作
            env_cls: 环境类，用于创建环境实例
            env_config: 环境配置字典，如果为 None 则使用空字典
            tool_manager: 工具管理器，提供给环境使用
            max_steps: 每个 episode 的最大步数，默认 8
        """
        # 保存策略网络
        self.policy = policy
        # 保存环境类（用于创建实例）
        self.env_cls = env_cls
        # 保存环境配置（如果为 None 则使用空字典）
        self.env_config = env_config or {}
        # 保存工具管理器
        self.tool_manager = tool_manager
        # 保存最大步数
        self.max_steps = max_steps

    def run_episode(self, task_sample: Dict[str, Any]) -> tuple[Episode, RolloutStats]:
        """
        运行一个完整的 episode 并收集轨迹。
        
        该方法执行完整的 episode 流程：
        1. 创建环境实例
        2. 重置环境获取初始观察
        3. 循环执行策略和环境交互
        4. 记录所有步骤
        5. 计算最终奖励
        6. 返回轨迹和统计信息
        
        Args:
            task_sample: 任务样本字典，包含任务信息
                通常包括 "task_id" 和 "prompt" 等字段
                传递给环境构造函数
        
        Returns:
            tuple[Episode, RolloutStats]: 包含两个元素的元组：
                - Episode: 完整的 episode 轨迹
                - RolloutStats: 轨迹收集统计信息
        
        Examples:
            >>> task = {"task_id": "task_1", "prompt": "写一个函数"}
            >>> episode, stats = worker.run_episode(task)
            >>> print(f"Steps: {stats.num_steps}, Reward: {stats.final_reward}")
        
        Note:
            - 如果达到最大步数仍未结束，会强制结束
            - 所有步骤都会被记录，包括提前结束的情况
            - 最终奖励由环境计算
        """
        # 步骤1: 创建环境实例
        env = self.env_cls(task_sample, self.env_config, self.tool_manager)
        
        # 步骤2: 重置环境，获取初始观察
        obs = env.reset()
        
        # 步骤3: 初始化步骤列表和其他变量
        steps: list[Step] = []
        episode_id = str(uuid.uuid4())  # 生成唯一的 episode ID
        start = time.time()  # 记录开始时间
        
        # 步骤4: 循环执行策略和环境交互
        for _ in range(self.max_steps):
            # 4.1: 使用策略采样动作
            action = self._sample_action(obs)
            
            # 4.2: 环境步进，执行动作
            next_obs, reward, done, env_info = env.step(action)
            
            # 4.3: 记录步骤信息
            steps.append(
                Step(
                    obs=obs,  # 当前观察
                    action=action,  # 执行的动作
                    logprob=action.get("metadata", {}).get("response_logprobs"),  # 动作的对数概率
                    reward=reward,  # 获得的奖励
                    done=done,  # 是否结束
                    info={"env": env_info, "policy": action.get("metadata", {})},  # 额外信息
                )
            )
            
            # 4.4: 更新观察
            obs = next_obs
            
            # 4.5: 如果 episode 结束，提前退出循环
            if done:
                break
        
        # 步骤5: 创建 Episode 对象
        episode = Episode(
            steps=steps,
            episode_id=episode_id,
            task_id=str(task_sample.get("task_id", "unknown")),  # 从任务样本中获取任务 ID
        )
        
        # 步骤6: 计算最终奖励
        episode.final_reward = env.compute_final_reward()
        
        # 步骤7: 计算延迟时间
        latency = time.time() - start
        
        # 步骤8: 创建统计信息对象
        stats = RolloutStats(
            episode_id=episode_id,
            task_id=episode.task_id,
            num_steps=len(steps),
            latency_s=latency,
            final_reward=episode.final_reward,
        )
        
        # 步骤9: 返回轨迹和统计信息
        return episode, stats

    def _sample_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        默认动作采样器，委托给策略的 generate_action 方法。
        
        该方法从观察中提取消息，然后调用策略生成动作。
        这是默认实现，子类可以覆盖以提供自定义采样逻辑。
        
        Args:
            obs: 观察字典，应该包含 "messages" 字段
                消息列表用于策略生成响应
        
        Returns:
            Dict[str, Any]: 动作字典，包含策略生成的响应
                通常包括 "content"、"metadata" 等字段
        
        Raises:
            ValueError: 如果观察中不包含 "messages" 字段
        
        Examples:
            >>> obs = {"messages": [{"role": "user", "content": "Hello"}]}
            >>> action = worker._sample_action(obs)
            >>> print(action["content"])
        
        Note:
            - 这是受保护方法，通常由 run_episode 调用
            - 子类可以覆盖此方法以提供自定义采样逻辑
            - 观察必须包含 "messages" 字段
        """
        # 从观察中提取消息列表
        messages = obs.get("messages")
        
        # 验证消息是否存在
        if messages is None:
            raise ValueError(
                "Observation must include 'messages' when using default sampler."
            )
        
        # 调用策略生成动作
        return self.policy.generate_action(messages=messages)


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["RolloutWorker", "RolloutStats"]

