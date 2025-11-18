"""
可重用的奖励函数，用于不同任务的奖励计算。

该模块定义了奖励函数的抽象接口和注册表机制，支持：
1. 定义自定义奖励函数（继承 RewardFunction）
2. 注册和管理多个奖励函数（使用 RewardRegistry）
3. 根据任务类型动态选择奖励函数

主要组件：
- RewardFunction: 奖励函数抽象基类
- RewardRegistry: 奖励函数注册表，用于管理和查找奖励函数

设计模式：
- 策略模式：不同的奖励函数可以互换使用
- 注册表模式：集中管理所有可用的奖励函数
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class RewardFunction(ABC):
    """
    奖励函数抽象基类。
    
    所有具体的奖励函数都应该继承此类并实现 __call__ 方法。
    奖励函数用于评估一个完整 episode 的表现，返回奖励值和相关信息。
    
    Attributes:
        name: str, default="base"
            奖励函数的名称，用于在注册表中标识该函数
            子类应该覆盖此属性以提供唯一的名称
    
    Examples:
        >>> class CodeExecReward(RewardFunction):
        ...     name = "code_exec"
        ...     def __call__(self, episode_data):
        ...         # 计算代码执行任务的奖励
        ...         reward = 1.0 if episode_data["success"] else 0.0
        ...         return reward, {"success": episode_data["success"]}
        >>> reward_fn = CodeExecReward()
        >>> reward, info = reward_fn({"success": True})
    
    Note:
        - 这是抽象类，不能直接实例化
        - 子类必须实现 __call__ 方法
        - name 属性应该唯一，用于注册表查找
    """
    # 奖励函数名称，子类应该覆盖此属性
    name: str = "base"

    @abstractmethod
    def __call__(self, episode_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算并返回 episode 的奖励值和相关信息。
        
        这是奖励函数的核心方法，子类必须实现。该方法接收一个完整的
        episode 数据，计算奖励值并返回额外的信息。
        
        Args:
            episode_data: 包含 episode 信息的字典，通常包括：
                - "steps": episode 的所有步骤
                - "task_id": 任务 ID
                - "final_reward": 最终奖励（如果有）
                - 其他任务特定的信息
        
        Returns:
            Tuple[float, Dict[str, Any]]: 包含两个元素的元组：
                - 第一个元素（float）: 奖励值，可以是任意实数
                  正数表示好的表现，负数表示差的表现，0 表示中性
                - 第二个元素（Dict）: 额外的信息字典，可以包含：
                  - 奖励计算的详细信息
                  - 调试信息
                  - 任务特定的指标
        
        Examples:
            >>> reward, info = reward_fn({
            ...     "steps": [...],
            ...     "task_id": "task_1",
            ...     "success": True
            ... })
            >>> print(f"Reward: {reward}, Info: {info}")
        
        Note:
            - 奖励值应该是标量（单个浮点数）
            - 奖励函数应该是确定性的（相同输入产生相同输出）
            - 奖励值的大小应该合理，避免过大或过小
        """
        pass


class RewardRegistry:
    """
    奖励函数注册表，用于管理和查找奖励函数。
    
    该注册表使用名称作为键，存储和管理所有可用的奖励函数。
    支持注册新函数和根据名称查找函数。
    
    Attributes:
        _registry: dict[str, RewardFunction]
            内部字典，存储名称到奖励函数的映射
            键是奖励函数的名称，值是奖励函数实例
    
    Examples:
        >>> registry = RewardRegistry()
        >>> registry.register(CodeExecReward())
        >>> reward_fn = registry.get("code_exec")
        >>> reward, info = reward_fn(episode_data)
    
    Note:
        - 注册表是线程不安全的，如果多线程使用需要加锁
        - 同一个名称只能注册一个函数，后注册的会覆盖先注册的
    """

    def __init__(self) -> None:
        """
        初始化奖励函数注册表。
        
        创建一个空的注册表，准备存储奖励函数。
        """
        # 内部字典，存储名称到奖励函数的映射
        self._registry: dict[str, RewardFunction] = {}

    def register(self, reward_fn: RewardFunction) -> None:
        """
        注册一个奖励函数到注册表中。
        
        使用奖励函数的 name 属性作为键，将函数实例存储到注册表中。
        如果该名称已存在，则覆盖原有的函数。
        
        Args:
            reward_fn: 要注册的奖励函数实例
                必须有一个有效的 name 属性
        
        Examples:
            >>> registry = RewardRegistry()
            >>> registry.register(CodeExecReward())
            >>> len(registry._registry)  # 1
        
        Note:
            - 如果名称已存在，会覆盖原有的函数
            - 不会检查函数是否实现了 __call__ 方法
        """
        # 使用奖励函数的名称作为键，存储函数实例
        self._registry[reward_fn.name] = reward_fn

    def get(self, name: str) -> RewardFunction:
        """
        根据名称获取奖励函数。
        
        Args:
            name: 奖励函数的名称（字符串）
                应该与注册时使用的名称一致
        
        Returns:
            RewardFunction: 对应的奖励函数实例
        
        Raises:
            KeyError: 如果指定的名称未在注册表中找到
        
        Examples:
            >>> registry = RewardRegistry()
            >>> registry.register(CodeExecReward())
            >>> reward_fn = registry.get("code_exec")
            >>> isinstance(reward_fn, CodeExecReward)
            True
        
        Note:
            - 如果名称不存在，会抛出 KeyError
            - 可以使用 `name in registry._registry` 检查名称是否存在
        """
        # 从注册表中查找并返回对应的奖励函数
        return self._registry[name]


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["RewardFunction", "RewardRegistry"]

