"""
基于搜索的问答任务的简单占位符环境。

该模块实现了一个简化的问答环境，用于训练搜索和问答任务。
这是一个占位符实现，支持基本的问答功能。

主要组件：
- SearchEnv: 搜索问答环境，继承自 AgentEnv

特点：
- 单轮交互：代理只需生成一次回答
- 简单匹配：通过字符串包含检查答案是否正确
- 不支持工具调用：当前实现不支持搜索工具（占位符）
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.messages import Message
from agent_rl.envs.base_env import AgentEnv


class SearchEnv(AgentEnv):
    """
    基于搜索的问答环境（占位符实现）。
    
    该环境用于训练问答任务，支持：
    - 单轮问答：代理生成一次回答后结束
    - 答案验证：通过字符串匹配检查答案是否正确
    
    工作流程：
    1. 环境初始化，加载问题和系统提示
    2. 代理生成回答
    3. 环境检查答案是否匹配目标答案
    4. 根据匹配结果给予奖励并结束 episode
    
    Attributes:
        messages: List[Message]
            对话消息列表，包含系统提示、用户问题和助手回答
            按时间顺序排列
    
    Examples:
        >>> env = SearchEnv(
        ...     task_sample={"question": "What is Python?", "answer": "programming language"},
        ...     config={},
        ...     tool_manager=None
        ... )
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(action)
    
    Note:
        - 这是占位符实现，功能较为简单
        - 当前不支持工具调用（如搜索工具）
        - 答案匹配是简单的字符串包含检查（不区分大小写）
        - 单轮交互，代理生成回答后立即结束
    """

    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: Any | None,
    ) -> None:
        """
        初始化搜索问答环境。
        
        Args:
            task_sample: 任务样本字典，包含：
                - "question": 问题文本
                - "answer": 目标答案（用于验证）
            config: 环境配置字典，包含：
                - "system_prompt": 系统提示（可选）
            tool_manager: 工具管理器（当前未使用，保留用于未来扩展）
        """
        # 调用基类初始化方法
        super().__init__(task_sample, config, tool_manager)
        
        # 初始化对话消息列表（空列表）
        self.messages: List[Message] = []

    def reset(self) -> Dict[str, Any]:
        """
        重置环境到初始状态并返回第一个观察。
        
        该方法会：
        1. 从任务样本中加载问题
        2. 初始化对话消息（系统提示和用户问题）
        
        Returns:
            Dict[str, Any]: 初始观察字典，包含：
                - "messages": 初始消息列表（系统提示和用户问题）
        
        Examples:
            >>> obs = env.reset()
            >>> print(len(obs["messages"]))  # 2 (system + user)
        
        Note:
            - 每次调用 reset 都会重新初始化环境状态
            - 系统提示可以从配置中自定义，否则使用默认值
            - 问题从任务样本中加载，如果不存在则使用默认值
        """
        # 从任务样本中获取问题，如果不存在则使用默认值
        question = self.task_sample.get("question", "What is the answer?")
        
        # 从配置中获取系统提示，如果不存在则使用默认值
        system_prompt = self.config.get(
            "system_prompt",
            "You answer questions and may use search tools.",
        )
        
        # 初始化对话消息列表，包含系统提示和用户问题
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=question),
        ]
        
        # 返回初始观察（包含消息列表）
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行动作并返回下一个观察、奖励、完成标志和信息。
        
        该方法处理代理的回答动作：
        1. 添加助手回答到消息列表
        2. 检查答案是否匹配目标答案
        3. 根据匹配结果给予奖励
        4. 立即结束 episode（单轮交互）
        
        Args:
            action: 动作字典，包含：
                - "content": 助手生成的回答文本
        
        Returns:
            tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 包含四个元素的元组：
                - 观察字典：包含更新后的消息列表
                - 奖励值：1.0（答案正确）或 0.0（答案错误）
                - 完成标志：始终为 True（单轮交互）
                - 信息字典：空字典（当前未使用）
        
        Examples:
            >>> action = {"content": "Python is a programming language"}
            >>> obs, reward, done, info = env.step(action)
            >>> print(f"Reward: {reward}, Done: {done}")  # Reward: 1.0, Done: True
        
        Note:
            - 这是单轮交互，回答后立即结束
            - 答案匹配是简单的字符串包含检查（不区分大小写）
            - 如果没有目标答案，奖励始终为 0.0
        """
        # 步骤1: 将助手回答添加到消息列表
        self.messages.append(Message(role="assistant", content=action.get("content", "")))
        
        # 步骤2: 单轮交互，回答后立即结束
        done = True
        
        # 步骤3: 检查答案是否匹配，匹配则给予奖励 1.0，否则为 0.0
        reward = 1.0 if self._answer_matches() else 0.0
        
        # 步骤4: 返回观察、奖励、完成标志和信息
        return {"messages": self.messages}, reward, done, {}

    def _answer_matches(self) -> bool:
        """
        检查助手回答是否匹配目标答案。
        
        该方法使用简单的字符串包含检查（不区分大小写）来验证答案。
        如果目标答案包含在助手回答中（或相反），则认为匹配。
        
        Returns:
            bool: True 表示答案匹配，False 表示不匹配
        
        Examples:
            >>> if env._answer_matches():
            ...     print("Answer is correct!")
        
        Note:
            - 匹配检查不区分大小写
            - 如果任务样本中没有目标答案，返回 False
            - 使用简单的字符串包含检查，可能不够精确
        """
        # 从任务样本中获取目标答案
        target = self.task_sample.get("answer")
        
        # 如果没有目标答案，无法验证，返回 False
        if not target:
            return False
        
        # 检查目标答案（小写）是否包含在助手回答（小写）中
        # 使用不区分大小写的匹配
        return target.lower() in self.messages[-1].content.lower()


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["SearchEnv"]

