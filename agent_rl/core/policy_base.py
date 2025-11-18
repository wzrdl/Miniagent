"""
策略抽象基类，用于包装 MiniMind 语言模型以支持强化学习训练。

该模块定义了策略网络的标准接口，包括：
- PolicyOutput: 策略输出的数据结构，包含 logits、logprobs、values 等
- BasePolicy: 抽象基类，定义了策略网络必须实现的方法

主要用途：
1. 为强化学习训练器提供统一的策略接口
2. 封装语言模型的生成和推理功能
3. 支持策略梯度算法（如 GRPO）所需的 token 级别信息

设计模式：
- 使用抽象基类（ABC）定义接口契约
- 子类（如 MiniMindPolicy）实现具体的模型逻辑
- 通过 PolicyOutput 统一返回格式，便于训练器处理
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Sequence

import torch
from torch import nn


class PolicyOutput(NamedTuple):
    """
    策略输出的数据结构，由 :class:`BasePolicy.forward` 方法产生。
    
    这是一个命名元组（NamedTuple），用于统一策略网络前向传播的返回格式。
    强化学习训练器（如 GRPOTrainer）需要这些信息来计算策略梯度、优势函数等。
    
    Attributes:
        logprobs: torch.Tensor
            每个 token 的对数概率，形状通常为 (batch_size, sequence_length)
            用于计算策略梯度，表示模型对生成每个 token 的置信度
            在强化学习中，用于计算 log π(a|s)，即策略在状态 s 下选择动作 a 的对数概率
        
        logits: torch.Tensor
            未归一化的 logits，形状通常为 (batch_size, sequence_length, vocab_size)
            模型输出的原始分数，经过 softmax 后得到概率分布
            用于计算 logprobs 和进行采样
        
        values: torch.Tensor | None, optional
            状态值函数 V(s) 的估计，形状通常为 (batch_size, sequence_length)
            用于计算优势函数 A(s,a) = Q(s,a) - V(s)
            如果为 None，表示该策略不提供值函数估计（仅策略网络，无价值网络）
        
        metadata: Dict[str, Any] | None, optional
            额外的元数据字典，可以包含：
            - 生成的文本内容
            - 工具调用信息
            - 注意力权重
            - 其他调试或分析信息
            如果为 None，表示没有额外元数据
    
    Examples:
        >>> output = policy.forward(input_ids, attention_mask)
        >>> loss = -output.logprobs.mean()  # 简单的负对数似然损失
        >>> probs = torch.softmax(output.logits, dim=-1)  # 转换为概率分布
        >>> if output.values is not None:
        ...     advantage = rewards - output.values  # 计算优势
    
    Note:
        - 使用 NamedTuple 而不是普通类，因为它是不可变的，适合作为函数返回值
        - logprobs 和 logits 是必需的，values 和 metadata 是可选的
        - 所有张量应该在相同的设备上（CPU 或 GPU）
    """
    # 每个 token 的对数概率，用于策略梯度计算
    logprobs: torch.Tensor
    # 未归一化的 logits，用于计算概率分布
    logits: torch.Tensor
    # 状态值函数估计（可选），用于优势计算
    values: torch.Tensor | None = None
    # 额外的元数据字典（可选），用于存储调试信息等
    metadata: Dict[str, Any] | None = None


class BasePolicy(nn.Module, ABC):
    """
    基于 MiniMind 语言模型的策略网络抽象基类。
    
    该类定义了策略网络的标准接口，所有具体的策略实现（如 MiniMindPolicy）
    都必须继承此类并实现抽象方法。该类同时继承自 nn.Module 和 ABC，使其：
    - 可以作为 PyTorch 模块使用（支持参数管理、设备移动等）
    - 强制子类实现抽象方法（保证接口一致性）
    
    主要职责：
    1. 提供统一的前向传播接口（forward），返回 PolicyOutput
    2. 提供动作生成接口（generate_action），用于在环境中执行策略
    3. 管理模型参数，支持梯度计算和优化器更新
    
    设计模式：
    - 模板方法模式：定义接口，子类实现具体逻辑
    - 策略模式：不同的策略实现可以互换使用
    
    Note:
        - 这是一个抽象类，不能直接实例化
        - 子类必须实现 forward 和 generate_action 方法
        - 所有方法都应该支持批处理（batch processing）
    """

    def __init__(self) -> None:
        """
        初始化策略基类。
        
        调用父类 nn.Module 的初始化方法，设置模块的基本属性。
        子类应该在此方法中初始化模型权重、分词器等。
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> PolicyOutput:
        """
        前向传播方法，返回 token 级别的 logits、logprobs 和 values。
        
        这是策略网络的核心方法，用于：
        - 计算策略在给定输入下的输出分布
        - 为强化学习训练器提供必要的梯度信息
        - 支持策略梯度算法的计算
        
        Args:
            *args: 可变位置参数，通常包括：
                - input_ids: 输入 token IDs，形状为 (batch_size, seq_len)
                - attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
                - 其他模型特定的参数
            
            **kwargs: 可变关键字参数，可能包括：
                - labels: 标签（用于监督学习）
                - return_dict: 是否返回字典格式
                - 其他模型生成参数
        
        Returns:
            PolicyOutput: 包含以下字段的命名元组：
                - logprobs: 每个 token 的对数概率
                - logits: 未归一化的 logits
                - values: 状态值函数估计（可选）
                - metadata: 额外元数据（可选）
        
        Note:
            - 这是抽象方法，子类必须实现
            - 应该支持批处理，即可以同时处理多个样本
            - 返回的张量应该在正确的设备上（CPU 或 GPU）
            - 在训练模式下，应该计算梯度；在评估模式下，可以使用 @torch.no_grad()
        
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        pass

    @torch.no_grad()
    def generate_action(
        self,
        messages: Sequence[Dict[str, Any]],
        max_new_tokens: int = 128,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        将聊天消息转换为自回归动作（生成的文本响应）。
        
        该方法用于在环境中执行策略，生成动作（通常是文本响应）。
        使用 @torch.no_grad() 装饰器，因为在推理时不需要计算梯度。
        
        具体子类应该按以下步骤实现：
        1. 使用 MiniMind 的聊天模板将消息列表转换为模型输入格式
        2. 调用底层模型的 generate 方法进行自回归生成
        3. 解码生成的 token IDs 为文本
        4. 返回解码后的响应和元数据（如工具调用、生成 token 的 logprobs 等）
        
        Args:
            messages: 消息序列，每个消息是一个字典，通常包含：
                - "role": 角色（"user", "assistant", "system" 等）
                - "content": 消息内容（字符串）
                示例: [{"role": "user", "content": "写一个 Python 函数"}]
            
            max_new_tokens: 最大生成 token 数，默认 128
                控制生成文本的最大长度，防止无限生成
            
            **gen_kwargs: 其他生成参数，可能包括：
                - temperature: 采样温度（控制随机性）
                - top_p: 核采样参数（控制多样性）
                - top_k: top-k 采样参数
                - do_sample: 是否使用采样（True）或贪婪解码（False）
                - repetition_penalty: 重复惩罚系数
                - 其他模型特定的生成参数
        
        Returns:
            Dict[str, Any]: 包含生成结果的字典，通常包括：
                - "text": 生成的文本响应（字符串）
                - "logprobs": 生成 token 的对数概率（可选）
                - "tool_calls": 工具调用信息（如果有，可选）
                - 其他元数据
        
        Examples:
            >>> messages = [{"role": "user", "content": "计算 1+1"}]
            >>> action = policy.generate_action(messages, max_new_tokens=50)
            >>> print(action["text"])  # 输出生成的文本
        
        Note:
            - 使用 @torch.no_grad() 禁用梯度计算，提高推理速度
            - 子类必须实现此方法，否则会抛出 NotImplementedError
            - 生成过程是自回归的，即逐个 token 生成
            - 应该处理生成停止条件（如遇到 EOS token）
        
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            "generate_action must be implemented by concrete policy wrappers."
        )


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["PolicyOutput", "BasePolicy"]

