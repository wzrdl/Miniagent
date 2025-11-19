"""
MiniMind 因果语言模型的策略包装器实现。

该模块提供了 BasePolicy 的具体实现，将 MiniMind 语言模型封装为可用于
强化学习训练的策略网络。主要功能包括：

1. 模型加载和初始化：从预训练路径加载模型和分词器
2. 前向传播：计算 token 级别的 logits 和 logprobs
3. 动作生成：生成文本响应并解析工具调用
4. 概率计算：重新计算响应序列的对数概率

主要组件：
- MiniMindPolicyConfig: 策略配置数据类
- MiniMindPolicy: 策略网络实现，继承自 BasePolicy

设计特点：
- 支持聊天模板格式化
- 支持工具调用解析
- 支持多种生成参数（温度、top_p 等）
- 提供完整的元数据支持（logprobs、token IDs 等）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import json
import threading

import torch
from transformers import AutoTokenizer

from agent_rl.agent.chat_template import ChatTemplate
from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.core.policy_base import BasePolicy, PolicyOutput
from model.model_minimind import MiniMindForCausalLM


@dataclass
class MiniMindPolicyConfig:
    """
    MiniMind 策略网络的配置数据类。
    
    该数据类使用 @dataclass 装饰器，自动生成 __init__、__repr__ 等方法。
    包含策略网络初始化所需的所有配置参数。
    
    Attributes:
        model_path: str
            模型路径，包含预训练的 MiniMind 模型权重和分词器文件
            例如: "model/minimind-7b" 或 "/path/to/model"
        
        device: str, default="cpu"
            计算设备，指定模型运行在哪个设备上
            可选值: "cpu", "cuda:0", "cuda:1" 等
        
        dtype: torch.dtype, default=torch.float32
            模型数据类型，影响内存占用和计算精度
            常用值: torch.float32, torch.bfloat16, torch.float16
        
        max_new_tokens: int, default=128
            生成时的最大新 token 数量
            控制生成文本的最大长度，防止无限生成
        
        temperature: float, default=0.7
            采样温度，控制生成的随机性
            - 值越大（>1.0），生成越随机、越多样化
            - 值越小（<1.0），生成越确定、越保守
            - 值为 0 时使用贪婪解码（总是选择概率最高的 token）
        
        top_p: float, default=0.95
            核采样（nucleus sampling）参数，控制采样的多样性
            - 从累积概率达到 top_p 的 token 集合中采样
            - 值越大，候选 token 越多，生成越多样
            - 值越小，候选 token 越少，生成越保守
        
        do_sample: bool, default=True
            是否使用采样生成
            - True: 使用采样（考虑 temperature 和 top_p）
            - False: 使用贪婪解码（总是选择概率最高的 token）
    
    Examples:
        >>> config = MiniMindPolicyConfig(
        ...     model_path="model/minimind-7b",
        ...     device="cuda:0",
        ...     dtype=torch.bfloat16,
        ...     max_new_tokens=256,
        ...     temperature=0.8,
        ...     top_p=0.9
        ... )
        >>> policy = MiniMindPolicy(config)
    
    Note:
        - 所有参数都有默认值，只有 model_path 是必需的
        - 使用 dataclass 可以方便地创建和传递配置
        - 配置对象是不可变的（除非显式修改字段）
    """
    # 模型路径（必需参数）
    model_path: str
    # 计算设备，默认为 CPU
    device: str = "cpu"
    # 模型数据类型，默认为 float32
    dtype: torch.dtype = torch.float32
    # 最大生成 token 数，默认 128
    max_new_tokens: int = 128
    # 采样温度，默认 0.7（中等随机性）
    temperature: float = 0.7
    # 核采样参数，默认 0.95（较多样）
    top_p: float = 0.95
    # 是否使用采样，默认 True
    do_sample: bool = True


class MiniMindPolicy(BasePolicy):
    """
    MiniMind 策略网络实现，适配器模式，为轨迹收集工作器提供 forward/generate API。
    
    该类是 BasePolicy 的具体实现，将 MiniMind 语言模型封装为可用于强化学习
    训练的策略网络。主要功能包括：
    
    1. 模型初始化：加载预训练模型和分词器
    2. 前向传播：计算 token 级别的 logits 和 logprobs
    3. 动作生成：生成文本响应并解析工具调用
    4. 概率计算：重新计算响应序列的对数概率
    
    主要组件：
    - model: MiniMindForCausalLM 模型实例
    - tokenizer: 分词器，用于文本编码和解码
    - chat_template: 聊天模板，用于格式化对话消息
    - tool_parser: 工具解析器，用于解析工具调用
    
    Note:
        - 继承自 BasePolicy，必须实现 forward 和 generate_action 方法
        - 支持批处理，但当前实现主要针对单样本
        - 自动处理 pad_token，如果未设置则使用 eos_token
    """

    def __init__(self, config: MiniMindPolicyConfig) -> None:
        """
        初始化 MiniMind 策略网络。
        
        该初始化方法会：
        1. 加载分词器
        2. 设置 pad_token（如果未设置则使用 eos_token）
        3. 加载预训练模型
        4. 将模型移动到指定设备并设置数据类型
        5. 初始化聊天模板和工具解析器
        
        Args:
            config: MiniMindPolicyConfig 配置对象，包含所有初始化参数
        
        Raises:
            FileNotFoundError: 如果模型路径不存在
            OSError: 如果无法加载模型或分词器
        """
        super().__init__()
        # 保存配置对象
        self.config = config
        
        # 加载分词器，use_fast=False 确保兼容性
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
        
        # 如果分词器没有 pad_token，使用 eos_token 作为 pad_token
        # 这是必要的，因为生成过程需要 pad_token 来处理不同长度的序列
        if self.tokenizer.pad_token_id is None:
            # 生成需要 pad_token；如果未指定，则回退到 EOS token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载预训练的 MiniMind 因果语言模型
        self.model = MiniMindForCausalLM.from_pretrained(config.model_path)
        
        # 将模型移动到指定设备（CPU 或 GPU）并设置数据类型
        self.model.to(config.device, dtype=config.dtype)
        
        # 初始化聊天模板，用于格式化对话消息
        self.chat_template = ChatTemplate()
        
        # 初始化工具解析器，用于解析工具调用
        self.tool_parser = ToolParser()
        # 生成锁：当在多线程环境中调用 generate_action 时，确保模型推理串行化。
        self._gen_lock = threading.Lock()

    def forward(self, messages: Sequence[Message], **_) -> PolicyOutput:
        """
        计算当前对话前缀的 token 级别 logits 和 logprobs。
        
        这是策略网络的核心前向传播方法，用于：
        - 计算策略在给定对话上下文下的输出分布
        - 为强化学习训练器提供梯度信息
        - 支持策略梯度算法的计算
        
        该方法处理流程：
        1. 使用聊天模板将消息序列编码为提示文本
        2. 将提示文本分词为 token IDs
        3. 通过模型前向传播获取 logits
        4. 提取最后一个位置的 logits（用于预测下一个 token）
        5. 计算对数概率分布
        
        Args:
            messages: 消息序列，每个消息是 Message 对象
                包含对话历史，例如用户消息和助手消息
            **_: 其他关键字参数（当前未使用，保留以兼容基类接口）
        
        Returns:
            PolicyOutput: 包含以下字段的命名元组：
                - logprobs: 最后一个位置的对数概率，形状为 (batch_size, vocab_size)
                - logits: 最后一个位置的 logits，形状为 (batch_size, vocab_size)
                - values: None（当前实现不提供值函数估计）
                - metadata: 包含原始提示文本的字典
        
        Note:
            - 只返回最后一个位置的 logits/logprobs，用于预测下一个 token
            - 当前实现不提供值函数估计（values=None）
            - 支持批处理，但主要针对单样本场景
            - 在训练模式下会计算梯度，在评估模式下可以使用 @torch.no_grad()
        
        Examples:
            >>> messages = [Message(role="user", content="写一个函数")]
            >>> output = policy.forward(messages)
            >>> next_token_probs = torch.exp(output.logprobs)  # 转换为概率
        """
        # 步骤1: 使用聊天模板将消息序列编码为格式化的提示文本
        prompt = self.chat_template.encode(messages)
        
        # 步骤2: 将提示文本分词为 token IDs 和 attention mask
        tokenized = self._tokenize(prompt)
        
        # 步骤3: 通过模型前向传播，获取所有位置的 logits
        outputs = self.model(**tokenized)
        
        # 步骤4: 提取最后一个位置的 logits，形状为 (batch_size, vocab_size)
        # 这表示模型对下一个 token 的预测分布
        logits = outputs.logits[:, -1, :]
        
        # 步骤5: 计算对数概率分布（log softmax）
        # 用于策略梯度计算，表示模型对每个可能 token 的对数概率
        logprobs = torch.log_softmax(logits, dim=-1)
        
        # 返回 PolicyOutput，包含 logprobs、logits 和元数据
        # values=None 表示当前实现不提供值函数估计
        return PolicyOutput(
            logprobs=logprobs,
            logits=logits,
            values=None,  # 当前实现不提供值函数
            metadata={"prompt": prompt}  # 保存原始提示文本用于调试
        )

    @torch.no_grad()
    def generate_action(
        self,
        messages: Sequence[Dict[str, Any] | Message],
        max_new_tokens: int | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        生成助手响应以及可选的工具调用标记。
        
        这是策略网络的动作生成方法，用于在环境中执行策略。该方法：
        1. 将消息序列格式化为提示文本
        2. 使用模型进行自回归生成
        3. 解码生成的 token IDs 为文本
        4. 解析工具调用（如果存在）
        5. 收集生成 token 的对数概率
        
        使用 @torch.no_grad() 装饰器，因为在推理时不需要计算梯度。
        
        Args:
            messages: 消息序列，可以是字典或 Message 对象
                每个消息包含 "role" 和 "content" 字段
                示例: [{"role": "user", "content": "计算 1+1"}]
            
            max_new_tokens: int | None, optional
                最大生成 token 数，如果为 None 则使用配置中的默认值
                控制生成文本的最大长度
            
            **gen_kwargs: 其他生成参数，可能包括：
                - temperature: 采样温度（覆盖配置中的值）
                - top_p: 核采样参数（覆盖配置中的值）
                - do_sample: 是否使用采样（覆盖配置中的值）
                - 其他模型特定的生成参数
        
        Returns:
            Dict[str, Any]: 包含生成结果的字典：
                - "content": 生成的文本响应（字符串）
                - "tool_call": 工具调用标记（如果有），格式为 "<tool_call>...</tool_call>"
                - "metadata": 元数据字典，包含：
                    - "prompt_text": 原始提示文本
                    - "response_ids": 生成的 token IDs（CPU 张量）
                    - "response_logprobs": 每个生成 token 的对数概率（CPU 张量）
        
        Examples:
            >>> messages = [{"role": "user", "content": "写一个 Python 函数"}]
            >>> action = policy.generate_action(messages, max_new_tokens=100)
            >>> print(action["content"])  # 输出生成的文本
            >>> if action["tool_call"]:
            ...     print("包含工具调用:", action["tool_call"])
        
        Note:
            - 使用 @torch.no_grad() 禁用梯度计算，提高推理速度
            - 生成过程是自回归的，逐个 token 生成
            - 自动处理生成停止条件（遇到 EOS token 或达到最大长度）
            - 所有返回的张量都移动到 CPU，便于序列化和存储
        """
        # 步骤1: 规范化消息格式，确保所有消息都是 Message 对象
        normalized_messages = self._normalize_messages(messages)
        
        # 步骤2: 使用聊天模板将消息序列编码为格式化的提示文本
        prompt = self.chat_template.encode(normalized_messages)
        
        # 步骤3: 将提示文本分词为 token IDs 和 attention mask
        tokenized = self._tokenize(prompt)
        
        # 步骤4: 使用模型进行自回归生成
        # output_scores=True 和 return_dict_in_generate=True 用于获取生成分数
        with self._gen_lock:
            generation = self.model.generate(
                **tokenized,  # 传入 input_ids 和 attention_mask
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,  # 最大生成长度
                temperature=gen_kwargs.get("temperature", self.config.temperature),  # 采样温度
                top_p=gen_kwargs.get("top_p", self.config.top_p),  # 核采样参数
                do_sample=gen_kwargs.get("do_sample", self.config.do_sample),  # 是否采样
                pad_token_id=self.tokenizer.pad_token_id,  # 填充 token ID
                eos_token_id=self.tokenizer.eos_token_id,  # 结束 token ID
                output_scores=True,  # 输出每个步骤的分数（用于计算 logprobs）
                return_dict_in_generate=True,  # 返回字典格式（包含 scores）
            )
        
        # 步骤5: 提取生成的 token IDs（排除提示部分）
        prompt_length = tokenized["input_ids"].shape[-1]  # 提示的长度
        # generation.sequences 包含完整的序列（提示+生成），我们只需要生成的部分
        completion_ids = generation.sequences[0, prompt_length:]  # 只取生成的部分
        
        # 步骤6: 将 token IDs 解码为文本，跳过特殊 token 并去除首尾空白
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        
        # 步骤7: 从生成的文本中提取工具调用（如果存在）
        tool_call = self._extract_tool_call(decoded)
        
        # 步骤8: 收集每个生成 token 的对数概率
        response_logprobs = self._gather_generated_logprobs(generation.scores, completion_ids)
        
        # 步骤9: 构建元数据字典，包含提示文本、响应 IDs 和 logprobs
        # 所有张量都移动到 CPU，便于序列化和存储
        metadata = {
            "prompt_text": prompt,  # 原始提示文本
            "response_ids": completion_ids.detach().cpu(),  # 生成的 token IDs
            "response_logprobs": response_logprobs.detach().cpu(),  # 每个 token 的 logprobs
        }
        
        # 返回包含内容、工具调用和元数据的字典
        return {
            "content": decoded,  # 生成的文本内容
            "tool_call": tool_call,  # 工具调用标记（如果有）
            "metadata": metadata  # 元数据
        }

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        将提示文本分词为 token IDs 和 attention mask。
        
        这是一个辅助方法，用于将字符串提示转换为模型输入格式。
        
        Args:
            prompt: 输入提示文本（字符串）
        
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典：
                - "input_ids": token IDs，形状为 (1, seq_len)
                - "attention_mask": 注意力掩码，形状为 (1, seq_len)
                所有张量都在配置指定的设备上
        
        Note:
            - 使用 return_tensors="pt" 返回 PyTorch 张量
            - 自动将张量移动到配置指定的设备（CPU 或 GPU）
            - 返回的是单样本格式（batch_size=1）
        """
        # 使用分词器编码提示文本，返回 PyTorch 张量
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        return encoded

    @staticmethod
    def _normalize_messages(messages: Sequence[Dict[str, Any] | Message]) -> List[Message]:
        """
        确保所有消息都是 Message 对象格式。
        
        这是一个静态辅助方法，用于统一消息格式。如果消息是字典格式，
        则转换为 Message 对象；如果已经是 Message 对象，则直接使用。
        
        Args:
            messages: 消息序列，可以是字典或 Message 对象
                字典格式应包含：
                - "role": 角色（"user", "assistant", "system" 等）
                - "content": 消息内容（字符串）
                - "tool_name": 工具名称（可选）
        
        Returns:
            List[Message]: Message 对象列表，所有消息都已规范化
        
        Examples:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> normalized = MiniMindPolicy._normalize_messages(messages)
            >>> isinstance(normalized[0], Message)
            True
        
        Note:
            - 这是静态方法，不依赖于实例状态
            - 如果消息已经是 Message 对象，则直接添加到列表
            - 如果消息是字典，则创建新的 Message 对象
        """
        normalized: list[Message] = []
        # 遍历消息序列，规范化每个消息
        for msg in messages:
            if isinstance(msg, Message):
                # 如果已经是 Message 对象，直接添加
                normalized.append(msg)
            else:
                # 如果是字典，转换为 Message 对象
                # tool_name 是可选的，使用 .get() 方法安全获取
                normalized.append(
                    Message(
                        role=msg["role"],
                        content=msg["content"],
                        tool_name=msg.get("tool_name")
                    )
                )
        return normalized

    def _extract_tool_call(self, text: str) -> str | None:
        """
        从文本中提取第一个 <tool_call>...</tool_call> 块（如果存在）。
        
        该方法用于解析生成的文本中的工具调用标记。工具调用通常以
        XML 风格的标签包裹，例如：<tool_call>{"name": "code_exec", ...}</tool_call>
        
        Args:
            text: 要解析的文本（字符串）
        
        Returns:
            str | None: 
                - 如果找到有效的工具调用块，返回完整的标记字符串
                - 如果未找到或解析失败，返回 None
        
        Examples:
            >>> text = "我需要执行代码 <tool_call>{\"name\": \"code_exec\"}</tool_call>"
            >>> tool_call = policy._extract_tool_call(text)
            >>> print(tool_call)  # '<tool_call>{"name": "code_exec"}</tool_call>'
        
        Note:
            - 使用 ToolParser 验证工具调用的 JSON 格式是否有效
            - 只返回第一个匹配的工具调用块
            - 如果 JSON 格式无效，返回 None（即使标签存在）
            - 捕获 ValueError 和 JSONDecodeError 异常，确保健壮性
        """
        # 快速检查：如果文本中不包含工具调用标记，直接返回 None
        if "<tool_call>" not in text:
            return None
        
        try:
            # 使用 ToolParser 验证工具调用的 JSON 格式是否有效
            # 这确保返回的工具调用是格式正确的
            self.tool_parser.parse(text)
            
            # 找到工具调用块的起始和结束位置
            start = text.index("<tool_call>")
            end = text.index("</tool_call>") + len("</tool_call>")
            
            # 返回完整的工具调用标记字符串
            return text[start:end]
        except (ValueError, json.JSONDecodeError):  # type: ignore[attr-defined]
            # 如果解析失败（JSON 格式错误或找不到结束标签），返回 None
            return None

    def compute_sequence_logprob(self, prompt: str, response_ids: torch.Tensor) -> torch.Tensor:
        """
        在当前策略下重新计算响应的对数概率。
        
        该方法用于计算给定响应序列在策略网络下的对数概率。这在强化学习
        训练中很重要，因为需要计算旧策略和新策略的概率比（importance sampling）。
        
        处理流程：
        1. 将提示和响应拼接为完整序列
        2. 通过模型前向传播获取所有位置的 logits
        3. 计算每个位置的对数概率
        4. 提取响应部分的对数概率
        
        Args:
            prompt: 提示文本（字符串）
            response_ids: 响应的 token IDs，形状为 (response_len,)
                这是要计算概率的响应序列
        
        Returns:
            torch.Tensor: 响应序列中每个 token 的对数概率
                形状为 (response_len,)
                每个元素表示对应 token 的对数概率
        
        Examples:
            >>> prompt = "写一个函数"
            >>> response_ids = torch.tensor([123, 456, 789])  # 示例 token IDs
            >>> logprobs = policy.compute_sequence_logprob(prompt, response_ids)
            >>> print(logprobs.shape)  # torch.Size([3])
        
        Note:
            - 用于计算策略概率比，支持重要性采样
            - 响应序列应该是在生成时产生的 token IDs
            - 返回的对数概率是每个 token 在其上下文下的条件概率
            - 用于 PPO、GRPO 等算法的策略更新
        """
        # 步骤1: 将提示文本分词为 token IDs
        prompt_encoding = self._tokenize(prompt)
        prompt_ids = prompt_encoding["input_ids"]
        
        # 步骤2: 将响应 IDs 添加 batch 维度，然后与提示拼接
        response_ids = response_ids.unsqueeze(0)  # (1, response_len)
        concat_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (1, prompt_len + response_len)
        
        # 步骤3: 创建注意力掩码（全1，表示所有位置都参与计算）
        attention_mask = torch.ones_like(concat_ids, device=self.config.device)
        
        # 步骤4: 通过模型前向传播，获取所有位置的 logits
        outputs = self.model(input_ids=concat_ids, attention_mask=attention_mask)
        
        # 步骤5: 提取除最后一个位置外的所有 logits
        # logits[i] 表示在位置 i 的 token 条件下，位置 i+1 的 token 分布
        logits = outputs.logits[:, :-1, :]  # (1, prompt_len + response_len - 1, vocab_size)
        
        # 步骤6: 计算对数概率分布
        logprobs = torch.log_softmax(logits, dim=-1)  # (1, seq_len - 1, vocab_size)
        
        # 步骤7: 提取目标 token IDs（从位置1开始，因为位置0是提示的最后一个token）
        target_ids = concat_ids[:, 1:]  # (1, seq_len - 1)
        
        # 步骤8: 使用 gather 提取每个目标 token 的对数概率
        # 对于每个位置，从 logprobs 中提取对应 token 的概率
        token_logprobs = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        # token_logprobs 形状: (1, seq_len - 1)
        
        # 步骤9: 只返回响应部分的对数概率（排除提示部分）
        response_len = response_ids.shape[-1]
        return token_logprobs[:, -response_len:].squeeze(0)  # (response_len,)

    def _gather_generated_logprobs(self, scores: List[torch.Tensor], completion_ids: torch.Tensor) -> torch.Tensor:
        """
        将生成步骤的 logits 转换为 token 级别的对数概率。
        
        该方法用于从模型生成过程中收集的分数（scores）中提取每个生成 token
        的对数概率。这在生成时已经计算好了，只需要提取对应 token 的概率。
        
        Args:
            scores: 生成过程中每个步骤的 logits 列表
                每个元素是一个张量，形状通常为 (1, vocab_size)
                表示该步骤所有可能 token 的分数
            completion_ids: 生成的 token IDs，形状为 (completion_len,)
                这是实际生成的 token 序列
        
        Returns:
            torch.Tensor: 每个生成 token 的对数概率
                形状为 (completion_len,)
                每个元素表示对应生成 token 的对数概率
        
        Examples:
            >>> scores = [torch.randn(1, 1000), torch.randn(1, 1000)]  # 2个步骤
            >>> completion_ids = torch.tensor([123, 456])  # 生成的 token IDs
            >>> logprobs = policy._gather_generated_logprobs(scores, completion_ids)
            >>> print(logprobs.shape)  # torch.Size([2])
        
        Note:
            - scores 和 completion_ids 的长度应该相同
            - 每个步骤的 logits 需要先转换为对数概率
            - 然后提取对应 token 的概率
            - 用于保存生成时的概率信息，供后续分析使用
        """
        logprob_values: List[torch.Tensor] = []
        
        # 遍历每个生成步骤，提取对应 token 的对数概率
        for step_scores, token_id in zip(scores, completion_ids):
            # 步骤1: 将 logits 转换为对数概率分布
            step_logprob = torch.log_softmax(step_scores, dim=-1)[0, token_id]
            # step_logprob 是标量，表示该步骤生成该 token 的对数概率
            
            # 步骤2: 将提取的对数概率添加到列表
            logprob_values.append(step_logprob)
        
        # 将所有步骤的对数概率堆叠为一个张量
        return torch.stack(logprob_values)  # (completion_len,)


# 定义模块的公共 API，控制 from module import * 时导入的内容
__all__ = ["MiniMindPolicy", "MiniMindPolicyConfig"]

