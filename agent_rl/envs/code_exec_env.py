"""
代码执行任务的环境，利用沙箱工具执行代码。

该模块实现了代码执行任务的强化学习环境，支持：
1. 多轮对话交互
2. 代码执行工具调用
3. 自动化测试验证
4. 任务完成检测

主要组件：
- CodeExecEnv: 代码执行环境，继承自 AgentEnv
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.envs.base_env import AgentEnv
from agent_rl.tools.registry import ToolRegistry


class CodeExecEnv(AgentEnv):
    """
    多轮对话环境，代理可以调用 ``exec_code`` 工具执行代码。
    
    该环境用于训练代码生成和执行任务，支持：
    - 多轮交互：代理可以多次调用工具执行代码
    - 测试验证：自动运行测试用例验证代码正确性
    - 任务完成检测：通过测试或参考解决方案检测任务是否完成
    
    工作流程：
    1. 环境初始化，加载任务和测试用例
    2. 代理生成代码并调用 exec_code 工具
    3. 环境执行代码并运行测试
    4. 根据测试结果给予奖励
    5. 如果所有测试通过或达到最大轮数，结束 episode
    
    Attributes:
        messages: List[Message]
            对话消息列表，包含系统提示、用户提示、助手响应和工具结果
            按时间顺序排列
        
        tool_parser: ToolParser
            工具调用解析器，用于解析代理生成的工具调用字符串
        
        max_turns: int
            最大轮数，防止无限循环
            默认值为 6，表示最多 6 轮交互
        
        tests: List[Dict[str, Any]]
            测试用例列表，每个测试用例包含输入、期望输出等信息
            从任务样本中加载
        
        _tests_passed: bool
            内部标志，表示是否所有测试都已通过
            用于快速检查任务完成状态
        
        _latest_test_info: Dict[str, Any] | None
            最新一次测试的结果信息
            包含测试用例详情、通过数量等
    
    Examples:
        >>> env = CodeExecEnv(
        ...     task_sample={"prompt": "写一个函数", "tests": [...]},
        ...     config={"max_turns": 10},
        ...     tool_manager=tool_registry
        ... )
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(action)
    
    Note:
        - 需要工具管理器来执行代码
        - 支持两种任务完成检测方式：测试用例或参考解决方案
        - 如果达到最大轮数仍未完成，会给予超时惩罚
    """

    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: ToolRegistry | None,
    ) -> None:
        """
        初始化代码执行环境。
        
        Args:
            task_sample: 任务样本字典，包含：
                - "prompt": 任务提示文本
                - "tests": 测试用例列表（可选）
                - "reference_solution": 参考解决方案（可选）
            config: 环境配置字典，包含：
                - "max_turns": 最大轮数（默认: 6）
                - "system_prompt": 系统提示（可选）
                - "success_reward": 成功奖励（默认: 1.0）
                - "timeout_penalty": 超时惩罚（默认: 0.0）
                - "test_timeout": 测试超时时间（默认: 5.0）
            tool_manager: 工具注册表，用于获取和执行代码执行工具
                如果为 None，在调用工具时会抛出错误
        """
        # 调用基类初始化方法
        super().__init__(task_sample, config, tool_manager)
        
        # 初始化对话消息列表（空列表）
        self.messages: List[Message] = []
        
        # 初始化工具调用解析器
        self.tool_parser = ToolParser()
        
        # 从配置中获取最大轮数，默认值为 6
        self.max_turns = self.config.get("max_turns", 6)
        
        # 初始化测试用例列表（将在 reset 中加载）
        self.tests: List[Dict[str, Any]] = []
        
        # 初始化测试通过标志
        self._tests_passed = False
        
        # 初始化最新测试信息（None 表示尚未运行测试）
        self._latest_test_info: Dict[str, Any] | None = None

    def reset(self) -> Dict[str, Any]:
        """
        重置环境到初始状态并返回第一个观察。
        
        该方法会：
        1. 初始化对话消息（系统提示和用户提示）
        2. 加载测试用例
        3. 重置内部状态标志
        
        Returns:
            Dict[str, Any]: 初始观察字典，包含：
                - "messages": 初始消息列表（系统提示和用户提示）
        
        Examples:
            >>> obs = env.reset()
            >>> print(len(obs["messages"]))  # 2 (system + user)
        
        Note:
            - 每次调用 reset 都会重新初始化环境状态
            - 测试用例从任务样本中加载，如果不存在则为空列表
            - 系统提示可以从配置中自定义，否则使用默认值
        """
        # 从配置中获取系统提示，如果不存在则使用默认值
        system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful coding assistant that can execute Python code via tools.",
        )
        
        # 从任务样本中获取用户提示，如果不存在则使用默认值
        user_prompt = self.task_sample.get("prompt", "Provide a coding task.")
        
        # 初始化对话消息列表，包含系统提示和用户提示
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        
        # 从任务样本中加载测试用例，如果不存在则为空列表
        self.tests = self.task_sample.get("tests", []) or []
        
        # 重置测试通过标志
        self._tests_passed = False
        
        # 清空最新测试信息
        self._latest_test_info = None
        
        # 返回初始观察（包含消息列表）
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行动作并返回下一个观察、奖励、完成标志和信息。
        
        该方法处理代理的动作，包括：
        1. 添加助手响应到消息列表
        2. 如果包含工具调用，解析并执行工具
        3. 如果执行了代码，运行测试用例
        4. 根据测试结果或任务完成情况给予奖励
        5. 检查是否达到最大轮数
        
        Args:
            action: 动作字典，包含：
                - "content": 助手生成的文本响应
                - "tool_call": 工具调用字符串（可选），格式为 "<tool_call>...</tool_call>"
        
        Returns:
            tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 包含四个元素的元组：
                - 观察字典：包含更新后的消息列表
                - 奖励值：该步骤获得的奖励（0.0 或成功奖励或超时惩罚）
                - 完成标志：True 表示 episode 结束，False 表示继续
                - 信息字典：包含工具结果、测试结果等额外信息
        
        Raises:
            RuntimeError: 如果工具管理器为 None 但需要执行工具
        
        Examples:
            >>> action = {"content": "我将执行代码", "tool_call": "<tool_call>...</tool_call>"}
            >>> obs, reward, done, info = env.step(action)
            >>> if done:
            ...     print(f"Episode ended with reward: {reward}")
        
        Note:
            - 如果没有测试用例，使用参考解决方案检测任务完成
            - 如果达到最大轮数，会强制结束并给予超时惩罚
            - 测试结果会保存在 info["test_results"] 中
        """
        # 步骤1: 从动作中提取助手响应内容
        assistant_content = action.get("content", "")
        
        # 步骤2: 从动作中提取工具调用字符串（如果存在）
        tool_call_str = action.get("tool_call")
        
        # 步骤3: 将助手响应添加到消息列表
        self.messages.append(Message(role="assistant", content=assistant_content))
        
        # 步骤4: 初始化返回变量
        info: Dict[str, Any] = {}  # 额外信息字典
        reward = 0.0  # 默认奖励为 0
        done = False  # 默认未完成
        # test_results 汇总最新代码运行的自动化检查结果
        test_results: Dict[str, Any] | None = None
        
        # 步骤5: 如果动作包含工具调用，解析并执行
        if tool_call_str:
            # 5.1: 验证工具管理器是否存在
            if not self.tool_manager:
                raise RuntimeError("Tool manager is required for code execution env.")
            
            # 5.2: 解析工具调用字符串
            tool_call = self.tool_parser.parse(tool_call_str)
            
            # 5.3: 从工具管理器获取对应的工具
            tool = self.tool_manager.get(tool_call.name)
            
            # 5.4: 执行工具，传入解析后的参数
            tool_result = tool(**tool_call.arguments)
            
            # 5.5: 将工具结果添加到消息列表
            self.messages.append(
                Message(role="tool", content=tool_result, tool_name=tool_call.name)
            )
            
            # 5.6: 保存工具结果到信息字典
            info["tool_result"] = tool_result
            
            # 步骤6: 如果执行了代码且有测试用例，运行测试
            if self.tests and tool_call.name == "exec_code" and "code" in tool_call.arguments:
                # 6.1: 提取代码字符串
                code_str = str(tool_call.arguments["code"])
                
                # 6.2: 运行所有测试用例
                test_results = self._run_dataset_tests(code_str, tool)
                
                # 6.3: 保存测试结果到信息字典
                info["test_results"] = test_results
                
                # 6.4: 更新最新测试信息
                self._latest_test_info = test_results
                
                # 6.5: 如果所有测试通过，给予成功奖励并结束 episode
                if test_results["all_passed"]:
                    reward = self._success_reward()
                    done = True
                    self._tests_passed = True
        
        # 步骤7: 如果没有测试用例，使用参考解决方案检测任务完成
        if not self.tests and self._task_solved():
            # 经典回退方法：代理打印一个标记短语
            reward = self._success_reward()
            done = True
        # 步骤8: 如果达到最大轮数，强制结束并给予超时惩罚
        elif len(self.messages) // 2 >= self.max_turns:
            # 消息数量除以 2 得到轮数（每轮包含助手和工具两条消息）
            done = True
            reward = self.config.get("timeout_penalty", 0.0)
        
        # 步骤9: 返回观察、奖励、完成标志和信息
        return {"messages": self.messages}, reward, done, info

    def compute_final_reward(self) -> float | None:
        """
        计算 episode 的最终奖励。
        
        该方法在 episode 结束时调用，用于计算最终奖励值。
        优先检查测试是否通过，如果没有测试则检查任务是否解决。
        
        Returns:
            float | None: 最终奖励值
                - 如果任务成功完成，返回成功奖励（通常为 1.0）
                - 如果任务未完成，返回 0.0
                - 如果基类实现返回 None，这里不会返回 None
        
        Examples:
            >>> final_reward = env.compute_final_reward()
            >>> print(f"Final reward: {final_reward}")
        
        Note:
            - 该方法在 episode 结束时由 RolloutWorker 调用
            - 优先使用测试结果，如果没有测试则使用参考解决方案
        """
        # 如果有测试用例且所有测试都通过，返回成功奖励
        if self.tests and self._tests_passed:
            return self._success_reward()
        
        # 如果没有测试用例但任务已解决，返回成功奖励
        if self._task_solved():
            return self._success_reward()
        
        # 否则返回 0.0（任务未完成）
        return 0.0

    def _task_solved(self) -> bool:
        """
        检查任务是否已解决。
        
        该方法使用两种方式检测任务完成：
        1. 如果有测试用例，检查是否所有测试都通过
        2. 如果没有测试用例，检查工具输出是否包含参考解决方案
        
        Returns:
            bool: True 表示任务已解决，False 表示未解决
        
        Examples:
            >>> if env._task_solved():
            ...     print("Task completed!")
        
        Note:
            - 优先使用测试结果
            - 如果没有测试用例和参考解决方案，返回 False
            - 参考解决方案检测是简单的字符串包含检查
        """
        # 如果有测试用例，直接返回测试通过状态
        if self.tests:
            return self._tests_passed
        
        # 如果没有测试用例，尝试使用参考解决方案检测
        reference = self.task_sample.get("reference_solution")
        if not reference:
            # 如果没有参考解决方案，无法检测，返回 False
            return False
        
        # 检查最新的工具输出是否包含参考解决方案
        return reference in self._last_tool_output()

    def _last_tool_output(self) -> str:
        """
        获取最后一次工具调用的输出。
        
        该方法从消息列表中反向查找，返回最后一个工具消息的内容。
        
        Returns:
            str: 最后一次工具调用的输出内容
                如果没有工具调用，返回空字符串
        
        Examples:
            >>> output = env._last_tool_output()
            >>> print(f"Last tool output: {output}")
        
        Note:
            - 从后往前查找，返回第一个找到的工具消息
            - 如果没有工具消息，返回空字符串
        """
        # 从后往前遍历消息列表
        for message in reversed(self.messages):
            # 如果找到工具消息，返回其内容
            if message.role == "tool":
                return message.content
        # 如果没有找到工具消息，返回空字符串
        return ""

    def _run_dataset_tests(self, code: str, tool: Any) -> Dict[str, Any]:
        """
        使用提供的代码片段执行所有可用的测试。
        
        该方法会遍历所有测试用例，对每个测试用例：
        1. 提取输入、期望输出和超时时间
        2. 使用工具执行代码并获取实际输出
        3. 比较实际输出和期望输出
        4. 记录测试结果
        
        Args:
            code: 要测试的代码字符串
            tool: 代码执行工具实例，用于执行代码
        
        Returns:
            Dict[str, Any]: 测试结果字典，包含：
                - "cases": 测试用例结果列表，每个元素包含：
                    - "index": 测试用例索引
                    - "description": 测试用例描述（如果有）
                    - "expected": 期望输出
                    - "actual": 实际输出
                    - "passed": 是否通过（布尔值）
                - "passed_count": 通过的测试用例数量
                - "total": 总测试用例数量
                - "all_passed": 是否所有测试都通过（布尔值）
        
        Examples:
            >>> test_results = env._run_dataset_tests(code, tool)
            >>> print(f"Passed: {test_results['passed_count']}/{test_results['total']}")
        
        Note:
            - 每个测试用例可以有自己的超时时间
            - 如果测试用例没有指定超时时间，使用配置中的默认值
            - 输出比较是精确匹配（去除首尾空白后）
        """
        # 初始化测试用例结果列表和通过计数
        cases: List[Dict[str, Any]] = []
        passed_count = 0
        
        # 遍历所有测试用例
        for idx, test_case in enumerate(self.tests):
            # 步骤1: 提取测试用例的输入
            stdin_payload = test_case.get("input", "")
            
            # 步骤2: 提取期望输出并去除首尾空白
            expected = str(test_case.get("output", "")).strip()
            
            # 步骤3: 提取超时时间，如果未指定则使用配置中的默认值
            timeout = float(test_case.get("timeout", self.config.get("test_timeout", 5.0)))
            
            # 步骤4: 使用工具执行代码，传入输入和超时时间
            actual = str(tool(code=code, stdin=stdin_payload, timeout=timeout)).strip()
            
            # 步骤5: 比较实际输出和期望输出
            passed = actual == expected
            
            # 步骤6: 更新通过计数
            passed_count += int(passed)
            
            # 步骤7: 记录测试用例结果
            cases.append(
                {
                    "index": idx,  # 测试用例索引
                    "description": test_case.get("description"),  # 测试用例描述（如果有）
                    "expected": expected,  # 期望输出
                    "actual": actual,  # 实际输出
                    "passed": passed,  # 是否通过
                }
            )
        
        # 返回测试结果汇总
        return {
            "cases": cases,  # 所有测试用例的详细结果
            "passed_count": passed_count,  # 通过的测试用例数量
            "total": len(self.tests),  # 总测试用例数量
            "all_passed": passed_count == len(self.tests),  # 是否所有测试都通过
        }

    def _success_reward(self) -> float:
        """
        从配置中读取成功完成的奖励值（默认为 1.0）。
        
        该方法返回任务成功完成时的奖励值，可以从配置中自定义。
        
        Returns:
            float: 成功奖励值，默认值为 1.0
        
        Examples:
            >>> reward = env._success_reward()
            >>> print(f"Success reward: {reward}")  # 1.0 (默认)
        
        Note:
            - 默认奖励为 1.0
            - 可以通过配置中的 "success_reward" 自定义
        """
        # 从配置中获取成功奖励，如果不存在则使用默认值 1.0
        return self.config.get("success_reward", 1.0)


__all__ = ["CodeExecEnv"]

