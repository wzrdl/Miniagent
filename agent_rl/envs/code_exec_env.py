"""
Environment for code-execution tasks that leverage a sandbox tool.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.envs.base_env import AgentEnv
from agent_rl.tools.registry import ToolRegistry


class CodeExecEnv(AgentEnv):
    """
    Multi-turn environment where the agent can invoke an ``exec_code`` tool.
    """

    def __init__(
        self,
        task_sample: Dict[str, Any],
        config: Dict[str, Any] | None,
        tool_manager: ToolRegistry | None,
    ) -> None:
        super().__init__(task_sample, config, tool_manager)
        self.messages: List[Message] = []
        self.tool_parser = ToolParser()
        self.max_turns = self.config.get("max_turns", 6)
        self.tests: List[Dict[str, Any]] = []
        self._tests_passed = False
        self._latest_test_info: Dict[str, Any] | None = None

    def reset(self) -> Dict[str, Any]:
        system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful coding assistant that can execute Python code via tools.",
        )
        user_prompt = self.task_sample.get("prompt", "Provide a coding task.")
        self.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        self.tests = self.task_sample.get("tests", []) or []
        self._tests_passed = False
        self._latest_test_info = None
        return {"messages": self.messages}

    def step(
        self, action: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assistant_content = action.get("content", "")
        tool_call_str = action.get("tool_call")

        self.messages.append(Message(role="assistant", content=assistant_content))

        info: Dict[str, Any] = {}
        reward = 0.0
        done = False
        # ``test_results`` summarizes automated checks for the latest code run.
        test_results: Dict[str, Any] | None = None

        if tool_call_str:
            if not self.tool_manager:
                raise RuntimeError("Tool manager is required for code execution env.")
            tool_call = self.tool_parser.parse(tool_call_str)
            tool = self.tool_manager.get(tool_call.name)
            tool_result = tool(**tool_call.arguments)
            self.messages.append(
                Message(role="tool", content=tool_result, tool_name=tool_call.name)
            )
            info["tool_result"] = tool_result
            if self.tests and tool_call.name == "exec_code" and "code" in tool_call.arguments:
                code_str = str(tool_call.arguments["code"])
                test_results = self._run_dataset_tests(code_str, tool)
                info["test_results"] = test_results
                self._latest_test_info = test_results
                if test_results["all_passed"]:
                    reward = self._success_reward()
                    done = True
                    self._tests_passed = True

        if not self.tests and self._task_solved():
            # Classic fallback where the agent prints a sentinel phrase.
            reward = self._success_reward()
            done = True
        elif len(self.messages) // 2 >= self.max_turns:
            done = True
            reward = self.config.get("timeout_penalty", 0.0)

        return {"messages": self.messages}, reward, done, info

    def compute_final_reward(self) -> float | None:
        if self.tests and self._tests_passed:
            return self._success_reward()
        if self._task_solved():
            return self._success_reward()
        return 0.0

    def _task_solved(self) -> bool:
        if self.tests:
            return self._tests_passed
        reference = self.task_sample.get("reference_solution")
        if not reference:
            return False
        return reference in self._last_tool_output()

    def _last_tool_output(self) -> str:
        for message in reversed(self.messages):
            if message.role == "tool":
                return message.content
        return ""

    def _run_dataset_tests(self, code: str, tool: Any) -> Dict[str, Any]:
        """
        Execute all available tests using the provided code snippet.
        """

        cases: List[Dict[str, Any]] = []
        passed_count = 0
        for idx, test_case in enumerate(self.tests):
            stdin_payload = test_case.get("input", "")
            expected = str(test_case.get("output", "")).strip()
            timeout = float(test_case.get("timeout", self.config.get("test_timeout", 5.0)))
            actual = str(tool(code=code, stdin=stdin_payload, timeout=timeout)).strip()
            passed = actual == expected
            passed_count += int(passed)
            cases.append(
                {
                    "index": idx,
                    "description": test_case.get("description"),
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                }
            )
        return {
            "cases": cases,
            "passed_count": passed_count,
            "total": len(self.tests),
            "all_passed": passed_count == len(self.tests),
        }

    def _success_reward(self) -> float:
        """
        Read reward from config (default to 1.0) for successful completion.
        """

        return self.config.get("success_reward", 1.0)


__all__ = ["CodeExecEnv"]

