"""
Python code execution tool (local subprocess prototype).
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any

from agent_rl.tools.base_tool import BaseTool


class CodeExecTool(BaseTool):
    name = "exec_code"
    description = "Execute Python code safely with a timeout constraint."

    def __init__(self, python_executable: str = "python", workdir: str | None = None) -> None:
        self.python_executable = python_executable
        self.workdir = Path(workdir) if workdir else None

    def __call__(self, code: str, timeout: float = 5.0, **_: Any) -> str:
        safe_code = textwrap.dedent(code)
        cmd = [self.python_executable, "-c", safe_code]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workdir,
            )
        except subprocess.TimeoutExpired:
            return "Execution timed out."
        except Exception as exc:  # pragma: no cover - defensive
            return f"Execution failed: {exc}"

        output = result.stdout.strip()
        error = result.stderr.strip()
        if error and output:
            return f"stdout:\n{output}\n\nstderr:\n{error}"
        if error:
            return f"stderr:\n{error}"
        return output or "[no output]"


__all__ = ["CodeExecTool"]

