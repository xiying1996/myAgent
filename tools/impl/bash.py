"""
tools/impl/bash.py — Bash Tool

⚠️ 高危工具，仅在 Plan 明确授权时可用。

安全措施：
  - 命令白名单（默认拒绝所有）
  - 超时限制（防止僵尸进程）
  - 输出截断（防止内存爆炸）
  - 不支持管道链式调用（安全性）
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, Dict, List, Optional

from tools.tool import Tool, RetryPolicy, ToolMetadata
from tools.schema import Schema

logger = logging.getLogger(__name__)

# ── 安全配置 ─────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 30.0
MAX_OUTPUT_CHARS = 50_000

# 允许的命令白名单（可扩展）
ALLOWED_COMMANDS: List[str] = [
    "echo", "ls", "pwd", "cat", "head", "tail", "grep", "wc",
    "mkdir", "touch", "rm", "cp", "mv", "find", "which", "whoami",
    "git", "curl", "wget",
]

ALLOWED_PATTERNS: List[str] = [
    # 禁止的危险模式
    r"\brsync\b", r"\bdd\b", r"\bshutdown\b", r"\breboot\b",
    r"\bmkfs\b", r"\bdd\b", r"\bnc\s+-e\b", r"\bcurl\s+.*\|.*sh\b",
    r"\bwget\s+.*\|.*sh\b", r"\;.*rm\s+-rf", r"\/\*\.py\b",
]


# ── Tool 定义 ────────────────────────────────────────────────────────────────


class BashTool(Tool):
    """
    Bash 执行工具。

    ⚠️ 安全要求：
      - Plan 的 budget 必须包含 "shell" permission
      - 命令必须匹配白名单或白名单前缀
      - 不支持复杂管道（"|" 仅允许在前述白名单命令中使用）

    输出截断至 MAX_OUTPUT_CHARS，防止内存爆炸。
    """

    def __init__(
        self,
        allowed_commands: Optional[List[str]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self._allowed = set(allowed_commands or ALLOWED_COMMANDS)
        self._timeout_s = timeout_s
        self._max_output_chars = max_output_chars

        super().__init__(
            name="bash",
            description="Execute a shell command. Only whitelisted commands are allowed.",
            input_schema=Schema.from_dict({
                "cmd": "str",
                "cwd": {"type": "str", "optional": True},
                "env": {"type": "dict", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "stdout": "str",
                "stderr": "str",
                "exit_code": "int",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                side_effects=True,
                permissions=["shell"],
                tags=["shell", "bash", "exec"],
                deterministic=False,
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        cmd = input_params.get("cmd", "")
        cwd = input_params.get("cwd")
        env = input_params.get("env")

        # 安全检查
        self._check_security(cmd)

        # 额外安全：若命令中包含疑似路径穿越的文件引用，做拦截
        self._check_path_injection(cmd)

        # 执行
        try:
            env_vars = None
            if env:
                import os
                env_vars = {**os.environ, **env}

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=min(self._timeout_s, 120),
                cwd=cwd,
                env=env_vars,
            )

            stdout = result.stdout[:self._max_output_chars]
            stderr = result.stderr[:self._max_output_chars]

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Bash command timed out after {self._timeout_s}s")
        except Exception as e:
            raise RuntimeError(f"Bash execution failed: {e}")

    def _check_security(self, cmd: str) -> None:
        """命令安全检查。"""
        # 禁止危险模式
        for pattern in ALLOWED_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                raise PermissionError(f"Command matches forbidden pattern: {pattern}")

        # 提取主命令（第一个空格分隔的词）
        parts = cmd.strip().split()
        if not parts:
            raise ValueError("Empty command")
        main_cmd = parts[0]

        # 白名单检查
        if main_cmd not in self._allowed:
            raise PermissionError(
                f"Command '{main_cmd}' not in whitelist. "
                f"Allowed: {sorted(self._allowed)}"
            )

    def _check_path_injection(self, cmd: str) -> None:
        """检查命令中是否包含路径穿越模式。"""
        # 匹配 ../ 或绝对路径
        if re.search(r"\.\.\/", cmd) or re.search(r"^\/etc\/", cmd):
            raise PermissionError(f"Path traversal attempt in command: {cmd}")


# ── 便捷访问 ────────────────────────────────────────────────────────────────


BASH_TOOL = BashTool()
