"""
tool.py — Typed Tool 定义

核心类：
  ToolMetadata  — 工具元信息（permissions / side_effects / cost / tags）
  Tool          — 抽象工具基类，子类实现 invoke()
  RetryPolicy   — 重试策略（可绑定到 Tool）

设计原则：
  - Tool.invoke() 返回 ToolResult（标准化结果）
  - Tool 是抽象的，同步/异步执行由子类决定
  - RetryPolicy 内嵌在 Tool 中，也可独立使用
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tools.result import ToolResult, ToolStatus, ToolErrorType
from tools.schema import Schema

logger = logging.getLogger(__name__)


# ── RetryPolicy ────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """
    重试策略。
    绑定到 Tool，ToolExecutor 在执行时参考此策略。
    """
    max_attempts: int = 1
    base_delay_ms: int = 1000
    exponential_base: float = 2.0
    max_delay_ms: int = 30000
    retriable_errors: List[ToolErrorType] = field(default_factory=list)
    # retriable_errors 建议值: [TIMEOUT, NETWORK]

    def should_retry(self, error_type: ToolErrorType) -> bool:
        return error_type in self.retriable_errors

    def delay_ms(self, attempt: int) -> int:
        """计算第 attempt 次重试的延迟（从 1 开始）。"""
        delay = int(self.base_delay_ms * (self.exponential_base ** (attempt - 1)))
        return min(delay, self.max_delay_ms)


# ── ToolMetadata ─────────────────────────────────────────────────────────────


@dataclass
class ToolMetadata:
    """
    工具元信息，用于 routing / 安全 / cost 感知 planner。
    """
    side_effects: bool = False
    permissions: List[str] = field(default_factory=list)
    # permissions 例子: ["network", "file:read", "file:write", "shell"]
    cost_estimate: Optional[float] = None    # 预估 token 成本
    deterministic: bool = False
    tags: List[str] = field(default_factory=list)
    # tags 例子: ["web", "search", "browsing"]


# ── Tool ─────────────────────────────────────────────────────────────────────


class Tool:
    """
    抽象工具基类。

    所有工具（web_search / bash / filesystem 等）都继承 Tool，
    实现 invoke() 方法返回 ToolResult。

    子类可覆盖：
      _do_invoke()  — 实际执行逻辑（可选，默认调用 invoke）
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Schema,
        output_schema: Schema,
        timeout_s: float = 30.0,
        metadata: Optional[ToolMetadata] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.timeout_s = timeout_s
        self.metadata = metadata or ToolMetadata()
        self.retry_policy = retry_policy or RetryPolicy()

    def invoke(
        self,
        input_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        同步调用入口。执行一次，内部处理 retry。

        参数：
          input_params — 工具入参（dict）
          context      — 额外上下文（如 trace_id，request_id）

        返回：
          ToolResult（标准化结果）
        """
        context = context or {}
        last_error: Optional[Exception] = None
        last_error_type: Optional[ToolErrorType] = None
        last_elapsed_ms: Optional[int] = None
        last_result: Optional[ToolResult] = None

        try:
            normalized_input = self.input_schema.validate_and_coerce(input_params)
        except ValueError as e:
            return ToolResult.failure(
                error=f"input schema mismatch: {e}",
                error_type=ToolErrorType.SCHEMA_MISMATCH,
                metadata={"trace_id": context.get("trace_id")},
            )

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            start_ms = int(time.monotonic() * 1000)
            try:
                output = self._do_invoke(normalized_input)
                elapsed_ms = int(time.monotonic() * 1000) - start_ms

                if isinstance(output, ToolResult):
                    result = self._finalize_tool_result(output, elapsed_ms, context, attempt)
                else:
                    normalized_output = self._validate_output(output)
                    result = ToolResult.success(
                        output=normalized_output,
                        exec_time_ms=elapsed_ms,
                        metadata=self._build_metadata(context, attempt),
                    )

                if result.is_success:
                    return result

                last_result = result
                last_error_type = result.error_type or ToolErrorType.UNKNOWN
                last_elapsed_ms = result.exec_time_ms
                if not result.is_retryable or not self.retry_policy.should_retry(last_error_type):
                    return result
            except Exception as e:
                elapsed_ms = int(time.monotonic() * 1000) - start_ms
                last_error = e
                error_type = self._classify_error(e)
                last_error_type = error_type
                last_elapsed_ms = elapsed_ms

                logger.warning(
                    "[Tool:%s] attempt %d/%d 失败: %s (%s)",
                    self.name, attempt, self.retry_policy.max_attempts,
                    e, error_type.value,
                )

                # 不可重试的错误，立即返回
                if not self.retry_policy.should_retry(error_type):
                    return ToolResult.failure(
                        error=str(e),
                        error_type=error_type,
                        exec_time_ms=elapsed_ms,
                        metadata=self._build_metadata(context, attempt),
                    )

            # 可重试：等待后重试
            if attempt < self.retry_policy.max_attempts:
                delay_ms = self.retry_policy.delay_ms(attempt)
                time.sleep(delay_ms / 1000.0)

        # 所有重试均失败
        if last_result is not None:
            return self._clone_result(
                last_result,
                metadata={
                    **last_result.metadata,
                    "attempts": self.retry_policy.max_attempts,
                },
            )

        terminal_metadata = {
            **self._build_metadata(context, self.retry_policy.max_attempts),
            "attempts": self.retry_policy.max_attempts,
        }
        if last_error_type == ToolErrorType.TIMEOUT:
            terminal_metadata.setdefault("timeout_s", self.timeout_s)
            return ToolResult.timeout(
                timeout_s=self.timeout_s,
                metadata=terminal_metadata,
            )

        return ToolResult.failure(
            error=str(last_error),
            error_type=last_error_type or ToolErrorType.EXCEPTION,
            exec_time_ms=last_elapsed_ms,
            metadata=terminal_metadata,
        )

    def _build_metadata(
        self,
        context: Dict[str, Any],
        attempt: int,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"attempt": attempt}
        trace_id = context.get("trace_id")
        if trace_id is not None:
            metadata["trace_id"] = trace_id
        return metadata

    def _validate_output(self, output: Any) -> Dict[str, Any]:
        if not isinstance(output, dict):
            raise ValueError(
                f"output schema mismatch: Tool[{self.name}] 输出必须是 dict，实际为 {type(output).__name__}"
            )
        try:
            return self.output_schema.validate_and_coerce(output)
        except ValueError as e:
            raise ValueError(f"output schema mismatch: {e}") from e

    def _finalize_tool_result(
        self,
        result: ToolResult,
        elapsed_ms: int,
        context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        metadata = {**self._build_metadata(context, attempt), **result.metadata}
        exec_time_ms = result.exec_time_ms if result.exec_time_ms is not None else elapsed_ms

        if result.status == ToolStatus.SUCCESS:
            normalized_output = self._validate_output(result.output)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=normalized_output,
                exec_time_ms=exec_time_ms,
                raw_response=result.raw_response,
                metadata=metadata,
            )

        if result.status == ToolStatus.TIMEOUT:
            metadata.setdefault("timeout_s", self.timeout_s)

        return ToolResult(
            status=result.status,
            output=dict(result.output),
            error=result.error,
            error_type=result.error_type,
            exec_time_ms=exec_time_ms,
            raw_response=result.raw_response,
            metadata=metadata,
        )

    def _clone_result(
        self,
        result: ToolResult,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        return ToolResult(
            status=result.status,
            output=dict(result.output),
            error=result.error,
            error_type=result.error_type,
            exec_time_ms=result.exec_time_ms,
            raw_response=result.raw_response,
            metadata=metadata or dict(result.metadata),
        )

    def _do_invoke(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        子类实现实际的工具调用逻辑。
        默认抛出 NotImplementedError。
        """
        raise NotImplementedError(f"Tool[{self.name}] 未实现 _do_invoke()")

    def _classify_error(self, e: Exception) -> ToolErrorType:
        """
        将异常分类为 ToolErrorType。
        子类可覆盖以提供更精确的分类。
        """
        name = type(e).__name__
        msg = str(e).lower()
        if "timeout" in msg or "timed out" in msg:
            return ToolErrorType.TIMEOUT
        if "permission" in msg or "denied" in msg or "forbidden" in msg:
            return ToolErrorType.PERMISSION_DENIED
        if "network" in msg or "connection" in msg or "refused" in msg:
            return ToolErrorType.NETWORK
        if "schema" in msg or "validation" in msg:
            return ToolErrorType.SCHEMA_MISMATCH
        return ToolErrorType.EXCEPTION

    def __repr__(self) -> str:
        return f"Tool[{self.name}](timeout={self.timeout_s}s)"
