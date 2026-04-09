"""
result.py — Tool Execution Contract

标准化工具执行结果，驱动 StateMachine：

  ToolStatus  — 执行状态（SUCCESS / FAILURE / TIMEOUT / ERROR）
  ToolErrorType — 失败归因（SCHEMA_MISMATCH / TIMEOUT / EXCEPTION / PERMISSION_DENIED / NETWORK）
  ToolResult  — 标准结果，含 error_type / exec_time_ms / raw_response / metadata

Failure 策略：
  SCHEMA_MISMATCH  → 不 retry，尝试 fallback + adapter
  TIMEOUT          → 依 retry policy 重试 N 次
  EXCEPTION        → sandboxed 可重试，外部 service 不重试
  PERMISSION_DENIED → human approval
  NETWORK          → 重试 + exponential backoff
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"   # 工具返回的显式失败
    TIMEOUT = "TIMEOUT"
    ERROR   = "ERROR"      # 异常


class ToolErrorType(Enum):
    SCHEMA_MISMATCH   = "SCHEMA_MISMATCH"
    TIMEOUT           = "TIMEOUT"
    EXCEPTION         = "EXCEPTION"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NETWORK           = "NETWORK"
    UNKNOWN           = "UNKNOWN"


@dataclass(frozen=True)
class ToolResult:
    """
    标准化工具执行结果。

    字段：
      status        — 执行状态
      output        — 成功时的输出（dict）
      error         — 错误描述（人类可读）
      error_type    — 错误归因（用于 routing / retry 决策）
      exec_time_ms  — 执行耗时（毫秒）
      raw_response  — 原始返回（用于 replay / debug）
      metadata      — 扩展信息（provider / request_id / 等）
    """
    status: ToolStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[ToolErrorType] = None
    exec_time_ms: Optional[int] = None
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def is_retryable(self) -> bool:
        """是否应该根据 error_type 决定重试。"""
        if not self.is_success:
            et = self.error_type
            if et in (ToolErrorType.TIMEOUT, ToolErrorType.NETWORK, ToolErrorType.EXCEPTION):
                return True
        return False

    @classmethod
    def success(
        cls,
        output: Dict[str, Any],
        exec_time_ms: Optional[int] = None,
        raw_response: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.SUCCESS,
            output=output,
            exec_time_ms=exec_time_ms,
            raw_response=raw_response,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        error: str,
        error_type: ToolErrorType = ToolErrorType.EXCEPTION,
        output: Optional[Dict[str, Any]] = None,
        exec_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.FAILURE if error_type == ToolErrorType.SCHEMA_MISMATCH else ToolStatus.ERROR,
            output=output or {},
            error=error,
            error_type=error_type,
            exec_time_ms=exec_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def timeout(
        cls,
        timeout_s: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.TIMEOUT,
            output={},
            error=f"执行超时（{timeout_s}s）",
            error_type=ToolErrorType.TIMEOUT,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "exec_time_ms": self.exec_time_ms,
            "metadata": self.metadata,
        }
