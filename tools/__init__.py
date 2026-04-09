"""
tools — 工业级 Tool 系统

导出：
  Schema, Field                — 类型系统
  ToolResult, ToolStatus, ToolErrorType — 执行结果契约
  Tool, ToolMetadata, RetryPolicy — 工具定义
  ToolRegistry                 — 工具注册中心
  Adapter, AdapterRegistry     — 转换器 / 数据流修复
"""

from tools.schema import Schema, Field
from tools.result import ToolResult, ToolStatus, ToolErrorType
from tools.tool import Tool, ToolMetadata, RetryPolicy
from tools.registry import ToolRegistry
from tools.adapter import Adapter, AdapterRegistry

__all__ = [
    "Schema",
    "Field",
    "ToolResult",
    "ToolStatus",
    "ToolErrorType",
    "Tool",
    "ToolMetadata",
    "RetryPolicy",
    "ToolRegistry",
    "Adapter",
    "AdapterRegistry",
]
