"""
registry.py — Tool Registry

核心类：
  ToolRegistry — 工具注册中心，支持：
    - 按 name 注册 / 查找
    - 反向索引（output field → tools）
    - 按 tag / permission 查找
    - 兼容注册层（Tool 实例 或 callable + schemas）

设计原则：
  - backward compatible：支持 register(Tool) 和 register(callable, name=..., ...)
  - 反向索引：find_compatible() 使用倒排表高效查找
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from tools.schema import Schema
from tools.tool import Tool, ToolMetadata

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    """注册工具的完整信息。"""
    tool: Tool
    func: Optional[Callable] = None  # 旧路途：原始 callable


class ToolRegistry:
    """
    工具注册中心。

    使用示例:
        registry = ToolRegistry()

        # 方式1：注册 Tool 实例
        registry.register(MyTool())

        # 方式2：注册 callable（自动 wrap）
        registry.register(some_fn, name="my_tool",
                          input_schema=Schema.from_dict({"q": "str"}),
                          output_schema=Schema.from_dict({"results": "list"}))

        # 查找
        tool = registry.get("my_tool")
        compatible = registry.find_compatible({"results": "str"})
        by_tag = registry.find_by_tag("web")
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, RegisteredTool] = {}
        # 反向索引：output field → {tool_name}
        self._index_by_output: Dict[str, Set[str]] = defaultdict(set)
        # 按 tag 索引：tag → {tool_name}
        self._index_by_tag: Dict[str, Set[str]] = defaultdict(set)
        # 按 permission 索引：permission → {tool_name}
        self._index_by_perm: Dict[str, Set[str]] = defaultdict(set)

    # ── 注册 ──────────────────────────────────────────────────────────────

    def register(
        self,
        tool_or_func: Union[Tool, Callable],
        *,
        name: Optional[str] = None,
        input_schema: Optional[Schema] = None,
        output_schema: Optional[Schema] = None,
        description: str = "",
        timeout_s: float = 30.0,
        metadata: Optional[ToolMetadata] = None,
    ) -> None:
        """
        兼容注册层。

        方式1 — Tool 实例：
            registry.register(MyTool())

        方式2 — callable + schemas：
            registry.register(fn, name="search",
                              input_schema=Schema(...),
                              output_schema=Schema(...))
        """
        if isinstance(tool_or_func, Tool):
            tool = tool_or_func
            self._register_tool(tool)
            return

        # 方式2：包装为匿名 Tool
        func = tool_or_func
        if not name:
            raise ValueError("register(callable) 必须指定 name")
        if not input_schema or not output_schema:
            raise ValueError("register(callable) 必须指定 input_schema 和 output_schema")

        wrapped = _CallableTool(
            name=name,
            description=description or f"Wrapped from {func.__name__}",
            input_schema=input_schema,
            output_schema=output_schema,
            func=func,
            timeout_s=timeout_s,
            metadata=metadata,
        )
        self._register_tool(wrapped)

    def _register_tool(self, tool: Tool) -> None:
        """内部：注册 Tool 实例。"""
        if tool.name in self._by_name:
            logger.warning("ToolRegistry: 工具 '%s' 已被注册，将被覆盖", tool.name)

        self._by_name[tool.name] = RegisteredTool(tool=tool)

        # 更新反向索引
        for field_name in tool.output_schema.keys():
            self._index_by_output[field_name].add(tool.name)

        # 更新 tag 索引
        for tag in tool.metadata.tags:
            self._index_by_tag[tag].add(tool.name)

        # 更新 permission 索引
        for perm in tool.metadata.permissions:
            self._index_by_perm[perm].add(tool.name)

        logger.debug(
            "ToolRegistry: 注册 tool=%s output_fields=%s tags=%s permissions=%s",
            tool.name,
            list(tool.output_schema.keys()),
            tool.metadata.tags,
            tool.metadata.permissions,
        )

    def unregister(self, name: str) -> None:
        """注销工具。"""
        if name not in self._by_name:
            return
        tool = self._by_name.pop(name).tool
        for field_name in tool.output_schema.keys():
            self._index_by_output[field_name].discard(name)
        for tag in tool.metadata.tags:
            self._index_by_tag[tag].discard(name)
        for perm in tool.metadata.permissions:
            self._index_by_perm[perm].discard(name)

    # ── 查询 ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Tool]:
        """按 name 查找工具。"""
        rt = self._by_name.get(name)
        return rt.tool if rt else None

    def get_or_raise(self, name: str) -> Tool:
        t = self.get(name)
        if t is None:
            raise KeyError(f"ToolRegistry: 工具 '{name}' 未注册")
        return t

    def list_all(self) -> List[Tool]:
        """返回所有已注册工具。"""
        return [rt.tool for rt in self._by_name.values()]

    def find_compatible(
        self,
        required_fields: Union[Set[str], Dict[str, Any], Schema],
        required_tags: Optional[List[str]] = None,
    ) -> List[Tool]:
        """
        找所有 output_schema.fields 包含 required_fields 的工具。

        倒排索引加速：对于每个 required_field，取对应 tool_name 集合的交集。
        """
        required_schema = self._normalize_required_schema(required_fields)
        field_names = required_schema.keys()

        if not field_names:
            candidates = set(self._by_name.keys())
        else:
            # 取所有包含 required_fields 的工具的交集
            field_lists = [self._index_by_output.get(f, set()) for f in field_names]
            candidates = set.intersection(*field_lists) if field_lists else set()
            if not candidates:
                return []

        # 过滤 tag
        if required_tags:
            tag_sets = [self._index_by_tag.get(t, set()) for t in required_tags]
            tag_candidates = set.intersection(*tag_sets) if tag_sets else set()
            candidates &= tag_candidates

        return [
            self._by_name[n].tool
            for n in candidates
            if self._by_name[n].tool.output_schema.is_superset_of(required_schema)
        ]

    def find_by_tag(self, tag: str) -> List[Tool]:
        """按 tag 查找工具。"""
        names = self._index_by_tag.get(tag, set())
        return [self._by_name[n].tool for n in names]

    def find_with_permissions(self, permissions: List[str]) -> List[Tool]:
        """找所有满足指定 permissions 的工具。"""
        perm_sets = [self._index_by_perm.get(p, set()) for p in permissions]
        candidates = set.intersection(*perm_sets) if perm_sets else set()
        return [self._by_name[n].tool for n in candidates]

    def find_by_name_pattern(self, pattern: str) -> List[Tool]:
        """按 name 前缀/包含模糊查找。"""
        return [
            rt.tool for rt in self._by_name.values()
            if pattern in rt.tool.name
        ]

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def _normalize_required_schema(
        self,
        required_fields: Union[Set[str], Dict[str, Any], Schema],
    ) -> Schema:
        if isinstance(required_fields, Schema):
            return required_fields
        if isinstance(required_fields, dict):
            return Schema.from_dict(required_fields)
        return Schema.from_dict({field_name: "str" for field_name in required_fields})


# ── 内部：Callable 包装 ─────────────────────────────────────────────────────


class _CallableTool(Tool):
    """
    将普通函数包装为 Tool 的内部实现。
    不暴露给外部，只通过 registry.register(callable, ...) 使用。
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Schema,
        output_schema: Schema,
        func: Callable[..., Any],
        timeout_s: float = 30.0,
        metadata: Optional[ToolMetadata] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout_s=timeout_s,
            metadata=metadata,
        )
        self._func = func

    def _do_invoke(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        result = self._func(**input_params)
        if isinstance(result, dict):
            return result
        return {"value": result}
