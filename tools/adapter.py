"""
adapter.py — Adapter / Transformation Pipeline

核心类：
  Adapter          — 带 schema 的转换器节点
  AdapterRegistry  — 转换器注册 + find_path(BFS) 路径查找

设计原则：
  - Adapter 和 Tool 一样是 schema-aware 节点
  - find_path 用 BFS 找从 from_schema 到 to_schema 的转换路径
  - 支持多步串联：html → text → json

使用场景：
  当 tool.output_schema 不匹配 step.input_schema 时，
  查找 adapter 链来自动填补 schema gap。
"""

from __future__ import annotations

import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from tools.schema import Schema

logger = logging.getLogger(__name__)


# ── Adapter ─────────────────────────────────────────────────────────────────


@dataclass
class Adapter:
    """
    带 schema 的转换器。

    例：
      html_to_text = Adapter(
          name="html_to_text",
          input_schema=Schema.from_dict({"html": "str"}),
          output_schema=Schema.from_dict({"text": "str"}),
          transform=lambda d: {"text": strip_html(d["html"])},
      )
    """
    name: str
    input_schema: Schema
    output_schema: Schema
    transform: Callable[[Dict[str, Any]], Dict[str, Any]]
    tags: List[str] = field(default_factory=list)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用转换（无校验，快速路径）。

        使用场景：
        - 内部工具链，调用方已知输入合法
        - 测试代码，快速构造期望结果
        - AdapterRegistry.find_path() 的内部 BFS 探索

        注意：不对输入/输出做 schema 校验，失败直接抛出异常。
        如需校验，请使用 apply_checked()。
        """
        return self.transform(data)

    def apply_checked(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用转换，带完整 schema 校验。

        使用场景：
        - Scheduler 的 normalize 阶段（外部输入，需要严格校验）
        - 用户输入/工具返回结果的规范化
        - 任何需要确保数据符合 schema 的场景

        处理流程：
          1. 输入校验 + coercion（validate_and_coerce）
          2. 执行转换（transform）
          3. 输出校验（validate_and_coerce）

        失败时抛出 ValueError。
        """

        # 1. 输入校验 + coercion
        validated_input = self.input_schema.validate_and_coerce(dict(data))

        # 2. 执行转换
        result = self.transform(validated_input)

        # 3. 输出校验
        validated_output = self.output_schema.validate_and_coerce(result)

        return validated_output

    def can_accept(self, schema: Schema) -> bool:
        """input_schema 是否与给定 schema 兼容。"""
        return schema.is_compatible_with(self.input_schema)

    def can_produce(self, schema: Schema) -> bool:
        """output_schema 是否满足给定 schema。"""
        return self.output_schema.is_superset_of(schema)


# ── Built-in Adapters ──────────────────────────────────────────────────────


def _html_to_text(data: Dict[str, Any]) -> Dict[str, Any]:
    """html → text（简单去除 HTML 标签）"""
    import re
    html = data.get("html", "")
    text = re.sub(r"<[^>]+>", "", html)
    text = text.strip()
    return {"text": text}


def _str_to_int(data: Dict[str, Any]) -> Dict[str, Any]:
    """str → int（尝试解析）"""
    s = data.get("value", "0")
    try:
        return {"value": int(s)}
    except (ValueError, TypeError):
        return {"value": 0, "_parse_error": f"无法解析为 int: {s}"}


def _str_to_float(data: Dict[str, Any]) -> Dict[str, Any]:
    """str → float"""
    s = data.get("value", "0.0")
    try:
        return {"value": float(s)}
    except (ValueError, TypeError):
        return {"value": 0.0, "_parse_error": f"无法解析为 float: {s}"}


# 内置常用 adapters
BUILTIN_ADAPTERS: List[Adapter] = [
    Adapter(
        name="html_to_text",
        input_schema=Schema.from_dict({"html": "str"}),
        output_schema=Schema.from_dict({"text": "str"}),
        transform=_html_to_text,
        tags=["html", "text", "cleanup"],
    ),
    Adapter(
        name="str_to_int",
        input_schema=Schema.from_dict({"value": "str"}),
        output_schema=Schema.from_dict({"value": "int"}),
        transform=_str_to_int,
        tags=["coercion", "type"],
    ),
    Adapter(
        name="str_to_float",
        input_schema=Schema.from_dict({"value": "str"}),
        output_schema=Schema.from_dict({"value": "float"}),
        transform=_str_to_float,
        tags=["coercion", "type"],
    ),
]


# ── AdapterRegistry ─────────────────────────────────────────────────────────


class AdapterRegistry:
    """
    转换器注册中心 + 路径查找。

    find_path 用 BFS 找从 from_schema 到 to_schema 的转换路径。
    支持多步串联（adapter chain）。

    使用示例:
        reg = AdapterRegistry()
        reg.register(Adapter(...))
        path = reg.find_path(
            from_schema=Schema.from_dict({"html": "str"}),
            to_schema=Schema.from_dict({"text": "str"}),
        )
        # path = [Adapter(html_to_text)]
    """

    def __init__(self) -> None:
        self._adapters: List[Adapter] = []
        self._by_tag: Dict[str, List[Adapter]] = defaultdict(list)

    def register(self, adapter: Adapter) -> None:
        """注册 Adapter。"""
        self._adapters.append(adapter)
        for tag in adapter.tags:
            self._by_tag[tag].append(adapter)
        logger.debug("AdapterRegistry: 注册 adapter=%s", adapter.name)

    def get(self, name: str) -> Optional[Adapter]:
        """按名称查找 Adapter，未找到返回 None。"""
        for a in self._adapters:
            if a.name == name:
                return a
        return None

    def find_direct(
        self,
        from_schema: Schema,
        to_schema: Schema,
    ) -> Optional[Adapter]:
        """找单个直接转换器（input 兼容 from_schema，output 是 to_schema 的超集）。"""
        for a in self._adapters:
            if from_schema.is_compatible_with(a.input_schema) and a.output_schema.is_superset_of(to_schema):
                return a
        return None

    def find_path(
        self,
        from_schema: Schema,
        to_schema: Schema,
        max_hops: int = 3,
    ) -> List[Adapter]:
        """
        BFS 找从 from_schema 到 to_schema 的转换路径。

        规则：
          - 每步 adapter 的 input_schema 必须与当前 schema 兼容
          - 每步 adapter 的 output_schema 必须能推进到目标
          - 最多 max_hops 步
          - 返回的路径可能为空（找不到转换路径）

        返回：
          List[Adapter]，串联后从 from_schema 到达 to_schema
        """
        if from_schema.is_superset_of(to_schema):
            # 无需转换
            return []

        # BFS 队列元素: (current_schema, path_so_far)
        queue: deque[tuple[Schema, List[Adapter]]] = deque()
        queue.append((from_schema, []))

        visited: Set[str] = set()

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue

            # 用 schema 的字段集合作为 cache key
            cache_key = ",".join(sorted(current.keys()))
            if cache_key in visited:
                continue
            visited.add(cache_key)

            for adapter in self._adapters:
                # adapter 是否能接住 current？
                if not current.is_compatible_with(adapter.input_schema):
                    continue

                next_schema = adapter.output_schema
                new_path = path + [adapter]

                # 检查是否满足目标
                if next_schema.is_superset_of(to_schema):
                    return new_path

                # 继续探索
                queue.append((next_schema, new_path))

        logger.debug(
            "AdapterRegistry: 找不到从 %s 到 %s 的路径（max_hops=%d）",
            from_schema.to_simple_dict(),
            to_schema.to_simple_dict(),
            max_hops,
        )
        return []

    def find_by_tag(self, tag: str) -> List[Adapter]:
        return list(self._by_tag.get(tag, []))

    def __len__(self) -> int:
        return len(self._adapters)
