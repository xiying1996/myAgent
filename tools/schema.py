"""
schema.py — Schema 类型系统

Typed, type-safe schema 定义：
  Field   — 单字段定义（type + optional + nested）
  Schema  — 字段集合，支持 is_superset_of / is_compatible_with

核心原则：
  - is_superset_of: 严格匹配（字段 + 类型），用于 tool.output ⊇ step.input
  - is_compatible_with: 宽松匹配（字段存在 + 类型可 coercion），用于 tool selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

# 支持的原子类型
ATOMIC_TYPES: set[str] = {"str", "int", "float", "bool"}


@dataclass(frozen=True)
class Field:
    """
    单字段定义。

    type: 原子类型或 "list"（内部元素类型由 nested 表达）
    optional: 该字段是否可选
    nested: 嵌套字段（仅当 type="dict" 时有效），例：
        Field(type="dict", nested={"name": Field(type="str"), "age": Field(type="int", optional=True)})
    """
    type: str
    optional: bool = False
    nested: Optional[Dict[str, Field]] = None

    def __post_init__(self) -> None:
        if self.nested is not None and self.type != "dict":
            raise ValueError(f"Field[{self}]: nested 只在 type='dict' 时有效")


@dataclass(frozen=True)
class Schema:
    """
    字段集合。

    fields: {field_name: Field}

    is_superset_of(other):      严格匹配，self 的字段覆盖 other
    is_compatible_with(other):  宽松匹配，类型可 coercion
    """
    fields: Dict[str, Field]

    def keys(self) -> Set[str]:
        return set(self.fields.keys())

    def get(self, name: str) -> Optional[Field]:
        return self.fields.get(name)

    def validate_and_coerce(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        按当前 schema 校验并规范化一个 payload。

        规则：
          - 缺少必填字段 → ValueError
          - 缺少可选字段 → 忽略
          - 已声明字段会按目标类型做 coercion
          - 未声明字段保留原样（向前兼容）
        """
        if not isinstance(data, dict):
            raise ValueError(f"payload 必须是 dict，实际为 {type(data).__name__}")

        normalized = dict(data)
        for name, field_def in self.fields.items():
            if name not in data:
                if field_def.optional:
                    continue
                raise ValueError(f"缺少必填字段 '{name}'")

            value = data[name]
            if value is None:
                if field_def.optional:
                    normalized[name] = None
                    continue
                raise ValueError(f"字段 '{name}' 不能为空")

            normalized[name] = self._coerce_value(name, value, field_def)

        return normalized

    # ── 严格匹配 ─────────────────────────────────────────────────────────

    def is_superset_of(self, other: Schema) -> bool:
        """
        严格超集：self（通常是 tool.output）是否包含
        other（通常是 step.input_required）的所有必填字段。

        条件：对于 other 的每个必填字段 f，
          - self 必须有同名同类型字段
          - self 的该字段不能是 optional（除非 f 也是 optional，但这里用严格模式）
        """
        for name, f in other.fields.items():
            if name not in self.fields:
                return False
            mine = self.fields[name]
            if not self._field_compatible(mine, f, strict=True):
                return False
        return True

    # ── 宽松匹配 ─────────────────────────────────────────────────────────

    def is_compatible_with(self, other: Schema) -> bool:
        """
        兼容匹配：other 的每个字段在 self 中都能找到兼容版本。

        规则：
          - self 有该字段：类型可 coercion 或 self 是 dict
          - self 无该字段：该字段在 other 中必须是 optional
        """
        for name, f in other.fields.items():
            if name not in self.fields:
                if not f.optional:
                    return False
                continue
            mine = self.fields[name]
            if not self._field_compatible(mine, f):
                return False
        return True

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _field_compatible(self, mine: Field, theirs: Field, strict: bool = True) -> bool:
        """检查单个字段是否兼容。"""
        if strict and mine.optional and not theirs.optional:
            return False

        if mine.type == "dict" and theirs.type == "dict":
            mine_nested = Schema(mine.nested or {})
            theirs_nested = Schema(theirs.nested or {})
            if strict:
                return mine_nested.is_superset_of(theirs_nested)
            return mine_nested.is_compatible_with(theirs_nested)

        return self._can_coerce(mine.type, theirs.type)

    def _can_coerce(self, from_t: str, to_t: str) -> bool:
        """判断 from_t 能否 coercion 到 to_t。"""
        if from_t == to_t:
            return True
        # 字符串可解析为 int
        if to_t == "int" and from_t == "str":
            return True
        # 字符串/int 可转为 float
        if to_t == "float" and from_t in ("str", "int"):
            return True
        # 任何类型可到 str
        if to_t == "str":
            return True
        return False

    def _coerce_value(self, name: str, value: Any, field_def: Field) -> Any:
        target = field_def.type

        if target == "dict":
            if not isinstance(value, dict):
                raise ValueError(f"字段 '{name}' 期望 dict，实际为 {type(value).__name__}")
            if field_def.nested:
                return Schema(field_def.nested).validate_and_coerce(value)
            return dict(value)

        if target == "list":
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            raise ValueError(f"字段 '{name}' 期望 list，实际为 {type(value).__name__}")

        if target == "str":
            return str(value)

        if target == "int":
            if isinstance(value, bool):
                return int(value)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"字段 '{name}' 无法转为 int: {value}") from exc

        if target == "float":
            if isinstance(value, bool):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"字段 '{name}' 无法转为 float: {value}") from exc

        if target == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            raise ValueError(f"字段 '{name}' 无法转为 bool: {value}")

        raise ValueError(f"字段 '{name}' 使用了不支持的类型 '{target}'")

    # ── 工厂 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Schema:
        """
        从简单 dict 构建 Schema。
        支持格式：
          {"name": "str"}  → Field(type="str")
          {"name": {"type": "str", "optional": True}}
          {"name": {"type": "dict", "nested": {"age": "int"}}}
        """
        fields: Dict[str, Field] = {}
        for name, spec in d.items():
            if isinstance(spec, str):
                fields[name] = Field(type=spec)
            elif isinstance(spec, dict):
                nested = None
                if "nested" in spec and spec["nested"]:
                    nested = {}
                    for nk, nv in spec["nested"].items():
                        if isinstance(nv, str):
                            nested[nk] = Field(type=nv)
                        else:
                            nested[nk] = Field(**nv)
                fields[name] = Field(
                    type=spec.get("type", "str"),
                    optional=spec.get("optional", False),
                    nested=nested,
                )
            else:
                fields[name] = Field(type="str")
        return cls(fields=fields)

    def to_simple_dict(self) -> Dict[str, str]:
        """简化输出：{field_name: type_name}（用于日志/兼容）"""
        return {k: v.type for k, v in self.fields.items()}
