"""
event.py — Event 数据模型

Phase 1 只定义事件本身，不包含队列与分发逻辑。
约束:
  - event_id / agent_id 不能为空
  - payload 必须是 dict
  - priority 必须与 event_type 的系统约定一致，避免脏数据进入队列层
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .event_types import EventPriority, EventType, default_priority_for


@dataclass(frozen=True)
class Event:
    event_id: str
    agent_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: float
    priority: EventPriority
    is_plan_snapshot: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("Event.event_id 不能为空")
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("Event.agent_id 不能为空")
        if not isinstance(self.payload, dict):
            raise ValueError("Event.payload 必须是 dict")

        expected = default_priority_for(self.event_type)
        if self.priority != expected:
            raise ValueError(
                f"Event[{self.event_id}] 优先级与类型不匹配: "
                f"{self.event_type.value} 应为 {expected.value}，实际为 {self.priority.value}"
            )

        # 冻结前做一层浅拷贝，避免外部持有同一个 payload 引用继续修改。
        object.__setattr__(self, "payload", dict(self.payload))

    @classmethod
    def create(
        cls,
        agent_id: str,
        event_type: EventType,
        payload: Dict[str, Any],
        *,
        event_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        priority: Optional[EventPriority] = None,
        is_plan_snapshot: bool = False,
    ) -> "Event":
        return cls(
            event_id=event_id or f"evt_{uuid.uuid4().hex[:10]}",
            agent_id=agent_id,
            event_type=event_type,
            payload=payload,
            timestamp=time.monotonic() if timestamp is None else timestamp,
            priority=priority or default_priority_for(event_type),
            is_plan_snapshot=is_plan_snapshot,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """从持久化 dict 恢复 Event。"""
        return cls(
            event_id=data["event_id"],
            agent_id=data["agent_id"],
            event_type=EventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data["timestamp"],
            priority=EventPriority(data.get("priority", default_priority_for(EventType(data["event_type"])).value)),
            is_plan_snapshot=data.get("is_plan_snapshot", False),
        )

    @property
    def is_high_priority(self) -> bool:
        return self.priority == EventPriority.HIGH

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "event_type": self.event_type.value,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "is_plan_snapshot": self.is_plan_snapshot,
        }
