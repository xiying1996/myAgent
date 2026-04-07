"""
event_types.py — EventType / EventPriority

定义事件类型及其默认优先级归类。
EventQueue 会据此决定进入 HighPriority 还是 Normal 队列。
"""

from __future__ import annotations

from enum import Enum


class EventPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"


class EventType(Enum):
    # HighPriority
    TIMEOUT         = "TIMEOUT"
    ERROR           = "ERROR"           # 系统级错误
    TOOL_ERROR      = "TOOL_ERROR"      # 工具执行抛异常
    SYSTEM_FALLBACK = "SYSTEM_FALLBACK"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    # Normal
    TOOL_RESULT     = "TOOL_RESULT"
    TOOL_FAILED     = "TOOL_FAILED"     # 工具业务层失败
    PLAN_UPDATE     = "PLAN_UPDATE"
    OBSERVATION     = "OBSERVATION"
    STATE_UPDATE    = "STATE_UPDATE"


HIGH_PRIORITY_EVENT_TYPES = frozenset(
    {
        EventType.TIMEOUT,
        EventType.ERROR,
        EventType.TOOL_ERROR,
        EventType.SYSTEM_FALLBACK,
        EventType.BUDGET_EXCEEDED,
    }
)


def default_priority_for(event_type: EventType) -> EventPriority:
    """根据事件类型返回系统约定的默认优先级。"""
    if event_type in HIGH_PRIORITY_EVENT_TYPES:
        return EventPriority.HIGH
    return EventPriority.NORMAL
