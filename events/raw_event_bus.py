"""
raw_event_bus.py — RawEventBus + Dispatcher

职责拆分:
  - RawEventBus: 只负责事件发布与订阅，不做优先级判断
  - Dispatcher: 订阅 RawEventBus，把事件路由到 PriorityEventQueue
"""

from __future__ import annotations

from threading import Lock
from typing import Callable, List

from .event import Event
from .event_queue import PriorityEventQueue


EventHandler = Callable[[Event], None]


class RawEventBus:
    """线程安全的同步发布总线。"""

    def __init__(self) -> None:
        self._subscribers: List[EventHandler] = []
        self._lock = Lock()

    def subscribe(self, handler: EventHandler) -> None:
        with self._lock:
            if handler not in self._subscribers:
                self._subscribers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers = [h for h in self._subscribers if h != handler]

    def publish(self, event: Event) -> None:
        """
        同步发布事件。
        为避免 handler 回调时导致锁重入或阻塞，先拍平订阅者快照再调用。
        """
        with self._lock:
            subscribers = list(self._subscribers)
        for handler in subscribers:
            handler(event)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class Dispatcher:
    """订阅 RawEventBus，把事件投递到 PriorityEventQueue。"""

    def __init__(self, event_queue: PriorityEventQueue) -> None:
        self._queue = event_queue

    def dispatch(self, event: Event) -> None:
        self._queue.put(event)

    def attach(self, bus: RawEventBus) -> None:
        bus.subscribe(self.dispatch)

    def detach(self, bus: RawEventBus) -> None:
        bus.unsubscribe(self.dispatch)
