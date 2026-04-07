"""
event_queue.py — PriorityEventQueue

双优先级事件队列：
  - HIGH 事件必须先于 NORMAL 被消费
  - 同一优先级内保持 FIFO
  - 初版按串行消费设计，但允许多个生产者并发 put()

实现上不直接依赖两个 queue.Queue，而是用 deque + Condition。
这样在消费者等待 NORMAL 事件时，如果期间来了 HIGH 事件，也能被立即优先取出。
"""

from __future__ import annotations

import time
from collections import deque
from queue import Empty
from threading import Condition

# Alias for backwards compatibility
QueueEmpty = Empty
from typing import Deque, Optional

from .event import Event
from .event_types import EventPriority


class PriorityEventQueue:
    """双优先级、线程安全的事件队列。"""

    def __init__(self) -> None:
        self._high_priority_queue: Deque[Event] = deque()
        self._normal_queue: Deque[Event] = deque()
        self._cv = Condition()

    def put(self, event: Event) -> None:
        """按事件优先级入队，并唤醒等待中的消费者。"""
        with self._cv:
            if event.priority == EventPriority.HIGH:
                self._high_priority_queue.append(event)
            else:
                self._normal_queue.append(event)
            self._cv.notify()

    def get_next_event(self, timeout: Optional[float] = 1.0) -> Event:
        """
        获取下一条事件。

        规则:
          1. 只要 HIGH 非空，永远优先返回 HIGH
          2. 两个队列都空时阻塞等待
          3. timeout 到期仍无事件则抛 queue.Empty
        """
        with self._cv:
            if timeout is None:
                while True:
                    event = self._pop_next()
                    if event is not None:
                        return event
                    self._cv.wait()

            deadline = time.monotonic() + timeout
            while True:
                event = self._pop_next()
                if event is not None:
                    return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Empty()
                waited = self._cv.wait(remaining)
                if not waited:
                    raise Empty()
                event = self._pop_next()
                if event is not None:
                    return event

    def get_nowait(self) -> Event:
        """非阻塞获取；无事件时抛 queue.Empty。"""
        with self._cv:
            event = self._pop_next()
            if event is None:
                raise Empty()
            return event

    def empty(self) -> bool:
        with self._cv:
            return not self._high_priority_queue and not self._normal_queue

    def qsize(self) -> int:
        with self._cv:
            return len(self._high_priority_queue) + len(self._normal_queue)

    @property
    def high_priority_size(self) -> int:
        with self._cv:
            return len(self._high_priority_queue)

    @property
    def normal_priority_size(self) -> int:
        with self._cv:
            return len(self._normal_queue)

    def _pop_next(self) -> Optional[Event]:
        if self._high_priority_queue:
            return self._high_priority_queue.popleft()
        if self._normal_queue:
            return self._normal_queue.popleft()
        return None
