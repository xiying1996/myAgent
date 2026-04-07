"""
test_event_system.py — EventQueue / RawEventBus / Dispatcher 测试
"""

from __future__ import annotations

import time
from queue import Empty
from threading import Thread

import pytest

from events.event import Event
from events.event_queue import PriorityEventQueue
from events.event_types import EventType
from events.raw_event_bus import Dispatcher, RawEventBus


class TestPriorityEventQueue:
    def test_high_priority_preempts_normal(self):
        q = PriorityEventQueue()
        normal = Event.create("agent_1", EventType.TOOL_RESULT, {"step_id": "s0"})
        high = Event.create("agent_1", EventType.TIMEOUT, {"step_id": "s0"})

        q.put(normal)
        q.put(high)

        assert q.get_next_event().event_type == EventType.TIMEOUT
        assert q.get_next_event().event_type == EventType.TOOL_RESULT

    def test_fifo_within_same_priority(self):
        q = PriorityEventQueue()
        e1 = Event.create("agent_1", EventType.OBSERVATION, {"seq": 1})
        e2 = Event.create("agent_1", EventType.OBSERVATION, {"seq": 2})

        q.put(e1)
        q.put(e2)

        assert q.get_next_event().payload["seq"] == 1
        assert q.get_next_event().payload["seq"] == 2

    def test_get_nowait_raises_when_empty(self):
        q = PriorityEventQueue()
        with pytest.raises(Empty):
            q.get_nowait()

    def test_timeout_raises_empty(self):
        q = PriorityEventQueue()
        with pytest.raises(Empty):
            q.get_next_event(timeout=0.01)

    def test_waiting_consumer_still_prefers_late_high_priority(self):
        q = PriorityEventQueue()
        received = []

        def consumer() -> None:
            received.append(q.get_next_event(timeout=0.2))

        t = Thread(target=consumer)
        t.start()
        time.sleep(0.02)

        q.put(Event.create("agent_1", EventType.TIMEOUT, {"step_id": "s0"}))
        t.join()

        assert len(received) == 1
        assert received[0].event_type == EventType.TIMEOUT


class TestRawEventBusAndDispatcher:
    def test_publish_routes_to_priority_queue(self):
        q = PriorityEventQueue()
        bus = RawEventBus()
        dispatcher = Dispatcher(q)
        dispatcher.attach(bus)

        bus.publish(Event.create("agent_1", EventType.TOOL_RESULT, {"step_id": "s0"}))
        bus.publish(Event.create("agent_1", EventType.ERROR, {"reason": "crash"}))

        assert q.high_priority_size == 1
        assert q.normal_priority_size == 1
        assert q.get_next_event().event_type == EventType.ERROR
        assert q.get_next_event().event_type == EventType.TOOL_RESULT

    def test_multiple_subscribers_are_called(self):
        bus = RawEventBus()
        seen = []

        def subscriber(event: Event) -> None:
            seen.append(event.event_id)

        bus.subscribe(subscriber)
        bus.subscribe(subscriber)

        event = Event.create("agent_1", EventType.STATE_UPDATE, {"state": "READY"})
        bus.publish(event)

        assert seen == [event.event_id]
        assert bus.subscriber_count() == 1

    def test_unsubscribe_detaches_handler(self):
        bus = RawEventBus()
        q = PriorityEventQueue()
        dispatcher = Dispatcher(q)
        dispatcher.attach(bus)
        dispatcher.detach(bus)

        bus.publish(Event.create("agent_1", EventType.TIMEOUT, {"step_id": "s0"}))

        assert q.empty() is True
