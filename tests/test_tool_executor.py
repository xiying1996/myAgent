"""
test_tool_executor.py — ToolExecutor + EventQueue + Bus 测试套件

覆盖范围:
  PriorityEventQueue  — put/get_next_event/get_nowait/empty/sizes
  RawEventBus         — publish/subscribe/unsubscribe
  Dispatcher          — dispatch 到正确队列
  ToolExecutor        — submit/超时/异常/成功/工具注册

运行:
  pytest tests/test_tool_executor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time

from events.event_queue import PriorityEventQueue, QueueEmpty
from events.raw_event_bus import RawEventBus, Dispatcher
from events.event_types import EventType, EventPriority
from events.event import Event
from execution.tool_executor import ToolExecutor, Action
from execution.llm_interface import ToolExecutionFailed, ToolTimeoutError


# ===========================================================================
# PriorityEventQueue 测试
# ===========================================================================

class TestPriorityEventQueue:
    def _make(self) -> PriorityEventQueue:
        return PriorityEventQueue()

    def test_put_high_get_high(self):
        q = self._make()
        e = Event.create(agent_id="a1", event_type=EventType.TIMEOUT, payload={"step": "s0"})
        q.put(e)
        assert q.get_nowait().event_type == EventType.TIMEOUT

    def test_put_normal_get_normal(self):
        q = self._make()
        e = Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={"step": "s0"})
        q.put(e)
        assert q.get_nowait().event_type == EventType.TOOL_RESULT

    def test_high优先于_normal(self):
        q = self._make()
        q.put(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))  # normal 先入队
        q.put(Event.create(agent_id="a1", event_type=EventType.TIMEOUT, payload={}))       # high 后入队
        # get_next_event 优先返回 high
        evt1 = q.get_next_event(timeout=0.1)
        assert evt1.event_type == EventType.TIMEOUT
        evt2 = q.get_next_event(timeout=0.1)
        assert evt2.event_type == EventType.TOOL_RESULT

    def test_get_next_event_timeout_raises(self):
        q = self._make()
        with pytest.raises(QueueEmpty):
            q.get_next_event(timeout=0.01)

    def test_sizes(self):
        q = self._make()
        assert q.qsize() == 0
        q.put(Event.create(agent_id="a1", event_type=EventType.TIMEOUT, payload={}))
        q.put(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))
        assert q.high_priority_size == 1
        assert q.normal_priority_size == 1
        assert q.qsize() == 2


# ===========================================================================
# RawEventBus + Dispatcher 测试
# ===========================================================================

class TestRawEventBus:
    def test_publish_triggers_subscriber(self):
        bus = RawEventBus()
        received = []
        bus.subscribe(received.append)
        bus.publish(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={"x": 1}))
        assert len(received) == 1
        assert received[0].event_type == EventType.TOOL_RESULT

    def test_unsubscribe(self):
        bus = RawEventBus()
        handler = lambda e: None
        bus.subscribe(handler)
        assert bus.subscriber_count() == 1
        bus.unsubscribe(handler)
        assert bus.subscriber_count() == 0

    def test_multiple_subscribers(self):
        bus = RawEventBus()
        a, b = [], []
        bus.subscribe(a.append)
        bus.subscribe(b.append)
        bus.publish(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))
        assert len(a) == 1
        assert len(b) == 1


class TestDispatcher:
    def test_dispatch_routes_to_correct_queue(self):
        q = PriorityEventQueue()
        d = Dispatcher(q)
        # high
        d.dispatch(Event.create(agent_id="a1", event_type=EventType.TIMEOUT, payload={}))
        assert q.high_priority_size == 1
        assert q.normal_priority_size == 0
        # normal
        d.dispatch(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))
        assert q.high_priority_size == 1
        assert q.normal_priority_size == 1

    def test_attach_to_bus(self):
        q = PriorityEventQueue()
        d = Dispatcher(q)
        bus = RawEventBus()
        d.attach(bus)
        bus.publish(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))
        assert q.normal_priority_size == 1


# ===========================================================================
# ToolExecutor 测试
# ===========================================================================

class TestToolExecutor:
    def _make(self) -> tuple[ToolExecutor, PriorityEventQueue]:
        q = PriorityEventQueue()
        bus = RawEventBus()
        # ToolExecutor 需要通过 bus 连接到 queue
        d = Dispatcher(q)
        d.attach(bus)
        executor = ToolExecutor(bus=bus, max_workers=2)
        return executor, q

    def test_register_tool(self):
        q = PriorityEventQueue()
        bus = RawEventBus()
        d = Dispatcher(q)
        d.attach(bus)
        executor = ToolExecutor(bus=bus)
        executor.register_tool("echo", lambda **kw: kw)
        assert executor.is_registered("echo")

    def test_submit_success(self):
        executor, q = self._make()
        executor.register_tool("echo", lambda **kw: {"ok": True, **kw})

        action = Action(tool_name="echo", params={"x": 1}, agent_id="a1", step_id="s0")
        executor.submit(action)

        # 等待事件到达队列（异步）
        time.sleep(0.05)
        assert q.qsize() == 1
        evt = q.get_nowait()
        assert evt.event_type == EventType.TOOL_RESULT
        assert evt.payload["result"]["x"] == 1

    def test_submit_unregistered_tool_sends_error(self):
        executor, q = self._make()

        action = Action(tool_name="unknown_tool", params={}, agent_id="a1", step_id="s0")
        executor.submit(action)

        time.sleep(0.05)
        evt = q.get_nowait()
        assert evt.event_type == EventType.TOOL_ERROR
        assert "未注册" in evt.payload["error_msg"]

    def test_submit_failure_sends_tool_failed(self):
        executor, q = self._make()
        executor.register_tool("failing", lambda **kw: (_ for _ in ()).throw(
            ToolExecutionFailed("mock failure")
        ))

        action = Action(tool_name="failing", params={}, agent_id="a1", step_id="s0")
        executor.submit(action)

        time.sleep(0.05)
        evt = q.get_nowait()
        assert evt.event_type == EventType.TOOL_FAILED
        assert "mock failure" in evt.payload["reason"]

    def test_submit_exception_sends_tool_error(self):
        executor, q = self._make()
        executor.register_tool("crash", lambda **kw: (_ for _ in ()).throw(
            RuntimeError("unexpected")
        ))

        action = Action(tool_name="crash", params={}, agent_id="a1", step_id="s0")
        executor.submit(action)

        time.sleep(0.05)
        evt = q.get_nowait()
        assert evt.event_type == EventType.TOOL_ERROR
        assert "RuntimeError" in evt.payload["error_type"]

    def test_submit_with_timeout(self):
        import time
        executor, q = self._make()
        executor.register_tool(
            "slow",
            lambda **kw: time.sleep(0.3),
        )

        action = Action(
            tool_name="slow", params={},
            agent_id="a1", step_id="s0",
            timeout_s=0.05,
        )
        executor.submit(action)

        # 等待超时事件
        time.sleep(0.2)
        evt = q.get_nowait()
        assert evt.event_type == EventType.TIMEOUT
        assert evt.payload["timeout_s"] == 0.05

        executor.shutdown()

    def test_shutdown(self):
        executor, q = self._make()
        executor.shutdown()
        # shutdown 后不再接受任务（静默忽略）
        # 不抛异常


class TestMockToolRegistry:
    def test_register_and_call(self):
        from execution.llm_interface import MockToolRegistry
        reg = MockToolRegistry()
        reg.register("add", lambda **kw: kw.get("a", 0) + kw.get("b", 0))
        assert reg.call("add", a=2, b=3) == 5

    def test_simulate_failure(self):
        from execution.llm_interface import MockToolRegistry, ToolExecutionFailed
        reg = MockToolRegistry()
        reg.register("fail_once", lambda **kw: "ok")
        reg.simulate_failure("fail_once")
        with pytest.raises(ToolExecutionFailed, match="Mock 模拟失败"):
            reg.call("fail_once")
        # 第二次应该成功
        assert reg.call("fail_once") == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
