"""
test_day6.py — DependencyValidator + StateManager + Scheduler e2e 测试套件

覆盖范围:
  DependencyValidator   — register_plan / resolve_bindings / propagate_output / validate_replan
  StateManager          — add_agent / on_event 各事件类型 / metrics
  Scheduler             — submit_task / 主循环 / 状态引导
  PolicyEngine          — check_transition / check_tool / check_before_replan

运行:
  pytest tests/test_day6.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
import threading

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget, BudgetUsage
from core.agent import Agent
from core.state_machine import AgentState, RetryMode

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import RawEventBus, Dispatcher
from events.event import Event
from events.event_types import EventType

from execution.llm_interface import MockLLM
from execution.step_runner import StepRunner
from execution.tool_executor import ToolExecutor

from scheduler.policy_engine import PolicyEngine
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager
from scheduler.scheduler import Scheduler
from tools.registry import ToolRegistry
from tools.schema import Schema
from tools.tool import Tool


# ===========================================================================
# DependencyValidator 测试
# ===========================================================================

class TestDependencyValidator:
    def _make_plan(self) -> Plan:
        return Plan.create([
            Step("s0", "web_search", {"q": "x"},
                 output_schema={"url": "str", "title": "str"}),
            Step("s1", "fetch", {"target_url": ""},
                 input_bindings={"target_url": "s0.url"},
                 dependencies=["s0"]),
            Step("s2", "summarize", {"content": ""},
                 input_bindings={"content": "s1.html"},
                 dependencies=["s1"]),
        ])

    def test_register_plan(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)
        assert plan.plan_id in dv._schemas
        assert "s0" in dv._schemas[plan.plan_id]

    def test_resolve_bindings(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)

        step1 = plan.steps[1]
        completed = {
            "s0": {"url": "http://example.com", "title": "Example"},
        }
        resolved = dv.resolve_bindings(step1, completed)
        assert resolved["target_url"] == "http://example.com"

    def test_resolve_bindings_missing_field(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)

        step1 = plan.steps[1]
        completed = {
            "s0": {"title": "Example"},
        }
        resolved = dv.resolve_bindings(step1, completed)
        assert resolved["target_url"] == ""

    def test_propagate_output(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)

        completed = {}
        dv.propagate_output("s0", {"url": "http://x", "title": "X"}, completed)
        assert completed["s0"]["url"] == "http://x"

    def test_validate_replan_downstream_still_valid(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)

        result = dv.validate_replan(plan, "s0", new_output_schema={"url": "str", "title": "str"})
        assert result.ok is True

    def test_validate_replan_downstream_broken(self):
        dv = DependencyValidator()
        plan = self._make_plan()
        dv.register_plan(plan)

        result = dv.validate_replan(
            plan, "s0",
            new_output_schema={"title": "str"},
        )
        assert result.ok is False
        assert any("s1" in p for p in result.problems)

    def test_register_plan_is_isolated_per_plan(self):
        dv = DependencyValidator()
        plan_a = Plan.create([
            Step("s0", "search", {}, output_schema={"url": "str"}),
            Step("s1", "fetch", {"url": ""}, input_bindings={"url": "s0.url"}, dependencies=["s0"]),
        ])
        plan_b = Plan.create([
            Step("s0", "search", {}, output_schema={"title": "str"}),
            Step("s1", "fetch", {"title": ""}, input_bindings={"title": "s0.title"}, dependencies=["s0"]),
        ])

        dv.register_plan(plan_a)
        dv.register_plan(plan_b)

        result = dv.validate_replan(plan_a, "s0")
        assert result.ok is True


# ===========================================================================
# StateManager 测试
# ===========================================================================

class TestStateManager:
    def _make_agent(self) -> Agent:
        plan = Plan.create([
            Step("s0", "search", {}),
            Step("s1", "fetch", {}, dependencies=["s0"]),
        ])
        return Agent.create(plan=plan, agent_id="a_test")

    def test_add_and_get_agent(self):
        sm = StateManager()
        agent = self._make_agent()
        sm.add_agent(agent)
        assert sm.get_agent("a_test") is agent

    def test_list_agents(self):
        sm = StateManager()
        a1 = Agent.create(plan=Plan.create([Step("s0", "a", {})]), agent_id="a1")
        a2 = Agent.create(plan=Plan.create([Step("s0", "b", {})]), agent_id="a2")
        sm.add_agent(a1)
        sm.add_agent(a2)
        assert len(sm.list_agents()) == 2

    def test_on_event_tool_result_advances_plan(self):
        sm = StateManager()
        agent = self._make_agent()
        sm.add_agent(agent)

        agent.transition(AgentState.RUNNING, "test")
        agent.transition(AgentState.WAITING, "test")

        evt = Event.create(
            agent_id="a_test",
            event_type=EventType.TOOL_RESULT,
            payload={"step_id": "s0", "tool_name": "search", "result": {}},
        )
        sm.on_event(evt)

        assert agent.state == AgentState.READY
        assert agent.current_step().step_id == "s1"
        assert agent.plan._current_index == 1

    def test_on_event_timeout进入_retrying(self):
        sm = StateManager()
        agent = self._make_agent()
        sm.add_agent(agent)

        agent.transition(AgentState.RUNNING, "test")
        agent.transition(AgentState.WAITING, "test")

        evt = Event.create(
            agent_id="a_test",
            event_type=EventType.TIMEOUT,
            payload={"step_id": "s0", "tool_name": "search", "timeout_s": 30.0},
        )
        sm.on_event(evt)

        assert agent.state == AgentState.RETRYING
        assert agent.retry_mode == RetryMode.FALLBACK_MODE

    def test_on_event_budget_exceeded进入_error(self):
        sm = StateManager()
        agent = self._make_agent()
        sm.add_agent(agent)

        evt = Event.create(
            agent_id="a_test",
            event_type=EventType.BUDGET_EXCEEDED,
            payload={"violations": ["MAX_STEPS_EXCEEDED"], "message": "steps >= 20"},
        )
        sm.on_event(evt)

        assert agent.state == AgentState.ERROR

    def test_metrics_initial(self):
        sm = StateManager()
        agent = self._make_agent()
        sm.add_agent(agent)

        m = sm.get_metrics("a_test")
        assert m is not None
        assert m.agent_id == "a_test"
        # 初始值都是 0
        assert m.step_count == 0
        assert m.llm_call_count == 0
        assert m.replan_count == 0


# ===========================================================================
# PolicyEngine 测试
# ===========================================================================

class TestPolicyEngine:
    def _make_agent(self, budget=None) -> Agent:
        plan = Plan.create([Step("s0", "x", {})])
        return Agent.create(plan=plan, budget=budget or ExecutionBudget.default())

    def test_check_transition_passes_within_budget(self):
        pe = PolicyEngine()
        agent = self._make_agent()
        result = pe.check_transition(agent, AgentState.RUNNING)
        assert result.ok is True

    def test_check_transition_fails_when_exceeded(self):
        pe = PolicyEngine()
        # max_steps=1, step_count=1 → 触发 MAX_STEPS_EXCEEDED
        agent = self._make_agent(budget=ExecutionBudget(max_steps=1))
        agent.budget_usage.consume_step()  # step_count = 1 >= max_steps = 1
        result = pe.check_transition(agent, AgentState.RUNNING)
        assert result.ok is False

    def test_check_tool_whitelist_allowed(self):
        pe = PolicyEngine()
        agent = self._make_agent(budget=ExecutionBudget(allowed_tools=["x", "y"]))
        result = pe.check_tool(agent, "x")
        assert result.ok is True

    def test_check_tool_whitelist_blocked(self):
        pe = PolicyEngine()
        agent = self._make_agent(budget=ExecutionBudget(allowed_tools=["x"]))
        result = pe.check_tool(agent, "z")
        assert result.ok is False

    def test_check_before_replan_fails_at_limit(self):
        pe = PolicyEngine()
        budget = ExecutionBudget(max_replans=1)
        agent = self._make_agent(budget=budget)
        agent.budget_usage.consume_replan()
        result = pe.check_before_replan(agent)
        assert result.ok is False


# ===========================================================================
# Scheduler e2e 测试
# ===========================================================================

class TestScheduler:
    def _make_scheduler(self):
        queue = PriorityEventQueue()
        bus = RawEventBus()
        d = Dispatcher(queue)
        d.attach(bus)
        sm = StateManager()
        llm = MockLLM(simulate_delay_s=0)
        sr = StepRunner(llm)
        dv = DependencyValidator()
        pe = PolicyEngine()
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dv,
            policy_engine=pe,
        )
        executor = ToolExecutor(bus=bus, max_workers=2)
        sched.set_tool_executor(executor)
        return sched, queue, executor

    class _DummyExecutor:
        def __init__(self):
            self.submitted = []

        def submit(self, action):
            self.submitted.append(action)

    def test_submit_task_returns_agent_id(self):
        sched, _, _ = self._make_scheduler()
        plan = Plan.create([
            Step("s0", "echo", {"v": 1}),
        ])
        agent_id = sched.submit_task(plan=plan, agent_id="agent_001")
        assert agent_id == "agent_001"

    def test_submit_task_registers_plan(self):
        sched, _, _ = self._make_scheduler()
        plan = Plan.create([
            Step("s0", "echo", {"v": 1}),
            Step("s1", "echo", {"v": 2}, dependencies=["s0"]),
        ])
        sched.submit_task(plan=plan, agent_id="a1")
        agent = sched._state_mgr.get_agent("a1")
        assert agent is not None
        assert agent.current_step().step_id == "s0"

    def test_submit_multiple_tasks(self):
        sched, _, _ = self._make_scheduler()
        for i in range(3):
            plan = Plan.create([Step(f"s0_{i}", "echo", {"v": i})])
            sched.submit_task(plan=plan, agent_id=f"a{i}")
        assert len(sched._state_mgr.list_agents()) == 3

    def test_scheduler_runs_in_thread(self):
        sched, queue, executor = self._make_scheduler()
        plan = Plan.create([Step("s0", "x", {})])
        sched.submit_task(plan=plan, agent_id="a1")
        sched.start()
        time.sleep(0.1)
        sched.stop()
        assert sched._running is False

    def test_get_status(self):
        sched, _, _ = self._make_scheduler()
        plan = Plan.create([Step("s0", "x", {})])
        sched.submit_task(plan=plan, agent_id="a1")
        status = sched.get_status()
        assert status["agent_count"] == 1
        assert len(status["agents"]) == 1
        assert status["agents"][0]["agent_id"] == "a1"

    def test_scheduler_does_not_double_route_bus_events(self):
        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=StateManager(),
            step_runner=StepRunner(MockLLM(simulate_delay_s=0)),
            dependency_validator=DependencyValidator(),
            policy_engine=PolicyEngine(),
        )

        bus.publish(Event.create(agent_id="a1", event_type=EventType.TOOL_RESULT, payload={}))
        assert queue.qsize() == 1

    def test_retrying_fallback_submits_action_and_enters_waiting(self):
        sched, _, _ = self._make_scheduler()
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([
            Step("s0", "primary", {"x": "primary"},
                 fallback_chain=[FallbackOption("backup", {"x": "fallback", "y": 1})]),
        ])
        agent = Agent.create(plan=plan, agent_id="a_retry")
        sched._state_mgr.add_agent(agent)
        sched._dep_validator.register_plan(plan)
        sched._completed_outputs[agent.agent_id] = {}

        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.transition(AgentState.RETRYING, "failed", retry_mode=RetryMode.FALLBACK_MODE)

        sched._handle_retrying(agent, Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TIMEOUT,
            payload={"step_id": "s0", "timeout_s": 1.0},
        ))

        assert agent.state == AgentState.WAITING
        assert len(dummy.submitted) == 1
        assert dummy.submitted[0].tool_name == "backup"
        assert dummy.submitted[0].params == {"x": "fallback", "y": 1}

    def test_tool_result_propagates_output_to_next_step_binding(self):
        sched, _, _ = self._make_scheduler()
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([
            Step("s0", "search", {"q": "x"}, output_schema={"url": "str"}),
            Step("s1", "fetch", {"url": ""}, input_bindings={"url": "s0.url"}, dependencies=["s0"]),
        ])
        agent = Agent.create(plan=plan, agent_id="a_bind")
        sched._state_mgr.add_agent(agent)
        sched._dep_validator.register_plan(plan)
        sched._completed_outputs[agent.agent_id] = {}

        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")

        sched._process_event(Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TOOL_RESULT,
            payload={
                "step_id": "s0",
                "tool_name": "search",
                "result": {"url": "http://example.com"},
            },
        ))

        assert sched.get_agent_outputs(agent.agent_id)["s0"]["url"] == "http://example.com"
        assert len(dummy.submitted) == 1
        assert dummy.submitted[0].tool_name == "fetch"
        assert dummy.submitted[0].params["url"] == "http://example.com"

    def test_tool_whitelist_blocks_submission_and_moves_agent_to_error(self):
        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)
        sm = StateManager()
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=StepRunner(MockLLM(simulate_delay_s=0)),
            dependency_validator=DependencyValidator(),
            policy_engine=PolicyEngine(),
        )
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([Step("s0", "blocked_tool", {})])
        agent_id = sched.submit_task(
            plan=plan,
            budget=ExecutionBudget(allowed_tools=["allowed_tool"]),
            agent_id="a_policy",
        )

        evt = queue.get_nowait()
        assert evt.event_type == EventType.BUDGET_EXCEEDED
        assert dummy.submitted == []

        sched._process_event(evt)
        agent = sm.get_agent(agent_id)
        assert agent.state == AgentState.ERROR

    def test_retrying_fallback_skips_typed_tool_with_broken_output_schema(self):
        class BadFallbackTool(Tool):
            def __init__(self):
                super().__init__(
                    name="bad_fallback",
                    description="",
                    input_schema=Schema.from_dict({}),
                    output_schema=Schema.from_dict({"html": "str"}),
                    timeout_s=11.0,
                )

            def _do_invoke(self, params):
                return {"html": "<html></html>"}

        class GoodFallbackTool(Tool):
            def __init__(self):
                super().__init__(
                    name="good_fallback",
                    description="",
                    input_schema=Schema.from_dict({}),
                    output_schema=Schema.from_dict({"url": "str"}),
                    timeout_s=6.5,
                )

            def _do_invoke(self, params):
                return {"url": "http://example.com"}

        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)
        sm = StateManager()
        dv = DependencyValidator()
        reg = ToolRegistry()
        reg.register(BadFallbackTool())
        reg.register(GoodFallbackTool())
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=StepRunner(
                MockLLM(simulate_delay_s=0),
                tool_registry=reg,
                dependency_validator=dv,
            ),
            dependency_validator=dv,
            policy_engine=PolicyEngine(),
        )
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([
            Step(
                "s0",
                "search",
                {"q": "x"},
                output_schema={"seed": "str"},
            ),
            Step(
                "s1",
                "primary_fetch",
                {"seed": ""},
                fallback_chain=[
                    FallbackOption("bad_fallback", {"seed": "x"}),
                    FallbackOption("good_fallback", {"seed": "x"}),
                ],
                output_schema={"url": "str"},
                input_bindings={"seed": "s0.seed"},
                dependencies=["s0"],
            ),
            Step(
                "s2",
                "consume",
                {"url": ""},
                input_bindings={"url": "s1.url"},
                dependencies=["s1"],
            ),
        ])
        agent = Agent.create(plan=plan, agent_id="a_typed_fallback")
        agent.plan._current_index = 1
        sm.add_agent(agent)
        dv.register_plan(plan)
        sched._completed_outputs[agent.agent_id] = {"s0": {"seed": "hello"}}

        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.transition(AgentState.RETRYING, "failed", retry_mode=RetryMode.FALLBACK_MODE)

        sched._handle_retrying(agent, Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TOOL_FAILED,
            payload={"step_id": "s1", "reason": "boom"},
        ))

        assert len(dummy.submitted) == 1
        assert dummy.submitted[0].tool_name == "good_fallback"
        assert dummy.submitted[0].timeout_s == 6.5
        assert dummy.submitted[0].params["seed"] == "hello"


# ===========================================================================
# 集成场景：完整失败 → Fallback → 成功
# ===========================================================================

class TestIntegration:
    def test_tool_result_advances_plan(self):
        """验证 StateManager 处理 TOOL_RESULT 后推进 Plan"""
        sm = StateManager()
        plan = Plan.create([
            Step("s0", "search", {}),
            Step("s1", "fetch", {}, dependencies=["s0"]),
        ])
        agent = Agent.create(plan=plan, agent_id="a1")
        sm.add_agent(agent)

        # 模拟状态机推进：READY → RUNNING → WAITING
        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submit")

        # 收到 TOOL_RESULT
        evt = Event.create(
            agent_id="a1",
            event_type=EventType.TOOL_RESULT,
            payload={"step_id": "s0", "tool_name": "search", "result": {}},
        )
        sm.on_event(evt)

        assert agent.state == AgentState.READY
        assert agent.current_step().step_id == "s1"
        assert "s0" in agent.plan.completed_step_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
