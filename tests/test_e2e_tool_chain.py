"""
test_e2e_tool_chain.py — End-to-End Tool Chain Adapter Test

Run: pytest tests/test_e2e_tool_chain.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.plan import Plan, Step, FallbackOption
from core.agent import Agent
from core.budget import ExecutionBudget
from core.state_machine import AgentState
from execution.step_runner import StepRunner
from execution.llm_interface import MockLLM
from execution.tool_executor import Action
from tools.adapter import AdapterRegistry, Adapter, BUILTIN_ADAPTERS
from tools.registry import ToolRegistry
from tools.schema import Schema
from tools.tool import Tool, ToolMetadata
from state.dependency_validator import DependencyValidator


# ── 测试工具 ────────────────────────────────────────────────────────────────


class HtmlTool(Tool):
    def __init__(self):
        super().__init__(
            name="html_fetch",
            description="Fetch HTML",
            input_schema=Schema.from_dict({"url": "str"}),
            output_schema=Schema.from_dict({"html": "str"}),
            metadata=ToolMetadata(tags=["web"]),
        )

    def _do_invoke(self, params):
        return {"html": f"<p>{params['url']}</p>"}


class TextTool(Tool):
    def __init__(self):
        super().__init__(
            name="text_fetch",
            description="Fetch text",
            input_schema=Schema.from_dict({"url": "str"}),
            output_schema=Schema.from_dict({"text": "str"}),
            metadata=ToolMetadata(tags=["web"]),
        )

    def _do_invoke(self, params):
        return {"text": params["url"]}


# ── 测试用例 ────────────────────────────────────────────────────────────────


class TestE2EToolChain:

    def test_fallback_adapter_bridges_gap(self):
        """HtmlTool 输出 {html}，html_to_text adapter 桥接 {html}→{text}"""
        adapter_reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            adapter_reg.register(a)

        path = adapter_reg.find_path(
            from_schema=Schema.from_dict({"html": "str"}),
            to_schema=Schema.from_dict({"text": "str"}),
        )
        assert len(path) == 1
        assert path[0].name == "html_to_text"

        raw = {"html": "<p>Hello</p>"}
        adapted = path[0].apply(raw)
        assert adapted == {"text": "Hello"}

    def test_fallback_problems_empty_when_adapter_available(self):
        """adapter 可桥接时 _fallback_problems 返回空"""
        adapter_reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            adapter_reg.register(a)

        tool_reg = ToolRegistry()
        tool_reg.register(HtmlTool())
        dep_val = DependencyValidator()

        step = Step(
            step_id="step0",
            tool_name="html_fetch",
            params={"url": "http://x.com"},
            output_schema={"text": "str"},
            input_schema={"url": "str"},
            fallback_chain=[
                FallbackOption(tool="html_fetch", params={"url": "http://y.com"}),
            ],
        )
        plan = Plan.create([step])
        dep_val.register_plan(plan)

        sr = StepRunner(
            llm=MockLLM(),
            tool_registry=tool_reg,
            dependency_validator=dep_val,
            adapter_registry=adapter_reg,
        )

        # HtmlTool output {html} 不满足 {text}，但 adapter 可桥接
        problems = sr._fallback_problems(plan, "step0", "html_fetch")
        assert problems == [], f"Expected no problems, got: {problems}"

    def test_fallback_rejected_when_no_adapter_path(self):
        """无 adapter 路径时 _fallback_problems 返回拒绝"""
        adapter_reg = AdapterRegistry()  # 空
        tool_reg = ToolRegistry()
        tool_reg.register(HtmlTool())
        dep_val = DependencyValidator()

        step = Step(
            step_id="step0",
            tool_name="html_fetch",
            params={"url": "http://x.com"},
            output_schema={"text": "str"},
            input_schema={"url": "str"},
        )
        plan = Plan.create([step])
        dep_val.register_plan(plan)

        sr = StepRunner(
            llm=MockLLM(),
            tool_registry=tool_reg,
            dependency_validator=dep_val,
            adapter_registry=adapter_reg,
        )

        problems = sr._fallback_problems(plan, "step0", "html_fetch")
        assert len(problems) == 1
        assert "无 adapter 路径" in problems[0]

    def test_adapter_apply_checked_raises_on_invalid_input(self):
        """adapter.apply_checked() 在输入缺必填字段时抛 ValueError"""
        adapter_reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            adapter_reg.register(a)

        html_adapter = adapter_reg.get("html_to_text")
        assert html_adapter is not None

        with pytest.raises(ValueError, match="缺少必填字段"):
            html_adapter.apply_checked({"not_html": "<p>Hi</p>"})

    def test_step_snapshot_includes_input_schema(self):
        """StepSnapshot 包含 input_schema"""
        from execution.step_runner import _step_to_snapshot

        step = Step(
            step_id="s0",
            tool_name="dummy",
            params={"value": "test"},
            input_schema={"value": "str"},
            output_schema={"value": "str"},
        )
        snap = _step_to_snapshot(step)
        assert snap.input_schema == {"value": "str"}

    def test_normalized_event_preserves_event_id(self):
        """归一化后 Event 保留原 event_id / timestamp / priority"""
        from events.event import Event
        from events.event_types import EventType, EventPriority
        from scheduler.scheduler import Scheduler
        from state.state_manager import StateManager
        from events.event_queue import PriorityEventQueue
        from events.raw_event_bus import RawEventBus

        queue = PriorityEventQueue()
        bus = RawEventBus()
        sm = StateManager()
        dep_val = DependencyValidator()
        sr = StepRunner(MockLLM())
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dep_val,
        )

        step = Step(
            step_id="s0",
            tool_name="dummy",
            params={"value": "x"},
            output_schema={"value": "str"},
            input_schema={"value": "str"},
        )
        plan = Plan.create([step])
        agent = Agent.create(plan=plan, budget=ExecutionBudget.default())
        sm.add_agent(agent)

        original_event = Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TOOL_RESULT,
            payload={
                "step_id": "s0",
                "tool_name": "dummy",
                "result": {"value": "test"},
                "adapter_chain": [],
            },
        )

        normalized = sched._normalize_tool_result_event(agent, original_event)
        assert normalized is not None
        assert normalized.event_id == original_event.event_id
        assert normalized.timestamp == original_event.timestamp
        assert normalized.priority == original_event.priority
        assert normalized.payload["result"] == {"value": "test"}

    def test_input_schema_validation_rejects_at_dispatch(self):
        """submit 前 params 不满足 input_schema → agent 直接进 ERROR"""
        from scheduler.scheduler import Scheduler
        from state.state_manager import StateManager
        from events.event_queue import PriorityEventQueue
        from events.raw_event_bus import RawEventBus

        tool_reg = ToolRegistry()
        tool_reg.register(TextTool())

        dep_val = DependencyValidator()
        sr = StepRunner(MockLLM(), tool_registry=tool_reg)

        queue = PriorityEventQueue()
        bus = RawEventBus()
        sm = StateManager()

        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dep_val,
            tool_registry=tool_reg,
        )

        # step.input_schema 声明需要 url，但 params 为空
        step = Step(
            step_id="s0",
            tool_name="text_fetch",
            params={},
            output_schema={"text": "str"},
            input_schema={"url": "str"},
        )
        plan = Plan.create([step])
        agent = Agent.create(plan=plan, budget=ExecutionBudget.default())
        sm.add_agent(agent)
        dep_val.register_plan(plan)

        sched._dispatch_agent(agent)
        assert agent.state == AgentState.ERROR

    def test_normalize_failure_emits_tool_failed_and_retries(self):
        """
        e2e：adapter.apply_checked 失败 → TOOL_FAILED → agent 进 RETRYING。

        S2 修正：agent 需要在 WAITING 状态才能处理 TOOL_FAILED → RETRYING。
        模拟流程：手动把 agent 设为 WAITING，然后发 TOOL_RESULT 触发归一化失败。
        """
        from events.event import Event
        from events.event_types import EventType
        from scheduler.scheduler import Scheduler
        from state.state_manager import StateManager
        from events.event_queue import PriorityEventQueue
        from events.raw_event_bus import RawEventBus

        adapter_reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            adapter_reg.register(a)

        tool_reg = ToolRegistry()
        tool_reg.register(HtmlTool())

        dep_val = DependencyValidator()
        sr = StepRunner(
            MockLLM(), tool_registry=tool_reg, adapter_registry=adapter_reg
        )

        queue = PriorityEventQueue()
        bus = RawEventBus()
        sm = StateManager()

        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dep_val,
            tool_registry=tool_reg,
            adapter_registry=adapter_reg,
        )

        # step.output_schema = {text: str}，但 tool 输出 {html}
        # adapter html_to_text 存在，但输入缺必填字段 "html"（result 里是 wrong_field）
        step = Step(
            step_id="s0",
            tool_name="html_fetch",
            params={"url": "http://x.com"},
            output_schema={"text": "str"},
            input_schema={"url": "str"},
        )
        plan = Plan.create([step])
        agent = Agent.create(plan=plan, budget=ExecutionBudget.default())
        sm.add_agent(agent)
        dep_val.register_plan(plan)

        # S2 修正：手动将 agent 设为 WAITING 状态
        # 模拟工具已提交、正在等待结果的场景
        agent.transition(AgentState.RUNNING, reason="模拟 dispatch")
        agent.transition(AgentState.WAITING, reason="模拟提交工具")

        # 构造 TOOL_RESULT 事件（result 缺少 "html" 字段，adapter 会失败）
        bad_result_event = Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TOOL_RESULT,
            payload={
                "step_id": "s0",
                "tool_name": "html_fetch",
                "result": {"wrong_field": "<p>Hi</p>"},  # 缺 html 字段
                "adapter_chain": ["html_to_text"],
            },
        )

        # _process_event 内部归一化失败，emit TOOL_FAILED，同步处理使 agent 进入 RETRYING
        sched._process_event(bad_result_event)

        # 检查 agent 状态为 RETRYING
        updated_agent = sm.get_agent(agent.agent_id)
        assert updated_agent is not None
        assert updated_agent.state == AgentState.RETRYING

    def test_adapter_registry_get_returns_correct_adapter(self):
        """AdapterRegistry.get(name) 返回对应 adapter"""
        adapter_reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            adapter_reg.register(a)

        adapter = adapter_reg.get("html_to_text")
        assert adapter is not None
        assert adapter.name == "html_to_text"
        assert adapter_reg.get("nonexistent") is None

    def test_normalize_returns_none_on_adapter_not_found(self):
        """adapter_chain 里引用不存在的 adapter → normalize 返回 None"""
        from events.event import Event
        from events.event_types import EventType
        from scheduler.scheduler import Scheduler
        from state.state_manager import StateManager
        from events.event_queue import PriorityEventQueue
        from events.raw_event_bus import RawEventBus

        adapter_reg = AdapterRegistry()  # 空 registry
        dep_val = DependencyValidator()
        sr = StepRunner(MockLLM())

        queue = PriorityEventQueue()
        bus = RawEventBus()
        sm = StateManager()

        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dep_val,
            adapter_registry=adapter_reg,
        )

        step = Step(
            step_id="s0",
            tool_name="dummy",
            params={"value": "x"},
            output_schema={"value": "str"},
            input_schema={"value": "str"},
        )
        plan = Plan.create([step])
        agent = Agent.create(plan=plan, budget=ExecutionBudget.default())
        sm.add_agent(agent)

        event = Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.TOOL_RESULT,
            payload={
                "step_id": "s0",
                "tool_name": "dummy",
                "result": {"value": "test"},
                "adapter_chain": ["nonexistent_adapter"],
            },
        )

        result = sched._normalize_tool_result_event(agent, event)
        assert result is None
