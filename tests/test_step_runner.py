"""
test_step_runner.py — StepRunner + MockLLM 测试套件

覆盖范围:
  MockLLM               — propose_replan 固定返回 fallback plan
  StepRunner.decide()   — 三条路径（首次 / FALLBACK_MODE / REPLAN_MODE）
  StepRunner.validate_replan() — 六规则校验
  StepRunner.build_context()  — ReplanContext 构建

运行:
  pytest tests/test_step_runner.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget, BudgetUsage
from core.agent import Agent
from core.state_machine import AgentState, RetryMode
from execution.llm_interface import (
    FallbackSuggestion, MockLLM, ReplanContext, ReplanResult, StepSnapshot
)
from execution.step_runner import (
    FailureRecord, NoFallbackAvailableError, ReplanFailedError,
    ReplanValidationResult, StepRunner
)


# ===========================================================================
# MockLLM 测试
# ===========================================================================

class TestMockLLM:
    def test_propose_replan_returns_fallbacks(self):
        llm = MockLLM()
        context = ReplanContext(
            agent_id="a1",
            failed_step=StepSnapshot(
                step_id="s0", tool_name="web_search", params={"q": "x"},
                output_schema={}, input_bindings={}, dependencies=[],
                fallback_tools=["bing"],
            ),
            failure_reason="timeout",
            failure_history=[{"tool": "web_search", "reason": "timeout"}],
            completed_steps=[],
            pending_steps=[],
            budget_remaining={"replans_left": 2, "llm_calls_left": 5},
        )
        result = llm.propose_replan(context)
        assert result.give_up is False
        assert len(result.new_fallbacks) == 2
        assert result.new_fallbacks[0].tool == "web_search_v2"

    def test_call_count_increments(self):
        llm = MockLLM()
        context = ReplanContext(
            agent_id="a1",
            failed_step=StepSnapshot(
                step_id="s0", tool_name="x", params={},
                output_schema={}, input_bindings={}, dependencies=[],
                fallback_tools=[],
            ),
            failure_reason="err",
            failure_history=[],
            completed_steps=[],
            pending_steps=[],
            budget_remaining={},
        )
        assert llm.call_count == 0
        llm.propose_replan(context)
        assert llm.call_count == 1

    def test_always_give_up(self):
        llm = MockLLM(always_give_up=True)
        context = ReplanContext(
            agent_id="a1",
            failed_step=StepSnapshot(
                step_id="s0", tool_name="x", params={},
                output_schema={}, input_bindings={}, dependencies=[],
                fallback_tools=[],
            ),
            failure_reason="err",
            failure_history=[],
            completed_steps=[],
            pending_steps=[],
            budget_remaining={},
        )
        result = llm.propose_replan(context)
        assert result.give_up is True
        assert result.is_empty() is True


# ===========================================================================
# StepRunner.validate_replan 测试
# ===========================================================================

class TestValidateReplan:
    def _make_result(
        self,
        new_fallbacks=None,
        step_param_updates=None,
        give_up=False,
    ) -> ReplanResult:
        return ReplanResult(
            new_fallbacks=new_fallbacks or [],
            step_param_updates=step_param_updates or {},
            reasoning="test",
            give_up=give_up,
        )

    def test_R1_empty_fallbacks_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        result = self._make_result(new_fallbacks=[])
        v = sr.validate_replan(result, frozenset(), "s0", ["s0", "s1"])
        assert v.ok is False
        assert "R1" in v.reasons[0]

    def test_R2_empty_tool_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        result = self._make_result(new_fallbacks=[FallbackSuggestion(tool="", params={})])
        v = sr.validate_replan(result, frozenset(), "s0", ["s0"])
        assert v.ok is False
        assert "R2" in v.reasons[0]

    def test_R3_invalid_step_id_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        result = self._make_result(
            new_fallbacks=[FallbackSuggestion(tool="x", params={})],
            step_param_updates={"nonexistent": {"p": 1}},
        )
        v = sr.validate_replan(result, frozenset(), "s0", ["s0"])
        assert v.ok is False
        assert "R3" in v.reasons[0]

    def test_R4_completed_step_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        result = self._make_result(
            new_fallbacks=[FallbackSuggestion(tool="x", params={})],
            step_param_updates={"s0": {"p": 1}},  # s0 已完成
        )
        v = sr.validate_replan(result, frozenset({"s0"}), "s1", ["s0", "s1"])
        assert v.ok is False
        assert "R4" in v.reasons[0]

    def test_R5_current_step_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        result = self._make_result(
            new_fallbacks=[FallbackSuggestion(tool="x", params={})],
            step_param_updates={"s0": {"p": 1}},  # s0 是当前失败 step
        )
        v = sr.validate_replan(result, frozenset(), "s0", ["s0"])
        assert v.ok is False
        assert "R5" in v.reasons[0]

    def test_R6_forbidden_field_rejected(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        # s0 已完成，s1 失败，s2 是后续步骤
        # s2 不允许改 step_id
        result = self._make_result(
            new_fallbacks=[FallbackSuggestion(tool="x", params={})],
            step_param_updates={"s2": {"step_id": "s999"}},  # 不允许改 step_id
        )
        v = sr.validate_replan(result, frozenset({"s0"}), "s1", ["s0", "s1", "s2"])
        assert v.ok is False
        assert "R6" in v.reasons[0]

    def test_valid_result_passes(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        # s0 已完成，s1 失败，s2 是后续步骤
        # 允许：修改 s1 的 fallback chain + 修改 s2 的 params
        result = self._make_result(
            new_fallbacks=[FallbackSuggestion(tool="y", params={})],
            step_param_updates={"s2": {"timeout": 30}},  # s2 是后续步骤
        )
        v = sr.validate_replan(result, frozenset({"s0"}), "s1", ["s0", "s1", "s2"])
        assert v.ok is True


# ===========================================================================
# StepRunner.decide() 测试
# ===========================================================================

class TestDecide:
    def _make_agent(
        self,
        steps=None,
        retry_mode: RetryMode = None,
    ) -> Agent:
        steps = steps or [
            Step("s0", "primary", {"q": "x"},
                 fallback_chain=[FallbackOption("backup", {})]),
        ]
        plan = Plan.create(steps)
        agent = Agent.create(plan=plan)
        if retry_mode:
            agent.transition(AgentState.RUNNING, "test")
            agent.transition(AgentState.WAITING, "test")
            agent.transition(AgentState.RETRYING, "test", retry_mode=retry_mode)
        return agent

    def test_no_current_step_returns_none(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        # Plan 已完成
        agent = self._make_agent()
        agent.plan._current_index = 99  # 假装已完成
        assert sr.decide(agent) is None

    def test_first_execution_uses_primary(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        agent = self._make_agent()
        action = sr.decide(agent)
        assert action is not None
        assert action.tool_name == "primary"
        assert action.params["q"] == "x"

    def test_fallback_mode_uses_next_fallback(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        agent = self._make_agent(retry_mode=RetryMode.FALLBACK_MODE)
        action = sr.decide(agent)
        assert action is not None
        assert action.tool_name == "backup"

    def test_fallback_exhausted_raises(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        agent = self._make_agent(
            steps=[Step("s0", "primary", {},
                       fallback_chain=[FallbackOption("fb1", {})])],
            retry_mode=RetryMode.FALLBACK_MODE,
        )
        # 消耗掉唯一的 fallback
        agent.current_step().next_fallback()
        with pytest.raises(NoFallbackAvailableError):
            sr.decide(agent)

    def test_replan_mode_calls_llm(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        agent = self._make_agent(retry_mode=RetryMode.REPLAN_MODE)
        action = sr.decide(agent)
        assert action is not None
        # MockLLM 返回 web_search_v2
        assert action.tool_name == "primary_v2"
        assert llm.call_count == 1


# ===========================================================================
# StepRunner.build_context 测试
# ===========================================================================

class TestBuildContext:
    def test_build_context_contains_required_fields(self):
        llm = MockLLM()
        sr = StepRunner(llm)
        plan = Plan.create([
            Step("s0", "web_search", {"q": "x"},
                 output_schema={"url": "str"},
                 fallback_chain=[FallbackOption("bing", {})]),
            Step("s1", "fetch", {"url": "x"},
                 dependencies=["s0"]),
        ])
        agent = Agent.create(plan=plan)

        context = sr.build_context(
            agent=agent,
            last_failure_reason="timeout",
            failure_records=[FailureRecord(tool_name="web_search", reason="timeout")],
        )

        assert context.agent_id == agent.agent_id
        assert context.failed_step.step_id == "s0"
        assert context.failed_step.tool_name == "web_search"
        assert "s0" not in context.completed_steps
        assert len(context.pending_steps) == 2
        assert context.budget_remaining["replans_left"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
