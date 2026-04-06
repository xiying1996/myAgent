"""
tests/test_core.py — 数据模型 + 状态机测试套件

覆盖范围:
  plan.py        — Step fallback 推进、Plan 推进、DAG 循环检测
  budget.py      — 各项预算检查、工具白名单
  state_machine  — 合法迁移、非法迁移、终态保护、retry_mode 约束
  agent.py       — 委托行为、snapshot

运行:
  pytest tests/test_core.py -v
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.plan import FallbackOption, Step, Plan
from core.budget import ExecutionBudget, BudgetUsage, BudgetViolation
from core.state_machine import (
    AgentState, RetryMode, StateMachine, InvalidTransitionError
)
from core.agent import Agent, HistoryEntry


# ===========================================================================
# plan.py 测试
# ===========================================================================

class TestFallbackOption:
    def test_empty_tool_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            FallbackOption(tool="", params={})

    def test_valid_creation(self):
        f = FallbackOption(tool="bing_search", params={"q": "test"})
        assert f.tool == "bing_search"


class TestStep:
    def _make_step(self, sid="s0", fallbacks=None):
        return Step(
            step_id=sid,
            tool_name="web_search",
            params={"query": "hello"},
            fallback_chain=fallbacks or [
                FallbackOption("bing", {}),
                FallbackOption("ddg", {}),
            ],
        )

    def test_has_fallback_initially(self):
        step = self._make_step()
        assert step.has_fallback() is True

    def test_next_fallback_advances(self):
        step = self._make_step()
        f1 = step.next_fallback()
        assert f1.tool == "bing"
        f2 = step.next_fallback()
        assert f2.tool == "ddg"
        assert step.next_fallback() is None
        assert step.has_fallback() is False

    def test_reset_fallback(self):
        step = self._make_step()
        step.next_fallback()
        step.next_fallback()
        assert step.has_fallback() is False
        step.reset_fallback()
        assert step.has_fallback() is True
        assert step.next_fallback().tool == "bing"

    def test_exhausted_after_all_fail(self):
        step = self._make_step()
        step.mark_failed()
        step.next_fallback()
        step.next_fallback()
        assert step.exhausted is True

    def test_current_tool_before_fallback(self):
        step = self._make_step()
        tool, params = step.current_tool()
        assert tool == "web_search"

    def test_current_tool_after_one_fallback(self):
        step = self._make_step()
        step.next_fallback()
        tool, _ = step.current_tool()
        assert tool == "bing"

    def test_empty_step_id_raises(self):
        with pytest.raises(ValueError, match="step_id"):
            Step(step_id="", tool_name="x", params={})


class TestPlan:
    def _make_plan(self):
        steps = [
            Step("s0", "search", {"q": "a"}, dependencies=[]),
            Step("s1", "fetch",  {"url": ""},  dependencies=["s0"]),
            Step("s2", "summarize", {},        dependencies=["s1"]),
        ]
        return Plan.create(steps)

    def test_current_step_initially_s0(self):
        plan = self._make_plan()
        assert plan.current_step().step_id == "s0"

    def test_advance_moves_to_s1(self):
        plan = self._make_plan()
        next_step = plan.advance()
        assert next_step.step_id == "s1"
        assert "s0" in plan.completed_step_ids

    def test_advance_through_all(self):
        plan = self._make_plan()
        plan.advance()   # s0 → s1
        plan.advance()   # s1 → s2
        plan.advance()   # s2 → done
        assert plan.is_complete() is True
        assert plan.current_step() is None

    def test_mark_current_failed_no_advance(self):
        plan = self._make_plan()
        plan.mark_current_failed()
        # 不应该推进
        assert plan.current_step().step_id == "s0"

    def test_replan_budget(self):
        plan = self._make_plan()
        assert plan.replan_budget_exhausted() is False
        for _ in range(plan.max_replans):
            plan.increment_replan()
        assert plan.replan_budget_exhausted() is True

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            Plan(plan_id="p", steps=[])

    def test_dag_cycle_detection(self):
        # s0 依赖 s1，s1 依赖 s0 → 循环
        with pytest.raises(ValueError, match="循环依赖"):
            Plan(plan_id="p", steps=[
                Step("s0", "a", {}, dependencies=["s1"]),
                Step("s1", "b", {}, dependencies=["s0"]),
            ])

    def test_dag_missing_dependency_raises(self):
        with pytest.raises(ValueError, match="不存在的 step_id"):
            Plan(plan_id="p", steps=[
                Step("s0", "a", {}, dependencies=["nonexistent"]),
            ])

    def test_completed_step_ids_frozen(self):
        plan = self._make_plan()
        plan.advance()
        ids = plan.completed_step_ids
        # frozenset 不能被外部修改
        assert isinstance(ids, frozenset)

    def test_pending_steps(self):
        plan = self._make_plan()
        plan.advance()  # 完成 s0
        pending = plan.pending_steps
        assert [s.step_id for s in pending] == ["s1", "s2"]


# ===========================================================================
# budget.py 测试
# ===========================================================================

class TestBudgetUsage:
    def test_initial_all_zero(self):
        u = BudgetUsage()
        assert u.step_count == 0
        assert u.llm_call_count == 0
        assert u.replan_count == 0

    def test_consume_step(self):
        u = BudgetUsage()
        u.consume_step()
        u.consume_step()
        assert u.step_count == 2

    def test_snapshot_round_trip(self):
        u = BudgetUsage()
        u.consume_step()
        u.consume_llm_call()
        u.consume_replan()
        snap = u.snapshot()
        u2 = BudgetUsage.from_snapshot(snap)
        assert u2.step_count == 1
        assert u2.llm_call_count == 1
        assert u2.replan_count == 1


class TestExecutionBudget:
    def test_default_passes_empty_usage(self):
        b = ExecutionBudget.default()
        u = BudgetUsage()
        result = b.check(u)
        assert result.ok is True

    def test_step_exceeded(self):
        b = ExecutionBudget(max_steps=2)
        u = BudgetUsage()
        u.consume_step(); u.consume_step()
        result = b.check(u)
        assert not result.ok
        assert BudgetViolation.MAX_STEPS_EXCEEDED in result.violations

    def test_llm_calls_exceeded(self):
        b = ExecutionBudget(max_llm_calls=1)
        u = BudgetUsage()
        u.consume_llm_call()
        result = b.check(u)
        assert BudgetViolation.MAX_LLM_CALLS_EXCEEDED in result.violations

    def test_replan_exceeded(self):
        b = ExecutionBudget(max_replans=1)
        u = BudgetUsage()
        u.consume_replan()
        result = b.check(u)
        assert BudgetViolation.MAX_REPLANS_EXCEEDED in result.violations

    def test_multiple_violations_collected(self):
        b = ExecutionBudget(max_steps=1, max_llm_calls=1)
        u = BudgetUsage()
        u.consume_step(); u.consume_llm_call()
        result = b.check(u)
        assert not result.ok
        assert len(result.violations) == 2

    def test_tool_whitelist_allowed(self):
        b = ExecutionBudget(allowed_tools=["web_search", "calculator"])
        result = b.check_tool("web_search")
        assert result.ok

    def test_tool_whitelist_blocked(self):
        b = ExecutionBudget(allowed_tools=["web_search"])
        result = b.check_tool("shell_exec")
        assert not result.ok
        assert BudgetViolation.TOOL_NOT_ALLOWED in result.violations

    def test_no_whitelist_allows_all(self):
        b = ExecutionBudget(allowed_tools=None)
        assert b.check_tool("any_tool_name").ok

    def test_check_before_replan(self):
        b = ExecutionBudget(max_replans=2)
        u = BudgetUsage()
        u.consume_replan(); u.consume_replan()
        result = b.check_before_replan(u)
        assert not result.ok
        assert BudgetViolation.MAX_REPLANS_EXCEEDED in result.violations

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            ExecutionBudget(max_steps=0)
        with pytest.raises(ValueError):
            ExecutionBudget(max_replans=-1)


# ===========================================================================
# state_machine.py 测试
# ===========================================================================

class TestStateMachine:
    def _make_sm(self) -> StateMachine:
        return StateMachine(agent_id="test_agent")

    # 初始状态
    def test_initial_state_is_ready(self):
        sm = self._make_sm()
        assert sm.state == AgentState.READY

    # 合法迁移
    def test_ready_to_running(self):
        sm = self._make_sm()
        record = sm.transition(AgentState.RUNNING, reason="Scheduler 分配")
        assert sm.state == AgentState.RUNNING
        assert record.from_state == AgentState.READY
        assert record.to_state == AgentState.RUNNING

    def test_full_happy_path(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "tool submitted")
        sm.transition(AgentState.READY,   "tool success")
        sm.transition(AgentState.DONE,    "all done")
        assert sm.state == AgentState.DONE
        assert sm.is_terminal is True

    def test_retrying_fallback_mode(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "tool submitted")
        sm.transition(AgentState.RETRYING, "timeout", retry_mode=RetryMode.FALLBACK_MODE)
        assert sm.state == AgentState.RETRYING
        assert sm.retry_mode == RetryMode.FALLBACK_MODE

    def test_retrying_replan_mode(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "tool submitted")
        sm.transition(AgentState.RETRYING, "fallback exhausted", retry_mode=RetryMode.REPLAN_MODE)
        assert sm.retry_mode == RetryMode.REPLAN_MODE

    def test_retry_mode_clears_after_leaving_retrying(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "submitted")
        sm.transition(AgentState.RETRYING, "fail", retry_mode=RetryMode.FALLBACK_MODE)
        sm.transition(AgentState.WAITING, "fallback submitted")
        assert sm.retry_mode is None

    # 非法迁移
    def test_invalid_ready_to_waiting_raises(self):
        sm = self._make_sm()
        with pytest.raises(InvalidTransitionError):
            sm.transition(AgentState.WAITING, "invalid")

    def test_invalid_ready_to_error_allowed(self):
        # 任意状态都可以 → ERROR（系统级故障）
        sm = self._make_sm()
        sm.transition(AgentState.ERROR, "system crash")
        assert sm.state == AgentState.ERROR

    def test_terminal_done_blocks_transition(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "submitted")
        sm.transition(AgentState.READY, "success")
        sm.transition(AgentState.DONE, "complete")
        with pytest.raises(InvalidTransitionError):
            sm.transition(AgentState.RUNNING, "should fail")

    def test_terminal_error_blocks_transition(self):
        sm = self._make_sm()
        sm.transition(AgentState.ERROR, "crash")
        with pytest.raises(InvalidTransitionError):
            sm.transition(AgentState.READY, "should fail")

    # retry_mode 约束
    def test_retrying_without_retry_mode_raises(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "submitted")
        with pytest.raises(ValueError, match="retry_mode"):
            sm.transition(AgentState.RETRYING, "fail")  # 缺少 retry_mode

    def test_non_retrying_with_retry_mode_raises(self):
        sm = self._make_sm()
        with pytest.raises(ValueError, match="retry_mode"):
            sm.transition(AgentState.RUNNING, "start", retry_mode=RetryMode.FALLBACK_MODE)

    # history
    def test_history_records_all_transitions(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        sm.transition(AgentState.WAITING, "submitted")
        assert len(sm.history) == 2
        assert sm.history[0].to_state == AgentState.RUNNING

    def test_history_is_copy(self):
        sm = self._make_sm()
        sm.transition(AgentState.RUNNING, "start")
        h = sm.history
        h.clear()   # 修改副本
        assert len(sm.history) == 1   # 原始不受影响

    def test_can_transition_true(self):
        sm = self._make_sm()
        assert sm.can_transition(AgentState.RUNNING) is True

    def test_can_transition_false(self):
        sm = self._make_sm()
        assert sm.can_transition(AgentState.WAITING) is False

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError):
            StateMachine(agent_id="")


# ===========================================================================
# agent.py 测试
# ===========================================================================

class TestAgent:
    def _make_plan(self):
        return Plan.create([
            Step("s0", "tool_a", {}),
            Step("s1", "tool_b", {}, dependencies=["s0"]),
        ])

    def test_create_with_defaults(self):
        agent = Agent.create(plan=self._make_plan())
        assert agent.state == AgentState.READY
        assert agent.agent_id.startswith("agent_")

    def test_state_delegates_to_sm(self):
        agent = Agent.create(plan=self._make_plan())
        agent.transition(AgentState.RUNNING, "test")
        assert agent.state == AgentState.RUNNING

    def test_current_step_delegates_to_plan(self):
        agent = Agent.create(plan=self._make_plan())
        assert agent.current_step().step_id == "s0"

    def test_is_complete_false_initially(self):
        agent = Agent.create(plan=self._make_plan())
        assert agent.is_complete() is False

    def test_is_terminal_false_initially(self):
        agent = Agent.create(plan=self._make_plan())
        assert agent.is_terminal() is False

    def test_add_history(self):
        agent = Agent.create(plan=self._make_plan())
        entry = HistoryEntry.create("step_started", {"step_id": "s0"})
        agent.add_history(entry)
        assert len(agent.history) >= 1

    def test_snapshot_contains_key_fields(self):
        agent = Agent.create(plan=self._make_plan(), task_id="task_001")
        snap = agent.snapshot()
        assert snap["agent_id"] == agent.agent_id
        assert snap["state"] == "READY"
        assert snap["task_id"] == "task_001"
        assert snap["current_step"] == "s0"

    def test_invalid_transition_propagates(self):
        agent = Agent.create(plan=self._make_plan())
        with pytest.raises(InvalidTransitionError):
            agent.transition(AgentState.WAITING, "invalid")

    def test_transition_appends_to_history(self):
        agent = Agent.create(plan=self._make_plan())
        agent.transition(AgentState.RUNNING, "start")
        # state_change 应该在 history 里
        kinds = [e.kind for e in agent.history]
        assert "state_change" in kinds

    def test_repr(self):
        agent = Agent.create(plan=self._make_plan())
        r = repr(agent)
        assert "READY" in r
        assert "s0" in r


# ===========================================================================
# 集成场景：完整失败 → Fallback → 成功路径
# ===========================================================================

class TestIntegration:
    def test_step_fail_then_fallback_then_success(self):
        """
        模拟: Step0 主工具失败 → 使用 Fallback → 成功 → 推进到 Step1
        """
        plan = Plan.create([
            Step("s0", "primary_tool", {},
                 fallback_chain=[FallbackOption("backup_tool", {})]),
            Step("s1", "final_tool", {}, dependencies=["s0"]),
        ])
        agent = Agent.create(plan=plan)

        # 开始执行
        agent.transition(AgentState.RUNNING, "Scheduler 分配 s0")
        agent.transition(AgentState.WAITING, "提交 primary_tool")

        # 主工具失败
        agent.transition(AgentState.RETRYING, "primary_tool 超时",
                         retry_mode=RetryMode.FALLBACK_MODE)
        plan.mark_current_failed()

        # 取 Fallback
        fb = plan.current_step().next_fallback()
        assert fb.tool == "backup_tool"

        # 执行 Fallback
        agent.transition(AgentState.WAITING, "提交 backup_tool")

        # Fallback 成功
        agent.transition(AgentState.READY, "backup_tool 成功")
        next_step = plan.advance()   # 推进 s0 → s1

        assert next_step.step_id == "s1"
        assert agent.state == AgentState.READY
        assert "s0" in plan.completed_step_ids

    def test_budget_exhaustion_leads_to_error(self):
        """
        模拟: Replan 次数耗尽 → PolicyEngine 注入 ERROR
        """
        budget = ExecutionBudget(max_replans=2)
        usage = BudgetUsage()
        plan = Plan.create([Step("s0", "tool", {})])
        agent = Agent.create(plan=plan, budget=budget)

        usage.consume_replan()
        usage.consume_replan()

        result = budget.check_before_replan(usage)
        assert not result.ok
        assert BudgetViolation.MAX_REPLANS_EXCEEDED in result.violations

        # PolicyEngine 会做这个动作：
        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.transition(AgentState.RETRYING, "fail", retry_mode=RetryMode.REPLAN_MODE)
        agent.transition(AgentState.ERROR, "budget exhausted")

        assert agent.is_terminal() is True
        assert agent.state == AgentState.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
