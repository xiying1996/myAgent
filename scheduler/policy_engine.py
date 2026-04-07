"""
policy_engine.py — PolicyEngine (Budget 安全边界)

PolicyEngine 在 Scheduler 的状态转换前运行，是系统的安全边界。

对外接口:
  PolicyEngine.check_transition(agent, to_state) → 违规时抛出 PolicyViolation
  PolicyEngine.check_tool(agent, tool_name)      → 工具白名单检查
  PolicyEngine.check_budget(agent)               → 全量预算检查（每次 Event 处理前调用）

设计原则:
  - 所有检查不抛异常只返回结果，由 Scheduler 决定是否 raise
  - 但最严重的违规（BUDGET_EXCEEDED）会直接注入 BUDGET_EXCEEDED 事件
  - 幂等：check() 本身不修改任何状态
"""

from __future__ import annotations

import logging
from typing import List, Optional

from core.agent import Agent
from core.budget import BudgetCheckResult, BudgetViolation, ExecutionBudget
from core.state_machine import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class PolicyViolation(Exception):
    """
    违反 PolicyEngine 约束时抛出。
    Scheduler 捕获后注入相应事件或直接推入 ERROR。
    """
    def __init__(self, message: str, violations: List[BudgetViolation]) -> None:
        self.message    = message
        self.violations = violations
        super().__init__(message)


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    预算 + 白名单的安全边界。

    Scheduler 在每次状态转换前调用 check_transition()。
    ToolExecutor 在提交工具前调用 check_tool()。
    """

    def __init__(self) -> None:
        pass

    # ── 状态转换前检查 ───────────────────────────────────────────────────

    def check_transition(self, agent: Agent, to_state: AgentState) -> BudgetCheckResult:
        """
        状态转换前的预算检查。
        返回 BudgetCheckResult，ok=True → 允许转换。

        检查顺序（优先级从高到低）:
          1. wall_clock_timeout（硬超时）
          2. max_total_steps
          3. max_llm_calls
          4. max_replans
        """
        budget  = agent.budget
        usage   = agent.budget_usage
        result  = budget.check(usage)

        if not result.ok:
            logger.warning(
                "[Agent:%s] PolicyEngine 拦截转换 %s → %s: %s",
                agent.agent_id,
                agent.state.value,
                to_state.value,
                result.message,
            )

        return result

    # ── 工具白名单检查 ──────────────────────────────────────────────────

    def check_tool(self, agent: Agent, tool_name: str) -> BudgetCheckResult:
        """
        工具白名单检查。
        ToolExecutor.submit() 前调用，拦截不合法的工具调用。
        """
        return agent.budget.check_tool(tool_name)

    # ── Replan 前专项检查 ────────────────────────────────────────────────

    def check_before_replan(self, agent: Agent) -> BudgetCheckResult:
        """
        进入 REPLAN_MODE 前的专项检查。
        StepRunner 在调用 LLM 前调用，快速 fail-fast。
        """
        return agent.budget.check_before_replan(agent.budget_usage)

    # ── 辅助：是否应注入 BUDGET_EXCEEDED ─────────────────────────────────

    def should_inject_budget_exceeded(self, agent: Agent) -> bool:
        """
        判断是否应注入 BUDGET_EXCEEDED 事件。
        由 Scheduler 在处理高优先级 Event 前调用。
        """
        result = agent.budget.check(agent.budget_usage)
        return not result.ok

    # ── 辅助：提取违规消息 ──────────────────────────────────────────────

    @staticmethod
    def format_violations(result: BudgetCheckResult) -> str:
        """格式化违规消息，供日志 / Event payload 使用。"""
        return result.message or ""
