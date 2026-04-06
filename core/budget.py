"""
budget.py — ExecutionBudget / BudgetUsage

职责分离：
  ExecutionBudget  — 上限配置（只读，不变）
  BudgetUsage      — 当前消耗记录（可变）
  BudgetViolation  — 违规原因枚举
  BudgetCheckResult— check() 的返回值，携带违规原因

对外接口（PolicyEngine 调用）:
  BudgetUsage.consume_step()       → 消耗一个步骤计数
  BudgetUsage.consume_llm_call()   → 消耗一次 LLM 调用
  BudgetUsage.consume_replan()     → 消耗一次 Replan 次数
  ExecutionBudget.check(usage)     → BudgetCheckResult（ok 或 violation）
  ExecutionBudget.check_tool(name) → bool（工具是否在白名单）

设计原则:
  - Budget 不抛异常，只返回 BudgetCheckResult
  - 真正抛异常的是 PolicyEngine（它决定要不要 raise）
  - 这样 Budget 可以被测试，也可以被 dry-run（只检查不执行）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ---------------------------------------------------------------------------
# BudgetViolation
# ---------------------------------------------------------------------------

class BudgetViolation(Enum):
    """预算违规类型，PolicyEngine 用来决定注入哪种 Event。"""
    MAX_STEPS_EXCEEDED      = auto()   # 总步骤数超限
    MAX_LLM_CALLS_EXCEEDED  = auto()   # LLM 调用次数超限
    MAX_REPLANS_EXCEEDED    = auto()   # Replan 次数超限
    WALL_CLOCK_TIMEOUT      = auto()   # 整体执行超时
    TOOL_NOT_ALLOWED        = auto()   # 工具不在白名单


# ---------------------------------------------------------------------------
# BudgetCheckResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetCheckResult:
    """
    check() 的返回值。
    ok=True  → 可以继续执行
    ok=False → violations 列表说明为什么不行
    """
    ok: bool
    violations: List[BudgetViolation] = field(default_factory=list)
    message: str = ""

    @classmethod
    def passed(cls) -> "BudgetCheckResult":
        return cls(ok=True)

    @classmethod
    def failed(cls, violations: List[BudgetViolation], message: str = "") -> "BudgetCheckResult":
        return cls(ok=False, violations=violations, message=message)


# ---------------------------------------------------------------------------
# BudgetUsage  （可变，记录消耗）
# ---------------------------------------------------------------------------

@dataclass
class BudgetUsage:
    """
    记录当前 Task 的资源消耗。
    由 StateManager 持有，每次操作后更新。
    CheckpointManager 会把它序列化进 Snapshot。
    """
    step_count:    int   = 0
    llm_call_count: int  = 0
    replan_count:  int   = 0
    start_time:    float = field(default_factory=time.monotonic)

    # ── 消耗接口（StateManager 调用）─────────────────────────────────────

    def consume_step(self) -> None:
        """每次 RUNNING → WAITING 时调用（提交工具调用时）。"""
        self.step_count += 1

    def consume_llm_call(self) -> None:
        """每次 StepRunner 调用 LLM 时调用。"""
        self.llm_call_count += 1

    def consume_replan(self) -> None:
        """每次进入 replan_mode 时调用。"""
        self.replan_count += 1

    # ── 查询接口 ─────────────────────────────────────────────────────────

    def elapsed_seconds(self) -> float:
        """任务已运行的秒数。"""
        return time.monotonic() - self.start_time

    def snapshot(self) -> dict:
        """序列化为 dict，供 CheckpointManager 保存。"""
        return {
            "step_count":     self.step_count,
            "llm_call_count": self.llm_call_count,
            "replan_count":   self.replan_count,
            "elapsed_seconds": round(self.elapsed_seconds(), 2),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "BudgetUsage":
        """从 Snapshot dict 恢复（Recovery Replay 使用）。"""
        usage = cls()
        usage.step_count     = data.get("step_count", 0)
        usage.llm_call_count = data.get("llm_call_count", 0)
        usage.replan_count   = data.get("replan_count", 0)
        # start_time 在恢复时重置为当前时刻，elapsed 从 0 重新计
        return usage


# ---------------------------------------------------------------------------
# ExecutionBudget  （不可变，只读配置）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionBudget:
    """
    预算上限配置。创建后不可修改。
    由任务提交方（Task Submission API）在创建 Agent 时传入。

    allowed_tools=None  → 不限制（所有工具都允许）
    allowed_tools=[...] → 白名单模式，不在列表中的工具被拒绝
    """
    max_steps:           int            = 20
    max_llm_calls:       int            = 10
    max_replans:         int            = 3
    wall_clock_timeout:  float          = 300.0      # 秒
    allowed_tools:       Optional[List[str]] = None  # None = 不限制

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须 > 0")
        if self.max_llm_calls <= 0:
            raise ValueError("max_llm_calls 必须 > 0")
        if self.max_replans < 0:
            raise ValueError("max_replans 不能为负数")
        if self.wall_clock_timeout <= 0:
            raise ValueError("wall_clock_timeout 必须 > 0")

    # ── 核心检查接口（PolicyEngine 调用）─────────────────────────────────

    def check(self, usage: BudgetUsage) -> BudgetCheckResult:
        """
        全量检查：一次性收集所有违规，返回 BudgetCheckResult。
        PolicyEngine 拿到结果后决定是否注入 BUDGET_EXCEEDED 事件。

        注意：check() 本身不修改任何状态，幂等且无副作用。
        """
        violations = []
        messages = []

        if usage.step_count >= self.max_steps:
            violations.append(BudgetViolation.MAX_STEPS_EXCEEDED)
            messages.append(
                f"步骤数 {usage.step_count} >= 上限 {self.max_steps}"
            )

        if usage.llm_call_count >= self.max_llm_calls:
            violations.append(BudgetViolation.MAX_LLM_CALLS_EXCEEDED)
            messages.append(
                f"LLM 调用 {usage.llm_call_count} >= 上限 {self.max_llm_calls}"
            )

        if usage.replan_count >= self.max_replans:
            violations.append(BudgetViolation.MAX_REPLANS_EXCEEDED)
            messages.append(
                f"Replan 次数 {usage.replan_count} >= 上限 {self.max_replans}"
            )

        elapsed = usage.elapsed_seconds()
        if elapsed >= self.wall_clock_timeout:
            violations.append(BudgetViolation.WALL_CLOCK_TIMEOUT)
            messages.append(
                f"已运行 {elapsed:.1f}s >= 超时上限 {self.wall_clock_timeout}s"
            )

        if violations:
            return BudgetCheckResult.failed(violations, " | ".join(messages))
        return BudgetCheckResult.passed()

    def check_before_replan(self, usage: BudgetUsage) -> BudgetCheckResult:
        """
        Replan 前的专项检查（只检查 replan_count 和 llm_calls）。
        StepRunner 进入 replan_mode 前调用，快速 fail-fast。
        """
        violations = []
        messages = []

        if usage.replan_count >= self.max_replans:
            violations.append(BudgetViolation.MAX_REPLANS_EXCEEDED)
            messages.append(
                f"Replan 已达上限 {self.max_replans}，不再允许 LLM 重新规划"
            )

        if usage.llm_call_count >= self.max_llm_calls:
            violations.append(BudgetViolation.MAX_LLM_CALLS_EXCEEDED)
            messages.append(
                f"LLM 调用次数已达上限 {self.max_llm_calls}"
            )

        if violations:
            return BudgetCheckResult.failed(violations, " | ".join(messages))
        return BudgetCheckResult.passed()

    def check_tool(self, tool_name: str) -> BudgetCheckResult:
        """
        工具白名单检查。
        ToolExecutor 提交工具前调用，拦截不合法的工具调用。
        """
        if self.allowed_tools is None:
            return BudgetCheckResult.passed()
        if tool_name not in self.allowed_tools:
            return BudgetCheckResult.failed(
                [BudgetViolation.TOOL_NOT_ALLOWED],
                f"工具 '{tool_name}' 不在白名单 {self.allowed_tools} 中",
            )
        return BudgetCheckResult.passed()

    # ── 工厂方法 ─────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ExecutionBudget":
        """开发 / 测试用默认预算。"""
        return cls()

    @classmethod
    def strict(cls, allowed_tools: List[str]) -> "ExecutionBudget":
        """生产用严格预算，指定工具白名单。"""
        return cls(
            max_steps=10,
            max_llm_calls=5,
            max_replans=2,
            wall_clock_timeout=120.0,
            allowed_tools=allowed_tools,
        )
