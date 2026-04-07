"""
agent.py — Agent 数据结构

Agent 是执行的基本单元，职责极简：
  持有状态机、Plan、Budget 消耗记录、执行历史。
  不做任何业务决策，所有判断交给 Scheduler / StepRunner / PolicyEngine。

对外接口（Scheduler / StateManager 读取）:
  agent.state              → 当前 AgentState（委托给 StateMachine）
  agent.current_step()     → 当前 Step（委托给 Plan）
  agent.is_complete()      → Plan 是否全部完成
  agent.is_terminal()      → 是否到达终态（DONE / ERROR）
  agent.transition(...)    → 委托 StateMachine.transition()
  agent.add_history(entry) → 追加执行历史条目
  agent.snapshot()         → 序列化为 dict（CheckpointManager 用）

设计约束:
  - Agent 不直接修改自己的 state 字段，所有状态变更通过 state_machine.transition()
  - Agent 不调用 LLM，不调用工具，不做调度决策
  - StateManager 是唯一有权更新 Agent 内部状态的模块
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import BudgetUsage, ExecutionBudget
from .plan import Plan, Step
from .state_machine import AgentState, RetryMode, StateMachine, TransitionRecord


# ---------------------------------------------------------------------------
# HistoryEntry — 执行历史的单条记录
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    """
    记录 Agent 执行过程中的一个事件。
    StateManager 每次处理 Event 后追加一条。

    kind 取值示例:
      "step_started"  — 开始执行某个 Step
      "step_success"  — Step 成功
      "step_failed"   — Step 失败（主工具）
      "fallback_used" — 使用了 Fallback 工具
      "replan"        — LLM 重新规划
      "state_change"  — 状态机迁移
    """
    entry_id:  str
    kind:      str
    timestamp: float
    data:      Dict[str, Any]

    @classmethod
    def create(cls, kind: str, data: Dict[str, Any]) -> "HistoryEntry":
        return cls(
            entry_id=uuid.uuid4().hex[:10],
            kind=kind,
            timestamp=time.monotonic(),
            data=data,
        )

    def to_dict(self) -> dict:
        return {
            "entry_id":  self.entry_id,
            "kind":      self.kind,
            "timestamp": self.timestamp,
            "data":      self.data,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """
    Agent 是执行单元的数据容器。

    创建方式推荐使用 Agent.create() 工厂方法。
    直接构造时需要自行传入 StateMachine 实例。
    """
    agent_id:     str
    plan:         Plan
    budget:       ExecutionBudget
    _sm:          StateMachine          # 不暴露为公开字段，通过属性访问
    budget_usage: BudgetUsage = field(default_factory=BudgetUsage)
    _history:     List[HistoryEntry]   = field(default_factory=list)
    created_at:   float                = field(default_factory=time.monotonic)
    task_id:      Optional[str]        = None   # 关联的上层任务 ID

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        plan: Plan,
        budget: Optional[ExecutionBudget] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> "Agent":
        """
        推荐的创建方式。自动生成 agent_id，初始化 StateMachine。

        示例:
            agent = Agent.create(plan=my_plan, budget=ExecutionBudget.default())
        """
        _id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        return cls(
            agent_id=_id,
            plan=plan,
            budget=budget or ExecutionBudget.default(),
            _sm=StateMachine(agent_id=_id),
            task_id=task_id,
        )

    # ── 状态委托（核心）──────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """当前状态，委托给 StateMachine（唯一真相来源）。"""
        return self._sm.state

    @property
    def retry_mode(self) -> Optional[RetryMode]:
        """当前 RETRYING 子模式，非 RETRYING 时为 None。"""
        return self._sm.retry_mode

    def transition(
        self,
        to: AgentState,
        reason: str,
        retry_mode: Optional[RetryMode] = None,
    ) -> TransitionRecord:
        """
        状态迁移。实际逻辑在 StateMachine 里。
        StateManager 是唯一应该调用此方法的模块。

        非法迁移会抛 InvalidTransitionError，让 Scheduler 感知并处理。
        """
        record = self._sm.transition(to, reason, retry_mode)
        # 状态变更自动写入执行历史
        self._history.append(
            HistoryEntry.create("state_change", record.to_dict())
        )
        return record

    def can_transition(self, to: AgentState) -> bool:
        """非抛异常版本的合法性检查，Scheduler 分支判断用。"""
        return self._sm.can_transition(to)

    def switch_retry_mode(
        self,
        retry_mode: RetryMode,
        reason: str,
    ) -> TransitionRecord:
        """
        在保持 RETRYING 状态不变的前提下切换其子模式。
        由 Scheduler 在 fallback 链耗尽后进入 replan_mode 时调用。
        """
        record = self._sm.switch_retry_mode(retry_mode, reason)
        self._history.append(
            HistoryEntry.create("state_change", record.to_dict())
        )
        return record

    # ── Plan 委托 ──────────────────────────────────────────────────────────

    def current_step(self) -> Optional[Step]:
        """当前待执行的 Step，None 表示全部完成。"""
        return self.plan.current_step()

    def is_complete(self) -> bool:
        """Plan 是否全部完成（所有步骤成功）。"""
        return self.plan.is_complete()

    def is_terminal(self) -> bool:
        """是否到达终态（DONE 或 ERROR）。"""
        return self._sm.is_terminal

    # ── 历史记录 ──────────────────────────────────────────────────────────

    def add_history(self, entry: HistoryEntry) -> None:
        """
        追加一条执行历史。
        StateManager 每处理一个 Event 后调用，供 Replay / Debug 使用。
        """
        self._history.append(entry)

    @property
    def history(self) -> List[HistoryEntry]:
        """完整执行历史（浅拷贝）。"""
        return list(self._history)

    @property
    def state_history(self) -> list:
        """仅状态迁移记录，来自 StateMachine。"""
        return self._sm.history

    # ── 序列化（CheckpointManager 使用）──────────────────────────────────

    def snapshot(self) -> dict:
        """
        序列化为 dict，供 CheckpointManager 保存 Snapshot。

        注意：Plan 内部的 _fallback_index、_current_index 等运行时状态
        需要单独处理，此处只存结构性信息。
        完整序列化方案由 CheckpointManager 负责（不在 Agent 层做）。
        """
        return {
            "agent_id":     self.agent_id,
            "task_id":      self.task_id,
            "state":        self.state.value,
            "plan_id":      self.plan.plan_id,
            "created_at":   self.created_at,
            "budget_usage": self.budget_usage.snapshot(),
            "history_len":  len(self._history),
            "current_step": (
                self.current_step().step_id if self.current_step() else None
            ),
        }

    def __repr__(self) -> str:
        step = self.current_step()
        return (
            f"Agent(id={self.agent_id!r}, "
            f"state={self.state.value}, "
            f"step={step.step_id if step else 'None'}, "
            f"replan={self.budget_usage.replan_count}/{self.budget.max_replans})"
        )
