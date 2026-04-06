"""
state_machine.py — Agent 状态机

对外接口（Scheduler / StateManager 调用）:
  StateMachine.transition(to, reason)  → TransitionRecord，非法转换抛 InvalidTransitionError
  StateMachine.can_transition(to)      → bool，用于 if 判断，不抛异常
  StateMachine.state                   → 当前状态（AgentState）
  StateMachine.history                 → 完整转换记录列表（只读）
  StateMachine.retry_mode              → 当前 RETRYING 的子模式（可为 None）

对内实现:
  VALID_TRANSITIONS dict 是核心：定义所有合法的状态迁移
  transition() 检查合法性 → 记录日志 → 更新状态
  非法迁移立刻抛 InvalidTransitionError，永远不静默失败

AgentState 枚举:
  READY / RUNNING / WAITING / RETRYING / DONE / ERROR

RetryMode 枚举（RETRYING 状态的子语义）:
  FALLBACK_MODE — 还有 Fallback 工具可以尝试
  REPLAN_MODE   — Fallback 耗尽，需要 LLM 重新规划
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class AgentState(Enum):
    READY    = "READY"     # 准备好，等待 Scheduler 分配
    RUNNING  = "RUNNING"   # 正在执行（提交工具调用中）
    WAITING  = "WAITING"   # 等待工具异步结果
    RETRYING = "RETRYING"  # 工具失败，进入 Fallback / Replan 流程
    DONE     = "DONE"      # 所有步骤完成（终态）
    ERROR    = "ERROR"     # 不可恢复错误（终态）


class RetryMode(Enum):
    FALLBACK_MODE = "FALLBACK_MODE"  # 当前 Step 还有 Fallback 可用
    REPLAN_MODE   = "REPLAN_MODE"    # Fallback 耗尽，走 LLM Replan


# ---------------------------------------------------------------------------
# 合法状态迁移表
# ---------------------------------------------------------------------------

# key   = 当前状态
# value = 可以迁移到的目标状态集合
VALID_TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
    AgentState.READY:    frozenset({AgentState.RUNNING, AgentState.DONE}),
    AgentState.RUNNING:  frozenset({AgentState.WAITING, AgentState.ERROR}),
    AgentState.WAITING:  frozenset({AgentState.READY, AgentState.RETRYING, AgentState.ERROR}),
    AgentState.RETRYING: frozenset({AgentState.WAITING, AgentState.ERROR}),
    AgentState.DONE:     frozenset(),   # 终态，不再迁移
    AgentState.ERROR:    frozenset(),   # 终态，不再迁移
}

# 任意状态都可以直接跳 ERROR（系统级故障）
_GLOBAL_ERROR_TARGET: AgentState = AgentState.ERROR


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """非法状态迁移。包含足够的上下文方便排查。"""
    def __init__(
        self,
        agent_id: str,
        from_state: AgentState,
        to_state: AgentState,
        reason: str = "",
    ) -> None:
        self.agent_id   = agent_id
        self.from_state = from_state
        self.to_state   = to_state
        self.reason     = reason
        msg = (
            f"[Agent:{agent_id}] 非法状态迁移: {from_state.value} → {to_state.value}"
            + (f"（原因: {reason}）" if reason else "")
            + f"\n合法目标: {[s.value for s in VALID_TRANSITIONS.get(from_state, frozenset())]}"
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# TransitionRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionRecord:
    """
    单次状态迁移的不可变记录。
    StateManager 会把它追加到 history，CheckpointManager 会序列化它。
    """
    record_id:  str
    agent_id:   str
    from_state: AgentState
    to_state:   AgentState
    reason:     str
    timestamp:  float
    retry_mode: Optional[RetryMode] = None   # 仅 to_state=RETRYING 时有值

    @classmethod
    def create(
        cls,
        agent_id: str,
        from_state: AgentState,
        to_state: AgentState,
        reason: str,
        retry_mode: Optional[RetryMode] = None,
    ) -> "TransitionRecord":
        return cls(
            record_id=uuid.uuid4().hex[:12],
            agent_id=agent_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=time.monotonic(),
            retry_mode=retry_mode,
        )

    def to_dict(self) -> dict:
        """序列化为 dict，供 Event Log 存储。"""
        return {
            "record_id":  self.record_id,
            "agent_id":   self.agent_id,
            "from":       self.from_state.value,
            "to":         self.to_state.value,
            "reason":     self.reason,
            "timestamp":  self.timestamp,
            "retry_mode": self.retry_mode.value if self.retry_mode else None,
        }


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    单个 Agent 的状态机。

    使用方式：
        sm = StateMachine(agent_id="agent_001")
        record = sm.transition(AgentState.RUNNING, reason="Scheduler 分配")
        print(sm.state)   # AgentState.RUNNING

    线程安全：
        StateMachine 本身不加锁。
        调用方（StateManager）负责保证同一个 Agent 的更新是串行的。
    """

    def __init__(self, agent_id: str, initial_state: AgentState = AgentState.READY) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id 不能为空")
        self._agent_id:   str        = agent_id
        self._state:      AgentState = initial_state
        self._retry_mode: Optional[RetryMode] = None
        self._history:    List[TransitionRecord] = []

    # ── 核心对外接口 ───────────────────────────────────────────────────────

    def transition(
        self,
        to: AgentState,
        reason: str,
        retry_mode: Optional[RetryMode] = None,
    ) -> TransitionRecord:
        """
        执行状态迁移。

        参数:
          to         — 目标状态
          reason     — 迁移原因（写进日志，调试关键）
          retry_mode — 仅当 to=RETRYING 时有意义，区分 fallback_mode / replan_mode

        返回:
          TransitionRecord（已追加到 history）

        异常:
          InvalidTransitionError — 非法迁移，立刻抛，不静默
          ValueError             — retry_mode 使用不当

        语义约束（系统自动检查）:
          1. 终态（DONE/ERROR）不允许再迁移
          2. 非 RETRYING 目标不应传入 retry_mode
          3. RETRYING 目标必须传入 retry_mode（明确子模式）
        """
        self._validate(to, retry_mode)

        from_state = self._state
        self._state = to
        self._retry_mode = retry_mode if to == AgentState.RETRYING else None

        record = TransitionRecord.create(
            agent_id=self._agent_id,
            from_state=from_state,
            to_state=to,
            reason=reason,
            retry_mode=self._retry_mode,
        )
        self._history.append(record)
        return record

    def can_transition(self, to: AgentState) -> bool:
        """
        检查当前状态是否允许迁移到 to，不抛异常。
        Scheduler 做分支判断时使用。
        """
        if self._state in (AgentState.DONE, AgentState.ERROR):
            return to == AgentState.ERROR   # 终态唯一例外：允许再次 → ERROR（幂等）
        allowed = VALID_TRANSITIONS.get(self._state, frozenset())
        return to in allowed

    # ── 查询接口 ──────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """当前状态（只读）。"""
        return self._state

    @property
    def retry_mode(self) -> Optional[RetryMode]:
        """
        当前 RETRYING 子模式。
        非 RETRYING 状态时为 None。
        StepRunner 用来判断走 fallback 还是走 LLM Replan。
        """
        return self._retry_mode

    @property
    def history(self) -> List[TransitionRecord]:
        """
        完整转换记录（浅拷贝，防止外部篡改）。
        StateManager / CheckpointManager 读取时使用。
        """
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """是否已到达终态（DONE 或 ERROR）。"""
        return self._state in (AgentState.DONE, AgentState.ERROR)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def last_transition(self) -> Optional[TransitionRecord]:
        """最近一次迁移记录。可为 None（刚初始化未发生过迁移）。"""
        return self._history[-1] if self._history else None

    def time_in_current_state(self) -> float:
        """当前状态已持续的秒数（用于超时检测辅助）。"""
        if not self._history:
            return 0.0
        return time.monotonic() - self._history[-1].timestamp

    # ── 内部校验（不对外暴露）────────────────────────────────────────────

    def _validate(self, to: AgentState, retry_mode: Optional[RetryMode]) -> None:
        """集中做所有前置校验，有问题立刻抛。"""

        # 1. 终态检测
        if self._state in (AgentState.DONE, AgentState.ERROR):
            if to != AgentState.ERROR:
                raise InvalidTransitionError(
                    self._agent_id, self._state, to,
                    reason=f"终态 {self._state.value} 不允许再迁移（除了 → ERROR 的幂等调用）",
                )
            # DONE/ERROR → ERROR 是幂等的（允许重复注入 ERROR 事件），直接返回
            # 但我们仍然记录一条 record 方便排查
            return

        # 2. 任意非终态都允许直接 → ERROR（系统级故障、PolicyEngine 注入）
        if to == AgentState.ERROR:
            return

        # 3. 常规合法迁移检测
        allowed = VALID_TRANSITIONS.get(self._state, frozenset())
        if to not in allowed:
            raise InvalidTransitionError(self._agent_id, self._state, to)

        # 3. retry_mode 语义约束
        if to == AgentState.RETRYING and retry_mode is None:
            raise ValueError(
                f"迁移到 RETRYING 时必须传入 retry_mode（FALLBACK_MODE 或 REPLAN_MODE）"
            )
        if to != AgentState.RETRYING and retry_mode is not None:
            raise ValueError(
                f"retry_mode 只在迁移到 RETRYING 时有效，当前目标是 {to.value}"
            )
