"""
replay.py — Replay (Debug Replay + Recovery Replay)

支持两种恢复语义：

  Debug Replay（确定性）:
    - 使用 Event Log + PlanSnapshot（不重新调用 LLM）
    - 用于复现问题，确定性重放
    - 读取 Event Log，按顺序重放事件，遇到 LLM 调用点跳过

  Recovery Replay（从断点恢复）:
    - 使用 Snapshot + Snapshot 之后的 Event Log
    - 从断点继续执行，LLM 可重新调用
    - 用于崩溃恢复

对外接口:
  ReplayAgent.replay_from_snapshot(snapshot, event_log)    → Agent（带完整状态）
  DebugReplayAgent.replay_events(events)                    → 按事件重放（不调 LLM）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.agent import Agent
from core.budget import BudgetUsage
from core.state_machine import AgentState, RetryMode, StateMachine
from checkpoint.checkpoint_manager import (
    Snapshot,
    deserialize_execution_budget,
    deserialize_history_entries,
    deserialize_plan_from_snapshot,
    deserialize_state_history,
)
from events.event import Event
from state.state_manager import StateManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReplayMixin（恢复时重建 Agent 状态）
# ---------------------------------------------------------------------------

class ReplayMixin:
    """
    从 Snapshot + Event Log 恢复 Agent 状态的逻辑。

    使用场景：
      - Recovery Replay（崩溃恢复）
      - Debug Replay（确定性重放）

    注意：
      Plan 内部的 _fallback_index 等运行时状态无法从 Snapshot 完整恢复。
      这部分在 Replay 时由 Event Log 逐步重放还原。
    """

    @staticmethod
    def restore_from_snapshot(
        snapshot: Snapshot,
        event_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Agent:
        """
        从 Snapshot 恢复 Agent。

        步骤：
          1. 从 Snapshot 恢复 Plan（current_index + completed_ids）
          2. 从 Snapshot 恢复 BudgetUsage
          3. 从 Snapshot 之后的事件恢复 StateMachine 历史
        """
        plan = deserialize_plan_from_snapshot(snapshot.plan)
        budget = deserialize_execution_budget(snapshot.budget)
        budget_usage = BudgetUsage.from_snapshot(snapshot.budget_usage)
        sm = StateMachine(
            agent_id=snapshot.agent_id,
            initial_state=AgentState(snapshot.agent_state),
        )
        sm._retry_mode = RetryMode(snapshot.retry_mode) if snapshot.retry_mode else None
        sm._history = deserialize_state_history(snapshot.state_history)

        agent = Agent(
            agent_id=snapshot.agent_id,
            plan=plan,
            budget=budget,
            _sm=sm,
            budget_usage=budget_usage,
            _history=deserialize_history_entries(snapshot.history),
        )
        return agent


# ---------------------------------------------------------------------------
# DebugReplayAgent（确定性重放，不调 LLM）
# ---------------------------------------------------------------------------

class DebugReplayAgent:
    """
    Debug Replay：使用 Event Log + PlanSnapshot 确定性重放。

    特点：
      - 不重新调用 LLM（读取 PlanSnapshot）
      - 按顺序重放所有事件
      - 遇到 TOOL_RESULT → 推进 Plan
      - 遇到 TIMEOUT / TOOL_FAILED → 重放到下一个成功结果

    使用示例:
        replay = DebugReplayAgent(event_log=event_log)
        final_state = replay.replay_all()
    """

    def __init__(
        self,
        event_log: List[Dict[str, Any]],
        plan_snapshots: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._event_log = event_log
        self._plan_snapshots = plan_snapshots or {}

    def replay_all(self) -> Dict[str, Any]:
        """
        完整重放所有事件，返回最终状态摘要。
        """
        return self._replay_subset(self._event_log)

    def replay_until(
        self,
        agent_id: str,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        重放直到指定 event_id（不含），用于定位问题。
        """
        stopped_at = None
        subset = []
        for i, event_data in enumerate(self._event_log):
            if event_data.get("event_id") == event_id:
                stopped_at = event_data
                break
            subset.append(event_data)

        if stopped_at is None:
            raise ValueError(f"Event {event_id} 不存在于 Event Log")

        result = self._replay_subset(subset)
        result["stopped_at"] = stopped_at
        return result

    def _replay_subset(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        completed_ids: set[str] = set()
        current_index = 0
        final_state = "UNKNOWN"
        latest_plan_snapshot = None
        plan_snapshot_count = 0

        for event_data in events:
            event_type = event_data.get("event_type")
            payload = event_data.get("payload", {})

            if event_data.get("is_plan_snapshot"):
                latest_plan_snapshot = payload.get("plan")
                plan_snapshot_count += 1
                if latest_plan_snapshot:
                    current_index = latest_plan_snapshot.get("current_index", current_index)
                    completed_ids = set(latest_plan_snapshot.get("completed_ids", list(completed_ids)))
                final_state = payload.get("agent_state", final_state)
                continue

            if event_type == "TOOL_RESULT":
                step_id = payload.get("step_id", "")
                if step_id:
                    completed_ids.add(step_id)
                current_index += 1
                final_state = "READY"
            elif event_type == "BUDGET_EXCEEDED":
                final_state = "ERROR"

        if latest_plan_snapshot and latest_plan_snapshot.get("steps"):
            if current_index >= len(latest_plan_snapshot["steps"]) and final_state != "ERROR":
                final_state = "DONE"

        return {
            "completed_steps": sorted(completed_ids),
            "total_events": len(events),
            "final_index": current_index,
            "plan_snapshot_count": plan_snapshot_count,
            "latest_plan_snapshot": latest_plan_snapshot,
            "final_state": final_state,
        }


# ---------------------------------------------------------------------------
# RecoveryReplayAgent（从 Snapshot 恢复后继续执行）
# ---------------------------------------------------------------------------

class RecoveryReplayAgent:
    """
    Recovery Replay：从 Snapshot 恢复 + 继续执行。

    流程：
      1. 从 Snapshot 恢复 Agent 状态（current_index, budget_usage 等）
      2. 重放 Snapshot 之后的 Event Log
      3. 到达崩溃点后，Agent 恢复到最后一次 Snapshot 状态
      4. Scheduler 可以继续驱动执行（LLM 可重新调用）

    使用示例:
        replay = RecoveryReplayAgent(
            snapshot=snapshot,
            event_log=event_log_after_snapshot,
        )
        agent = replay.recover()
        # agent 已恢复到崩溃前状态，Scheduler 可继续驱动
    """

    def __init__(
        self,
        snapshot: Snapshot,
        event_log: List[Dict[str, Any]],
    ) -> None:
        self._snapshot = snapshot
        self._event_log = event_log

    def recover(self) -> Agent:
        """
        从 Snapshot 恢复。
        """
        agent = ReplayMixin.restore_from_snapshot(self._snapshot)
        pending = self.pending_events()

        if pending:
            state_mgr = StateManager()
            state_mgr.add_agent(agent)
            for event_data in pending:
                state_mgr.on_event(Event.from_dict(event_data))

        return agent

    def pending_events(self) -> List[Dict[str, Any]]:
        last_eid = self._snapshot.last_event_id
        pending = []
        found = False

        for event_data in self._event_log:
            if found:
                pending.append(event_data)
                continue
            if event_data.get("event_id") == last_eid:
                found = True

        return pending


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def load_event_log_from_file(log_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载 Event Log。"""
    import json
    events = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events
