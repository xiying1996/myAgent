"""
checkpoint_manager.py — CheckpointManager (Snapshot 管理)

职责:
  1. 定期 Snapshot（每 N 个 Step、每次 Replan 后、Task 结束时）
  2. 保存 Agent 状态 + Plan + BudgetUsage + History + Event Log 位置
  3. 管理 Snapshot 目录（JSON 文件）

对外接口:
  CheckpointManager.save_snapshot(agent, event_id)   → snapshot_id
  CheckpointManager.load_snapshot(agent_id)           → Snapshot | None
  CheckpointManager.list_snapshots(agent_id)           → List[Snapshot]
  CheckpointManager.trigger_if_needed(agent, event)   → 是否触发了 snapshot

Snapshot 内容:
  snapshot_id, agent_id, timestamp
  agent_state (READY/WAITING/...)
  plan (plan_id + 所有 step + _current_index + _completed_ids)
  budget_usage (step_count, llm_call_count, replan_count, elapsed_seconds)
  history (浅拷贝)
  last_event_id (对应 Event Log 的位置)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agent import Agent, HistoryEntry
from core.budget import ExecutionBudget
from core.plan import FallbackOption, Plan, Step
from core.state_machine import AgentState
from core.state_machine import RetryMode, TransitionRecord

logger = logging.getLogger(__name__)


def serialize_execution_budget(budget: ExecutionBudget) -> dict:
    return {
        "max_steps": budget.max_steps,
        "max_llm_calls": budget.max_llm_calls,
        "max_replans": budget.max_replans,
        "wall_clock_timeout": budget.wall_clock_timeout,
        "allowed_tools": list(budget.allowed_tools) if budget.allowed_tools is not None else None,
    }


def deserialize_execution_budget(data: dict) -> ExecutionBudget:
    return ExecutionBudget(
        max_steps=data["max_steps"],
        max_llm_calls=data["max_llm_calls"],
        max_replans=data["max_replans"],
        wall_clock_timeout=data["wall_clock_timeout"],
        allowed_tools=data.get("allowed_tools"),
    )


def serialize_plan_for_snapshot(plan: Plan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "replan_count": plan.replan_count,
        "max_replans": plan.max_replans,
        "current_index": plan._current_index,
        "completed_ids": list(plan._completed_ids),
        "steps": [
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "params": dict(step.params),
                "fallback_chain": [
                    {"tool": fb.tool, "params": dict(fb.params)}
                    for fb in step.fallback_chain
                ],
                "output_schema": dict(step.output_schema),
                "input_schema": dict(step.input_schema),
                "input_bindings": dict(step.input_bindings),
                "dependencies": list(step.dependencies),
                "fallback_index": step._fallback_index,
                "failed": step._failed,
            }
            for step in plan.steps
        ],
    }


def deserialize_plan_from_snapshot(data: dict) -> Plan:
    steps = []
    for step_data in data["steps"]:
        step = Step(
            step_id=step_data["step_id"],
            tool_name=step_data["tool_name"],
            params=dict(step_data["params"]),
            fallback_chain=[
                FallbackOption(tool=fb["tool"], params=dict(fb["params"]))
                for fb in step_data.get("fallback_chain", [])
            ],
            output_schema=dict(step_data.get("output_schema", {})),
            input_schema=dict(step_data.get("input_schema", {})),
            input_bindings=dict(step_data.get("input_bindings", {})),
            dependencies=list(step_data.get("dependencies", [])),
        )
        step._fallback_index = step_data.get("fallback_index", 0)
        step._failed = step_data.get("failed", False)
        steps.append(step)

    plan = Plan(
        plan_id=data["plan_id"],
        steps=steps,
        replan_count=data.get("replan_count", 0),
        max_replans=data.get("max_replans", 3),
    )
    plan._current_index = data.get("current_index", 0)
    plan._completed_ids = set(data.get("completed_ids", []))
    return plan


def serialize_history_entries(entries: List[HistoryEntry]) -> List[dict]:
    return [entry.to_dict() for entry in entries]


def deserialize_history_entries(data: List[dict]) -> List[HistoryEntry]:
    return [
        HistoryEntry(
            entry_id=item["entry_id"],
            kind=item["kind"],
            timestamp=item["timestamp"],
            data=dict(item.get("data", {})),
        )
        for item in data
    ]


def serialize_state_history(records: List[TransitionRecord]) -> List[dict]:
    return [record.to_dict() for record in records]


def deserialize_state_history(data: List[dict]) -> List[TransitionRecord]:
    return [
        TransitionRecord(
            record_id=item["record_id"],
            agent_id=item["agent_id"],
            from_state=AgentState(item["from"]),
            to_state=AgentState(item["to"]),
            reason=item["reason"],
            timestamp=item["timestamp"],
            retry_mode=RetryMode(item["retry_mode"]) if item.get("retry_mode") else None,
        )
        for item in data
    ]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """
    单次快照的不可变记录。
    """
    snapshot_id:   str
    agent_id:      str
    timestamp:     float
    agent_state:   str                      # AgentState.value
    retry_mode:    Optional[str]
    plan_id:       str
    plan:          Dict[str, Any]
    budget:        Dict[str, Any]
    current_index: int                      # Plan._current_index
    completed_ids: List[str]                # Plan._completed_ids
    budget_usage:  Dict[str, Any]           # BudgetUsage.snapshot()
    history:       List[Dict[str, Any]]
    state_history: List[Dict[str, Any]]
    history_len:   int
    last_event_id: str                      # Event Log 中的最后一条 event_id


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Snapshot 与 Replay 的管理。

    Snapshot 触发策略（trigger_if_needed 内部判断）:
      - 每 N 个 Step 完成（N=3，可配置）
      - 每次成功 Replan 后
      - Task 完成时（DONE / ERROR）
    """

    def __init__(
        self,
        snapshot_dir: str = ".snapshots",
        snapshot_interval_steps: int = 3,
    ) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._interval = snapshot_interval_steps

        # 内存缓存：agent_id → 最新 Snapshot
        self._latest: Dict[str, Snapshot] = {}

        # 已完成的 step 数（用于判断是否达到 snapshot 间隔）
        self._step_counts: Dict[str, int] = {}

    # ── 触发判断 ────────────────────────────────────────────────────────

    def trigger_if_needed(
        self,
        agent: Agent,
        event_id: str,
        is_replan: bool = False,
    ) -> bool:
        """
        判断是否需要触发 Snapshot。

        触发条件：
          - 每 N 个 Step 完成（spending step_count）
          - Replan 成功后
          - Agent 进入 DONE / ERROR
        """
        aid = agent.agent_id

        # 计数
        current_count = self._step_counts.get(aid, 0)
        should_snapshot = False

        if is_replan:
            # Replan 后立刻 Snapshot
            should_snapshot = True
            logger.debug("CheckpointManager: Agent %s Replan 后触发 Snapshot", aid)

        elif agent.state in (AgentState.DONE, AgentState.ERROR):
            # 终态时 Snapshot
            should_snapshot = True
            logger.debug("CheckpointManager: Agent %s 进入 %s 触发 Snapshot", aid, agent.state.value)

        else:
            completed_count = len(agent.plan.completed_step_ids)
            if completed_count - current_count >= self._interval and completed_count > 0:
                should_snapshot = True
                self._step_counts[aid] = completed_count

        if should_snapshot:
            self.save_snapshot(agent, event_id)
            return True
        return False

    # ── 保存 / 加载 ────────────────────────────────────────────────────

    def save_snapshot(self, agent: Agent, last_event_id: str) -> str:
        """
        保存 Agent 的完整快照。
        返回 snapshot_id。
        """
        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            agent_id=agent.agent_id,
            timestamp=time.time(),
            agent_state=agent.state.value,
            retry_mode=agent.retry_mode.value if agent.retry_mode else None,
            plan_id=agent.plan.plan_id,
            plan=serialize_plan_for_snapshot(agent.plan),
            budget=serialize_execution_budget(agent.budget),
            current_index=agent.plan._current_index,
            completed_ids=list(agent.plan._completed_ids),
            budget_usage=agent.budget_usage.snapshot(),
            history=serialize_history_entries(agent.history),
            state_history=serialize_state_history(agent.state_history),
            history_len=len(agent.history),
            last_event_id=last_event_id,
        )

        # 写入文件
        file_path = self._snapshot_file(agent.agent_id, snapshot_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self._snapshot_to_dict(snapshot), f, ensure_ascii=False)

        # 更新内存缓存
        self._latest[agent.agent_id] = snapshot

        logger.info(
            "CheckpointManager: 保存 Snapshot %s (agent=%s, state=%s, step=%d/%d)",
            snapshot_id, agent.agent_id, agent.state.value,
            agent.plan._current_index, len(agent.plan.steps),
        )

        return snapshot_id

    def load_snapshot(self, agent_id: str) -> Optional[Snapshot]:
        """
        加载指定 Agent 的最新 Snapshot。
        用于 Recovery Replay。
        """
        cached = self._latest.get(agent_id)
        if cached is not None:
            return cached
        snapshots = self.list_snapshots(agent_id)
        return snapshots[0] if snapshots else None

    def load_snapshot_from_file(
        self,
        agent_id: str,
        snapshot_id: str,
    ) -> Optional[Snapshot]:
        """从文件加载指定 Snapshot。"""
        file_path = self._snapshot_file(agent_id, snapshot_id)
        if not file_path.exists():
            return None
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return self._dict_to_snapshot(data)

    def list_snapshots(self, agent_id: str) -> List[Snapshot]:
        """列出指定 Agent 的所有 Snapshot（按时间倒序）。"""
        pattern = f"{agent_id}_*.json"
        files = sorted(
            self._snapshot_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        snapshots = []
        for f in files:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            snapshots.append(self._dict_to_snapshot(data))
        return snapshots

    # ── 辅助 ───────────────────────────────────────────────────────────

    def _snapshot_file(self, agent_id: str, snapshot_id: str) -> Path:
        return self._snapshot_dir / f"{agent_id}_{snapshot_id}.json"

    def _snapshot_to_dict(self, s: Snapshot) -> dict:
        return {
            "snapshot_id":   s.snapshot_id,
            "agent_id":      s.agent_id,
            "timestamp":     s.timestamp,
            "agent_state":   s.agent_state,
            "retry_mode":    s.retry_mode,
            "plan_id":       s.plan_id,
            "plan":          s.plan,
            "budget":        s.budget,
            "current_index": s.current_index,
            "completed_ids": s.completed_ids,
            "budget_usage":  s.budget_usage,
            "history":       s.history,
            "state_history": s.state_history,
            "history_len":   s.history_len,
            "last_event_id": s.last_event_id,
        }

    def _dict_to_snapshot(self, d: dict) -> Snapshot:
        return Snapshot(
            snapshot_id=d["snapshot_id"],
            agent_id=d["agent_id"],
            timestamp=d["timestamp"],
            agent_state=d["agent_state"],
            retry_mode=d.get("retry_mode"),
            plan_id=d["plan_id"],
            plan=d.get("plan", {}),
            budget=d.get("budget", serialize_execution_budget(ExecutionBudget.default())),
            current_index=d["current_index"],
            completed_ids=d["completed_ids"],
            budget_usage=d["budget_usage"],
            history=d.get("history", []),
            state_history=d.get("state_history", []),
            history_len=d["history_len"],
            last_event_id=d["last_event_id"],
        )
