"""
state_manager.py — StateManager (Event → 状态变更的唯一入口)

StateManager 是所有 Agent 状态变更的唯一入口。

职责:
  1. 接收 Event → 更新 Agent 状态
  2. 维护完整执行历史 history[]
  3. 记录 Metrics（步骤数、LLM 调用数、总耗时）
  4. 持久化 Event Log（JSON）
  5. 触发 Snapshot（由 CheckpointManager 调用）

关键约束:
  状态更新必须通过 StateManager，任何模块不直接修改 Agent 状态字段。
  所有 Agent 的更新串行化（同一个 Agent 的两个 Event 不会并发处理）。

对外接口:
  StateManager.on_event(event)        → 处理事件，更新状态
  StateManager.get_agent(agent_id)   → 获取 Agent 实例
  StateManager.add_agent(agent)      → 注册新 Agent
  StateManager.get_metrics(agent_id) → 获取执行指标
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from core.agent import Agent, HistoryEntry
from core.plan import Plan
from core.state_machine import AgentState, InvalidTransitionError, RetryMode
from events.event import Event
from events.event_types import EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class AgentMetrics:
    """单 Agent 的执行指标。"""
    agent_id:       str
    step_count:     int = 0
    llm_call_count: int = 0
    replan_count:   int = 0
    error_count:    int = 0
    total_duration_s: float = 0.0
    start_time:     float = 0.0
    end_time:       float = 0.0

    def record_step(self) -> None:
        self.step_count += 1

    def record_llm_call(self) -> None:
        self.llm_call_count += 1

    def record_replan(self) -> None:
        self.replan_count += 1

    def record_error(self) -> None:
        self.error_count += 1

    def finalize(self) -> None:
        if self.start_time > 0 and self.end_time == 0:
            self.end_time = time.monotonic()
        if self.end_time > 0:
            self.total_duration_s = round(self.end_time - self.start_time, 3)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    所有 Agent 状态变更的唯一入口。

    线程安全：
      - _agents dict 的读写需要锁
      - 每个 Agent 的更新（on_event）是串行的
      - 不同 Agent 的 Event 可以并发处理（按 agent_id 分桶加锁）

    Event 处理顺序（on_event 内部）：
      1. 根据 event_type 分发到 _handle_* 方法
      2. _handle_* 更新 Agent 状态 + history
      3. 写入 Event Log（JSON 追加）
      4. 返回更新后的 Agent
    """

    def __init__(self, event_log_dir: Optional[str] = None) -> None:
        self._agents: Dict[str, Agent] = {}
        self._metrics: Dict[str, AgentMetrics] = {}
        self._lock = Lock()

        # Event Log 持久化
        self._event_log_dir: Optional[Path] = (
            Path(event_log_dir) if event_log_dir else None
        )
        if self._event_log_dir:
            self._event_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Agent 管理 ───────────────────────────────────────────────────────

    def add_agent(self, agent: Agent) -> None:
        """注册新 Agent。Scheduler.submit_task() 后调用。"""
        with self._lock:
            if agent.agent_id in self._agents:
                raise ValueError(f"Agent {agent.agent_id} 已存在")
            self._agents[agent.agent_id] = agent
            self._metrics[agent.agent_id] = AgentMetrics(
                agent_id=agent.agent_id,
                start_time=time.monotonic(),
            )
        logger.info("StateManager: 注册 Agent %s", agent.agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        with self._lock:
            return list(self._agents.values())

    def remove_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agents.pop(agent_id, None)
            m = self._metrics.get(agent_id)
            if m:
                m.finalize()

    # ── 指标 ─────────────────────────────────────────────────────────────

    def get_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        with self._lock:
            return self._metrics.get(agent_id)

    # ── 事件处理 ─────────────────────────────────────────────────────────

    def on_event(self, event: Event) -> Agent:
        """
        处理单个 Event，更新 Agent 状态。
        返回更新后的 Agent（供 Scheduler 后续处理）。

        事件处理表：
          TOOL_RESULT     → _handle_tool_result
          TOOL_FAILED     → _handle_tool_failed
          TIMEOUT         → _handle_timeout
          TOOL_ERROR      → _handle_tool_error
          BUDGET_EXCEEDED → _handle_budget_exceeded
          PLAN_UPDATE     → _handle_plan_update（Replan 成功）
          STATE_UPDATE    → _handle_state_update
        """
        agent = self.get_agent(event.agent_id)
        if agent is None:
            logger.warning("StateManager: 收到未知 Agent 的 Event: %s", event.agent_id)
            return None

        handler_map = {
            EventType.TOOL_RESULT:     self._handle_tool_result,
            EventType.TOOL_FAILED:     self._handle_tool_failed,
            EventType.TIMEOUT:         self._handle_timeout,
            EventType.TOOL_ERROR:      self._handle_tool_error,
            EventType.BUDGET_EXCEEDED: self._handle_budget_exceeded,
            EventType.PLAN_UPDATE:      self._handle_plan_update,
            EventType.STATE_UPDATE:    self._handle_state_update,
        }

        handler = handler_map.get(event.event_type)
        if handler is None:
            logger.debug("StateManager: 未处理 EventType %s", event.event_type)
            return agent

        try:
            handler(agent, event)
        except InvalidTransitionError:
            raise
        finally:
            self._append_event_log(event)

        return agent

    # ── 事件处理实现 ────────────────────────────────────────────────────

    def _handle_tool_result(self, agent: Agent, event: Event) -> None:
        """工具成功完成。"""
        payload = event.payload
        step_id = payload.get("step_id", "")

        # 推进 Plan
        next_step = agent.plan.advance()

        # 状态转换：WAITING → READY
        if agent.can_transition(AgentState.READY):
            agent.transition(AgentState.READY, reason=f"tool_result: {step_id} 成功")

        # 记录
        agent.add_history(HistoryEntry.create("step_success", {
            "step_id":     step_id,
            "tool_name":   payload.get("tool_name"),
            "result":      payload.get("result"),
            "duration_s":  payload.get("duration_s", 0.0),
        }))

        # 更新指标
        m = self._metrics.get(agent.agent_id)
        if m:
            m.record_step()

        logger.info(
            "[Agent:%s] Step %s 完成，next=%s",
            agent.agent_id, step_id,
            next_step.step_id if next_step else "None（Plan 完成）",
        )

    def _handle_tool_failed(self, agent: Agent, event: Event) -> None:
        """工具业务层失败。"""
        payload = event.payload
        step_id = payload.get("step_id", "")

        # 记录当前 Step 失败
        agent.plan.mark_current_failed()

        # 状态转换：WAITING → RETRYING (FALLBACK_MODE)
        if agent.can_transition(AgentState.RETRYING):
            agent.transition(
                AgentState.RETRYING,
                reason=f"tool_failed: {payload.get('reason')}",
                retry_mode=RetryMode.FALLBACK_MODE,
            )

        agent.add_history(HistoryEntry.create("step_failed", {
            "step_id":   step_id,
            "tool_name": payload.get("tool_name"),
            "reason":    payload.get("reason"),
        }))

        logger.info(
            "[Agent:%s] Step %s 失败，进入 FALLBACK_MODE",
            agent.agent_id, step_id,
        )

    def _handle_timeout(self, agent: Agent, event: Event) -> None:
        """工具执行超时。"""
        payload = event.payload
        step_id = payload.get("step_id", "")

        agent.plan.mark_current_failed()

        if agent.can_transition(AgentState.RETRYING):
            agent.transition(
                AgentState.RETRYING,
                reason=f"timeout（{payload.get('timeout_s')}s）",
                retry_mode=RetryMode.FALLBACK_MODE,
            )

        agent.add_history(HistoryEntry.create("step_timeout", {
            "step_id":   step_id,
            "tool_name": payload.get("tool_name"),
            "timeout_s": payload.get("timeout_s"),
        }))

        logger.info("[Agent:%s] Step %s 超时，进入 FALLBACK_MODE", agent.agent_id, step_id)

    def _handle_tool_error(self, agent: Agent, event: Event) -> None:
        """工具执行抛异常。"""
        payload = event.payload
        step_id = payload.get("step_id", "")

        agent.plan.mark_current_failed()

        if agent.can_transition(AgentState.RETRYING):
            agent.transition(
                AgentState.RETRYING,
                reason=f"tool_error: {payload.get('error_type')}: {payload.get('error_msg')}",
                retry_mode=RetryMode.FALLBACK_MODE,
            )

        agent.add_history(HistoryEntry.create("step_error", {
            "step_id":    step_id,
            "tool_name":  payload.get("tool_name"),
            "error_type": payload.get("error_type"),
            "error_msg":  payload.get("error_msg"),
        }))

        m = self._metrics.get(agent.agent_id)
        if m:
            m.record_error()

    def _handle_budget_exceeded(self, agent: Agent, event: Event) -> None:
        """预算耗尽。"""
        payload = event.payload

        if agent.can_transition(AgentState.ERROR):
            agent.transition(
                AgentState.ERROR,
                reason=f"budget_exceeded: {payload.get('message', '')}",
            )

        agent.add_history(HistoryEntry.create("budget_exceeded", {
            "violations": payload.get("violations", []),
            "message":    payload.get("message", ""),
        }))

        m = self._metrics.get(agent.agent_id)
        if m:
            m.finalize()

    def _handle_plan_update(self, agent: Agent, event: Event) -> None:
        """Replan 成功，Plan 更新。"""
        payload = event.payload

        agent.add_history(HistoryEntry.create("plan_update", {
            "step_id": payload.get("step_id"),
            "changes": payload.get("changes", {}),
        }))

        m = self._metrics.get(agent.agent_id)
        if m:
            m.record_replan()

    def _handle_state_update(self, agent: Agent, event: Event) -> None:
        """外部 STATE_UPDATE 事件（Agent 间依赖通知等）。"""
        # 目前是占位符，具体逻辑由 Scheduler 根据场景注入
        logger.debug(
            "[Agent:%s] STATE_UPDATE: %s",
            agent.agent_id, event.payload,
        )

    # ── Event Log 持久化 ─────────────────────────────────────────────────

    def _append_event_log(self, event: Event) -> None:
        """追加单条 Event 到 JSON 文件（按 agent_id 分文件）。"""
        if not self._event_log_dir:
            return

        log_file = self._event_log_dir / f"{event.agent_id}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("StateManager: 写入 Event Log 失败: %s", e)

    def load_event_log(self, agent_id: str) -> List[Event]:
        """从 JSONL 文件加载指定 Agent 的所有 Event（用于 Replay）。"""
        if not self._event_log_dir:
            return []

        log_file = self._event_log_dir / f"{agent_id}.jsonl"
        if not log_file.exists():
            return []

        events = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # 重建 Event 对象（简化版本，直接用 dict 代替）
                events.append(data)
        return events
