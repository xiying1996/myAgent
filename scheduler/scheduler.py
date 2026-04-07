"""
scheduler.py — Scheduler (主循环)

Scheduler 是系统的调度中枢，负责：

  1. 维护所有 Agent 的生命周期
  2. 消费 EventQueue（优先消费 HighPriority）
  3. 根据 Event 触发状态转换（每次转换前调用 PolicyEngine）
  4. 唤醒处于 READY 状态的 Agent，交给 StepRunner
  5. 维护已完成 Step 的输出（completed_outputs），供 DependencyValidator 使用

关键原则：
  Scheduler 不做业务决策，只做调度决策。
  所有业务判断委托给 StepRunner / PolicyEngine / DependencyValidator。

事件处理流程：
  Event 到达
    → StateManager.on_event(Event) → 更新 Agent 状态 + history
    → Scheduler 收到 StateManager 返回的 Agent
    → 根据 Agent 状态决定下一步：
        READY   → StepRunner.decide() → Action → ToolExecutor.submit()
        WAITING → 等待（不操作）
        RETRYING → 走 Fallback 或 Replan 路径
        DONE    → 清理 / 触发 Checkpoint
        ERROR   → 清理 / 触发 Checkpoint

对外接口：
  Scheduler.submit_task(plan, budget, task_id) → agent_id
  Scheduler.start()                            → 启动主循环
  Scheduler.stop()                             → 停止主循环
  Scheduler.get_status()                       → 返回调度状态摘要
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from core.agent import Agent
from core.budget import BudgetCheckResult
from core.budget import ExecutionBudget
from core.plan import Plan
from core.state_machine import AgentState, InvalidTransitionError, RetryMode
from checkpoint.checkpoint_manager import CheckpointManager, serialize_plan_for_snapshot
from events.event import Event
from events.event_queue import PriorityEventQueue, QueueEmpty
from events.event_types import EventType
from events.raw_event_bus import RawEventBus
from execution.step_runner import NoFallbackAvailableError, ReplanFailedError, StepRunner
from execution.tool_executor import Action
from scheduler.policy_engine import PolicyEngine
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    Agent 调度中枢。

    使用示例:
        queue  = PriorityEventQueue()
        bus    = RawEventBus(queue)
        sm     = StateManager()
        sr     = StepRunner(MockLLM())
        dv     = DependencyValidator()
        sched  = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dv,
        )

        # 提交任务
        agent_id = sched.submit_task(plan=my_plan, task_id="task_001")

        # 启动调度循环（在单独线程中）
        sched.start()
    """

    def __init__(
        self,
        event_queue: PriorityEventQueue,
        raw_event_bus: RawEventBus,
        state_manager: StateManager,
        step_runner: StepRunner,
        dependency_validator: DependencyValidator,
        policy_engine: Optional[PolicyEngine] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self._queue      = event_queue
        self._bus        = raw_event_bus
        self._state_mgr  = state_manager
        self._step_runner = step_runner
        self._dep_validator = dependency_validator
        self._policy     = policy_engine or PolicyEngine()
        self._checkpoint_mgr = checkpoint_manager

        self._running    = False
        self._thread: Optional[threading.Thread] = None

        # 每个 Agent 的已完成 Step 输出（agent_id → {step_id → output}）
        self._completed_outputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._tool_executor: Optional[Any] = None

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """在单独线程中启动调度主循环。"""
        if self._running:
            logger.warning("Scheduler 已启动，忽略重复调用")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="SchedulerLoop", daemon=True)
        self._thread.start()
        logger.info("Scheduler: 主循环已启动")

    def stop(self) -> None:
        """停止调度主循环。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler: 主循环已停止")

    # ── 任务提交 ─────────────────────────────────────────────────────────

    def submit_task(
        self,
        plan: Plan,
        budget: Optional[ExecutionBudget] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        提交一个 Agent 执行计划。
        返回生成的 agent_id。
        """
        budget = budget or ExecutionBudget.default()
        agent = Agent.create(
            plan=plan,
            budget=budget,
            task_id=task_id,
            agent_id=agent_id,
        )

        self._state_mgr.add_agent(agent)
        self._dep_validator.register_plan(plan)
        self._completed_outputs[agent.agent_id] = {}

        logger.info(
            "Scheduler: 提交任务 task=%s agent=%s plan=%s (%d steps)",
            task_id, agent.agent_id, plan.plan_id, len(plan.steps),
        )

        # Agent 初始状态是 READY，立刻尝试调度
        self._dispatch_agent(agent)

        return agent.agent_id

    # ── 主循环 ──────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Scheduler 主循环。"""
        while self._running:
            try:
                event = self._queue.get_next_event(timeout=0.05)
            except QueueEmpty:
                # timeout：检查是否有 READY 的 Agent 需要调度
                self._tick()
                continue

            self._process_event(event)

    def _process_event(self, event: Event) -> None:
        """处理单个事件。"""
        logger.debug("Scheduler: 处理 Event %s (agent=%s)", event.event_type, event.agent_id)

        try:
            agent = self._state_mgr.on_event(event)
        except InvalidTransitionError as e:
            logger.warning("Scheduler: 非法状态转换 %s", e)
            return

        if agent is None:
            return

        if event.event_type == EventType.TOOL_RESULT:
            self._record_completed_output(agent, event)

        # ── Budget 检查（高优先级事件后）─────────────────────────────────
        if (
            event.is_high_priority
            and event.event_type != EventType.BUDGET_EXCEEDED
            and not agent.is_terminal()
        ):
            if self._policy.should_inject_budget_exceeded(agent):
                self._inject_budget_exceeded(agent)
                return

        # ── 状态引导 ──────────────────────────────────────────────────────
        self._引导_agent(agent, event)

        if self._checkpoint_mgr is not None:
            self._checkpoint_mgr.trigger_if_needed(
                agent,
                event.event_id,
                is_replan=event.is_plan_snapshot,
            )

    def _tick(self) -> None:
        """
        每 timeout 检查一次 READY Agent，尝试调度。
        主要是捕获从 WAITING/RETRYING 进入 READY 的 Agent。
        """
        for agent in self._state_mgr.list_agents():
            if agent.state == AgentState.READY and not agent.is_terminal():
                self._dispatch_agent(agent)

    def _引导_agent(self, agent: Agent, event: Event) -> None:
        """
        根据 Agent 当前状态决定下一步操作。
        这是 Scheduler 唯一的"业务决策"点。
        """
        state = agent.state

        if state == AgentState.READY:
            self._dispatch_agent(agent)

        elif state == AgentState.WAITING:
            # 等待中，不操作
            pass

        elif state == AgentState.RETRYING:
            self._handle_retrying(agent, event)

        elif state in (AgentState.DONE, AgentState.ERROR):
            self._handle_terminal(agent)

    # ── 调度动作 ─────────────────────────────────────────────────────────

    def _dispatch_agent(self, agent: Agent) -> None:
        """
        唤醒 READY/RETRYING Agent，交给 StepRunner 决定下一步 Action。
        """
        step = agent.current_step()
        if step is None:
            # Plan 已完成
            if agent.can_transition(AgentState.DONE):
                agent.transition(AgentState.DONE, reason="所有 Step 已完成")
            return

        state = agent.state
        if state not in (AgentState.READY, AgentState.RETRYING):
            return

        # PolicyEngine: 检查 Budget
        target_state = AgentState.RUNNING if state == AgentState.READY else AgentState.WAITING
        check = self._policy.check_transition(agent, target_state)
        if not check.ok:
            self._inject_budget_exceeded(agent, check)
            return

        is_replan_dispatch = (
            state == AgentState.RETRYING
            and agent.retry_mode == RetryMode.REPLAN_MODE
        )

        # 状态转换：READY → RUNNING
        if state == AgentState.READY and agent.can_transition(AgentState.RUNNING):
            agent.transition(AgentState.RUNNING, reason="Scheduler 分配执行")
        elif state == AgentState.READY:
            return

        # StepRunner 决定 Action
        try:
            action = self._step_runner.decide(agent)
        except NoFallbackAvailableError:
            # Fallback 耗尽 → 切换到 REPLAN_MODE
            self._switch_to_replan_mode(agent)
            return
        except ReplanFailedError as e:
            logger.error("Scheduler: Agent %s Replan 失败: %s", agent.agent_id, e)
            agent.transition(AgentState.ERROR, reason=f"replan_failed: {e.reason}")
            return

        if action is None:
            # Plan 完成
            agent.transition(AgentState.DONE, reason="所有 Step 已完成")
            return

        tool_check = self._policy.check_tool(agent, action.tool_name)
        if not tool_check.ok:
            self._inject_budget_exceeded(agent, tool_check)
            return

        # 只把 input_bindings 解析出来的字段覆盖到 action.params，避免主 step 的
        # 默认 params 把 fallback/replan 专属参数覆盖回去。
        completed = self._completed_outputs.get(agent.agent_id, {})
        resolved_params = self._dep_validator.resolve_bindings(step, completed)
        bound_params = {
            param_name: resolved_params[param_name]
            for param_name in step.input_bindings
            if param_name in resolved_params
        }

        resolved_action = Action(
            tool_name=action.tool_name,
            params={**action.params, **bound_params},
            agent_id=action.agent_id,
            step_id=action.step_id,
            timeout_s=action.timeout_s,
        )

        # 状态转换：READY 路径是 RUNNING → WAITING，RETRYING 路径是 RETRYING → WAITING
        if agent.can_transition(AgentState.WAITING):
            if state == AgentState.READY:
                agent.transition(AgentState.WAITING, reason=f"提交工具 {resolved_action.tool_name}")
            else:
                agent.transition(AgentState.WAITING, reason=f"重试提交工具 {resolved_action.tool_name}")
        else:
            return

        if is_replan_dispatch:
            self._publish_plan_snapshot(agent, step.step_id)

        # 消耗 Step 计数
        agent.budget_usage.consume_step()

        # 提交到 ToolExecutor（异步，不阻塞）
        # 注意：ToolExecutor 持有 RawEventBus 引用，已在 __init__ 中绑定
        if self._tool_executor is not None:
            self._tool_executor.submit(resolved_action)

    def _handle_retrying(self, agent: Agent, event: Event) -> None:
        """
        处理 RETRYING 状态的 Agent。
        - FALLBACK_MODE：直接取下一个 Fallback 执行
        - REPLAN_MODE：调用 StepRunner 的 Replan 路径
        """
        if agent.retry_mode == RetryMode.FALLBACK_MODE:
            self._dispatch_agent(agent)
        elif agent.retry_mode == RetryMode.REPLAN_MODE:
            self._dispatch_agent(agent)

    def _handle_terminal(self, agent: Agent) -> None:
        """终态 Agent 的清理。"""
        logger.info(
            "Scheduler: Agent %s 进入终态 %s",
            agent.agent_id, agent.state.value,
        )
        # 清理输出缓存
        self._completed_outputs.pop(agent.agent_id, None)

    def _switch_to_replan_mode(self, agent: Agent) -> None:
        """
        Fallback 耗尽，切换到 REPLAN_MODE。
        """
        # 检查 Budget
        check = self._policy.check_before_replan(agent)
        if not check.ok:
            self._inject_budget_exceeded(agent, check)
            return

        agent.switch_retry_mode(RetryMode.REPLAN_MODE, "fallback 耗尽")

        # 继续调度（会进入 Replan 路径）
        self._dispatch_agent(agent)

    # ── Budget 事件注入 ──────────────────────────────────────────────────

    def _inject_budget_exceeded(
        self,
        agent: Agent,
        result: Optional[BudgetCheckResult] = None,
    ) -> None:
        """注入 BUDGET_EXCEEDED 事件。"""
        result = result or agent.budget.check(agent.budget_usage)
        event = Event.create(
            event_type=EventType.BUDGET_EXCEEDED,
            agent_id=agent.agent_id,
            payload={
                "violations": [v.name for v in result.violations],
                "message":    result.message,
            },
        )
        self._bus.publish(event)

    def _record_completed_output(self, agent: Agent, event: Event) -> None:
        """
        保存已完成 Step 的输出，供后续 input_bindings 解析使用。
        """
        step_id = event.payload.get("step_id")
        if not step_id:
            return

        result = event.payload.get("result")
        if isinstance(result, dict):
            output = result
        else:
            output = {"value": result}

        completed_outputs = self._completed_outputs.setdefault(agent.agent_id, {})
        self._dep_validator.propagate_output(step_id, output, completed_outputs)

    def _publish_plan_snapshot(self, agent: Agent, step_id: str) -> None:
        """
        把 replan 后的完整 Plan 快照记录为 PLAN_UPDATE 事件，供 Debug Replay 使用。
        """
        event = Event.create(
            agent_id=agent.agent_id,
            event_type=EventType.PLAN_UPDATE,
            payload={
                "step_id": step_id,
                "changes": {
                    "replanned_step": step_id,
                    "fallback_tools": [
                        fb["tool"]
                        for fb in serialize_plan_for_snapshot(agent.plan)["steps"][agent.plan._current_index]["fallback_chain"]
                    ] if agent.current_step() else [],
                },
                "plan": serialize_plan_for_snapshot(agent.plan),
                "agent_state": agent.state.value,
                "retry_mode": agent.retry_mode.value if agent.retry_mode else None,
            },
            is_plan_snapshot=True,
        )
        self._bus.publish(event)

    # ── 工具执行器绑定 ──────────────────────────────────────────────────

    def set_tool_executor(self, executor: Any) -> None:
        """
        绑定 ToolExecutor（解决循环依赖）。
        Scheduler 持有引用用于提交 Action。
        """
        self._tool_executor = executor

    # ── 状态查询 ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """返回调度状态摘要（供监控/调试用）。"""
        agents = self._state_mgr.list_agents()
        return {
            "running":  self._running,
            "agent_count": len(agents),
            "queue_size": self._queue.qsize(),
            "high_priority_size": self._queue.high_priority_size,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "state":    a.state.value,
                    "step":     a.current_step().step_id if a.current_step() else None,
                    "is_terminal": a.is_terminal(),
                }
                for a in agents
            ],
        }

    def get_agent_outputs(self, agent_id: str) -> Dict[str, Dict[str, Any]]:
        """获取 Agent 的已完成 Step 输出。"""
        return self._completed_outputs.get(agent_id, {})
