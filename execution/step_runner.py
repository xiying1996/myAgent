"""
execution/step_runner.py — StepRunner

StepRunner 是 Scheduler 和 LLM 之间的桥梁，也是系统里最核心的决策模块之一。

职责（严格边界）:
  1. 根据 Agent 当前状态，决定下一个 Action（工具调用参数）
  2. 在 REPLAN_MODE 时调用 LLM，校验结果，更新 Plan
  3. 系统强制约束局部 Replan 范围（不依赖 LLM 自律）
  4. 向 BudgetUsage 记账 LLM 调用次数

不做的事（重要）:
  - 不修改 Agent 状态机
  - 不直接操作 EventQueue
  - 不知道 Scheduler 的存在
  - 不做 PolicyEngine 的工作（budget 检查由调用方做）

对外接口:
  StepRunner.decide(agent)            → Action | None
    None 表示 Plan 已完成，没有下一步要执行。

  StepRunner.validate_replan(...)     → ReplanValidationResult
    系统强制校验 LLM 返回的修改是否合法。
    合法 → 应用到 Plan；不合法 → 拒绝，记录原因。

  StepRunner.build_context(agent, reason) → ReplanContext
    构建喂给 LLM 的上下文快照（public，方便测试断言）。

执行路径（decide() 内部）:

  ┌─ 没有当前 Step ──────────────────────── return None（Plan 完成）
  │
  ├─ retry_mode == None（首次执行）─────── 用主工具构建 Action
  │
  ├─ retry_mode == FALLBACK_MODE ────────── 取下一个 Fallback 构建 Action
  │     Fallback 耗尽？→ 不在这里处理，由 Scheduler 切换到 REPLAN_MODE
  │
  └─ retry_mode == REPLAN_MODE ─────────── 调用 LLM → 校验 → 更新 Plan → 用新 Fallback 构建 Action
        LLM 失败 / give_up？→ raise ReplanFailedError → Scheduler → ERROR

校验规则（validate_replan）:
  ✅ 允许：修改当前失败 Step 的 fallback_chain
  ✅ 允许：微调后续未执行 Step 的 params（step_param_updates）
  ❌ 拒绝：修改已完成 Step 的任何字段
  ❌ 拒绝：更改 Step 顺序
  ❌ 拒绝：在 step_param_updates 里引用已完成的 Step
  ❌ 拒绝：new_fallbacks 里有空 tool_name

线程安全:
  StepRunner 实例不持有可变状态（除了 _llm.call_count 的委托），
  同一个实例可以被多个 Agent 共用，但每次 decide() 调用必须串行（由 Scheduler 保证）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.agent import Agent
from core.budget import BudgetUsage
from core.plan import FallbackOption, Plan, Step
from core.state_machine import RetryMode
from execution.llm_interface import (
    FallbackSuggestion,
    LLMCallError,
    LLMInterface,
    ReplanContext,
    ReplanResult,
    StepSnapshot,
)
from execution.tool_executor import Action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class ReplanFailedError(Exception):
    """
    Replan 无法产生有效方案时抛出。
    Scheduler 捕获后把 Agent 推入 ERROR 状态。

    原因可能是：
      - LLM 调用技术失败（LLMCallError）
      - LLM 返回 give_up()
      - LLM 返回的建议全部被校验拒绝
    """
    def __init__(self, agent_id: str, reason: str, original: Optional[Exception] = None) -> None:
        self.agent_id = agent_id
        self.reason   = reason
        self.original = original
        super().__init__(f"[Agent:{agent_id}] Replan 失败: {reason}")


class NoFallbackAvailableError(Exception):
    """
    FALLBACK_MODE 下 Fallback 已耗尽，StepRunner 无法给出下一个 Action。
    Scheduler 捕获后把 retry_mode 切换到 REPLAN_MODE 并重新调用 decide()。
    """
    def __init__(self, agent_id: str, step_id: str) -> None:
        self.agent_id = agent_id
        self.step_id  = step_id
        super().__init__(f"[Agent:{agent_id}] Step[{step_id}] Fallback 链已耗尽")


# ---------------------------------------------------------------------------
# ReplanValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ReplanValidationResult:
    """validate_replan() 的返回值，携带详细拒绝原因。"""
    ok:       bool
    reasons:  List[str]   # 违规原因列表（ok=True 时为空）

    @classmethod
    def passed(cls) -> "ReplanValidationResult":
        return cls(ok=True, reasons=[])

    @classmethod
    def failed(cls, *reasons: str) -> "ReplanValidationResult":
        return cls(ok=False, reasons=list(reasons))

    def __str__(self) -> str:
        if self.ok:
            return "ReplanValidation: OK"
        return "ReplanValidation: FAILED — " + " | ".join(self.reasons)


# ---------------------------------------------------------------------------
# FailureRecord — 单次工具失败记录
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    """记录一次工具执行失败，StepRunner 用于构建 ReplanContext。"""
    tool_name: str
    reason:    str

    def to_dict(self) -> Dict[str, str]:
        return {"tool": self.tool_name, "reason": self.reason}


# ---------------------------------------------------------------------------
# StepRunner
# ---------------------------------------------------------------------------

class StepRunner:
    """
    决定 Agent 下一步执行什么工具，并在必要时调用 LLM Replan。

    使用示例:
        llm    = MockLLM()
        runner = StepRunner(llm)

        # Scheduler 在 Agent READY 后调用：
        action = runner.decide(agent, failure_records)
        # → Action(tool_name=..., params=..., agent_id=..., step_id=...)
        # ToolExecutor 执行这个 Action
    """

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    # ── 核心接口 ──────────────────────────────────────────────────────────

    def decide(
        self,
        agent: Agent,
        failure_records: Optional[List[FailureRecord]] = None,
        last_failure_reason: str = "",
    ) -> Optional[Action]:
        """
        根据 Agent 当前状态，决定下一个要执行的 Action。

        参数:
          agent               — 当前 Agent（只读，不修改状态）
          failure_records     — 当前 Step 的历史失败记录（构建 Replan Context 用）
          last_failure_reason — 最近一次失败原因（TIMEOUT / error message 等）

        返回:
          Action  — 下一个工具调用
          None    — Plan 已全部完成，没有下一步

        抛出:
          NoFallbackAvailableError — FALLBACK_MODE 但 Fallback 已耗尽
          ReplanFailedError        — REPLAN_MODE 但 LLM 无法给出有效建议
        """
        step = agent.current_step()
        if step is None:
            return None  # Plan 已完成

        retry_mode = agent.retry_mode
        records    = failure_records or []

        if retry_mode is None:
            # ── 路径 1：首次执行此 Step，使用主工具 ──────────────────────
            return self._build_action(agent, step.tool_name, step.params)

        elif retry_mode == RetryMode.FALLBACK_MODE:
            # ── 路径 2：使用 Fallback Chain 中的下一个工具 ───────────────
            return self._decide_fallback(agent, step)

        else:
            # ── 路径 3：Fallback 耗尽，调用 LLM Replan ───────────────────
            return self._decide_replan(agent, step, records, last_failure_reason)

    # ── 校验接口（public，测试可直接调用）────────────────────────────────

    def validate_replan(
        self,
        result: ReplanResult,
        completed_step_ids: frozenset,
        current_step_id: str,
        all_step_ids: List[str],
    ) -> ReplanValidationResult:
        """
        系统强制校验 LLM 返回的修改建议。不依赖 LLM 自律。

        规则:
          R1. new_fallbacks 不能为空（give_up 在此之前已处理）
          R2. new_fallbacks 里每个 tool 不能为空字符串
          R3. step_param_updates 里引用的 step_id 必须存在于 all_step_ids
          R4. step_param_updates 里引用的 step_id 不能是已完成的 Step
          R5. step_param_updates 里引用的 step_id 不能是当前失败 Step
              （当前 Step 的 fallback 参数在 new_fallbacks 里，不在 updates 里）
          R6. step_param_updates 不允许修改 step_id / tool_name 字段
        """
        reasons = []

        # R1：必须有 fallback 建议
        if not result.new_fallbacks:
            reasons.append("R1: new_fallbacks 不能为空")
            return ReplanValidationResult.failed(*reasons)  # 后续规则无意义

        # R2：tool 不能为空
        for i, fb in enumerate(result.new_fallbacks):
            if not fb.tool or not fb.tool.strip():
                reasons.append(f"R2: new_fallbacks[{i}].tool 为空")

        # R3 + R4 + R5 + R6：step_param_updates 校验
        valid_ids = set(all_step_ids)
        for sid, updates in result.step_param_updates.items():
            if sid not in valid_ids:
                reasons.append(f"R3: step_param_updates 引用了不存在的 step_id: '{sid}'")
                continue
            if sid in completed_step_ids:
                reasons.append(f"R4: step_param_updates 尝试修改已完成的 Step: '{sid}'")
            if sid == current_step_id:
                reasons.append(f"R5: step_param_updates 不应引用当前失败 Step '{sid}'，请用 new_fallbacks")
            for forbidden_key in ("step_id", "tool_name"):
                if forbidden_key in updates:
                    reasons.append(f"R6: step_param_updates['{sid}'] 不允许修改 '{forbidden_key}' 字段")

        if reasons:
            return ReplanValidationResult.failed(*reasons)
        return ReplanValidationResult.passed()

    # ── Context 构建（public，方便测试断言 Prompt 内容）──────────────────

    def build_context(
        self,
        agent: Agent,
        last_failure_reason: str,
        failure_records: List[FailureRecord],
    ) -> ReplanContext:
        """
        把 Agent 当前状态打包成 ReplanContext 快照。
        只读操作，不修改任何运行时状态。
        """
        plan    = agent.plan
        step    = agent.current_step()
        usage   = agent.budget_usage
        budget  = agent.budget

        # 当前失败 Step 快照
        failed_snapshot = _step_to_snapshot(step)

        # 后续 Step（包含当前 Step）快照
        pending_snapshots = [_step_to_snapshot(s) for s in plan.pending_steps]

        # 预算剩余
        budget_remaining = {
            "replans_left":    budget.max_replans    - usage.replan_count,
            "llm_calls_left":  budget.max_llm_calls  - usage.llm_call_count,
            "steps_left":      budget.max_steps      - usage.step_count,
            "elapsed_s":       round(usage.elapsed_seconds(), 1),
            "timeout_s":       budget.wall_clock_timeout,
        }

        return ReplanContext(
            agent_id        = agent.agent_id,
            failed_step     = failed_snapshot,
            failure_reason  = last_failure_reason,
            failure_history = [r.to_dict() for r in failure_records],
            completed_steps = list(plan.completed_step_ids),
            pending_steps   = pending_snapshots,
            budget_remaining= budget_remaining,
        )

    # ── 内部：路径 2（Fallback）────────────────────────────────────────────

    def _decide_fallback(self, agent: Agent, step: Step) -> Action:
        """取 Fallback Chain 下一个工具，耗尽则抛 NoFallbackAvailableError。"""
        fb = step.next_fallback()
        if fb is None:
            raise NoFallbackAvailableError(agent.agent_id, step.step_id)
        logger.info(
            "[Agent:%s] Step[%s] 使用 Fallback: %s",
            agent.agent_id, step.step_id, fb.tool,
        )
        return self._build_action(agent, fb.tool, fb.params)

    # ── 内部：路径 3（Replan）─────────────────────────────────────────────

    def _decide_replan(
        self,
        agent:               Agent,
        step:                Step,
        failure_records:     List[FailureRecord],
        last_failure_reason: str,
    ) -> Action:
        """
        调用 LLM → 校验 → 应用到 Plan → 返回第一个新 Fallback 的 Action。
        失败时抛 ReplanFailedError。
        """
        context = self.build_context(agent, last_failure_reason, failure_records)

        # ── 调用 LLM ──────────────────────────────────────────────────────
        logger.info(
            "[Agent:%s] Step[%s] 进入 REPLAN_MODE，调用 LLM（第 %d 次）",
            agent.agent_id, step.step_id, self._llm.call_count + 1,
        )
        try:
            result = self._llm.propose_replan(context)
        except LLMCallError as e:
            raise ReplanFailedError(agent.agent_id, f"LLM 调用失败: {e.reason}", e)
        except Exception as e:
            raise ReplanFailedError(agent.agent_id, f"LLM 调用抛出未知异常: {e}", e)

        # 记录 LLM 调用（Budget 记账）
        agent.budget_usage.consume_llm_call()

        # ── LLM 放弃 ──────────────────────────────────────────────────────
        if result.is_empty():
            raise ReplanFailedError(
                agent.agent_id,
                f"LLM 放弃 Replan: {result.reasoning}",
            )

        # ── 校验 ──────────────────────────────────────────────────────────
        all_step_ids = [s.step_id for s in agent.plan.steps]
        validation = self.validate_replan(
            result          = result,
            completed_step_ids = agent.plan.completed_step_ids,
            current_step_id = step.step_id,
            all_step_ids    = all_step_ids,
        )
        if not validation.ok:
            raise ReplanFailedError(
                agent.agent_id,
                f"LLM 返回的 Replan 结果未通过校验: {validation}",
            )

        # ── 应用 Replan ────────────────────────────────────────────────────
        self._apply_replan(agent.plan, step, result)
        agent.budget_usage.consume_replan()

        logger.info(
            "[Agent:%s] Replan 成功，新 Fallback Chain: %s",
            agent.agent_id,
            [fb.tool for fb in result.new_fallbacks],
        )

        # ── 取第一个新 Fallback 执行 ───────────────────────────────────────
        new_fb = step.next_fallback()
        if new_fb is None:
            # 理论上不会到达（validate_replan 已保证 new_fallbacks 非空）
            raise ReplanFailedError(agent.agent_id, "Replan 后 Fallback Chain 仍为空（bug）")

        return self._build_action(agent, new_fb.tool, new_fb.params)

    # ── 内部：应用 Replan 到 Plan ─────────────────────────────────────────

    def _apply_replan(
        self,
        plan:   Plan,
        step:   Step,
        result: ReplanResult,
    ) -> None:
        """
        把校验通过的 ReplanResult 应用到 Plan：
          1. 替换当前 Step 的 fallback_chain
          2. 重置 Fallback 指针
          3. 应用 step_param_updates 到后续 Step

        只修改 fallback_chain 和 params，不修改 step_id / tool_name / dependencies。
        """
        # 1. 替换 fallback_chain（转换 FallbackSuggestion → FallbackOption）
        step.fallback_chain = [
            FallbackOption(tool=fb.tool, params=fb.params)
            for fb in result.new_fallbacks
        ]
        step.reset_fallback()

        # 2. 应用后续 Step 的参数微调
        if result.step_param_updates:
            step_map = {s.step_id: s for s in plan.steps}
            for sid, updates in result.step_param_updates.items():
                target = step_map.get(sid)
                if target is None:
                    continue  # 校验阶段已拦截，这里防御性跳过
                # 只允许更新 params，不允许改 tool_name / step_id
                for key, val in updates.items():
                    if key not in ("step_id", "tool_name"):
                        target.params[key] = val

    # ── 内部：构建 Action ─────────────────────────────────────────────────

    def _build_action(
        self,
        agent:     Agent,
        tool_name: str,
        params:    Dict[str, Any],
        timeout_s: float = 30.0,
    ) -> Action:
        """
        构建 Action，注入 agent_id / step_id。
        timeout_s 目前使用默认值，后续可从 Step 或 Budget 里读取。
        """
        step = agent.current_step()
        return Action(
            tool_name    = tool_name,
            params       = params,
            agent_id     = agent.agent_id,
            step_id      = step.step_id,
            timeout_s    = timeout_s,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _step_to_snapshot(step: Step) -> StepSnapshot:
    """把 Step 转成只读快照（不暴露内部指针）。"""
    return StepSnapshot(
        step_id        = step.step_id,
        tool_name      = step.tool_name,
        params         = dict(step.params),
        output_schema  = dict(step.output_schema),
        input_bindings = dict(step.input_bindings),
        dependencies   = list(step.dependencies),
        fallback_tools = [fb.tool for fb in step.fallback_chain],
    )
