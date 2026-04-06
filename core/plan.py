"""
plan.py — Plan / Step / FallbackOption

对外接口（其他模块会调用的）:
  Plan.current_step()       → 当前要执行的 Step（None 表示全部完成）
  Plan.advance()            → 标记当前 Step 成功，返回下一个 Step
  Plan.mark_current_failed()→ 记录当前 Step 失败（不推进）
  Plan.completed_step_ids   → 已成功的 Step id 集合（Replan 校验用）
  Plan.is_complete()        → 是否全部完成

  Step.next_fallback()      → 取下一个 Fallback 工具（None 表示耗尽）
  Step.has_fallback()       → 是否还有 Fallback 可用
  Step.reset_fallback()     → Replan 后重置 Fallback 指针

对内实现:
  Step 持有 _fallback_index 指针，每次 next_fallback() 推进一格
  Plan 持有 _current_index，advance() 推进，mark_current_failed() 不动
  Plan 在构建时做 DAG 循环检测，有环则 fail-fast
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# FallbackOption
# ---------------------------------------------------------------------------

@dataclass
class FallbackOption:
    """单个备选工具配置。"""
    tool: str
    params: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.tool or not self.tool.strip():
            raise ValueError("FallbackOption.tool 不能为空")


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """
    单个执行步骤。

    字段说明:
      step_id        — 唯一标识，用于依赖引用（如 "step0"）
      tool_name      — 主工具名（先尝试这个）
      params         — 主工具参数
      fallback_chain — 备选工具列表，按顺序尝试
      output_schema  — 输出字段类型声明，{"url": "str", "title": "str"}
      input_bindings — 输入来源，{"param_name": "step_id.field"}
      dependencies   — 依赖的 step_id 列表（用于 DAG 检测）
    """
    step_id: str
    tool_name: str
    params: Dict[str, Any]
    fallback_chain: List[FallbackOption] = field(default_factory=list)
    output_schema: Dict[str, str] = field(default_factory=dict)
    input_bindings: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    # 内部指针（不序列化）
    _fallback_index: int = field(default=0, init=False, repr=False)
    _failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.step_id or not self.step_id.strip():
            raise ValueError("Step.step_id 不能为空")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError(f"Step[{self.step_id}].tool_name 不能为空")

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def has_fallback(self) -> bool:
        """是否还有未尝试的 Fallback。"""
        return self._fallback_index < len(self.fallback_chain)

    def next_fallback(self) -> Optional[FallbackOption]:
        """
        取下一个 Fallback，同时推进内部指针。
        返回 None 表示 Fallback 链已耗尽。
        """
        if not self.has_fallback():
            return None
        option = self.fallback_chain[self._fallback_index]
        self._fallback_index += 1
        return option

    def reset_fallback(self) -> None:
        """
        Replan 后调用：重置 Fallback 指针，让新 Fallback Chain 从头开始。
        通常由 StepRunner 在写入新 fallback_chain 后调用。
        """
        self._fallback_index = 0

    def mark_failed(self) -> None:
        """记录该 Step 执行失败（主工具 + 所有 Fallback 均失败）。"""
        self._failed = True

    @property
    def exhausted(self) -> bool:
        """主工具失败 且 Fallback 全部耗尽。"""
        return self._failed and not self.has_fallback()

    def current_tool(self) -> tuple[str, Dict[str, Any]]:
        """
        返回当前应执行的工具名和参数。
        - 若 Fallback 指针为 0，返回主工具
        - 否则返回上一个已选出的 Fallback（peek，不推进）
        """
        if self._fallback_index == 0:
            return self.tool_name, self.params
        # 指针已推进过，返回最近一次 next_fallback 返回的 option
        option = self.fallback_chain[self._fallback_index - 1]
        return option.tool, option.params


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    完整执行计划。持有 Step 列表，维护当前执行位置。

    构建时自动做 DAG 循环依赖检测，发现问题立刻抛异常。
    """
    plan_id: str
    steps: List[Step]
    replan_count: int = 0
    max_replans: int = 3

    # 内部状态
    _current_index: int = field(default=0, init=False, repr=False)
    _completed_ids: Set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Plan.steps 不能为空")
        if self.max_replans < 0:
            raise ValueError("max_replans 不能为负数")
        self._validate_dag()          # 启动时 fail-fast

    # ── DAG 检测（内部）─────────────────────────────────────────────────

    def _validate_dag(self) -> None:
        """
        拓扑排序检测循环依赖。
        复杂度 O(V+E)，V=步骤数，E=依赖边数。
        """
        id_set = {s.step_id for s in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in id_set:
                    raise ValueError(
                        f"Step[{step.step_id}] 依赖了不存在的 step_id: '{dep}'"
                    )

        # Kahn 算法
        in_degree: Dict[str, int] = {s.step_id: 0 for s in self.steps}
        adj: Dict[str, List[str]] = {s.step_id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                adj[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.steps):
            raise ValueError("Plan.steps 存在循环依赖，请检查 dependencies 字段")

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def current_step(self) -> Optional[Step]:
        """
        返回当前待执行的 Step。
        None 表示所有步骤已完成（is_complete() == True）。
        """
        if self._current_index >= len(self.steps):
            return None
        return self.steps[self._current_index]

    def advance(self) -> Optional[Step]:
        """
        标记当前 Step 执行成功，推进到下一个 Step。
        返回下一个 Step，None 表示全部完成。

        外部调用者（StateManager）在收到 TOOL_RESULT 成功时调用。
        """
        current = self.current_step()
        if current is None:
            raise RuntimeError("advance() 调用时没有当前 Step（Plan 已完成）")
        self._completed_ids.add(current.step_id)
        self._current_index += 1
        return self.current_step()

    def mark_current_failed(self) -> None:
        """
        记录当前 Step 失败（主工具失败，进入 Fallback 流程）。
        不推进 _current_index，StepRunner 会继续在当前 Step 上尝试 Fallback。

        外部调用者（StateManager）在收到 TOOL_FAIL / TIMEOUT 时调用。
        """
        current = self.current_step()
        if current is None:
            raise RuntimeError("mark_current_failed() 调用时没有当前 Step")
        current.mark_failed()

    def increment_replan(self) -> None:
        """
        消耗一次 Replan 次数。PolicyEngine 调用前会先 check_replan_budget()。
        """
        self.replan_count += 1

    def replan_budget_exhausted(self) -> bool:
        """是否已超出 Replan 次数上限。"""
        return self.replan_count >= self.max_replans

    def is_complete(self) -> bool:
        """所有步骤是否全部成功完成。"""
        return self._current_index >= len(self.steps)

    @property
    def completed_step_ids(self) -> Set[str]:
        """
        已成功完成的 step_id 集合。
        StepRunner 做 Replan 校验时使用：已完成的 Step 不允许 LLM 修改。
        """
        return frozenset(self._completed_ids)

    @property
    def pending_steps(self) -> List[Step]:
        """未执行的步骤列表（包含当前正在执行的）。"""
        return self.steps[self._current_index:]

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        steps: List[Step],
        max_replans: int = 3,
        plan_id: Optional[str] = None,
    ) -> "Plan":
        """便捷构造，自动生成 plan_id。"""
        return cls(
            plan_id=plan_id or f"plan_{uuid.uuid4().hex[:8]}",
            steps=steps,
            max_replans=max_replans,
        )
