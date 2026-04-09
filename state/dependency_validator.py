"""
dependency_validator.py — DependencyValidator (DAG binding 校验 + 输出传播)

职责:
  1. 启动时检测 Plan 的循环依赖（由 Plan._validate_dag() 负责）
  2. Replan 后检查 Step 依赖链是否仍然有效
  3. 执行时把上游 Step 的输出绑定到下游 Step 的参数（resolve_bindings）
  4. 检测下游 Step 是否引用了不存在的上游字段

对外接口:
  DependencyValidator.validate_replan(plan, step_id, new_fallback_chain) → ValidationResult
    在 LLM 返回 new_fallbacks 后、StepRunner 应用之前调用。

  DependencyValidator.resolve_bindings(step, completed_outputs) → Dict[str, Any]
    在 ToolExecutor 提交工具前，把 input_bindings 解析为实际参数值。

  DependencyValidator.propagate_output(step_id, output, completed_outputs)
    在 Step 成功完成后，保存其输出，供下游 Step 使用。

设计原则:
  - DependencyValidator 是只读校验层，不修改 Plan 结构
  - 发现问题返回 ValidationResult（不抛异常）
  - resolve_bindings 是纯函数，输入相同输出相同（幂等）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from core.plan import FallbackOption, Plan, Step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class BindingValidationResult:
    """
    input_bindings 校验结果。
    ok=True  → 所有 binding 都合法
    ok=False → problems 列表说明哪些 binding 失效
    """
    ok:       bool
    problems: List[str]   # ["Step[s1].input_bindings['url'] 引用了 step0 不存在的字段 'title'"]

    @classmethod
    def passed(cls) -> "BindingValidationResult":
        return cls(ok=True, problems=[])

    @classmethod
    def failed(cls, *problems: str) -> "BindingValidationResult":
        return cls(ok=False, problems=list(problems))


# ---------------------------------------------------------------------------
# DependencyValidator
# ---------------------------------------------------------------------------

class DependencyValidator:
    """
    Step 依赖链的校验与绑定解析。

    使用示例:
        validator = DependencyValidator()

        # 校验 Replan 后的 fallback chain 是否影响下游
        result = validator.validate_replan(
            plan=agent.plan,
            step_id="step0",
            new_output_schema={"url": "str", "title": "str"},
        )

        # 执行前解析 bindings
        resolved_params = validator.resolve_bindings(
            step=step1,
            completed_outputs={
                "step0": {"url": "http://...", "title": "Hello"},
            },
        )

        # Step 成功后传播输出
        validator.propagate_output(
            step_id="step0",
            output={"url": "http://...", "title": "Hello"},
            completed_outputs=...,  # 会直接修改传入的 dict
        )
    """

    def __init__(self) -> None:
        # plan_id → {step_id → output_schema}
        self._schemas: Dict[str, Dict[str, Dict[str, str]]] = {}

    # ── 初始化 ───────────────────────────────────────────────────────────

    def register_plan(self, plan: Plan) -> None:
        """
        注册 Plan，构建 step_id → output_schema 映射。
        Plan 创建后立刻调用（Scheduler.submit_task 时）。
        """
        self._schemas[plan.plan_id] = {
            step.step_id: dict(step.output_schema)
            for step in plan.steps
        }

    # ── Replan 校验 ──────────────────────────────────────────────────────

    def validate_replan(
        self,
        plan: Plan,
        step_id: str,
        new_output_schema: Optional[Dict[str, str]] = None,
    ) -> BindingValidationResult:
        """
        Replan 后检查依赖链是否仍然有效。

        检查逻辑：
          step_id 换了工具 → output_schema 可能变化
          → 检查所有下游 Step 的 input_bindings 是否仍能解析
          → 若引用了 step_id 不存在的字段 → 失败

        参数:
          plan              — Agent 的 Plan
          step_id           — 被 Replan 的 Step ID
          new_output_schema — 新的 output_schema（None = 无变化）

        注意：new_output_schema 目前是占位参数，
        真实场景需要从 ReplanResult 传入新工具的 output_schema。
        """
        problems: List[str] = []
        registered_schemas = self._schemas.setdefault(
            plan.plan_id,
            {step.step_id: dict(step.output_schema) for step in plan.steps},
        )
        schemas = {
            sid: dict(schema)
            for sid, schema in registered_schemas.items()
        }

        # 构建 step_id → Step 的 map
        step_map: Dict[str, Step] = {s.step_id: s for s in plan.steps}

        # 找到 step_id 对应的 Step 的所有下游
        downstream_ids = self._get_downstream(step_id, plan)
        logger.debug(
            "DependencyValidator: step %s 的下游: %s",
            step_id, downstream_ids,
        )

        # 更新 schema（如果提供了新的）
        if new_output_schema is not None:
            schemas[step_id] = dict(new_output_schema)

        # 检查每个下游 Step 的 input_bindings
        for sid in downstream_ids:
            downstream = step_map[sid]
            for param_name, binding in downstream.input_bindings.items():
                # binding 格式: "step_id.field"
                source_id, field_name = self._parse_binding(binding)
                if source_id != step_id:
                    continue

                # 检查字段是否存在于 schema
                if field_name not in schemas.get(source_id, {}):
                    problems.append(
                        f"Step[{sid}].input_bindings['{param_name}'] "
                        f"引用了 step '{source_id}' 不存在的字段 '{field_name}'"
                    )
                    logger.warning(
                        "DependencyValidator: 发现失效 binding → Step[%s].%s → %s.%s",
                        sid, param_name, source_id, field_name,
                    )

        if problems:
            return BindingValidationResult.failed(*problems)
        return BindingValidationResult.passed()

    # ── Binding 解析 ────────────────────────────────────────────────────

    def resolve_bindings(
        self,
        step: Step,
        completed_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        把 Step 的 input_bindings 解析为实际参数值。

        binding 格式: "step_id.field"
        例: {"target_url": "step0.url", "query": "step1.search_term"}

        参数:
          step              — 当前要执行的 Step
          completed_outputs — {step_id: {field: value}} 已完成 Step 的输出

        返回:
          解析后的 params dict（会在 ToolExecutor.submit() 前合并到 params）

        如果 binding 引用了不存在的 step_id 或字段：
          → 记录 warning，返回原始 params（不解析）
        """
        if not step.input_bindings:
            return dict(step.params)

        resolved = dict(step.params)

        for param_name, binding in step.input_bindings.items():
            source_id, field_name = self._parse_binding(binding)

            source_output = completed_outputs.get(source_id, {})
            if field_name in source_output:
                resolved[param_name] = source_output[field_name]
                logger.debug(
                    "DependencyValidator: 解析 %s.%s → %s = %s",
                    step.step_id, param_name, binding, resolved[param_name],
                )
            else:
                logger.warning(
                    "DependencyValidator: Step[%s] 无法解析 binding '%s': "
                    "step '%s' 没有字段 '%s'（可用字段: %s）",
                    step.step_id, binding, source_id, field_name,
                    list(source_output.keys()),
                )
                # 不覆盖原始 params，保持原样

        return resolved

    def propagate_output(
        self,
        step_id: str,
        output: Dict[str, Any],
        completed_outputs: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Step 成功后，把输出保存到 completed_outputs（供下游 resolve_bindings 使用）。
        """
        completed_outputs[step_id] = output
        logger.debug(
            "DependencyValidator: step %s 输出已传播: %s",
            step_id, list(output.keys()),
        )

    # ── 辅助方法 ─────────────────────────────────────────────────────────

    def _parse_binding(self, binding: str) -> tuple[str, str]:
        """
        解析 binding 字符串，返回 (step_id, field_name)。
        binding 格式必须是 "step_id.field"。
        """
        if "." not in binding:
            return binding, ""
        parts = binding.rsplit(".", 1)
        return parts[0], parts[1]

    def _get_downstream(self, step_id: str, plan: Plan) -> List[str]:
        """
        用 Kahn 算法找所有下游 Step（依赖 step_id 的 Step）。
        """
        step_map: Dict[str, Step] = {s.step_id: s for s in plan.steps}

        # 构建邻接表（dep → dependents）
        adj: Dict[str, List[str]] = {s.step_id: [] for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                adj[dep].append(step.step_id)

        # BFS/DFS 找所有可达节点
        visited: Set[str] = set()
        queue = [step_id]
        while queue:
            node = queue.pop(0)
            for dep in adj.get(node, []):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        return list(visited)
