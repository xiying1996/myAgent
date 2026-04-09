"""
llm_interface.py — LLM Interface + Mock Implementation

对外接口:
  LLMInterface           — 抽象基类，定义与 LLM 交互的协议
  MockLLM                — 开发/测试用 Mock 实现，固定返回 fallback plan
  FallbackSuggestion      — 单条 Fallback 建议
  ReplanContext          — 喂给 LLM 的上下文快照（StepRunner.build_context() 构建）
  ReplanResult           — LLM 的返回结果
  StepSnapshot           — Step 的只读快照（不暴露内部指针）
  LLMCallError           — LLM 调用失败异常

LLM 边界（由 StepRunner 强制校验）:
  ✅ 允许：修改当前失败 Step 的 fallback_chain
  ✅ 允许：微调后续未执行 Step 的 params（step_param_updates）
  ❌ 拒绝：修改已完成 Step 的任何字段
  ❌ 拒绝：更改 Step 顺序
  ❌ 拒绝：在 step_param_updates 里引用已完成的 Step
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepSnapshot:
    """
    Step 的只读快照。
    不暴露 _fallback_index、_failed 等内部指针。
    StepRunner 用它构建 ReplanContext。
    """
    step_id:        str
    tool_name:      str
    params:         Dict[str, Any]
    output_schema:  Dict[str, str]
    input_schema:   Dict[str, str]
    input_bindings: Dict[str, str]
    dependencies:   List[str]
    fallback_tools: List[str]          # 当前 fallback chain 里的工具名列表


@dataclass
class FallbackSuggestion:
    """LLM 返回的单条 Fallback 建议。"""
    tool: str
    params: Dict[str, Any]


@dataclass
class ReplanResult:
    """
    LLM 返回的重规划结果。
    StepRunner 负责校验，不信任原始输出。
    """
    new_fallbacks:      List[FallbackSuggestion]   # 替换当前 Step 的 fallback chain
    step_param_updates: Dict[str, Dict[str, Any]]  # sid → {param: value}
    reasoning:          str                         # LLM 的思考过程（供日志/debug）
    give_up:            bool = False                # True = LLM 放弃，建议直接 ERROR
    raw_response:       Optional[str] = None        # Provider 原始输出（供 replay/audit）

    def is_empty(self) -> bool:
        """LLM 放弃时返回 True。"""
        return self.give_up or not self.new_fallbacks

    def to_dict(self, *, include_raw_response: bool = True) -> Dict[str, Any]:
        data = {
            "new_fallbacks": [
                {"tool": fb.tool, "params": dict(fb.params)}
                for fb in self.new_fallbacks
            ],
            "step_param_updates": {
                step_id: dict(params)
                for step_id, params in self.step_param_updates.items()
            },
            "reasoning": self.reasoning,
            "give_up": self.give_up,
        }
        if include_raw_response:
            data["raw_response"] = self.raw_response
        return data


@dataclass
class ReplanContext:
    """
    喂给 LLM 的上下文快照（由 StepRunner.build_context() 构建）。
    包含：失败 Step + 后续 Steps + Budget 剩余 + 失败历史。
    """
    agent_id:        str
    failed_step:     StepSnapshot
    failure_reason:  str
    failure_history: List[Dict[str, str]]   # [{"tool": "...", "reason": "..."}]
    completed_steps: List[str]              # 已完成 step_id 列表
    pending_steps:   List[StepSnapshot]      # 后续所有 Step
    budget_remaining: Dict[str, Any]         # replans_left, llm_calls_left, ...


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class LLMCallError(Exception):
    """LLM 调用失败（网络问题、API 错误等）。"""
    def __init__(self, reason: str, original: Optional[Exception] = None) -> None:
        self.reason   = reason
        self.original = original
        super().__init__(f"LLM 调用失败: {reason}")


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------

class LLMInterface(ABC):
    """
    LLM 交互协议抽象基类。

    真实接入时： subclass + 实现 propose_replan() 即可。
    开发/测试：用 MockLLM。

    线程安全：
      LLMInterface 实例被 StepRunner 单例持有，
      propose_replan() 由 Scheduler 串行调用（StepRunner 是无状态的），
      不需要额外加锁。
    """

    @property
    @abstractmethod
    def call_count(self) -> int:
        """累计调用次数（供 Budget 记账）。"""

    @property
    def provider_name(self) -> str:
        """Provider 名称，供日志 / snapshot / debug replay 使用。"""
        return type(self).__name__

    @property
    def model_name(self) -> Optional[str]:
        """模型名称；无明确模型概念时返回 None。"""
        return None

    @abstractmethod
    def propose_replan(self, context: ReplanContext) -> ReplanResult:
        """
        给定上下文，返回重规划建议。
        不抛异常；技术失败应抛 LLMCallError。
        """


# ---------------------------------------------------------------------------
# Mock LLM（开发/测试用）
# ---------------------------------------------------------------------------

class MockLLM(LLMInterface):
    """
    固定返回 fallback plan 的 Mock 实现。

    行为：
      1. 始终返回 new_fallbacks = [当前 step 的主工具名 + "_v2" 版]
      2. 不修改已完成 Step
      3. 不改变 Step 顺序
      4. 模拟调用延迟（可选，默认 0.01s）
    """

    def __init__(self, simulate_delay_s: float = 0.01, always_give_up: bool = False) -> None:
        self._call_count: int = 0
        self._simulate_delay_s = simulate_delay_s
        self._always_give_up = always_give_up

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> Optional[str]:
        return "mock"

    def propose_replan(self, context: ReplanContext) -> ReplanResult:
        import time
        self._call_count += 1

        if self._simulate_delay_s > 0:
            time.sleep(self._simulate_delay_s)

        if self._always_give_up:
            return ReplanResult(
                new_fallbacks=[],
                step_param_updates={},
                reasoning="MockLLM 放弃（always_give_up=True）",
                give_up=True,
            )

        # Mock：总是建议把失败工具替换为 {tool_name}_v2
        failed_tool = context.failed_step.tool_name
        suggested_tool = f"{failed_tool}_v2"

        new_fallbacks = [
            FallbackSuggestion(
                tool=suggested_tool,
                params=dict(context.failed_step.params),
            ),
            # 再加一个更通用的兜底工具
            FallbackSuggestion(
                tool="generic_tool",
                params={"fallback": True},
            ),
        ]

        logger.info(
            "[MockLLM] Agent[%s] Step[%s] 失败原因: %s; "
            "建议 fallback: %s",
            context.agent_id,
            context.failed_step.step_id,
            context.failure_reason,
            [fb.tool for fb in new_fallbacks],
        )

        return ReplanResult(
            new_fallbacks=new_fallbacks,
            step_param_updates={},
            reasoning=(
                f"MockLLM 建议：用 {suggested_tool} 替换失败的 {failed_tool}，"
                f"参数保持不变"
            ),
            give_up=False,
        )


# ---------------------------------------------------------------------------
# Mock Tool Registry（配合 MockLLM 使用）
# ---------------------------------------------------------------------------

class MockToolRegistry:
    """
    配合 MockLLM 使用：注册工具名 → 模拟执行结果。
    ToolExecutor 在测试模式下使用此注册表。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, callable] = {}
        self._failures: Set[str] = set()   # 模拟失败的工具名

    def register(self, name: str, func: callable) -> None:
        self._tools[name] = func

    def simulate_failure(self, tool_name: str) -> None:
        """标记某个工具下次调用时失败（用于测试 Fallback）。"""
        self._failures.add(tool_name)

    def clear_failures(self) -> None:
        self._failures.clear()

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, **params) -> Any:
        if name not in self._tools:
            raise RuntimeError(f"MockToolRegistry: 工具 '{name}' 未注册")
        if name in self._failures:
            self._failures.discard(name)
            raise ToolExecutionFailed(f"Mock 模拟失败: {name}")
        return self._tools[name](**params)


# ---------------------------------------------------------------------------
# ToolExecutor 使用的异常（定义在这里避免循环 import）
# ---------------------------------------------------------------------------

class ToolExecutionFailed(Exception):
    """工具执行失败（业务层失败，不是异常）。"""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ToolTimeoutError(Exception):
    """工具执行超时。"""
    def __init__(self, tool_name: str, timeout_s: float) -> None:
        self.tool_name = tool_name
        self.timeout_s = timeout_s
        super().__init__(f"工具 {tool_name} 执行超时（{timeout_s}s）")
