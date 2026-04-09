"""
tool_executor.py — ToolExecutor (async execution + timeout)

对外接口:
  Action              — 单次工具调用的完整描述
  ToolFailure         — 工具失败信息（区分超时 / 异常 / 业务失败）
  ToolExecutor         — 异步工具执行器，线程池 + 超时检测

执行流程:
  StepRunner.decide() → Action
    → ToolExecutor.submit(action)
      → Policy pre-check（permissions / 权限）
      → ThreadPoolExecutor 异步执行
        → 执行完成 / 超时
          → Event (TOOL_RESULT / TIMEOUT / TOOL_ERROR) → RawEventBus
            → PriorityEventQueue
              → Scheduler 消费

设计约束:
  ToolExecutor 不做路由判断，统一发到 RawEventBus。
  RawEventBus → Dispatcher → PriorityEventQueue → Scheduler。
  Policy 是 pre-execution guard，不是 post-check。
"""

from __future__ import annotations

import logging
import time
import uuid
import threading
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from events.event import Event
from events.event_types import EventType
from events.raw_event_bus import RawEventBus
from execution.llm_interface import ToolExecutionFailed, ToolTimeoutError

# Tool system（可选引入，避免循环依赖）
try:
    from tools.tool import Tool
    from tools.result import ToolResult, ToolStatus, ToolErrorType
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    Tool = None
    ToolResult = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """
    单次工具调用的完整描述。
    由 StepRunner.decide() 构建，交给 ToolExecutor.submit() 执行。
    """
    tool_name: str
    params:    Dict[str, Any]
    agent_id:  str
    step_id:   str
    timeout_s: float = 30.0
    metadata:  Dict[str, Any] = field(default_factory=dict)
    adapter_chain: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolFailure
# ---------------------------------------------------------------------------

@dataclass
class ToolFailure:
    """
    工具失败信息（包含失败类型，供 StepRunner 构建 ReplanContext）。
    """
    step_id:    str
    tool_name:  str
    reason:     str
    failure_type: str  # "timeout" | "error" | "failed"

    def to_dict(self) -> Dict[str, str]:
        return {
            "step_id":     self.step_id,
            "tool_name":   self.tool_name,
            "reason":      self.reason,
            "failure_type": self.failure_type,
        }


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    异步工具执行器。

    使用 ThreadPoolExecutor 实现并发执行，支持：
      - 工具注册（fn 或异步版本）
      - 超时检测（独立计时器，不占用调用方时间）
      - 执行完成 → 自动发送 Event 到 RawEventBus

    使用示例:
        executor = ToolExecutor(bus=event_bus)
        executor.register_tool("web_search", web_search_fn)
        executor.register_tool("calculator", calculator_fn)

        # 提交执行（异步，立刻返回）
        future = executor.submit(Action(
            tool_name="web_search",
            params={"query": "hello"},
            agent_id="agent_001",
            step_id="step0",
        ))
    """

    def __init__(
        self,
        bus: RawEventBus,
        max_workers: int = 4,
        policy: Optional[Callable[..., bool]] = None,
    ) -> None:
        self._bus        = bus
        self._tools: Dict[str, Any] = {}  # str → Tool or Callable
        self._executor   = cf.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[cf.Future, Action] = {}
        self._timers: Dict[cf.Future, threading.Timer] = {}
        self._start_times: Dict[cf.Future, float] = {}  # future → start_time
        self._timed_out: Dict[cf.Future, bool] = {}     # future → 是否已发过 timeout
        self._policy = policy or (lambda tool, params: True)  # 默认允许所有

    # ── 工具注册 ─────────────────────────────────────────────────────────

    def register_tool(
        self,
        name_or_tool: Any,
        func: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        注册工具。

        方式1 — Tool 实例：
            executor.register_tool(my_tool)

        方式2 — name + callable：
            executor.register_tool("search", search_fn)
        """
        if TOOLS_AVAILABLE and isinstance(name_or_tool, Tool):
            tool = name_or_tool
            self._tools[tool.name] = tool
            logger.debug("ToolExecutor: 注册 Tool %s", tool.name)
        elif func is not None:
            name = name_or_tool
            self._tools[name] = func
            logger.debug("ToolExecutor: 注册 callable %s", name)
        else:
            raise TypeError("register_tool 需要 Tool 实例 或 (name, callable)")

    def register_async_tool(self, name: str, func: Callable[..., Any]) -> None:
        """
        注册一个异步工具函数（返回 coroutine）。
        目前同步执行，未来可升级为 asyncio 版本。
        """
        self._tools[name] = func
        logger.debug("ToolExecutor: 注册异步工具 %s", name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    # ── 提交执行 ─────────────────────────────────────────────────────────

    def submit(self, action: Action) -> cf.Future:
        """
        异步提交工具执行，立刻返回 Future。
        完成后自动发送 Event 到 RawEventBus。

        Policy pre-check 在执行前拦截（permissions / 权限）。

        不会抛异常：
          - 超时 → TOOL_ERROR (TIMEOUT)
          - 工具抛异常 → TOOL_ERROR (ERROR)
          - 工具返回失败信号 → TOOL_FAILED
        """
        # ── Policy pre-check（前置拦截）───────────────────────────────
        tool = self._tools.get(action.tool_name)
        if tool is not None:
            if not self._policy(tool, action.params):
                logger.warning(
                    "ToolExecutor: Policy 拒绝 tool=%s(params=%s)",
                    action.tool_name, action.params,
                )
                self._emit_error(
                    action,
                    error_type="PolicyDenied",
                    error_msg=f"Policy 拒绝工具 '{action.tool_name}'",
                )
                f = cf.Future()
                f.set_result(None)
                return f

        if action.tool_name not in self._tools:
            # 工具未注册 → 发送 ERROR 事件，不阻塞
            logger.error("ToolExecutor: 工具 '%s' 未注册", action.tool_name)
            self._emit_error(
                action,
                error_type="ToolNotRegistered",
                error_msg=f"工具 '{action.tool_name}' 未注册",
            )
            # 返回一个已完成但失败的 dummy future
            f = cf.Future()
            f.set_result(None)
            return f

        future = self._executor.submit(self._execute, action)
        self._futures[future] = action
        self._start_times[future] = time.monotonic()
        future.add_done_callback(self._on_complete)

        # 启动超时计时器（用于发送超时事件）
        if action.timeout_s > 0:
            timer = threading.Timer(action.timeout_s, self._on_timeout, args=(future, action))
            timer.daemon = True
            timer.start()
            self._timers[future] = timer
        logger.info(
            "ToolExecutor: 提交 %s(step=%s, agent=%s, timeout=%ss)",
            action.tool_name, action.step_id, action.agent_id, action.timeout_s,
        )
        return future

    # ── 内部执行 ─────────────────────────────────────────────────────────

    def _execute(self, action: Action) -> Any:
        """
        在线程池中执行工具调用。
        支持 Tool 实例（返回 ToolResult）和旧路途 callable。
        不直接抛异常到外部，由 _on_complete 统一处理。
        """
        tool = self._tools[action.tool_name]

        # 路径1：Tool 实例
        if TOOLS_AVAILABLE and isinstance(tool, Tool):
            try:
                start = time.monotonic()
                tool_result = tool.invoke(
                    action.params,
                    context={"trace_id": action.metadata.get("trace_id")},
                )
                duration = time.monotonic() - start
                if isinstance(tool_result, dict):
                    result = {
                        "ok": True,
                        "result": tool_result,
                        "duration_s": duration,
                    }
                    return result
                if not isinstance(tool_result, ToolResult):
                    return {
                        "ok": False,
                        "result": None,
                        "error_type": "InvalidToolResult",
                        "error_msg": (
                            f"Tool[{action.tool_name}] invoke() 必须返回 ToolResult 或 dict，"
                            f"实际为 {type(tool_result).__name__}"
                        ),
                        "failure_type": "error",
                    }

                if tool_result.is_success:
                    return {
                        "ok": True,
                        "result": tool_result.output,
                        "duration_s": duration,
                        "tool_result": tool_result,
                    }

                failure_type = self._map_tool_failure(tool_result)
                result = {
                    "ok": False,
                    "result": tool_result.output,
                    "failure_type": failure_type,
                    "tool_result": tool_result,
                }
                if failure_type == "timeout":
                    result["timeout_s"] = tool_result.metadata.get(
                        "timeout_s",
                        getattr(tool, "timeout_s", action.timeout_s),
                    )
                elif failure_type == "error":
                    result["error_type"] = (
                        tool_result.error_type.value
                        if tool_result.error_type is not None
                        else "Unknown"
                    )
                    result["error_msg"] = tool_result.error or "未知"
                else:
                    result["reason"] = tool_result.error or "未知"
                return result
            except Exception as e:  # noqa: BLE001
                return {
                    "ok": False,
                    "result": None,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "failure_type": "error",
                }

        # 路径2：旧路途 callable
        func = tool
        try:
            start = time.monotonic()
            result = func(**action.params)
            duration = time.monotonic() - start
            return {"ok": True, "result": result, "duration_s": duration}
        except ToolTimeoutError:
            raise
        except ToolExecutionFailed as e:
            return {"ok": False, "result": None, "reason": e.reason, "failure_type": "failed"}
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "result": None,
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "failure_type": "error",
            }

    def _on_complete(self, future: cf.Future) -> None:
        """
        工具执行完成（成功 / 超时 / 异常）后的回调。
        从 future 拿到 action 信息，发送对应 Event 到 RawEventBus。
        """
        action = self._futures.pop(future, None)
        if action is None:
            return

        # 取消超时计时器（如果还在运行）
        timer = self._timers.pop(future, None)
        if timer is not None:
            timer.cancel()

        # 如果已经发过 timeout 事件，跳过结果处理
        if self._timed_out.pop(future, False):
            self._start_times.pop(future, None)
            return

        try:
            outcome = future.result(timeout=0)
        except cf.TimeoutError:
            self._emit_timeout(action)
            self._start_times.pop(future, None)
            return
        except Exception as e:  # noqa: BLE001
            self._emit_error(action, error_type=type(e).__name__, error_msg=str(e))
            self._start_times.pop(future, None)
            return

        # 检查是否超时（基于实际耗时）
        start_time = self._start_times.pop(future, None)
        if start_time is not None and action.timeout_s > 0:
            elapsed = time.monotonic() - start_time
            if elapsed > action.timeout_s:
                self._emit_timeout(action)
                return

        # 成功
        if outcome.get("ok"):
            self._emit_result(action, outcome)
        elif outcome.get("failure_type") == "timeout":
            self._emit_timeout(
                action,
                timeout_s=outcome.get("timeout_s", action.timeout_s),
                tool_result=outcome.get("tool_result"),
            )
        elif outcome.get("failure_type") == "error":
            # 工具执行抛异常
            self._emit_error(
                action,
                error_type=outcome.get("error_type", "Unknown"),
                error_msg=outcome.get("error_msg", "未知"),
                tool_result=outcome.get("tool_result"),
            )
        else:
            # 业务层失败（工具自己返回的）
            self._emit_failed(
                action,
                reason=outcome.get("reason", "未知"),
                tool_result=outcome.get("tool_result"),
            )

    def _on_timeout(self, future: cf.Future, action: Action) -> None:
        """
        超时计时器触发：标记已超时，忽略后续结果。
        注意：无法真正取消正在运行的任务，结果仍会由 _on_complete 处理，但会被忽略。
        """
        # 标记为已超时，_on_complete 会忽略实际结果
        self._timed_out[future] = True
        logger.warning(
            "ToolExecutor: %s(step=%s) 超时（%.1fs）",
            action.tool_name, action.step_id, action.timeout_s,
        )
        self._bus.publish(Event.create(
            event_type=EventType.TIMEOUT,
            agent_id=action.agent_id,
            payload={
                "step_id":   action.step_id,
                "tool_name": action.tool_name,
                "timeout_s": action.timeout_s,
            },
        ))

    # ── Event 发送 ───────────────────────────────────────────────────────

    def _emit_result(self, action: Action, outcome: Dict[str, Any]) -> None:
        payload = {
            "step_id":       action.step_id,
            "tool_name":     action.tool_name,
            "result":        outcome["result"],
            "duration_s":    outcome.get("duration_s", 0.0),
            "adapter_chain": list(action.adapter_chain),
        }
        tool_result = outcome.get("tool_result")
        if tool_result is not None:
            payload["tool_result"] = self._serialize_tool_result(tool_result)

        self._bus.publish(Event.create(
            event_type=EventType.TOOL_RESULT,
            agent_id=action.agent_id,
            payload=payload,
        ))
        logger.info(
            "ToolExecutor: %s(step=%s) 成功，耗时 %.3fs",
            action.tool_name, action.step_id, outcome.get("duration_s", 0),
        )

    def _emit_failed(
        self,
        action: Action,
        reason: str,
        tool_result: Optional["ToolResult"] = None,
    ) -> None:
        payload = {
            "step_id":   action.step_id,
            "tool_name": action.tool_name,
            "reason":    reason,
        }
        if tool_result is not None:
            payload["tool_result"] = self._serialize_tool_result(tool_result)
            if tool_result.error_type is not None:
                payload["error_type"] = tool_result.error_type.value

        self._bus.publish(Event.create(
            event_type=EventType.TOOL_FAILED,
            agent_id=action.agent_id,
            payload=payload,
        ))
        logger.info(
            "ToolExecutor: %s(step=%s) 业务失败: %s",
            action.tool_name, action.step_id, reason,
        )

    def _emit_error(
        self,
        action: Action,
        error_type: str,
        error_msg: str,
        tool_result: Optional["ToolResult"] = None,
    ) -> None:
        payload = {
            "step_id":   action.step_id,
            "tool_name": action.tool_name,
            "error_type": error_type,
            "error_msg":  error_msg,
        }
        if tool_result is not None:
            payload["tool_result"] = self._serialize_tool_result(tool_result)

        self._bus.publish(Event.create(
            event_type=EventType.TOOL_ERROR,
            agent_id=action.agent_id,
            payload=payload,
        ))
        logger.warning(
            "ToolExecutor: %s(step=%s) 异常 [%s]: %s",
            action.tool_name, action.step_id, error_type, error_msg,
        )

    def _emit_timeout(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        tool_result: Optional["ToolResult"] = None,
    ) -> None:
        timeout_value = action.timeout_s if timeout_s is None else timeout_s
        payload = {
            "step_id":   action.step_id,
            "tool_name": action.tool_name,
            "timeout_s": timeout_value,
        }
        if tool_result is not None:
            payload["tool_result"] = self._serialize_tool_result(tool_result)

        self._bus.publish(Event.create(
            event_type=EventType.TIMEOUT,
            agent_id=action.agent_id,
            payload=payload,
        ))
        logger.warning(
            "ToolExecutor: %s(step=%s) 执行超时（%.1fs）",
            action.tool_name, action.step_id, timeout_value,
        )

    def _map_tool_failure(self, tool_result: "ToolResult") -> str:
        if (
            tool_result.status == ToolStatus.TIMEOUT
            or tool_result.error_type == ToolErrorType.TIMEOUT
        ):
            return "timeout"
        if (
            tool_result.status == ToolStatus.FAILURE
            or tool_result.error_type == ToolErrorType.SCHEMA_MISMATCH
        ):
            return "failed"
        return "error"

    def _serialize_tool_result(self, tool_result: Any) -> Dict[str, Any]:
        if TOOLS_AVAILABLE and isinstance(tool_result, ToolResult):
            return tool_result.to_dict()
        return {"value": str(tool_result)}

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池。"""
        self._executor.shutdown(wait=wait)
        logger.info("ToolExecutor: 线程池已关闭")

    def __enter__(self) -> "ToolExecutor":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()
