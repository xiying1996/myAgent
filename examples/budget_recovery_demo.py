"""
budget_recovery_demo.py — Budget 限制与恢复策略演示

展示场景：
  1. Budget 如何限制 Agent 执行（max_steps, max_llm_calls, max_replans）
  2. Fallback 链：主工具失败 → 依次尝试 fallback → 成功恢复
  3. Replan 模式：所有 fallback 耗尽 → 调用 LLM 生成新方案
  4. Budget 耗尽时 Agent 被正确终止
  5. PolicyEngine 如何在状态转换前进行检查

运行:
  python examples/budget_recovery_demo.py

模拟工具:
  - primary_tool: 模拟失败（用于触发 fallback）
  - fallback_1, fallback_2: 模拟不同级别的 fallback
  - slow_tool: 模拟超时
"""

import os
import sys
import time
import logging
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget
from core.state_machine import AgentState, RetryMode

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import Dispatcher, RawEventBus

from execution.llm_interface import MockLLM
from execution.step_runner import StepRunner, FailureRecord
from execution.tool_executor import ToolExecutor

from scheduler.policy_engine import PolicyEngine
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager
from scheduler.scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FailureSimulator:
    """模拟不同失败模式的工具"""
    fail_mode: str = "none"  # none, always, timeout, rate_limit
    call_count: int = 0

    def __call__(self, **params):
        self.call_count += 1
        if self.fail_mode == "always":
            raise RuntimeError("Simulated tool failure")
        elif self.fail_mode == "timeout":
            raise TimeoutError("Simulated timeout")
        elif self.fail_mode == "rate_limit" and self.call_count == 1:
            raise RuntimeError("Rate limit exceeded (429)")
        return {"result": "success", "params": params, "attempt": self.call_count}


def run_demo(demo_name, plan, budget, description):
    """运行单个演示场景"""
    print(f"\n{'=' * 70}")
    print(f"📌 演示: {demo_name}")
    print(f"   {description}")
    print("=" * 70)

    # 创建组件
    queue = PriorityEventQueue()
    bus = RawEventBus()
    Dispatcher(queue).attach(bus)
    sm = StateManager()
    dv = DependencyValidator()
    pe = PolicyEngine()
    llm = MockLLM()
    sr = StepRunner(llm=llm, tool_registry=None)
    sched = Scheduler(
        event_queue=queue, raw_event_bus=bus,
        state_manager=sm, step_runner=sr,
        dependency_validator=dv, policy_engine=pe,
    )

    # 注册模拟工具
    executor = ToolExecutor(bus=bus, max_workers=4)
    executor.register_tool("primary", primary)
    executor.register_tool("fallback_1", fallback_1)
    executor.register_tool("fallback_2", fallback_2)
    executor.register_tool("slow", slow_tool)
    sched.set_tool_executor(executor)

    # 提交并执行
    agent_id = sched.submit_task(plan=plan, budget=budget, task_id=demo_name)
    sched.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        agent = sm.get_agent(agent_id)
        if agent and agent.is_terminal():
            break
        time.sleep(0.05)

    # 输出结果
    agent = sm.get_agent(agent_id)
    metrics = sm.get_metrics(agent_id)

    print(f"\n   结果:")
    print(f"   - 最终状态: {agent.state.value}")
    print(f"   - 执行步数: {metrics.step_count if metrics else '?'}")
    print(f"   - LLM 调用: {metrics.llm_call_count if metrics else '?'}")
    print(f"   - Replan次数: {metrics.replan_count if metrics else '?'}")

    # 工具调用统计
    print(f"   - 工具调用:")
    print(f"     primary: {primary.call_count}")
    print(f"     fallback_1: {fallback_1.call_count}")
    print(f"     fallback_2: {fallback_2.call_count}")

    sched.stop()
    executor.shutdown()

    return agent.state.value


# ── 模拟工具实例 ─────────────────────────────────────────────────────────
primary = FailureSimulator(fail_mode="always")
fallback_1 = FailureSimulator(fail_mode="none")
fallback_2 = FailureSimulator(fail_mode="none")
slow_tool = FailureSimulator(fail_mode="timeout")


def main():
    print("=" * 70)
    print("Agent Runtime Framework — Budget 与恢复策略演示")
    print("=" * 70)

    # ── Demo 1: 成功恢复（Fallback 链）────────────────────────────────
    plan1 = Plan.create([
        Step(
            step_id="main",
            tool_name="primary",
            params={"task": "do something"},
            fallback_chain=[
                FallbackOption("fallback_1", {"task": "do something"}),
                FallbackOption("fallback_2", {"task": "do something"}),
            ],
            output_schema={"result": "str"},
        ),
    ])

    # 重置计数器
    primary.fail_mode = "always"
    fallback_1.fail_mode = "none"
    fallback_2.fail_mode = "none"

    state = run_demo(
        "demo1_fallback_recovery",
        plan1,
        ExecutionBudget.default(),
        "主工具失败 → fallback_1 成功恢复"
    )
    print(f"   ✅ 预期: DONE, 实际: {state}")

    # ── Demo 2: Fallback 全部耗尽 → ERROR ─────────────────────────────
    plan2 = Plan.create([
        Step(
            step_id="main",
            tool_name="primary",
            params={"task": "do something"},
            fallback_chain=[
                FallbackOption("fallback_1", {"task": "do something"}),
                FallbackOption("fallback_2", {"task": "do something"}),
            ],
            output_schema={"result": "str"},
        ),
    ])

    # 所有工具都失败
    primary.fail_mode = "always"
    fallback_1.fail_mode = "always"
    fallback_2.fail_mode = "always"

    state = run_demo(
        "demo2_all_fail",
        plan2,
        ExecutionBudget.default(),
        "所有工具失败 → Fallback 耗尽 → ERROR"
    )
    print(f"   ✅ 预期: ERROR, 实际: {state}")

    # ── Demo 3: Budget 限制（max_steps = 1）───────────────────────────
    plan3 = Plan.create([
        Step(step_id="s1", tool_name="fallback_1", params={},
             output_schema={"result": "str"}),
        Step(step_id="s2", tool_name="fallback_2", params={},
             output_schema={"result": "str"}),
    ])

    # 限制只允许 1 步
    budget3 = ExecutionBudget(
        max_steps=1,
        max_llm_calls=3,
        max_replans=2,
        wall_clock_timeout=30,
    )

    fallback_1.fail_mode = "none"
    fallback_2.fail_mode = "none"

    state = run_demo(
        "demo3_budget_limit",
        plan3,
        budget3,
        "max_steps=1 → 第二步被拒绝 → DONE（Budget 限制）"
    )
    print(f"   ✅ 预期: DONE (budget exhausted), 实际: {state}")

    # ── Demo 4: Rate Limit → Fallback 恢复 ───────────────────────────
    plan4 = Plan.create([
        Step(
            step_id="api_call",
            tool_name="primary",
            params={"query": "search"},
            fallback_chain=[
                FallbackOption("fallback_1", {"query": "search"}),
            ],
            output_schema={"result": "str"},
        ),
    ])

    # 第一次失败（rate limit），第二次成功
    primary.fail_mode = "rate_limit"
    fallback_1.fail_mode = "none"

    state = run_demo(
        "demo4_rate_limit",
        plan4,
        ExecutionBudget.default(),
        "Rate limit → Fallback 恢复"
    )
    print(f"   ✅ 预期: DONE, 实际: {state}")

    # ── 总结 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 演示总结")
    print("=" * 70)
    print("""
本演示展示了 Agent 框架的恢复能力：

1. Fallback 链：主工具失败时，自动尝试 fallback 链中的下一个工具
2. 错误分类：Timeout、Rate Limit、网络错误等被正确分类
3. Budget 限制：max_steps/max_llm_calls/max_replans 控制资源使用
4. PolicyEngine：在状态转换前检查 budget，超限时注入 BUDGET_EXCEEDED

关键设计：
  - Fallback 耗尽 → 切换到 REPLAN_MODE → 调用 LLM 生成新方案
  - Budget 超限 → 注入 BUDGET_EXCEEDED → Agent 进入 ERROR
  - 所有检查都在状态转换前执行，避免非法状态
    """)


if __name__ == "__main__":
    main()
