"""
parallel_agents_demo.py — 多 Agent 并行执行演示

展示场景：
  1. 同一个 Scheduler 管理多个 Agent
  2. Agent 独立执行，互不干扰
  3. PriorityEventQueue 高优先级事件优先处理
  4. 每个 Agent 有独立的 Budget 和 Plan
  5. Scheduler 统一调度，观察并发执行

运行:
  python examples/parallel_agents_demo.py
"""

import os
import sys
import time
import logging
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget
from core.state_machine import AgentState

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import Dispatcher, RawEventBus

from execution.llm_interface import MockLLM
from execution.step_runner import StepRunner
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


def main():
    print("=" * 70)
    print("Agent Runtime Framework — 多 Agent 并行执行演示")
    print("=" * 70)

    # ── 1. 构建共享组件 ─────────────────────────────────────────────
    queue = PriorityEventQueue()
    bus = RawEventBus()
    Dispatcher(queue).attach(bus)
    sm = StateManager()
    dv = DependencyValidator()
    pe = PolicyEngine()
    llm = MockLLM()
    sr = StepRunner(llm=llm, tool_registry=None)

    sched = Scheduler(
        event_queue=queue,
        raw_event_bus=bus,
        state_manager=sm,
        step_runner=sr,
        dependency_validator=dv,
        policy_engine=pe,
    )

    # ── 2. 注册模拟工具 ─────────────────────────────────────────────
    executor = ToolExecutor(bus=bus, max_workers=4)
    sched.set_tool_executor(executor)

    # 不同延迟的工具（模拟不同耗时任务）
    delays = {"quick": 0.1, "medium": 0.3, "slow": 0.5}

    def make_tool(name, delay):
        def tool(**params):
            logger.info(f"[{name}] 开始执行...")
            time.sleep(delay)
            logger.info(f"[{name}] 完成")
            return {"tool": name, "result": f"completed by {name}", "delay": delay}
        return tool

    for name, delay in delays.items():
        executor.register_tool(name, make_tool(name, delay))

    # ── 3. 创建多个 Agent（不同任务）────────────────────────────────

    # Agent 1: 快速任务
    plan1 = Plan.create([
        Step(step_id="q1", tool_name="quick", params={},
             output_schema={"tool": "str", "result": "str"}),
    ])
    agent1_id = sched.submit_task(
        plan=plan1,
        budget=ExecutionBudget.default(),
        agent_id="agent_quick",
        task_id="task_quick",
    )

    # Agent 2: 中等任务
    plan2 = Plan.create([
        Step(step_id="m1", tool_name="medium", params={},
             output_schema={"tool": "str", "result": "str"}),
        Step(step_id="m2", tool_name="quick", params={},
             output_schema={"tool": "str", "result": "str"},
             dependencies=["m1"]),
    ])
    agent2_id = sched.submit_task(
        plan=plan2,
        budget=ExecutionBudget.default(),
        agent_id="agent_medium",
        task_id="task_medium",
    )

    # Agent 3: 慢速任务
    plan3 = Plan.create([
        Step(step_id="s1", tool_name="slow", params={},
             output_schema={"tool": "str", "result": "str"}),
        Step(step_id="s2", tool_name="medium", params={},
             output_schema={"tool": "str", "result": "str"},
             dependencies=["s1"]),
    ])
    agent3_id = sched.submit_task(
        plan=plan3,
        budget=ExecutionBudget.default(),
        agent_id="agent_slow",
        task_id="task_slow",
    )

    # Agent 4: 带 fallback 的任务
    plan4 = Plan.create([
        Step(
            step_id="fb1", tool_name="slow", params={},
            fallback_chain=[
                FallbackOption("medium", {}),
                FallbackOption("quick", {}),
            ],
            output_schema={"tool": "str", "result": "str"},
        ),
    ])
    agent4_id = sched.submit_task(
        plan=plan4,
        budget=ExecutionBudget.default(),
        agent_id="agent_fallback",
        task_id="task_fallback",
    )

    print(f"\n📋 提交的 Agent:")
    print(f"   1. agent_quick:    快速任务 (0.1s)")
    print(f"   2. agent_medium:   中等任务 (0.3s + 0.1s)")
    print(f"   3. agent_slow:    慢速任务 (0.5s + 0.3s)")
    print(f"   4. agent_fallback: Fallback 测试")

    # ── 4. 启动调度并监控 ───────────────────────────────────────────
    sched.start()

    print("\n🚀 开始执行（监控进度）...")
    start_time = time.time()

    def monitor():
        while True:
            time.sleep(0.2)
            agents = sm.list_agents()
            done = sum(1 for a in agents if a.is_terminal())
            total = len(agents)
            if done == total:
                break
            statuses = {a.agent_id: a.state.value for a in agents}
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.1f}s] 完成: {done}/{total} - {statuses}")

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # 等待所有 Agent 完成
    deadline = time.time() + 15
    while time.time() < deadline:
        agents = sm.list_agents()
        if all(a.is_terminal() for a in agents):
            break
        time.sleep(0.1)

    monitor_thread.join()

    # ── 5. 输出结果 ─────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)

    agent_ids = [agent1_id, agent2_id, agent3_id, agent4_id]
    agent_names = ["agent_quick", "agent_medium", "agent_slow", "agent_fallback"]

    for name, aid in zip(agent_names, agent_ids):
        agent = sm.get_agent(aid)
        metrics = sm.get_metrics(aid)
        outputs = sched.get_agent_outputs(aid)

        print(f"\n   [{name}]")
        print(f"     状态: {agent.state.value}")
        print(f"     步数: {metrics.step_count if metrics else '?'}")

        if outputs:
            print(f"     输出:")
            for step_id, output in outputs.items():
                print(f"       {step_id}: {output}")

    # 理论最短时间 vs 实际时间
    print(f"\n⏱️  时间分析:")
    print(f"   理论最短 (并行): 0.5s (最慢任务)")
    print(f"   理论最长 (串行): 1.0s + 0.4s + 0.8s = 2.2s")
    print(f"   实际耗时: {elapsed:.2f}s")

    if elapsed < 1.5:
        print(f"   ✅ 确认并行执行")
    else:
        print(f"   ⚠️  可能是串行执行或并发有限")

    sched.stop()
    executor.shutdown()

    print("\n✅ 多 Agent 并行演示完成")


if __name__ == "__main__":
    main()
