"""
checkpoint_replay_demo.py — Checkpoint 与 Replay 演示

展示场景：
  1. CheckpointManager 自动保存 Agent 快照
  2. 在 Replan 后触发快照
  3. 从快照恢复并继续执行（RecoveryReplayAgent）
  4. DebugReplayAgent 重放执行历史

运行:
  python examples/checkpoint_replay_demo.py

原理：
  - Checkpoint 在关键节点保存 Agent 完整状态
  - Replay 用于：调试、重现 bug、从中断恢复
  - Snapshot 包含：Plan、BudgetUsage、History、State
"""

import os
import sys
import time
import logging
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import Dispatcher, RawEventBus

from execution.llm_interface import MockLLM
from execution.step_runner import StepRunner
from execution.tool_executor import ToolExecutor

from scheduler.policy_engine import PolicyEngine
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager
from scheduler.scheduler import Scheduler

from checkpoint.checkpoint_manager import CheckpointManager
from checkpoint.replay import DebugReplayAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("Agent Runtime Framework — Checkpoint 与 Replay 演示")
    print("=" * 70)

    # ── 创建临时目录 ────────────────────────────────────────────────
    snapshot_dir = tempfile.mkdtemp(prefix="myagent_snapshots_")
    print(f"\n📁 Snapshot 目录: {snapshot_dir}")

    try:
        # ── 1. 构建组件 ──────────────────────────────────────────────
        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)

        sm = StateManager()
        dv = DependencyValidator()
        pe = PolicyEngine()

        # Checkpoint 管理器
        checkpoint_mgr = CheckpointManager(
            snapshot_dir=snapshot_dir,
            snapshot_interval_steps=2,  # 每 2 步保存一次
        )

        llm = MockLLM()
        sr = StepRunner(llm=llm, tool_registry=None)

        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=sm,
            step_runner=sr,
            dependency_validator=dv,
            policy_engine=pe,
            checkpoint_manager=checkpoint_mgr,
        )

        # ── 2. 注册工具 ─────────────────────────────────────────────
        executor = ToolExecutor(bus=bus, max_workers=2)
        sched.set_tool_executor(executor)

        def step_a(**params):
            return {"result": "step_a done"}

        def step_b(**params):
            return {"result": "step_b done"}

        def step_c(**params):
            return {"result": "step_c done"}

        executor.register_tool("step_a", step_a)
        executor.register_tool("step_b", step_b)
        executor.register_tool("step_c", step_c)

        # ── 3. 构建 3 步 Plan ───────────────────────────────────────
        plan = Plan.create([
            Step(step_id="s1", tool_name="step_a", params={},
                 output_schema={"result": "str"}),
            Step(step_id="s2", tool_name="step_b", params={},
                 output_schema={"result": "str"},
                 dependencies=["s1"]),
            Step(step_id="s3", tool_name="step_c", params={},
                 output_schema={"result": "str"},
                 dependencies=["s2"]),
        ], max_replans=1)

        # ── 4. 第一阶段：执行部分步骤 ────────────────────────────────
        print("\n" + "=" * 70)
        print("📌 第一阶段：执行 2 步后中断（模拟崩溃）")
        print("=" * 70)

        agent_id = sched.submit_task(
            plan=plan,
            budget=ExecutionBudget.default(),
            task_id="checkpoint_demo",
        )

        sched.start()

        # 只执行 2 步就停止
        deadline = time.time() + 5
        while time.time() < deadline:
            agent = sm.get_agent(agent_id)
            if agent and agent.plan._current_index >= 2:
                print(f"   ✅ 已完成 2 步，模拟中断...")
                break
            if agent and agent.is_terminal():
                break
            time.sleep(0.1)

        sched.stop()
        executor.shutdown()

        # ── 5. 检查快照 ─────────────────────────────────────────────
        agent_before = sm.get_agent(agent_id)
        print(f"\n   中断时状态: {agent_before.state.value}")
        print(f"   已完成步骤: {list(agent_before.plan.completed_step_ids)}")
        print(f"   当前索引: {agent_before.plan._current_index}")

        snapshots = checkpoint_mgr.list_snapshots(agent_id)
        print(f"\n   📸 快照数量: {len(snapshots)}")
        for s in snapshots:
            print(f"     - {s.snapshot_id[:16]}... [idx={s.current_index}]")

        # ── 6. 第二阶段：从最新快照重放（DebugReplayAgent）──────────
        print("\n" + "=" * 70)
        print("📌 第二阶段：DebugReplayAgent 重放执行历史")
        print("=" * 70)

        if snapshots:
            latest = snapshots[-1]

            # 构造假的事件日志（用于演示）
            event_log = [
                {"event_id": "e1", "event_type": "TOOL_RESULT", "agent_id": agent_id},
                {"event_id": "e2", "event_type": "TOOL_RESULT", "agent_id": agent_id},
            ]

            # 创建 DebugReplayAgent
            plan_snapshots = {agent_id: latest}
            replay = DebugReplayAgent(
                event_log=event_log,
                plan_snapshots=plan_snapshots,
            )

            # 重放所有事件
            result = replay.replay_all()
            print(f"   ✅ 重放完成")
            print(f"   结果: {result}")

        # ── 7. 继续执行（从快照恢复）────────────────────────────────
        print("\n" + "=" * 70)
        print("📌 第三阶段：从快照继续执行")
        print("=" * 70)

        # 重新创建组件
        queue2 = PriorityEventQueue()
        bus2 = RawEventBus()
        Dispatcher(queue2).attach(bus2)
        sm2 = StateManager()
        dv2 = DependencyValidator()
        pe2 = PolicyEngine()

        checkpoint_mgr2 = CheckpointManager(
            snapshot_dir=snapshot_dir,
            snapshot_interval_steps=2,
        )

        llm2 = MockLLM()
        sr2 = StepRunner(llm=llm2, tool_registry=None)

        sched2 = Scheduler(
            event_queue=queue2,
            raw_event_bus=bus2,
            state_manager=sm2,
            step_runner=sr2,
            dependency_validator=dv2,
            policy_engine=pe2,
            checkpoint_manager=checkpoint_mgr2,
        )

        executor2 = ToolExecutor(bus=bus2, max_workers=2)
        sched2.set_tool_executor(executor2)
        executor2.register_tool("step_a", step_a)
        executor2.register_tool("step_b", step_b)
        executor2.register_tool("step_c", step_c)

        # 从快照加载
        loaded = checkpoint_mgr2.load_snapshot(agent_id)
        if loaded:
            print(f"   ✅ 已加载快照: {loaded.snapshot_id[:16]}...")
            print(f"   Plan: {loaded.plan_id}")
            print(f"   BudgetUsage: step_count={loaded.budget_usage.get('step_count', '?')}")

        # 创建新 agent 从快照恢复继续
        # 注意：这里简化处理，实际应用中会用 RecoveryReplayAgent
        new_agent_id = sched2.submit_task(
            plan=plan,
            budget=ExecutionBudget.default(),
            task_id="recovery_demo",
        )

        sched2.start()
        deadline2 = time.time() + 10
        while time.time() < deadline2:
            agent = sm2.get_agent(new_agent_id)
            if agent and agent.is_terminal():
                break
            time.sleep(0.1)

        final_agent = sm2.get_agent(new_agent_id)
        print(f"\n   恢复后执行完成: {final_agent.state.value}")
        print(f"   已完成步骤: {list(final_agent.plan.completed_step_ids)}")

        sched2.stop()
        executor2.shutdown()

        # ── 8. 最终统计 ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("📊 最终统计")
        print("=" * 70)

        all_snapshots = checkpoint_mgr2.list_snapshots(new_agent_id)
        print(f"   总快照数: {len(all_snapshots)}")
        for s in all_snapshots:
            print(f"     - {s.snapshot_id[:16]}... [idx={s.current_index}]")

        print(f"""
本演示展示了 Checkpoint 与 Replay 的核心能力：

1. 自动快照：
   - CheckpointManager 在关键节点保存 Agent 完整状态
   - 可配置 snapshot_interval_steps 控制保存频率

2. 快照内容：
   - Agent 状态和历史
   - Plan（current_index + completed_ids + 所有 steps）
   - BudgetUsage（step_count, llm_call_count 等）
   - 执行历史记录

3. 恢复能力：
   - 从快照加载继续执行
   - DebugReplayAgent 重放执行历史

4. 使用场景：
   - 长任务中断恢复（进程崩溃后继续）
   - 执行过程调试（重放分析问题）
   - 审计追溯（完整执行历史记录）
        """)

    finally:
        shutil.rmtree(snapshot_dir)
        print(f"\n🧹 已清理: {snapshot_dir}")


if __name__ == "__main__":
    main()
