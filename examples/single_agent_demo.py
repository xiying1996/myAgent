"""
single_agent_demo.py — 单 Agent 端到端演示

演示场景：
  1. 创建 3 步 Plan（search → fetch → summarize）
  2. 默认使用 MockLLM，可通过环境变量切到 DeepSeek
  3. Step1 失败 → Fallback → 成功
  4. 输出完整执行日志

运行:
  python examples/single_agent_demo.py

切换到 DeepSeek:
  export MYAGENT_LLM_PROVIDER=deepseek
  export DEEPSEEK_API_KEY=...
  python examples/single_agent_demo.py
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget
from core.agent import Agent
from core.state_machine import AgentState

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import Dispatcher, RawEventBus

from execution.llm_factory import create_llm_from_env
from execution.llm_interface import MockToolRegistry
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
    print("=" * 60)
    print("Agent Runtime Framework — 单 Agent 演示")
    print("=" * 60)

    # ── 1. 构建组件 ──────────────────────────────────────────────────
    queue  = PriorityEventQueue()
    bus    = RawEventBus()
    Dispatcher(queue).attach(bus)
    sm     = StateManager()
    llm    = create_llm_from_env(mock_delay_s=0.01)
    sr     = StepRunner(llm)
    dv     = DependencyValidator()
    pe     = PolicyEngine()

    print(f"LLM Provider: {llm.provider_name} ({llm.model_name or 'n/a'})")
    print("Note: 当前示例只有 fallback 耗尽时才会真正调用 LLM。")

    sched  = Scheduler(
        event_queue=queue,
        raw_event_bus=bus,
        state_manager=sm,
        step_runner=sr,
        dependency_validator=dv,
        policy_engine=pe,
    )

    # ── 2. 注册工具 ──────────────────────────────────────────────────
    tools = MockToolRegistry()

    def web_search(**params):
        q = params.get("query", "")
        logger.info(f"[Tool] web_search query={q}")
        return {"url": f"http://result/{q}", "title": f"Result for {q}"}

    def web_search_v2(**params):
        """Fallback 工具"""
        q = params.get("query", "")
        logger.info(f"[Tool] web_search_v2 (fallback) query={q}")
        return {"url": f"http://v2/{q}", "title": f"V2 Result for {q}"}

    def fetch_page(**params):
        url = params.get("url", "")
        logger.info(f"[Tool] fetch_page url={url}")
        return {"html": f"<html>{url}</html>", "status": 200}

    def summarize(**params):
        content = params.get("content", "")
        logger.info(f"[Tool] summarize content={content[:30]}...")
        return {"summary": f"Summary of {content[:20]}..."}

    tools.register("web_search", web_search)
    tools.register("web_search_v2", web_search_v2)
    tools.register("fetch_page", fetch_page)
    tools.register("summarize", summarize)

    # ToolExecutor 使用 MockToolRegistry
    executor = ToolExecutor(bus=bus, max_workers=4)
    # 手动给 executor 一个注册表（这里用 lambda 包装）
    # 注：实际使用需要改造，这里用注册函数的方式
    def echo(**params):
        return params

    executor.register_tool("echo", echo)
    executor.register_tool("web_search", web_search)
    executor.register_tool("web_search_v2", web_search_v2)
    executor.register_tool("fetch_page", fetch_page)
    executor.register_tool("summarize", summarize)

    sched.set_tool_executor(executor)

    # ── 3. 构建 Plan ──────────────────────────────────────────────────
    plan = Plan.create([
        Step(
            step_id="step0",
            tool_name="web_search",
            params={"query": "latest AI news"},
            fallback_chain=[
                FallbackOption(tool="web_search_v2", params={"query": "latest AI news"}),
            ],
            output_schema={"url": "str", "title": "str"},
        ),
        Step(
            step_id="step1",
            tool_name="fetch_page",
            params={"url": ""},
            input_bindings={"url": "step0.url"},
            output_schema={"html": "str", "status": "int"},
            dependencies=["step0"],
        ),
        Step(
            step_id="step2",
            tool_name="summarize",
            params={"content": ""},
            input_bindings={"content": "step1.html"},
            output_schema={"summary": "str"},
            dependencies=["step1"],
        ),
    ], max_replans=3)

    # ── 4. 提交任务 ──────────────────────────────────────────────────
    print("\n📋 提交任务:")
    print(f"   Plan: {plan.plan_id}")
    print(f"   Steps: {[s.step_id for s in plan.steps]}")
    print()

    agent_id = sched.submit_task(
        plan=plan,
        budget=ExecutionBudget.default(),
        agent_id="demo_agent",
        task_id="task_demo_001",
    )
    print(f"✅ Agent 已提交: {agent_id}")

    # ── 5. 启动调度循环 ───────────────────────────────────────────────
    sched.start()
    print("✅ Scheduler 启动")

    # ── 6. 模拟执行：让 ToolExecutor 同步执行完 ──────────────────────
    # 手动 pump 事件循环（演示用）
    print("\n📨 手动 pump 事件循环...")
    for _ in range(20):  # 最多 20 轮
        time.sleep(0.05)
        if sm.get_agent(agent_id).is_terminal():
            break

    # ── 7. 等待完成 ──────────────────────────────────────────────────
    print("\n等待 Agent 执行完成...")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        agent = sm.get_agent(agent_id)
        if agent and agent.is_terminal():
            break
        time.sleep(0.1)

    # ── 8. 输出结果 ──────────────────────────────────────────────────
    agent = sm.get_agent(agent_id)
    metrics = sm.get_metrics(agent_id)

    print("\n" + "=" * 60)
    print("📊 执行结果")
    print("=" * 60)
    print(f"   Agent ID:    {agent.agent_id}")
    print(f"   Final State: {agent.state.value}")
    print(f"   Steps:       {metrics.step_count if metrics else '?'}")
    print(f"   LLM Calls:   {metrics.llm_call_count if metrics else '?'}")
    print(f"   Replans:     {metrics.replan_count if metrics else '?'}")

    print("\n📜 状态历史:")
    for record in agent.state_history:
        retry_info = f" ({record.retry_mode.value})" if record.retry_mode else ""
        print(f"   {record.from_state.value} → {record.to_state.value}{retry_info}  [{record.reason}]")

    print("\n📜 执行历史:")
    for entry in agent.history:
        print(f"   [{entry.kind}] {entry.data}")

    # ── 9. 清理 ───────────────────────────────────────────────────────
    sched.stop()
    executor.shutdown()
    print("\n✅ 演示完成")


if __name__ == "__main__":
    main()
