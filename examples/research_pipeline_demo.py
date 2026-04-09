"""
research_pipeline_demo.py — 多步骤研究 pipeline 演示

展示场景：
  1. 完整的 4 步研究 pipeline：
     - 搜索查询 → 提取 URL → 获取页面 → 生成摘要
  2. 使用真实内置工具（WebSearchTool + HttpFetchTool）
  3. 使用真正的 DeepSeek LLM 进行 replan
  4. 演示 input_bindings 自动参数注入
  5. Fallback 链处理网络错误
  6. 完整执行 trace 和 metrics

环境变量配置:
  DEEPSEEK_API_KEY      — DeepSeek API Key（用于 replan）
  SHUYAN_API_KEY        — 数眼智能 API Key（国内搜索，优先）
  SERPAPI_API_KEY       — SerpAPI API Key（国际搜索，可选）

运行:
  export DEEPSEEK_API_KEY="sk-..."
  export SHUYAN_API_KEY="sk-..."   # 数眼 API 用于搜索
  python examples/research_pipeline_demo.py
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plan import FallbackOption, Plan, Step
from core.budget import ExecutionBudget

from events.event_queue import PriorityEventQueue
from events.raw_event_bus import Dispatcher, RawEventBus

from execution.llm_factory import create_llm_from_env
from execution.step_runner import StepRunner
from execution.tool_executor import ToolExecutor

from scheduler.policy_engine import PolicyEngine
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager
from scheduler.scheduler import Scheduler

from tools.impl import register_all
from tools.registry import ToolRegistry
from tools.adapter import BUILTIN_ADAPTERS, AdapterRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("Agent Runtime Framework — 多步骤研究 Pipeline 演示")
    print("=" * 70)

    # ── 1. 构建核心组件 ────────────────────────────────────────────────
    queue = PriorityEventQueue()
    bus = RawEventBus()
    Dispatcher(queue).attach(bus)

    sm = StateManager()
    dv = DependencyValidator()
    pe = PolicyEngine()

    # 工具注册表
    tool_reg = ToolRegistry()
    register_all(tool_reg)

    # Adapter 注册表
    adapter_reg = AdapterRegistry()
    for a in BUILTIN_ADAPTERS:
        adapter_reg.register(a)

    # ── 2. 创建 LLM（支持环境变量配置）───────────────────────────────
    #
    # 优先级：
    #   1. 显式设置 MYAGENT_LLM_PROVIDER
    #   2. 若有 DEEPSEEK_API_KEY → deepseek
    #   3. 否则 mock
    #
    llm = create_llm_from_env()
    logger.info(f"LLM Provider: {llm.provider_name} / Model: {llm.model_name}")

    sr = StepRunner(
        llm=llm,
        tool_registry=tool_reg,
        dependency_validator=dv,
        adapter_registry=adapter_reg,
    )

    sched = Scheduler(
        event_queue=queue,
        raw_event_bus=bus,
        state_manager=sm,
        step_runner=sr,
        dependency_validator=dv,
        policy_engine=pe,
        tool_registry=tool_reg,
        adapter_registry=adapter_reg,
    )

    # ── 3. 注册工具到 ToolExecutor ─────────────────────────────────────
    executor = ToolExecutor(bus=bus, max_workers=4)
    for name, tool in tool_reg._by_name.items():
        executor.register_tool(name, tool.tool)
    sched.set_tool_executor(executor)

    # ── 4. 注册辅助工具（提取 URL 和摘要）────────────────────────────

    def extract_url_tool(**params):
        """从搜索结果列表提取第一个 URL"""
        results = params.get("results", [])
        if not results:
            return {"url": ""}

        first = results[0] if isinstance(results[0], dict) else results[0]
        if isinstance(first, dict):
            url = first.get("url", "")
        else:
            url = str(first)

        title = ""
        if isinstance(first, dict):
            title = first.get("title", "")

        return {"url": url, "title": title}

    def summarize_tool(**params):
        """
        简单文本摘要工具。
        在真实场景中，这里可以调用 LLM 进行更智能的摘要。
        """
        text = params.get("text", params.get("content", ""))
        if not text:
            return {"summary": "", "word_count": 0}

        # 简单截取前 200 字符作为摘要
        summary = text[:200].strip()
        if len(text) > 200:
            summary += "..."

        # 统计词数
        word_count = len(text.split())

        return {
            "summary": summary,
            "word_count": word_count,
            "source": params.get("source", "unknown"),
        }

    executor.register_tool("extract_url", extract_url_tool)
    executor.register_tool("summarize_text", summarize_tool)

    # ── 5. 构建 5 步研究 Pipeline ─────────────────────────────────────
    #
    # Step 0: 搜索关键词（使用真正的 web_search 工具）
    # Step 1: 提取第一个 URL
    # Step 2: 获取页面内容
    # Step 3: 提取纯文本
    # Step 4: 生成摘要

    # 研究主题
    research_topic = "Python agent framework 2024"

    plan = Plan.create([
        Step(
            step_id="search",
            tool_name="web_search",
            params={"q": research_topic, "max_results": 5},
            output_schema={
                "results": "list",
                "answer": "str",
                "source": "str",
            },
        ),
        Step(
            step_id="extract_url",
            tool_name="extract_url",
            params={"results": []},
            input_bindings={"results": "search.results"},
            output_schema={"url": "str", "title": "str"},
        ),
        Step(
            step_id="fetch_page",
            tool_name="http_fetch",
            params={"url": ""},
            input_bindings={"url": "extract_url.url"},
            output_schema={"content": "str", "status_code": "int", "url": "str"},
            fallback_chain=[
                FallbackOption("http_fetch", {
                    "url": "https://httpbin.org/html"
                }),
            ],
        ),
        Step(
            step_id="summarize",
            tool_name="summarize_text",
            params={"text": "", "source": ""},
            input_bindings={
                "text": "fetch_page.content",
                "source": "fetch_page.url",
            },
            output_schema={"summary": "str", "word_count": "int", "source": "str"},
        ),
    ], max_replans=2)

    print(f"\n📋 Pipeline: {plan.plan_id}")
    print(f"   主题: {research_topic}")
    print(f"   步骤数: {len(plan.steps)}")
    print(f"   工具链: {[s.tool_name for s in plan.steps]}")
    print(f"\n   数据流:")
    for s in plan.steps:
        if s.input_bindings:
            for param, binding in s.input_bindings.items():
                print(f"     {s.step_id}.{param} ← {binding}")

    # 显示搜索 API 后端
    search_tool = tool_reg.get("web_search")
    backend = getattr(search_tool, '_backend', 'unknown')
    print(f"\n🔍 搜索后端: {backend}")

    # ── 6. 提交并执行 ─────────────────────────────────────────────────
    print("\n🚀 提交研究任务...")
    agent_id = sched.submit_task(
        plan=plan,
        budget=ExecutionBudget.default(),
        task_id="research_001",
    )
    print(f"   Agent: {agent_id}")

    sched.start()

    # Pump 事件循环
    deadline = time.time() + 30
    while time.time() < deadline:
        agent = sm.get_agent(agent_id)
        if agent and agent.is_terminal():
            break
        time.sleep(0.2)

    # ── 7. 输出结果 ────────────────────────────────────────────────────
    agent = sm.get_agent(agent_id)
    metrics = sm.get_metrics(agent_id)

    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    print(f"   Agent:      {agent.agent_id}")
    print(f"   状态:       {agent.state.value}")
    print(f"   执行步数:   {metrics.step_count if metrics else '?'}")
    print(f"   LLM 调用:   {metrics.llm_call_count if metrics else '?'}")
    print(f"   Replan次数: {metrics.replan_count if metrics else '?'}")

    # 输出已完成步骤的输出
    outputs = sched.get_agent_outputs(agent_id)
    if outputs:
        print("\n📤 步骤输出:")
        for step_id, output in outputs.items():
            if step_id == "search":
                # 搜索结果详细展示
                results = output.get("results", [])
                print(f"   [{step_id}]:")
                print(f"       source: {output.get('source', 'unknown')}")
                print(f"       results count: {len(results)}")
                for i, r in enumerate(results[:3], 1):
                    print(f"       [{i}] {r.get('title', '')[:60]}")
                    print(f"           URL: {r.get('url', '')[:60]}")
            elif step_id == "extract_url":
                print(f"   [{step_id}]:")
                print(f"       url: {output.get('url', '')}")
                print(f"       title: {output.get('title', '')}")
            elif step_id == "summarize":
                print(f"   [{step_id}]:")
                print(f"       summary: {output.get('summary', '')[:100]}...")
                print(f"       word_count: {output.get('word_count', 0)}")
                print(f"       source: {output.get('source', '')}")
            else:
                preview = str(output)[:150]
                print(f"   [{step_id}]: {preview}...")

    print("\n🔄 状态转换:")
    for record in agent.state_history:
        retry_info = f" ({record.retry_mode.value})" if record.retry_mode else ""
        print(f"   {record.from_state.value} → {record.to_state.value}{retry_info}")

    sched.stop()
    executor.shutdown()

    print("\n" + "=" * 70)
    print("✅ 研究 Pipeline 演示完成")
    print("""
本演示展示了 Agent 框架的核心能力：

1. 多步骤 Pipeline：4 个步骤自动串联执行
2. 数据绑定：input_bindings 自动将上一步输出注入下一步
3. Fallback 链：主工具失败时自动降级（触发 FALLBACK_MODE）
4. 真实 LLM：使用 DeepSeek 进行智能 replan（需要 API Key）
5. 真实搜索：集成数眼/SerpAPI/DuckDuckGo 搜索服务
    """)


if __name__ == "__main__":
    main()
