"""
file_processing_demo.py — 多步骤文件处理 Pipeline 演示

展示场景：
  1. 文件处理工作流：列出文件 → 读取内容 → 处理 → 写入结果
  2. 使用真实内置工具（FileReadTool, FileWriteTool, FileListTool）
  3. Workspace 沙箱安全：文件操作限制在 ~/project 目录
  4. 演示 Step 间的数据流：file_list → file_read → process → file_write

运行:
  python examples/file_processing_demo.py

注意:
  - 文件操作限制在 WORKSPACE_ROOT（默认 ~/project）
  - 演示会自动创建临时文件，完成后清理
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

from tools.impl import register_all
from tools.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("Agent Runtime Framework — 文件处理 Pipeline 演示")
    print("=" * 70)

    # ── 创建临时工作目录 ─────────────────────────────────────────────
    workspace = tempfile.mkdtemp(prefix="myagent_demo_")
    print(f"\n📁 工作目录: {workspace}")

    # 创建演示文件
    test_file = os.path.join(workspace, "input.txt")
    with open(test_file, "w") as f:
        f.write("Hello, Agent Framework!\n")
        f.write("This is a test file.\n")
        f.write("Line 3: Testing file operations.\n")
        f.write("Line 4: Another line of content.\n")

    test_file2 = os.path.join(workspace, "data.csv")
    with open(test_file2, "w") as f:
        f.write("name,value\n")
        f.write("item1,100\n")
        f.write("item2,200\n")

    print(f"   已创建: input.txt, data.csv")

    # ── 1. 构建组件 ─────────────────────────────────────────────────
    queue = PriorityEventQueue()
    bus = RawEventBus()
    Dispatcher(queue).attach(bus)

    sm = StateManager()
    dv = DependencyValidator()
    pe = PolicyEngine()

    # 工具注册表（使用 FileReadTool/FileWriteTool/FileListTool）
    tool_reg = ToolRegistry()

    # 创建带自定义 workspace 的文件工具
    from tools.impl.filesystem import FileReadTool, FileWriteTool, FileListTool
    from tools.impl.echo import EchoTool

    file_read = FileReadTool(workspace_root=workspace)
    file_write = FileWriteTool(workspace_root=workspace)
    file_list = FileListTool(workspace_root=workspace)
    echo = EchoTool()

    tool_reg.register(file_read)
    tool_reg.register(file_write)
    tool_reg.register(file_list)
    tool_reg.register(echo)

    llm = MockLLM()
    sr = StepRunner(
        llm=llm,
        tool_registry=tool_reg,
        dependency_validator=dv,
    )

    sched = Scheduler(
        event_queue=queue,
        raw_event_bus=bus,
        state_manager=sm,
        step_runner=sr,
        dependency_validator=dv,
        policy_engine=pe,
        tool_registry=tool_reg,
    )

    # ── 2. 注册工具到 ToolExecutor ───────────────────────────────────
    executor = ToolExecutor(bus=bus, max_workers=2)
    for name, reg_tool in tool_reg._by_name.items():
        executor.register_tool(name, reg_tool.tool)

    # 模拟的文本处理工具
    def process_text(**params):
        content = params.get("content", "")
        # 简单处理：转大写 + 行数统计
        lines = content.strip().split("\n")
        return {
            "processed": content.upper(),
            "line_count": len(lines),
            "word_count": len(content.split()),
        }

    executor.register_tool("process_text", process_text)
    sched.set_tool_executor(executor)

    # ── 3. 构建文件处理 Pipeline ─────────────────────────────────────
    #
    # Step 0: 列出目录中的 .txt 文件
    # Step 1: 读取第一个文件内容
    # Step 2: 处理文本（大写 + 统计）
    # Step 3: 写入处理结果

    plan = Plan.create([
        Step(
            step_id="list_files",
            tool_name="file_list",
            params={"path": workspace, "pattern": "*.txt"},
            output_schema={"entries": "list", "path": "str"},
        ),
        Step(
            step_id="read_file",
            tool_name="file_read",
            params={"path": ""},
            input_bindings={"path": "list_files.entries.0"},  # 取第一个文件
            output_schema={"content": "str", "path": "str", "size": "int", "truncated": "bool"},
        ),
        Step(
            step_id="process",
            tool_name="process_text",
            params={"content": ""},
            input_bindings={"content": "read_file.content"},
            output_schema={"processed": "str", "line_count": "int", "word_count": "int"},
        ),
        Step(
            step_id="write_result",
            tool_name="file_write",
            params={"path": "", "content": ""},
            input_bindings={
                "path": "read_file.path",
                "content": "process.processed",
            },
            output_schema={"path": "str", "bytes_written": "int"},
        ),
    ], max_replans=1)

    print(f"\n📋 Pipeline: {plan.plan_id}")
    print(f"   步骤数: {len(plan.steps)}")
    print(f"   工具链: {[s.tool_name for s in plan.steps]}")
    print(f"   Workspace: {workspace}")

    # ── 4. 提交并执行 ───────────────────────────────────────────────
    print("\n🚀 提交文件处理任务...")
    agent_id = sched.submit_task(
        plan=plan,
        budget=ExecutionBudget.default(),
        task_id="file_process_001",
    )
    print(f"   Agent: {agent_id}")

    sched.start()

    # Pump 事件循环
    deadline = time.time() + 15
    while time.time() < deadline:
        agent = sm.get_agent(agent_id)
        if agent and agent.is_terminal():
            break
        time.sleep(0.1)

    # ── 5. 输出结果 ─────────────────────────────────────────────────
    agent = sm.get_agent(agent_id)
    metrics = sm.get_metrics(agent_id)

    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    print(f"   Agent:      {agent.agent_id}")
    print(f"   State:      {agent.state.value}")
    print(f"   Steps:      {metrics.step_count if metrics else '?'}")

    # 输出步骤输出
    outputs = sched.get_agent_outputs(agent_id)
    if outputs:
        print("\n📤 步骤输出:")
        for step_id, output in list(outputs.items())[:4]:
            output_str = str(output)
            print(f"   [{step_id}]: {output_str[:100]}")

    # 检查输出文件
    output_file = os.path.join(workspace, "input.txt")  # 被覆盖
    if os.path.exists(output_file):
        print(f"\n📄 输出文件内容:")
        with open(output_file) as f:
            print(f.read())

    print("\n🔄 状态转换:")
    for record in agent.state_history:
        retry_info = f" ({record.retry_mode.value})" if record.retry_mode else ""
        print(f"   {record.from_state.value} → {record.to_state.value}{retry_info}")

    # ── 6. 清理 ────────────────────────────────────────────────────
    sched.stop()
    executor.shutdown()
    shutil.rmtree(workspace)
    print(f"\n🧹 已清理临时目录: {workspace}")

    print("\n✅ 文件处理 Pipeline 演示完成")


if __name__ == "__main__":
    main()
