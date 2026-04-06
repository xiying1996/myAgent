# Agent Runtime Framework

> 一个模块化、可观测、自愈能力强的 Agent 执行框架。控制流与数据流分离，LLM 仅负责智能决策，调度完全由状态机驱动。

---

## 目录

- [Agent Runtime Framework](#agent-runtime-framework)
  - [目录](#目录)
  - [1. 设计理念](#1-设计理念)
    - [核心原则](#核心原则)
    - [为什么这样设计](#为什么这样设计)
  - [2. 架构总览](#2-架构总览)
  - [3. 分层设计](#3-分层设计)
  - [4. 核心模块详解](#4-核心模块详解)
    - [4.1 Agent](#41-agent)
    - [4.2 Scheduler](#42-scheduler)
    - [4.3 StepRunner + LLM](#43-steprunner--llm)
    - [4.4 ToolExecutor](#44-toolexecutor)
    - [4.5 EventQueue（双优先级）](#45-eventqueue双优先级)
    - [4.6 StateManager](#46-statemanager)
    - [4.7 DependencyValidator](#47-dependencyvalidator)
    - [4.8 PolicyEngine](#48-policyengine)
    - [4.9 CheckpointManager](#49-checkpointmanager)
  - [5. 状态机](#5-状态机)
    - [合法状态转换表](#合法状态转换表)
    - [状态机图](#状态机图)
    - [RETRYING 内部语义](#retrying-内部语义)
  - [6. 局部 Replan 机制](#6-局部-replan-机制)
    - [Replan Prompt 构建](#replan-prompt-构建)
    - [Replan 范围约束（系统强制，不依赖 LLM 自律）](#replan-范围约束系统强制不依赖-llm-自律)
  - [7. Fallback Chain 与 RETRYING](#7-fallback-chain-与-retrying)
    - [Fallback Chain 定义](#fallback-chain-定义)
    - [执行逻辑](#执行逻辑)
  - [8. Checkpoint 与 Replay](#8-checkpoint-与-replay)
    - [Event Log 结构](#event-log-结构)
    - [Replay 流程](#replay-流程)
    - [Snapshot 触发策略](#snapshot-触发策略)
  - [9. 多 Agent 并行](#9-多-agent-并行)
    - [Agent 间通信](#agent-间通信)
    - [跨 Agent 依赖声明](#跨-agent-依赖声明)
    - [并发安全](#并发安全)
  - [10. 数据结构定义](#10-数据结构定义)
  - [11. 执行流程示例](#11-执行流程示例)
    - [场景：3 步 Plan，Step1 失败后 fallback 成功](#场景3-步-planstep1-失败后-fallback-成功)
  - [12. 开发路线图](#12-开发路线图)
    - [Phase 1：数据模型 + 状态机（Day 1-3）](#phase-1数据模型--状态机day-1-3)
    - [Phase 2：执行层（Day 4-7）](#phase-2执行层day-4-7)
    - [Phase 3：可观测 + 可恢复（Day 8-11）](#phase-3可观测--可恢复day-8-11)
    - [Phase 4：多 Agent + 收尾（Day 12-15）](#phase-4多-agent--收尾day-12-15)
  - [13. 已知限制与后续规划](#13-已知限制与后续规划)
    - [当前版本限制（Python 原型）](#当前版本限制python-原型)
    - [C++ 重构优先级（后续阶段）](#c-重构优先级后续阶段)
    - [接入真实 LLM 的注意点](#接入真实-llm-的注意点)
  - [快速开始](#快速开始)
  - [目录结构](#目录结构)

---

## 1. 设计理念

### 核心原则

| 原则 | 说明 |
|------|------|
| **控制流与数据流分离** | Scheduler + 状态机决定谁执行、何时执行；Event 驱动数据更新，互不干扰 |
| **责任单一化** | 每个模块只做一件事，任何模块故障不拖垮全系统 |
| **LLM 边界明确** | LLM 只参与局部 Replan 决策，不干预调度、等待、状态更新 |
| **Plan + Policy 分离** | Plan 由 LLM 提出，Policy 由系统约束，智能与可控性兼顾 |
| **可观测优先** | 全链路事件日志 + 状态历史，支持 debug / replay / tracing |
| **自愈能力** | fallback chain → 局部 Replan → System Fallback，多层防护 |

### 为什么这样设计

大多数 Agent 框架的问题在于让 LLM 既做决策又影响调度，导致系统行为不可预测。本框架将 LLM 严格限制在"计划修改"角色，Scheduler 由确定性状态机驱动，从根本上保证了系统稳定性。

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   Task Submission API                       │
│              submit_task(plan) → task_id                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       Scheduler                             │
│   多 Agent 调度 │ 状态机驱动 │ 事件驱动唤醒 │ Policy 检查   │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────────────┐
│      StepRunner     │   │         PolicyEngine            │
│  LLM 局部 Replan    │   │  Budget / Whitelist / Timeout   │
│  Fallback Chain 决策│   └─────────────────────────────────┘
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ToolExecutor (async)                     │
│            ThreadPool / asyncio │ 超时检测                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RawEventBus → Dispatcher                 │
│   HighPriority Queue          Normal Queue                  │
│   TIMEOUT/ERROR/SYSTEM        TOOL_RESULT/PLAN_UPDATE       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      StateManager                           │
│   Agent 状态更新 │ 执行历史 │ Metrics │ Event Log           │
└───────────┬─────────────────────┬───────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────┐   ┌────────────────────────────────────┐
│    Scheduler    │   │         CheckpointManager          │
│   (唤醒 Agent)  │   │   Snapshot │ Event Log │ Replay    │
└─────────────────┘   └────────────────────────────────────┘
```

---

## 3. 分层设计

```
Layer 0  │  Task Submission API          对外入口，接收任务
─────────┼────────────────────────────────────────────────
Layer 1  │  Scheduler + PolicyEngine     调度与安全边界
─────────┼────────────────────────────────────────────────
Layer 2  │  Agent + StateMachine         Agent 持有状态机
─────────┼────────────────────────────────────────────────
Layer 3  │  StepRunner + LLM             决策层（局部 Replan）
─────────┼────────────────────────────────────────────────
Layer 4  │  DependencyValidator          依赖图检查与修复
─────────┼────────────────────────────────────────────────
Layer 5  │  ToolExecutor                 异步执行层
─────────┼────────────────────────────────────────────────
Layer 6  │  RawEventBus + EventQueue     事件路由层
─────────┼────────────────────────────────────────────────
Layer 7  │  StateManager                 状态持久层
─────────┼────────────────────────────────────────────────
Layer 8  │  CheckpointManager            可恢复层
```

---

## 4. 核心模块详解

### 4.1 Agent

Agent 是执行的基本单元，持有：
- 当前 Plan（Step 列表 + 依赖关系）
- 状态机实例
- 执行历史
- Execution Budget 消耗记录

**职责：** 持有状态，不做决策。

```python
@dataclass
class Agent:
    agent_id: str
    plan: Plan
    state: AgentState
    history: List[HistoryEntry]
    budget_usage: BudgetUsage
    created_at: float
```

---

### 4.2 Scheduler

Scheduler 是系统的调度中枢，负责：

- 维护所有 Agent 的生命周期
- 消费 EventQueue（优先消费 HighPriority）
- 根据 Event 触发状态转换（每次转换前调用 PolicyEngine）
- 唤醒处于 READY 状态的 Agent，交给 StepRunner

**关键原则：** Scheduler 不做业务决策，只做调度决策。

```
事件到达 → Scheduler 消费 → 调用 PolicyEngine 检查 
→ 合法则更新 StateManager → 唤醒对应 Agent → 交给 StepRunner
```

---

### 4.3 StepRunner + LLM

StepRunner 负责决定 Agent 的下一步行动，是唯一与 LLM 交互的模块。

**LLM 的边界：**

| 范围 | 是否允许 LLM 修改 | 说明 |
|------|:-----------------:|------|
| 当前 Step 的 fallback chain | ✅ | 替换工具、调整参数 |
| 后续未执行的 Step | ⚠️ 有限 | 仅 fallback 范围内微调，不全局重排 |
| 已成功执行的 Step | ❌ | 保持系统稳定性 |
| 全 Plan 重新排序 | ❌ 受限 | 仅 Budget 允许时考虑 |

**局部 Replan 流程：**

```
Step 失败
  → StepRunner 构建 Replan Prompt
    （当前失败原因 + 当前 Step + 后续 Steps + Budget 剩余）
  → LLM 返回更新后的局部 Plan
  → StepRunner 校验修改范围是否合法
  → 合法则交给 DependencyValidator 检查下游影响
  → 更新 Plan，继续执行
```

**Mock LLM（测试用）：**

开发阶段 LLM 接口使用 Mock 实现，固定返回 fallback plan，不依赖真实 API。

---

### 4.4 ToolExecutor

ToolExecutor 异步执行工具调用，与 StepRunner 解耦。

**职责：**
- 接收 Action（tool_name + params）
- 通过 ThreadPoolExecutor / asyncio 异步执行
- 超时检测（独立计时器）
- 执行完成 → 发送 Event 到 RawEventBus

**工具注册：**

```python
executor = ToolExecutor()
executor.register_tool("web_search", web_search_fn)
executor.register_tool("calculator", calculator_fn)
```

**TE 不做路由判断**，统一发到 RawEventBus，由 Dispatcher 根据事件类型分发：

```
ToolExecutor → RawEventBus → Dispatcher → HighPriority / Normal Queue
```

---

### 4.5 EventQueue（双优先级）

系统使用双队列设计，Scheduler 优先消费 HighPriority：

| 队列 | 事件类型 | 说明 |
|------|---------|------|
| HighPriority | `TIMEOUT` / `ERROR` / `SYSTEM_FALLBACK` / `BUDGET_EXCEEDED` | 必须立即处理 |
| Normal | `TOOL_RESULT` / `PLAN_UPDATE` / `OBSERVATION` / `STATE_UPDATE` | 按序处理 |

**消费逻辑：**

```python
def get_next_event(self) -> Event:
    if not self.high_priority_queue.empty():
        return self.high_priority_queue.get()
    return self.normal_queue.get(timeout=1.0)
```

---

### 4.6 StateManager

StateManager 是所有状态变更的唯一入口，负责：

- 接收 Event → 更新 Agent 状态
- 维护完整执行历史 `history[]`
- 记录 Metrics（步骤数、LLM 调用数、总耗时）
- 持久化 Event Log（JSON / protobuf）
- 触发 Snapshot（由 CheckpointManager 调用）

**关键约束：** 状态更新必须通过 StateManager，任何模块不直接修改 Agent 状态字段。

---

### 4.7 DependencyValidator

负责在每次 Replan 后检查 Step 依赖链是否仍然有效。

**Step 依赖声明：**

```python
@dataclass
class Step:
    step_id: str
    tool_name: str
    params: Dict[str, Any]
    fallback_chain: List[FallbackOption]
    output_schema: Dict[str, str]        # 输出字段类型声明
    input_bindings: Dict[str, str]       # {"param_name": "step_id.field"}
    dependencies: List[str]              # 依赖的 step_id 列表
```

**示例：**

```
Step0: tool=web_search, output_schema={url: str, title: str}
Step1: tool=fetch_page, input_bindings={target_url: "step0.url"}
Step2: tool=summarize,  input_bindings={content: "step1.html"}
```

**Replan 后触发流程：**

```
Step0 换了工具 → 新工具 output_schema 变化
→ DependencyValidator 检测到 Step1.input_bindings["target_url"] 
   引用的 "step0.url" 字段仍存在 → 合法，继续
→ 若字段消失 → 通知 StepRunner 要求 LLM 修复 Step1 的 input_bindings
```

**启动时检查循环依赖**（DAG 拓扑排序），发现循环则 fail-fast。

---

### 4.8 PolicyEngine

PolicyEngine 在每次状态转换前运行，是系统的安全边界。

**Execution Budget：**

```python
@dataclass
class ExecutionBudget:
    max_replans: int = 3           # 最大 Replan 次数
    max_total_steps: int = 20      # 防止 Plan 爆炸
    max_llm_calls: int = 10        # LLM 调用次数上限
    wall_clock_timeout: float = 300.0  # 整个 Task 的硬超时（秒）
    allowed_tools: List[str] = None    # 工具白名单（None = 不限制）
```

**检查顺序：**

```
1. wall_clock_timeout 检查（优先级最高）
2. max_total_steps 检查
3. max_llm_calls 检查（进入 Replan 前）
4. max_replans 检查（进入 Replan 前）
5. allowed_tools 检查（工具执行前）
```

任何检查失败 → 注入 `BUDGET_EXCEEDED` 事件到 HighPriority 队列 → Agent 进入 ERROR 状态。

---

### 4.9 CheckpointManager

负责 Snapshot 与 Replay，支持两种恢复语义：

| 模式 | 触发 | LLM 调用 | 用途 |
|------|------|---------|------|
| Debug Replay | 手动 | ❌ 用 PlanSnapshot | 复现问题，确定性重放 |
| Recovery Replay | 崩溃恢复 | ✅ 可重新调用 | 从断点继续执行 |

**Snapshot 内容：**

```python
@dataclass
class Snapshot:
    snapshot_id: str
    agent_id: str
    timestamp: float
    agent_state: AgentState
    plan: Plan
    budget_usage: BudgetUsage
    history: List[HistoryEntry]
    last_event_id: str             # 对应 Event Log 的位置
```

---

## 5. 状态机

### 合法状态转换表

```
READY     → RUNNING     # Scheduler 分配 Agent 执行当前 Step
RUNNING   → WAITING     # ToolExecutor 开始异步执行，等待结果
WAITING   → READY       # Tool 成功完成，准备下一步
WAITING   → RETRYING    # Tool 失败 / Timeout，触发 Fallback 机制
RETRYING  → WAITING     # 执行 Fallback 或 Replan 后，重新等待
RETRYING  → ERROR       # Budget 耗尽，无法继续
READY     → DONE        # 所有 Step 执行完成
任意状态  → ERROR       # System Fallback / wall_clock_timeout
```

### 状态机图

```
                    ┌──────────────────────────────┐
                    │             READY             │
                    └──────┬───────────────────┬───┘
                           │ start step        │ all steps done
                           ▼                   ▼
                    ┌──────────────┐     ┌──────────┐
                    │   RUNNING    │     │   DONE   │
                    └──────┬───────┘     └──────────┘
                           │ submit tool async
                           ▼
                    ┌──────────────┐
                    │   WAITING    │
                    └──┬───────┬───┘
           tool done   │       │ tool fail / timeout
                       ▼       ▼
                    READY   ┌──────────────┐
                            │   RETRYING   │
                            └──┬───────┬───┘
               fallback/replan │       │ budget exceeded
                               ▼       ▼
                           WAITING   ERROR
```

### RETRYING 内部语义

RETRYING 状态区分两种内部模式（通过 `retry_mode` 字段标记）：

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| `fallback_mode` | 当前 Step 有 fallback 可用 | 直接执行 fallback chain 中下一个工具 |
| `replan_mode` | fallback chain 耗尽 | 调用 StepRunner → LLM 局部 Replan |

---

## 6. 局部 Replan 机制

### Replan Prompt 构建

```
System: 你是一个 Agent 执行规划助手。
        你只能修改 [当前失败 Step] 的 fallback chain 和后续未执行步骤。
        不能修改已成功执行的步骤。
        不能全局重排 Plan 顺序。
        
User: 当前执行状态:
  - 已完成 Steps: [step0(成功), step1(成功)]
  - 当前失败 Step: step2, 工具: web_scraper, 错误: ConnectionTimeout
  - 剩余 Budget: max_replans=2, max_llm_calls=5
  - 后续 Steps: [step3, step4]
  
  请提出修复 step2 的局部 Plan（允许调整 step3/step4 的参数，但不改变顺序）。
```

### Replan 范围约束（系统强制，不依赖 LLM 自律）

```python
def validate_replan(original_plan: Plan, new_plan: Plan, 
                    completed_step_ids: List[str]) -> bool:
    for step_id in completed_step_ids:
        if original_plan[step_id] != new_plan[step_id]:
            raise ReplanViolation(f"禁止修改已完成的 Step: {step_id}")
    
    if new_plan.step_order != original_plan.step_order:
        raise ReplanViolation("局部 Replan 不允许重排 Step 顺序")
    
    return True
```

---

## 7. Fallback Chain 与 RETRYING

### Fallback Chain 定义

每个 Step 在 Plan 初始化时声明 fallback chain：

```python
Step(
    step_id="step0",
    tool_name="web_search_v2",          # Primary Tool
    params={"query": "..."},
    fallback_chain=[
        FallbackOption(tool="web_search_v1", params={"query": "..."}),
        FallbackOption(tool="bing_search",   params={"query": "..."}),
    ],
    output_schema={"url": "str", "title": "str"}
)
```

### 执行逻辑

```
Step0 执行 Primary Tool(web_search_v2) → 失败
  → RETRYING (fallback_mode)
  → 执行 fallback[0]: web_search_v1 → 失败
  → 执行 fallback[1]: bing_search → 成功
  → READY，继续下一 Step

若 fallback chain 全部耗尽:
  → RETRYING (replan_mode)
  → StepRunner → LLM 局部 Replan
  → 生成新 fallback chain
  → 若 Replan 次数耗尽 → PolicyEngine → BUDGET_EXCEEDED → ERROR
```

---

## 8. Checkpoint 与 Replay

### Event Log 结构

每条 Event 记录：

```python
@dataclass
class EventRecord:
    event_id: str                  # 全局唯一 ID
    agent_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: float
    is_plan_snapshot: bool = False  # LLM 输出是否被记录为 PlanSnapshot
```

`PlanSnapshot` 是特殊的 Event，记录 LLM 的完整输出，用于 Debug Replay 的确定性重放。

### Replay 流程

**Debug Replay（确定性）：**

```
1. 读取 Event Log（从头或从指定 event_id）
2. 清空 Agent 状态
3. 按顺序重放事件
4. 遇到 LLM 调用点 → 读取 PlanSnapshot，不重新调用 LLM
5. 完整复现执行过程
```

**Recovery Replay（从断点恢复）：**

```
1. 读取最近 Snapshot
2. 恢复 Agent 状态 + Budget 消耗
3. 重放 Snapshot 之后的 Event Log
4. 到达崩溃点后继续执行，LLM 可重新调用
```

### Snapshot 触发策略

- 每完成 N 个 Step（默认 N=3）
- 每次成功 Replan 后
- Task 完成时（DONE / ERROR）

---

## 9. 多 Agent 并行

### Agent 间通信

所有 Agent 输出通过 EventQueue 统一路由，Scheduler 按 `agent_id` 分发，Agent 之间没有直接通信通道。

```
Agent A 完成 → 发送 TOOL_RESULT(agent_id=A, output={data}) 
→ EventQueue → Scheduler
→ Scheduler 检查：是否有 Agent B 等待 Agent A 的输出？
→ 若有：将 Agent A 的 output 绑定到 Agent B 的下一步 Step 参数
→ Agent B 状态 WAITING → READY
```

### 跨 Agent 依赖声明

```python
Plan(
    agents=[
        AgentPlan(agent_id="A", steps=[...]),
        AgentPlan(agent_id="B", steps=[...],
                  depends_on={"step0.input_data": "A.final_output"})
    ]
)
```

### 并发安全

- 每个 Agent 有独立状态（无共享可变状态）
- EventQueue 消费为串行（初版），避免竞态
- StateManager 内部对单个 Agent 的更新串行化

---

## 10. 数据结构定义

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ===== 枚举 =====

class AgentState(Enum):
    READY    = "READY"
    RUNNING  = "RUNNING"
    WAITING  = "WAITING"
    RETRYING = "RETRYING"
    DONE     = "DONE"
    ERROR    = "ERROR"

class EventType(Enum):
    # HighPriority
    TIMEOUT          = "TIMEOUT"
    ERROR            = "ERROR"
    SYSTEM_FALLBACK  = "SYSTEM_FALLBACK"
    BUDGET_EXCEEDED  = "BUDGET_EXCEEDED"
    # Normal
    TOOL_RESULT      = "TOOL_RESULT"
    PLAN_UPDATE      = "PLAN_UPDATE"
    OBSERVATION      = "OBSERVATION"
    STATE_UPDATE     = "STATE_UPDATE"

class RetryMode(Enum):
    FALLBACK_MODE = "FALLBACK_MODE"
    REPLAN_MODE   = "REPLAN_MODE"

# ===== 核心数据结构 =====

@dataclass
class FallbackOption:
    tool: str
    params: Dict[str, Any]

@dataclass
class Step:
    step_id: str
    tool_name: str
    params: Dict[str, Any]
    fallback_chain: List[FallbackOption]
    output_schema: Dict[str, str]
    input_bindings: Dict[str, str]         # {"param": "step_id.field"}
    dependencies: List[str]                # 依赖的 step_id

@dataclass
class Plan:
    plan_id: str
    steps: List[Step]
    replan_count: int = 0
    max_replans: int = 3

@dataclass
class ExecutionBudget:
    max_replans: int = 3
    max_total_steps: int = 20
    max_llm_calls: int = 10
    wall_clock_timeout: float = 300.0
    allowed_tools: Optional[List[str]] = None

@dataclass
class BudgetUsage:
    replan_count: int = 0
    total_steps: int = 0
    llm_call_count: int = 0
    start_time: float = 0.0

@dataclass
class Event:
    event_id: str
    agent_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: float
    priority: str = "normal"           # "high" / "normal"

@dataclass
class Agent:
    agent_id: str
    plan: Plan
    state: AgentState
    budget: ExecutionBudget
    budget_usage: BudgetUsage
    history: List[Dict] = field(default_factory=list)
    retry_mode: Optional[RetryMode] = None
    created_at: float = 0.0

@dataclass
class Snapshot:
    snapshot_id: str
    agent_id: str
    timestamp: float
    agent_state: AgentState
    plan: Plan
    budget_usage: BudgetUsage
    history: List[Dict]
    last_event_id: str
```

---

## 11. 执行流程示例

### 场景：3 步 Plan，Step1 失败后 fallback 成功

```
Task 提交
  Plan: [Step0(web_search), Step1(fetch_page), Step2(summarize)]

[Step0]
  Scheduler: Agent → READY → RUNNING
  StepRunner: 提交 Tool(web_search) 给 ToolExecutor
  Agent → WAITING
  ToolExecutor: 执行成功 → Event(TOOL_RESULT, step0, {url:...})
  StateManager: 更新 history
  Agent → READY

[Step1]
  Scheduler: Agent → RUNNING
  StepRunner: 提交 Tool(fetch_page, url=step0.url) 给 ToolExecutor
  Agent → WAITING
  ToolExecutor: 执行失败(ConnectionTimeout) → Event(TIMEOUT, step1)
  Dispatcher: TIMEOUT → HighPriority Queue
  Scheduler: 消费 HighPriority 事件
  PolicyEngine: 检查 Budget → 通过
  Agent → RETRYING(fallback_mode)
  StepRunner: 执行 fallback[0](fetch_page_v2, url=step0.url)
  Agent → WAITING
  ToolExecutor: 执行成功 → Event(TOOL_RESULT, step1, {html:...})
  Agent → READY

[Step2]
  DependencyValidator: 确认 step2.input_bindings["content"] = "step1.html" 有效 ✓
  Scheduler: Agent → RUNNING
  StepRunner: 提交 Tool(summarize, content=step1.html)
  Agent → WAITING
  ToolExecutor: 执行成功 → Event(TOOL_RESULT, step2, {summary:...})
  Agent → READY → DONE

CheckpointManager: 保存 DONE Snapshot
```

---

## 12. 开发路线图

### Phase 1：数据模型 + 状态机（Day 1-3）

| Day | 目标 | 关键产出 |
|-----|------|---------|
| 1 | 核心数据结构 | Step / Plan / Event / Agent / Budget dataclass |
| 2 | 状态机实现 | StateMachine 类，合法转换表，非法转换异常 |
| 3 | EventQueue | PriorityEventQueue，RawEventBus + Dispatcher |

### Phase 2：执行层（Day 4-7）

| Day | 目标 | 关键产出 |
|-----|------|---------|
| 4 | ToolExecutor | 异步执行，工具注册，超时检测 |
| 5 | StepRunner + LLM Interface | Mock LLM，局部 Replan 范围校验 |
| 6 | DependencyValidator + Scheduler | DAG 解析，循环检测，Scheduler 主循环 |
| 7 | 单 Agent 端到端集成测试 | 4 种场景全部通过 |

### Phase 3：可观测 + 可恢复（Day 8-11）

| Day | 目标 | 关键产出 |
|-----|------|---------|
| 8 | StateManager + Event Log | 全链路日志，JSON 持久化 |
| 9 | PolicyEngine | Budget 检查，工具白名单 |
| 10 | CheckpointManager + Snapshot | 定期快照，PlanSnapshot 事件 |
| 11 | Replay（Debug + Recovery） | 两种 Replay 模式完整测试 |

### Phase 4：多 Agent + 收尾（Day 12-15）

| Day | 目标 | 关键产出 |
|-----|------|---------|
| 12 | 多 Agent 调度 | 并行执行，跨 Agent 依赖 |
| 13 | RETRYING 子状态细化 | fallback_mode / replan_mode 区分 |
| 14 | 压力测试 | 10 Agent 并发，30% 随机失败，无死锁 |
| 15 | 文档 + 接口整理 | 对外 API，README，C++ 重构接口准备 |

---

## 13. 已知限制与后续规划

### 当前版本限制（Python 原型）

| 限制 | 影响 | 规划解决方式 |
|------|------|-------------|
| EventQueue 串行消费 | 高并发吞吐受限 | Phase 2：Agent 粒度锁 + 并行消费 |
| Event Log 存 JSON 文件 | 大规模场景性能差 | 替换为 SQLite 或 protobuf |
| LLM 接口为 Mock | 不测试真实 Replan 效果 | Day 5 后接入真实 LLM 进行集成测试 |
| 无分布式支持 | 单进程限制 | C++ 重构阶段引入分布式 Scheduler |
| Snapshot 粒度固定 | 可能浪费存储 | 自适应 Snapshot 策略（基于 Step 重要性） |

### C++ 重构优先级（后续阶段）

1. **StateMachine + Scheduler**（核心热路径）
2. **EventQueue**（无锁队列，性能关键）
3. **ToolExecutor**（线程池 + io_uring）
4. **StateManager**（内存模型优化）
5. **CheckpointManager**（protobuf 序列化）

### 接入真实 LLM 的注意点

- LLM 接口需实现幂等重试（网络抖动）
- LLM 输出必须经过 `validate_replan()` 校验，不信任原始输出
- LLM 调用耗时纳入 `wall_clock_timeout` 计算
- 所有 LLM 输出保存为 `PlanSnapshot` 事件，保证 Debug Replay 确定性

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单 Agent 示例
python examples/single_agent_demo.py

# 运行测试套件
pytest tests/ -v

# 运行压力测试
python tests/stress_test.py --agents 10 --fail-rate 0.3
```

---

## 目录结构

```
agent_runtime/
├── core/
│   ├── agent.py              # Agent 数据结构
│   ├── state_machine.py      # 状态机
│   ├── plan.py               # Plan / Step / FallbackOption
│   └── budget.py             # ExecutionBudget / BudgetUsage
├── scheduler/
│   ├── scheduler.py          # Scheduler 主循环
│   └── policy_engine.py      # PolicyEngine
├── execution/
│   ├── step_runner.py        # StepRunner
│   ├── tool_executor.py      # ToolExecutor（异步）
│   └── llm_interface.py      # LLM 抽象接口 + Mock 实现
├── events/
│   ├── event_queue.py        # PriorityEventQueue
│   ├── raw_event_bus.py      # RawEventBus + Dispatcher
│   └── event_types.py        # EventType 枚举
├── state/
│   ├── state_manager.py      # StateManager
│   └── dependency_validator.py # DependencyValidator
├── checkpoint/
│   ├── checkpoint_manager.py # CheckpointManager
│   └── replay.py             # Debug / Recovery Replay
├── examples/
│   └── single_agent_demo.py  # 快速演示
└── tests/
    ├── test_state_machine.py
    ├── test_event_queue.py
    ├── test_tool_executor.py
    ├── test_step_runner.py
    ├── test_dependency_validator.py
    ├── test_policy_engine.py
    ├── test_checkpoint.py
    ├── test_integration.py   # 端到端集成测试
    └── stress_test.py        # 压力测试
```

---

*本文档随开发进度持续更新。当前版本对应 Phase 1 设计阶段。*