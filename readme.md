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
  - [9. 多 Agent 并行（设计目标，当前未完成）](#9-多-agent-并行设计目标当前未完成)
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
  - [13. 当前状态、限制与后续规划](#13-当前状态限制与后续规划)
    - [当前实现快照（2026-04）](#当前实现快照2026-04)
    - [当前版本限制（Python 原型）](#当前版本限制python-原型)
    - [Treasure Map 对齐与下一阶段](#treasure-map-对齐与下一阶段)
      - [当前定位](#当前定位)
      - [下一阶段：补齐 Level 2（最优先）](#下一阶段补齐-level-2最优先)
      - [再下一阶段：冲击 Level 3](#再下一阶段冲击-level-3)
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
| 后续未执行的 Step | ⚠️ 有限 | 仅允许参数微调（`step_param_updates`），不改 `tool_name / step_id / 顺序` |
| 已成功执行的 Step | ❌ | 保持系统稳定性 |
| 全 Plan 重新排序 | ❌ | 当前实现明确禁止 |

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

**LLM Provider（当前实现）：**

当前仓库同时支持：

- `MockLLM`：开发 / 单测 / 离线演示
- `DeepSeekLLM`：真实 API 接入，支持重试、结构化解析、`LLMCallError` 语义
- `create_llm_from_env()`：运行时 provider 选择（`mock` / `deepseek`）

成功的 Replan 会把 `provider / model / raw_response / normalized_result`
一起写入 `PLAN_UPDATE` 事件，供 Checkpoint / Debug Replay 审计。

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
    priority: EventPriority
    is_plan_snapshot: bool = False  # LLM 输出是否被记录为 PlanSnapshot
```

当 `is_plan_snapshot=True` 时：

- `payload["plan"]` 保存 Replan 后的完整 Plan 快照
- `payload["llm"]` 保存 `provider / model / raw_response / normalized_result`

也就是说，当前实现已经能把真实 LLM 输出作为 `PlanSnapshot` 事件的一部分落到日志里。

### Replay 流程

**Debug Replay（确定性）：**

```
1. 读取 Event Log（从头或从指定 event_id）
2. 清空 Agent 状态
3. 按顺序重放事件
4. 遇到 LLM 调用点 → 读取 `PlanSnapshot.payload["llm"]`，不重新调用 LLM
5. 输出确定性的状态摘要（`completed_steps / final_state / latest_plan_snapshot / latest_llm_output`）
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

## 9. 多 Agent 并行（设计目标，当前未完成）

当前仓库的 `Scheduler` 可以同时持有多个 Agent，但还没有完成跨 Agent 数据依赖、
通信协议和冲突解决。因此本节描述的是目标形态，而不是已经落地的 API。

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
# 目标形态（当前代码尚未实现）
Plan(
    agents=[
        AgentPlan(agent_id="A", steps=[...]),
        AgentPlan(agent_id="B", steps=[...],
                  depends_on={"step0.input_data": "A.final_output"})
    ]
)
```

### 并发安全

- 当前已具备：每个 Agent 独立状态、EventQueue 串行消费、StateManager 单 Agent 串行更新
- 当前未具备：跨 Agent 共享状态、依赖唤醒协议、冲突解决策略

---

## 10. 数据结构定义

以下为当前实现的近似结构，省略了部分 helper / 校验方法。
需要注意：当前系统使用 `dependencies + input_bindings` 表达数据流，
还没有独立的 `Edge` 类型。

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
    TOOL_ERROR       = "TOOL_ERROR"
    SYSTEM_FALLBACK  = "SYSTEM_FALLBACK"
    BUDGET_EXCEEDED  = "BUDGET_EXCEEDED"
    # Normal
    TOOL_RESULT      = "TOOL_RESULT"
    TOOL_FAILED      = "TOOL_FAILED"
    PLAN_UPDATE      = "PLAN_UPDATE"
    OBSERVATION      = "OBSERVATION"
    STATE_UPDATE     = "STATE_UPDATE"

class EventPriority(Enum):
    HIGH = "high"
    NORMAL = "normal"

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
    max_steps: int = 20
    max_llm_calls: int = 10
    wall_clock_timeout: float = 300.0
    allowed_tools: Optional[List[str]] = None

@dataclass
class BudgetUsage:
    replan_count: int = 0
    step_count: int = 0
    llm_call_count: int = 0
    start_time: float = 0.0

@dataclass
class Event:
    event_id: str
    agent_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: float
    priority: EventPriority
    is_plan_snapshot: bool = False

@dataclass
class Agent:
    agent_id: str
    plan: Plan
    budget: ExecutionBudget
    budget_usage: BudgetUsage
    state_machine: "StateMachine"
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0

@dataclass
class Snapshot:
    snapshot_id: str
    agent_id: str
    timestamp: float
    agent_state: str
    retry_mode: Optional[str]
    plan_id: str
    plan: Dict[str, Any]
    budget: Dict[str, Any]
    current_index: int
    completed_ids: List[str]
    budget_usage: Dict[str, Any]
    history: List[Dict[str, Any]]
    state_history: List[Dict[str, Any]]
    history_len: int
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

状态：`已完成`

| Day | 目标 | 当前结果 |
|-----|------|---------|
| 1 | 核心数据结构 | `Step / Plan / Event / Agent / Budget` 已落地 |
| 2 | 状态机实现 | 严格 `StateMachine` 已落地，支持 `RetryMode` |
| 3 | EventQueue | `PriorityEventQueue + RawEventBus + Dispatcher` 已落地 |

### Phase 2：执行层（Day 4-7）

状态：`已完成`

| Day | 目标 | 当前结果 |
|-----|------|---------|
| 4 | ToolExecutor | 异步执行、超时检测、事件发布已落地 |
| 5 | StepRunner + LLM Interface | `MockLLM + DeepSeekLLM + llm_factory` 已落地 |
| 6 | DependencyValidator + Scheduler | DAG 校验、binding 解析、Scheduler 主循环已落地 |
| 7 | 单 Agent 集成 | 单 Agent 主链路、fallback、replan、白名单路径已覆盖 |

### Phase 3：可观测 + 可恢复（Day 8-11）

状态：`已完成`

| Day | 目标 | 当前结果 |
|-----|------|---------|
| 8 | StateManager + Event Log | JSONL Event Log、metrics、history 已落地 |
| 9 | PolicyEngine | Budget 检查、工具白名单、Replan 前检查已落地 |
| 10 | CheckpointManager + Snapshot | Snapshot、`PlanSnapshot` 事件已落地 |
| 11 | Replay（Debug + Recovery） | Debug / Recovery Replay 已落地并有测试覆盖 |

### Phase 4：多 Agent + 收尾（Day 12-15）

状态：`进行中`

| Day | 目标 | 当前结果 |
|-----|------|---------|
| 12 | 多 Agent 调度 | `未完成`，当前仍以单 Agent 闭环为主 |
| 13 | RETRYING 子状态细化 | `已完成`，`fallback_mode / replan_mode` 已落地 |
| 14 | 压力测试 | `未完成`，尚无正式 stress harness |
| 15 | 文档 + 接口整理 | `部分完成`，README 已更新，稳定对外 API 仍待整理 |

---

## 13. 当前状态、限制与后续规划

### 当前实现快照（2026-04）

当前仓库已经形成一个可运行、可回放、可接真实 LLM 的单 Agent Runtime：

- `Plan / Step` 已支持 DAG 校验、`dependencies`、`input_bindings`、`output_schema`
- `StateMachine` 已严格落地，包含 `READY / RUNNING / WAITING / RETRYING / DONE / ERROR`
- `RetryMode` 已落地，支持 `fallback_mode / replan_mode`
- `ToolExecutor`、`Scheduler`、`PolicyEngine`、`StateManager` 已闭环
- `StepRunner` 已支持局部 Replan，且真实接入了 `DeepSeekLLM`
- `CheckpointManager + DebugReplay + RecoveryReplay` 已落地
- 成功 Replan 会把 `provider / model / raw_response / normalized_result` 写入 `PlanSnapshot`
- 当前回归结果：`pytest -q` → `173 passed, 1 skipped`

### 当前版本限制（Python 原型）

| 限制 | 影响 | 规划解决方式 |
|------|------|-------------|
| 还没有显式 `Edge` 类型 | 数据流仍由 `dependencies + input_bindings` 隐式表达 | 引入 `Edge + data_mapping`，把流程图升级为显式数据流图 |
| Schema Contract 还不完整 | 目前只有 `output_schema`，没有正式 `input_schema` / tool contract registry | 补齐 `input_schema`、fallback schema 兼容校验、工具能力注册 |
| Debug Replay 是确定性摘要，不是完整二次执行 | 能审计状态和 LLM 输出，但不能 1:1 重放工具副作用 | 持久化 resolved params、action metadata、tool outcome envelope |
| 多 Agent 依赖未实现 | 还不能做 manager-workers / shared state / conflict resolution | 完成单 Agent Runtime 后再做跨 Agent 协议 |
| EventQueue 串行消费 | 高并发吞吐受限 | Agent 粒度并发 / queue sharding |
| Event Log 存 JSON 文件 | 大规模场景性能差 | 替换为 SQLite 或 protobuf |
| 对外 API 仍偏 demo 风格 | 真实集成时需要自己拼组件 | 整理稳定的 runtime config / factory / submit API |
| 无分布式支持 | 单进程限制 | C++ 重构阶段引入分布式 Scheduler |
| 没有 memory / learning | 还没进入 Level 3 后半段和 Level 5 | 后续增加 working memory / episodic memory / trajectory learning |

### Treasure Map 对齐与下一阶段

#### 当前定位

按你给的 Treasure Map 来看，这个项目已经不再是纯 `Level 1: Structured Agent`。

更准确的定位是：

- `Level 2: Agent Runtime` 已经基本成型
- 并且已经踩进了一部分 `Level 3: Self-Healing Agent`

原因很直接：

| 能力 | 当前状态 | 说明 |
|------|----------|------|
| DAG Plan | `已实现` | `Plan` 在构建时做循环依赖检测 |
| Typed Dataflow | `部分实现` | 有 `output_schema + input_bindings`，但还没有独立 `Edge` |
| Strict State Machine | `已实现` | 有明确迁移表和非法迁移异常 |
| Runtime vs LLM 解耦 | `已实现` | LLM 只参与局部 Replan，不掌控调度 |
| Deterministic Replay | `部分实现` | `PlanSnapshot + DebugReplay` 已具备，但还不是字节级完整复演 |
| Local Replan | `已实现` | 失败后可走 fallback / replan |
| Failure Attribution | `基础版` | 已区分 `TIMEOUT / TOOL_FAILED / TOOL_ERROR`，但还没有统一归因体系 |
| Dynamic Tool Selection | `基础版` | LLM 可在 Replan 中建议新 fallback 工具，但还没有 tool capability 层 |

结论：当前最重要的不是立刻冲 Multi-Agent，而是把 `Level 2` 的最后那层“系统感”补齐。

#### 下一阶段：补齐 Level 2（最优先）

下一阶段建议只做这 4 件事，不要分心：

1. **显式 `Edge` 模型**
   把 `dependencies + input_bindings` 编译成真正的数据流边，支持 `from_step / to_step / data_mapping`。

2. **完整 Schema System**
   给 `Step` 增加 `input_schema`，再补一个 tool contract registry。
   这样 fallback / replan 后才能做真正的 schema compatibility check。

3. **更强的 Deterministic Replay**
   除了 `plan snapshot`，还要持久化：
   `resolved params`、`action metadata`、`tool outcome envelope`、`policy decision reason`。

4. **稳定的 Runtime API**
   把“demo 脚本拼装组件”收敛成明确入口：
   `RuntimeConfig / create_runtime() / submit_task()`。

这一阶段做完，你的项目才会彻底从“好看的原型”跨到“工业味很强的 Agent Runtime”。

#### 再下一阶段：冲击 Level 3

当 Level 2 补齐后，再进入下面这些能力：

1. **Failure Attribution**
   给失败建立统一分类：
   `tool_error / timeout / policy_violation / schema_mismatch / llm_invalid_output / bad_plan`

2. **约束下的 Dynamic Tool Selection**
   不是让 LLM 随便选工具，而是基于 whitelist / capability / schema contract 约束选择。

3. **Memory**
   先做 `working_memory`，再做 `episodic_memory`。

4. **Agent Runtime Design Doc**
   这是杀招。
   当 runtime 真补齐后，写一份设计文档会显著放大项目价值。

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
# 安装最小依赖
pip install pytest openai

# 运行单 Agent 示例（默认使用 MockLLM）
python examples/single_agent_demo.py

# 运行真实 DeepSeek Replan 示例
export MYAGENT_LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_api_key
python examples/deepseek_replan_demo.py

# 运行测试套件
pytest -q
```

---

## 目录结构

```
agent_runtime/
├── core/
│   ├── agent.py              # Agent 数据结构
│   ├── state_machine.py      # 状态机
│   ├── plan.py               # Plan / Step / FallbackOption
│   ├── budget.py             # ExecutionBudget / BudgetUsage
│   └── test_core.py          # 核心数据模型 + 状态机测试
├── scheduler/
│   ├── scheduler.py          # Scheduler 主循环
│   └── policy_engine.py      # PolicyEngine
├── execution/
│   ├── step_runner.py        # StepRunner
│   ├── tool_executor.py      # ToolExecutor（异步）
│   ├── llm_interface.py      # LLM 抽象接口 + Mock 实现
│   ├── llm_deepseek.py       # DeepSeek 实现
│   └── llm_factory.py        # 运行时 Provider 选择
├── events/
│   ├── event_queue.py        # PriorityEventQueue
│   ├── raw_event_bus.py      # RawEventBus + Dispatcher
│   ├── event_types.py        # EventType 枚举
│   └── test_event_system.py  # 事件系统测试
├── state/
│   ├── state_manager.py      # StateManager
│   └── dependency_validator.py # DependencyValidator
├── checkpoint/
│   ├── checkpoint_manager.py # CheckpointManager
│   └── replay.py             # Debug / Recovery Replay
├── examples/
│   ├── single_agent_demo.py  # 单 Agent 快速演示
│   └── deepseek_replan_demo.py # 真实 DeepSeek Replan 演示
└── tests/
    ├── test_tool_executor.py
    ├── test_step_runner.py
    ├── test_day6.py          # Scheduler / StateManager / DependencyValidator / PolicyEngine
    ├── test_checkpoint.py
    └── test_llm_deepseek.py
```

---

*本文档随开发进度持续更新。当前版本对应：Level 2 Agent Runtime（单 Agent 主链路完成） + Level 3 起步。*
