"""
deepseek_replan_demo.py — 使用真实 DeepSeek 触发一次 Replan

运行:
  export MYAGENT_LLM_PROVIDER=deepseek
  export DEEPSEEK_API_KEY=...
  python examples/deepseek_replan_demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.agent import Agent
from core.plan import FallbackOption, Plan, Step
from core.state_machine import AgentState, RetryMode
from execution.llm_factory import create_llm_from_env
from execution.step_runner import FailureRecord, StepRunner


def main() -> None:
    llm = create_llm_from_env(mock_delay_s=0)
    if llm.provider_name != "deepseek":
        raise SystemExit(
            "该示例要求真实 DeepSeek。请设置 MYAGENT_LLM_PROVIDER=deepseek 和 DEEPSEEK_API_KEY。"
        )

    plan = Plan.create([
        Step(
            step_id="step0",
            tool_name="web_search",
            params={"query": "open source agent runtime framework design"},
            fallback_chain=[
                FallbackOption("bing_search", {"query": "open source agent runtime framework design"}),
                FallbackOption("duckduckgo_search", {"query": "open source agent runtime framework design"}),
            ],
            output_schema={"url": "str", "title": "str"},
        ),
    ], max_replans=2)

    agent = Agent.create(plan=plan, agent_id="deepseek_demo_agent")
    agent.transition(AgentState.RUNNING, "start")
    agent.transition(AgentState.WAITING, "submitted")
    agent.transition(
        AgentState.RETRYING,
        "search provider failed repeatedly",
        retry_mode=RetryMode.REPLAN_MODE,
    )

    runner = StepRunner(llm)

    failure_records = [
        FailureRecord("web_search", "HTTP 429 rate limit"),
        FailureRecord("bing_search", "captcha blocked"),
        FailureRecord("duckduckgo_search", "empty search results"),
    ]
    last_failure_reason = "主搜索超时，现有 fallback 也不可用，需要新的恢复方案"

    context = runner.build_context(
        agent=agent,
        last_failure_reason=last_failure_reason,
        failure_records=failure_records,
    )
    result = llm.propose_replan(context)
    normalized = result.to_dict(include_raw_response=False)
    validation = runner.validate_replan(
        result=result,
        completed_step_ids=agent.plan.completed_step_ids,
        current_step_id=agent.current_step().step_id,
        all_step_ids=[step.step_id for step in agent.plan.steps],
    )

    print("=" * 60)
    print("DeepSeek Replan Demo")
    print("=" * 60)
    print(f"provider:   {llm.provider_name}")
    print(f"model:      {llm.model_name}")
    print(f"llm_calls:  {llm.call_count}")
    print(f"agent:      {agent.agent_id}")
    print(f"state:      {agent.state.value}")
    print(f"retry_mode: {agent.retry_mode.value if agent.retry_mode else None}")
    print(f"validation: {'PASS' if validation.ok else 'FAIL'}")
    print()
    print("normalized_replan_result:")
    print(json.dumps(normalized, ensure_ascii=False, indent=2))
    print()
    if validation.reasons:
        print("validation_reasons:")
        print(json.dumps(validation.reasons, ensure_ascii=False, indent=2))
        print()
    raw_response = result.raw_response or ""
    print("raw_response_excerpt:")
    print(raw_response[:1200])
    print()
    print("suggested_fallback_tools:")
    print([fb.tool for fb in result.new_fallbacks])


if __name__ == "__main__":
    main()
