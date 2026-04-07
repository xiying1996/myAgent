"""
llm_deepseek.py — DeepSeek LLM Implementation

接入 DeepSeek Chat API（OpenAI-Compatible），实现 LLMInterface.propose_replan()。

依赖:
    pip install openai

使用:
    from execution.llm_deepseek import DeepSeekLLM
    llm = DeepSeekLLM(api_key="sk-...", model="deepseek-chat")
    result = llm.propose_replan(context)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, RateLimitError, APITimeoutError

from execution.llm_interface import (
    LLMCallError,
    LLMInterface,
    ReplanContext,
    ReplanResult,
    FallbackSuggestion,
)

logger = logging.getLogger(__name__)

# DeepSeek API base URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Default model
DEFAULT_MODEL = "deepseek-chat"

# Max retries for rate limit / timeout
DEFAULT_MAX_RETRIES = 3

# Base delay for exponential backoff (seconds)
BASE_BACKOFF_DELAY = 2.0


class DeepSeekLLM(LLMInterface):
    """
    DeepSeek Chat API 实现（OpenAI-Compatible）。

    通过更换 base_url + api_key，也可适配其他 OpenAI-Compatible 接口。

    行为:
      propose_replan() 接收 ReplanContext，构建 Prompt，调用 DeepSeek API，
      解析 JSON 响应为 ReplanResult。技术失败（超时/RateLimit）抛 LLMCallError，
      由 StepRunner 上层的 Scheduler 决定是否重试。
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key 不能为空")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._call_count: int = 0

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_s,
        )

    # ── LLMInterface ────────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self._model

    def propose_replan(self, context: ReplanContext) -> ReplanResult:
        """
        给定 ReplanContext，调用 DeepSeek API 获取重规划建议。

        异常:
          LLMCallError — 技术失败（超时、RateLimit、解析错误等）
        """
        self._call_count += 1

        prompt = self._build_prompt(context)

        raw_response = self._call_with_retry(prompt)

        result = self._parse_response(raw_response, context)

        logger.info(
            "[DeepSeekLLM] Agent[%s] Step[%s] replan result: "
            "fallbacks=%s, param_updates=%s, give_up=%s",
            context.agent_id,
            context.failed_step.step_id,
            [fb.tool for fb in result.new_fallbacks],
            list(result.step_param_updates.keys()),
            result.give_up,
        )

        return result

    # ── Prompt 构建 ─────────────────────────────────────────────────────────

    def _build_prompt(self, ctx: ReplanContext) -> str:
        """将 ReplanContext 转换为发送给 LLM 的 Prompt。"""

        failed = ctx.failed_step
        pending = ctx.pending_steps

        # 格式化失败历史
        failure_history_lines = ""
        if ctx.failure_history:
            for record in ctx.failure_history[-5:]:  # 最多显示最近 5 条
                failure_history_lines += f'  - tool="{record.get("tool", "?")}", reason="{record.get("reason", "?")}"\n'
        else:
            failure_history_lines = "  (无历史失败记录)\n"

        # 格式化已完成步骤
        completed_lines = ""
        if ctx.completed_steps:
            for sid in ctx.completed_steps:
                completed_lines += f'  - {sid}\n'
        else:
            completed_lines = "  (无)\n"

        # 格式化待处理步骤
        pending_lines = ""
        for ps in pending:
            pending_lines += (
                f'  - step_id="{ps.step_id}", tool="{ps.tool_name}", '
                f'params={ps.params}, dependencies={ps.dependencies}\n'
            )

        # 格式化 Budget
        budget = ctx.budget_remaining
        budget_lines = ", ".join(f"{k}={v}" for k, v in budget.items())

        prompt = f"""You are an expert AI task planning assistant. A task execution step has failed and you need to suggest a recovery plan.

## Current Failed Step
- step_id: {failed.step_id}
- tool: {failed.tool_name}
- params: {json.dumps(failed.params, ensure_ascii=False)}
- output_schema: {json.dumps(failed.output_schema, ensure_ascii=False)}
- input_bindings: {json.dumps(failed.input_bindings, ensure_ascii=False)}
- existing_fallback_tools: {failed.fallback_tools}

## Failure Reason
{ctx.failure_reason}

## Failure History (recent)
{failure_history_lines}## Completed Steps
{completed_lines}## Pending Steps (not yet executed)
{pending_lines}## Budget Remaining
{budget_lines}

## Your Task
Based on the above context, suggest a recovery plan. You can:
1. Suggest new fallback tools for the failed step (via new_fallbacks)
2. Adjust parameters for pending steps (via step_param_updates)

## Constraints
- You CANNOT modify completed steps
- You CANNOT change step order
- You CANNOT change step_id or tool_name of any step
- step_param_updates MUST only reference downstream pending steps
- step_param_updates MUST NOT include the current failed step; changes for the failed step belong in new_fallbacks
- If step_param_updates is unnecessary, return an empty object
- If you believe the task cannot be completed, set give_up=true

## Output Format
Return a JSON object with the following fields:
{{
  "new_fallbacks": [
    {{"tool": "tool_name", "params": {{"param_name": "param_value"}}}}
  ],
  "step_param_updates": {{
    "step_id": {{"param_name": "new_value"}}
  }},
  "reasoning": "Explain your reasoning here",
  "give_up": false
}}

Return ONLY the JSON object, no additional text."""
        return prompt

    # ── API 调用（带重试） ─────────────────────────────────────────────────

    def _call_with_retry(self, prompt: str) -> str:
        """调用 DeepSeek API，带指数退避重试。"""
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise LLMCallError("LLM 返回空响应")
                return content

            except APITimeoutError as e:
                last_error = e
                logger.warning(
                    "[DeepSeekLLM] API 超时（attempt %d/%d）",
                    attempt + 1, self._max_retries,
                )

            except RateLimitError as e:
                last_error = e
                delay = BASE_BACKOFF_DELAY * (2 ** attempt)
                logger.warning(
                    "[DeepSeekLLM] Rate limit（attempt %d/%d），"
                    "%.1fs 后重试",
                    attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)

            except Exception as e:
                last_error = e
                logger.warning(
                    "[DeepSeekLLM] API 调用异常（attempt %d/%d）: %s",
                    attempt + 1, self._max_retries, e,
                )
                # 非重试异常，直接抛出
                if not isinstance(e, (APITimeoutError, RateLimitError)):
                    break

        # 所有重试均失败
        raise LLMCallError(
            f"DeepSeek API 调用失败（已重试 {self._max_retries} 次）: {last_error}",
            original=last_error,
        )

    # ── Response 解析 ──────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        ctx: ReplanContext,
    ) -> ReplanResult:
        """
        将 LLM 原始输出解析为 ReplanResult。
        解析失败视为技术失败，抛出 LLMCallError。
        """
        try:
            # 尝试从 markdown 代码块中提取 JSON
            json_str = self._extract_json(raw)
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise TypeError(f"期望 JSON object，实际为 {type(data).__name__}")

            new_fallbacks: List[FallbackSuggestion] = []
            for fb in data.get("new_fallbacks") or []:
                if not fb.get("tool"):
                    continue
                new_fallbacks.append(FallbackSuggestion(
                    tool=fb["tool"],
                    params=fb.get("params") or {},
                ))

            step_param_updates: Dict[str, Dict[str, Any]] = {}
            for sid, params in data.get("step_param_updates", {}).items():
                if isinstance(params, dict):
                    step_param_updates[sid] = params

            return ReplanResult(
                new_fallbacks=new_fallbacks,
                step_param_updates=step_param_updates,
                reasoning=data.get("reasoning", ""),
                give_up=bool(data.get("give_up", False)),
                raw_response=raw,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(
                "[DeepSeekLLM] 解析 LLM 响应失败。原始响应: %s",
                raw[:500],
            )
            raise LLMCallError(
                f"解析 LLM 响应失败: {e}。原始响应片段: {raw[:200]}",
                original=e,
            )

    def _extract_json(self, raw: str) -> str:
        """
        从 LLM 输出中提取 JSON 字符串。
        支持 plain JSON 和 ```json ... ``` 格式。
        """
        raw = raw.strip()

        # 尝试直接解析
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return match.group(1)

        # 尝试找到第一个 { 到最后一个 } 的内容
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 无法提取，返回原始内容让 json.loads 报错
        return raw
