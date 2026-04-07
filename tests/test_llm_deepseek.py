"""
test_llm_deepseek.py — DeepSeekLLM 测试套件

覆盖范围:
  DeepSeekLLM.call_count       — 计数递增
  DeepSeekLLM._build_prompt     — Prompt 包含所有必要字段
  DeepSeekLLM._extract_json     — plain JSON / markdown 代码块 / 混排
  DeepSeekLLM._parse_response   — 正常 / 解析失败抛 LLMCallError
  DeepSeekLLM._call_with_retry  — 超时重试 / RateLimit 退避

运行:
  pytest tests/test_llm_deepseek.py -v

注意:
  真实 API 调用需要 DEEPSEEK_API_KEY 环境变量。
  无 key 时跳过真实调用测试。
"""

import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from execution.llm_interface import (
    FallbackSuggestion,
    LLMCallError,
    ReplanContext,
    ReplanResult,
    StepSnapshot,
)
from execution.llm_deepseek import DeepSeekLLM
from execution.llm_factory import create_llm_from_env


# ===========================================================================
# Helper
# ===========================================================================

def _make_context(
    failed_tool: str = "web_search",
    failed_step_id: str = "s0",
    failure_reason: str = "timeout",
    failure_history: list = None,
    completed_steps: list = None,
    pending_steps: list = None,
    budget: dict = None,
) -> ReplanContext:
    return ReplanContext(
        agent_id="a_test",
        failed_step=StepSnapshot(
            step_id=failed_step_id,
            tool_name=failed_tool,
            params={"q": "test query"},
            output_schema={"url": "str", "title": "str"},
            input_bindings={},
            dependencies=[],
            fallback_tools=["bing"],
        ),
        failure_reason=failure_reason,
        failure_history=failure_history or [],
        completed_steps=completed_steps or [],
        pending_steps=pending_steps or [],
        budget_remaining=budget or {"replans_left": 3, "llm_calls_left": 10},
    )


# ===========================================================================
# DeepSeekLLM 初始化
# ===========================================================================

class TestInit:
    def test_api_key_required(self):
        with pytest.raises(ValueError, match="api_key 不能为空"):
            DeepSeekLLM(api_key="")

    def test_default_values(self):
        llm = DeepSeekLLM(api_key="test-key-123")
        assert llm._model == "deepseek-chat"
        assert llm._base_url == "https://api.deepseek.com"
        assert llm._max_retries == 3
        assert llm._timeout_s == 60.0

    def test_custom_values(self):
        llm = DeepSeekLLM(
            api_key="test-key",
            model="deepseek-coder",
            base_url="https://custom.api/v1",
            max_retries=5,
            timeout_s=30.0,
        )
        assert llm._model == "deepseek-coder"
        assert llm._base_url == "https://custom.api/v1"
        assert llm._max_retries == 5


# ===========================================================================
# call_count
# ===========================================================================

class TestCallCount:
    def test_call_count_starts_at_zero(self):
        llm = DeepSeekLLM(api_key="test-key")
        assert llm.call_count == 0


# ===========================================================================
# Prompt 构建
# ===========================================================================

class TestBuildPrompt:
    def test_prompt_contains_failed_step(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(failed_tool="fetch", failed_step_id="s1")
        prompt = llm._build_prompt(ctx)
        assert "fetch" in prompt
        assert "s1" in prompt
        assert "timeout" in prompt

    def test_prompt_contains_failure_history(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(
            failure_history=[
                {"tool": "web_search", "reason": "timeout"},
                {"tool": "bing", "reason": "rate limit"},
            ]
        )
        prompt = llm._build_prompt(ctx)
        assert "web_search" in prompt
        assert "bing" in prompt
        assert "timeout" in prompt

    def test_prompt_contains_completed_steps(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(completed_steps=["s0", "s1"])
        prompt = llm._build_prompt(ctx)
        assert "s0" in prompt
        assert "s1" in prompt

    def test_prompt_contains_pending_steps(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(
            pending_steps=[
                StepSnapshot(
                    step_id="s2",
                    tool_name="summarize",
                    params={"text": ""},
                    output_schema={},
                    input_bindings={},
                    dependencies=["s1"],
                    fallback_tools=[],
                )
            ]
        )
        prompt = llm._build_prompt(ctx)
        assert "s2" in prompt
        assert "summarize" in prompt

    def test_prompt_contains_budget(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(budget={"replans_left": 2, "llm_calls_left": 5})
        prompt = llm._build_prompt(ctx)
        assert "replans_left" in prompt
        assert "2" in prompt


# ===========================================================================
# JSON 提取
# ===========================================================================

class TestExtractJson:
    def test_plain_json(self):
        llm = DeepSeekLLM(api_key="test-key")
        raw = '{"new_fallbacks": [], "reasoning": "test"}'
        assert llm._extract_json(raw) == raw

    def test_markdown_json_block(self):
        llm = DeepSeekLLM(api_key="test-key")
        raw = '```json\n{"new_fallbacks": [], "reasoning": "test"}\n```'
        result = llm._extract_json(raw)
        assert result == '{"new_fallbacks": [], "reasoning": "test"}'

    def test_markdown_code_block_no_lang(self):
        llm = DeepSeekLLM(api_key="test-key")
        raw = '```\n{"new_fallbacks": [], "reasoning": "test"}\n```'
        result = llm._extract_json(raw)
        assert result == '{"new_fallbacks": [], "reasoning": "test"}'

    def test_json_with_surrounding_text(self):
        llm = DeepSeekLLM(api_key="test-key")
        raw = 'Here is the result: {"new_fallbacks": [{"tool": "x", "params": {}}], "reasoning": "ok"}\nGood luck!'
        result = llm._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["new_fallbacks"][0]["tool"] == "x"

    def test_invalid_json_returns_raw_for_parsing(self):
        """_extract_json 对完全无效的 JSON 不抛异常，而是原样返回由 parse 来报错"""
        llm = DeepSeekLLM(api_key="test-key")
        raw = "this is not json at all"
        result = llm._extract_json(raw)
        # 返回原始字符串，解析由外层 json.loads 负责
        assert result == raw


# ===========================================================================
# Response 解析
# ===========================================================================

class TestParseResponse:
    def test_valid_full_response(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()
        raw = json.dumps({
            "new_fallbacks": [
                {"tool": "bing_v2", "params": {"q": "test"}}
            ],
            "step_param_updates": {"s2": {"timeout": 30}},
            "reasoning": "use bing instead",
            "give_up": False,
        })
        result = llm._parse_response(raw, ctx)
        assert len(result.new_fallbacks) == 1
        assert result.new_fallbacks[0].tool == "bing_v2"
        assert result.step_param_updates == {"s2": {"timeout": 30}}
        assert result.reasoning == "use bing instead"
        assert result.give_up is False

    def test_give_up_true(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()
        raw = '{"new_fallbacks": [], "reasoning": "cannot recover", "give_up": true}'
        result = llm._parse_response(raw, ctx)
        assert result.give_up is True
        assert result.is_empty()

    def test_invalid_json_raises_llmcallerror(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()
        with pytest.raises(LLMCallError, match="解析 LLM 响应失败"):
            llm._parse_response("not json at all", ctx)

    def test_missing_fields_uses_defaults(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()
        raw = '{"new_fallbacks": [{"tool": "x", "params": {}}]}'
        result = llm._parse_response(raw, ctx)
        assert len(result.new_fallbacks) == 1
        assert result.step_param_updates == {}
        assert result.reasoning == ""
        assert result.give_up is False

    def test_empty_fallbacks_give_up(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()
        raw = '{"new_fallbacks": [], "step_param_updates": {}, "reasoning": "no more options"}'
        result = llm._parse_response(raw, ctx)
        assert result.is_empty()


# ===========================================================================
# Mock patched（无真实 API 调用）
# ===========================================================================

class TestDeepSeekLLMMocked:
    """使用 mock 直接 patch client.chat.completions.create 方法。"""

    def test_propose_replan_success(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context(failed_tool="search", failure_reason="timeout")

        mock_response = json.dumps({
            "new_fallbacks": [
                {"tool": "search_v2", "params": {"q": "test query"}},
                {"tool": "generic_search", "params": {}},
            ],
            "step_param_updates": {},
            "reasoning": "try v2 version",
            "give_up": False,
        })

        mock_result = mock.MagicMock()
        mock_result.choices = [mock.MagicMock()]
        mock_result.choices[0].message.content = mock_response

        with mock.patch.object(
            llm._client.chat.completions, "create", return_value=mock_result
        ):
            result = llm.propose_replan(ctx)

        assert len(result.new_fallbacks) == 2
        assert result.new_fallbacks[0].tool == "search_v2"
        assert llm.call_count == 1

    def test_propose_replan_timeout_raises_llmcallerror(self):
        from openai import APITimeoutError

        llm = DeepSeekLLM(api_key="test-key", max_retries=2)
        ctx = _make_context()

        with mock.patch.object(
            llm._client.chat.completions, "create"
        ) as mock_create:
            mock_create.side_effect = APITimeoutError(request=mock.MagicMock())

            with pytest.raises(LLMCallError) as exc_info:
                llm.propose_replan(ctx)
            assert "已重试" in str(exc_info.value) or "超时" in str(exc_info.value)

    def test_propose_replan_parse_error_raises_llmcallerror(self):
        llm = DeepSeekLLM(api_key="test-key")
        ctx = _make_context()

        mock_result = mock.MagicMock()
        mock_result.choices = [mock.MagicMock()]
        mock_result.choices[0].message.content = "not json"

        with mock.patch.object(
            llm._client.chat.completions, "create", return_value=mock_result
        ):
            with pytest.raises(LLMCallError, match="解析 LLM 响应失败"):
                llm.propose_replan(ctx)


class TestLLMFactory:
    def test_factory_defaults_to_mock_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            llm = create_llm_from_env(mock_delay_s=0)
        assert llm.provider_name == "mock"

    def test_factory_uses_deepseek_when_key_present(self):
        env = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "deepseek-chat",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            llm = create_llm_from_env(mock_delay_s=0)
        assert llm.provider_name == "deepseek"
        assert llm.model_name == "deepseek-chat"

    def test_factory_rejects_deepseek_without_key(self):
        env = {"MYAGENT_LLM_PROVIDER": "deepseek"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                create_llm_from_env(mock_delay_s=0)


# ===========================================================================
# 真实 API 调用测试（需要 DEEPSEEK_API_KEY 环境变量）
# ===========================================================================

class TestRealAPI:
    """仅在设置 DEEPSEEK_API_KEY 环境变量时运行真实 API 测试。"""

    def test_propose_replan_real_api(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY 环境变量未设置")

        llm = DeepSeekLLM(api_key=api_key)
        ctx = _make_context(
            failed_tool="web_search",
            failed_step_id="s0",
            failure_reason="timeout",
            failure_history=[{"tool": "web_search", "reason": "timeout"}],
            completed_steps=["s_pre"],
            pending_steps=[
                StepSnapshot(
                    step_id="s1",
                    tool_name="fetch",
                    params={"url": ""},
                    output_schema={"html": "str"},
                    input_bindings={},
                    dependencies=["s0"],
                    fallback_tools=[],
                )
            ],
            budget={"replans_left": 3, "llm_calls_left": 10},
        )

        result = llm.propose_replan(ctx)

        assert llm.call_count == 1
        # 验证返回结构
        assert isinstance(result, ReplanResult)
        assert isinstance(result.new_fallbacks, list)
        assert isinstance(result.reasoning, str)
        # give_up 可能是 True 或 False，但不应抛异常
        print(f"\n[Real API] result: fallbacks={[fb.tool for fb in result.new_fallbacks]}, "
              f"param_updates={result.step_param_updates}, give_up={result.give_up}, "
              f"reasoning={result.reasoning[:100]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
