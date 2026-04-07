"""
llm_factory.py — Runtime LLM Factory

统一从环境变量构建运行时使用的 LLM 实例。

支持：
  - mock（默认）
  - deepseek

环境变量：
  MYAGENT_LLM_PROVIDER  — mock | deepseek
  DEEPSEEK_API_KEY      — DeepSeek API Key
  DEEPSEEK_MODEL        — 可选，默认 deepseek-chat
  DEEPSEEK_BASE_URL     — 可选，默认 https://api.deepseek.com
  DEEPSEEK_MAX_RETRIES  — 可选，默认 3
  DEEPSEEK_TIMEOUT_S    — 可选，默认 60
"""

from __future__ import annotations

import os

from execution.llm_deepseek import DeepSeekLLM
from execution.llm_interface import LLMInterface, MockLLM


def create_llm_from_env(mock_delay_s: float = 0.01) -> LLMInterface:
    """
    根据环境变量创建运行时 LLM。

    优先级：
      1. 显式 MYAGENT_LLM_PROVIDER
      2. 若存在 DEEPSEEK_API_KEY，则默认 deepseek
      3. 否则默认 mock
    """
    provider = os.environ.get("MYAGENT_LLM_PROVIDER")
    if provider is None:
        provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "mock"
    provider = provider.strip().lower()

    if provider == "mock":
        return MockLLM(simulate_delay_s=mock_delay_s)

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("MYAGENT_LLM_PROVIDER=deepseek 时必须设置 DEEPSEEK_API_KEY")

        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        max_retries = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "3"))
        timeout_s = float(os.environ.get("DEEPSEEK_TIMEOUT_S", "60"))
        return DeepSeekLLM(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )

    raise ValueError(f"不支持的 MYAGENT_LLM_PROVIDER: {provider}")
