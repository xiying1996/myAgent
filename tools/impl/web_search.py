"""
tools/impl/web_search.py — Web Search Tool

基于 HTTP API 的搜索工具（支持数眼智能 / SerpAPI / DuckDuckGo）。

⚠️ 安全注意：
  - API Key 通过环境变量注入，不硬编码
  - 不支持任意 JS 执行（安全）
  - 输出截断，防止过大 payload
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import requests

from tools.tool import Tool, RetryPolicy, ToolMetadata
from tools.result import ToolErrorType
from tools.schema import Schema

logger = logging.getLogger(__name__)

# ── 默认配置 ─────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 10.0
MAX_RESULTS = 10
OUTPUT_TRUNCATE_CHARS = 8000

# 数眼智能 API 配置
SHUYAN_API_BASE = "https://api.shuyanai.com"
SHUYAN_API_KEY_ENV = "SHUYAN_API_KEY"


# ── Tool 定义 ────────────────────────────────────────────────────────────────


class WebSearchTool(Tool):
    """
    Web Search 工具。

    支持三种后端（按优先级）：
      1. 数眼智能 API（需 SHUYAN_API_KEY 环境变量，国内搜索首选）
      2. SerpAPI（需 SERPAPI_API_KEY 环境变量）
      3. DuckDuckGo HTML（无需 key，免费备用）

    output_schema: {results: list, answer: str, source: str}
    """

    def __init__(
        self,
        shuyan_api_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # 优先级：数眼 > SerpAPI > DuckDuckGo
        self._shuyan_api_key = shuyan_api_key or os.environ.get(SHUYAN_API_KEY_ENV)
        self._serpapi_key = serpapi_key or os.environ.get("SERPAPI_API_KEY")

        # 确定使用哪个后端
        if self._shuyan_api_key:
            self._backend = "shuyan"
        elif self._serpapi_key:
            self._backend = "serpapi"
        else:
            self._backend = "duckduckgo"

        logger.info("WebSearchTool 初始化，使用后端: %s", self._backend)

        super().__init__(
            name="web_search",
            description="Search the web for information. Returns top results with snippets.",
            input_schema=Schema.from_dict({
                "q": "str",
                "max_results": {"type": "int", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "results": "list",
                "answer": "str",
                "source": "str",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                permissions=["network"],
                tags=["search", "web", "information"],
                deterministic=False,
            ),
            retry_policy=RetryPolicy(
                max_attempts=2,
                base_delay_ms=1000,
                exponential_base=2.0,
                retriable_errors=[ToolErrorType.TIMEOUT, ToolErrorType.NETWORK],
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        q = input_params.get("q", "")
        max_results = input_params.get("max_results", MAX_RESULTS)
        max_results = min(max_results, 20)  # 上限

        if self._backend == "shuyan":
            return self._search_shuyan(q, max_results)
        elif self._backend == "serpapi":
            return self._search_serpapi(q, max_results)
        else:
            return self._search_ddg(q, max_results)

    def _search_shuyan(self, q: str, max_results: int) -> dict:
        """
        使用数眼智能 API 进行搜索。

        API: POST https://api.shuyanai.com/v1/search
        参数: {"q": "搜索词"}
        返回: {"code": 0, "data": {"webPages": [...]}, "message": "success"}
        """
        try:
            resp = requests.post(
                f"{SHUYAN_API_BASE}/v1/search",
                json={"q": q},
                headers={
                    "Authorization": f"Bearer {self._shuyan_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()

            # 检查 API 返回状态
            if data.get("code") != 0:
                error_msg = data.get("message", "未知错误")
                raise RuntimeError(f"数眼 API 错误: {error_msg}")

            results = []
            web_pages = data.get("data", {}).get("webPages", [])
            for item in web_pages[:max_results]:
                snippet = (item.get("snippet", "") or "")[:500]
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": snippet,
                    "score": item.get("score", 0),
                    "date": item.get("datePublished", ""),
                })

            return {
                "results": results,
                "answer": "",
                "source": "shuyan",
            }
        except requests.exceptions.RequestException as e:
            logger.warning("WebSearchTool[Shuyan] 网络错误: %s，尝试 DuckDuckGo", e)
            return self._search_ddg(q, max_results)
        except Exception as e:
            logger.warning("WebSearchTool[Shuyan] 失败: %s，尝试 DuckDuckGo", e)
            return self._search_ddg(q, max_results)

    def _search_serpapi(self, q: str, max_results: int) -> dict:
        """使用 SerpAPI 进行搜索。"""
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={
                    "q": q,
                    "api_key": self._serpapi_key,
                    "num": max_results,
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                snippet = (item.get("snippet", "") or "")[:500]
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": snippet,
                })

            answer = data.get("answer_box", {}).get("answer", "")
            return {
                "results": results,
                "answer": str(answer)[:OUTPUT_TRUNCATE_CHARS],
                "source": "serpapi",
            }
        except Exception as e:
            logger.warning("WebSearchTool[SerpAPI] 失败: %s，尝试 DuckDuckGo", e)
            return self._search_ddg(q, max_results)

    def _search_ddg(self, q: str, max_results: int) -> dict:
        """DuckDuckGo HTML 搜索（备用，无需 API Key）。"""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": q},
                headers=headers,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()

            # 简单解析（不依赖 BeautifulSoup）
            html = resp.text
            results = self._parse_ddg_html(html, max_results)
            return {
                "results": results,
                "answer": "",
                "source": "duckduckgo",
            }
        except Exception as e:
            logger.warning("WebSearchTool[DuckDuckGo] 失败: %s", e)
            raise

    def _parse_ddg_html(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """从 DuckDuckGo HTML 中提取搜索结果（简单正则）。"""
        import re
        results = []
        # 匹配 <a class="result__a" href="...">Title</a>
        links = re.findall(r'<a class="result__a" href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)
        for url, title in links[:max_results]:
            title = re.sub(r'<[^>]+>', "", title).strip()
            results.append({
                "title": title[:200],
                "url": url,
                "snippet": "",
            })
        return results


# ── HTTP Fetch Tool ──────────────────────────────────────────────────────────


class HttpFetchTool(Tool):
    """
    HTTP GET 工具。根据 URL 获取页面内容。

    ⚠️ 仅用于获取页面，不执行 JS。
    """

    def __init__(self, timeout_s: float = 15.0) -> None:
        super().__init__(
            name="http_fetch",
            description="Fetch the content of a URL via HTTP GET",
            input_schema=Schema.from_dict({
                "url": "str",
                "max_chars": {"type": "int", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "content": "str",
                "status_code": "int",
                "url": "str",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                permissions=["network"],
                tags=["web", "fetch", "http"],
                deterministic=False,
            ),
            retry_policy=RetryPolicy(
                max_attempts=2,
                base_delay_ms=500,
                retriable_errors=[ToolErrorType.TIMEOUT, ToolErrorType.NETWORK],
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        url = input_params["url"]
        max_chars = input_params.get("max_chars", OUTPUT_TRUNCATE_CHARS)

        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()

        content = resp.text[:max_chars]
        return {
            "content": content,
            "status_code": resp.status_code,
            "url": resp.url,
        }


# ── 便捷访问 ────────────────────────────────────────────────────────────────


WEB_SEARCH_TOOL = WebSearchTool()
HTTP_FETCH_TOOL = HttpFetchTool()
