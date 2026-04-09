"""
tools/impl — 内置工具实现

导出所有内置工具实例，供 ToolRegistry 批量注册。
"""

from tools.impl.echo import EchoTool, ECHO_TOOL
from tools.impl.web_search import WebSearchTool, HttpFetchTool, WEB_SEARCH_TOOL, HTTP_FETCH_TOOL
from tools.impl.bash import BashTool, BASH_TOOL
from tools.impl.filesystem import (
    FileReadTool, FileWriteTool, FileListTool,
    FILE_READ_TOOL, FILE_WRITE_TOOL, FILE_LIST_TOOL,
)

__all__ = [
    "EchoTool",
    "ECHO_TOOL",
    "WebSearchTool",
    "HttpFetchTool",
    "WEB_SEARCH_TOOL",
    "HTTP_FETCH_TOOL",
    "BashTool",
    "BASH_TOOL",
    "FileReadTool",
    "FileWriteTool",
    "FileListTool",
    "FILE_READ_TOOL",
    "FILE_WRITE_TOOL",
    "FILE_LIST_TOOL",
]


def register_all(registry) -> None:
    """将所有内置工具注册到 ToolRegistry。"""
    from tools.registry import ToolRegistry
    for tool in [
        ECHO_TOOL,
        WEB_SEARCH_TOOL,
        HTTP_FETCH_TOOL,
        BASH_TOOL,
        FILE_READ_TOOL,
        FILE_WRITE_TOOL,
        FILE_LIST_TOOL,
    ]:
        registry.register(tool)
