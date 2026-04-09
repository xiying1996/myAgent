"""
tools/impl/filesystem.py — Filesystem Tool

安全约束：
  - 只能访问 WORKSPACE_ROOT（默认 ~/project）
  - 禁止路径穿越（../）
  - 超时限制
  - 输出截断
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, Optional

from tools.tool import Tool, RetryPolicy, ToolMetadata
from tools.schema import Schema

logger = logging.getLogger(__name__)

# ── 安全配置 ─────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 10.0
MAX_FILE_SIZE = 1_000_000   # 1MB
WORKSPACE_ROOT = os.path.expanduser("~/project")


# ── Tool 定义 ────────────────────────────────────────────────────────────────


def _sanitize_path(base: str, relative: str) -> str:
    """
    安全路径解析：确保相对路径不会穿越 base。

    例: base=/project, relative=../etc/passwd → /project/../etc/passwd (rejected)
    """
    base = os.path.abspath(base)
    if not os.path.commonpath([base, os.path.abspath(os.path.join(base, relative))]).startswith(base):
        raise PermissionError(f"Path traversal attempt: {relative}")
    return os.path.join(base, relative)


class FileReadTool(Tool):
    """安全文件读取工具。"""

    def __init__(
        self,
        workspace_root: str = WORKSPACE_ROOT,
        max_file_size: int = MAX_FILE_SIZE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._workspace = os.path.abspath(workspace_root)
        self._max_size = max_file_size

        super().__init__(
            name="file_read",
            description="Read the content of a file within the workspace",
            input_schema=Schema.from_dict({
                "path": "str",
                "max_chars": {"type": "int", "optional": True},
                "offset": {"type": "int", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "content": "str",
                "path": "str",
                "size": "int",
                "truncated": "bool",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                permissions=["file:read"],
                tags=["filesystem", "file", "read"],
                deterministic=True,
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        rel_path = input_params.get("path", "")
        max_chars = input_params.get("max_chars", 50_000)
        offset = input_params.get("offset", 0)

        # 安全路径解析
        safe_path = _sanitize_path(self._workspace, rel_path)

        if not os.path.isfile(safe_path):
            raise FileNotFoundError(f"File not found: {rel_path}")

        size = os.path.getsize(safe_path)
        if size > self._max_size:
            raise ValueError(f"File too large: {size} bytes (max {self._max_size})")

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            if offset > 0:
                f.seek(offset)
            content = f.read(max_chars)

        truncated = len(content) >= max_chars or size > len(content)
        return {
            "content": content,
            "path": rel_path,
            "size": size,
            "truncated": truncated,
        }


class FileWriteTool(Tool):
    """安全文件写入工具。"""

    def __init__(
        self,
        workspace_root: str = WORKSPACE_ROOT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._workspace = os.path.abspath(workspace_root)

        super().__init__(
            name="file_write",
            description="Write content to a file within the workspace",
            input_schema=Schema.from_dict({
                "path": "str",
                "content": "str",
                "append": {"type": "bool", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "path": "str",
                "bytes_written": "int",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                side_effects=True,
                permissions=["file:write"],
                tags=["filesystem", "file", "write"],
                deterministic=False,
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        rel_path = input_params.get("path", "")
        content = input_params.get("content", "")
        append = input_params.get("append", False)

        safe_path = _sanitize_path(self._workspace, rel_path)

        # 确保父目录存在
        parent = os.path.dirname(safe_path)
        if parent and not os.path.isdir(parent):
            raise FileNotFoundError(f"Parent directory not found: {parent}")

        mode = "a" if append else "w"
        with open(safe_path, mode, encoding="utf-8") as f:
            bytes_written = f.write(content)

        return {
            "path": rel_path,
            "bytes_written": bytes_written,
        }


class FileListTool(Tool):
    """列出目录内容。"""

    def __init__(
        self,
        workspace_root: str = WORKSPACE_ROOT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._workspace = os.path.abspath(workspace_root)

        super().__init__(
            name="file_list",
            description="List directory contents within the workspace",
            input_schema=Schema.from_dict({
                "path": {"type": "str", "optional": True},
                "pattern": {"type": "str", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "entries": "list",
                "path": "str",
            }),
            timeout_s=timeout_s,
            metadata=ToolMetadata(
                permissions=["file:read"],
                tags=["filesystem", "file", "list"],
                deterministic=False,
            ),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        rel_path = input_params.get("path", ".")
        pattern = input_params.get("pattern")

        safe_path = _sanitize_path(self._workspace, rel_path)
        if not os.path.isdir(safe_path):
            raise NotADirectoryError(f"Not a directory: {rel_path}")

        entries = []
        for name in os.listdir(safe_path):
            if pattern:
                import fnmatch
                if not fnmatch.fnmatch(name, pattern):
                    continue
            full = os.path.join(safe_path, name)
            is_dir = os.path.isdir(full)
            entries.append({
                "name": name,
                "type": "dir" if is_dir else "file",
                "size": 0 if is_dir else os.path.getsize(full),
            })

        return {
            "entries": entries,
            "path": rel_path,
        }


# ── 便捷访问 ────────────────────────────────────────────────────────────────


FILE_READ_TOOL = FileReadTool()
FILE_WRITE_TOOL = FileWriteTool()
FILE_LIST_TOOL = FileListTool()
