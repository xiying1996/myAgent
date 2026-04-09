"""
tools/impl/echo.py — Echo Tool（调试用）

简单的调试工具，回显输入参数。用于测试 schema matching 和 pipeline。
"""

from tools.tool import Tool, ToolMetadata
from tools.schema import Schema
from tools.result import ToolResult


class EchoTool(Tool):
    """回显工具：输入什么就输出什么。"""

    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echo back the input parameters (debug tool)",
            input_schema=Schema.from_dict({
                "message": "str",
                "optional_extra": {"type": "str", "optional": True},
            }),
            output_schema=Schema.from_dict({
                "echoed": "str",
                "extra": {"type": "str", "optional": True},
            }),
            timeout_s=5.0,
            metadata=ToolMetadata(tags=["debug", "echo"]),
        )

    def _do_invoke(self, input_params: dict) -> dict:
        message = input_params.get("message", "")
        extra = input_params.get("optional_extra")
        return {
            "echoed": message,
            "extra": extra,
        }


# 便捷访问
ECHO_TOOL = EchoTool()
