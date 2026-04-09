"""
test_tools.py — Tool System 测试套件

覆盖范围:
  Schema         — Field / is_superset_of / is_compatible_with / coercion
  ToolResult     — success / failure / timeout 工厂方法
  Tool           — invoke / retry / error classification
  ToolRegistry   — register / get / find_compatible / find_by_tag
  Adapter        — apply / can_accept / can_produce
  AdapterRegistry — find_path / find_direct
  BashTool       — security checks

运行:
  pytest tests/test_tools.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tools.schema import Schema, Field
from tools.result import ToolResult, ToolStatus, ToolErrorType
from tools.tool import Tool, ToolMetadata, RetryPolicy
from tools.registry import ToolRegistry
from tools.adapter import Adapter, AdapterRegistry, BUILTIN_ADAPTERS
from tools.impl.echo import EchoTool
from tools.impl.bash import BashTool


# ===========================================================================
# Schema 测试
# ===========================================================================

class TestSchema:
    def test_from_dict_simple(self):
        s = Schema.from_dict({"name": "str", "age": "int"})
        assert "name" in s.fields
        assert s.fields["name"].type == "str"
        assert s.fields["age"].type == "int"

    def test_from_dict_with_optional(self):
        s = Schema.from_dict({"name": "str", "age": {"type": "int", "optional": True}})
        assert s.fields["age"].optional is True
        assert s.fields["name"].optional is False

    def test_is_superset_of_basic(self):
        # tool output: {name: str, url: str}
        output = Schema.from_dict({"name": "str", "url": "str"})
        # step input: {name: str}
        step_input = Schema.from_dict({"name": "str"})
        assert output.is_superset_of(step_input) is True

    def test_is_superset_of_missing_field(self):
        output = Schema.from_dict({"name": "str"})
        step_input = Schema.from_dict({"name": "str", "url": "str"})
        assert output.is_superset_of(step_input) is False

    def test_is_superset_of_type_coercion(self):
        # tool returns int, step requires str → coercion allowed
        output = Schema.from_dict({"count": "int"})
        step_input = Schema.from_dict({"count": "str"})
        assert output.is_superset_of(step_input) is True

    def test_is_superset_of_nested(self):
        s = Schema.from_dict({
            "user": {"type": "dict", "nested": {"name": "str", "age": "int"}}
        })
        other = Schema.from_dict({
            "user": {"type": "dict", "nested": {"name": "str"}}
        })
        assert s.is_superset_of(other) is True

    def test_is_compatible_with_optional_missing(self):
        output = Schema.from_dict({"name": "str"})
        step_input = Schema.from_dict({"name": "str", "extra": {"type": "str", "optional": True}})
        assert output.is_compatible_with(step_input) is True

    def test_is_compatible_with_int_from_str(self):
        output = Schema.from_dict({"val": "str"})
        step_input = Schema.from_dict({"val": "int"})  # str → int coercion
        assert output.is_compatible_with(step_input) is True

    def test_to_simple_dict(self):
        s = Schema.from_dict({"name": "str", "age": "int"})
        d = s.to_simple_dict()
        assert d == {"name": "str", "age": "int"}


# ===========================================================================
# ToolResult 测试
# ===========================================================================

class TestToolResult:
    def test_success_factory(self):
        r = ToolResult.success({"url": "http://x"}, exec_time_ms=100)
        assert r.is_success is True
        assert r.status == ToolStatus.SUCCESS
        assert r.output == {"url": "http://x"}
        assert r.exec_time_ms == 100

    def test_failure_factory(self):
        r = ToolResult.failure("not found", ToolErrorType.SCHEMA_MISMATCH)
        assert r.is_success is False
        assert r.error == "not found"
        assert r.error_type == ToolErrorType.SCHEMA_MISMATCH

    def test_timeout_factory(self):
        r = ToolResult.timeout(timeout_s=5.0)
        assert r.status == ToolStatus.TIMEOUT
        assert r.error_type == ToolErrorType.TIMEOUT

    def test_is_retryable(self):
        t = ToolResult.timeout(timeout_s=1.0)
        n = ToolResult.failure("err", ToolErrorType.NETWORK)
        s = ToolResult.failure("err", ToolErrorType.SCHEMA_MISMATCH)
        assert t.is_retryable is True
        assert n.is_retryable is True
        assert s.is_retryable is False


# ===========================================================================
# Tool 测试
# ===========================================================================

class TestTool:
    def test_invoke_success(self):
        class DummyTool(Tool):
            def _do_invoke(self, params):
                return {"echo": params.get("x", 0) * 2}

        tool = DummyTool(
            name="dummy",
            description="",
            input_schema=Schema.from_dict({"x": "int"}),
            output_schema=Schema.from_dict({"echo": "int"}),
        )
        result = tool.invoke({"x": 5})
        assert result.is_success is True
        assert result.output == {"echo": 10}

    def test_invoke_failure(self):
        class FailTool(Tool):
            def _do_invoke(self, params):
                raise ValueError("test error")

        tool = FailTool(
            name="fail",
            description="",
            input_schema=Schema.from_dict({}),
            output_schema=Schema.from_dict({}),
        )
        result = tool.invoke({})
        assert result.is_success is False
        assert result.error_type == ToolErrorType.EXCEPTION

    def test_invoke_retry_on_timeout(self):
        class FlakeyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="flakey",
                    description="",
                    input_schema=Schema.from_dict({}),
                    output_schema=Schema.from_dict({"ok": "bool"}),
                    retry_policy=RetryPolicy(
                        max_attempts=3,
                        base_delay_ms=10,
                        retriable_errors=[ToolErrorType.TIMEOUT],
                    ),
                )
                self._attempts = 0

            def _do_invoke(self, params):
                self._attempts += 1
                if self._attempts < 3:
                    raise TimeoutError("simulated timeout")
                return {"ok": True}

        tool = FlakeyTool()
        result = tool.invoke({})
        assert result.is_success is True
        assert tool._attempts == 3

    def test_error_classification(self):
        class ExTool(Tool):
            def _do_invoke(self, params):
                raise PermissionError("access denied")

        tool = ExTool(
            name="ex",
            description="",
            input_schema=Schema.from_dict({}),
            output_schema=Schema.from_dict({}),
        )
        result = tool.invoke({})
        assert result.error_type == ToolErrorType.PERMISSION_DENIED

    def test_invoke_input_schema_mismatch(self):
        class DummyTool(Tool):
            def _do_invoke(self, params):
                return {"echo": params["x"]}

        tool = DummyTool(
            name="dummy",
            description="",
            input_schema=Schema.from_dict({"x": "int"}),
            output_schema=Schema.from_dict({"echo": "int"}),
        )
        result = tool.invoke({})
        assert result.is_success is False
        assert result.status == ToolStatus.FAILURE
        assert result.error_type == ToolErrorType.SCHEMA_MISMATCH

    def test_invoke_output_schema_mismatch(self):
        class BadOutputTool(Tool):
            def _do_invoke(self, params):
                return {"wrong": "shape"}

        tool = BadOutputTool(
            name="bad_output",
            description="",
            input_schema=Schema.from_dict({}),
            output_schema=Schema.from_dict({"echo": "int"}),
        )
        result = tool.invoke({})
        assert result.is_success is False
        assert result.status == ToolStatus.FAILURE
        assert result.error_type == ToolErrorType.SCHEMA_MISMATCH

    def test_retry_timeout_preserves_timeout_result(self):
        class AlwaysTimeoutTool(Tool):
            def __init__(self):
                super().__init__(
                    name="always_timeout",
                    description="",
                    input_schema=Schema.from_dict({}),
                    output_schema=Schema.from_dict({"ok": "bool"}),
                    timeout_s=1.5,
                    retry_policy=RetryPolicy(
                        max_attempts=2,
                        base_delay_ms=1,
                        retriable_errors=[ToolErrorType.TIMEOUT],
                    ),
                )

            def _do_invoke(self, params):
                raise TimeoutError("simulated timeout")

        result = AlwaysTimeoutTool().invoke({})
        assert result.status == ToolStatus.TIMEOUT
        assert result.error_type == ToolErrorType.TIMEOUT
        assert result.metadata["attempts"] == 2
        assert result.metadata["timeout_s"] == 1.5


# ===========================================================================
# ToolRegistry 测试
# ===========================================================================

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool
        assert "echo" in reg

    def test_register_callable(self):
        reg = ToolRegistry()
        reg.register(
            lambda q: {"r": q},
            name="test_fn",
            input_schema=Schema.from_dict({"q": "str"}),
            output_schema=Schema.from_dict({"r": "str"}),
        )
        t = reg.get("test_fn")
        assert t is not None
        assert t.name == "test_fn"

    def test_find_compatible_by_field(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        results = reg.find_compatible({"echoed": "str"})
        assert len(results) >= 1
        assert any(t.name == "echo" for t in results)

    def test_find_compatible_empty(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        results = reg.find_compatible({"nonexistent": "str"})
        assert results == []

    def test_find_compatible_respects_schema_types(self):
        reg = ToolRegistry()
        reg.register(
            lambda raw_count: {"count": raw_count},
            name="count_reader",
            input_schema=Schema.from_dict({"raw_count": "str"}),
            output_schema=Schema.from_dict({"count": "str"}),
        )

        compatible = reg.find_compatible({"count": "int"})
        incompatible = reg.find_compatible({"count": "bool"})

        assert [tool.name for tool in compatible] == ["count_reader"]
        assert incompatible == []

    def test_find_by_tag(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        results = reg.find_by_tag("echo")
        assert any(t.name == "echo" for t in results)

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.unregister("echo")
        assert reg.get("echo") is None

    def test_duplicate_register_overwrites(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        t2 = EchoTool()
        t2.name = "echo"
        reg.register(t2)
        assert len(reg) == 1


# ===========================================================================
# Adapter 测试
# ===========================================================================

class TestAdapter:
    def test_apply(self):
        a = Adapter(
            name="double",
            input_schema=Schema.from_dict({"x": "int"}),
            output_schema=Schema.from_dict({"x2": "int"}),
            transform=lambda d: {"x2": d["x"] * 2},
        )
        result = a.apply({"x": 5})
        assert result == {"x2": 10}

    def test_can_accept(self):
        a = Adapter(
            name="test",
            input_schema=Schema.from_dict({"html": "str"}),
            output_schema=Schema.from_dict({"text": "str"}),
            transform=lambda d: d,
        )
        s = Schema.from_dict({"html": "str", "extra": "str"})
        assert a.can_accept(s) is True

    def test_can_produce(self):
        a = Adapter(
            name="test",
            input_schema=Schema.from_dict({"html": "str"}),
            output_schema=Schema.from_dict({"text": "str"}),
            transform=lambda d: d,
        )
        s = Schema.from_dict({"text": "str"})
        assert a.can_produce(s) is True


class TestAdapterRegistry:
    def test_find_path_direct(self):
        reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            reg.register(a)

        html_to_text = reg.find_direct(
            from_schema=Schema.from_dict({"html": "str"}),
            to_schema=Schema.from_dict({"text": "str"}),
        )
        assert html_to_text is not None
        assert html_to_text.name == "html_to_text"

    def test_find_path_chain(self):
        reg = AdapterRegistry()
        reg.register(Adapter(
            name="html_to_text",
            input_schema=Schema.from_dict({"html": "str"}),
            output_schema=Schema.from_dict({"text": "str"}),
            transform=lambda d: {"text": d["html"]},
        ))
        reg.register(Adapter(
            name="text_to_upper",
            input_schema=Schema.from_dict({"text": "str"}),
            output_schema=Schema.from_dict({"text": "str"}),
            transform=lambda d: {"text": d["text"].upper()},
        ))

        path = reg.find_path(
            from_schema=Schema.from_dict({"html": "str"}),
            to_schema=Schema.from_dict({"text": "str"}),
        )
        assert len(path) >= 1  # 至少有一个直接匹配

    def test_find_path_no_path(self):
        reg = AdapterRegistry()
        # 不注册任何 adapter
        path = reg.find_path(
            from_schema=Schema.from_dict({"html": "str"}),
            to_schema=Schema.from_dict({"json": "str"}),
        )
        assert path == []

    def test_builtin_adapters_loaded(self):
        reg = AdapterRegistry()
        for a in BUILTIN_ADAPTERS:
            reg.register(a)
        assert len(reg) >= 2  # html_to_text + str_to_int + str_to_float


# ===========================================================================
# BashTool 安全测试
# ===========================================================================

class TestBashToolSecurity:
    def test_whitelisted_command_allowed(self):
        tool = BashTool(allowed_commands=["echo", "ls"])
        result = tool._do_invoke({"cmd": "echo hello"})
        assert result["stdout"].strip() == "hello"

    def test_non_whitelisted_rejected(self):
        tool = BashTool(allowed_commands=["echo"])
        with pytest.raises(PermissionError):
            tool._do_invoke({"cmd": "ls"})

    def test_path_traversal_blocked(self):
        tool = BashTool(allowed_commands=["cat"])
        # 路径穿越: 相对路径尝试穿越 workspace
        with pytest.raises(PermissionError):
            tool._do_invoke({"cmd": "cat ../../etc/passwd"})


# ===========================================================================
# EchoTool 集成
# ===========================================================================

class TestEchoTool:
    def test_echo_invoke(self):
        tool = EchoTool()
        result = tool.invoke({"message": "hello world"})
        assert result.is_success is True
        assert result.output["echoed"] == "hello world"

    def test_echo_optional_missing(self):
        tool = EchoTool()
        result = tool.invoke({"message": "hi"})
        assert result.output.get("extra") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
