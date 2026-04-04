# SPDX-License-Identifier: MIT
"""Tests for model-aware tool call parsers."""

import pytest

from agent_memory.adapters.outbound.tool_call_parsers import (
    GemmaToolCallParser,
    JsonLinesToolCallParser,
    MistralToolCallParser,
    QwenToolCallParser,
    create_parser_for_model,
)
from agent_memory.adapters.outbound.tool_call_parsers.llama_server_native import (
    extract_from_openai_tool_calls,
)
from agent_memory.application.tool_call_parsing import ToolCallParserChain
from agent_memory.domain.value_objects import ParsedToolCall


# ── Gemma Parser ───────────────────────────────────────────────


class TestGemmaParser:
    parser = GemmaToolCallParser()

    def test_simple_call(self):
        text = 'call:Read{file_path: "/foo/bar.py"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert calls[0].input["file_path"] == "/foo/bar.py"
        assert remaining == ""

    def test_namespaced_call(self):
        text = 'call:agent_memory:Agent{subagent_type: "Explore", task: "search"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Agent"
        assert calls[0].input["subagent_type"] == "Explore"

    def test_double_namespaced(self):
        text = 'call:a:b:Read{path: "/foo"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"

    def test_space_before_brace(self):
        text = 'call:Read {file_path: "/foo"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert calls[0].input["file_path"] == "/foo"

    def test_newline_before_brace(self):
        text = 'call:Agent\n{"subagent_type": "Explore", "task": "search"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Agent"

    def test_concatenated_calls(self):
        text = 'call:A{x: "1"}call:B{y: "2"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2
        assert calls[0].name == "A"
        assert calls[1].name == "B"
        assert remaining == ""

    def test_nested_arrays(self):
        text = 'call:TaskCreate{tasks: [{description: "Research oil prices", title: "Analysis"}]}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "TaskCreate"
        assert isinstance(calls[0].input["tasks"], list)

    def test_nested_objects(self):
        text = 'call:Configure{opts: {debug: true, level: 3}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["opts"]["debug"] is True

    def test_unquoted_keys(self):
        text = 'call:Read{file_path: "/foo", limit: 100}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["file_path"] == "/foo"
        assert calls[0].input["limit"] == 100

    def test_valid_json_args(self):
        text = 'call:Read{"file_path": "/foo/bar.py"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["file_path"] == "/foo/bar.py"

    def test_boolean_values(self):
        text = "call:SetFlag{verbose: true, quiet: false}"
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["verbose"] is True
        assert calls[0].input["quiet"] is False

    def test_mixed_text_and_calls(self):
        text = "I'll read the file now.\ncall:Read{path: \"/foo\"}\nDone."
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert "read the file" in remaining

    def test_empty_args(self):
        text = "call:NoArgs{}"
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input == {}

    def test_no_call_prefix_returns_empty(self):
        text = "Just some regular text without any tool calls."
        remaining, calls = self.parser.parse(text)
        assert calls == []
        assert remaining == text

    def test_unbalanced_braces_skipped(self):
        text = "call:Bad{key: \"value\ncall:Good{x: \"1\"}"
        remaining, calls = self.parser.parse(text)
        # Bad has unbalanced braces, Good should still parse
        assert any(c.name == "Good" for c in calls)

    def test_strips_calls_from_text(self):
        text = "before call:Read{path: \"/foo\"} after"
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert "call:" not in remaining
        assert "before" in remaining

    def test_gemma_delimiter_remnants(self):
        text = 'call:Read{path: <|"|>/foo/bar<|"|>}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["path"] == "/foo/bar"

    # Real model output from production logs
    def test_real_gemma_multi_tool(self):
        text = (
            'call:TaskCreate{tasks: [{description: "Research current oil/gasoline '
            'price trends", tittle: "Physical Reality Analysis"}]}'
            'call:TaskCreate{tasks: [{description: "Analyze oil futures", '
            'tittle: "Economic Reality Analysis"}]}'
            'call:Agent{subagent_type: "Explore", task: "Analyze the current '
            'physical reality"}'
        )
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 3
        assert calls[0].name == "TaskCreate"
        assert calls[1].name == "TaskCreate"
        assert calls[2].name == "Agent"

    def test_returns_parsed_tool_call_type(self):
        text = 'call:Read{path: "/foo"}'
        _, calls = self.parser.parse(text)
        assert isinstance(calls[0], ParsedToolCall)
        assert calls[0].id.startswith("toolu_")


# ── Qwen Parser ────────────────────────────────────────────────


class TestQwenParser:
    parser = QwenToolCallParser()

    def test_single_call(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "/foo"}}</tool_call>'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].input["path"] == "/foo"
        assert remaining == ""

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "read", "arguments": {"p": "a"}}</tool_call>\n'
            '<tool_call>{"name": "write", "arguments": {"p": "b"}}</tool_call>'
        )
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2

    def test_arguments_as_string(self):
        text = '<tool_call>{"name": "f", "arguments": "{\\"key\\": \\"val\\"}"}</tool_call>'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["key"] == "val"

    def test_parameters_key(self):
        text = '<tool_call>{"name": "f", "parameters": {"key": "val"}}</tool_call>'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["key"] == "val"

    def test_no_tags_returns_empty(self):
        text = "No tool calls here."
        remaining, calls = self.parser.parse(text)
        assert calls == []

    def test_malformed_json_skipped(self):
        text = "<tool_call>not json</tool_call>"
        remaining, calls = self.parser.parse(text)
        assert calls == []


# ── Mistral Parser ─────────────────────────────────────────────


class TestMistralParser:
    parser = MistralToolCallParser()

    def test_single_call(self):
        text = '[TOOL_CALLS][{"name": "f", "arguments": {"key": "val"}}]'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "f"
        assert calls[0].input["key"] == "val"

    def test_multiple_calls(self):
        text = '[TOOL_CALLS][{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2

    def test_with_preceding_text(self):
        text = 'I will call tools now. [TOOL_CALLS][{"name": "f", "arguments": {}}]'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert "I will call tools" in remaining

    def test_no_marker_returns_empty(self):
        text = "No tools here."
        remaining, calls = self.parser.parse(text)
        assert calls == []


# ── JSON Lines Parser ──────────────────────────────────────────


class TestJsonLinesParser:
    parser = JsonLinesToolCallParser()

    def test_name_parameters(self):
        text = '{"name": "read", "parameters": {"path": "/foo"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "read"
        assert calls[0].input["path"] == "/foo"

    def test_name_arguments(self):
        text = '{"name": "read", "arguments": {"path": "/foo"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1

    def test_tool_use_format(self):
        text = '{"tool_use": {"name": "read", "input": {"path": "/foo"}}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "read"

    def test_function_call_format(self):
        text = '{"function_call": {"name": "read", "arguments": {"path": "/foo"}}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1

    def test_multiple_json_lines(self):
        text = (
            '{"name": "a", "parameters": {"x": 1}}\n'
            '{"name": "b", "parameters": {"y": 2}}'
        )
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2

    def test_json_in_code_block(self):
        text = 'Here is a tool call:\n```json\n{"name": "read", "parameters": {"path": "/foo"}}\n```'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "read"

    def test_mixed_text_and_json(self):
        text = 'Some text\n{"name": "read", "parameters": {"path": "/foo"}}\nMore text'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1

    def test_no_json_returns_empty(self):
        text = "Just plain text."
        remaining, calls = self.parser.parse(text)
        assert calls == []

    def test_json_without_tool_keys_ignored(self):
        text = '{"status": "ok", "count": 42}'
        remaining, calls = self.parser.parse(text)
        assert calls == []


# ── Native Server Parser ──────────────────────────────────────


class TestNativeServerParser:
    def test_single_call(self):
        raw = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "/foo"}'},
            }
        ]
        calls = extract_from_openai_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "read"
        assert calls[0].id == "call_123"

    def test_arguments_as_dict(self):
        raw = [
            {
                "id": "call_456",
                "type": "function",
                "function": {"name": "write", "arguments": {"content": "hello"}},
            }
        ]
        calls = extract_from_openai_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].input["content"] == "hello"

    def test_empty_name_skipped(self):
        raw = [{"function": {"name": "", "arguments": "{}"}}]
        calls = extract_from_openai_tool_calls(raw)
        assert calls == []


# ── Parser Chain ───────────────────────────────────────────────


class TestParserChain:
    def test_gemma_chain_prefers_native(self):
        chain = ToolCallParserChain([GemmaToolCallParser(), JsonLinesToolCallParser()])
        text = 'call:Read{path: "/foo"}'
        _, calls = chain.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"

    def test_falls_through_to_json(self):
        chain = ToolCallParserChain([GemmaToolCallParser(), JsonLinesToolCallParser()])
        text = '{"name": "read", "parameters": {"path": "/foo"}}'
        _, calls = chain.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "read"

    def test_no_parsers_raises(self):
        with pytest.raises(ValueError):
            ToolCallParserChain([])

    def test_no_calls_returns_original_text(self):
        chain = ToolCallParserChain([GemmaToolCallParser(), JsonLinesToolCallParser()])
        text = "Just regular text."
        remaining, calls = chain.parse(text)
        assert calls == []
        assert remaining == text


# ── Factory ────────────────────────────────────────────────────


class TestFactory:
    def test_gemma_model_ids(self):
        for mid in ("gemma-4-26b-a4b", "gemma-4-31b", "Gemma-4-E2B"):
            chain = create_parser_for_model(mid)
            assert len(chain._parsers) == 2
            assert isinstance(chain._parsers[0], GemmaToolCallParser)

    def test_qwen_model_ids(self):
        for mid in ("qwen3-coder-next", "Qwen2.5-32B"):
            chain = create_parser_for_model(mid)
            assert isinstance(chain._parsers[0], QwenToolCallParser)

    def test_mistral_model_id(self):
        chain = create_parser_for_model("mistral-nemo-12b")
        assert isinstance(chain._parsers[0], MistralToolCallParser)

    def test_unknown_model_gets_fallback(self):
        chain = create_parser_for_model("some-unknown-model-v3")
        assert len(chain._parsers) == 1
        assert isinstance(chain._parsers[0], JsonLinesToolCallParser)
