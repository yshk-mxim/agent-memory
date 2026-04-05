# SPDX-License-Identifier: MIT
"""Tests for JsonLinesToolCallParser — all JSON formats including ReAct."""

from agent_memory.adapters.outbound.tool_call_parsers.json_lines import (
    JsonLinesToolCallParser,
)


class TestReActFormat:
    """ReAct: {"action": "Name", "action_input": ...}"""

    def setup_method(self):
        self.parser = JsonLinesToolCallParser()

    def test_react_with_json_string_input(self):
        text = '{"action": "Agent", "action_input": "{\\"subagent_type\\": \\"Explore\\", \\"instructions\\": \\"Research topic.\\"}"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Agent"
        # "instructions" normalized to "prompt"
        assert calls[0].input["prompt"] == "Research topic."
        assert calls[0].input["subagent_type"] == "Explore"

    def test_react_with_dict_input(self):
        text = '{"action": "WebSearch", "action_input": {"query": "test"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "WebSearch"
        assert calls[0].input["query"] == "test"

    def test_react_in_code_block(self):
        text = '```json\n{"action": "Agent", "action_input": {"instructions": "do X"}}\n```'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Agent"
        assert calls[0].input["prompt"] == "do X"  # Normalized

    def test_react_unparseable_string_agent(self):
        """Unparseable action_input string for Agent → wrapped as prompt."""
        text = '{"action": "Agent", "action_input": "just do the research"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["prompt"] == "just do the research"

    def test_react_unparseable_string_other(self):
        """Unparseable action_input string for non-Agent → wrapped as input."""
        text = '{"action": "CustomTool", "action_input": "some value"}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input == {"input": "some value"}

    def test_react_websearch_param_normalization(self):
        text = '{"action": "WebSearch", "action_input": {"search_query": "crude oil prices"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].input["query"] == "crude oil prices"
        assert "search_query" not in calls[0].input


class TestNameParametersFormat:
    """Template-instructed: {"name": ..., "parameters": ...}"""

    def setup_method(self):
        self.parser = JsonLinesToolCallParser()

    def test_basic(self):
        text = '{"name": "Read", "parameters": {"file_path": "/foo.py"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert calls[0].input["file_path"] == "/foo.py"

    def test_multiple_on_separate_lines(self):
        text = '{"name": "Read", "parameters": {"file_path": "/a.py"}}\n{"name": "Grep", "parameters": {"pattern": "foo"}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2


class TestToolUseFormat:
    """Anthropic proxy: {"tool_use": {"name": ..., "input": ...}}"""

    def setup_method(self):
        self.parser = JsonLinesToolCallParser()

    def test_inline(self):
        text = 'Let me read. {"tool_use": {"name": "Read", "input": {"path": "x"}}}'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert "Let me read." in remaining

    def test_missing_input_rejected(self):
        text = '{"tool_use": {"name": "test"}}'
        remaining, calls = self.parser.parse(text)
        assert calls == []


class TestCodeBlocks:
    def setup_method(self):
        self.parser = JsonLinesToolCallParser()

    def test_json_code_block(self):
        text = '```json\n{"name": "Bash", "parameters": {"command": "ls"}}\n```'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "Bash"

    def test_multiple_code_blocks(self):
        text = (
            '```json\n{"action": "Agent", "action_input": {"instructions": "A"}}\n```\n'
            '```json\n{"action": "WebSearch", "action_input": {"query": "B"}}\n```'
        )
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 2
        names = {c.name for c in calls}
        assert names == {"Agent", "WebSearch"}


class TestEdgeCases:
    def setup_method(self):
        self.parser = JsonLinesToolCallParser()

    def test_empty_text(self):
        remaining, calls = self.parser.parse("")
        assert remaining == ""
        assert calls == []

    def test_no_json(self):
        text = "Just plain text, no JSON here."
        remaining, calls = self.parser.parse(text)
        assert remaining == text
        assert calls == []

    def test_invalid_json(self):
        text = '{"name": "broken", parameters: missing quotes}'
        remaining, calls = self.parser.parse(text)
        assert calls == []

    def test_text_preserved_around_tool_calls(self):
        text = 'Before {"name": "Tool", "parameters": {"x": 1}} After'
        remaining, calls = self.parser.parse(text)
        assert len(calls) == 1
        assert "Before" in remaining
        assert "After" in remaining
