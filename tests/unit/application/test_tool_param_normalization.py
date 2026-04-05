# SPDX-License-Identifier: MIT
"""Tests for tool_param_normalization — canonical parameter name mapping."""

from agent_memory.application.tool_param_normalization import normalize_tool_params


class TestNormalizeToolParams:
    def test_agent_instructions_to_prompt(self):
        result = normalize_tool_params("Agent", {"instructions": "do stuff"})
        assert result["prompt"] == "do stuff"
        assert "instructions" not in result

    def test_agent_task_to_prompt(self):
        result = normalize_tool_params("Agent", {"task": "research X"})
        assert result["prompt"] == "research X"

    def test_agent_input_to_prompt(self):
        result = normalize_tool_params("Agent", {"input": "look up Y"})
        assert result["prompt"] == "look up Y"

    def test_agent_auto_description(self):
        result = normalize_tool_params("Agent", {"prompt": "research recent news"})
        assert "description" in result
        assert result["description"] == "research recent news"

    def test_agent_preserves_existing_description(self):
        result = normalize_tool_params("Agent", {
            "prompt": "research",
            "description": "custom desc",
        })
        assert result["description"] == "custom desc"

    def test_agent_preserves_subagent_type(self):
        result = normalize_tool_params("Agent", {
            "subagent_type": "Explore",
            "instructions": "find X",
        })
        assert result["subagent_type"] == "Explore"
        assert result["prompt"] == "find X"

    def test_websearch_search_query_to_query(self):
        result = normalize_tool_params("WebSearch", {"search_query": "test"})
        assert result["query"] == "test"
        assert "search_query" not in result

    def test_webfetch_link_to_url(self):
        result = normalize_tool_params("WebFetch", {"link": "http://example.com"})
        assert result["url"] == "http://example.com"

    def test_unknown_tool_passthrough(self):
        result = normalize_tool_params("Bash", {"command": "ls"})
        assert result == {"command": "ls"}

    def test_unknown_params_passthrough(self):
        result = normalize_tool_params("Agent", {
            "prompt": "test",
            "custom_field": 42,
        })
        assert result["custom_field"] == 42

    def test_empty_params(self):
        result = normalize_tool_params("Agent", {})
        assert "description" in result  # Auto-added
