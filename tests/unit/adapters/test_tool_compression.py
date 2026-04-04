# SPDX-License-Identifier: MIT
"""Tests for tool definition compression."""

from agent_memory.adapters.inbound.tool_compression import (
    KNOWN_TOOLS,
    compress_tool_definitions,
)


# ── Sample tool definitions (Anthropic format) ────────────────


def _make_tool(name: str, description: str = "", properties: dict | None = None,
               required: list[str] | None = None) -> dict:
    schema = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema}


SAMPLE_READ = _make_tool(
    "Read",
    "Reads a file from the local filesystem.",
    {"file_path": {"type": "string", "description": "Absolute path"},
     "offset": {"type": "integer", "description": "Start line"},
     "limit": {"type": "integer", "description": "Max lines"}},
    required=["file_path"],
)

SAMPLE_WRITE = _make_tool(
    "Write",
    "Writes a file to the local filesystem.",
    {"file_path": {"type": "string"}, "content": {"type": "string"}},
    required=["file_path", "content"],
)

SAMPLE_UNKNOWN = _make_tool(
    "CustomWidget",
    "Renders a custom widget in the UI. Supports dark mode and animations.",
    {"widget_id": {"type": "string"}, "theme": {"type": "string"},
     "animate": {"type": "boolean"}},
    required=["widget_id"],
)


# ── Known tools ───────────────────────────────────────────────


class TestKnownToolCompression:
    def test_known_tool_uses_prewritten_description(self):
        result = compress_tool_definitions([SAMPLE_READ])
        assert "Read: Read file contents" in result
        # Should NOT contain the original verbose description
        assert "Reads a file from the local filesystem" not in result

    def test_multiple_known_tools(self):
        result = compress_tool_definitions([SAMPLE_READ, SAMPLE_WRITE])
        assert "- Read:" in result
        assert "- Write:" in result

    def test_all_claude_code_tools_have_entries(self):
        """Verify common Claude Code tools are in KNOWN_TOOLS."""
        expected = {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent", "TodoWrite"}
        assert expected.issubset(set(KNOWN_TOOLS.keys()))

    def test_deduplicates_tools(self):
        result = compress_tool_definitions([SAMPLE_READ, SAMPLE_READ])
        assert result.count("- Read:") == 1


# ── Unknown tools ─────────────────────────────────────────────


class TestUnknownToolCompression:
    def test_unknown_tool_extracts_from_schema(self):
        result = compress_tool_definitions([SAMPLE_UNKNOWN])
        assert "- CustomWidget:" in result
        assert "widget_id" in result
        assert "required" in result

    def test_unknown_tool_first_sentence_only(self):
        result = compress_tool_definitions([SAMPLE_UNKNOWN])
        # Should have first sentence, not full description
        assert "Renders a custom widget in the UI." in result
        assert "animations" not in result

    def test_unknown_tool_without_schema(self):
        tool = {"name": "Bare", "description": "A bare tool.", "input_schema": {}}
        result = compress_tool_definitions([tool])
        assert "- Bare:" in result


# ── Output format ─────────────────────────────────────────────


class TestOutputFormat:
    def test_includes_header(self):
        result = compress_tool_definitions([SAMPLE_READ])
        assert "You have access to the following tools" in result
        assert '{"name": "tool_name"' in result

    def test_includes_available_tools_label(self):
        result = compress_tool_definitions([SAMPLE_READ])
        assert "Available tools:" in result

    def test_no_markdown_code_blocks(self):
        result = compress_tool_definitions([SAMPLE_READ, SAMPLE_UNKNOWN])
        assert "```" not in result

    def test_empty_tools_list(self):
        result = compress_tool_definitions([])
        assert "Available tools:" in result
        # Should still have header but no tool lines
        assert "- " not in result


# ── OpenAI format ─────────────────────────────────────────────


class TestOpenAIFormat:
    def test_openai_function_format(self):
        tool = {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }
        result = compress_tool_definitions([tool])
        assert "- Read:" in result


# ── Token efficiency ──────────────────────────────────────────


class TestTokenEfficiency:
    def test_compressed_much_shorter_than_raw(self):
        """Compressed output should be significantly smaller than raw JSON schemas."""
        import json

        all_tools = [
            _make_tool(name, f"Tool {name} does things.", {
                "param1": {"type": "string", "description": "A long description " * 10},
                "param2": {"type": "integer", "description": "Another param " * 5},
            }, required=["param1"])
            for name in KNOWN_TOOLS
        ]
        raw_size = sum(len(json.dumps(t, indent=2)) for t in all_tools)
        compressed = compress_tool_definitions(all_tools)
        compressed_size = len(compressed)

        # Compressed should be at most 30% of raw JSON size
        assert compressed_size < raw_size * 0.3, (
            f"Compressed ({compressed_size}) should be much smaller than raw ({raw_size})"
        )


# ── Prefix hash normalization ────────────────────────────────


class TestPrefixHashNormalization:
    def test_tool_order_independent_hash(self):
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        h1 = SharedPrefixCache.compute_hash("system prompt", "Read\nWrite\nBash")
        h2 = SharedPrefixCache.compute_hash("system prompt", "Write\nRead\nBash")
        h3 = SharedPrefixCache.compute_hash("system prompt", "Bash\nWrite\nRead")
        assert h1 == h2 == h3

    def test_different_tools_different_hash(self):
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        h1 = SharedPrefixCache.compute_hash("sys", "Read\nWrite")
        h2 = SharedPrefixCache.compute_hash("sys", "Read\nBash")
        assert h1 != h2

    def test_empty_tools_stable(self):
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        h1 = SharedPrefixCache.compute_hash("sys", "")
        h2 = SharedPrefixCache.compute_hash("sys", "")
        assert h1 == h2
