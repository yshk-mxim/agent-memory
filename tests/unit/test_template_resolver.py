"""Unit tests for demo/lib/template_resolver.py."""

from __future__ import annotations

import pytest
from demo.lib.template_resolver import (
    extract_phase_refs,
    has_template_refs,
    resolve_template,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# resolve_template
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    def test_single_ref_substituted(self) -> None:
        template = "Context:\n${debate.messages[alice]}"
        phase_messages = {
            "debate": [
                {"sender_name": "Alice", "content": "I agree."},
                {"sender_name": "Bob", "content": "I disagree."},
            ],
        }
        result = resolve_template(template, phase_messages)
        assert result == "Context:\nAlice: I agree.\nBob: I disagree."

    def test_multiple_refs_substituted(self) -> None:
        template = "Phase1:\n${intro.messages[a]}\n\nPhase2:\n${outro.messages[b]}"
        phase_messages = {
            "intro": [{"sender_name": "Host", "content": "Welcome"}],
            "outro": [{"sender_name": "Host", "content": "Goodbye"}],
        }
        result = resolve_template(template, phase_messages)
        assert "Host: Welcome" in result
        assert "Host: Goodbye" in result

    def test_no_refs_passthrough(self) -> None:
        plain = "No placeholders here."
        result = resolve_template(plain, {"phase": []})
        assert result == plain

    def test_missing_phase_returns_no_messages(self) -> None:
        template = "Prior: ${missing.messages[agent]}"
        result = resolve_template(template, {})
        assert result == "Prior: (no messages yet)"

    def test_empty_message_list_returns_no_messages(self) -> None:
        template = "${empty.messages[x]}"
        result = resolve_template(template, {"empty": []})
        assert result == "(no messages yet)"

    def test_missing_sender_name_defaults_to_unknown(self) -> None:
        template = "${phase.messages[a]}"
        phase_messages = {"phase": [{"content": "hello"}]}
        result = resolve_template(template, phase_messages)
        assert result == "Unknown: hello"

    def test_missing_content_defaults_to_empty(self) -> None:
        template = "${phase.messages[a]}"
        phase_messages = {"phase": [{"sender_name": "Bot"}]}
        result = resolve_template(template, phase_messages)
        assert result == "Bot: "

    def test_empty_template(self) -> None:
        assert resolve_template("", {"p": []}) == ""

    def test_special_chars_in_message_content(self) -> None:
        template = "${phase.messages[a]}"
        phase_messages = {
            "phase": [
                {"sender_name": "Bot", "content": "Price is $100 (50% off)"},
            ],
        }
        result = resolve_template(template, phase_messages)
        assert "Price is $100 (50% off)" in result

    def test_multiple_messages_joined_with_newlines(self) -> None:
        template = "${chat.messages[x]}"
        phase_messages = {
            "chat": [
                {"sender_name": "A", "content": "line1"},
                {"sender_name": "B", "content": "line2"},
                {"sender_name": "A", "content": "line3"},
            ],
        }
        result = resolve_template(template, phase_messages)
        assert result == "A: line1\nB: line2\nA: line3"

    def test_surrounding_text_preserved(self) -> None:
        template = "BEGIN ${p.messages[x]} END"
        phase_messages = {
            "p": [{"sender_name": "S", "content": "hi"}],
        }
        result = resolve_template(template, phase_messages)
        assert result == "BEGIN S: hi END"


# ---------------------------------------------------------------------------
# resolve_template — agent perspective formatting
# ---------------------------------------------------------------------------


class TestResolveTemplatePerspective:
    """Tests for perspective-aware formatting with agent_display_names."""

    def test_own_messages_labeled_you(self) -> None:
        template = "${chat.messages[alice]}"
        phase_messages = {
            "chat": [
                {"sender_name": "Alice", "content": "I think so."},
                {"sender_name": "Bob", "content": "I disagree."},
            ],
        }
        agent_names = {"alice": "Alice", "bob": "Bob"}
        result = resolve_template(template, phase_messages, agent_names)
        assert result == "You: I think so.\nBob: I disagree."

    def test_other_agent_perspective_different(self) -> None:
        template = "${chat.messages[bob]}"
        phase_messages = {
            "chat": [
                {"sender_name": "Alice", "content": "I think so."},
                {"sender_name": "Bob", "content": "I disagree."},
            ],
        }
        agent_names = {"alice": "Alice", "bob": "Bob"}
        result = resolve_template(template, phase_messages, agent_names)
        assert result == "Alice: I think so.\nYou: I disagree."

    def test_system_messages_skipped(self) -> None:
        template = "${phase.messages[alice]}"
        phase_messages = {
            "phase": [
                {"sender_name": "System", "content": "Topic intro"},
                {"sender_name": "Alice", "content": "My response."},
            ],
        }
        agent_names = {"alice": "Alice"}
        result = resolve_template(template, phase_messages, agent_names)
        assert "System" not in result
        assert result == "You: My response."

    def test_no_agent_names_shows_all_senders(self) -> None:
        template = "${chat.messages[alice]}"
        phase_messages = {
            "chat": [
                {"sender_name": "Alice", "content": "Hi"},
                {"sender_name": "Bob", "content": "Hello"},
            ],
        }
        result = resolve_template(template, phase_messages)
        assert result == "Alice: Hi\nBob: Hello"

    def test_unknown_agent_key_no_perspective(self) -> None:
        template = "${chat.messages[unknown]}"
        phase_messages = {
            "chat": [{"sender_name": "Alice", "content": "Hi"}],
        }
        agent_names = {"alice": "Alice"}
        result = resolve_template(template, phase_messages, agent_names)
        assert result == "Alice: Hi"

    def test_per_agent_templates_different_perspectives(self) -> None:
        """Simulate gossip scenario: same phase, different agent perspectives."""
        phase_messages = {
            "private_chat": [
                {"sender_name": "Alice", "content": "Did you hear about Eve?"},
                {"sender_name": "Bob", "content": "No, what happened?"},
                {"sender_name": "Alice", "content": "She got promoted!"},
            ],
        }
        agent_names = {"alice": "Alice", "bob": "Bob", "eve": "Eve"}

        alice_tmpl = "Your memories:\n${private_chat.messages[alice]}"
        bob_tmpl = "Your memories:\n${private_chat.messages[bob]}"

        alice_result = resolve_template(
            alice_tmpl,
            phase_messages,
            agent_names,
        )
        bob_result = resolve_template(
            bob_tmpl,
            phase_messages,
            agent_names,
        )

        assert "You: Did you hear about Eve?" in alice_result
        assert "Bob: No, what happened?" in alice_result

        assert "Alice: Did you hear about Eve?" in bob_result
        assert "You: No, what happened?" in bob_result


# ---------------------------------------------------------------------------
# has_template_refs
# ---------------------------------------------------------------------------


class TestHasTemplateRefs:
    def test_true_for_valid_ref(self) -> None:
        assert has_template_refs("See ${debate.messages[alice]}")

    def test_true_for_multiple_refs(self) -> None:
        assert has_template_refs("${a.messages[x]} and ${b.messages[y]}")

    def test_false_for_plain_text(self) -> None:
        assert not has_template_refs("No references here")

    def test_false_for_empty_string(self) -> None:
        assert not has_template_refs("")

    def test_false_for_partial_pattern(self) -> None:
        assert not has_template_refs("${incomplete.messages}")
        assert not has_template_refs("${.messages[x]}")


# ---------------------------------------------------------------------------
# extract_phase_refs
# ---------------------------------------------------------------------------


class TestExtractPhaseRefs:
    def test_single_phase(self) -> None:
        refs = extract_phase_refs("${debate.messages[alice]}")
        assert refs == {"debate"}

    def test_multiple_distinct_phases(self) -> None:
        template = "${intro.messages[a]} then ${outro.messages[b]}"
        assert extract_phase_refs(template) == {"intro", "outro"}

    def test_duplicate_phase_deduplicated(self) -> None:
        template = "${p.messages[x]} ${p.messages[y]}"
        assert extract_phase_refs(template) == {"p"}

    def test_no_phases_returns_empty_set(self) -> None:
        assert extract_phase_refs("plain text") == set()

    def test_empty_template_returns_empty_set(self) -> None:
        assert extract_phase_refs("") == set()
