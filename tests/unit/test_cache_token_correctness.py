# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for cache token and prefix matching correctness.

Verifies that AgentBlocks correctly computes prefix matches at both
token-level (common_prefix_length) and character-level (common_prefix_chars).
Character-level matching exists to solve BPE tokenization boundary mismatches.
"""

import pytest

from agent_memory.domain.entities import AgentBlocks

pytestmark = pytest.mark.unit


def _make_agent(
    token_sequence: list[int] | None = None,
    prompt_text: str = "",
) -> AgentBlocks:
    """Helper to create an AgentBlocks with minimal valid structure."""
    return AgentBlocks(
        agent_id="test_agent",
        blocks={},
        total_tokens=0,
        token_sequence=token_sequence or [],
        prompt_text=prompt_text,
    )


class TestCommonPrefixLength:
    def test_full_match(self) -> None:
        agent = _make_agent(token_sequence=[10, 20, 30, 40, 50])
        assert agent.common_prefix_length([10, 20, 30, 40, 50]) == 5

    def test_partial_match(self) -> None:
        agent = _make_agent(token_sequence=[10, 20, 30, 40, 50])
        assert agent.common_prefix_length([10, 20, 30, 99, 99]) == 3

    def test_no_match(self) -> None:
        agent = _make_agent(token_sequence=[10, 20, 30])
        assert agent.common_prefix_length([99, 88, 77]) == 0

    def test_empty_cached_tokens(self) -> None:
        agent = _make_agent(token_sequence=[])
        assert agent.common_prefix_length([10, 20, 30]) == 0

    def test_empty_query_tokens(self) -> None:
        agent = _make_agent(token_sequence=[10, 20, 30])
        assert agent.common_prefix_length([]) == 0

    def test_both_empty(self) -> None:
        agent = _make_agent(token_sequence=[])
        assert agent.common_prefix_length([]) == 0

    def test_single_token_match(self) -> None:
        agent = _make_agent(token_sequence=[42])
        assert agent.common_prefix_length([42]) == 1

    def test_single_token_no_match(self) -> None:
        agent = _make_agent(token_sequence=[42])
        assert agent.common_prefix_length([99]) == 0

    def test_cached_shorter_than_query(self) -> None:
        agent = _make_agent(token_sequence=[10, 20])
        assert agent.common_prefix_length([10, 20, 30, 40]) == 2

    def test_query_shorter_than_cached(self) -> None:
        agent = _make_agent(token_sequence=[10, 20, 30, 40])
        assert agent.common_prefix_length([10, 20]) == 2

    def test_diverges_at_first_token(self) -> None:
        agent = _make_agent(token_sequence=[1, 2, 3])
        assert agent.common_prefix_length([9, 2, 3]) == 0

    def test_long_sequences(self) -> None:
        seq = list(range(10000))
        agent = _make_agent(token_sequence=seq)
        query = list(range(5000)) + [99999] + list(range(5001, 10000))
        assert agent.common_prefix_length(query) == 5000

    def test_formula_matches_zip_computation(self) -> None:
        """Verify prefix length matches analytical zip-based computation."""
        cached = [1, 2, 3, 4, 5]
        query = [1, 2, 3, 6, 7]
        expected = sum(1 for a, b in zip(cached, query) if a == b)
        # zip stops at shorter sequence, but prefix stops at first mismatch
        # so we compute manually
        expected = 0
        for a, b in zip(cached, query):
            if a != b:
                break
            expected += 1
        agent = _make_agent(token_sequence=cached)
        assert agent.common_prefix_length(query) == expected == 3


class TestCommonPrefixChars:
    def test_full_match(self) -> None:
        agent = _make_agent(prompt_text="Hello, how are you today?")
        assert agent.common_prefix_chars("Hello, how are you today?") == 25

    def test_partial_match(self) -> None:
        # "Hello, how are you today?" vs "Hello, how are you?"
        # Common prefix: "Hello, how are you" = 18 chars
        # Char 19: ' ' vs '?' — mismatch
        agent = _make_agent(prompt_text="Hello, how are you today?")
        assert agent.common_prefix_chars("Hello, how are you?") == 18

    def test_no_match(self) -> None:
        agent = _make_agent(prompt_text="Hello")
        assert agent.common_prefix_chars("Goodbye") == 0

    def test_empty_cached_text(self) -> None:
        agent = _make_agent(prompt_text="")
        assert agent.common_prefix_chars("Hello") == 0

    def test_empty_query_text(self) -> None:
        agent = _make_agent(prompt_text="Hello")
        assert agent.common_prefix_chars("") == 0

    def test_both_empty(self) -> None:
        agent = _make_agent(prompt_text="")
        assert agent.common_prefix_chars("") == 0

    def test_single_char_match(self) -> None:
        agent = _make_agent(prompt_text="A")
        assert agent.common_prefix_chars("A") == 1

    def test_single_char_no_match(self) -> None:
        agent = _make_agent(prompt_text="A")
        assert agent.common_prefix_chars("B") == 0

    def test_bpe_boundary_space(self) -> None:
        """Space after comma is a common BPE boundary — char matching handles it."""
        agent = _make_agent(prompt_text="Hello, world")
        assert agent.common_prefix_chars("Hello, ") == 7

    def test_cached_shorter_than_query(self) -> None:
        agent = _make_agent(prompt_text="Hi")
        assert agent.common_prefix_chars("Hi there") == 2

    def test_query_shorter_than_cached(self) -> None:
        agent = _make_agent(prompt_text="Hello world")
        assert agent.common_prefix_chars("Hello") == 5

    def test_unicode_chars(self) -> None:
        agent = _make_agent(prompt_text="Caf\u00e9 au lait")
        assert agent.common_prefix_chars("Caf\u00e9 au") == 7

    def test_diverges_at_first_char(self) -> None:
        agent = _make_agent(prompt_text="abc")
        assert agent.common_prefix_chars("xyz") == 0

    def test_whitespace_differences(self) -> None:
        agent = _make_agent(prompt_text="hello world")
        assert agent.common_prefix_chars("hello\tworld") == 5

    def test_newline_handling(self) -> None:
        agent = _make_agent(prompt_text="line1\nline2")
        assert agent.common_prefix_chars("line1\nline2") == 11
        assert agent.common_prefix_chars("line1\rline2") == 5


class TestTokenSequenceRoundTrip:
    def test_stored_sequence_preserved(self) -> None:
        """Token sequence stored in AgentBlocks is retrievable unchanged."""
        seq = [101, 202, 303, 404, 505]
        agent = _make_agent(token_sequence=seq)
        assert agent.token_sequence == [101, 202, 303, 404, 505]

    def test_prompt_text_preserved(self) -> None:
        text = "Hello, how are you today?"
        agent = _make_agent(prompt_text=text)
        assert agent.prompt_text == "Hello, how are you today?"

    def test_prefix_match_after_construction(self) -> None:
        """Prefix matching works immediately after construction."""
        agent = _make_agent(
            token_sequence=[1, 2, 3, 4, 5],
            prompt_text="Hello world",
        )
        assert agent.common_prefix_length([1, 2, 3]) == 3
        assert agent.common_prefix_chars("Hello") == 5
