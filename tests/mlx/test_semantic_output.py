"""Semantic output quality validation tests.

Verifies that model output is meaningful text, not garbage. Tests run
against a real MLX model (SmolLM2-135M-Instruct) and validate:
- Generated text answers prompts coherently
- Output is valid UTF-8 without null bytes
- Math questions produce digits
- Instruction following works
- Output format is reasonable

Run:
  pytest tests/mlx/test_semantic_output.py -v -x --timeout=300

REQUIRES: Apple Silicon, dangerouslyDisableSandbox: true
"""

import pytest

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

pytestmark = pytest.mark.integration

MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load(MODEL_ID)


def _gen(model, tokenizer, content, max_tokens=50, temp=0.0, top_p=0.0):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temp, top_p=top_p)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)


class TestBasicLanguageCoherence:
    """Output must be valid, readable text — not garbage bytes."""

    def test_math_answer_contains_digit(self, model_and_tokenizer) -> None:
        """'What is 2 + 2?' should produce a digit in the response."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is 2 + 2? Answer with just the number.")

        assert isinstance(result, str)
        has_digit = any(c.isdigit() for c in result)
        assert has_digit, f"Math answer should contain a digit: {result!r}"

    def test_output_is_valid_utf8(self, model_and_tokenizer) -> None:
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is Python?")

        assert isinstance(result, str)
        assert result == result.encode("utf-8").decode("utf-8")

    def test_no_null_bytes(self, model_and_tokenizer) -> None:
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Tell me about the weather.", max_tokens=100)

        assert "\x00" not in result, f"Null byte in output: {result!r}"

    def test_reasonable_length(self, model_and_tokenizer) -> None:
        """Short question should produce a short-to-medium answer."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is gravity?", max_tokens=100)

        assert len(result) >= 3, f"Output too short: {result!r}"
        assert len(result) < 100 * 10, f"Output suspiciously long: {len(result)} chars"

    def test_output_has_alphabetic_content(self, model_and_tokenizer) -> None:
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Say hello.", max_tokens=30)

        alpha_chars = sum(1 for c in result if c.isalpha())
        assert alpha_chars > 3, f"Output lacks alphabetic characters: {result!r}"


class TestInstructionFollowing:
    """Model should follow basic instructions."""

    def test_list_colors_produces_color_words(self, model_and_tokenizer) -> None:
        """Asking for colors should produce color-related words."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "List three colors.", max_tokens=60)

        known_colors = {
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "brown", "gray", "grey", "violet", "cyan",
            "magenta", "indigo", "gold", "silver", "crimson", "scarlet",
        }
        result_lower = result.lower()
        matches = [c for c in known_colors if c in result_lower]
        assert len(matches) >= 1, f"Expected color words in: {result!r}"

    def test_short_answer_is_short(self, model_and_tokenizer) -> None:
        """Asking for a one-word answer should be reasonably short."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Respond with only the word 'yes'.", max_tokens=20)

        assert len(result.strip()) < 100, f"Expected short answer, got {len(result)} chars"
        assert "yes" in result.lower(), f"Expected 'yes' in response: {result!r}"

    def test_not_just_echoing_prompt(self, model_and_tokenizer) -> None:
        """Model should not just repeat the prompt back."""
        model, tokenizer = model_and_tokenizer
        prompt = "What is the capital of France?"
        result = _gen(model, tokenizer, prompt, max_tokens=30)

        assert result.strip() != prompt, "Model just echoed the prompt"
        assert len(result.strip()) > 0, "Empty response"


class TestOutputFormats:
    """Validate that output doesn't contain raw tensor data or garbage."""

    def test_no_raw_float_arrays(self, model_and_tokenizer) -> None:
        """Output should not contain raw floating point array data."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Hello!", max_tokens=30)

        # Raw tensor data looks like "[0.234532, -1.23456, ...]"
        import re

        float_pattern = r"\[[\d\.\-e,\s]{20,}\]"
        match = re.search(float_pattern, result)
        assert match is None, f"Output contains raw float array: {match.group()}"

    def test_no_binary_escape_sequences(self, model_and_tokenizer) -> None:
        """Output should not have binary escape sequences."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "How are you?", max_tokens=30)

        # Check for common binary artifacts
        for bad_char in ["\x00", "\x01", "\x02", "\x03", "\x04", "\x05"]:
            assert bad_char not in result, f"Binary char {ord(bad_char)} in output"

    def test_greedy_produces_words(self, model_and_tokenizer) -> None:
        """T=0 output should contain recognizable English words."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is a cat?", max_tokens=50, temp=0.0)

        common_words = {"the", "a", "is", "an", "it", "and", "of", "to", "in", "for", "cat"}
        result_words = set(result.lower().split())
        matches = result_words & common_words
        assert len(matches) >= 1, (
            f"Expected common English words in: {result!r}"
        )
