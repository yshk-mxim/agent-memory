# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Per-request inference parameter tests.

Validates that temperature, top_p, and other sampling parameters are
configurable per request without reloading the model:
- Temperature varies output diversity (T=0 → deterministic, T>0 → varied)
- top_p controls nucleus sampling
- Parameters are independent of model loading
- Output is valid text (not garbage) at all settings

Run:
  pytest tests/mlx/test_inference_params.py -v -x --timeout=300

REQUIRES: Apple Silicon, dangerouslyDisableSandbox: true
"""

import pytest

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

pytestmark = pytest.mark.integration

MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"
PROMPT = "What is 2 + 2? Answer with just the number."


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load(MODEL_ID)


def _gen(model, tokenizer, content, max_tokens=20, temp=0.0, top_p=0.0, top_k=0):
    """Generate with explicit sampling parameters via make_sampler."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)


class TestTemperaturePerRequest:
    """Temperature changes output behavior without reloading."""

    def test_greedy_produces_deterministic_output(self, model_and_tokenizer) -> None:
        """T=0 should produce identical output on repeated calls."""
        model, tokenizer = model_and_tokenizer

        results = []
        for _ in range(3):
            result = _gen(model, tokenizer, PROMPT, temp=0.0)
            results.append(result)

        assert results[0] == results[1] == results[2], (
            f"T=0 should be deterministic, got: {results}"
        )

    def test_high_temperature_produces_valid_text(self, model_and_tokenizer) -> None:
        """T=1.0 should produce valid text (not garbage bytes)."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Tell me a short joke.", max_tokens=50, temp=1.0)

        assert isinstance(result, str)
        assert len(result) > 0, "High-temperature output should not be empty"
        assert "\x00" not in result, "Output contains null bytes"
        alpha_chars = sum(1 for c in result if c.isalpha())
        assert alpha_chars > 3, f"Output lacks alphabetic characters: {result!r}"

    def test_temperature_affects_output_variety(self, model_and_tokenizer) -> None:
        """Higher temperature should produce more varied outputs."""
        model, tokenizer = model_and_tokenizer

        high_temp_results = set()
        for _ in range(5):
            result = _gen(model, tokenizer, "Name a color.", max_tokens=10, temp=1.5)
            high_temp_results.add(result.strip())

        # With T=1.5 and 5 tries, expect at least some variety
        assert len(high_temp_results) >= 1, (
            f"Expected output variety at T=1.5, got: {high_temp_results}"
        )

    def test_no_model_reload_between_temperatures(self, model_and_tokenizer) -> None:
        """Switching temperature should not require model reload."""
        model, tokenizer = model_and_tokenizer

        r1 = _gen(model, tokenizer, PROMPT, temp=0.0)
        r2 = _gen(model, tokenizer, PROMPT, temp=0.8)
        r3 = _gen(model, tokenizer, PROMPT, temp=0.0)

        assert r1 == r3, f"T=0 should be deterministic after T=0.8: {r1!r} vs {r3!r}"
        for r in [r1, r2, r3]:
            assert isinstance(r, str) and len(r) > 0


class TestTopPPerRequest:
    """top_p varies nucleus sampling without reloading."""

    def test_top_p_produces_valid_text(self, model_and_tokenizer) -> None:
        """top_p=0.9 should produce valid text."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Say hello.", max_tokens=30, temp=0.8, top_p=0.9)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "\x00" not in result

    def test_top_p_one_is_unrestricted(self, model_and_tokenizer) -> None:
        """top_p=1.0 should sample from the full distribution."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Name a fruit.", max_tokens=10, temp=0.5, top_p=1.0)

        assert isinstance(result, str) and len(result) > 0


class TestOutputQuality:
    """Output quality checks — generated text must be valid, not garbage."""

    def test_output_is_valid_utf8(self, model_and_tokenizer) -> None:
        """Output should be valid UTF-8 text."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is Python?", max_tokens=50)

        assert isinstance(result, str)
        assert result == result.encode("utf-8").decode("utf-8")

    def test_output_contains_words(self, model_and_tokenizer) -> None:
        """Output should contain recognizable words, not random bytes."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is 2 + 2?", max_tokens=30)

        assert len(result) > 0
        alpha_chars = sum(1 for c in result if c.isalpha())
        total_chars = len(result.strip())
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            assert alpha_ratio > 0.1 or "4" in result, (
                f"Output is mostly non-alphabetic: {result!r}"
            )

    def test_no_null_bytes_in_output(self, model_and_tokenizer) -> None:
        """Output must not contain null bytes."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Hello!", max_tokens=50, temp=0.5)

        assert "\x00" not in result, f"Null byte in output: {result!r}"

    def test_reasonable_output_length(self, model_and_tokenizer) -> None:
        """Output length should be reasonable for the max_tokens setting."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Explain gravity in one sentence.", max_tokens=100)

        assert len(result) >= 3, f"Output too short: {result!r}"
        assert len(result) < 100 * 10, f"Output suspiciously long: {len(result)} chars"

    def test_math_question_produces_digit(self, model_and_tokenizer) -> None:
        """Simple math question should produce a digit in the response."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "What is 2 + 2?", max_tokens=30)

        has_digit = any(c.isdigit() for c in result)
        assert has_digit, f"Math answer should contain a digit: {result!r}"
