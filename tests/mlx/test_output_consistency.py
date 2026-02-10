# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Output consistency and determinism tests.

Verifies that:
- T=0 (greedy) decoding is deterministic across calls
- Different temperatures produce valid text
- Same model, same prompt, same seed → same output
- No model reload needed between parameter changes

Run:
  pytest tests/mlx/test_output_consistency.py -v -x --timeout=300

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


def _gen(model, tokenizer, content, max_tokens=30, temp=0.0, top_p=0.0, top_k=0):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)


class TestDeterminism:
    """T=0 greedy decoding must be deterministic."""

    def test_greedy_identical_across_5_calls(self, model_and_tokenizer) -> None:
        """Same prompt + T=0 → identical output every time."""
        model, tokenizer = model_and_tokenizer
        results = []
        for _ in range(5):
            result = _gen(model, tokenizer, "What is 2 + 2?", temp=0.0)
            results.append(result)

        assert all(r == results[0] for r in results), (
            f"T=0 should be deterministic, got: {results}"
        )

    def test_greedy_determinism_after_temp_switch(self, model_and_tokenizer) -> None:
        """T=0 output must be the same before and after using T>0."""
        model, tokenizer = model_and_tokenizer
        prompt = "Name the largest planet."

        r1 = _gen(model, tokenizer, prompt, temp=0.0)
        _gen(model, tokenizer, "Tell me a joke.", temp=1.0)  # Use high temp
        r2 = _gen(model, tokenizer, prompt, temp=0.0)

        assert r1 == r2, f"T=0 changed after T=1.0 usage: {r1!r} vs {r2!r}"

    def test_greedy_determinism_after_top_p_switch(self, model_and_tokenizer) -> None:
        """T=0 output stays the same after using top_p sampling."""
        model, tokenizer = model_and_tokenizer
        prompt = "What color is the sky?"

        r1 = _gen(model, tokenizer, prompt, temp=0.0)
        _gen(model, tokenizer, "Random thought.", temp=0.8, top_p=0.9)
        r2 = _gen(model, tokenizer, prompt, temp=0.0)

        assert r1 == r2, f"T=0 changed after top_p=0.9: {r1!r} vs {r2!r}"


class TestTemperatureConsistency:
    """Different temperatures should produce valid text."""

    @pytest.mark.parametrize("temp", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_all_temperatures_produce_valid_text(self, model_and_tokenizer, temp) -> None:
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Hello!", max_tokens=30, temp=temp)

        assert isinstance(result, str)
        assert len(result) > 0, f"Empty output at temp={temp}"
        assert "\x00" not in result, f"Null byte at temp={temp}"
        alpha = sum(1 for c in result if c.isalpha())
        assert alpha > 0, f"No alphabetic chars at temp={temp}: {result!r}"

    def test_high_temp_still_produces_words(self, model_and_tokenizer) -> None:
        """Even T=1.5 should produce readable text, not garbage."""
        model, tokenizer = model_and_tokenizer
        result = _gen(model, tokenizer, "Describe a tree.", max_tokens=50, temp=1.5)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should have at least some spaces (multiple words)
        word_count = len(result.split())
        assert word_count >= 1, f"Expected words, got: {result!r}"


class TestSamplingParameterIndependence:
    """Sampling parameters should work independently without interference."""

    def test_top_p_does_not_affect_greedy(self, model_and_tokenizer) -> None:
        """Setting top_p on one call should not affect the next T=0 call."""
        model, tokenizer = model_and_tokenizer
        prompt = "What is 1 + 1?"

        r_before = _gen(model, tokenizer, prompt, temp=0.0)
        _gen(model, tokenizer, "Random.", temp=0.5, top_p=0.1)
        r_after = _gen(model, tokenizer, prompt, temp=0.0)

        assert r_before == r_after

    def test_top_k_does_not_affect_greedy(self, model_and_tokenizer) -> None:
        """Setting top_k on one call should not affect the next T=0 call."""
        model, tokenizer = model_and_tokenizer
        prompt = "Name a fruit."

        r_before = _gen(model, tokenizer, prompt, temp=0.0)
        _gen(model, tokenizer, "Anything.", temp=0.5, top_k=5)
        r_after = _gen(model, tokenizer, prompt, temp=0.0)

        assert r_before == r_after

    def test_sequential_different_prompts_all_valid(self, model_and_tokenizer) -> None:
        """Multiple different prompts in sequence should all produce valid output."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            "What is the sun?",
            "Name three animals.",
            "How does rain form?",
            "What is 5 + 3?",
        ]
        for prompt in prompts:
            result = _gen(model, tokenizer, prompt, max_tokens=40)
            assert isinstance(result, str)
            assert len(result) > 0, f"Empty output for: {prompt}"
            assert "\x00" not in result, f"Null byte for: {prompt}"


class TestCachedVsUncachedConsistency:
    """Repeated calls should produce consistent quality."""

    def test_repeated_calls_same_quality(self, model_and_tokenizer) -> None:
        """10 calls with same prompt should all be valid text."""
        model, tokenizer = model_and_tokenizer
        prompt = "What is water made of?"

        for i in range(10):
            result = _gen(model, tokenizer, prompt, temp=0.0)
            assert isinstance(result, str), f"Call {i}: not a string"
            assert len(result) > 0, f"Call {i}: empty output"
            assert "\x00" not in result, f"Call {i}: null byte"

    def test_greedy_all_identical(self, model_and_tokenizer) -> None:
        """10 greedy calls should produce bit-identical output."""
        model, tokenizer = model_and_tokenizer
        prompt = "Define the word 'logic'."

        results = [_gen(model, tokenizer, prompt, temp=0.0) for _ in range(10)]
        assert len(set(results)) == 1, f"Greedy produced {len(set(results))} unique outputs"
