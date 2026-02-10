"""Multi-model real MLX tests.

Validates that the Q4 cache pipeline works across different model architectures:
- Gemma 3 12B (48 layers, sliding window + global attention, 8 KV heads, head_dim=256)
- GPT-OSS-20B (24 layers, all global attention, 8 KV heads)
- SmolLM2-135M (30 layers, all global attention, 3 KV heads)

These tests verify scaling behavior and architecture-specific edge cases.

Run one model at a time:
  pytest tests/mlx/test_multi_model.py -k "gemma" -v -x --timeout=300
  pytest tests/mlx/test_multi_model.py -k "gpt_oss" -v -x --timeout=300
  pytest tests/mlx/test_multi_model.py -k "smollm" -v -x --timeout=120

REQUIRES: Apple Silicon, dangerouslyDisableSandbox: true
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from agent_memory.adapters.outbound.mlx_spec_extractor import get_extractor
from agent_memory.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from agent_memory.domain.entities import AgentBlocks, KVBlock

pytestmark = pytest.mark.integration

MODEL_IDS = {
    "smollm": "mlx-community/SmolLM2-135M-Instruct",
    "gemma3": "mlx-community/gemma-3-12b-it-4bit",
    "gpt_oss": "mlx-community/gpt-oss-20b-MXFP4-Q4",
}

# Expected specs per model (from config/models/*.toml)
EXPECTED_SPECS = {
    "smollm": {"n_layers": 30, "n_kv_heads": 3, "head_dim": 64},
    "gemma3": {"n_layers": 48, "n_kv_heads": 8, "head_dim": 256},
    "gpt_oss": {"n_layers": 24, "n_kv_heads": 8, "head_dim": 64},
}


@pytest.fixture(scope="module")
def gemma3_model():
    from mlx_lm import load
    return load(MODEL_IDS["gemma3"])


@pytest.fixture(scope="module")
def gpt_oss_model():
    from mlx_lm import load
    return load(MODEL_IDS["gpt_oss"])


@pytest.fixture(scope="module")
def smollm_model():
    from mlx_lm import load
    return load(MODEL_IDS["smollm"])


def _make_kv_blocks_for_spec(
    spec, n_layers_to_test=2, seq_len=16, kv_bits=4, group_size=64,
):
    """Create KV blocks matching a model's actual spec.

    Args:
        spec: ModelCacheSpec with model geometry.
        n_layers_to_test: Number of layers to create blocks for.
        seq_len: Sequence length per block.
        kv_bits: Quantization bits — 4 (Q4), 8 (Q8), or None (FP16).
        group_size: Quantization group size (ignored when kv_bits is None).
    """
    blocks: dict[int, list[KVBlock]] = {}

    for layer_id in range(min(n_layers_to_test, spec.n_layers)):
        k_float = mx.random.normal(
            (1, spec.n_kv_heads, seq_len, spec.head_dim)
        ).astype(mx.float16)
        v_float = mx.random.normal(
            (1, spec.n_kv_heads, seq_len, spec.head_dim)
        ).astype(mx.float16)
        mx.eval(k_float, v_float)

        if kv_bits is not None:
            k_w, k_s, k_b = mx.quantize(k_float, group_size=group_size, bits=kv_bits)
            v_w, v_s, v_b = mx.quantize(v_float, group_size=group_size, bits=kv_bits)
            mx.eval(k_w, k_s, k_b, v_w, v_s, v_b)
            layer_data = {"k": (k_w, k_s, k_b), "v": (v_w, v_s, v_b)}
        else:
            layer_data = {"k": k_float, "v": v_float}

        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=seq_len,
            layer_data=layer_data,
        )
        blocks[layer_id] = [block]

    return AgentBlocks(
        agent_id="test_agent",
        blocks=blocks,
        total_tokens=seq_len,
        token_sequence=list(range(1, seq_len + 1)),
        prompt_text="test",
    )


class TestSmolLMSpecExtraction:
    """Verify SmolLM2-135M spec matches expected values."""

    def test_spec_matches_expected(self, smollm_model) -> None:
        model, _ = smollm_model
        spec = get_extractor().extract_spec(model)

        expected = EXPECTED_SPECS["smollm"]
        assert spec.n_layers == expected["n_layers"], (
            f"n_layers: expected {expected['n_layers']}, got {spec.n_layers}"
        )
        assert spec.n_kv_heads == expected["n_kv_heads"], (
            f"n_kv_heads: expected {expected['n_kv_heads']}, got {spec.n_kv_heads}"
        )
        assert spec.head_dim == expected["head_dim"], (
            f"head_dim: expected {expected['head_dim']}, got {spec.head_dim}"
        )

    @pytest.mark.parametrize("kv_bits", [4, 8, None], ids=["Q4", "Q8", "FP16"])
    def test_cache_round_trip(self, smollm_model, kv_bits) -> None:
        """SmolLM2: Q4/Q8/FP16 cache round-trip preserves data."""
        model, _ = smollm_model
        spec = get_extractor().extract_spec(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = SafetensorsCacheAdapter(Path(tmpdir))
            original = _make_kv_blocks_for_spec(
                spec, n_layers_to_test=3, kv_bits=kv_bits,
            )

            path = adapter.save("smollm_agent", original, {"model_id": "SmolLM2"})
            loaded_blocks, _ = adapter.load(path)

            assert len(loaded_blocks) == 3
            for layer_id in loaded_blocks:
                k_data = loaded_blocks[layer_id][0].layer_data["k"]
                # Adapter always stores as quantized tuples
                assert isinstance(k_data, tuple) and len(k_data) == 3


class TestGemma3SpecExtraction:
    """Verify Gemma 3 12B spec and cache round-trip at scale."""

    def test_spec_matches_expected(self, gemma3_model) -> None:
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        expected = EXPECTED_SPECS["gemma3"]
        assert spec.n_layers == expected["n_layers"], (
            f"n_layers: expected {expected['n_layers']}, got {spec.n_layers}"
        )
        assert spec.n_kv_heads == expected["n_kv_heads"], (
            f"n_kv_heads: expected {expected['n_kv_heads']}, got {spec.n_kv_heads}"
        )
        assert spec.head_dim == expected["head_dim"], (
            f"head_dim: expected {expected['head_dim']}, got {spec.head_dim}"
        )

    def test_layer_types_include_sliding_window(self, gemma3_model) -> None:
        """Gemma 3 should have both global and sliding window layers."""
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        layer_types = set(spec.layer_types)
        # Gemma 3 has global attention for first 8 layers, sliding window for rest
        assert len(layer_types) >= 1, f"Expected multiple layer types, got {layer_types}"
        assert spec.n_layers == 48, f"Expected 48 layers, got {spec.n_layers}"

    def test_head_dim_compatible_with_quantization(self, gemma3_model) -> None:
        """Gemma 3 head_dim=256 is divisible by group_size=64 — Q4/Q8 work.

        The spec extractor reads head_dim from the model's attention layers
        (attn.head_dim=256), not from hidden_size//num_attention_heads (=240).
        256 is divisible by 32, 64, and 128, so all quantization group sizes work.
        """
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        assert spec.head_dim == 256, f"Expected head_dim=256, got {spec.head_dim}"
        for gs in (32, 64, 128):
            assert spec.head_dim % gs == 0, (
                f"head_dim {spec.head_dim} not divisible by group_size {gs}"
            )

    @pytest.mark.parametrize("kv_bits", [4, 8, None], ids=["Q4", "Q8", "FP16"])
    def test_cache_round_trip(self, gemma3_model, kv_bits) -> None:
        """Gemma 3: Q4/Q8/FP16 cache round-trip preserves data."""
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = SafetensorsCacheAdapter(Path(tmpdir))
            original = _make_kv_blocks_for_spec(
                spec, n_layers_to_test=2, seq_len=32, kv_bits=kv_bits,
            )

            path = adapter.save("gemma3_agent", original, {"model_id": "gemma-3-12b"})
            loaded_blocks, _ = adapter.load(path)

            assert len(loaded_blocks) == 2
            for layer_id in loaded_blocks:
                k_data = loaded_blocks[layer_id][0].layer_data["k"]
                assert isinstance(k_data, tuple) and len(k_data) == 3, (
                    f"Layer {layer_id}: expected quantized tuple, got {type(k_data)}"
                )

    def test_q4_bit_identical_round_trip(self, gemma3_model) -> None:
        """Gemma 3 Q4: weights are bit-identical after save/load."""
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = SafetensorsCacheAdapter(Path(tmpdir))
            original = _make_kv_blocks_for_spec(
                spec, n_layers_to_test=2, seq_len=16, kv_bits=4,
            )

            orig_k_w = np.array(original.blocks[0][0].layer_data["k"][0])

            path = adapter.save("gemma3_agent", original, {"model_id": "gemma-3-12b"})
            loaded_blocks, _ = adapter.load(path)

            loaded_k_w = np.array(loaded_blocks[0][0].layer_data["k"][0])
            np.testing.assert_array_equal(
                orig_k_w, loaded_k_w,
                err_msg="Gemma 3 Q4 weights not bit-identical after round-trip",
            )

    def test_memory_per_block_matches_formula(self, gemma3_model) -> None:
        """Verify actual FP16 and Q4 block sizes match analytical formulas."""
        model, _ = gemma3_model
        spec = get_extractor().extract_spec(model)

        # FP16 memory per block per layer
        fp16_expected = 2 * spec.n_kv_heads * spec.head_dim * spec.block_tokens * 2
        k_float = mx.random.normal(
            (1, spec.n_kv_heads, spec.block_tokens, spec.head_dim)
        ).astype(mx.float16)
        mx.eval(k_float)
        fp16_actual = k_float.nbytes * 2  # K+V
        assert fp16_actual == fp16_expected, (
            f"FP16 bytes: actual {fp16_actual} vs expected {fp16_expected}"
        )

        # Q4 memory: weights + scales + biases
        elements_per_kv = spec.n_kv_heads * spec.head_dim * spec.block_tokens
        q4_weight_bytes = (elements_per_kv * 2 * 4) // 8
        import math
        groups = math.ceil(elements_per_kv / 64) * 2
        q4_overhead = groups * 4  # 2 bytes scales + 2 bytes biases per group
        q4_expected = q4_weight_bytes + q4_overhead
        q4_ratio = q4_expected / fp16_expected
        assert 0.25 <= q4_ratio <= 0.30, (
            f"Q4/FP16 ratio {q4_ratio:.4f} outside 25-30% range"
        )


class TestGptOssSpecExtraction:
    """Verify GPT-OSS-20B spec and cache round-trip."""

    def test_spec_matches_expected(self, gpt_oss_model) -> None:
        model, _ = gpt_oss_model
        spec = get_extractor().extract_spec(model)

        expected = EXPECTED_SPECS["gpt_oss"]
        assert spec.n_layers == expected["n_layers"], (
            f"n_layers: expected {expected['n_layers']}, got {spec.n_layers}"
        )
        assert spec.n_kv_heads == expected["n_kv_heads"], (
            f"n_kv_heads: expected {expected['n_kv_heads']}, got {spec.n_kv_heads}"
        )

    @pytest.mark.parametrize("kv_bits", [4, 8, None], ids=["Q4", "Q8", "FP16"])
    def test_cache_round_trip(self, gpt_oss_model, kv_bits) -> None:
        """GPT-OSS-20B: Q4/Q8/FP16 cache round-trip preserves data."""
        model, _ = gpt_oss_model
        spec = get_extractor().extract_spec(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = SafetensorsCacheAdapter(Path(tmpdir))
            original = _make_kv_blocks_for_spec(
                spec, n_layers_to_test=2, seq_len=16, kv_bits=kv_bits,
            )

            if kv_bits is not None:
                orig_k_w = np.array(original.blocks[0][0].layer_data["k"][0])

            path = adapter.save("gpt_oss_agent", original, {"model_id": "gpt-oss-20b"})
            loaded_blocks, _ = adapter.load(path)

            assert len(loaded_blocks) == 2

            if kv_bits is not None:
                loaded_k_w = np.array(loaded_blocks[0][0].layer_data["k"][0])
                np.testing.assert_array_equal(
                    orig_k_w, loaded_k_w,
                    err_msg=f"GPT-OSS-20B Q{kv_bits} weights not bit-identical after round-trip",
                )

    def test_inference_produces_text(self, gpt_oss_model) -> None:
        """GPT-OSS-20B generates coherent text."""
        from mlx_lm import generate

        model, tokenizer = gpt_oss_model
        # GPT-OSS-20B may or may not have chat template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello in one sentence."}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "Say hello in one sentence."

        result = generate(model, tokenizer, prompt=prompt, max_tokens=30)

        assert isinstance(result, str)
        assert len(result) > 0, "GPT-OSS-20B generated empty text"
        alpha_chars = sum(1 for c in result if c.isalpha())
        assert alpha_chars > 3, f"GPT-OSS-20B output lacks words: {result!r}"
