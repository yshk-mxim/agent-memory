# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Real MLX Q4 cache round-trip tests.

NO MOCKING. Tests actual Q4 quantization, safetensors persistence,
and cache reconstruction with real MLX tensors on Metal GPU.

Run: pytest tests/mlx/test_real_cache_round_trip.py -v -x --timeout=180
REQUIRES: Apple Silicon, dangerouslyDisableSandbox: true
"""

import numpy as np
import pytest

import mlx.core as mx

from agent_memory.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from agent_memory.domain.entities import AgentBlocks, KVBlock

pytestmark = pytest.mark.integration

# Q4 quantization parameters matching production config
KV_BITS = 4
KV_GROUP_SIZE = 64


def _quantize_tensor(data: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a float tensor to Q4 using real MLX quantize."""
    w, s, b = mx.quantize(data, group_size=KV_GROUP_SIZE, bits=KV_BITS)
    mx.eval(w, s, b)
    return w, s, b


def _build_real_q4_blocks(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    agent_id: str = "test_agent",
    prompt_text: str = "Hello, how are you today?",
    token_sequence: list[int] | None = None,
) -> AgentBlocks:
    """Build AgentBlocks with REAL MLX Q4 quantized tensors."""
    blocks: dict[int, list[KVBlock]] = {}
    tok_seq = token_sequence or list(range(1, seq_len + 1))

    for layer_id in range(n_layers):
        # Create real float16 KV data (simulating model output)
        k_float = mx.random.normal((1, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
        v_float = mx.random.normal((1, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
        mx.eval(k_float, v_float)

        # Quantize to Q4 — this is what _extract_cache does in production
        k_quant = _quantize_tensor(k_float)
        v_quant = _quantize_tensor(v_float)

        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=seq_len,
            layer_data={"k": k_quant, "v": v_quant},
        )
        blocks[layer_id] = [block]

    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks,
        total_tokens=seq_len,
        token_sequence=tok_seq,
        prompt_text=prompt_text,
    )


class TestQ4FormatPreservation:
    """Phase 3.0c: Q4 format survives save → load → verify."""

    def test_quantized_dtypes_correct(self, real_spec, cache_dir) -> None:
        """Freshly quantized tensors have correct Q4 dtypes."""
        data = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        mx.eval(data)
        w, s, b = _quantize_tensor(data)

        assert w.dtype == mx.uint32, f"Weights should be uint32, got {w.dtype}"
        assert s.dtype == mx.float16, f"Scales should be float16, got {s.dtype}"
        assert b.dtype == mx.float16, f"Biases should be float16, got {b.dtype}"

    def test_q4_round_trip_through_safetensors(self, real_spec, cache_dir) -> None:
        """Q4 data saved to safetensors and loaded back must be bit-identical."""
        adapter = SafetensorsCacheAdapter(cache_dir)

        # Build blocks with real Q4 data (2 layers for speed)
        original = _build_real_q4_blocks(
            n_layers=2, n_kv_heads=4, head_dim=64, seq_len=32,
        )

        # Capture original numpy arrays (before save)
        original_arrays: dict[str, np.ndarray] = {}
        for layer_id, layer_blocks in original.blocks.items():
            block = layer_blocks[0]
            k_w, k_s, k_b = block.layer_data["k"]
            v_w, v_s, v_b = block.layer_data["v"]
            original_arrays[f"L{layer_id}_K_w"] = np.array(k_w)
            original_arrays[f"L{layer_id}_K_s"] = np.array(k_s)
            original_arrays[f"L{layer_id}_K_b"] = np.array(k_b)
            original_arrays[f"L{layer_id}_V_w"] = np.array(v_w)
            original_arrays[f"L{layer_id}_V_s"] = np.array(v_s)
            original_arrays[f"L{layer_id}_V_b"] = np.array(v_b)

        # Save to disk
        metadata = {
            "model_id": "test-model",
            "n_layers": "2",
            "token_sequence": str(original.token_sequence),
            "prompt_text": original.prompt_text,
        }
        path = adapter.save("test_agent", original, metadata)
        assert path.exists(), "Safetensors file should exist"

        # Load from disk
        loaded_blocks, loaded_metadata = adapter.load(path)

        # Verify loaded data is STILL quantized tuples
        for layer_id in range(2):
            assert layer_id in loaded_blocks, f"Layer {layer_id} should be present"
            block = loaded_blocks[layer_id][0]
            k_data = block.layer_data["k"]
            v_data = block.layer_data["v"]

            # Must be 3-tuples (quantized format)
            assert isinstance(k_data, tuple), f"K data should be tuple, got {type(k_data)}"
            assert len(k_data) == 3, f"K tuple should have 3 elements, got {len(k_data)}"
            assert isinstance(v_data, tuple), f"V data should be tuple, got {type(v_data)}"
            assert len(v_data) == 3, f"V tuple should have 3 elements, got {len(v_data)}"

            # Verify dtypes preserved (NO FP16 conversion happened)
            k_w, k_s, k_b = k_data
            assert k_w.dtype == mx.uint32, f"Loaded K weights should be uint32, got {k_w.dtype}"
            assert k_s.dtype == mx.float16, f"Loaded K scales should be float16, got {k_s.dtype}"
            assert k_b.dtype == mx.float16, f"Loaded K biases should be float16, got {k_b.dtype}"

            v_w, v_s, v_b = v_data
            assert v_w.dtype == mx.uint32, f"Loaded V weights should be uint32, got {v_w.dtype}"
            assert v_s.dtype == mx.float16, f"Loaded V scales should be float16, got {v_s.dtype}"
            assert v_b.dtype == mx.float16, f"Loaded V biases should be float16, got {v_b.dtype}"

            # BIT-IDENTICAL comparison
            loaded_k_w = np.array(k_w)
            loaded_k_s = np.array(k_s)
            loaded_k_b = np.array(k_b)
            loaded_v_w = np.array(v_w)
            loaded_v_s = np.array(v_s)
            loaded_v_b = np.array(v_b)

            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_w"], loaded_k_w,
                err_msg=f"Layer {layer_id} K weights not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_s"], loaded_k_s,
                err_msg=f"Layer {layer_id} K scales not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_b"], loaded_k_b,
                err_msg=f"Layer {layer_id} K biases not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_w"], loaded_v_w,
                err_msg=f"Layer {layer_id} V weights not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_s"], loaded_v_s,
                err_msg=f"Layer {layer_id} V scales not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_b"], loaded_v_b,
                err_msg=f"Layer {layer_id} V biases not bit-identical",
            )

    def test_dequantized_values_match_after_round_trip(self, real_spec, cache_dir) -> None:
        """Dequantized output from original vs loaded Q4 must be identical."""
        adapter = SafetensorsCacheAdapter(cache_dir)

        # Create and quantize
        original_float = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        mx.eval(original_float)
        w, s, b = _quantize_tensor(original_float)

        # Dequantize original
        original_dequant = mx.dequantize(w, s, b, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        mx.eval(original_dequant)
        original_np = np.array(original_dequant)

        # Save → load
        block = KVBlock(
            block_id=0, layer_id=0, token_count=16,
            layer_data={"k": (w, s, b), "v": (w, s, b)},
        )
        blocks = AgentBlocks(
            agent_id="test", blocks={0: [block]}, total_tokens=16,
            token_sequence=[1, 2, 3], prompt_text="test",
        )
        path = adapter.save("test", blocks, {"model_id": "test"})
        loaded_blocks, _ = adapter.load(path)

        # Dequantize loaded
        k_loaded = loaded_blocks[0][0].layer_data["k"]
        loaded_dequant = mx.dequantize(
            k_loaded[0], k_loaded[1], k_loaded[2],
            group_size=KV_GROUP_SIZE, bits=KV_BITS,
        )
        mx.eval(loaded_dequant)
        loaded_np = np.array(loaded_dequant)

        # Must be IDENTICAL (deterministic Q4 → FP16 conversion)
        np.testing.assert_array_equal(
            original_np, loaded_np,
            err_msg="Dequantized values differ after round-trip",
        )

    def test_double_round_trip_no_accumulation_error(self, real_spec, cache_dir) -> None:
        """Save→load→save→load must produce identical Q4 data (no double quantization)."""
        adapter = SafetensorsCacheAdapter(cache_dir)

        original = _build_real_q4_blocks(
            n_layers=1, n_kv_heads=4, head_dim=64, seq_len=16,
        )

        # First round-trip
        path1 = adapter.save("agent_r1", original, {"model_id": "test"})
        loaded1_blocks, _ = adapter.load(path1)

        # Reconstruct AgentBlocks from loaded data for second save
        loaded1_agent = AgentBlocks(
            agent_id="agent_r2",
            blocks=loaded1_blocks,
            total_tokens=16,
            token_sequence=[1, 2, 3],
            prompt_text="test",
        )

        # Second round-trip
        path2 = adapter.save("agent_r2", loaded1_agent, {"model_id": "test"})
        loaded2_blocks, _ = adapter.load(path2)

        # Compare first and second load — must be identical
        for layer_id in loaded1_blocks:
            b1 = loaded1_blocks[layer_id][0]
            b2 = loaded2_blocks[layer_id][0]

            k1_w = np.array(b1.layer_data["k"][0])
            k2_w = np.array(b2.layer_data["k"][0])
            np.testing.assert_array_equal(
                k1_w, k2_w,
                err_msg=f"Layer {layer_id}: K weights differ after double round-trip",
            )

            v1_s = np.array(b1.layer_data["v"][1])
            v2_s = np.array(b2.layer_data["v"][1])
            np.testing.assert_array_equal(
                v1_s, v2_s,
                err_msg=f"Layer {layer_id}: V scales differ after double round-trip",
            )


def _build_bfloat16_q4_blocks(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> AgentBlocks:
    """Build AgentBlocks with Q4 quantized tensors from bfloat16 source data.

    Simulates models like Gemma 3 that use bfloat16 internally, producing
    bfloat16 scales/biases after mx.quantize().
    """
    blocks: dict[int, list[KVBlock]] = {}

    for layer_id in range(n_layers):
        # bfloat16 source data (as Gemma 3 would produce)
        k_float = mx.random.normal((1, n_kv_heads, seq_len, head_dim)).astype(mx.bfloat16)
        v_float = mx.random.normal((1, n_kv_heads, seq_len, head_dim)).astype(mx.bfloat16)
        mx.eval(k_float, v_float)

        k_quant = _quantize_tensor(k_float)
        v_quant = _quantize_tensor(v_float)

        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=seq_len,
            layer_data={"k": k_quant, "v": v_quant},
        )
        blocks[layer_id] = [block]

    return AgentBlocks(
        agent_id="bf16_test",
        blocks=blocks,
        total_tokens=seq_len,
        token_sequence=list(range(1, seq_len + 1)),
        prompt_text="bfloat16 test",
    )


class TestBfloat16Q4Preservation:
    """Verify bfloat16 scales/biases survive save → load round-trip."""

    def test_bfloat16_quantized_dtypes(self, cache_dir) -> None:
        """Quantizing bfloat16 data produces bfloat16 scales/biases."""
        data = mx.random.normal((1, 4, 32, 64)).astype(mx.bfloat16)
        mx.eval(data)
        w, s, b = _quantize_tensor(data)

        assert w.dtype == mx.uint32, f"Weights should be uint32, got {w.dtype}"
        assert s.dtype == mx.bfloat16, f"Scales should be bfloat16, got {s.dtype}"
        assert b.dtype == mx.bfloat16, f"Biases should be bfloat16, got {b.dtype}"

    def test_bfloat16_round_trip_through_safetensors(self, cache_dir) -> None:
        """bfloat16 Q4 data saved to safetensors and loaded back must be bit-identical."""
        adapter = SafetensorsCacheAdapter(cache_dir)

        original = _build_bfloat16_q4_blocks(
            n_layers=2, n_kv_heads=4, head_dim=64, seq_len=32,
        )

        # Capture original arrays (before save) via float32 for bfloat16 comparison
        original_arrays: dict[str, np.ndarray] = {}
        for layer_id, layer_blocks in original.blocks.items():
            block = layer_blocks[0]
            k_w, k_s, k_b = block.layer_data["k"]
            v_w, v_s, v_b = block.layer_data["v"]

            # Verify source scales/biases are bfloat16
            assert k_s.dtype == mx.bfloat16, f"Source K scales should be bfloat16"
            assert k_b.dtype == mx.bfloat16, f"Source K biases should be bfloat16"

            original_arrays[f"L{layer_id}_K_w"] = np.array(k_w)
            # Compare bfloat16 via float32 (numpy doesn't have native bfloat16)
            original_arrays[f"L{layer_id}_K_s"] = np.array(k_s.astype(mx.float32))
            original_arrays[f"L{layer_id}_K_b"] = np.array(k_b.astype(mx.float32))
            original_arrays[f"L{layer_id}_V_w"] = np.array(v_w)
            original_arrays[f"L{layer_id}_V_s"] = np.array(v_s.astype(mx.float32))
            original_arrays[f"L{layer_id}_V_b"] = np.array(v_b.astype(mx.float32))

        # Save to disk
        metadata = {"model_id": "test-bf16-model", "n_layers": "2"}
        path = adapter.save("bf16_test", original, metadata)
        assert path.exists()

        # Load from disk
        loaded_blocks, _ = adapter.load(path)

        for layer_id in range(2):
            assert layer_id in loaded_blocks
            block = loaded_blocks[layer_id][0]
            k_data = block.layer_data["k"]
            v_data = block.layer_data["v"]

            assert isinstance(k_data, tuple) and len(k_data) == 3
            assert isinstance(v_data, tuple) and len(v_data) == 3

            k_w, k_s, k_b = k_data
            v_w, v_s, v_b = v_data

            # CRITICAL: scales/biases must come back as bfloat16, NOT float16
            assert k_w.dtype == mx.uint32, f"L{layer_id} K weights: {k_w.dtype}"
            assert k_s.dtype == mx.bfloat16, f"L{layer_id} K scales: {k_s.dtype} (want bfloat16)"
            assert k_b.dtype == mx.bfloat16, f"L{layer_id} K biases: {k_b.dtype} (want bfloat16)"
            assert v_w.dtype == mx.uint32, f"L{layer_id} V weights: {v_w.dtype}"
            assert v_s.dtype == mx.bfloat16, f"L{layer_id} V scales: {v_s.dtype} (want bfloat16)"
            assert v_b.dtype == mx.bfloat16, f"L{layer_id} V biases: {v_b.dtype} (want bfloat16)"

            # Bit-identical comparison (via float32 for bfloat16 arrays)
            loaded_k_w = np.array(k_w)
            loaded_k_s = np.array(k_s.astype(mx.float32))
            loaded_k_b = np.array(k_b.astype(mx.float32))
            loaded_v_w = np.array(v_w)
            loaded_v_s = np.array(v_s.astype(mx.float32))
            loaded_v_b = np.array(v_b.astype(mx.float32))

            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_w"], loaded_k_w,
                err_msg=f"Layer {layer_id} K weights not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_s"], loaded_k_s,
                err_msg=f"Layer {layer_id} K scales not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_K_b"], loaded_k_b,
                err_msg=f"Layer {layer_id} K biases not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_w"], loaded_v_w,
                err_msg=f"Layer {layer_id} V weights not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_s"], loaded_v_s,
                err_msg=f"Layer {layer_id} V scales not bit-identical",
            )
            np.testing.assert_array_equal(
                original_arrays[f"L{layer_id}_V_b"], loaded_v_b,
                err_msg=f"Layer {layer_id} V biases not bit-identical",
            )

    def test_bfloat16_dequantized_values_match(self, cache_dir) -> None:
        """Dequantized output from original vs loaded bfloat16 Q4 must be identical."""
        adapter = SafetensorsCacheAdapter(cache_dir)

        # Create bfloat16 source and quantize
        original_float = mx.random.normal((1, 4, 16, 64)).astype(mx.bfloat16)
        mx.eval(original_float)
        w, s, b = _quantize_tensor(original_float)

        # Dequantize original
        original_dequant = mx.dequantize(w, s, b, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        mx.eval(original_dequant)
        original_np = np.array(original_dequant.astype(mx.float32))

        # Save → load
        block = KVBlock(
            block_id=0, layer_id=0, token_count=16,
            layer_data={"k": (w, s, b), "v": (w, s, b)},
        )
        blocks = AgentBlocks(
            agent_id="bf16_dequant", blocks={0: [block]}, total_tokens=16,
            token_sequence=[1, 2, 3], prompt_text="test",
        )
        path = adapter.save("bf16_dequant", blocks, {"model_id": "test"})
        loaded_blocks, _ = adapter.load(path)

        # Dequantize loaded
        k_loaded = loaded_blocks[0][0].layer_data["k"]
        loaded_dequant = mx.dequantize(
            k_loaded[0], k_loaded[1], k_loaded[2],
            group_size=KV_GROUP_SIZE, bits=KV_BITS,
        )
        mx.eval(loaded_dequant)
        loaded_np = np.array(loaded_dequant.astype(mx.float32))

        np.testing.assert_array_equal(
            original_np, loaded_np,
            err_msg="bfloat16 dequantized values differ after round-trip",
        )


def _chat_prompt(tokenizer, user_message: str) -> str:
    """Format a user message with the model's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_message


class TestRealModelInference:
    """Phase 5.1: Real model generates meaningful text."""

    def test_model_generates_text(self, real_model_and_tokenizer) -> None:
        """SmolLM2-135M generates non-empty valid UTF-8 text."""
        from mlx_lm import generate

        model, tokenizer = real_model_and_tokenizer
        prompt = _chat_prompt(tokenizer, "What is 2 + 2?")
        result = generate(model, tokenizer, prompt=prompt, max_tokens=50)

        assert isinstance(result, str), f"Expected string, got {type(result)}"
        assert len(result) > 0, "Generated text should be non-empty"
        result.encode("utf-8")
        assert "\x00" not in result, f"Generated text contains null bytes: {result!r}"

    def test_generated_text_is_coherent(self, real_model_and_tokenizer) -> None:
        """Generated text should contain alphabetic words, not raw tensor bytes."""
        from mlx_lm import generate

        model, tokenizer = real_model_and_tokenizer
        prompt = _chat_prompt(tokenizer, "List three colors:")
        result = generate(model, tokenizer, prompt=prompt, max_tokens=100)

        alpha_chars = sum(1 for c in result if c.isalpha())
        assert alpha_chars > 5, (
            f"Generated text has only {alpha_chars} alphabetic chars, "
            f"likely garbage: {result!r}"
        )
        assert len(result) > 5, f"Response too short: {result!r}"
        assert len(result) < 5000, f"Response unreasonably long: {len(result)} chars"


class TestRealBatchEngineEndToEnd:
    """End-to-end batch engine test with real model."""

    def test_submit_and_step_produces_completion(
        self, real_model_and_tokenizer, real_spec
    ) -> None:
        """Full engine pipeline: submit → step → completion with real tokens."""
        from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
        from agent_memory.application.batch_engine import BlockPoolBatchEngine
        from agent_memory.domain.services import BlockPool

        model, tokenizer = real_model_and_tokenizer
        pool = BlockPool(spec=real_spec, total_blocks=200)
        cache_adapter = MLXCacheAdapter()

        engine = BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=real_spec,
            cache_adapter=cache_adapter,
        )

        uid = engine.submit(agent_id="real_agent", prompt="Hello", max_tokens=20)

        completions = list(engine.step())

        assert len(completions) == 1, f"Expected 1 completion, got {len(completions)}"
        c = completions[0]
        assert c.uid == uid
        assert len(c.text) > 0, f"Empty text from real model"
        assert c.finish_reason in ("stop", "length")
        assert c.token_count > 0
        assert c.blocks is not None
        assert c.blocks.total_tokens > 0

    def test_extracted_cache_is_q4_quantized(
        self, real_model_and_tokenizer, real_spec
    ) -> None:
        """Cache extracted after real inference should be Q4 quantized."""
        from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
        from agent_memory.application.batch_engine import BlockPoolBatchEngine
        from agent_memory.domain.services import BlockPool

        model, tokenizer = real_model_and_tokenizer

        # Use Q4 spec
        q4_spec = real_spec
        if q4_spec.kv_bits is None:
            pytest.skip("Model spec has kv_bits=None, no quantization")

        pool = BlockPool(spec=q4_spec, total_blocks=200)
        cache_adapter = MLXCacheAdapter()

        engine = BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=q4_spec,
            cache_adapter=cache_adapter,
        )

        engine.submit(agent_id="q4_agent", prompt="Hello world", max_tokens=10)
        completions = list(engine.step())

        assert len(completions) == 1
        agent_blocks = completions[0].blocks

        # Verify at least one layer has Q4 data
        found_q4 = False
        float_dtypes = {mx.float16, mx.bfloat16}  # Model may use either
        for layer_id, layer_blocks in agent_blocks.blocks.items():
            for block in layer_blocks:
                if block.layer_data is None:
                    continue
                k_data = block.layer_data.get("k")
                if k_data is None:
                    continue

                if isinstance(k_data, tuple) and len(k_data) == 3:
                    found_q4 = True
                    w, s, b = k_data
                    assert w.dtype == mx.uint32, (
                        f"Layer {layer_id}: K weights should be uint32, got {w.dtype}"
                    )
                    assert s.dtype in float_dtypes, (
                        f"Layer {layer_id}: K scales should be float16/bfloat16, got {s.dtype}"
                    )
                    break
            if found_q4:
                break

        assert found_q4, (
            "No Q4 quantized cache found after inference. "
            "Cache extraction may not be quantizing correctly."
        )

    def test_cache_persist_and_reload(
        self, real_model_and_tokenizer, real_spec, cache_dir
    ) -> None:
        """Full pipeline: infer → extract Q4 → save to disk → load → verify."""
        from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
        from agent_memory.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
        from agent_memory.application.batch_engine import BlockPoolBatchEngine
        from agent_memory.domain.services import BlockPool

        model, tokenizer = real_model_and_tokenizer
        pool = BlockPool(spec=real_spec, total_blocks=200)
        cache_adapter = MLXCacheAdapter()
        disk_adapter = SafetensorsCacheAdapter(cache_dir)

        engine = BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=real_spec,
            cache_adapter=cache_adapter,
        )

        # Generate with real model
        engine.submit(agent_id="persist_agent", prompt="Hello", max_tokens=10)
        completions = list(engine.step())
        assert len(completions) == 1

        original_blocks = completions[0].blocks

        # Save to disk
        metadata = {
            "model_id": "SmolLM2-135M",
            "n_layers": str(real_spec.n_layers),
        }
        path = disk_adapter.save("persist_agent", original_blocks, metadata)
        assert path.exists()

        # Load from disk
        loaded_blocks, loaded_metadata = disk_adapter.load(path)

        # Verify structure preserved
        assert len(loaded_blocks) > 0, "No layers loaded"
        assert loaded_metadata["model_id"] == "SmolLM2-135M"

        # Verify Q4 format preserved
        float_dtypes = {mx.float16, mx.bfloat16}
        for layer_id in loaded_blocks:
            block = loaded_blocks[layer_id][0]
            k_data = block.layer_data["k"]
            v_data = block.layer_data["v"]

            if isinstance(k_data, tuple) and len(k_data) == 3:
                w, s, b = k_data
                assert w.dtype == mx.uint32, f"Layer {layer_id}: loaded K weights not uint32"
                assert s.dtype in float_dtypes, (
                    f"Layer {layer_id}: loaded K scales should be float16/bfloat16, got {s.dtype}"
                )
