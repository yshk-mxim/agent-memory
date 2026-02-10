"""Integration tests for cache round-trip validation.

Verifies the full cache lifecycle: create → store → recover → compare.
Tests at multiple levels:
- Safetensors save/load preserves numpy arrays bit-identically
- AgentCacheStore metadata round-trip (token_sequence, prompt_text)
- Q4 quantized format preservation (weights, scales, biases dtype/shape)
- Character-level prefix matching survives round-trip
- Double round-trip catches accumulation errors (e.g., double quantization)
"""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag
from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(n_layers: int = 4) -> ModelCacheSpec:
    return ModelCacheSpec(
        n_layers=n_layers,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * n_layers,
        sliding_window_size=None,
    )


def _make_tag(n_layers: int = 4) -> ModelTag:
    return ModelTag(
        model_id="test-model",
        n_layers=n_layers,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
    )


def _make_q4_numpy_arrays(
    n_kv_heads: int = 4, head_dim: int = 64, seq_len: int = 256, group_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic Q4-format numpy arrays matching MLX quantize output.

    Q4 packing: 8 values per uint32 word (4 bits each).
    Shape: (1, n_kv_heads, seq_len, head_dim // 8) for weights.
    """
    packed_dim = head_dim // 8
    weights = np.random.randint(0, 2**32, size=(1, n_kv_heads, seq_len, packed_dim), dtype=np.uint32)
    n_groups = head_dim // group_size
    scales = np.random.randn(1, n_kv_heads, seq_len, n_groups).astype(np.float16)
    biases = np.random.randn(1, n_kv_heads, seq_len, n_groups).astype(np.float16)
    return weights, scales, biases


def _build_agent_blocks_with_numpy(
    agent_id: str,
    n_layers: int = 4,
    token_sequence: list[int] | None = None,
    prompt_text: str = "",
) -> AgentBlocks:
    """Build AgentBlocks with real numpy Q4 data in layer_data."""
    blocks: dict[int, list[KVBlock]] = {}
    for layer_id in range(n_layers):
        k_w, k_s, k_b = _make_q4_numpy_arrays()
        v_w, v_s, v_b = _make_q4_numpy_arrays()
        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=256,
            layer_data={
                "k": (k_w, k_s, k_b),
                "v": (v_w, v_s, v_b),
            },
        )
        blocks[layer_id] = [block]
    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks,
        total_tokens=256,
        token_sequence=token_sequence or [],
        prompt_text=prompt_text,
    )


class InMemoryAdapter:
    """Minimal adapter that stores blocks+metadata in memory.

    Creates a placeholder file at the returned path so that
    AgentCacheStore._load_from_disk() passes the path.exists() check.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._store: dict[str, tuple] = {}
        self._cache_dir = cache_dir

    def save(self, agent_id: str, blocks: AgentBlocks, metadata: dict) -> Path:
        path = self._cache_dir / f"{agent_id}.safetensors"
        self._store[agent_id] = (blocks, metadata)
        path.write_bytes(b"placeholder")
        return path

    def load(self, path: Path) -> tuple[dict, dict]:
        agent_id = Path(path).stem
        if agent_id not in self._store:
            return {}, {}
        blocks_obj, metadata = self._store[agent_id]
        return blocks_obj.blocks, metadata


# ---------------------------------------------------------------------------
# Test 3.0b: Safetensors numpy round-trip (bit-identical)
# ---------------------------------------------------------------------------

class TestSafetensorsNumpyRoundTrip:
    """Verify safetensors save_file/load_file preserves arrays exactly."""

    def test_uint32_weights_preserved(self) -> None:
        """uint32 packed Q4 weights survive safetensors round-trip."""
        weights = np.random.randint(0, 2**32, size=(1, 4, 256, 8), dtype=np.uint32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file({"weights": weights}, str(path))
            loaded = load_file(str(path))
            np.testing.assert_array_equal(loaded["weights"], weights)
            assert loaded["weights"].dtype == np.uint32

    def test_float16_scales_preserved(self) -> None:
        """float16 scales survive safetensors round-trip."""
        scales = np.random.randn(1, 4, 256, 1).astype(np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file({"scales": scales}, str(path))
            loaded = load_file(str(path))
            np.testing.assert_array_equal(loaded["scales"], scales)
            assert loaded["scales"].dtype == np.float16

    def test_float16_biases_preserved(self) -> None:
        """float16 biases survive safetensors round-trip."""
        biases = np.random.randn(1, 4, 256, 1).astype(np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file({"biases": biases}, str(path))
            loaded = load_file(str(path))
            np.testing.assert_array_equal(loaded["biases"], biases)
            assert loaded["biases"].dtype == np.float16

    def test_multi_tensor_round_trip(self) -> None:
        """Multiple tensors with different dtypes survive single-file round-trip."""
        k_w, k_s, k_b = _make_q4_numpy_arrays()
        v_w, v_s, v_b = _make_q4_numpy_arrays()
        tensors = {
            "L0_B0_K_weights": k_w,
            "L0_B0_K_scales": k_s,
            "L0_B0_K_biases": k_b,
            "L0_B0_V_weights": v_w,
            "L0_B0_V_scales": v_s,
            "L0_B0_V_biases": v_b,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file(tensors, str(path))
            loaded = load_file(str(path))
            for key in tensors:
                np.testing.assert_array_equal(loaded[key], tensors[key])
                assert loaded[key].dtype == tensors[key].dtype

    def test_metadata_round_trip(self) -> None:
        """String metadata in safetensors header is preserved."""
        metadata = {
            "agent_id": "test-agent-42",
            "model_id": "gemma-3-12b-it-4bit",
            "n_layers": "48",
            "token_sequence": json.dumps([101, 202, 303]),
            "prompt_text": "Hello, how are you today?",
        }
        dummy = {"x": np.zeros(1, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file(dummy, str(path), metadata=metadata)
            # Read metadata from safetensors header
            with open(path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size).decode("utf-8"))
            recovered = header.get("__metadata__", {})
            for k, v in metadata.items():
                assert recovered[k] == v, f"Metadata key '{k}' mismatch: {recovered[k]} != {v}"

    def test_double_round_trip_identical(self) -> None:
        """Save → load → save → load produces identical arrays (no accumulation)."""
        k_w, k_s, k_b = _make_q4_numpy_arrays()
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "round1.safetensors"
            path2 = Path(tmpdir) / "round2.safetensors"

            tensors = {"K_weights": k_w, "K_scales": k_s, "K_biases": k_b}

            # Round 1
            save_file(tensors, str(path1))
            loaded1 = load_file(str(path1))

            # Round 2: save loaded data, load again
            save_file(loaded1, str(path2))
            loaded2 = load_file(str(path2))

            for key in tensors:
                np.testing.assert_array_equal(loaded1[key], loaded2[key])
                np.testing.assert_array_equal(loaded2[key], tensors[key])


# ---------------------------------------------------------------------------
# Test 3.0c: Q4 Format Preservation
# ---------------------------------------------------------------------------

class TestQ4FormatPreservation:
    """Verify Q4 quantized data stays Q4 through the safetensors pipeline."""

    def test_weights_dtype_is_uint32(self) -> None:
        """Q4 packed weights must be uint32 after round-trip."""
        k_w, k_s, k_b = _make_q4_numpy_arrays()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q4.safetensors"
            save_file({"w": k_w, "s": k_s, "b": k_b}, str(path))
            loaded = load_file(str(path))
            assert loaded["w"].dtype == np.uint32, f"Expected uint32, got {loaded['w'].dtype}"
            assert loaded["s"].dtype == np.float16, f"Expected float16, got {loaded['s'].dtype}"
            assert loaded["b"].dtype == np.float16, f"Expected float16, got {loaded['b'].dtype}"

    def test_no_fp16_conversion_in_pipeline(self) -> None:
        """Q4 data must NOT be converted to FP16 during save/load.

        If weights come back as float16/float32 instead of uint32,
        the Q4→FP16 conversion happened somewhere in the pipeline.
        """
        weights = np.random.randint(0, 2**32, size=(1, 4, 256, 8), dtype=np.uint32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q4.safetensors"
            save_file({"w": weights}, str(path))
            loaded = load_file(str(path))
            assert loaded["w"].dtype != np.float16, "Q4 weights were converted to FP16!"
            assert loaded["w"].dtype != np.float32, "Q4 weights were converted to FP32!"
            assert loaded["w"].dtype == np.uint32

    def test_shapes_preserved_for_all_components(self) -> None:
        """Q4 component shapes must be preserved through round-trip."""
        k_w, k_s, k_b = _make_q4_numpy_arrays(n_kv_heads=4, head_dim=64, seq_len=128)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q4.safetensors"
            save_file({"w": k_w, "s": k_s, "b": k_b}, str(path))
            loaded = load_file(str(path))
            assert loaded["w"].shape == k_w.shape
            assert loaded["s"].shape == k_s.shape
            assert loaded["b"].shape == k_b.shape

    def test_multi_layer_q4_format_preserved(self) -> None:
        """Q4 format preserved across multiple layers in same file."""
        n_layers = 4
        tensors = {}
        for layer_id in range(n_layers):
            k_w, k_s, k_b = _make_q4_numpy_arrays()
            v_w, v_s, v_b = _make_q4_numpy_arrays()
            tensors[f"L{layer_id}_B0_K_weights"] = k_w
            tensors[f"L{layer_id}_B0_K_scales"] = k_s
            tensors[f"L{layer_id}_B0_K_biases"] = k_b
            tensors[f"L{layer_id}_B0_V_weights"] = v_w
            tensors[f"L{layer_id}_B0_V_scales"] = v_s
            tensors[f"L{layer_id}_B0_V_biases"] = v_b

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_layer.safetensors"
            save_file(tensors, str(path))
            loaded = load_file(str(path))
            for layer_id in range(n_layers):
                assert loaded[f"L{layer_id}_B0_K_weights"].dtype == np.uint32
                assert loaded[f"L{layer_id}_B0_K_scales"].dtype == np.float16
                assert loaded[f"L{layer_id}_B0_V_weights"].dtype == np.uint32
                assert loaded[f"L{layer_id}_B0_V_scales"].dtype == np.float16


# ---------------------------------------------------------------------------
# Test 3.0d: Character-Level Prefix Match Round-Trip
# ---------------------------------------------------------------------------

class TestCharacterPrefixRoundTrip:
    """Verify prompt_text is preserved through AgentCacheStore save → load."""

    @pytest.fixture
    def store_with_adapter(self):
        """Create AgentCacheStore with in-memory adapter."""
        tag = _make_tag()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = InMemoryAdapter(cache_dir=Path(tmpdir))
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=tag,
                cache_adapter=adapter,
            )
            yield store, adapter

    def test_prompt_text_exact_preservation(self, store_with_adapter) -> None:
        """prompt_text survives save → evict → load without any character changes."""
        store, adapter = store_with_adapter
        original_text = "Hello, how are you today?"

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=[1, 2, 3],
            prompt_text=original_text,
        )

        store.save("agent_1", blocks)

        # Flush write-behind to disk, then clear hot tier
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.prompt_text == original_text
        # Character-by-character verification
        for i, (c1, c2) in enumerate(zip(original_text, recovered.prompt_text)):
            assert c1 == c2, f"Character mismatch at position {i}: '{c1}' vs '{c2}'"
        assert len(recovered.prompt_text) == len(original_text)

    def test_common_prefix_chars_full_match_after_round_trip(self, store_with_adapter) -> None:
        """common_prefix_chars returns full length after save → load."""
        store, adapter = store_with_adapter
        text = "Hello, how are you today?"

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            prompt_text=text,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.common_prefix_chars(text) == 25

    def test_common_prefix_chars_partial_match_after_round_trip(self, store_with_adapter) -> None:
        """common_prefix_chars returns correct partial match after round-trip."""
        store, adapter = store_with_adapter
        text = "Hello, how are you today?"

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            prompt_text=text,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.prompt_text == "Hello, how are you today?"
        # "Hello, how are you" = 18 chars, then ' ' vs '?' mismatch
        assert recovered.common_prefix_chars("Hello, how are you?") == 18

    def test_common_prefix_chars_no_match_after_round_trip(self, store_with_adapter) -> None:
        """common_prefix_chars returns 0 for completely different text."""
        store, adapter = store_with_adapter

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            prompt_text="Hello, how are you today?",
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.prompt_text == "Hello, how are you today?"
        assert recovered.common_prefix_chars("Goodbye") == 0

    def test_bpe_boundary_space_handling(self, store_with_adapter) -> None:
        """Character-level matching handles BPE boundary at comma+space."""
        store, adapter = store_with_adapter

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            prompt_text="Hello, world",
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.prompt_text == "Hello, world"
        # "Hello, " = 7 chars
        assert recovered.common_prefix_chars("Hello, ") == 7

    def test_unicode_prompt_text_preserved(self, store_with_adapter) -> None:
        """Unicode characters in prompt_text survive round-trip."""
        store, adapter = store_with_adapter

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            prompt_text="Caf\u00e9 au lait",
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.prompt_text == "Caf\u00e9 au lait"
        assert recovered.common_prefix_chars("Caf\u00e9 au") == 7


# ---------------------------------------------------------------------------
# Test 3.0e: Token Sequence Fidelity
# ---------------------------------------------------------------------------

class TestTokenSequenceFidelity:
    """Verify token_sequence is preserved exactly through save → load."""

    @pytest.fixture
    def store_with_adapter(self):
        """Create AgentCacheStore with in-memory adapter."""
        tag = _make_tag()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = InMemoryAdapter(cache_dir=Path(tmpdir))
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=tag,
                cache_adapter=adapter,
            )
            yield store

    def test_token_sequence_exact_match(self, store_with_adapter) -> None:
        """token_sequence values are identical after save → evict → load."""
        store = store_with_adapter
        seq = [101, 202, 303, 404, 505]

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.token_sequence == [101, 202, 303, 404, 505]

    def test_common_prefix_length_full_match(self, store_with_adapter) -> None:
        """common_prefix_length returns full length after round-trip."""
        store = store_with_adapter
        seq = [101, 202, 303, 404, 505]

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.common_prefix_length([101, 202, 303, 404, 505]) == 5

    def test_common_prefix_length_partial_match(self, store_with_adapter) -> None:
        """common_prefix_length returns correct partial match after round-trip."""
        store = store_with_adapter
        seq = [101, 202, 303, 404, 505]

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.common_prefix_length([101, 202, 999]) == 2

    def test_common_prefix_length_no_match(self, store_with_adapter) -> None:
        """common_prefix_length returns 0 for completely different tokens."""
        store = store_with_adapter
        seq = [101, 202, 303]

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.common_prefix_length([999, 888, 777]) == 0

    def test_common_prefix_length_empty_query(self, store_with_adapter) -> None:
        """common_prefix_length handles empty query after round-trip."""
        store = store_with_adapter
        seq = [101, 202, 303]

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.common_prefix_length([]) == 0

    def test_empty_token_sequence_preserved(self, store_with_adapter) -> None:
        """Empty token_sequence stays empty after round-trip."""
        store = store_with_adapter

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=[],
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.token_sequence == []

    def test_large_token_sequence_preserved(self, store_with_adapter) -> None:
        """Large token sequence (1000 tokens) survives round-trip."""
        store = store_with_adapter
        seq = list(range(1000))

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
        )
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered = store.load("agent_1")

        assert recovered is not None, "Load returned None after invalidate_hot"
        assert recovered.agent_id == "agent_1"
        assert recovered.token_sequence == list(range(1000))
        assert recovered.common_prefix_length(list(range(500)) + [99999]) == 500


# ---------------------------------------------------------------------------
# Test 3.0a: AgentCacheStore save → evict → load → compare
# ---------------------------------------------------------------------------

class TestAgentCacheStoreRoundTrip:
    """Full AgentCacheStore round-trip: save → evict → load → compare."""

    @pytest.fixture
    def store_with_adapter(self):
        """Create AgentCacheStore with in-memory adapter."""
        tag = _make_tag()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = InMemoryAdapter(cache_dir=Path(tmpdir))
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=tag,
                cache_adapter=adapter,
            )
            yield store, adapter

    def test_save_load_from_hot_tier(self, store_with_adapter) -> None:
        """save() followed by load() returns same blocks from hot tier."""
        store, _ = store_with_adapter
        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=[10, 20, 30],
            prompt_text="test prompt",
        )

        store.save("agent_1", blocks)
        recovered = store.load("agent_1")

        assert recovered is not None, "Hot tier load returned None"
        assert recovered.agent_id == "agent_1"
        assert recovered.total_tokens == 256
        assert recovered.token_sequence == [10, 20, 30]
        assert recovered.prompt_text == "test prompt"

    def test_save_evict_load_from_warm_tier(self, store_with_adapter) -> None:
        """save() → evict from hot → load() recovers from warm tier (disk)."""
        store, _ = store_with_adapter
        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=[10, 20, 30],
            prompt_text="test prompt",
        )

        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")

        # Verify hot tier is empty
        assert "agent_1" not in store._hot_cache

        recovered = store.load("agent_1")

        assert recovered is not None, "Warm tier load returned None"
        assert recovered.agent_id == "agent_1"
        assert recovered.total_tokens == 256
        assert recovered.token_sequence == [10, 20, 30]
        assert recovered.prompt_text == "test prompt"

    def test_model_tag_preserved_in_metadata(self, store_with_adapter) -> None:
        """Model tag fields are written to metadata and validated on load."""
        store, adapter = store_with_adapter
        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
        )

        store.save("agent_1", blocks)
        store.flush_dirty()

        # Verify metadata was passed to adapter
        _, metadata = adapter._store["agent_1"]
        assert metadata["model_id"] == "test-model"
        assert metadata["n_layers"] == 4
        assert metadata["n_kv_heads"] == 4
        assert metadata["head_dim"] == 64
        assert metadata["block_tokens"] == 256

    def test_incompatible_model_tag_rejected(self, store_with_adapter) -> None:
        """Loading cache with mismatched model tag returns None."""
        store, adapter = store_with_adapter
        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
        )

        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")

        # Change model tag to incompatible one
        store.model_tag = ModelTag(
            model_id="different-model",
            n_layers=48,  # Different from 4
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
        )

        recovered = store.load("agent_1")
        assert recovered is None

    def test_double_save_load_produces_same_result(self, store_with_adapter) -> None:
        """Save → load → save → load produces identical metadata (no accumulation)."""
        store, adapter = store_with_adapter
        seq = [101, 202, 303, 404]
        text = "Hello world, this is a test."

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=256,
            token_sequence=seq,
            prompt_text=text,
        )

        # Round 1
        store.save("agent_1", blocks)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered1 = store.load("agent_1")
        assert recovered1 is not None, "Round 1 load returned None"
        assert recovered1.agent_id == "agent_1"

        # Round 2: save recovered data, load again
        store.save("agent_1", recovered1)
        store.flush_dirty()
        store.invalidate_hot("agent_1")
        recovered2 = store.load("agent_1")
        assert recovered2 is not None, "Round 2 load returned None"
        assert recovered2.agent_id == "agent_1"

        # Compare: second recovery must match first
        assert recovered2.token_sequence == recovered1.token_sequence == seq
        assert recovered2.prompt_text == recovered1.prompt_text == text
        assert recovered2.total_tokens == recovered1.total_tokens == 256
        assert recovered2.agent_id == recovered1.agent_id == "agent_1"

    def test_lru_eviction_preserves_warm_tier(self, store_with_adapter) -> None:
        """LRU eviction moves agents to warm tier, which survives load."""
        store, adapter = store_with_adapter
        # Reduce max to trigger eviction
        store.max_hot_agents = 2

        for i in range(3):
            block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
            blocks = AgentBlocks(
                agent_id=f"agent_{i}",
                blocks={0: [block]},
                total_tokens=256,
                token_sequence=[i, i + 1, i + 2],
            )
            store.save(f"agent_{i}", blocks)

        # agent_0 should have been evicted from hot tier
        assert len(store._hot_cache) == 2

        # But it should still be loadable from warm tier
        recovered = store.load("agent_0")
        assert recovered is not None, "LRU-evicted agent_0 not recoverable from warm tier"
        assert recovered.agent_id == "agent_0"
        assert recovered.token_sequence == [0, 1, 2]

    def test_multiple_agents_independent(self, store_with_adapter) -> None:
        """Multiple agents' caches don't contaminate each other."""
        store, _ = store_with_adapter

        for i in range(4):
            block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake")
            blocks = AgentBlocks(
                agent_id=f"agent_{i}",
                blocks={0: [block]},
                total_tokens=256,
                token_sequence=[i * 100, i * 100 + 1],
                prompt_text=f"Prompt for agent {i}",
            )
            store.save(f"agent_{i}", blocks)

        # Evict all and load back
        store.evict_lru(target_count=0)

        for i in range(4):
            recovered = store.load(f"agent_{i}")
            assert recovered is not None, f"agent_{i} not recoverable after evict_lru"
            assert recovered.agent_id == f"agent_{i}"
            assert recovered.token_sequence == [i * 100, i * 100 + 1]
            assert recovered.prompt_text == f"Prompt for agent {i}"
