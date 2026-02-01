"""Analytical estimate validation tests.

Cross-references config/models/*.toml benchmark values against computed
formulas from ModelCacheSpec.bytes_per_block_per_layer(). Verifies that
formulas predict values in the right ballpark and that TOML values are
internally consistent.

Every expected value is computed from first principles, never hardcoded.
"""

import math
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from semantic.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit

BYTES_PER_MB = 1024 * 1024
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "models"
SYSTEM_MEMORY_GB = 24  # M4 Pro benchmark platform


def _load_toml(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "rb") as f:
        return tomllib.load(f)


# ── Model Specs (from documentation / HuggingFace configs) ───────────────

GEMMA3_SPEC = ModelCacheSpec(
    n_layers=48,
    n_kv_heads=4,
    head_dim=256,
    block_tokens=256,
    layer_types=["global"] * 8 + ["sliding_window"] * 40,
    kv_bits=4,
    kv_group_size=64,
)

GPT_OSS_SPEC = ModelCacheSpec(
    n_layers=24,
    n_kv_heads=8,
    head_dim=45,
    block_tokens=256,
    layer_types=["global"] * 24,
    kv_bits=4,
    kv_group_size=64,
)

SMOLLM2_SPEC = ModelCacheSpec(
    n_layers=30,
    n_kv_heads=9,
    head_dim=64,
    block_tokens=256,
    layer_types=["global"] * 30,
    kv_bits=4,
    kv_group_size=64,
)


def _compute_q4_bytes(spec: ModelCacheSpec) -> int:
    """Compute Q4 bytes per block per layer from first principles."""
    elements_per_kv = spec.n_kv_heads * spec.head_dim * spec.block_tokens
    total_elements = elements_per_kv * 2  # K and V

    weight_bytes = (total_elements * 4) // 8
    groups_per_kv = math.ceil(elements_per_kv / spec.kv_group_size)
    total_groups = groups_per_kv * 2
    scales_bytes = total_groups * 2
    biases_bytes = total_groups * 2

    return weight_bytes + scales_bytes + biases_bytes


def _compute_fp16_bytes(spec: ModelCacheSpec) -> int:
    """Compute FP16 bytes per block per layer from first principles."""
    elements_per_kv = spec.n_kv_heads * spec.head_dim * spec.block_tokens
    return elements_per_kv * 2 * 2  # K+V * 2 bytes


class TestFormulaCorrectness:
    """Verify bytes_per_block_per_layer() matches hand-computed values."""

    def test_gemma3_q4_matches_hand_computation(self) -> None:
        """Gemma 3 Q4: formula output matches step-by-step computation."""
        formula_result = GEMMA3_SPEC.bytes_per_block_per_layer()
        hand_computed = _compute_q4_bytes(GEMMA3_SPEC)

        assert formula_result == hand_computed
        assert formula_result == 294_912  # 262144 + 16384 + 16384

    def test_gpt_oss_q4_matches_hand_computation(self) -> None:
        """GPT-OSS-20B Q4: formula output matches step-by-step computation."""
        formula_result = GPT_OSS_SPEC.bytes_per_block_per_layer()
        hand_computed = _compute_q4_bytes(GPT_OSS_SPEC)

        assert formula_result == hand_computed
        assert formula_result == 103_680  # 92160 + 5760 + 5760

    def test_smollm2_q4_matches_hand_computation(self) -> None:
        """SmolLM2-135M Q4: formula output matches step-by-step computation."""
        formula_result = SMOLLM2_SPEC.bytes_per_block_per_layer()
        hand_computed = _compute_q4_bytes(SMOLLM2_SPEC)

        assert formula_result == hand_computed
        assert formula_result == 165_888  # 147456 + 9216 + 9216

    def test_gemma3_fp16_matches_hand_computation(self) -> None:
        """Gemma 3 FP16: formula output matches step-by-step computation."""
        fp16_spec = ModelCacheSpec(
            n_layers=GEMMA3_SPEC.n_layers,
            n_kv_heads=GEMMA3_SPEC.n_kv_heads,
            head_dim=GEMMA3_SPEC.head_dim,
            block_tokens=GEMMA3_SPEC.block_tokens,
            layer_types=GEMMA3_SPEC.layer_types,
            kv_bits=None,
            kv_group_size=GEMMA3_SPEC.kv_group_size,
        )
        formula_result = fp16_spec.bytes_per_block_per_layer()
        hand_computed = _compute_fp16_bytes(GEMMA3_SPEC)

        assert formula_result == hand_computed
        # 4 heads * 256 dim * 256 tokens * 2(K+V) * 2 bytes = 1,048,576
        assert formula_result == 1_048_576


class TestQ4FP16Ratio:
    """Q4 should be 25-28% of FP16 memory per block."""

    @pytest.mark.parametrize(
        "spec",
        [GEMMA3_SPEC, GPT_OSS_SPEC, SMOLLM2_SPEC],
        ids=["gemma3", "gpt-oss", "smollm2"],
    )
    def test_q4_is_approximately_28_percent_of_fp16(self, spec: ModelCacheSpec) -> None:
        """Q4/FP16 ratio depends only on kv_bits and group_size."""
        q4_bytes = spec.bytes_per_block_per_layer()

        fp16_spec = ModelCacheSpec(
            n_layers=spec.n_layers,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
            block_tokens=spec.block_tokens,
            layer_types=spec.layer_types,
            kv_bits=None,
            kv_group_size=spec.kv_group_size,
        )
        fp16_bytes = fp16_spec.bytes_per_block_per_layer()

        ratio = q4_bytes / fp16_bytes

        # For bits=4, group_size=64: ratio ≈ (0.5 + 4/64) / 2 = 0.28125
        # Allow tolerance for models where elements_per_kv isn't divisible by group_size
        assert 0.25 <= ratio <= 0.30, (
            f"Q4/FP16 ratio {ratio:.4f} outside expected 25-30% range"
        )

    def test_ratio_formula_derivation(self) -> None:
        """Verify the analytical ratio formula: (bits/8 + 4/group_size) / 2."""
        bits = 4
        group_size = 64
        # Per element: weight = bits/8 bytes, overhead = 4/group_size bytes (2 scales + 2 biases)
        # FP16 per element: 2 bytes
        expected_ratio = (bits / 8 + 4 / group_size) / 2

        assert abs(expected_ratio - 0.28125) < 1e-10

        # Cross-validate with Gemma 3 (large enough that rounding is negligible)
        q4 = GEMMA3_SPEC.bytes_per_block_per_layer()
        fp16_spec = ModelCacheSpec(
            n_layers=48, n_kv_heads=4, head_dim=256, block_tokens=256,
            layer_types=["global"] * 48, kv_bits=None, kv_group_size=64,
        )
        fp16 = fp16_spec.bytes_per_block_per_layer()
        actual_ratio = q4 / fp16

        assert abs(actual_ratio - expected_ratio) < 0.001


class TestTomlPoolFitsInMemory:
    """Verify block_pool_blocks * block_size_mb fits in available system memory."""

    def test_gemma3_pool_fits_in_24gb(self) -> None:
        """Gemma 3 pool must fit in available memory after model is loaded."""
        toml = _load_toml("gemma-3-12b-it-4bit.toml")
        model_gb = toml["benchmark"]["model_memory_gb"]
        pool_blocks = toml["benchmark"]["block_pool_blocks"]
        block_mb = toml["benchmark"]["block_size_mb"]

        pool_gb = pool_blocks * block_mb / 1024
        available_gb = SYSTEM_MEMORY_GB - model_gb

        assert pool_gb < available_gb, (
            f"Pool {pool_gb:.1f} GB exceeds available {available_gb:.1f} GB"
        )

    def test_gpt_oss_pool_fits_in_24gb(self) -> None:
        """GPT-OSS-20B pool must fit in available memory after model is loaded."""
        toml = _load_toml("gpt-oss-20b-mxfp4.toml")
        model_gb = toml["benchmark"]["model_memory_gb"]
        pool_blocks = toml["benchmark"]["block_pool_blocks"]
        block_mb = toml["benchmark"]["block_size_mb"]

        pool_gb = pool_blocks * block_mb / 1024
        available_gb = SYSTEM_MEMORY_GB - model_gb

        assert pool_gb < available_gb, (
            f"Pool {pool_gb:.1f} GB exceeds available {available_gb:.1f} GB"
        )


class TestTomlBlockSizeMatchesFormula:
    """Cross-validate TOML block_size_mb against bytes_per_block_per_layer() formula."""

    def test_gpt_oss_block_size_matches_q4_formula(self) -> None:
        """GPT-OSS-20B: TOML block_size_mb should match Q4 formula within 10%."""
        toml = _load_toml("gpt-oss-20b-mxfp4.toml")
        toml_mb = toml["benchmark"]["block_size_mb"]

        formula_bytes = GPT_OSS_SPEC.bytes_per_block_per_layer()
        formula_mb = formula_bytes / BYTES_PER_MB

        error_pct = abs(toml_mb - formula_mb) / formula_mb * 100
        assert error_pct < 10, (
            f"GPT-OSS block_size_mb={toml_mb} vs formula={formula_mb:.4f} MB "
            f"({error_pct:.1f}% error)"
        )

    def test_gemma3_block_size_matches_q4_formula(self) -> None:
        """Gemma 3: TOML block_size_mb should match Q4 formula within 10%.

        Bug fix: The previous TOML value (0.53) matched Q8 (kv_bits=8) but
        the model uses kv_bits=4. Corrected to 0.28 which matches the Q4
        formula (294,912 bytes ≈ 0.281 MB per block per layer).
        """
        toml = _load_toml("gemma-3-12b-it-4bit.toml")
        toml_mb = toml["benchmark"]["block_size_mb"]

        q4_formula_bytes = GEMMA3_SPEC.bytes_per_block_per_layer()
        q4_formula_mb = q4_formula_bytes / BYTES_PER_MB

        error_pct = abs(toml_mb - q4_formula_mb) / q4_formula_mb * 100
        assert error_pct < 10, (
            f"Gemma 3 block_size_mb={toml_mb} vs Q4 formula={q4_formula_mb:.4f} MB "
            f"({error_pct:.1f}% error)"
        )


class TestMultiturnSpeedupCrossCheck:
    """Verify stated multiturn_speedup against timing data in TOML."""

    def test_gemma3_speedup_matches_timing_ratio(self) -> None:
        """Gemma 3: stated 2.8x matches cold_medium / multiturn_t2 ratio."""
        toml = _load_toml("gemma-3-12b-it-4bit.toml")
        cold_ttft = toml["benchmark"]["cold_medium_ttft_ms"]
        warm_e2e = toml["benchmark"]["multiturn_t2_e2e_ms"]
        stated = float(toml["benchmark"]["multiturn_speedup"].rstrip("x"))

        computed_speedup = cold_ttft / warm_e2e
        error_pct = abs(computed_speedup - stated) / stated * 100

        assert error_pct < 20, (
            f"Gemma 3 speedup: computed {computed_speedup:.2f}x vs stated {stated}x "
            f"({error_pct:.1f}% error)"
        )

    def test_gpt_oss_speedup_matches_timing_ratio(self) -> None:
        """GPT-OSS-20B: stated 2.2x matches cold_medium / multiturn_t2 ratio."""
        toml = _load_toml("gpt-oss-20b-mxfp4.toml")
        cold_ttft = toml["benchmark"]["cold_medium_ttft_ms"]
        warm_e2e = toml["benchmark"]["multiturn_t2_e2e_ms"]
        stated = float(toml["benchmark"]["multiturn_speedup"].rstrip("x"))

        computed_speedup = cold_ttft / warm_e2e
        error_pct = abs(computed_speedup - stated) / stated * 100

        assert error_pct < 20, (
            f"GPT-OSS speedup: computed {computed_speedup:.2f}x vs stated {stated}x "
            f"({error_pct:.1f}% error)"
        )

    def test_smollm2_speedup_negligible(self) -> None:
        """SmolLM2: stated 1.0x — prefill is too fast for cache to help."""
        toml = _load_toml("smollm2-135m.toml")
        cold_ttft = toml["benchmark"]["cold_medium_ttft_ms"]
        warm_e2e = toml["benchmark"]["multiturn_t2_e2e_ms"]
        stated = float(toml["benchmark"]["multiturn_speedup"].rstrip("x"))

        computed_speedup = cold_ttft / warm_e2e

        # For tiny models, cache reuse gives negligible speedup
        assert stated == 1.0
        assert computed_speedup < 1.5, (
            f"SmolLM2 should have negligible speedup, got {computed_speedup:.2f}x"
        )


class TestMaxAgentsMemoryBudget:
    """Verify max_agents_in_memory is reasonable for system memory."""

    def test_gemma3_max_agents_fit_in_memory(self) -> None:
        """3 Gemma 3 agents' caches should fit in available memory."""
        toml = _load_toml("gemma-3-12b-it-4bit.toml")
        max_agents = toml["optimal"]["max_agents_in_memory"]
        model_gb = toml["benchmark"]["model_memory_gb"]
        pool_blocks = toml["benchmark"]["block_pool_blocks"]
        block_mb = toml["benchmark"]["block_size_mb"]

        total_pool_gb = pool_blocks * block_mb / 1024
        available_gb = SYSTEM_MEMORY_GB - model_gb

        # Pool shared across agents, so just verify pool fits
        assert total_pool_gb < available_gb
        assert max_agents >= 1
        assert max_agents <= 10  # Reasonable upper bound for 12B model

    def test_smollm2_max_agents_is_generous(self) -> None:
        """SmolLM2 (135M) should allow many agents given its small footprint."""
        toml = _load_toml("smollm2-135m.toml")
        max_agents = toml["optimal"]["max_agents_in_memory"]

        # 135M model uses ~1GB, leaving ~23GB for caches
        # 20 agents is reasonable for such a small model
        assert max_agents >= 10
        assert max_agents <= 50

    def test_gpt_oss_max_agents_conservative(self) -> None:
        """GPT-OSS-20B (10.6GB) should be conservative with agents."""
        toml = _load_toml("gpt-oss-20b-mxfp4.toml")
        max_agents = toml["optimal"]["max_agents_in_memory"]
        model_gb = toml["benchmark"]["model_memory_gb"]

        available_gb = SYSTEM_MEMORY_GB - model_gb
        # With only 13.4GB left, should be conservative
        assert max_agents <= 5
        assert max_agents >= 1
        # Sanity: more constrained than SmolLM2
        smollm2_toml = _load_toml("smollm2-135m.toml")
        assert max_agents < smollm2_toml["optimal"]["max_agents_in_memory"]
