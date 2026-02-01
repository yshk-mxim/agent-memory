"""Unit tests for AdaptiveConfig runtime configuration."""

import pytest

from semantic.application.adaptive_config import (
    DEFAULT_HIGH_BATCH_THRESHOLD,
    DEFAULT_MEMORY_PRESSURE_MB,
    EMA_ALPHA,
    AdaptiveConfig,
    AdaptiveState,
)


class TestAdaptiveStateDefaults:
    def test_initial_values(self) -> None:
        state = AdaptiveState()
        assert state.avg_input_tokens == 0.0
        assert state.cache_hit_ratio == 0.0
        assert state.peak_memory_mb == 0.0
        assert state.active_batch_size == 0
        assert state.pending_queue_depth == 0
        assert state.total_requests == 0
        assert state.total_cache_hits == 0
        assert state.last_update > 0  # monotonic clock


class TestAdaptiveConfigInit:
    def test_baseline_values(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        assert config.effective_prefill_step_size == 512
        assert config.effective_max_batch_size == 4
        assert config.effective_chunk_sizes == (256, 4096)

    def test_model_profile_overrides_thresholds(self) -> None:
        profile = {
            "thresholds": {
                "long_context_threshold": 2000,
                "high_batch_threshold": 2,
                "memory_pressure_mb": 8000,
                "min_cache_benefit_ratio": 0.5,
            }
        }
        config = AdaptiveConfig(
            prefill_step_size=1024,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
            model_profile=profile,
        )
        # Threshold overrides are used internally; verify by triggering them
        # Long context threshold at 2000 (not default 4000)
        # Need enough EMA updates to push avg above 2000
        for _ in range(20):
            config.update(input_tokens=5000)
        assert config.effective_prefill_step_size == 512  # reduced

    def test_none_model_profile_uses_defaults(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
            model_profile=None,
        )
        assert config.effective_prefill_step_size == 512


class TestAdaptiveConfigUpdate:
    @pytest.fixture
    def config(self) -> AdaptiveConfig:
        return AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )

    def test_ema_smoothing_avg_input_tokens(self, config: AdaptiveConfig) -> None:
        """EMA formula: alpha * new + (1-alpha) * old."""
        config.update(input_tokens=1000)
        # First update: 0.3 * 1000 + 0.7 * 0.0 = 300.0
        expected = EMA_ALPHA * 1000 + (1 - EMA_ALPHA) * 0.0
        assert config.state.avg_input_tokens == pytest.approx(expected)

        config.update(input_tokens=2000)
        # Second: 0.3 * 2000 + 0.7 * 300.0 = 600 + 210 = 810.0
        expected = EMA_ALPHA * 2000 + (1 - EMA_ALPHA) * expected
        assert config.state.avg_input_tokens == pytest.approx(expected)

    def test_cache_hit_ratio_computed(self, config: AdaptiveConfig) -> None:
        """Ratio = total_cache_hits / total_requests."""
        config.update(cache_hit=True)
        config.update(cache_hit=False)
        config.update(cache_hit=True)
        config.update(cache_hit=True)
        # 3 hits / 4 requests = 0.75
        assert config.state.cache_hit_ratio == pytest.approx(3 / 4)
        assert config.state.total_requests == 4
        assert config.state.total_cache_hits == 3

    def test_zero_input_tokens_skips_ema_update(self, config: AdaptiveConfig) -> None:
        config.update(input_tokens=1000)
        first = config.state.avg_input_tokens
        config.update(input_tokens=0)  # Should not change avg
        assert config.state.avg_input_tokens == first

    def test_zero_peak_mb_skips_memory_update(self, config: AdaptiveConfig) -> None:
        config.update(peak_mb=5000.0)
        first = config.state.peak_memory_mb
        config.update(peak_mb=0.0)  # Should not change peak
        assert config.state.peak_memory_mb == first

    def test_ema_peak_memory(self, config: AdaptiveConfig) -> None:
        config.update(peak_mb=10000.0)
        expected = EMA_ALPHA * 10000 + (1 - EMA_ALPHA) * 0.0
        assert config.state.peak_memory_mb == pytest.approx(expected)

    def test_batch_state_updated(self, config: AdaptiveConfig) -> None:
        config.update(active_batch=3, queue_depth=5)
        assert config.state.active_batch_size == 3
        assert config.state.pending_queue_depth == 5


class TestSetBatchState:
    def test_updates_without_full_request(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        config.set_batch_state(active_batch=2, queue_depth=7)
        assert config.state.active_batch_size == 2
        assert config.state.pending_queue_depth == 7
        assert config.state.total_requests == 0  # No request counted


class TestEffectivePrefillStepSize:
    def test_memory_pressure_reduces_to_256(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=1024,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        # Push memory EMA above default threshold (10500)
        # EMA converges from 0 towards 15000 — need enough updates
        for _ in range(20):
            config.update(peak_mb=15000.0)
        assert config.state.peak_memory_mb > DEFAULT_MEMORY_PRESSURE_MB
        assert config.effective_prefill_step_size == 256

    def test_long_context_reduces_to_512(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=1024,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        # Push avg tokens above threshold (4000)
        for _ in range(10):
            config.update(input_tokens=8000)
        assert config.effective_prefill_step_size == 512

    def test_no_pressure_returns_baseline(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=1024,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        config.update(input_tokens=500, peak_mb=2000.0)
        assert config.effective_prefill_step_size == 1024

    def test_baseline_already_below_256_unchanged(self) -> None:
        """If baseline is 128, memory pressure returns min(128, 256) = 128."""
        config = AdaptiveConfig(
            prefill_step_size=128,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        for _ in range(10):
            config.update(peak_mb=15000.0)
        assert config.effective_prefill_step_size == 128


class TestEffectiveMaxBatchSize:
    def test_memory_pressure_reduces_to_1(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=8,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        for _ in range(10):
            config.update(peak_mb=15000.0)
        assert config.effective_max_batch_size == 1

    def test_normal_returns_baseline(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=8,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        assert config.effective_max_batch_size == 8


class TestEffectiveChunkSizes:
    def test_memory_pressure_reduces_chunks(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=512,
            chunked_prefill_max_chunk=4096,
        )
        for _ in range(10):
            config.update(peak_mb=15000.0)
        assert config.effective_chunk_sizes == (256, 2048)

    def test_high_batch_reduces_chunks(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=512,
            chunked_prefill_max_chunk=4096,
        )
        config.set_batch_state(
            active_batch=DEFAULT_HIGH_BATCH_THRESHOLD, queue_depth=0
        )
        assert config.effective_chunk_sizes == (256, 2048)

    def test_normal_returns_baseline_chunks(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=512,
            chunked_prefill_max_chunk=4096,
        )
        assert config.effective_chunk_sizes == (512, 4096)


class TestCacheBenefitExpected:
    def test_above_threshold_returns_true(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        # 8 hits / 10 total = 0.8 >= 0.8 threshold
        for i in range(10):
            config.update(cache_hit=(i < 8))
        assert config.cache_benefit_expected is True

    def test_below_threshold_returns_false(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        # 2 hits / 10 total = 0.2 < 0.8
        for i in range(10):
            config.update(cache_hit=(i < 2))
        assert config.cache_benefit_expected is False

    def test_zero_requests_returns_false(self) -> None:
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
        )
        assert config.cache_benefit_expected is False

    def test_custom_threshold_from_profile(self) -> None:
        profile = {"thresholds": {"min_cache_benefit_ratio": 0.5}}
        config = AdaptiveConfig(
            prefill_step_size=512,
            max_batch_size=4,
            chunked_prefill_min_chunk=256,
            chunked_prefill_max_chunk=4096,
            model_profile=profile,
        )
        # 6 hits / 10 = 0.6 >= 0.5
        for i in range(10):
            config.update(cache_hit=(i < 6))
        assert config.cache_benefit_expected is True
