"""Adaptive runtime configuration for inference parameters.

Adjusts prefill step size, batch size, and chunk sizes based on
real-time workload characteristics (input length, memory pressure,
cache hit ratio, batch depth).

Uses exponential moving averages to smooth metrics and avoid
oscillation. All thresholds are loaded from the per-model profile
TOML or use sensible defaults.

Example:
    >>> from agent_memory.application.adaptive_config import AdaptiveConfig
    >>> config = AdaptiveConfig(mlx_settings, model_profile)
    >>> config.update(input_tokens=5000, cache_hit=True, peak_mb=8000)
    >>> config.effective_prefill_step_size
    512
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EMA_ALPHA = 0.3  # Smoothing factor for exponential moving averages

# Default thresholds (overridden by model profile TOML)
DEFAULT_LONG_CONTEXT_THRESHOLD = 4000
DEFAULT_HIGH_BATCH_THRESHOLD = 3
DEFAULT_MEMORY_PRESSURE_MB = 10500  # Profiling: peak at 16K input was 10202MB
DEFAULT_MIN_CACHE_BENEFIT_RATIO = 0.8


@dataclass
class AdaptiveState:
    """Current workload state tracked via exponential moving averages."""

    avg_input_tokens: float = 0.0
    cache_hit_ratio: float = 0.0
    peak_memory_mb: float = 0.0
    active_batch_size: int = 0
    pending_queue_depth: int = 0
    total_requests: int = 0
    total_cache_hits: int = 0
    last_update: float = field(default_factory=time.monotonic)


class AdaptiveConfig:
    """Runtime-adaptive configuration overlay.

    Reads baseline values from MLXSettings and adjusts them based on
    observed workload patterns. All properties return the effective
    value to use — callers don't need to know whether adaptation is
    active.

    The model profile dict should have a 'thresholds' section:
        {
            "thresholds": {
                "long_context_threshold": 4000,
                "high_batch_threshold": 3,
                "memory_pressure_mb": 12000,
                "min_cache_benefit_ratio": 0.8,
            }
        }
    """

    def __init__(
        self,
        prefill_step_size: int,
        max_batch_size: int,
        chunked_prefill_min_chunk: int,
        chunked_prefill_max_chunk: int,
        model_profile: dict[str, Any] | None = None,
    ) -> None:
        self._prefill_step_size = prefill_step_size
        self._max_batch_size = max_batch_size
        self._min_chunk = chunked_prefill_min_chunk
        self._max_chunk = chunked_prefill_max_chunk
        self._state = AdaptiveState()

        thresholds = (model_profile or {}).get("thresholds", {})
        self._long_context = int(
            thresholds.get("long_context_threshold", DEFAULT_LONG_CONTEXT_THRESHOLD)
        )
        self._high_batch = int(thresholds.get("high_batch_threshold", DEFAULT_HIGH_BATCH_THRESHOLD))
        self._memory_pressure = float(
            thresholds.get("memory_pressure_mb", DEFAULT_MEMORY_PRESSURE_MB)
        )
        self._min_cache_ratio = float(
            thresholds.get("min_cache_benefit_ratio", DEFAULT_MIN_CACHE_BENEFIT_RATIO)
        )

    def update(
        self,
        input_tokens: int = 0,
        cache_hit: bool = False,
        peak_mb: float = 0.0,
        active_batch: int = 0,
        queue_depth: int = 0,
    ) -> None:
        """Update workload state after a request completes."""
        s = self._state
        alpha = EMA_ALPHA

        if input_tokens > 0:
            s.avg_input_tokens = alpha * input_tokens + (1 - alpha) * s.avg_input_tokens

        s.total_requests += 1
        if cache_hit:
            s.total_cache_hits += 1
        s.cache_hit_ratio = s.total_cache_hits / s.total_requests if s.total_requests > 0 else 0.0

        if peak_mb > 0:
            s.peak_memory_mb = alpha * peak_mb + (1 - alpha) * s.peak_memory_mb

        s.active_batch_size = active_batch
        s.pending_queue_depth = queue_depth
        s.last_update = time.monotonic()

    def set_batch_state(self, active_batch: int, queue_depth: int) -> None:
        """Update batch/queue state without a full request update."""
        self._state.active_batch_size = active_batch
        self._state.pending_queue_depth = queue_depth

    @property
    def state(self) -> AdaptiveState:
        """Current adaptive state (read-only snapshot)."""
        return self._state

    @property
    def effective_prefill_step_size(self) -> int:
        """Adapt prefill step based on input length and memory pressure.

        - Long contexts (>threshold): reduce step to cap memory
        - Memory pressure: reduce step regardless of input length
        - Otherwise: use baseline value
        """
        s = self._state

        if s.peak_memory_mb > self._memory_pressure:
            return min(self._prefill_step_size, 256)

        if s.avg_input_tokens > self._long_context:
            return min(self._prefill_step_size, 512)

        return self._prefill_step_size

    @property
    def effective_max_batch_size(self) -> int:
        """Reduce batch size under memory pressure."""
        if self._state.peak_memory_mb > self._memory_pressure:
            return 1
        return self._max_batch_size

    @property
    def effective_chunk_sizes(self) -> tuple[int, int]:
        """Adapt chunk sizes based on batch and memory pressure.

        Returns (min_chunk, max_chunk) tuple.
        """
        s = self._state

        if s.peak_memory_mb > self._memory_pressure:
            return (256, 2048)

        if s.active_batch_size >= self._high_batch:
            return (256, 2048)

        return (self._min_chunk, self._max_chunk)

    @property
    def cache_benefit_expected(self) -> bool:
        """Whether cache hits are providing measurable benefit.

        Returns False if cache hit ratio is below the configured
        threshold, suggesting the workload is mostly cold starts
        and cache overhead may not be worth it.
        """
        return self._state.cache_hit_ratio >= self._min_cache_ratio

    def log_state(self) -> None:
        """Log current adaptive state for debugging."""
        s = self._state
        logger.info(
            f"[ADAPTIVE] "
            f"avg_input={s.avg_input_tokens:.0f} "
            f"cache_hit={s.cache_hit_ratio:.2f} "
            f"peak_mem={s.peak_memory_mb:.0f}MB "
            f"batch={s.active_batch_size} "
            f"queue={s.pending_queue_depth} | "
            f"eff_prefill={self.effective_prefill_step_size} "
            f"eff_batch={self.effective_max_batch_size} "
            f"eff_chunks={self.effective_chunk_sizes}"
        )
