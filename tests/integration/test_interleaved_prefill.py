"""Integration test: interleaved chunked prefill with decode.

Tests the full ConcurrentScheduler flow with a FakeBatchEngine,
verifying that:
1. Two concurrent requests complete successfully.
2. Long prompts go through chunked prefill (adapter called).
3. Short prompts bypass prefill (direct engine path).
4. Decode steps and prefill chunks interleave correctly.
5. Futures resolve with correct completions.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agent_memory.application.scheduler import ConcurrentScheduler
from agent_memory.application.shared_prefix_cache import SharedPrefixCache

# -------------------------------------------------------------------
# Fakes (self-contained — no MLX dependency)
# -------------------------------------------------------------------


@dataclass
class FakeCompletion:
    uid: str
    text: str = "generated text"
    blocks: Any = None
    finish_reason: str = "end_turn"
    token_count: int = 5


@dataclass
class FakeStepResult:
    """Mimics StepOneResult for scheduler compatibility."""

    uid: str
    text: str = ""
    token_count: int = 0
    finish_reason: str | None = None
    completion: FakeCompletion | None = None


class FakeEngine:
    """Simulates BatchGenerator: tracks submissions, completes after N steps."""

    def __init__(self, steps_to_complete: int = 2) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._uid_counter = 0
        self._steps_to_complete = steps_to_complete
        self.submit_calls: list[dict[str, Any]] = []
        self.submit_with_cache_calls: list[dict[str, Any]] = []
        self.step_count = 0

    def has_active_batch(self) -> bool:
        return len(self._active) > 0

    def submit(self, **kwargs: Any) -> str:
        self._uid_counter += 1
        uid = f"uid_{self._uid_counter}"
        self._active[uid] = {"steps": self._steps_to_complete}
        self.submit_calls.append({"uid": uid, **kwargs})
        return uid

    def submit_with_cache(self, **kwargs: Any) -> str:
        self._uid_counter += 1
        uid = f"uid_{self._uid_counter}"
        self._active[uid] = {"steps": self._steps_to_complete}
        self.submit_with_cache_calls.append({"uid": uid, **kwargs})
        return uid

    def step(self) -> Iterator[FakeCompletion]:
        self.step_count += 1
        completed = []
        for uid in list(self._active):
            self._active[uid]["steps"] -= 1
            if self._active[uid]["steps"] <= 0:
                completed.append(uid)
        for uid in completed:
            del self._active[uid]
            yield FakeCompletion(uid=uid)

    def step_once(self) -> list[FakeStepResult]:
        """Per-token decode step matching StepOneResult interface."""
        self.step_count += 1
        results: list[FakeStepResult] = []
        for uid in list(self._active):
            self._active[uid]["steps"] -= 1
            if self._active[uid]["steps"] <= 0:
                comp = FakeCompletion(uid=uid)
                results.append(
                    FakeStepResult(
                        uid=uid,
                        text=comp.text,
                        token_count=comp.token_count,
                        finish_reason=comp.finish_reason,
                        completion=comp,
                    )
                )
                del self._active[uid]
            else:
                results.append(
                    FakeStepResult(
                        uid=uid,
                        text="tok",
                        token_count=1,
                    )
                )
        return results


@dataclass
class FakeKV:
    offset: int = 0


class FakePrefillAdapter:
    """Tracks prefill calls, returns fake KV caches."""

    def __init__(self, chunk_size: int = 256) -> None:
        self._chunk_size = chunk_size
        self.chunks: list[tuple[int, int]] = []
        self.init_count = 0

    def init_prefill_caches(self, n_layers: int) -> list[FakeKV]:
        self.init_count += 1
        return [FakeKV() for _ in range(n_layers)]

    def process_prefill_chunk(
        self, tokens: list[int], start: int, end: int, kv_caches: list[FakeKV]
    ) -> None:
        self.chunks.append((start, end))
        for c in kv_caches:
            c.offset = end

    def chunk_size_for_position(self, cache_pos: int) -> int:
        return self._chunk_size


# -------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------


class TestInterleavedPrefillDecode:
    """End-to-end tests for scheduler interleaving."""

    def test_short_and_long_concurrent(self) -> None:
        """Short request (direct) + long request (prefill) both complete."""
        engine = FakeEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=300)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                short_task = asyncio.create_task(
                    scheduler.submit_and_wait("agent_short", list(range(50)), None, 10)
                )
                await asyncio.sleep(0.02)
                long_task = asyncio.create_task(
                    scheduler.submit_and_wait("agent_long", list(range(800)), None, 20)
                )
                return await asyncio.gather(short_task, long_task)

            loop = asyncio.new_event_loop()
            try:
                r_short, r_long = loop.run_until_complete(run())
            finally:
                loop.close()

            assert r_short.uid.startswith("uid_")
            assert r_long.uid.startswith("uid_")

            # Short went direct (no prefill adapter)
            assert len(engine.submit_calls) == 1
            assert engine.submit_calls[0]["agent_id"] == "agent_short"

            # Long went through prefill then submit_with_cache
            assert len(engine.submit_with_cache_calls) == 1
            assert engine.submit_with_cache_calls[0]["agent_id"] == "agent_long"

            # 800 tokens, prefill_end=799 → 3 chunks (last stops at 799)
            assert len(adapter.chunks) == 3
            assert adapter.chunks[0] == (0, 300)
            assert adapter.chunks[1] == (300, 600)
            assert adapter.chunks[2] == (600, 799)
        finally:
            scheduler.stop()

    def test_two_long_prompts_sequentially_prefilled(self) -> None:
        """Two long prompts: both go through prefill queue (FIFO order)."""
        engine = FakeEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=500)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                t1 = asyncio.create_task(
                    scheduler.submit_and_wait("a1", list(range(1000)), None, 10)
                )
                t2 = asyncio.create_task(
                    scheduler.submit_and_wait("a2", list(range(600)), None, 10)
                )
                return await asyncio.gather(t1, t2)

            loop = asyncio.new_event_loop()
            try:
                r1, r2 = loop.run_until_complete(run())
            finally:
                loop.close()

            assert r1.uid.startswith("uid_")
            assert r2.uid.startswith("uid_")

            # Both went through submit_with_cache
            assert len(engine.submit_with_cache_calls) == 2

            # a1: 1000/500 = 2 chunks, a2: 600/500 = 2 chunks → 4 total
            assert len(adapter.chunks) == 4
        finally:
            scheduler.stop()

    def test_scheduler_respects_threshold(self) -> None:
        """Prompts below threshold go direct, at/above go to prefill."""
        engine = FakeEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=200)
        threshold = 500
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                # Below threshold → direct
                r1 = await scheduler.submit_and_wait("below", list(range(499)), None, 5)
                # At threshold → prefill
                r2 = await scheduler.submit_and_wait("at", list(range(500)), None, 5)
                return r1, r2

            loop = asyncio.new_event_loop()
            try:
                r1, r2 = loop.run_until_complete(run())
            finally:
                loop.close()

            assert len(engine.submit_calls) == 1  # "below"
            assert len(engine.submit_with_cache_calls) == 1  # "at"
        finally:
            scheduler.stop()

    def test_decode_steps_happen_during_prefill(self) -> None:
        """Engine.step() called while prefill is in progress (interleaving)."""
        engine = FakeEngine(steps_to_complete=3)  # Needs 3 steps to complete
        adapter = FakePrefillAdapter(chunk_size=100)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=50)
        scheduler.start()

        try:

            async def run():
                # Submit short first (enters decode immediately)
                t_short = asyncio.create_task(
                    scheduler.submit_and_wait("short", list(range(30)), None, 10)
                )
                await asyncio.sleep(0.05)
                # Submit long (enters prefill queue)
                t_long = asyncio.create_task(
                    scheduler.submit_and_wait("long", list(range(400)), None, 10)
                )
                return await asyncio.gather(t_short, t_long)

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(run())
            finally:
                loop.close()

            # Engine should have been stepped multiple times
            # (decode for short + after long promoted)
            assert engine.step_count >= 3
        finally:
            scheduler.stop()


class TestSharedPrefixCacheIntegration:
    """Tests SharedPrefixCache in isolation (no MLX needed)."""

    def test_prefix_cache_hit_after_store(self) -> None:
        cache = SharedPrefixCache()
        h = SharedPrefixCache.compute_hash("You are a helpful assistant.", "Bash, Read, Write")

        assert cache.get(h) is None

        cache.put(h, kv_caches=["fake_kv_state"], n_tokens=200, token_sequence=list(range(200)))

        entry = cache.get(h)
        assert entry.n_tokens == 200
        assert entry.hit_count == 1

    def test_different_tools_different_hash(self) -> None:
        cache = SharedPrefixCache()
        h1 = SharedPrefixCache.compute_hash("system", "Bash, Read")
        h2 = SharedPrefixCache.compute_hash("system", "Bash, Read, Write")

        cache.put(h1, kv_caches=["v1"], n_tokens=100, token_sequence=[])
        cache.put(h2, kv_caches=["v2"], n_tokens=150, token_sequence=[])

        assert cache.get(h1).kv_caches == ["v1"]
        assert cache.get(h2).kv_caches == ["v2"]

    def test_prefix_cache_survives_many_agents(self) -> None:
        """Prefix cache holds across multiple agent lookups."""
        cache = SharedPrefixCache()
        h = SharedPrefixCache.compute_hash("system prompt", "tools")
        cache.put(h, kv_caches=["shared"], n_tokens=50, token_sequence=[])

        # 100 agents looking up same prefix
        for i in range(100):
            entry = cache.get(h)
            assert entry.kv_caches == ["shared"]

        assert cache.get(h).hit_count == 101  # 100 + 1

    def test_prefix_cache_clear_on_model_swap(self) -> None:
        """Prefix cache can be cleared (e.g. on model hot-swap)."""
        cache = SharedPrefixCache()
        h = SharedPrefixCache.compute_hash("sys", "tools")
        cache.put(h, kv_caches=["old"], n_tokens=50, token_sequence=[])

        cache.clear()
        assert cache.get(h) is None
        assert cache.size == 0
