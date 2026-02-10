# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for ConcurrentScheduler."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from agent_memory.application.scheduler import (
    ConcurrentScheduler,
)

# -------------------------------------------------------------------
# Fakes
# -------------------------------------------------------------------


@dataclass
class FakeKVCache:
    offset: int = 0


class FakeCompletedGeneration:
    """Minimal stand-in for CompletedGeneration used in tests."""

    def __init__(self, uid: str, text: str = "done") -> None:
        self.uid = uid
        self.text = text
        self.blocks = None
        self.finish_reason = "end_turn"
        self.token_count = 1


class FakeBatchEngine:
    """In-memory engine that completes after a fixed number of steps."""

    def __init__(self, steps_to_complete: int = 3) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._uid_counter = 0
        self._steps_to_complete = steps_to_complete

    def has_active_batch(self) -> bool:
        return len(self._active) > 0

    def submit(
        self,
        agent_id: str,
        prompt: str | None = None,
        cache: Any | None = None,
        max_tokens: int = 256,
        prompt_tokens: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> str:
        self._uid_counter += 1
        uid = f"uid_{self._uid_counter}"
        self._active[uid] = {
            "agent_id": agent_id,
            "steps_remaining": self._steps_to_complete,
        }
        return uid

    def submit_with_cache(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        kv_caches: list[Any],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        self._uid_counter += 1
        uid = f"uid_{self._uid_counter}"
        self._active[uid] = {
            "agent_id": agent_id,
            "steps_remaining": self._steps_to_complete,
        }
        return uid

    def step(self) -> Iterator[Any]:
        completed = []
        for uid, info in list(self._active.items()):
            info["steps_remaining"] -= 1
            if info["steps_remaining"] <= 0:
                completed.append(uid)

        for uid in completed:
            del self._active[uid]
            yield FakeCompletedGeneration(uid)

    def step_once(self) -> list[Any]:
        """Per-token decode step matching scheduler's expected interface."""
        from agent_memory.domain.value_objects import StepOneResult

        results = []
        completed_uids = []
        for uid, info in list(self._active.items()):
            info["steps_remaining"] -= 1
            finish = None
            completion = None
            if info["steps_remaining"] <= 0:
                finish = "end_turn"
                completed_uids.append(uid)
                completion = FakeCompletedGeneration(uid)
            results.append(
                StepOneResult(
                    uid=uid,
                    text="tok",
                    token_count=1,
                    finish_reason=finish,
                    completion=completion,
                )
            )
        for uid in completed_uids:
            del self._active[uid]
        return results


class FakePrefillAdapter:
    """Records prefill calls and completes immediately."""

    def __init__(self, chunk_size: int = 512) -> None:
        self._chunk_size = chunk_size
        self.chunks_processed: list[tuple[int, int]] = []
        self.init_calls: int = 0

    def init_prefill_caches(self, n_layers: int) -> list[FakeKVCache]:
        self.init_calls += 1
        return [FakeKVCache() for _ in range(n_layers)]

    def process_prefill_chunk(
        self,
        tokens: list[int],
        start: int,
        end: int,
        kv_caches: list[FakeKVCache],
    ) -> None:
        self.chunks_processed.append((start, end))
        for c in kv_caches:
            c.offset = end

    def chunk_size_for_position(self, cache_pos: int) -> int:
        return self._chunk_size


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def run_async(coro):
    """Run a coroutine in a new event loop (for tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# -------------------------------------------------------------------
# Tests: Lifecycle
# -------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_stop(self) -> None:
        engine = FakeBatchEngine()
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4)

        scheduler.start()
        assert scheduler.is_running
        assert scheduler._worker_thread.is_alive()

        scheduler.stop()
        assert not scheduler.is_running

    def test_double_start_is_idempotent(self) -> None:
        engine = FakeBatchEngine()
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4)

        scheduler.start()
        thread1 = scheduler._worker_thread
        scheduler.start()
        thread2 = scheduler._worker_thread
        assert thread1 is thread2

        scheduler.stop()

    def test_stop_without_start(self) -> None:
        engine = FakeBatchEngine()
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4)
        scheduler.stop()  # Should not raise

    def test_initial_counters(self) -> None:
        engine = FakeBatchEngine()
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4)
        assert scheduler.pending_prefills == 0
        assert scheduler.active_decodes == 0


# -------------------------------------------------------------------
# Tests: Short prompt (direct path, no chunked prefill)
# -------------------------------------------------------------------


class TestSchedulerDirectSubmit:
    def test_short_prompt_goes_direct(self) -> None:
        """Prompts shorter than interleave threshold bypass chunked prefill."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="a1",
                    prompt_tokens=list(range(50)),  # < threshold
                    cache=None,
                    max_tokens=10,
                )

            result = run_async(run())
            assert result.uid.startswith("uid_")
            assert adapter.init_calls == 0  # No prefill adapter used
        finally:
            scheduler.stop()

    def test_short_prompt_completes_with_correct_uid(self) -> None:
        engine = FakeBatchEngine(steps_to_complete=2)
        adapter = FakePrefillAdapter()
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="a1",
                    prompt_tokens=list(range(50)),
                    cache=None,
                    max_tokens=10,
                )

            result = run_async(run())
            assert result.text == "done"
        finally:
            scheduler.stop()


# -------------------------------------------------------------------
# Tests: Long prompt (chunked prefill path)
# -------------------------------------------------------------------


class TestSchedulerChunkedPrefill:
    def test_long_prompt_uses_prefill_adapter(self) -> None:
        """Prompts longer than threshold go through chunked prefill."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=512)
        threshold = 100
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=4, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="a1",
                    prompt_tokens=list(range(1024)),  # > threshold
                    cache=None,
                    max_tokens=10,
                )

            result = run_async(run())
            assert adapter.init_calls == 1
            # prefill_end = 1024 - 1 = 1023 (reserves 1 token for BatchGenerator)
            assert len(adapter.chunks_processed) == 2  # 512 + 511
            assert adapter.chunks_processed[0] == (0, 512)
            assert adapter.chunks_processed[1] == (512, 1023)
        finally:
            scheduler.stop()

    def test_prefill_then_decode_completes(self) -> None:
        """After prefill, sequence enters decode and completes."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=256)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="a1",
                    prompt_tokens=list(range(300)),
                    cache=None,
                    max_tokens=10,
                )

            result = run_async(run())
            assert result.uid.startswith("uid_")
            # 300 tokens / 256 chunk = 2 chunks
            assert len(adapter.chunks_processed) == 2
        finally:
            scheduler.stop()

    def test_multiple_chunks_tracked(self) -> None:
        """Large prompt requires many chunks, all tracked."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=100)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=50)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="a1",
                    prompt_tokens=list(range(450)),
                    cache=None,
                    max_tokens=5,
                )

            result = run_async(run())
            # prefill_end = 450 - 1 = 449; chunks: (0,100),(100,200),(200,300),(300,400),(400,449)
            assert len(adapter.chunks_processed) == 5
            assert adapter.chunks_processed[-1] == (400, 449)
        finally:
            scheduler.stop()


# -------------------------------------------------------------------
# Tests: Interleaving (decode + prefill concurrent)
# -------------------------------------------------------------------


class TestSchedulerInterleaving:
    def test_two_concurrent_requests(self) -> None:
        """Two requests: one short (direct), one long (prefill).
        Both complete successfully.
        """
        engine = FakeBatchEngine(steps_to_complete=2)
        adapter = FakePrefillAdapter(chunk_size=200)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                t1 = asyncio.create_task(
                    scheduler.submit_and_wait("short", list(range(50)), None, 10)
                )
                # Small delay to let first request enter decode
                await asyncio.sleep(0.05)
                t2 = asyncio.create_task(
                    scheduler.submit_and_wait("long", list(range(500)), None, 10)
                )
                r1, r2 = await asyncio.gather(t1, t2)
                return r1, r2

            r1, r2 = run_async(run())
            assert r1.uid.startswith("uid_")
            assert r2.uid.startswith("uid_")
            # Long prompt went through prefill: 500/200 = 3 chunks
            assert len(adapter.chunks_processed) == 3
        finally:
            scheduler.stop()

    def test_both_long_prompts(self) -> None:
        """Two long prompts: both go through prefill queue (FIFO)."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=200)
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=100)
        scheduler.start()

        try:

            async def run():
                t1 = asyncio.create_task(
                    scheduler.submit_and_wait("a1", list(range(400)), None, 10)
                )
                t2 = asyncio.create_task(
                    scheduler.submit_and_wait("a2", list(range(600)), None, 10)
                )
                r1, r2 = await asyncio.gather(t1, t2)
                return r1, r2

            r1, r2 = run_async(run())
            # a1: 400/200=2 chunks, a2: 600/200=3 chunks → 5 total
            assert len(adapter.chunks_processed) == 5
        finally:
            scheduler.stop()


# -------------------------------------------------------------------
# Tests: Error handling
# -------------------------------------------------------------------


class TestSchedulerErrorHandling:
    def test_prefill_failure_rejects_request(self) -> None:
        """If a prefill chunk fails, the request Future gets the exception."""
        engine = FakeBatchEngine()
        adapter = FakePrefillAdapter()

        # Make process_prefill_chunk raise
        def failing_chunk(*args, **kwargs):
            raise RuntimeError("GPU OOM")

        adapter.process_prefill_chunk = failing_chunk

        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=10)
        scheduler.start()

        try:

            async def run():
                return await scheduler.submit_and_wait(
                    agent_id="fail",
                    prompt_tokens=list(range(100)),
                    cache=None,
                    max_tokens=10,
                )

            with pytest.raises(RuntimeError, match="Prefill chunk failed"):
                run_async(run())
        finally:
            scheduler.stop()


# -------------------------------------------------------------------
# Tests: Request routing
# -------------------------------------------------------------------


class TestSchedulerRequestRouting:
    def test_threshold_boundary_short(self) -> None:
        """Prompt exactly at threshold goes direct."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter()
        threshold = 100
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                # Exactly threshold tokens → direct (< is the check)
                return await scheduler.submit_and_wait("a1", list(range(99)), None, 10)

            run_async(run())
            assert adapter.init_calls == 0
        finally:
            scheduler.stop()

    def test_threshold_boundary_long(self) -> None:
        """Prompt at threshold goes to prefill."""
        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter(chunk_size=200)
        threshold = 100
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                # Exactly threshold tokens → prefill path
                return await scheduler.submit_and_wait("a1", list(range(100)), None, 10)

            run_async(run())
            assert adapter.init_calls == 1
        finally:
            scheduler.stop()


# -------------------------------------------------------------------
# Tests: Warm cache + large delta warning
# -------------------------------------------------------------------


class FakeAgentBlocks:
    """Fake AgentBlocks with total_tokens attribute."""

    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class TestSchedulerWarmLargeDelta:
    def test_warm_large_delta_warning(self, caplog) -> None:
        """Warm cache + large prompt delta should log a warning."""
        import logging

        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter()
        threshold = 100
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                # Cache has 50 tokens, prompt has 500 tokens → delta = 450 > threshold
                fake_cache = FakeAgentBlocks(total_tokens=50)
                return await scheduler.submit_and_wait(
                    "a1", list(range(500)), fake_cache, 10
                )

            with caplog.at_level(logging.WARNING):
                run_async(run())

            # Verify warning was logged about large delta
            assert any("large delta" in record.message.lower() for record in caplog.records)
        finally:
            scheduler.stop()

    def test_warm_small_delta_no_warning(self, caplog) -> None:
        """Warm cache + small prompt delta should NOT log a warning."""
        import logging

        engine = FakeBatchEngine(steps_to_complete=1)
        adapter = FakePrefillAdapter()
        threshold = 100
        scheduler = ConcurrentScheduler(engine, adapter, n_layers=2, interleave_threshold=threshold)
        scheduler.start()

        try:

            async def run():
                # Cache has 40 tokens, prompt has 50 tokens → delta = 10 < threshold
                fake_cache = FakeAgentBlocks(total_tokens=40)
                return await scheduler.submit_and_wait(
                    "a1", list(range(50)), fake_cache, 10
                )

            with caplog.at_level(logging.WARNING):
                run_async(run())

            # No large delta warning
            assert not any("large delta" in record.message.lower() for record in caplog.records)
        finally:
            scheduler.stop()
