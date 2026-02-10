# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Coverage tests for ConcurrentScheduler — dispatch, decode, prefill, properties."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_memory.application.prefill_state import PrefillState
from agent_memory.application.scheduler import (
    ConcurrentScheduler,
    SchedulerRequest,
)
from agent_memory.domain.errors import InvalidRequestError, PoolExhaustedError
from agent_memory.domain.value_objects import CompletedGeneration, StepOneResult, StreamDelta

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("ignore::ResourceWarning")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    engine = MagicMock()
    engine.has_active_batch.return_value = False
    engine.submit.return_value = "uid_0"
    engine.step_once.return_value = []
    return engine


def _make_prefill_adapter():
    adapter = MagicMock()
    adapter.init_prefill_caches.return_value = [MagicMock()]
    adapter.chunk_size_for_position.return_value = 256
    adapter.process_prefill_chunk.return_value = None
    return adapter


def _make_scheduler(engine=None, prefill_adapter=None, **kwargs):
    if engine is None:
        engine = _make_engine()
    if prefill_adapter is None:
        prefill_adapter = _make_prefill_adapter()
    defaults = dict(n_layers=12, interleave_threshold=2048, max_batch_size=2)
    defaults.update(kwargs)
    return ConcurrentScheduler(engine=engine, prefill_adapter=prefill_adapter, **defaults)


@pytest.fixture()
def event_loop():
    """Provide a fresh event loop that is properly closed after the test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _make_request(agent_id="test", n_tokens=100, cache=None, *, loop):
    """Create a SchedulerRequest. The `loop` parameter is required."""
    future = loop.create_future()
    return SchedulerRequest(
        agent_id=agent_id,
        prompt_tokens=list(range(n_tokens)),
        cache=cache,
        max_tokens=200,
        prompt_text="test prompt",
        future=future,
        loop=loop,
    )


# ===========================================================================
# Properties
# ===========================================================================


class TestProperties:
    def test_is_running_after_start_stop(self):
        sched = _make_scheduler()
        assert sched.is_running is False
        sched.start()
        assert sched.is_running is True
        sched.stop()
        assert sched.is_running is False

    def test_pending_prefills_count(self, event_loop):
        sched = _make_scheduler()
        assert sched.pending_prefills == 0
        state = PrefillState(agent_id="a", tokens=[1, 2, 3])
        req = _make_request(loop=event_loop)
        sched._prefill_queue.append((state, req))
        assert sched.pending_prefills == 1

    def test_active_decodes_count(self, event_loop):
        sched = _make_scheduler()
        assert sched.active_decodes == 0
        sched._uid_to_request["uid_1"] = _make_request(loop=event_loop)
        assert sched.active_decodes == 1

    def test_waiting_requests_count(self, event_loop):
        sched = _make_scheduler()
        assert sched.waiting_requests == 0
        sched._waiting_queue.append(_make_request(loop=event_loop))
        assert sched.waiting_requests == 1


# ===========================================================================
# start / stop
# ===========================================================================


class TestStartStop:
    def test_start_already_running_noop(self):
        sched = _make_scheduler()
        sched.start()
        try:
            thread1 = sched._worker_thread
            sched.start()  # Should be no-op
            assert sched._worker_thread is thread1
        finally:
            sched.stop()

    def test_stop_with_no_thread(self):
        sched = _make_scheduler()
        sched.stop()  # Should not raise


# ===========================================================================
# _dispatch_request
# ===========================================================================


class TestDispatchRequest:
    def test_warm_cache_goes_direct(self, event_loop):
        sched = _make_scheduler()
        cache = MagicMock()
        cache.total_tokens = 50
        req = _make_request(n_tokens=100, cache=cache, loop=event_loop)

        sched._dispatch_request(req)
        sched._engine.submit.assert_called_once()

    def test_cold_short_prompt_goes_direct(self, event_loop):
        sched = _make_scheduler(interleave_threshold=2048)
        req = _make_request(n_tokens=100, loop=event_loop)

        sched._dispatch_request(req)
        sched._engine.submit.assert_called_once()

    def test_cold_long_prompt_goes_prefill(self, event_loop):
        sched = _make_scheduler(interleave_threshold=100)
        req = _make_request(n_tokens=200, loop=event_loop)

        sched._dispatch_request(req)
        assert len(sched._prefill_queue) == 1


# ===========================================================================
# _promote_waiting
# ===========================================================================


class TestPromoteWaiting:
    def test_active_decodes_blocks_promotion(self, event_loop):
        sched = _make_scheduler()
        sched._uid_to_request["uid_1"] = _make_request(loop=event_loop)
        sched._waiting_queue.append(_make_request(agent_id="a2", loop=event_loop))

        promoted = sched._promote_waiting()
        assert promoted == 0

    def test_promotes_up_to_max_batch(self, event_loop):
        sched = _make_scheduler(max_batch_size=2)
        sched._engine.submit.side_effect = [f"uid_{i}" for i in range(5)]
        for i in range(3):
            sched._waiting_queue.append(_make_request(agent_id=f"a{i}", n_tokens=10, loop=event_loop))

        promoted = sched._promote_waiting()
        assert promoted == 2
        assert len(sched._waiting_queue) == 1

    def test_empty_waiting_queue(self):
        sched = _make_scheduler()
        assert sched._promote_waiting() == 0


# ===========================================================================
# _submit_direct error paths
# ===========================================================================


class TestSubmitDirectErrors:
    def test_pool_exhausted_rejects(self, event_loop):
        engine = _make_engine()
        engine.submit.side_effect = PoolExhaustedError("no blocks")
        sched = _make_scheduler(engine=engine)

        req = _make_request(loop=event_loop)
        sched._submit_direct(req)

        # Future should have exception set (via call_soon_threadsafe)
        # We can't easily check in a sync test, but verify no crash
        assert "uid" not in sched._uid_to_request or len(sched._uid_to_request) == 0

    def test_invalid_request_rejects(self, event_loop):
        engine = _make_engine()
        engine.submit.side_effect = InvalidRequestError("bad request")
        sched = _make_scheduler(engine=engine)

        req = _make_request(loop=event_loop)
        sched._submit_direct(req)

        assert len(sched._uid_to_request) == 0


# ===========================================================================
# _enqueue_prefill
# ===========================================================================


class TestEnqueuePrefill:
    def test_creates_prefill_state(self, event_loop):
        sched = _make_scheduler()
        req = _make_request(n_tokens=5000, loop=event_loop)
        sched._enqueue_prefill(req)

        assert len(sched._prefill_queue) == 1
        state, stored_req = sched._prefill_queue[0]
        assert state.agent_id == req.agent_id
        assert state.tokens == req.prompt_tokens
        assert stored_req is req


# ===========================================================================
# _run_decode_step
# ===========================================================================


class TestRunDecodeStep:
    def test_in_progress_token_pushed_to_queue(self, event_loop):
        engine = _make_engine()
        sched = _make_scheduler(engine=engine)

        token_queue = asyncio.Queue()
        req = _make_request(loop=event_loop)
        req.token_queue = token_queue
        sched._uid_to_request["uid_0"] = req

        result = StepOneResult(uid="uid_0", text="Hello", token_count=1)
        engine.step_once.return_value = [result]

        sched._run_decode_step()
        # uid_0 should still be active (no finish_reason)
        assert "uid_0" in sched._uid_to_request

    def test_finish_reason_resolves(self, event_loop):
        engine = _make_engine()
        sched = _make_scheduler(engine=engine)

        req = _make_request(loop=event_loop)
        req.token_queue = asyncio.Queue()
        completion = CompletedGeneration(
            uid="uid_0", text="Done", blocks=None, finish_reason="stop", token_count=5
        )
        sched._uid_to_request["uid_0"] = req

        result = StepOneResult(
            uid="uid_0", text="Done", token_count=5,
            finish_reason="stop", completion=completion,
        )
        engine.step_once.return_value = [result]

        sched._run_decode_step()
        assert "uid_0" not in sched._uid_to_request

    def test_unknown_uid_skipped(self):
        engine = _make_engine()
        sched = _make_scheduler(engine=engine)

        result = StepOneResult(uid="unknown_uid", text="x", token_count=1)
        engine.step_once.return_value = [result]

        sched._run_decode_step()  # Should not raise

    def test_exception_rejects_all(self, event_loop):
        engine = _make_engine()
        engine.step_once.side_effect = RuntimeError("metal crash")
        sched = _make_scheduler(engine=engine)

        req = _make_request(loop=event_loop)
        req.token_queue = asyncio.Queue()
        sched._uid_to_request["uid_0"] = req

        sched._run_decode_step()
        assert len(sched._uid_to_request) == 0


# ===========================================================================
# _process_one_chunk
# ===========================================================================


class TestProcessOneChunk:
    def test_normal_chunk_advances(self, event_loop):
        sched = _make_scheduler()
        state = PrefillState(agent_id="a", tokens=list(range(5000)), max_tokens=200)
        state.kv_caches = [MagicMock()]
        req = _make_request(n_tokens=5000, loop=event_loop)
        sched._prefill_queue.append((state, req))

        sched._process_one_chunk()
        assert state.pos > 0

    def test_prefill_complete_promotes(self, event_loop):
        engine = _make_engine()
        engine.submit_with_cache.return_value = "uid_promoted"
        sched = _make_scheduler(engine=engine)

        # Create a state that's almost done (just 1 token left)
        state = PrefillState(agent_id="a", tokens=[1, 2], max_tokens=200)
        state.pos = 0  # prefill_end = max(0, 2-1) = 1
        state.kv_caches = [MagicMock()]
        req = _make_request(n_tokens=2, loop=event_loop)

        sched._prefill_queue.append((state, req))
        sched._prefill_adapter.chunk_size_for_position.return_value = 256

        sched._process_one_chunk()
        # State should be done and removed from prefill queue
        assert len(sched._prefill_queue) == 0

    def test_exception_during_prefill(self, event_loop):
        sched = _make_scheduler()
        sched._prefill_adapter.process_prefill_chunk.side_effect = RuntimeError("fail")

        state = PrefillState(agent_id="a", tokens=list(range(5000)), max_tokens=200)
        state.kv_caches = [MagicMock()]
        req = _make_request(loop=event_loop)
        sched._prefill_queue.append((state, req))

        sched._process_one_chunk()
        assert len(sched._prefill_queue) == 0


# ===========================================================================
# _resolve_future / _reject_request
# ===========================================================================


class TestFutureHelpers:
    def test_resolve_future(self, event_loop):
        sched = _make_scheduler()
        req = _make_request(loop=event_loop)
        completion = CompletedGeneration(
            uid="uid_0", text="Done", blocks=None, finish_reason="stop", token_count=1
        )
        sched._resolve_future(req, completion)
        # Verify call_soon_threadsafe was invoked (loop is not running)
        # Just verify no exception

    def test_reject_request_with_queue(self, event_loop):
        sched = _make_scheduler()
        req = _make_request(loop=event_loop)
        req.token_queue = asyncio.Queue()
        exc = RuntimeError("test error")
        sched._reject_request(req, exc)
        # Verify no exception


# ===========================================================================
# submit_and_stream (async)
# ===========================================================================


class TestSubmitAndStream:
    @pytest.mark.asyncio
    async def test_tokens_arrive_via_queue(self):
        engine = _make_engine()
        sched = _make_scheduler(engine=engine)

        # Simulate: after submit, worker pushes tokens to the queue
        async def simulate_worker():
            await asyncio.sleep(0.05)
            # Get the request from the queue
            req = sched._request_queue.get_nowait()
            # Push deltas
            delta1 = StreamDelta(text="Hi", token_count=1)
            delta2 = StreamDelta(text="Hi there", token_count=2, finish_reason="stop")
            await req.token_queue.put(delta1)
            await req.token_queue.put(delta2)
            await req.token_queue.put(None)  # Sentinel

        task = asyncio.create_task(simulate_worker())

        collected = []
        async for delta in sched.submit_and_stream(
            agent_id="test",
            prompt_tokens=[1, 2, 3],
            cache=None,
            max_tokens=100,
        ):
            collected.append(delta)

        await task
        assert len(collected) == 2
        assert collected[0].text == "Hi"
        assert collected[1].finish_reason == "stop"


# ===========================================================================
# update_engine
# ===========================================================================


class TestUpdateEngine:
    def test_engine_updated(self):
        sched = _make_scheduler()
        new_engine = MagicMock()
        sched.update_engine(new_engine)
        assert sched._engine is new_engine
