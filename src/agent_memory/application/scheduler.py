# mypy: disable-error-code="assignment"
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""ConcurrentScheduler: interleaves chunked prefill with decode steps.

The scheduler runs a dedicated worker thread that alternates between:
1. Draining the request queue for new submissions.
2. Running ONE decode step (one token per active sequence).
3. Processing one prefill chunk for the next queued long-prompt request.

This ensures that a long prefill (e.g. 40K tokens) does not stall an
active generation stream.  Per-token decode granularity enables true
streaming and responsive prefill interleaving.

Architecture layer: application service.
No MLX / infrastructure imports — interacts with adapters through ports.
"""

import asyncio
import logging
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent_memory.application.prefill_state import PrefillState
from agent_memory.domain.errors import InvalidRequestError, PoolExhaustedError
from agent_memory.domain.value_objects import CompletedGeneration, StreamDelta

logger = logging.getLogger(__name__)

INTERLEAVE_THRESHOLD_DEFAULT = 2048


@dataclass
class SchedulerRequest:
    """A pending inference request submitted by the HTTP layer."""

    agent_id: str
    prompt_tokens: list[int]
    cache: Any | None
    max_tokens: int
    prompt_text: str
    future: asyncio.Future[CompletedGeneration]
    loop: asyncio.AbstractEventLoop
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    token_queue: asyncio.Queue[StreamDelta | None] | None = None


class ConcurrentScheduler:
    """Interleaves chunked prefill with decode on a single worker thread.

    Public API (async, called from HTTP layer):
        submit_and_wait()   — enqueue request, return when generation completes.
        submit_and_stream() — enqueue request, yield per-token streaming deltas.
        start() / stop()    — lifecycle management.

    The scheduling loop runs in a daemon thread so the asyncio event loop
    remains free for HTTP requests.
    """

    def __init__(
        self,
        engine: Any,
        prefill_adapter: Any,
        n_layers: int,
        interleave_threshold: int = INTERLEAVE_THRESHOLD_DEFAULT,
        max_batch_size: int = 2,
    ) -> None:
        self._engine = engine
        self._prefill_adapter = prefill_adapter
        self._n_layers = n_layers
        self._interleave_threshold = interleave_threshold
        self._max_batch_size = max_batch_size

        self._request_queue: queue.Queue[SchedulerRequest] = queue.Queue()
        self._waiting_queue: deque[SchedulerRequest] = deque()
        self._prefill_queue: deque[tuple[PrefillState, SchedulerRequest]] = deque()
        self._uid_to_request: dict[str, SchedulerRequest] = {}

        self._running = False
        self._worker_thread: threading.Thread | None = None

    def update_engine(self, new_engine: Any) -> None:
        """Update engine reference after model hot-swap.

        Args:
            new_engine: New BatchEngine instance
        """
        self._engine = new_engine
        logger.info("[SCHEDULER] Engine updated after model swap")

    # ------------------------------------------------------------------
    # Public API (async)
    # ------------------------------------------------------------------

    async def submit_and_wait(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        cache: Any | None,
        max_tokens: int,
        prompt_text: str = "",
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> CompletedGeneration:
        """Submit a request and await its completion.

        Thread-safe: can be called from any asyncio coroutine.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[CompletedGeneration] = loop.create_future()
        request = SchedulerRequest(
            agent_id=agent_id,
            prompt_tokens=prompt_tokens,
            cache=cache,
            max_tokens=max_tokens,
            prompt_text=prompt_text,
            future=future,
            loop=loop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        self._request_queue.put(request)
        return await future

    async def submit_and_stream(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        cache: Any | None,
        max_tokens: int,
        prompt_text: str = "",
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> AsyncIterator[StreamDelta]:
        """Submit a request and yield per-token streaming deltas.

        Yields StreamDelta objects as tokens are generated, enabling
        true SSE streaming through the scheduler's batch=2 path.

        The final delta has finish_reason set. After that, the iterator ends.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[CompletedGeneration] = loop.create_future()
        token_queue: asyncio.Queue[StreamDelta | None] = asyncio.Queue()
        request = SchedulerRequest(
            agent_id=agent_id,
            prompt_tokens=prompt_tokens,
            cache=cache,
            max_tokens=max_tokens,
            prompt_text=prompt_text,
            future=future,
            loop=loop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            token_queue=token_queue,
        )
        self._request_queue.put(request)

        while True:
            delta = await token_queue.get()
            if delta is None:
                break
            yield delta

    def start(self) -> None:
        """Start the scheduling loop in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="scheduler-worker"
        )
        self._worker_thread.start()
        logger.info("[SCHEDULER] Worker thread started")

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        logger.info("[SCHEDULER] Worker thread stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pending_prefills(self) -> int:
        return len(self._prefill_queue)

    @property
    def active_decodes(self) -> int:
        return len(self._uid_to_request)

    @property
    def waiting_requests(self) -> int:
        return len(self._waiting_queue)

    # ------------------------------------------------------------------
    # Scheduling loop (worker thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running:
            did_work = False

            accepted = self._accept_requests()
            if accepted:
                did_work = True

            # Decode-first: one token per active sequence
            if self._engine.has_active_batch():
                self._run_decode_step()
                did_work = True

            # Then prefill: one chunk for the first prefilling sequence
            if self._prefill_queue:
                self._process_one_chunk()
                did_work = True

            if not did_work:
                self._wait_for_request(timeout=0.05)

    def _active_count(self) -> int:
        """Total sequences in-flight: active decodes + prefilling."""
        return len(self._uid_to_request) + len(self._prefill_queue)

    def _accept_requests(self) -> int:
        """Drain request queue; submit up to max_batch_size, queue the rest."""
        count = 0

        # First, drain incoming requests into the waiting queue
        while True:
            try:
                req = self._request_queue.get_nowait()
            except queue.Empty:
                break
            self._waiting_queue.append(req)
            count += 1

        # Then, promote from waiting queue up to max_batch_size
        submitted = self._promote_waiting()

        return count + submitted

    def _promote_waiting(self) -> int:
        """Submit waiting requests up to max_batch_size. Returns count submitted.

        Only promotes when there are free batch slots AND no active decodes.
        Mid-batch insertion (adding a new sequence while another is decoding)
        corrupts the shared Q4 batch cache state in the engine.  We wait for
        the current batch to drain completely before starting the next batch.
        """
        # Block promotion while any sequence is still decoding — the engine's
        # Q4 batch caches are shared tensors that can't be extended mid-decode.
        if self._uid_to_request:
            return 0

        submitted = 0
        while self._waiting_queue and self._active_count() < self._max_batch_size:
            req = self._waiting_queue.popleft()
            self._dispatch_request(req)
            submitted += 1
        if self._waiting_queue:
            logger.debug(
                "[SCHEDULER] %d requests waiting (batch full: %d/%d active)",
                len(self._waiting_queue),
                self._active_count(),
                self._max_batch_size,
            )
        return submitted

    def _dispatch_request(self, req: SchedulerRequest) -> None:
        """Route a single request to direct submit or chunked prefill."""
        n_tokens = len(req.prompt_tokens)

        # Warm requests with cached state: always use direct path.
        # engine.submit() handles character-level prefix matching and
        # only processes new tokens. _enqueue_prefill() creates fresh
        # caches, ignoring the stored KV state entirely.
        if req.cache is not None:
            cached_tokens = req.cache.total_tokens if req.cache else 0
            delta_tokens = n_tokens - cached_tokens
            if delta_tokens > self._interleave_threshold:
                logger.warning(
                    "[SCHEDULER] Warm cache + large delta (%d tokens, "
                    "cached=%d, prompt=%d) — direct path will block "
                    "decode. Consider chunked warm prefill.",
                    delta_tokens,
                    cached_tokens,
                    n_tokens,
                )
            self._submit_direct(req)
        elif n_tokens < self._interleave_threshold:
            self._submit_direct(req)
        else:
            self._enqueue_prefill(req)

    def _submit_direct(self, req: SchedulerRequest) -> None:
        """Short prompt — submit directly to BatchGenerator."""
        try:
            uid = self._engine.submit(
                agent_id=req.agent_id,
                prompt=req.prompt_text,
                cache=req.cache,
                max_tokens=req.max_tokens,
                prompt_tokens=req.prompt_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )
            self._uid_to_request[uid] = req
            logger.debug(
                "[SCHEDULER] Direct submit: agent=%s, tokens=%d, uid=%s",
                req.agent_id,
                len(req.prompt_tokens),
                uid,
            )
        except (PoolExhaustedError, InvalidRequestError) as exc:
            self._reject_request(req, exc)

    def _enqueue_prefill(self, req: SchedulerRequest) -> None:
        """Long prompt — enqueue for chunked interleaved prefill."""
        state = PrefillState(
            agent_id=req.agent_id,
            tokens=req.prompt_tokens,
            max_tokens=req.max_tokens,
        )
        state.kv_caches = self._prefill_adapter.init_prefill_caches(self._n_layers)
        state._request_ref = req
        self._prefill_queue.append((state, req))
        logger.debug(
            "[SCHEDULER] Enqueue prefill: agent=%s, tokens=%d",
            req.agent_id,
            len(req.prompt_tokens),
        )

    def _run_decode_step(self) -> None:
        """Execute ONE batch_gen.next() and dispatch tokens/completions.

        Uses step_once() for per-token granularity. Each call generates
        one token per active sequence, enabling:
        - True streaming: push tokens to client as generated
        - Responsive interleaving: prefill chunks between tokens
        """
        try:
            results = self._engine.step_once()
        except Exception as exc:
            logger.exception("[SCHEDULER] Decode step failed")
            # Reject all in-flight requests to avoid infinite loop
            for _uid, req in list(self._uid_to_request.items()):
                self._reject_request(req, exc)
            self._uid_to_request.clear()
            return

        for result in results:
            req = self._uid_to_request.get(result.uid)
            if req is None:
                continue

            if result.finish_reason is not None:
                # Sequence complete — free a batch slot
                self._uid_to_request.pop(result.uid, None)

                if req.token_queue is not None:
                    # Streaming: push final delta then sentinel
                    delta = StreamDelta(
                        text=result.text,
                        token_count=result.token_count,
                        finish_reason=result.finish_reason,
                    )
                    req.loop.call_soon_threadsafe(req.token_queue.put_nowait, delta)
                    req.loop.call_soon_threadsafe(req.token_queue.put_nowait, None)

                # Resolve future (used by submit_and_wait, ignored if streaming)
                if result.completion is not None:
                    self._resolve_future(req, result.completion)
            # In-progress token — only push to streaming queues
            elif req.token_queue is not None:
                delta = StreamDelta(
                    text=result.text,
                    token_count=result.token_count,
                )
                req.loop.call_soon_threadsafe(req.token_queue.put_nowait, delta)

    def _process_one_chunk(self) -> None:
        """Process one prefill chunk, round-robin across queued sequences.

        When multiple sequences are prefilling, rotates the queue after each
        chunk so each sequence gets fair GPU time. This prevents a single
        long prefill from starving shorter ones that arrive later.
        """
        state, req = self._prefill_queue[0]

        chunk_size = self._prefill_adapter.chunk_size_for_position(state.pos)
        start, end = state.next_chunk_range(chunk_size)

        try:
            self._prefill_adapter.process_prefill_chunk(
                state.tokens,
                start,
                end,
                state.kv_caches,
            )
        except Exception:
            logger.exception(f"[SCHEDULER] Prefill chunk failed: agent={state.agent_id}")
            self._prefill_queue.popleft()
            self._reject_request(req, RuntimeError("Prefill chunk failed"))
            return

        state.advance(end - start)

        if state.is_done:
            self._promote_to_decode(state, req)
            self._prefill_queue.popleft()
        elif len(self._prefill_queue) > 1:
            # Round-robin: rotate current sequence to back of queue
            self._prefill_queue.rotate(-1)

    def _promote_to_decode(self, state: PrefillState, req: SchedulerRequest) -> None:
        """Prefill complete — insert into BatchGenerator for decode."""
        try:
            uid = self._engine.submit_with_cache(
                agent_id=state.agent_id,
                prompt_tokens=state.tokens,
                kv_caches=state.kv_caches,
                max_tokens=state.max_tokens,
                prompt_text=req.prompt_text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )
            self._uid_to_request[uid] = req
            logger.debug(
                "[SCHEDULER] Prefill done, promoted: agent=%s, chunks=%d, uid=%s",
                state.agent_id,
                state.chunk_count,
                uid,
            )
        except (PoolExhaustedError, InvalidRequestError) as exc:
            self._reject_request(req, exc)

    def _wait_for_request(self, timeout: float) -> None:
        """Block until a request arrives or timeout expires."""
        try:
            req = self._request_queue.get(timeout=timeout)
            self._request_queue.put(req)  # Put it back for _accept_requests
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Future resolution helpers
    # ------------------------------------------------------------------

    def _resolve_future(self, req: SchedulerRequest, completion: CompletedGeneration) -> None:
        """Set the result on the asyncio Future (thread-safe)."""
        req.loop.call_soon_threadsafe(req.future.set_result, completion)

    def _reject_request(self, req: SchedulerRequest, exc: Exception) -> None:
        """Set an exception on the asyncio Future (thread-safe)."""
        logger.warning(f"[SCHEDULER] Rejecting request: {exc}")
        # Unblock streaming queue if present (avoids hanging async generator)
        if req.token_queue is not None:
            req.loop.call_soon_threadsafe(req.token_queue.put_nowait, None)
        req.loop.call_soon_threadsafe(req.future.set_exception, exc)
