"""ConcurrentScheduler: interleaves chunked prefill with decode steps.

The scheduler runs a dedicated worker thread that alternates between:
1. Draining the request queue for new submissions.
2. Running one decode step on the BatchGenerator (active sequences).
3. Processing one prefill chunk for the next queued long-prompt request.

This ensures that a long prefill (e.g. 40K tokens) does not stall an
active generation stream.  Decode-first priority minimises inter-token
latency for sequences already generating.

Architecture layer: application service.
No MLX / infrastructure imports — interacts with adapters through ports.
"""

import asyncio
import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from semantic.application.prefill_state import PrefillState
from semantic.domain.errors import InvalidRequestError, PoolExhaustedError
from semantic.domain.value_objects import CompletedGeneration

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


class ConcurrentScheduler:
    """Interleaves chunked prefill with decode on a single worker thread.

    Public API (async, called from HTTP layer):
        submit_and_wait() — enqueue request, return when generation completes.
        start() / stop()  — lifecycle management.

    The scheduling loop runs in a daemon thread so the asyncio event loop
    remains free for HTTP requests.
    """

    def __init__(
        self,
        engine: Any,
        prefill_adapter: Any,
        n_layers: int,
        interleave_threshold: int = INTERLEAVE_THRESHOLD_DEFAULT,
    ) -> None:
        self._engine = engine
        self._prefill_adapter = prefill_adapter
        self._n_layers = n_layers
        self._interleave_threshold = interleave_threshold

        self._request_queue: queue.Queue[SchedulerRequest] = queue.Queue()
        self._prefill_queue: deque[tuple[PrefillState, SchedulerRequest]] = deque()
        self._uid_to_request: dict[str, SchedulerRequest] = {}

        self._running = False
        self._worker_thread: threading.Thread | None = None

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
        )
        self._request_queue.put(request)
        return await future

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

    # ------------------------------------------------------------------
    # Scheduling loop (worker thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running:
            did_work = False

            # 1. Accept new requests
            accepted = self._accept_requests()
            if accepted:
                did_work = True

            # 2. DECODE-FIRST: one step for active sequences
            if self._engine.has_active_batch():
                self._run_decode_step()
                did_work = True

            # 3. THEN PREFILL: one chunk for the first prefilling sequence
            if self._prefill_queue:
                self._process_one_chunk()
                did_work = True

            # 4. If idle, block-wait on request queue
            if not did_work:
                self._wait_for_request(timeout=0.05)

    def _accept_requests(self) -> int:
        """Drain request queue into prefill queue or direct-to-decode."""
        count = 0
        while True:
            try:
                req = self._request_queue.get_nowait()
            except queue.Empty:
                break

            count += 1
            n_tokens = len(req.prompt_tokens)

            # Warm requests with cached state: always use direct path.
            # engine.submit() handles character-level prefix matching and
            # only processes new tokens. _enqueue_prefill() creates fresh
            # caches, ignoring the stored KV state entirely.
            if req.cache is not None:
                self._submit_direct(req)
            elif n_tokens < self._interleave_threshold:
                self._submit_direct(req)
            else:
                self._enqueue_prefill(req)

        return count

    def _submit_direct(self, req: SchedulerRequest) -> None:
        """Short prompt — submit directly to BatchGenerator."""
        try:
            uid = self._engine.submit(
                agent_id=req.agent_id,
                prompt=req.prompt_text,
                cache=req.cache,
                max_tokens=req.max_tokens,
                prompt_tokens=req.prompt_tokens,
            )
            self._uid_to_request[uid] = req
            logger.info(
                f"[SCHEDULER] Direct submit: agent={req.agent_id}, "
                f"tokens={len(req.prompt_tokens)}, uid={uid}"
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
        state.kv_caches = self._prefill_adapter.init_prefill_caches(
            self._n_layers
        )
        state._request_ref = req
        self._prefill_queue.append((state, req))
        logger.info(
            f"[SCHEDULER] Enqueue prefill: agent={req.agent_id}, "
            f"tokens={len(req.prompt_tokens)}"
        )

    def _run_decode_step(self) -> None:
        """Execute one decode step and dispatch completions."""
        try:
            completions = list(self._engine.step())
        except Exception:
            logger.exception("[SCHEDULER] Decode step failed")
            return

        for completion in completions:
            req = self._uid_to_request.pop(completion.uid, None)
            if req is not None:
                self._resolve_future(req, completion)

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
                state.tokens, start, end, state.kv_caches,
            )
        except Exception:
            logger.exception(
                f"[SCHEDULER] Prefill chunk failed: agent={state.agent_id}"
            )
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
            )
            self._uid_to_request[uid] = req
            logger.info(
                f"[SCHEDULER] Prefill done, promoted: agent={state.agent_id}, "
                f"chunks={state.chunk_count}, uid={uid}"
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

    def _resolve_future(
        self, req: SchedulerRequest, completion: CompletedGeneration
    ) -> None:
        """Set the result on the asyncio Future (thread-safe)."""
        req.loop.call_soon_threadsafe(req.future.set_result, completion)

    def _reject_request(self, req: SchedulerRequest, exc: Exception) -> None:
        """Set an exception on the asyncio Future (thread-safe)."""
        logger.warning(f"[SCHEDULER] Rejecting request: {exc}")
        req.loop.call_soon_threadsafe(req.future.set_exception, exc)
