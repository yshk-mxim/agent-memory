"""
Concurrent Agent Processing Manager

Async wrapper around PersistentAgentManager for concurrent multi-agent processing.
Enables processing Agent B while Agent A is blocked on tool/code execution.

Inspired by Halo's batch query processing for agentic workflows.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from .agent_manager import PersistentAgentManager
from .batched_engine import BatchedGenerationEngine

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Single generation request in queue."""
    agent_id: str
    prompt: str
    max_tokens: int
    temperature: float
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class UtilizationMetrics:
    """Agent utilization and throughput metrics."""
    total_requests: int = 0
    completed_requests: int = 0
    active_agents: int = 0
    queue_depth: int = 0
    avg_queue_wait_ms: float = 0.0
    total_generation_time_sec: float = 0.0
    throughput_req_per_sec: float = 0.0


class ConcurrentAgentManager:
    """
    Concurrent multi-agent processing with async queue scheduling.

    Features:
    - Async wrapper around synchronous PersistentAgentManager
    - Priority queue for request scheduling
    - Concurrent generation across multiple agents
    - Utilization metrics and monitoring

    Limitation:
    - MLX is synchronous and uses single GPU, so true parallelism is limited
    - Benefit comes from processing Agent B while Agent A is idle (tool calls, etc.)
    """

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-12b-it-4bit",
        max_agents: int = 3,
        cache_dir: str = "~/.agent_caches",
        max_batch_size: int = 5,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64
    ):
        """
        Initialize concurrent manager with batched generation.

        Args:
            model_name: HuggingFace model ID or local path
            max_agents: Maximum number of agents in memory
            cache_dir: Directory for cache persistence
            max_batch_size: Maximum sequences in a batch
            kv_bits: Optional KV cache quantization (2-8 bits, None=no quantization)
            kv_group_size: Group size for quantization (default 64)
        """
        logger.info(
            f"Initializing ConcurrentAgentManager: {model_name}, "
            f"batch_size={max_batch_size}, kv_bits={kv_bits}"
        )

        # Underlying synchronous manager
        self.manager = PersistentAgentManager(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size
        )

        # Batched generation engine
        self.engine = BatchedGenerationEngine(
            model=self.manager.model,
            tokenizer=self.manager.tokenizer,
            max_batch_size=max_batch_size
        )

        # Per-agent locks for sequential per-agent, cross-agent parallel
        self._agent_locks: Dict[str, asyncio.Lock] = {}

        # Tracking for pending futures (uid → future)
        self._pending_futures: Dict[int, asyncio.Future] = {}

        # Event to signal new work available
        self._submit_event = asyncio.Event()

        # Metrics
        self.metrics = UtilizationMetrics()
        self.start_time = time.time()
        self._worker_task: Optional[asyncio.Task] = None

        logger.info(f"ConcurrentAgentManager initialized with batch_size={max_batch_size}")

    async def start(self):
        """Start background worker task for batch processing."""
        # Start the batch engine
        self.engine.start()

        # Start the batch worker
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info("Started batch worker task")

    async def stop(self):
        """Stop background worker task."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped queue worker task")

    async def _batch_worker(self):
        """
        Background worker that processes batched requests.

        Collects pending requests and processes them in batches using the
        BatchedGenerationEngine. Runs continuously until cancelled.
        """
        logger.info("Batch worker started")

        while True:
            try:
                # Wait for work
                await self._submit_event.wait()
                self._submit_event.clear()

                # Optional: small delay to collect more requests for batching
                # This allows concurrent requests to "batch up" before processing
                await asyncio.sleep(0.01)  # 10ms batching window

                # Process all active requests
                while self.engine.has_active_requests():
                    # Run one decode step (generates one token per sequence)
                    loop = asyncio.get_running_loop()
                    completed = await loop.run_in_executor(
                        None,
                        self.engine.step
                    )

                    # Resolve futures for completed generations
                    for result in completed:
                        uid = result.uid
                        if uid in self._pending_futures:
                            self._pending_futures[uid].set_result(result)
                            del self._pending_futures[uid]

                            # Update metrics
                            self.metrics.completed_requests += 1

                            logger.debug(
                                f"Request uid={uid} for agent={result.agent_id} completed"
                            )

            except asyncio.CancelledError:
                logger.info("Batch worker cancelled")
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}")

    async def generate(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        priority: int = 0
    ) -> str:
        """
        Generate response asynchronously with batched processing.

        Per-agent sequential, cross-agent parallel semantics:
        - Same agent: requests execute sequentially (via per-agent lock)
        - Different agents: requests can execute in parallel (batched together)

        Args:
            agent_id: Unique identifier
            prompt: User input
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            priority: Request priority (unused with batch engine)

        Returns:
            str: Generated response
        """
        # Get or create per-agent lock
        if agent_id not in self._agent_locks:
            self._agent_locks[agent_id] = asyncio.Lock()

        # Acquire lock for this agent (ensures sequential per-agent)
        async with self._agent_locks[agent_id]:
            # Load agent's current cache (includes updates from previous requests)
            cache = self.manager.get_agent_cache(agent_id)

            # Submit to batch engine (non-blocking)
            uid = self.engine.submit(
                agent_id=agent_id,
                prompt=prompt,
                existing_cache=cache,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Create future for this request
            future = asyncio.get_event_loop().create_future()
            self._pending_futures[uid] = future

            # Signal worker that new work is available
            self._submit_event.set()

            # Update metrics
            self.metrics.total_requests += 1

            logger.debug(
                f"Submitted uid={uid} for agent={agent_id}, "
                f"active_requests={self.engine.get_active_count()}"
            )

            # Wait for this specific generation to complete
            result = await future

            # Update agent's cache with new state
            self.manager.update_agent_cache(agent_id, result.cache)

            # Return generated text
            return result.text

    async def generate_concurrent(
        self,
        requests: List[Tuple[str, str, int]]
    ) -> List[str]:
        """
        Generate responses for multiple agents concurrently.

        Args:
            requests: List of (agent_id, prompt, max_tokens) tuples

        Returns:
            List[str]: Generated responses in same order as requests
        """
        logger.info(f"Processing {len(requests)} concurrent requests")

        # Create tasks for all requests
        tasks = []
        for agent_id, prompt, max_tokens in requests:
            task = asyncio.create_task(
                self.generate(agent_id, prompt, max_tokens)
            )
            tasks.append(task)

        # Wait for all to complete
        responses = await asyncio.gather(*tasks)

        logger.info(f"Completed {len(responses)} concurrent requests")

        return responses

    @property
    def agents(self):
        """Access to underlying agent registry."""
        return self.manager.agents

    def create_agent(
        self,
        agent_id: str,
        agent_type: str,
        system_prompt: str
    ):
        """
        Create agent (synchronous, delegates to manager).

        Args:
            agent_id: Unique identifier
            agent_type: Agent role/type
            system_prompt: System-level instructions
        """
        return self.manager.create_agent(agent_id, agent_type, system_prompt)

    def load_agent(self, agent_id: str):
        """
        Load agent from disk (synchronous, delegates to manager).

        Args:
            agent_id: Unique identifier
        """
        return self.manager.load_agent(agent_id)

    def save_agent(self, agent_id: str):
        """
        Save agent to disk (synchronous, delegates to manager).

        Args:
            agent_id: Unique identifier
        """
        return self.manager.save_agent(agent_id)

    def get_utilization(self) -> Dict[str, Any]:
        """
        Get current utilization metrics.

        Returns:
            dict with keys:
            - total_requests: Total requests submitted
            - completed_requests: Completed requests
            - active_agents: Currently processing agents
            - queue_depth: Requests waiting in queue
            - avg_queue_wait_ms: Average queue wait time
            - throughput_req_per_sec: Requests per second
            - uptime_sec: Time since manager started
        """
        uptime = time.time() - self.start_time

        # Calculate throughput
        if uptime > 0:
            throughput = self.metrics.completed_requests / uptime
        else:
            throughput = 0.0

        return {
            "total_requests": self.metrics.total_requests,
            "completed_requests": self.metrics.completed_requests,
            "active_agents": self.metrics.active_agents,
            "queue_depth": self.metrics.queue_depth,
            "avg_queue_wait_ms": self.metrics.avg_queue_wait_ms,
            "throughput_req_per_sec": throughput,
            "uptime_sec": uptime,
            "total_generation_time_sec": self.metrics.total_generation_time_sec
        }
