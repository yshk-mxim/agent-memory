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
        max_queue_size: int = 100
    ):
        """
        Initialize concurrent manager.

        Args:
            model_name: HuggingFace model ID or local path
            max_agents: Maximum number of agents in memory
            cache_dir: Directory for cache persistence
            max_queue_size: Maximum requests in queue
        """
        logger.info(f"Initializing ConcurrentAgentManager: {model_name}")

        # Underlying synchronous manager
        self.manager = PersistentAgentManager(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir
        )

        # Request queue (priority queue)
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        # Per-agent locks for sequential per-agent, cross-agent parallel
        self._agent_locks: Dict[str, asyncio.Lock] = {}

        # Metrics
        self.metrics = UtilizationMetrics()
        self.start_time = time.time()
        self._worker_task: Optional[asyncio.Task] = None

        logger.info("ConcurrentAgentManager initialized")

    async def start(self):
        """Start background worker task for processing queue."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.info("Started queue worker task")

    async def stop(self):
        """Stop background worker task."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped queue worker task")

    async def _process_queue(self):
        """
        Background worker that processes requests from queue.

        Runs continuously, pulling requests and executing them.
        """
        logger.info("Queue worker started")

        while True:
            try:
                # Get request from queue (blocks if empty)
                priority, request = await self.request_queue.get()

                # Calculate queue wait time
                wait_time = (datetime.now() - request.created_at).total_seconds() * 1000

                logger.debug(
                    f"Processing request for {request.agent_id}, "
                    f"waited {wait_time:.1f}ms in queue"
                )

                # Update metrics
                self.metrics.active_agents += 1
                self.metrics.queue_depth = self.request_queue.qsize()

                # Execute request (run_in_executor for sync call)
                try:
                    loop = asyncio.get_running_loop()
                    start_time = time.time()

                    response = await loop.run_in_executor(
                        None,
                        self.manager.generate,
                        request.agent_id,
                        request.prompt,
                        request.max_tokens,
                        request.temperature
                    )

                    generation_time = time.time() - start_time

                    # Update metrics
                    self.metrics.completed_requests += 1
                    self.metrics.total_generation_time_sec += generation_time

                    # Update average wait time (exponential moving average)
                    alpha = 0.1
                    self.metrics.avg_queue_wait_ms = (
                        alpha * wait_time +
                        (1 - alpha) * self.metrics.avg_queue_wait_ms
                    )

                    # Set result
                    request.future.set_result(response)

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    request.future.set_exception(e)

                finally:
                    self.metrics.active_agents -= 1
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}")

    async def generate(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        priority: int = 0
    ) -> str:
        """
        Generate response asynchronously (queued).

        Per-agent sequential, cross-agent parallel semantics:
        - Same agent: requests execute sequentially (via per-agent lock)
        - Different agents: requests can execute in parallel (different locks)

        Args:
            agent_id: Unique identifier
            prompt: User input
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            priority: Request priority (lower = higher priority)

        Returns:
            str: Generated response
        """
        # Get or create per-agent lock
        if agent_id not in self._agent_locks:
            self._agent_locks[agent_id] = asyncio.Lock()

        # Acquire lock for this agent (ensures sequential per-agent)
        async with self._agent_locks[agent_id]:
            # Create request
            request = GenerationRequest(
                agent_id=agent_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority
            )

            # Add to queue
            await self.request_queue.put((priority, request))
            self.metrics.total_requests += 1
            self.metrics.queue_depth = self.request_queue.qsize()

            logger.debug(
                f"Queued request for {agent_id}, priority={priority}, "
                f"queue_depth={self.metrics.queue_depth}"
            )

            # Wait for result (lock held until request completes)
            return await request.future

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
