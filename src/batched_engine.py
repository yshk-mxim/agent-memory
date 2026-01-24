"""
Batched Generation Engine

Wraps mlx_lm's BatchGenerator for multi-agent continuous batching with
persistent per-agent KV caches.

Features:
- Continuous batching using mlx_lm.BatchGenerator
- Per-sequence cache extraction for persistence
- Agent-to-UID mapping for tracking
- Supports existing caches (resume from disk)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    """Tracks an agent's generation request."""
    agent_id: str
    uid: int


@dataclass
class CompletedGeneration:
    """Result of a completed generation."""
    uid: int
    agent_id: str
    text: str
    cache: List[Any]  # Per-sequence KVCache


class BatchedGenerationEngine:
    """
    Wraps mlx_lm's BatchGenerator for multi-agent continuous batching.

    Provides continuous batching with persistent per-agent KV caches:
    - Insert new generation requests with optional pre-computed caches
    - Step through decode loop yielding completed generations
    - Extract per-sequence caches for persistence
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 5,
        prefill_step_size: int = 512
    ):
        """
        Initialize batched generation engine.

        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance
            max_batch_size: Maximum sequences in a batch
            prefill_step_size: Tokens to process per prefill step
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.prefill_step_size = prefill_step_size
        self._batch_gen: Optional[BatchGenerator] = None
        self._active_requests: Dict[int, AgentRequest] = {}  # uid → request

        logger.info(
            f"BatchedGenerationEngine initialized: "
            f"max_batch_size={max_batch_size}"
        )

    def start(self):
        """Initialize the BatchGenerator."""
        if self._batch_gen is None:
            self._batch_gen = BatchGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                stop_strings=[self.tokenizer.eos_token],
                prefill_step_size=self.prefill_step_size
            )
            logger.info("BatchGenerator initialized")

    def has_active_requests(self) -> bool:
        """Check if there are active requests being processed."""
        return len(self._active_requests) > 0

    def submit(
        self,
        agent_id: str,
        prompt: str,
        existing_cache: Optional[List] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> int:
        """
        Submit a generation request.

        Args:
            agent_id: Unique agent identifier
            prompt: Input prompt
            existing_cache: Optional pre-computed KV cache (from disk)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            int: UID for tracking this request
        """
        if self._batch_gen is None:
            raise RuntimeError("Engine not started. Call start() first.")

        # Create sampler
        sampler = make_sampler(temperature)

        # Insert into BatchGenerator
        # If existing_cache is provided, it will be merged into the batch
        uids = self._batch_gen.insert(
            prompts=[prompt],
            max_tokens=[max_tokens],
            prompt_cache=[existing_cache] if existing_cache else None,
            samplers=[sampler],
        )

        uid = uids[0]
        self._active_requests[uid] = AgentRequest(agent_id=agent_id, uid=uid)

        logger.debug(
            f"Submitted request uid={uid} for agent={agent_id}, "
            f"active_requests={len(self._active_requests)}"
        )

        return uid

    def step(self) -> List[CompletedGeneration]:
        """
        Run one decode step.

        Processes all active requests in the batch, generating one token per
        sequence. Returns list of completed generations.

        Returns:
            List[CompletedGeneration]: Generations that completed this step
        """
        if self._batch_gen is None:
            raise RuntimeError("Engine not started. Call start() first.")

        if not self._active_requests:
            return []

        completed = []

        try:
            # Run one step of the batch generator
            # This generates one token for each active sequence
            results = self._batch_gen.next()

            # Process results
            for uid, text, metadata in results:
                if metadata.get("done", False):
                    # Generation complete - extract cache
                    if uid not in self._active_requests:
                        logger.warning(f"Completed UID {uid} not in active requests")
                        continue

                    request = self._active_requests[uid]

                    # Extract per-sequence cache from the batch
                    # This gives us the full KV cache for this sequence
                    cache = self._batch_gen.batch.extract_cache(uid)

                    completed.append(CompletedGeneration(
                        uid=uid,
                        agent_id=request.agent_id,
                        text=text,
                        cache=cache,
                    ))

                    # Remove from active
                    del self._active_requests[uid]

                    logger.debug(
                        f"Request uid={uid} completed, "
                        f"remaining={len(self._active_requests)}"
                    )

        except StopIteration:
            # Batch exhausted (all sequences complete)
            logger.debug("Batch exhausted, no more active requests")

        return completed

    def step_until_done(self) -> List[CompletedGeneration]:
        """
        Run decode loop until all active requests complete.

        Returns:
            List[CompletedGeneration]: All completed generations
        """
        all_completed = []
        while self._active_requests:
            completed = self.step()
            all_completed.extend(completed)
        return all_completed

    def get_active_count(self) -> int:
        """Get number of active requests."""
        return len(self._active_requests)

    def submit_with_cached_agent(
        self,
        agent_id: str,
        prompt: str,
        persistence,  # CachePersistence instance
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> int:
        """
        Submit request, loading agent's cache from disk if available.

        This enables cache persistence across server restarts:
        - If agent exists on disk: load cache → merge into batch
        - If agent is new: start with empty cache

        Args:
            agent_id: Unique agent identifier
            prompt: Input prompt
            persistence: CachePersistence instance
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            int: UID for tracking this request
        """
        existing_cache = None

        # Try loading cache from disk
        if persistence.agent_cache_exists(agent_id):
            try:
                existing_cache, metadata = persistence.load_agent_cache(agent_id)
                logger.info(
                    f"Loaded cache for agent={agent_id} with "
                    f"{metadata.get('cache_tokens', 0)} tokens"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache for {agent_id}: {e}")
                existing_cache = None

        return self.submit(
            agent_id=agent_id,
            prompt=prompt,
            existing_cache=existing_cache,
            max_tokens=max_tokens,
            temperature=temperature
        )
