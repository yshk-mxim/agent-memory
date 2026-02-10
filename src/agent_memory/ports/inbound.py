# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Inbound port interfaces (driving adapters).

These ports define the contracts for external systems to interact with
the application core. Implementations are provided by inbound adapters
(API handlers, CLI, etc.).

All interfaces use Protocol (PEP 544) for structural typing, allowing
implicit implementation without inheritance.
"""

from collections.abc import Iterator
from typing import Any, Protocol

from agent_memory.domain.value_objects import CompletedGeneration, GenerationResult


class InferencePort(Protocol):
    """Port for inference operations.

    This port defines the contract for text generation services.
    Implementations may use different backends (MLX, vLLM, etc.)
    but must provide consistent semantics.

    Thread safety: Implementations must be thread-safe for
    concurrent access from multiple agents.
    """

    def generate(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate text continuation from prompt.

        Args:
            agent_id: Unique identifier for the agent making the request.
            prompt: Input text to continue.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy, 1.0 = random).

        Returns:
            GenerationResult with text, tokens, and updated cache.

        Raises:
            AgentNotFoundError: If agent_id does not exist.
            ModelNotFoundError: If model not loaded.
            PoolExhaustedError: If no blocks available for allocation.
            InvalidRequestError: If prompt is empty or parameters invalid.
        """
        ...


class AgentManagementPort(Protocol):
    """Port for agent lifecycle management.

    This port defines the contract for creating, listing, and deleting
    agents (persistent conversation contexts).
    """

    def create_agent(self, agent_id: str, initial_context: str | None = None) -> None:
        """Create a new agent with optional initial context.

        Args:
            agent_id: Unique identifier for the new agent.
            initial_context: Optional initial prompt/context to cache.

        Raises:
            InvalidRequestError: If agent_id already exists or is invalid.
        """
        ...

    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent and free its resources.

        Args:
            agent_id: Unique identifier for the agent to delete.

        Raises:
            AgentNotFoundError: If agent_id does not exist.
        """
        ...

    def list_agents(self) -> list[str]:
        """List all active agent IDs.

        Returns:
            List of agent IDs currently in memory or on disk.
        """
        ...

    def get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """Get information about an agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Dictionary with agent metadata (cache_size, token_count, etc.).

        Raises:
            AgentNotFoundError: If agent_id does not exist.
        """
        ...


class ModelManagementPort(Protocol):
    """Port for model management operations.

    This port defines the contract for loading, unloading, and
    switching between models (hot-swap).
    """

    def load_model(self, model_id: str) -> None:
        """Load a model into memory.

        Args:
            model_id: HuggingFace model ID or local path.

        Raises:
            ModelNotFoundError: If model cannot be found or loaded.
            ModelSwapError: If loading fails.
        """
        ...

    def unload_model(self) -> None:
        """Unload the current model and free GPU memory.

        Raises:
            ModelSwapError: If unloading fails.
        """
        ...

    def get_current_model(self) -> str | None:
        """Get the currently loaded model ID.

        Returns:
            Model ID if a model is loaded, None otherwise.
        """
        ...

    def list_available_models(self) -> list[str]:
        """List all available models (cached + downloadable).

        Returns:
            List of model IDs.
        """
        ...


class GenerationEnginePort(Protocol):
    """Port for async batching inference engine.

    This port defines the contract for batch-based text generation
    where requests are submitted to a queue and processed in batches.
    Distinct from InferencePort which is synchronous request-response.

    Used by application services that need batching semantics
    (e.g., ConcurrentScheduler wrapping BlockPoolBatchEngine).

    Thread safety: Implementations must handle concurrent submit() calls
    but step() is single-threaded (only one caller should call step()).
    """

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: Any | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Submit a generation request to the batch queue.

        Args:
            agent_id: Unique identifier for the agent.
            prompt: Input text to continue.
            cache: Optional pre-built cache (AgentBlocks from previous generation).
            max_tokens: Maximum tokens to generate.

        Returns:
            Request UID for tracking this generation.

        Raises:
            PoolExhaustedError: If no blocks available for allocation.
            InvalidRequestError: If prompt is empty or parameters invalid.
            ModelNotFoundError: If no model is loaded.

        Notes:
            - Non-blocking: returns immediately with UID
            - Actual generation happens during step() calls
            - Multiple submit() calls can be batched together
        """
        ...

    def step(self) -> Iterator[CompletedGeneration]:
        """Execute one batch decode step and yield completed generations.

        Yields:
            CompletedGeneration for each sequence that finished this step.
            Sequences finish when:
            - EOS token generated (finish_reason="stop")
            - max_tokens limit reached (finish_reason="length")
            - Error occurred (finish_reason="error")

        Notes:
            - Call repeatedly until all in-flight requests complete
            - Non-blocking: returns empty iterator if no completions this step
            - Single-threaded: only one caller should invoke step()
            - Batching window: Waits briefly to collect concurrent submits

        Example:
            >>> engine = BlockPoolBatchEngine(...)
            >>> uid1 = engine.submit("agent_a", "Hello", max_tokens=50)
            >>> uid2 = engine.submit("agent_b", "World", max_tokens=50)
            >>> for completion in engine.step():
            ...     print(f"{completion.uid}: {completion.text[:20]}...")
            ...     if completion.finish_reason == "stop":
            ...         print(f"Completed with {completion.token_count} tokens")
        """
        ...
