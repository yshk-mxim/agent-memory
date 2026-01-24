"""Outbound port interfaces (driven adapters).

These ports define the contracts for the application core to interact
with external systems (infrastructure). Implementations are provided by
outbound adapters (MLX backend, disk persistence, etc.).

All interfaces use Protocol (PEP 544) for structural typing, allowing
implicit implementation without inheritance.
"""

from typing import Any, Protocol

from semantic.domain.value_objects import GenerationResult, ModelCacheSpec


class ModelBackendPort(Protocol):
    """Port for model inference backend.

    This port defines the contract for LLM inference operations.
    Implementations wrap specific backends (MLX, vLLM, etc.) with
    a unified interface.
    """

    def generate(
        self,
        prompt_tokens: list[int],
        cache: list[Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate text from tokenized prompt with optional cache.

        Args:
            prompt_tokens: Pre-tokenized input as list of token IDs.
            cache: Optional pre-built KV cache (from previous generation).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            GenerationResult with generated text, tokens, and updated cache.

        Raises:
            ModelNotFoundError: If model not loaded.
        """
        ...

    def extract_model_spec(self) -> ModelCacheSpec:
        """Extract cache specification from the loaded model.

        Returns:
            ModelCacheSpec describing the model's cache geometry
            (layers, attention pattern, block requirements).

        Raises:
            ModelNotFoundError: If no model is loaded.
        """
        ...


class CachePersistencePort(Protocol):
    """Port for KV cache persistence.

    This port defines the contract for saving and loading KV caches
    to/from disk. Implementations handle serialization formats
    (safetensors, HDF5, etc.).
    """

    def save(
        self,
        agent_id: str,
        cache: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save KV cache to disk.

        Args:
            agent_id: Unique identifier for the agent (used as filename).
            cache: List of KV cache objects (per-layer).
            metadata: Optional metadata (model_id, token_count, etc.).

        Raises:
            CachePersistenceError: If save fails (disk full, permissions, etc.).
        """
        ...

    def load(self, agent_id: str) -> tuple[list[Any], dict[str, Any]]:
        """Load KV cache from disk.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Tuple of (cache, metadata).

        Raises:
            AgentNotFoundError: If cache file does not exist.
            CachePersistenceError: If load fails (corruption, version mismatch).
            IncompatibleCacheError: If cache model_id != current model.
        """
        ...

    def exists(self, agent_id: str) -> bool:
        """Check if cache exists on disk.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            True if cache file exists, False otherwise.
        """
        ...

    def delete(self, agent_id: str) -> None:
        """Delete cache from disk.

        Args:
            agent_id: Unique identifier for the agent.

        Raises:
            AgentNotFoundError: If cache file does not exist.
            CachePersistenceError: If deletion fails (permissions, etc.).
        """
        ...

    def list_cached_agents(self) -> list[str]:
        """List all agent IDs with caches on disk.

        Returns:
            List of agent IDs.
        """
        ...


class TokenizerPort(Protocol):
    """Port for tokenization operations.

    This port defines the contract for encoding/decoding text.
    Implementations wrap tokenizer libraries (transformers, tiktoken, etc.).
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID.

        Returns:
            EOS token ID.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens in vocabulary.
        """
        ...
