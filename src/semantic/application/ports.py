"""Application layer ports (interfaces for adapters).

Defines protocols that external adapters must implement to interact
with the application layer. This maintains clean architecture by preventing
direct framework dependencies in the application layer.
"""

from typing import Any, Protocol

from semantic.domain.value_objects import ModelCacheSpec


class SpecExtractorPort(Protocol):
    """Port for extracting ModelCacheSpec from a loaded model."""

    def extract_spec(self, model: Any) -> ModelCacheSpec: ...


class ChatTemplatePort(Protocol):
    """Port for model-specific chat template behavior.

    Abstracts model-family detection (Llama, ChatML, GPT-OSS Harmony)
    away from the application layer so that template-specific logic
    lives in adapters.
    """

    def needs_message_merging(self, tokenizer: Any) -> bool:
        """Whether consecutive same-role messages should be merged."""
        ...

    def get_template_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        """Extra kwargs for tokenizer.apply_chat_template()."""
        ...


class ModelLoaderPort(Protocol):
    """Port for loading and unloading ML models.

    This protocol abstracts away the specific ML framework (MLX, PyTorch, etc.)
    from the application layer, allowing different backends to be swapped.

    Implementations:
        - MLXModelLoader: Apple Silicon optimized via MLX
        - (Future) PyTorchModelLoader: CUDA/CPU via PyTorch
    """

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        """Load a model from HuggingFace.

        Args:
            model_id: HuggingFace model ID
                (e.g., "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            Exception: If model cannot be loaded
        """
        ...

    def get_active_memory(self) -> int:
        """Get currently allocated memory in bytes.

        Returns:
            Memory in bytes currently allocated by the ML framework
        """
        ...

    def clear_cache(self) -> None:
        """Clear ML framework memory cache.

        Frees cached allocations to reclaim memory after model unload.
        """
        ...
