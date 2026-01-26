"""Model hot-swap orchestrator for dynamic model switching.

Coordinates the full hot-swap sequence:
1. Drain active requests from BatchEngine
2. Evict all caches to disk via AgentCacheStore
3. Shutdown BatchEngine
4. Unload old model via ModelRegistry
5. Load new model via ModelRegistry
6. Reconfigure BlockPool for new model spec
7. Create new BatchEngine with new model
8. Update application state

Implements rollback on failure to restore previous model.
"""

import logging
from typing import Any

from semantic.application.agent_cache_store import AgentCacheStore, ModelTag
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.application.model_registry import ModelRegistry
from semantic.domain.errors import ModelNotFoundError
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

logger = logging.getLogger(__name__)


class ModelSwapOrchestrator:
    """Orchestrates hot-swap of models while preserving agent caches.

    Responsibilities:
    - Coordinate drain → evict → unload → load → reconfigure sequence
    - Handle errors and rollback to previous model
    - Ensure no memory leaks or orphaned blocks
    - Provide swap progress/status reporting

    Thread Safety:
    - Not thread-safe. Caller must ensure exclusive access during swap.
    - Designed for single-threaded FastAPI lifespan management.

    Example:
        >>> orchestrator = ModelSwapOrchestrator(registry, pool, cache_store, cache_adapter)
        >>> old_engine = app.state.batch_engine
        >>> new_engine = orchestrator.swap_model(
        ...     old_engine=old_engine,
        ...     new_model_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
        ...     timeout_seconds=60.0,
        ... )
        >>> app.state.batch_engine = new_engine
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        block_pool: BlockPool,
        cache_store: AgentCacheStore,
        cache_adapter: Any,  # CacheOperationsPort (MLXCacheAdapter)
    ) -> None:
        """Initialize orchestrator with dependencies.

        Args:
            model_registry: ModelRegistry for load/unload operations
            block_pool: BlockPool for reconfiguration
            cache_store: AgentCacheStore for cache eviction
            cache_adapter: CacheOperationsPort for BatchEngine creation
        """
        self._registry = model_registry
        self._pool = block_pool
        self._cache_store = cache_store
        self._cache_adapter = cache_adapter

    def swap_model(
        self,
        old_engine: BlockPoolBatchEngine | None,
        new_model_id: str,
        timeout_seconds: float = 30.0,
    ) -> BlockPoolBatchEngine:
        """Execute full hot-swap sequence.

        Args:
            old_engine: Current BatchEngine (None if first load)
            new_model_id: HuggingFace model ID to load
            timeout_seconds: Max time to wait for drain (default: 30s)

        Returns:
            New BatchEngine with loaded model

        Raises:
            ModelNotFoundError: If new model cannot be loaded
            GenerationError: If drain times out
            PoolReconfigurationError: If active allocations prevent reconfigure

        Notes:
            - On failure, attempts to rollback to old model
            - If rollback fails, raises error (system in degraded state)
            - Caller must update app.state.batch_engine with returned engine

        Example:
            >>> new_engine = orchestrator.swap_model(
            ...     old_engine=app.state.batch_engine,
            ...     new_model_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
            ... )
            >>> app.state.batch_engine = new_engine
        """
        old_model_id = self._registry.get_current_id()
        logger.info(f"Starting model swap: {old_model_id} → {new_model_id}")

        # Store old spec for rollback
        old_spec = self._registry.get_current_spec()

        try:
            # Step 1: Drain active requests (if engine exists)
            if old_engine is not None:
                logger.info("Step 1/7: Draining active requests...")
                old_engine.drain(timeout_seconds=timeout_seconds)
                logger.info("Drain complete")

            # Step 2: Evict all caches to disk
            logger.info("Step 2/7: Evicting all caches to disk...")
            evicted_count = self._cache_store.evict_all_to_disk()
            logger.info(f"Evicted {evicted_count} caches")

            # Step 3: Shutdown old BatchEngine
            if old_engine is not None:
                logger.info("Step 3/7: Shutting down BatchEngine...")
                old_engine.shutdown()
                logger.info("BatchEngine shutdown complete")

            # Step 4: Unload old model
            if old_model_id is not None:
                logger.info(f"Step 4/7: Unloading model {old_model_id}...")
                self._registry.unload_model()
                logger.info("Model unloaded")

            # Step 5: Load new model
            logger.info(f"Step 5/7: Loading model {new_model_id}...")
            new_model, new_tokenizer = self._registry.load_model(new_model_id)
            new_spec = self._registry.get_current_spec()
            if new_spec is None:
                raise ModelNotFoundError(f"Failed to extract spec for {new_model_id}")
            logger.info("Model loaded")

            # Step 6: Reconfigure BlockPool for new model dimensions
            logger.info("Step 6/7: Reconfiguring BlockPool...")
            self._pool.reconfigure(new_spec)
            logger.info("BlockPool reconfigured")

            # Step 7: Update cache store model tag
            logger.info("Step 7/8: Updating cache store model tag...")
            new_tag = self._create_model_tag(new_model_id, new_spec)
            self._cache_store.update_model_tag(new_tag)
            logger.info("Model tag updated")

            # Step 8: Create new BatchEngine with new model
            logger.info("Step 8/8: Creating new BatchEngine...")
            new_engine = BlockPoolBatchEngine(
                model=new_model,
                tokenizer=new_tokenizer,
                pool=self._pool,
                spec=new_spec,
                cache_adapter=self._cache_adapter,
            )
            logger.info("BatchEngine created")

            logger.info(f"Model swap complete: {old_model_id} → {new_model_id}")
            return new_engine

        except Exception as e:
            logger.error(f"Model swap failed: {e}")

            # Attempt rollback to old model
            if old_model_id is not None and old_spec is not None:
                logger.warning(f"Attempting rollback to {old_model_id}...")
                try:
                    self._rollback(old_model_id, old_spec)
                    logger.info("Rollback successful")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
                    logger.critical(
                        "System in degraded state - manual intervention required"
                    )
                    raise  # Re-raise original error

            # Re-raise original swap error
            raise

    def _rollback(self, old_model_id: str, old_spec: ModelCacheSpec) -> None:
        """Rollback to previous model after failed swap.

        Args:
            old_model_id: Previous model ID
            old_spec: Previous model spec

        Raises:
            Exception: If rollback fails (system in degraded state)
        """
        logger.info(f"Rolling back to {old_model_id}...")

        # Unload failed new model (if any)
        if self._registry.is_loaded():
            self._registry.unload_model()

        # Reload old model
        self._registry.load_model(old_model_id)

        # Reconfigure pool back to old spec
        self._pool.reconfigure(old_spec)

        logger.info(f"Rollback to {old_model_id} complete")

    def _create_model_tag(self, model_id: str, spec: ModelCacheSpec) -> ModelTag:
        """Create ModelTag from model ID and spec.

        Args:
            model_id: HuggingFace model ID
            spec: Model cache specification

        Returns:
            ModelTag for cache compatibility validation
        """
        return ModelTag.from_spec(model_id, spec)
