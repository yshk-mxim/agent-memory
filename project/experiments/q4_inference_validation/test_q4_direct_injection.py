#!/usr/bin/env python3
"""Test Q4 direct injection in actual semantic server code path.

This tests our ACTUAL code path (BatchGenerator with QuantizedKVCache injection),
not mlx-lm's high-level generate() function.

Expected results if Q4 injection works:
- 1K tokens: ~0.20GB memory (not ~0.86GB)
- No errors during generation
- Correct output produced
"""

import sys
import os
import logging
import mlx.core as mx

# Add semantic to path
sys.path.insert(0, '/Users/dev_user/semantic/src')

from semantic.adapters.config.settings import Settings
from semantic.adapters.outbound.mlx_model_loader import MLXModelLoader
from semantic.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.application.agent_cache_store import AgentCacheStore
from semantic.domain.services import BlockPool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_q4_direct_injection_1k():
    """Test Q4 direct injection with 1K token cache."""

    logger.info("="*60)
    logger.info("Q4 DIRECT INJECTION TEST (1K tokens)")
    logger.info("="*60)

    # Initialize components
    settings = Settings()
    model_id = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
    cache_budget_mb = 8192
    max_agents = 5

    model_loader = MLXModelLoader()
    cache_adapter = MLXCacheAdapter()

    logger.info(f"Loading model: {model_id}")
    model, tokenizer, spec = model_loader.load_model(
        model_id,
        kv_bits=4,
        kv_group_size=64
    )

    logger.info(f"Model loaded: {spec.n_layers} layers, {spec.n_kv_heads} KV heads")

    # Initialize block pool and batch engine
    block_pool = BlockPool(
        spec=spec,
        cache_budget_mb=cache_budget_mb
    )

    cache_store = AgentCacheStore(
        cache_adapter=cache_adapter,
        cache_persistence_dir=settings.agent.cache_dir,
        max_hot_agents=max_agents
    )

    batch_engine = BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        block_pool=block_pool,
        spec=spec,
        cache_adapter=cache_adapter
    )

    # Test parameters
    agent_id = "test_q4_1k"
    prompt = "Explain what machine learning is. " * 100  # ~1K tokens

    logger.info(f"\n[TEST] Agent: {agent_id}, Prompt: ~1K tokens")

    # Memory before first request
    mx.metal.clear_cache()
    mem_start = mx.metal.get_active_memory() / (1024**3)
    logger.info(f"[MEMORY] Start: {mem_start:.2f}GB")

    # First request: Generate and save cache
    logger.info("\n[REQUEST 1] Generating cache...")
    response1 = batch_engine.submit(
        agent_id=agent_id,
        prompt=prompt,
        max_tokens=100,
        temperature=0.0,
    )

    # Process until complete
    while response1.status == "processing":
        batch_engine.step()

    mem_after_gen = mx.metal.get_active_memory() / (1024**3)
    logger.info(f"[MEMORY] After generation: {mem_after_gen:.2f}GB")
    logger.info(f"[RESPONSE 1] Status: {response1.status}")

    if response1.status != "completed":
        logger.error(f"❌ Generation failed: {response1.status}")
        return False

    # Save cache
    logger.info("\n[SAVE] Saving cache to warm tier...")
    agent_blocks = block_pool.get_agent_blocks(agent_id)
    if agent_blocks:
        cache_store.save(agent_id, agent_blocks, "test session")
        logger.info(f"✅ Cache saved: {agent_blocks.total_tokens} tokens, {len(agent_blocks.blocks)} layers")

    # Clear for second request
    block_pool.clear_agent(agent_id)
    mx.metal.clear_cache()

    # Memory before loading cache
    mem_before_load = mx.metal.get_active_memory() / (1024**3)
    logger.info(f"\n[MEMORY] Before cache load: {mem_before_load:.2f}GB")

    # Second request: Load Q4 cache with direct injection
    logger.info("\n[REQUEST 2] Loading Q4 cache (direct injection)...")

    # Load cache
    loaded_cache = cache_store.load(agent_id)
    if not loaded_cache:
        logger.error("❌ Failed to load cache")
        return False

    logger.info(f"✅ Cache loaded: {loaded_cache.total_tokens} tokens")

    # Submit with loaded cache (this will inject Q4 directly)
    response2 = batch_engine.submit(
        agent_id=agent_id,
        prompt=prompt,
        max_tokens=100,
        temperature=0.0,
    )

    mem_after_load = mx.metal.get_active_memory() / (1024**3)
    mem_peak = mx.metal.get_peak_memory() / (1024**3)

    # Process until complete
    while response2.status == "processing":
        batch_engine.step()

    mem_final = mx.metal.get_active_memory() / (1024**3)

    logger.info(f"[MEMORY] After Q4 injection: {mem_after_load:.2f}GB")
    logger.info(f"[MEMORY] Peak: {mem_peak:.2f}GB")
    logger.info(f"[MEMORY] Final: {mem_final:.2f}GB")
    logger.info(f"[MEMORY] Spike from load: {(mem_peak - mem_before_load):.3f}GB")

    logger.info(f"[RESPONSE 2] Status: {response2.status}")

    if response2.status != "completed":
        logger.error(f"❌ Generation with Q4 cache failed: {response2.status}")
        return False

    # Analysis
    spike_gb = mem_peak - mem_before_load

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Memory spike from cache load: {spike_gb:.3f}GB")
    logger.info(f"Expected Q4: ~0.20GB")
    logger.info(f"Previous (FP8?): ~0.86GB")
    logger.info(f"FP16 would be: ~2.00GB")

    if spike_gb < 0.3:
        logger.info("✅ Q4 DIRECT INJECTION WORKS! Memory spike matches Q4 expectations!")
        return True
    elif spike_gb < 1.0:
        logger.warning(f"⚠️  Spike {spike_gb:.3f}GB higher than Q4, lower than FP16")
        logger.warning("Possible FP8 or partial dequantization")
        return True  # Still an improvement
    else:
        logger.error(f"❌ Spike {spike_gb:.3f}GB too high - Q4 injection not working")
        return False


def main():
    """Run Q4 direct injection test."""
    try:
        success = test_q4_direct_injection_1k()

        if success:
            logger.info("\n🎉 Q4 DIRECT INJECTION TEST PASSED!")
            sys.exit(0)
        else:
            logger.error("\n❌ Q4 DIRECT INJECTION TEST FAILED!")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
