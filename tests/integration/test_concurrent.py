"""Concurrent tests for thread-safety.

These tests verify that BlockPool and batch engine are thread-safe.
They use real threading to catch race conditions.

Requires: No MLX dependency (uses mocks)
"""

import threading
import time

import pytest

from semantic.domain.entities import KVBlock
from semantic.domain.errors import PoolExhaustedError
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec


@pytest.fixture
def spec():
    """Create a simple ModelCacheSpec for testing."""
    return ModelCacheSpec(
        n_layers=4,
        n_kv_heads=8,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 4,
        sliding_window_size=None,
    )


@pytest.fixture
def pool(spec):
    """Create a BlockPool with 1000 blocks for concurrent testing."""
    return BlockPool(spec=spec, total_blocks=1000)


@pytest.mark.integration
class TestConcurrentBlockPool:
    """Test BlockPool thread-safety under concurrent access."""

    def test_concurrent_allocation_10_threads(self, pool: BlockPool) -> None:
        """10 threads allocating blocks simultaneously should not crash."""
        results: list[tuple[str, list[KVBlock] | Exception]] = []
        barrier = threading.Barrier(10)  # Synchronize thread start

        def allocate_worker(thread_id: int) -> None:
            try:
                # Wait for all threads to be ready
                barrier.wait()

                # Allocate blocks
                blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id=f"thread_{thread_id}")
                results.append(("success", blocks))
            except Exception as e:
                results.append(("error", e))

        # Create and start threads
        threads = [threading.Thread(target=allocate_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (or properly error if pool exhausted)
        assert len(results) == 10, "All threads should complete"

        # Count successes and errors
        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]

        # Should all succeed (10 threads x 10 blocks = 100 blocks < 1000 available)
        assert len(successes) == 10, f"All should succeed, got {len(errors)} errors"
        assert len(errors) == 0, f"No errors expected, got {errors}"

        # Verify all block IDs are unique (no double-allocation)
        all_block_ids = set()
        for _status, blocks in successes:
            for block in blocks:
                assert block.block_id not in all_block_ids, (
                    f"Block {block.block_id} allocated twice!"
                )
                all_block_ids.add(block.block_id)

        assert len(all_block_ids) == 100, "Should have 100 unique blocks"

    def test_concurrent_free_no_double_free(self, pool: BlockPool) -> None:
        """Concurrent free operations should not cause double-free errors."""
        # Allocate blocks for 5 agents
        agent_blocks = {}
        for i in range(5):
            agent_id = f"agent_{i}"
            blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id=agent_id)
            agent_blocks[agent_id] = blocks

        results: list[tuple[str, str | Exception]] = []
        barrier = threading.Barrier(5)

        def free_worker(agent_id: str, blocks: list[KVBlock]) -> None:
            try:
                barrier.wait()
                pool.free(blocks, agent_id)
                results.append(("success", agent_id))
            except Exception as e:
                results.append(("error", e))

        # Free all agents concurrently
        threads = [
            threading.Thread(target=free_worker, args=(aid, blocks))
            for aid, blocks in agent_blocks.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(results) == 5
        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 5, f"All frees should succeed, got {results}"

        # Pool should be back to full capacity
        assert pool.available_blocks() == 1000

    def test_concurrent_allocate_and_free_mixed(self, pool: BlockPool) -> None:
        """Mixed allocate/free operations should remain consistent."""
        results: list[tuple[str, str]] = []
        barrier = threading.Barrier(20)

        def allocate_worker(thread_id: int) -> None:
            try:
                barrier.wait()
                time.sleep(0.001 * thread_id)  # Stagger slightly
                _blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id=f"alloc_{thread_id}")
                results.append(("alloc_success", f"alloc_{thread_id}"))
            except Exception as e:
                results.append(("alloc_error", str(e)))

        def free_worker(thread_id: int) -> None:
            try:
                barrier.wait()
                # First allocate, then free
                agent_id = f"free_{thread_id}"
                blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id=agent_id)
                time.sleep(0.001)  # Let other threads run
                pool.free(blocks, agent_id)
                results.append(("free_success", agent_id))
            except Exception as e:
                results.append(("free_error", str(e)))

        # 10 allocate threads + 10 free threads = 20 threads
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=allocate_worker, args=(i,)))
            threads.append(threading.Thread(target=free_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check results
        alloc_successes = [r for r in results if r[0] == "alloc_success"]
        free_successes = [r for r in results if r[0] == "free_success"]

        # All allocations should succeed (10 x 5 = 50 blocks)
        assert len(alloc_successes) == 10, "All allocations should succeed"
        # All frees should succeed (allocate then free)
        assert len(free_successes) == 10, "All frees should succeed"

        # Final pool state: 950 blocks allocated (10 agents x 5 blocks)
        assert pool.available_blocks() == 950

    def test_pool_exhaustion_concurrent(self, pool: BlockPool) -> None:
        """When pool exhausts under concurrent load, errors should be clean."""
        # Pool has 1000 blocks. 200 threads each want 10 blocks = 2000 blocks needed
        results: list[tuple[str, str | Exception]] = []
        barrier = threading.Barrier(200)

        def allocate_worker(thread_id: int) -> None:
            try:
                barrier.wait()
                _blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id=f"thread_{thread_id}")
                results.append(("success", f"thread_{thread_id}"))
            except PoolExhaustedError as e:
                results.append(("exhausted", str(e)))
            except Exception as e:
                results.append(("error", str(e)))

        threads = [threading.Thread(target=allocate_worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check results
        successes = [r for r in results if r[0] == "success"]
        exhausted = [r for r in results if r[0] == "exhausted"]
        errors = [r for r in results if r[0] == "error"]

        # Should have some successes (first ~100 threads)
        assert len(successes) > 0, "Some allocations should succeed"
        # Should have some exhausted (after pool runs out)
        assert len(exhausted) > 0, "Some should get PoolExhaustedError"
        # Should have NO other errors (no crashes)
        assert len(errors) == 0, f"No other errors expected, got {errors}"

        # Exact number depends on timing, but should be around 100 successes
        # (100 threads x 10 blocks = 1000 blocks)
        assert 90 <= len(successes) <= 110, f"Expected ~100 successes, got {len(successes)}"

        # Pool should be exhausted
        assert pool.available_blocks() == 0

    def test_reconfigure_blocks_thread_safety(self, pool: BlockPool, spec: ModelCacheSpec) -> None:
        """Reconfigure should safely fail if allocations exist."""
        from semantic.domain.errors import PoolConfigurationError

        # Allocate blocks
        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="test_agent")

        # Try to reconfigure (should fail due to active allocations)
        with pytest.raises(PoolConfigurationError, match="active allocations"):
            pool.reconfigure(spec)

        # Free blocks
        pool.free(blocks, "test_agent")

        # Now reconfigure should succeed
        pool.reconfigure(spec)
        assert pool.available_blocks() == 1000


@pytest.mark.integration
class TestMemoryLeakPrevention:
    """Test memory leak prevention in step() and free()."""

    def test_layer_data_cleared_on_free(self, pool: BlockPool) -> None:
        """Freeing blocks should clear layer_data to prevent memory leaks."""
        # Allocate blocks
        blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id="test_agent")

        # Populate layer_data (simulate tensor storage)
        for block in blocks:
            block.layer_data = {"k": "fake_tensor_k", "v": "fake_tensor_v"}

        # Free blocks
        pool.free(blocks, "test_agent")

        # layer_data should be cleared (None)
        for block in blocks:
            assert block.layer_data is None, "layer_data should be None after free"

    def test_free_agent_blocks_clears_layer_data(self, pool: BlockPool) -> None:
        """free_agent_blocks should also clear layer_data."""
        # Allocate blocks for agent
        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="test_agent")

        # Populate layer_data
        for block in blocks:
            block.layer_data = {"k": "fake_tensor", "v": "fake_tensor"}

        # Free all agent blocks
        freed_count = pool.free_agent_blocks("test_agent")
        assert freed_count == 10

        # layer_data should be cleared
        for block in blocks:
            assert block.layer_data is None, "layer_data should be cleared"
