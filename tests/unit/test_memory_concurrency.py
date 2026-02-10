"""Concurrency tests for BlockPool memory tracking.

Tests that used_memory() and available_memory() are thread-safe and return
consistent values under concurrent allocation/deallocation.
"""

import threading

import pytest

from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec


@pytest.fixture
def spec() -> ModelCacheSpec:
    """Create model cache spec for testing."""
    return ModelCacheSpec(
        n_layers=12,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 12,
        sliding_window_size=None,
    )


@pytest.fixture
def pool(spec: ModelCacheSpec) -> BlockPool:
    """Create block pool with 100 blocks."""
    return BlockPool(spec=spec, total_blocks=100)


class TestMemoryConcurrency:
    """Tests for thread-safe memory tracking."""

    def test_used_memory_consistent_under_concurrent_allocation(self, pool: BlockPool) -> None:
        """used_memory() should return consistent values during concurrent allocations.

        Validates thatMemory tracking remains accurate even when multiple threads
        allocate blocks simultaneously.
        """
        results: list[int] = []
        errors: list[Exception] = []

        def allocate_and_measure():
            """Allocate blocks and measure memory usage."""
            try:
                # Allocate 5 blocks
                blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id="test_agent")

                # Read memory usage (should be thread-safe)
                used = pool.used_memory()
                results.append(used)

                # Free blocks
                pool.free(blocks, agent_id="test_agent")

            except Exception as e:
                errors.append(e)

        # Run 10 threads concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=allocate_and_measure)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all memory readings were non-negative
        assert all(m >= 0 for m in results), "Memory readings should be non-negative"

        # Verify memory is consistent after all operations
        final_used = pool.used_memory()
        final_available = pool.available_memory()
        total = pool.total_memory()

        assert final_used + final_available == total, (
            f"Memory invariant violated: {final_used} + {final_available} != {total}"
        )

    def test_available_memory_consistent_under_concurrent_operations(self, pool: BlockPool) -> None:
        """available_memory() should return consistent values during concurrent ops.

        Validates thatAvailable memory tracking is thread-safe.
        """
        results: list[int] = []
        errors: list[Exception] = []

        def allocate_read_free():
            """Allocate, read available memory, then free."""
            try:
                # Allocate
                blocks = pool.allocate(n_blocks=3, layer_id=0, agent_id="test_agent")

                # Read available memory (should be thread-safe)
                available = pool.available_memory()
                results.append(available)

                # Free
                pool.free(blocks, agent_id="test_agent")

            except Exception as e:
                errors.append(e)

        # Run 20 threads concurrently
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=allocate_read_free)
            threads.append(thread)

        # Start all
        for thread in threads:
            thread.start()

        # Wait
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all readings were valid (non-negative, <= total)
        total = pool.total_memory()
        assert all(0 <= m <= total for m in results), (
            f"Invalid memory readings: {[m for m in results if m < 0 or m > total]}"
        )

        # Verify final state is consistent
        final_available = pool.available_memory()
        final_used = pool.used_memory()

        assert final_available + final_used == total, (
            f"Final state inconsistent: {final_available} + {final_used} != {total}"
        )

    def test_memory_invariant_maintained_under_load(self, pool: BlockPool) -> None:
        """Memory invariant (used + available = total) maintained under concurrent load.

        Validates thatNo race conditions in memory tracking.
        """
        total = pool.total_memory()
        snapshots: list[tuple[int, int]] = []
        errors: list[Exception] = []

        def stress_test():
            """Rapidly allocate and free while taking memory snapshots."""
            try:
                for _ in range(10):
                    # Allocate
                    blocks = pool.allocate(n_blocks=2, layer_id=0, agent_id="test_agent")

                    # Take snapshot (should be consistent under concurrency)
                    used = pool.used_memory()
                    available = pool.available_memory()
                    snapshots.append((used, available))

                    # Free
                    pool.free(blocks, agent_id="test_agent")

            except Exception as e:
                errors.append(e)

        # Run 5 threads concurrently
        threads = [threading.Thread(target=stress_test) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify EVERY snapshot maintained the invariant
        for i, (used, available) in enumerate(snapshots):
            assert used + available == total, (
                f"Snapshot {i} violated invariant: {used} + {available} != {total}"
            )

    def test_concurrent_memory_reads_no_deadlock(self, pool: BlockPool) -> None:
        """Multiple concurrent reads should not deadlock.

        Validates thatLock implementation doesn't cause deadlock.
        """
        results: list[int] = []

        def read_memory():
            """Repeatedly read memory metrics."""
            for _ in range(100):
                used = pool.used_memory()
                available = pool.available_memory()
                results.append(used + available)

        # Run 10 reader threads
        threads = [threading.Thread(target=read_memory) for _ in range(10)]

        for thread in threads:
            thread.start()

        # Join with timeout (if deadlock, this will timeout)
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "Deadlock detected - thread didn't complete"

        # Verify we got expected number of results
        assert len(results) == 1000, f"Expected 1000 results, got {len(results)}"

        # All results should equal total
        total = pool.total_memory()
        assert all(r == total for r in results), "Memory invariant violated in reads"
