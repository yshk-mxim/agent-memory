# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Benchmark lock overhead for memory tracking methods (NEW-2).

Validates that lock acquisition for used_memory() and available_memory()
has <1ms overhead as required.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec


def benchmark_memory_methods():
    """Benchmark used_memory() and available_memory() lock overhead."""
    print("=== NEW-2 Memory Lock Overhead Benchmark ===\n")

    # Create spec and pool
    spec = ModelCacheSpec(
        n_layers=12,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 12,
        sliding_window_size=None,
    )

    pool = BlockPool(spec=spec, total_blocks=100)

    # Allocate some blocks to make it realistic
    pool.allocate(n_blocks=50, layer_id=0, agent_id="bench_agent")

    print("Pool state:")
    print(f"  Total blocks: {pool.total_blocks}")
    print(f"  Allocated: {pool.allocated_block_count()}")
    print(f"  Available: {pool.available_blocks()}")
    print()

    # Benchmark used_memory()
    print("Benchmarking used_memory() (with lock)...")
    iterations = 100000

    start = time.perf_counter()
    for _ in range(iterations):
        _ = pool.used_memory()
    end = time.perf_counter()

    used_memory_total = (end - start) * 1000  # Convert to ms
    used_memory_per_call = used_memory_total / iterations

    print(f"  Total time: {used_memory_total:.3f} ms")
    print(f"  Per-call:   {used_memory_per_call * 1000:.3f} µs")
    print(f"  Throughput: {iterations / (end - start):.0f} calls/sec")

    # Benchmark available_memory()
    print("\nBenchmarking available_memory() (with lock)...")

    start = time.perf_counter()
    for _ in range(iterations):
        _ = pool.available_memory()
    end = time.perf_counter()

    available_memory_total = (end - start) * 1000  # Convert to ms
    available_memory_per_call = available_memory_total / iterations

    print(f"  Total time: {available_memory_total:.3f} ms")
    print(f"  Per-call:   {available_memory_per_call * 1000:.3f} µs")
    print(f"  Throughput: {iterations / (end - start):.0f} calls/sec")

    # Benchmark total_memory() (baseline - no lock needed, spec is immutable)
    print("\nBenchmarking total_memory() (baseline, no lock)...")

    start = time.perf_counter()
    for _ in range(iterations):
        _ = pool.total_memory()
    end = time.perf_counter()

    total_memory_total = (end - start) * 1000  # Convert to ms
    total_memory_per_call = total_memory_total / iterations

    print(f"  Total time: {total_memory_total:.3f} ms")
    print(f"  Per-call:   {total_memory_per_call * 1000:.3f} µs")
    print(f"  Throughput: {iterations / (end - start):.0f} calls/sec")

    # Verify requirement: <1ms per call
    print("\n" + "=" * 50)
    print("NEW-2 REQUIREMENT: Lock overhead <1ms per call")
    print("=" * 50)

    used_memory_pass = used_memory_per_call < 1.0
    available_memory_pass = available_memory_per_call < 1.0

    print(f"\nused_memory():      {used_memory_per_call * 1000:.3f} µs  ", end="")
    print("✅ PASS" if used_memory_pass else "❌ FAIL (<1ms required)")

    print(f"available_memory(): {available_memory_per_call * 1000:.3f} µs  ", end="")
    print("✅ PASS" if available_memory_pass else "❌ FAIL (<1ms required)")

    # Calculate lock overhead vs baseline
    used_overhead = ((used_memory_per_call - total_memory_per_call) / total_memory_per_call) * 100
    available_overhead = ((available_memory_per_call - total_memory_per_call) / total_memory_per_call) * 100

    print(f"\nLock overhead:")
    print(f"  used_memory():      +{used_overhead:.1f}% vs baseline")
    print(f"  available_memory(): +{available_overhead:.1f}% vs baseline")

    # Overall result
    print("\n" + "=" * 50)
    if used_memory_pass and available_memory_pass:
        print("🎉 NEW-2 BENCHMARK: PASS")
        print("=" * 50)
        print("\nLock overhead is well within <1ms requirement.")
        print("Thread-safe memory tracking has minimal performance impact.")
        return True
    else:
        print("❌ NEW-2 BENCHMARK: FAIL")
        print("=" * 50)
        print("\nLock overhead exceeds 1ms requirement!")
        return False


if __name__ == "__main__":
    success = benchmark_memory_methods()
    sys.exit(0 if success else 1)
