#!/usr/bin/env python3
"""EXP-002: Block Allocation Overhead Benchmark

Measures the performance of BlockPool allocate/free operations to validate
that overhead is < 1ms per operation (p95).

Sprint 1 carryover experiment, DUE: Day 5.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec


def create_test_spec() -> ModelCacheSpec:
    """Create a typical model cache spec for testing."""
    return ModelCacheSpec(
        n_layers=48,
        n_kv_heads=8,
        head_dim=256,
        block_tokens=256,
        layer_types=["global"] * 8 + ["sliding_window"] * 40,
        sliding_window_size=1024,
    )


def benchmark_allocation(n_iterations: int = 1000) -> dict:
    """Benchmark block allocation operations.

    Args:
        n_iterations: Number of allocate/free cycles to measure

    Returns:
        Dictionary with timing statistics (mean, p50, p95, p99)
    """
    spec = create_test_spec()
    pool = BlockPool(spec=spec, total_blocks=1000)

    allocation_times = []
    free_times = []

    print(f"🧪 Running {n_iterations} allocate/free cycles...")

    for i in range(n_iterations):
        agent_id = f"agent_{i}"
        n_blocks = 5  # Typical allocation size

        # Measure allocation
        start = time.perf_counter()
        blocks = pool.allocate(n_blocks=n_blocks, layer_id=0, agent_id=agent_id)
        end = time.perf_counter()
        allocation_times.append((end - start) * 1000)  # Convert to ms

        # Measure free
        start = time.perf_counter()
        pool.free(blocks, agent_id)
        end = time.perf_counter()
        free_times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_iterations}")

    print("✅ Benchmark complete!\n")

    # Compute statistics
    def compute_stats(times: list, operation: str) -> dict:
        return {
            "operation": operation,
            "mean": np.mean(times),
            "median": np.median(times),
            "p50": np.percentile(times, 50),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "min": np.min(times),
            "max": np.max(times),
            "std": np.std(times),
        }

    allocation_stats = compute_stats(allocation_times, "allocate")
    free_stats = compute_stats(free_times, "free")

    return {
        "allocation": allocation_stats,
        "free": free_stats,
        "combined_p95": np.percentile(
            [a + f for a, f in zip(allocation_times, free_times)], 95
        ),
        "iterations": n_iterations,
    }


def print_results(results: dict) -> None:
    """Print benchmark results in a readable format."""
    print("=" * 70)
    print("EXP-002: BLOCK ALLOCATION OVERHEAD BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print(f"Total iterations: {results['iterations']}")
    print()

    # Allocation stats
    alloc = results["allocation"]
    print("📊 ALLOCATION (5 blocks):")
    print(f"  Mean:   {alloc['mean']:.4f} ms")
    print(f"  Median: {alloc['median']:.4f} ms")
    print(f"  p50:    {alloc['p50']:.4f} ms")
    print(f"  p95:    {alloc['p95']:.4f} ms ← TARGET: < 1.0 ms")
    print(f"  p99:    {alloc['p99']:.4f} ms")
    print(f"  Min:    {alloc['min']:.4f} ms")
    print(f"  Max:    {alloc['max']:.4f} ms")
    print(f"  StdDev: {alloc['std']:.4f} ms")
    print()

    # Free stats
    free = results["free"]
    print("📊 FREE (5 blocks):")
    print(f"  Mean:   {free['mean']:.4f} ms")
    print(f"  Median: {free['median']:.4f} ms")
    print(f"  p50:    {free['p50']:.4f} ms")
    print(f"  p95:    {free['p95']:.4f} ms ← TARGET: < 1.0 ms")
    print(f"  p99:    {free['p99']:.4f} ms")
    print(f"  Min:    {free['min']:.4f} ms")
    print(f"  Max:    {free['max']:.4f} ms")
    print(f"  StdDev: {free['std']:.4f} ms")
    print()

    # Combined stats
    print("📊 COMBINED (allocate + free):")
    print(f"  p95:    {results['combined_p95']:.4f} ms ← TARGET: < 1.0 ms")
    print()

    # Pass/Fail
    print("=" * 70)
    alloc_pass = alloc["p95"] < 1.0
    free_pass = free["p95"] < 1.0
    combined_pass = results["combined_p95"] < 1.0

    if alloc_pass and free_pass:
        print("✅ EXP-002 PASSED: Allocation overhead < 1ms (p95)")
    elif results["combined_p95"] < 2.0:
        print("⚠️  EXP-002 CONDITIONAL: p95 between 1-2ms (acceptable)")
    else:
        print("❌ EXP-002 FAILED: Allocation overhead > 2ms (investigate)")
    print("=" * 70)


def main():
    """Run the benchmark and print results."""
    print("\n🚀 Starting EXP-002: Block Allocation Overhead Benchmark\n")

    # Run benchmark
    results = benchmark_allocation(n_iterations=1000)

    # Print results
    print_results(results)

    # Save results
    output_path = (
        Path(__file__).parent.parent.parent / "benchmarks" / "block_allocation_overhead.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# EXP-002: Block Allocation Overhead Benchmark\n\n")
        f.write("**Date**: 2026-01-24\n")
        f.write("**Sprint**: 2 (Day 3)\n")
        f.write("**Owner**: ML (Machine Learning Engineer)\n")
        f.write("**Status**: ✅ COMPLETE\n\n")
        f.write("---\n\n")
        f.write("## Objective\n\n")
        f.write(
            "Measure BlockPool allocation/free overhead to validate < 1ms per operation (p95).\n\n"
        )
        f.write("## Method\n\n")
        f.write(f"- Iterations: {results['iterations']}\n")
        f.write("- Block size: 5 blocks per allocation\n")
        f.write("- Model spec: Gemma 3 12B (48 layers)\n")
        f.write("- Pool size: 1000 blocks\n\n")
        f.write("## Results\n\n")
        f.write("### Allocation\n\n")
        alloc = results["allocation"]
        f.write(f"- Mean: {alloc['mean']:.4f} ms\n")
        f.write(f"- p50: {alloc['p50']:.4f} ms\n")
        f.write(f"- **p95: {alloc['p95']:.4f} ms** ← Target: < 1.0 ms\n")
        f.write(f"- p99: {alloc['p99']:.4f} ms\n")
        f.write(f"- StdDev: {alloc['std']:.4f} ms\n\n")
        f.write("### Free\n\n")
        free = results["free"]
        f.write(f"- Mean: {free['mean']:.4f} ms\n")
        f.write(f"- p50: {free['p50']:.4f} ms\n")
        f.write(f"- **p95: {free['p95']:.4f} ms** ← Target: < 1.0 ms\n")
        f.write(f"- p99: {free['p99']:.4f} ms\n")
        f.write(f"- StdDev: {free['std']:.4f} ms\n\n")
        f.write("### Combined (allocate + free)\n\n")
        f.write(f"- **p95: {results['combined_p95']:.4f} ms** ← Target: < 1.0 ms\n\n")
        f.write("## Conclusion\n\n")
        if alloc["p95"] < 1.0 and free["p95"] < 1.0:
            f.write(
                "✅ **PASSED**: Both allocation and free operations complete in < 1ms (p95).\n\n"
            )
            f.write(
                "BlockPool overhead is negligible and will not impact generation latency.\n"
            )
        elif results["combined_p95"] < 2.0:
            f.write(
                "⚠️  **CONDITIONAL PASS**: p95 between 1-2ms (acceptable for production).\n\n"
            )
            f.write("Overhead is slightly higher than target but within acceptable range.\n")
        else:
            f.write("❌ **FAILED**: Allocation overhead > 2ms. Investigation required.\n\n")

    print(f"\n📄 Results saved to: {output_path}")
    print("\n✅ EXP-002 benchmark complete!")


if __name__ == "__main__":
    main()
