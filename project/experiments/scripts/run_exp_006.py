#!/usr/bin/env python3
"""EXP-006: Block Gather Performance Benchmark

Measures the performance of BlockPoolBatchEngine._reconstruct_cache() to
validate that cache reconstruction overhead is < 5ms (p95).

Sprint 2, Day 8
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


def setup_test_environment():
    """Load model and create test infrastructure.

    Returns:
        tuple: (model, tokenizer, pool, spec, engine)
    """
    # TODO: Day 8 implementation
    # from mlx_lm import load
    # from semantic.domain.services import BlockPool
    # from semantic.domain.value_objects import ModelCacheSpec
    # from semantic.application.batch_engine import BlockPoolBatchEngine

    # # Use Gemma 3 12B for realistic layer count
    # model, tokenizer = load("mlx-community/gemma-3-12b-it-4bit")
    # spec = ModelCacheSpec.from_model(model)
    # pool = BlockPool(spec, total_blocks=1000)
    # engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)

    # return model, tokenizer, pool, spec, engine
    raise NotImplementedError("Setup pending BlockPoolBatchEngine implementation")


def create_synthetic_blocks(pool, spec, context_tokens: int):
    """Create synthetic blocks for testing cache reconstruction.

    Args:
        pool: BlockPool instance
        spec: ModelCacheSpec
        context_tokens: Context size (e.g., 8192 for 8K)

    Returns:
        AgentBlocks with allocated blocks
    """
    # TODO: Day 8 implementation
    # from semantic.domain.entities import AgentBlocks

    # agent_blocks = AgentBlocks(agent_id="test_agent", total_tokens=context_tokens)

    # for layer_id in range(spec.n_layers):
    #     layer_type = spec.layer_types[layer_id]
    #
    #     if layer_type == "global":
    #         n_blocks = context_tokens // spec.block_tokens
    #     else:  # sliding_window
    #         window_tokens = spec.sliding_window_size or 1024
    #         n_blocks = window_tokens // spec.block_tokens
    #
    #     blocks = pool.allocate(n_blocks, layer_id, "test_agent")
    #     for block in blocks:
    #         agent_blocks.add_block(block)

    # return agent_blocks
    raise NotImplementedError("Synthetic blocks pending AgentBlocks implementation")


def benchmark_block_gather(engine, agent_blocks, n_runs: int = 100) -> dict:
    """Benchmark cache reconstruction from blocks.

    Args:
        engine: BlockPoolBatchEngine instance
        agent_blocks: AgentBlocks with allocated blocks
        n_runs: Number of benchmark iterations

    Returns:
        dict: Statistics (p50, p95, p99, mean, std, min, max)
    """
    # TODO: Day 8 implementation
    # times = []

    # print(f"🧪 Running {n_runs} cache reconstruction iterations...")

    # for run in range(n_runs):
    #     # Measure reconstruction time
    #     start = time.perf_counter()
    #     cache = engine._reconstruct_cache(agent_blocks)
    #     end = time.perf_counter()
    #
    #     elapsed_ms = (end - start) * 1000  # Convert to milliseconds
    #     times.append(elapsed_ms)
    #
    #     if (run + 1) % 10 == 0:
    #         print(f"  Progress: {run + 1}/{n_runs}")

    # print("✅ Benchmark complete!")
    # print()

    # # Compute statistics
    # return {
    #     "p50": np.percentile(times, 50),
    #     "p95": np.percentile(times, 95),
    #     "p99": np.percentile(times, 99),
    #     "mean": np.mean(times),
    #     "std": np.std(times),
    #     "min": min(times),
    #     "max": max(times),
    #     "samples": n_runs,
    #     "raw_times": times,
    # }
    raise NotImplementedError("Benchmark pending _reconstruct_cache() implementation")


def print_results(stats: dict, context_tokens: int) -> None:
    """Print benchmark results in a readable format.

    Args:
        stats: Statistics dictionary
        context_tokens: Context size tested
    """
    print("=" * 70)
    print("EXP-006: BLOCK GATHER PERFORMANCE BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print(f"Context size: {context_tokens} tokens ({context_tokens // 256} blocks)")
    print(f"Total iterations: {stats['samples']}")
    print()

    # Statistics table
    print("📊 CACHE RECONSTRUCTION PERFORMANCE:")
    print(f"  Mean:   {stats['mean']:.4f} ms")
    print(f"  Median: {stats['p50']:.4f} ms")
    print(f"  p95:    {stats['p95']:.4f} ms ← TARGET: < 5.0 ms")
    print(f"  p99:    {stats['p99']:.4f} ms ← STRETCH: < 10.0 ms")
    print(f"  Min:    {stats['min']:.4f} ms")
    print(f"  Max:    {stats['max']:.4f} ms")
    print(f"  StdDev: {stats['std']:.4f} ms")
    print()

    # Variance check
    variance_ratio = stats['std'] / stats['mean']
    variance_ok = variance_ratio < 0.2

    if variance_ok:
        print(f"✅ Variance check PASSED: {variance_ratio:.1%} < 20%")
    else:
        print(f"⚠️  Variance check FAILED: {variance_ratio:.1%} >= 20%")
    print()

    # Pass/Fail
    print("=" * 70)
    p95_pass = stats['p95'] < 5.0
    p99_pass = stats['p99'] < 10.0

    if p95_pass:
        print("✅ EXP-006 PASSED: p95 < 5ms (cache reconstruction is fast)")
    elif stats['p95'] < 10.0:
        print("⚠️  EXP-006 CONDITIONAL: p95 between 5-10ms (acceptable, document in ADR)")
    else:
        print("❌ EXP-006 FAILED: p95 > 10ms (investigate performance)")

    if p99_pass:
        print("✅ Tail latency acceptable: p99 < 10ms")

    print("=" * 70)


def run_exp_006():
    """Execute EXP-006 benchmark.

    Returns:
        bool: True if benchmark passes, False otherwise
    """
    print("=" * 70)
    print("EXP-006: BLOCK GATHER PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    print("🔧 Setting up test environment...")
    try:
        model, tokenizer, pool, spec, engine = setup_test_environment()
        print("✅ Test environment ready")
        print()
    except NotImplementedError as e:
        print(f"⏳ Skipping EXP-006: {e}")
        return False

    # Test configurations
    test_contexts = [
        (2048, "2K"),   # Small context
        (4096, "4K"),   # Medium context
        (8192, "8K"),   # Large context (primary test)
    ]

    all_results = {}

    for context_tokens, label in test_contexts:
        print(f"\n{'=' * 70}")
        print(f"Testing {label} context ({context_tokens} tokens)")
        print(f"{'=' * 70}")
        print()

        # Create synthetic blocks
        print(f"📦 Allocating {context_tokens // 256} blocks per layer...")
        agent_blocks = create_synthetic_blocks(pool, spec, context_tokens)
        print("✅ Blocks allocated")
        print()

        # Benchmark
        stats = benchmark_block_gather(engine, agent_blocks, n_runs=100)

        # Print results
        print_results(stats, context_tokens)

        # Store results
        all_results[label] = stats

        # Clean up
        pool.free_agent_blocks("test_agent")

    # Save results
    output_path = (
        Path(__file__).parent.parent / "data" / "exp_006_timings.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove raw_times for JSON serialization
    save_results = {}
    for label, stats in all_results.items():
        save_results[label] = {k: v for k, v in stats.items() if k != "raw_times"}

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print()
    print(f"💾 Results saved to: {output_path}")

    # Final verdict
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Check 8K context (primary test)
    stats_8k = all_results["8K"]
    p95_pass = stats_8k["p95"] < 5.0

    if p95_pass:
        print("🎉 EXP-006 PASSED: Cache reconstruction is sufficiently fast")
        print()
        print(f"8K context p95: {stats_8k['p95']:.2f}ms < 5.0ms target")
        print("One-time cache restoration overhead is negligible.")
        return True
    elif stats_8k["p95"] < 10.0:
        print("⚠️  EXP-006 CONDITIONAL PASS: p95 between 5-10ms")
        print()
        print(f"8K context p95: {stats_8k['p95']:.2f}ms")
        print()
        print("RECOMMENDATION:")
        print("- Document in ADR-004 as acceptable trade-off")
        print("- One-time cost at restore, not per-step overhead")
        print("- Per-step gather would be 10-100x more expensive")
        return True
    else:
        print("❌ EXP-006 FAILED: Cache reconstruction is too slow")
        print()
        print(f"8K context p95: {stats_8k['p95']:.2f}ms > 10.0ms")
        print()
        print("NEXT STEPS:")
        print("1. Profile mx.concatenate performance")
        print("2. Check block count scaling (O(n) vs O(n²))")
        print("3. Validate MLX lazy evaluation (mx.eval() called?)")
        print("4. Consider alternative strategies:")
        print("   - Pre-allocated buffer (copy blocks into contiguous memory)")
        print("   - Lazy gather (concatenate on first use, cache result)")
        print("   - Padded approach (avoid gather entirely)")
        return False


def main():
    """Main entry point."""
    success = run_exp_006()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
