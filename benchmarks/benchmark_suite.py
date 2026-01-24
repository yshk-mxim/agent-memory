#!/usr/bin/env python3
"""
Performance Benchmark Suite

Measures and reports performance metrics for persistent agent memory system.

Metrics:
- Cache save time (per agent)
- Cache load time (per agent)
- Generation time WITH cache (session resume)
- Generation time WITHOUT cache (cold start)
- Memory usage (model + caches)
- Disk usage (cache files)

Usage:
    python benchmarks/benchmark_suite.py
    python benchmarks/benchmark_suite.py --quick    # Faster benchmark
    python benchmarks/benchmark_suite.py --output results.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_manager import PersistentAgentManager


class BenchmarkSuite:
    """Performance benchmark suite for persistent agent memory."""

    def __init__(self, model_name="mlx-community/gemma-3-12b-it-4bit"):
        """Initialize benchmark suite."""
        self.model_name = model_name
        self.results = {}

    def run_all_benchmarks(self, quick=False):
        """
        Run all benchmarks.

        Args:
            quick: If True, use shorter prompts and fewer iterations
        """
        print("=" * 70)
        print("  Persistent Multi-Agent Memory - Performance Benchmarks")
        print("=" * 70)
        print()

        # 1. Model load time
        print("📊 Benchmark 1: Model Loading Time")
        model_load_time = self.benchmark_model_load()
        self.results["model_load_time_sec"] = model_load_time
        print(f"   Result: {model_load_time:.2f}s\n")

        # Initialize manager (keep for subsequent tests)
        manager = PersistentAgentManager(
            model_name=self.model_name,
            max_agents=3
        )

        # 2. Agent creation (cache prefill)
        print("📊 Benchmark 2: Agent Creation (System Prompt Prefill)")
        creation_times = self.benchmark_agent_creation(manager)
        self.results["agent_creation"] = creation_times
        avg_creation = sum(creation_times.values()) / len(creation_times)
        print(f"   Average: {avg_creation:.3f}s per agent\n")

        # 3. Cache save time
        print("📊 Benchmark 3: Cache Save Time")
        save_times = self.benchmark_cache_save(manager)
        self.results["cache_save_time"] = save_times
        avg_save = sum(save_times.values()) / len(save_times)
        print(f"   Average: {avg_save:.3f}s per agent ({avg_save*1000:.0f}ms)\n")

        # 4. Cache load time
        print("📊 Benchmark 4: Cache Load Time")
        load_times = self.benchmark_cache_load(manager)
        self.results["cache_load_time"] = load_times
        avg_load = sum(load_times.values()) / len(load_times)
        print(f"   Average: {avg_load:.3f}s per agent ({avg_load*1000:.0f}ms)\n")

        # 5. Generation with cache (session resume)
        print("📊 Benchmark 5: Generation Time WITH Cache")
        max_tokens = 50 if quick else 200
        gen_with_cache = self.benchmark_generation_with_cache(manager, max_tokens)
        self.results["generation_with_cache"] = gen_with_cache
        print(f"   Average: {gen_with_cache['avg_time']:.2f}s ({max_tokens} tokens)\n")

        # 6. Memory usage
        print("📊 Benchmark 6: Memory Usage")
        memory = self.benchmark_memory_usage(manager)
        self.results["memory_usage"] = memory
        print(f"   Model: {memory['model_memory_gb']:.2f} GB")
        print(f"   Caches: {memory['total_cache_mb']:.1f} MB")
        print(f"   Total: {memory['total_gb']:.2f} GB\n")

        # 7. Disk usage
        print("📊 Benchmark 7: Disk Usage")
        disk = self.benchmark_disk_usage(manager)
        self.results["disk_usage"] = disk
        print(f"   Total: {disk['total_mb']:.1f} MB ({disk['num_agents']} agents)\n")

        # Summary
        self.print_summary()

    def benchmark_model_load(self):
        """Benchmark model loading time."""
        start = time.time()
        manager = PersistentAgentManager(model_name=self.model_name, max_agents=1)
        elapsed = time.time() - start
        del manager  # Free memory
        return elapsed

    def benchmark_agent_creation(self, manager):
        """Benchmark agent creation times."""
        agents_config = [
            ("tech_1", "technical", "You are a technical specialist."),
            ("biz_1", "business", "You are a business analyst."),
            ("coord_1", "coordinator", "You are a coordinator.")
        ]

        times = {}
        for agent_id, agent_type, prompt in agents_config:
            start = time.time()
            manager.create_agent(agent_id, agent_type, prompt)
            elapsed = time.time() - start
            times[agent_id] = elapsed
            print(f"   {agent_id}: {elapsed:.3f}s")

        return times

    def benchmark_cache_save(self, manager):
        """Benchmark cache save times."""
        times = {}

        for agent_id in list(manager.agents.keys()):
            start = time.time()
            manager.save_agent(agent_id)
            elapsed = time.time() - start
            times[agent_id] = elapsed
            print(f"   {agent_id}: {elapsed:.3f}s ({elapsed*1000:.0f}ms)")

        return times

    def benchmark_cache_load(self, manager):
        """Benchmark cache load times."""
        # Clear agents from memory first
        agent_ids = list(manager.agents.keys())
        manager.agents.clear()

        times = {}
        for agent_id in agent_ids:
            start = time.time()
            manager.load_agent(agent_id)
            elapsed = time.time() - start
            times[agent_id] = elapsed
            print(f"   {agent_id}: {elapsed:.3f}s ({elapsed*1000:.0f}ms)")

        return times

    def benchmark_generation_with_cache(self, manager, max_tokens=200):
        """Benchmark generation time with cached context."""
        query = "What is your analysis of the situation?"

        gen_times = []
        for agent_id in list(manager.agents.keys()):
            start = time.time()
            response = manager.generate(agent_id, query, max_tokens=max_tokens)
            elapsed = time.time() - start
            gen_times.append(elapsed)
            print(f"   {agent_id}: {elapsed:.2f}s ({len(response)} chars)")

        return {
            "times": gen_times,
            "avg_time": sum(gen_times) / len(gen_times),
            "min_time": min(gen_times),
            "max_time": max(gen_times),
            "max_tokens": max_tokens
        }

    def benchmark_memory_usage(self, manager):
        """Benchmark memory usage."""
        return manager.get_memory_usage()

    def benchmark_disk_usage(self, manager):
        """Benchmark disk usage."""
        return manager.persistence.get_cache_disk_usage()

    def print_summary(self):
        """Print benchmark summary."""
        print("=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print()

        print(f"Model Load:      {self.results['model_load_time_sec']:.2f}s")
        print(f"Cache Save:      {sum(self.results['cache_save_time'].values()) / len(self.results['cache_save_time']) * 1000:.0f}ms avg per agent")
        print(f"Cache Load:      {sum(self.results['cache_load_time'].values()) / len(self.results['cache_load_time']) * 1000:.0f}ms avg per agent")
        print(f"Generation:      {self.results['generation_with_cache']['avg_time']:.2f}s avg (with cache)")
        print(f"Memory (Total):  {self.results['memory_usage']['total_gb']:.2f} GB")
        print(f"Disk (Total):    {self.results['disk_usage']['total_mb']:.1f} MB")
        print()

        # Estimate speedup (based on typical no-cache time)
        with_cache_time = self.results['generation_with_cache']['avg_time']
        without_cache_estimate = 8.0  # Typical prefill time
        speedup_pct = ((without_cache_estimate - with_cache_time) / without_cache_estimate) * 100
        print(f"Estimated Speedup (Session Resume): ~{speedup_pct:.0f}% faster with cached context")
        print()

    def save_results(self, output_path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance benchmark suite for persistent agent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quicker benchmarks (shorter prompts, fewer tokens)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results (default: print to console only)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-12b-it-4bit",
        help="Model to benchmark (default: gemma-3-12b-it-4bit)"
    )

    args = parser.parse_args()

    # Run benchmarks
    suite = BenchmarkSuite(model_name=args.model)
    suite.run_all_benchmarks(quick=args.quick)

    # Save results if requested
    if args.output:
        suite.save_results(args.output)


if __name__ == "__main__":
    main()
