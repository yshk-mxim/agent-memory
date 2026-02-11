# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Batched vs Sequential Benchmark

Compares performance of:
1. Sequential: 5 agents × 1 request each, processed one at a time
2. Batched: 5 agents × 1 request each, processed in one batch

Measures:
- Total time
- Throughput (tokens/sec)
- Per-agent latency
"""

import asyncio
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class BatchedBenchmark:
    """Benchmark sequential vs batched multi-agent generation."""

    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.num_agents = 5
        self.tokens_per_request = 50

    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70 + "\n")

    async def run_single_request(
        self, client: httpx.AsyncClient, agent_id: int, system_prompt: str, question: str
    ) -> tuple[float, dict]:
        """Run a single request and return (latency, result)."""
        start = time.time()

        response = await client.post(
            f"{self.api_base}/v1/messages",
            json={
                "model": "gemma-3-12b-it-4bit",
                "system": system_prompt,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": self.tokens_per_request,
                "stream": False,
            },
            timeout=60.0,
        )

        latency = time.time() - start
        result = response.json()

        return latency, result

    async def benchmark_sequential(self) -> dict:
        """Benchmark sequential processing (one agent at a time)."""
        self.print_header("Sequential Processing Benchmark")

        async with httpx.AsyncClient() as client:
            print(f"→ Processing {self.num_agents} agents SEQUENTIALLY...")

            total_tokens_in = 0
            total_tokens_out = 0
            latencies = []

            overall_start = time.time()

            for i in range(self.num_agents):
                agent_id = f"seq_agent_{i}"
                system_prompt = f"You are agent {i}, a concise technical expert."
                question = f"What is feature {i} of MLX?"

                latency, result = await self.run_single_request(client, i, system_prompt, question)

                latencies.append(latency)
                total_tokens_in += result["usage"]["input_tokens"]
                total_tokens_out += result["usage"]["output_tokens"]

                print(f"  Agent {i}: {latency:.2f}s ({result['usage']['output_tokens']} tokens)")

            total_time = time.time() - overall_start

            print(f"\n✓ Sequential complete in {total_time:.2f}s")
            print(f"  Total tokens: {total_tokens_in} in, {total_tokens_out} out")
            print(f"  Throughput: {total_tokens_out / total_time:.2f} tokens/sec")
            print(f"  Avg latency: {sum(latencies) / len(latencies):.2f}s")

            return {
                "mode": "sequential",
                "total_time": total_time,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "throughput": total_tokens_out / total_time,
                "avg_latency": sum(latencies) / len(latencies),
                "latencies": latencies,
            }

    async def benchmark_batched(self) -> dict:
        """Benchmark batched processing (all agents in parallel)."""
        self.print_header("Batched Processing Benchmark")

        async with httpx.AsyncClient() as client:
            print(f"→ Processing {self.num_agents} agents in BATCHED mode...")

            tasks = []
            overall_start = time.time()

            for i in range(self.num_agents):
                agent_id = f"batch_agent_{i}"
                system_prompt = f"You are agent {i}, a concise technical expert."
                question = f"What is feature {i} of MLX?"

                task = self.run_single_request(client, i, system_prompt, question)
                tasks.append(task)

            # Execute all concurrently
            results = await asyncio.gather(*tasks)
            total_time = time.time() - overall_start

            latencies = [r[0] for r in results]
            responses = [r[1] for r in results]

            total_tokens_in = sum(r["usage"]["input_tokens"] for r in responses)
            total_tokens_out = sum(r["usage"]["output_tokens"] for r in responses)

            for i, (latency, response) in enumerate(results):
                print(f"  Agent {i}: {latency:.2f}s ({response['usage']['output_tokens']} tokens)")

            print(f"\n✓ Batched complete in {total_time:.2f}s")
            print(f"  Total tokens: {total_tokens_in} in, {total_tokens_out} out")
            print(f"  Throughput: {total_tokens_out / total_time:.2f} tokens/sec")
            print(f"  Avg latency: {sum(latencies) / len(latencies):.2f}s")

            return {
                "mode": "batched",
                "total_time": total_time,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "throughput": total_tokens_out / total_time,
                "avg_latency": sum(latencies) / len(latencies),
                "latencies": latencies,
            }

    async def run_comparison(self):
        """Run both benchmarks and compare."""
        print("\n" + "=" * 70)
        print("  BATCHED VS SEQUENTIAL BENCHMARK")
        print("  Continuous Batching Performance Comparison")
        print("=" * 70)

        # Check server
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{self.api_base}/health", timeout=2.0)
                print("✓ API server running at localhost:8000\n")
        except httpx.ConnectError:
            print("✗ API server not running. Start with: python -m src.api_server")
            return

        # Run benchmarks
        seq_results = await self.benchmark_sequential()

        print("\n⏳ Waiting 2s between benchmarks...\n")
        await asyncio.sleep(2)

        batch_results = await self.benchmark_batched()

        # Comparison
        self.print_header("Comparison")

        speedup = seq_results["total_time"] / batch_results["total_time"]
        throughput_improvement = (
            (batch_results["throughput"] - seq_results["throughput"])
            / seq_results["throughput"]
            * 100
        )

        print("Sequential:")
        print(f"  Total time: {seq_results['total_time']:.2f}s")
        print(f"  Throughput: {seq_results['throughput']:.2f} tokens/sec")
        print(f"  Avg latency: {seq_results['avg_latency']:.2f}s\n")

        print("Batched:")
        print(f"  Total time: {batch_results['total_time']:.2f}s")
        print(f"  Throughput: {batch_results['throughput']:.2f} tokens/sec")
        print(f"  Avg latency: {batch_results['avg_latency']:.2f}s\n")

        print("Improvement:")
        print(f"  Speedup: {speedup:.2f}× faster")
        print(f"  Throughput: +{throughput_improvement:.1f}%")
        print(f"  Time saved: {seq_results['total_time'] - batch_results['total_time']:.2f}s")

        if speedup >= 2.0:
            print(f"\n🎉 Batching achieved {speedup:.1f}× speedup!")
        else:
            print(f"\n✓ Batching improved performance by {speedup:.1f}×")

        print()


async def main():
    """Run the benchmark."""
    benchmark = BatchedBenchmark()
    await benchmark.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())
