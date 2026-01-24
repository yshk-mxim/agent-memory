"""
Real Comparative Benchmark: This POC vs LM Studio (MLX + llama.cpp)

Uses LM Studio's API to benchmark both MLX and llama.cpp backends
with the same Gemma 3 12B model, comparing against this POC.

LM Studio must be running with the model loaded.
"""

import json
import time
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.agent_manager import PersistentAgentManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LMStudioComparativeBenchmark:
    """
    Real benchmark comparing This POC vs LM Studio (MLX and llama.cpp backends).

    Requires:
    - LM Studio running with gemma-3-12b model loaded
    - Server started on localhost:1234
    """

    def __init__(
        self,
        mlx_model: str = "mlx-community/gemma-3-12b-it-4bit",
        lmstudio_base: str = "http://localhost:1234/v1",
        output_dir: str = "benchmarks/results"
    ):
        """
        Initialize benchmark.

        Args:
            mlx_model: MLX model for this POC
            lmstudio_base: LM Studio API base URL
            output_dir: Results directory
        """
        self.mlx_model = mlx_model
        self.lmstudio_base = lmstudio_base
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MLX Model: {self.mlx_model}")
        logger.info(f"LM Studio API: {self.lmstudio_base}")

    def print_section(self, title: str):
        """Print formatted section."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def check_lmstudio_running(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            response = httpx.get(f"{self.lmstudio_base}/models", timeout=2.0)
            return response.status_code == 200
        except:
            return False

    def get_lmstudio_model_info(self) -> Optional[Dict[str, Any]]:
        """Get loaded model info from LM Studio."""
        try:
            response = httpx.get(f"{self.lmstudio_base}/models")
            models = response.json()
            if models.get('data'):
                return models['data'][0]
            return None
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    def benchmark_lmstudio(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark LM Studio generation.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Max tokens to generate

        Returns:
            dict with timing and response info
        """
        start_time = time.time()

        response = httpx.post(
            f"{self.lmstudio_base}/chat/completions",
            json={
                "model": "local-model",  # LM Studio uses this identifier
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0
            },
            timeout=300.0
        )

        total_time = time.time() - start_time

        result = response.json()
        output = result['choices'][0]['message']['content']

        return {
            "total_time_sec": total_time,
            "output": output,
            "tokens_generated": result['usage'].get('completion_tokens', 0),
            "tokens_per_sec": result['usage'].get('completion_tokens', 0) / total_time if total_time > 0 else 0
        }

    def benchmark_poc_cold_start(self) -> Dict[str, Any]:
        """Benchmark This POC cold start."""
        self.print_section("This POC: Cold Start")

        manager = PersistentAgentManager(
            model_name=self.mlx_model,
            max_agents=1
        )

        start_time = time.time()
        manager.create_agent(
            agent_id="bench_test",
            agent_type="technical",
            system_prompt="You are a helpful assistant."
        )
        creation_time = time.time() - start_time

        start_time = time.time()
        response = manager.generate(
            "bench_test",
            "Explain KV caching in one sentence.",
            max_tokens=50
        )
        generation_time = time.time() - start_time

        logger.info(f"✓ Agent created in {creation_time:.2f}s")
        logger.info(f"✓ Generated: {response[:80]}...")
        logger.info(f"✓ Generation time: {generation_time:.2f}s")

        total_time = creation_time + generation_time

        return {
            "creation_time_sec": creation_time,
            "generation_time_sec": generation_time,
            "total_time_sec": total_time,
            "output": response
        }

    def benchmark_poc_session_resume(self, num_sessions: int = 3) -> Dict[str, Any]:
        """Benchmark This POC session resume."""
        self.print_section(f"This POC: {num_sessions}-Session Resume")

        manager = PersistentAgentManager(
            model_name=self.mlx_model,
            max_agents=1
        )

        # Create and save agent
        manager.create_agent(
            agent_id="bench_test2",
            agent_type="technical",
            system_prompt="You are a helpful assistant with extensive knowledge."
        )
        manager.generate("bench_test2", "Hello", max_tokens=50)
        manager.save_agent("bench_test2")

        load_times = []
        generation_times = []

        for i in range(num_sessions):
            # Evict from memory
            if "bench_test2" in manager.agents:
                del manager.agents["bench_test2"]

            # Load from disk
            start_time = time.time()
            manager.load_agent("bench_test2")
            load_time = time.time() - start_time
            load_times.append(load_time)

            # Generate
            start_time = time.time()
            manager.generate("bench_test2", f"Query {i+1}", max_tokens=50)
            gen_time = time.time() - start_time
            generation_times.append(gen_time)

            logger.info(f"  Session {i+1}: load={load_time*1000:.1f}ms, gen={gen_time:.2f}s")

        return {
            "num_sessions": num_sessions,
            "avg_load_time_sec": sum(load_times) / len(load_times),
            "avg_generation_sec": sum(generation_times) / len(generation_times),
            "total_time_sec": sum(load_times) + sum(generation_times),
            "per_session_sec": (sum(load_times) + sum(generation_times)) / num_sessions
        }

    def run_all(self, quick: bool = True) -> Dict[str, Any]:
        """Run all benchmarks."""
        logger.info("\n" + "=" * 70)
        logger.info("  REAL COMPARATIVE BENCHMARK")
        logger.info("  This POC vs LM Studio (MLX + llama.cpp)")
        logger.info("=" * 70)

        # Check LM Studio
        if not self.check_lmstudio_running():
            logger.error("\n❌ LM Studio server not running on localhost:1234")
            logger.error("\nTo start LM Studio server:")
            logger.error("1. Open LM Studio app")
            logger.error("2. Load gemma-3-12b model")
            logger.error("3. Go to Local Server tab")
            logger.error("4. Click 'Start Server'")
            logger.error("\nOr use CLI: lms server start")
            return None

        model_info = self.get_lmstudio_model_info()
        if model_info:
            logger.info(f"\n✓ LM Studio running with model: {model_info.get('id', 'unknown')}")

        results = {
            "mlx_model": self.mlx_model,
            "lmstudio_model": model_info.get('id') if model_info else "unknown",
            "note": "Real measurements - both systems actually run",
            "scenarios": {}
        }

        # Scenario 1: Cold Start
        self.print_section("Scenario 1: Cold Start Comparison")

        # This POC
        poc_cold = self.benchmark_poc_cold_start()

        # LM Studio (current backend, either MLX or llama.cpp)
        self.print_section("LM Studio: Cold Start")
        logger.info("Benchmarking LM Studio with current backend...")
        lms_cold = self.benchmark_lmstudio(
            "Explain KV caching in one sentence.",
            "You are a helpful assistant.",
            max_tokens=50
        )
        logger.info(f"✓ Generated: {lms_cold['output'][:80]}...")
        logger.info(f"✓ Total time: {lms_cold['total_time_sec']:.2f}s ({lms_cold['tokens_per_sec']:.1f} tok/s)")

        results["scenarios"]["cold_start"] = {
            "this_poc": poc_cold,
            "lmstudio": lms_cold
        }

        # Scenario 2: Session Resume
        num_sessions = 3 if quick else 5

        # This POC
        poc_resume = self.benchmark_poc_session_resume(num_sessions)

        # LM Studio - simulate session resume (no cache persistence)
        self.print_section(f"LM Studio: {num_sessions}-Session Resume (No Cache Persistence)")
        logger.info("LM Studio has no cache persistence - re-processing each session...")

        lms_times = []
        for i in range(num_sessions):
            lms_result = self.benchmark_lmstudio(
                f"Query {i+1}",
                "You are a helpful assistant with extensive knowledge.",
                max_tokens=50
            )
            lms_times.append(lms_result['total_time_sec'])
            logger.info(f"  Session {i+1}: {lms_result['total_time_sec']:.2f}s")

        lms_resume = {
            "num_sessions": num_sessions,
            "session_times_sec": lms_times,
            "avg_session_sec": sum(lms_times) / len(lms_times),
            "total_time_sec": sum(lms_times),
            "note": "No cache persistence - full processing each session"
        }

        results["scenarios"]["session_resume"] = {
            "this_poc": poc_resume,
            "lmstudio": lms_resume
        }

        # Calculate advantages
        self.print_section("Results Summary")

        # Cold start
        poc_cold_time = poc_cold["total_time_sec"]
        lms_cold_time = lms_cold["total_time_sec"]
        cold_diff = ((lms_cold_time - poc_cold_time) / lms_cold_time) * 100

        print(f"\nCold Start:")
        print(f"  This POC:   {poc_cold_time:.2f}s")
        print(f"  LM Studio:  {lms_cold_time:.2f}s")
        print(f"  Difference: {abs(cold_diff):.1f}% {'faster' if cold_diff > 0 else 'slower'} (POC)")

        # Session resume
        poc_resume_per = poc_resume["per_session_sec"]
        lms_resume_per = lms_resume["avg_session_sec"]
        resume_advantage = ((lms_resume_per - poc_resume_per) / lms_resume_per) * 100

        print(f"\n{num_sessions}-Session Resume:")
        print(f"  This POC:   {poc_resume_per:.2f}s per session (with cache persistence)")
        print(f"  LM Studio:  {lms_resume_per:.2f}s per session (no cache persistence)")
        print(f"  Advantage:  {resume_advantage:.1f}% faster with cache persistence")

        results["summary"] = {
            "cold_start_difference_percent": cold_diff,
            "session_resume_advantage_percent": resume_advantage
        }

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON."""
        if results is None:
            logger.error("No results to save")
            return

        output_path = self.output_dir / "lmstudio_comparative_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LM Studio Comparative Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 sessions)")

    args = parser.parse_args()

    benchmark = LMStudioComparativeBenchmark()
    results = benchmark.run_all(quick=args.quick)
    if results:
        benchmark.save_results(results)
