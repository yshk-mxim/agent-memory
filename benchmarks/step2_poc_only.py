"""
Step 2: Benchmark This POC Only
LM Studio should be shut down - no memory competition
"""

import json
import time
from pathlib import Path
import logging

from src.agent_manager import PersistentAgentManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def benchmark_poc_cold_start(mlx_model: str):
    """Benchmark This POC cold start."""
    print_section("This POC: Cold Start")

    manager = PersistentAgentManager(
        model_name=mlx_model,
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


def benchmark_poc_session_resume(mlx_model: str, num_sessions: int = 3):
    """Benchmark This POC session resume."""
    print_section(f"This POC: {num_sessions}-Session Resume")

    manager = PersistentAgentManager(
        model_name=mlx_model,
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


def main(quick: bool = True):
    """Run POC benchmarks only."""
    mlx_model = "mlx-community/gemma-3-12b-it-4bit"
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("STEP 2: This POC Benchmarks Only")
    print("LM Studio should be shut down - no memory competition")

    results = {
        "mlx_model": mlx_model,
        "note": "POC only - no memory competition with LM Studio",
        "scenarios": {}
    }

    # Scenario 1: Cold Start
    poc_cold = benchmark_poc_cold_start(mlx_model)
    results["scenarios"]["cold_start"] = poc_cold

    # Scenario 2: Session Resume
    num_sessions = 3 if quick else 5
    poc_resume = benchmark_poc_session_resume(mlx_model, num_sessions)
    results["scenarios"]["session_resume"] = poc_resume

    # Save results
    output_path = output_dir / "poc_only_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_section("Step 2 Complete!")
    logger.info(f"✓ POC results saved to: {output_path}")
    logger.info("\n📋 Next step: Combine results")
    logger.info("   Run: python -m benchmarks.step3_combine")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 2: POC Only")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 sessions)")
    args = parser.parse_args()

    main(quick=args.quick)
