"""
Step 1: Benchmark LM Studio Only
No MLX imports - purely HTTP API calls to LM Studio
"""

import json
import time
import httpx
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_lmstudio_running(base_url: str) -> bool:
    """Check if LM Studio server is running."""
    try:
        response = httpx.get(f"{base_url}/models", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def get_lmstudio_model_info(base_url: str):
    """Get loaded model info from LM Studio."""
    try:
        response = httpx.get(f"{base_url}/models")
        models = response.json()
        if models.get('data'):
            return models['data'][0]
        return None
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def benchmark_lmstudio(
    base_url: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 50
):
    """Benchmark LM Studio generation via API."""
    start_time = time.time()

    response = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": "local-model",
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


def main(quick: bool = True):
    """Run LM Studio benchmarks only."""
    base_url = "http://localhost:1234/v1"
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("STEP 1: LM Studio Benchmarks Only")
    print("No MLX model loaded - no memory competition")

    # Check LM Studio
    if not check_lmstudio_running(base_url):
        logger.error("\n❌ LM Studio server not running on localhost:1234")
        logger.error("\nTo start LM Studio server:")
        logger.error("1. Open LM Studio app")
        logger.error("2. Load gemma-3-12b model")
        logger.error("3. Go to Local Server tab")
        logger.error("4. Click 'Start Server'")
        return None

    model_info = get_lmstudio_model_info(base_url)
    if model_info:
        logger.info(f"\n✓ LM Studio running with model: {model_info.get('id', 'unknown')}")

    # Warmup
    print_section("Warmup: LM Studio")
    logger.info("Running warmup request to compile/optimize...")
    warmup_result = benchmark_lmstudio(
        base_url,
        "warmup",
        "You are a helpful assistant.",
        max_tokens=10
    )
    logger.info(f"✓ Warmup complete ({warmup_result['total_time_sec']:.2f}s)")

    results = {
        "lmstudio_model": model_info.get('id') if model_info else "unknown",
        "note": "LM Studio only - no memory competition with MLX",
        "scenarios": {}
    }

    # Scenario 1: Cold Start
    print_section("LM Studio: Cold Start")
    logger.info("Benchmarking LM Studio...")
    lms_cold = benchmark_lmstudio(
        base_url,
        "Explain KV caching in one sentence.",
        "You are a helpful assistant.",
        max_tokens=50
    )
    logger.info(f"✓ Generated: {lms_cold['output'][:80]}...")
    logger.info(f"✓ Total time: {lms_cold['total_time_sec']:.2f}s ({lms_cold['tokens_per_sec']:.1f} tok/s)")

    results["scenarios"]["cold_start"] = lms_cold

    # Scenario 2: Session Resume
    num_sessions = 3 if quick else 5
    print_section(f"LM Studio: {num_sessions}-Session Resume")
    logger.info("Testing session resume (no cache persistence)...")

    lms_times = []
    for i in range(num_sessions):
        lms_result = benchmark_lmstudio(
            base_url,
            f"Query {i+1}",
            "You are a helpful assistant with extensive knowledge.",
            max_tokens=50
        )
        lms_times.append(lms_result['total_time_sec'])
        logger.info(f"  Session {i+1}: {lms_result['total_time_sec']:.2f}s")

    results["scenarios"]["session_resume"] = {
        "num_sessions": num_sessions,
        "session_times_sec": lms_times,
        "avg_session_sec": sum(lms_times) / len(lms_times),
        "total_time_sec": sum(lms_times),
        "note": "No cache persistence - full processing each session"
    }

    # Save results
    output_path = output_dir / "lmstudio_only_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_section("Step 1 Complete!")
    logger.info(f"✓ LM Studio results saved to: {output_path}")
    logger.info("\n📋 Next step: SHUT DOWN LM STUDIO")
    logger.info("   Then run: python -m benchmarks.step2_poc_only --quick")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: LM Studio Only")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 sessions)")
    args = parser.parse_args()

    main(quick=args.quick)
