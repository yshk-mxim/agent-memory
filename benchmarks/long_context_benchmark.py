"""
Long Context Multi-Turn Benchmark: Real Engine Simulation

Simulates realistic multi-agent engine behavior:
- Start with 500-token system prompt
- Multi-turn conversation incrementing by ~100 tokens per turn
- Build up to 4096 tokens total
- Session resume with full context
"""

import json
import time
import httpx
from pathlib import Path
import logging

from src.agent_manager import PersistentAgentManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# Long system prompt (~500 tokens)
LONG_SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge in:
- Machine learning and deep learning architectures
- Natural language processing and transformer models
- Distributed systems and edge computing
- Performance optimization and benchmarking
- Software engineering best practices
- Multi-agent systems and coordination protocols
- Knowledge representation and reasoning
- Ethical AI and responsible deployment

Your role is to provide detailed, technical analysis of complex problems. You should:
1. Break down problems into clear components
2. Provide concrete examples and code when relevant
3. Consider trade-offs and alternative approaches
4. Reference specific papers, techniques, or implementations
5. Explain both theoretical foundations and practical implications
6. Maintain technical accuracy while being accessible
7. Acknowledge uncertainties and limitations
8. Suggest follow-up questions for deeper exploration

When analyzing systems or architectures, consider:
- Performance characteristics (latency, throughput, memory)
- Scalability and resource efficiency
- Maintainability and code quality
- Security and privacy implications
- User experience and developer ergonomics
- Cost and operational considerations
- Edge cases and failure modes
- Integration with existing tools and workflows

You communicate with clarity and precision, avoiding jargon when simpler terms suffice, but using technical terminology when it adds precision."""

# Questions that will incrementally build context (~50-100 tokens each)
CONVERSATION_QUESTIONS = [
    "What are the key bottlenecks in transformer models on edge devices?",
    "How does KV cache quantization affect model quality?",
    "What about cache persistence across sessions?",
    "For multi-agent systems with large contexts, how much time is wasted?",
    "What architecture would enable persistent KV cache?",
    "How would cache invalidation work?",
    "What are the storage overhead implications?",
    "How does this compare to prompt caching in API services?",
    "What optimizations can reduce cache load time?",
    "How would batch processing affect cache persistence?",
    "What security considerations exist for cached data?",
    "How would multi-tenancy work with persistent caches?",
    "What happens when the model is updated?",
    "How would compression affect cache size?",
    "What metadata should be stored with each cache?",
    "How would distributed caching work across machines?",
    "What cleanup strategies prevent cache bloat?",
    "How does cache warmth affect inference latency?",
    "What monitoring is needed for cache health?",
    "How would rollback work if a cache is corrupted?"
]


def benchmark_lmstudio_multiturn(base_url: str, target_tokens: int = 4096):
    """Benchmark LM Studio with multi-turn conversation to target tokens."""
    print_section(f"LM Studio: Multi-Turn to {target_tokens} tokens")

    messages = [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
    current_tokens = estimate_tokens(LONG_SYSTEM_PROMPT)
    logger.info(f"Starting with system prompt: ~{current_tokens} tokens")

    turn_times = []
    turn_count = 0

    # Execute turns until we reach target
    for question in CONVERSATION_QUESTIONS:
        if current_tokens >= target_tokens:
            break

        turn_count += 1
        messages.append({"role": "user", "content": question})
        current_tokens += estimate_tokens(question)

        # Generate response
        start_time = time.time()
        response = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "max_tokens": 150,  # ~100 token responses
                "temperature": 0.7
            },
            timeout=300.0
        )
        turn_time = time.time() - start_time
        turn_times.append(turn_time)

        result = response.json()
        assistant_msg = result['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": assistant_msg})
        current_tokens += estimate_tokens(assistant_msg)

        logger.info(f"  Turn {turn_count}: {turn_time:.2f}s (context: ~{current_tokens} tokens)")

    final_tokens = result['usage'].get('prompt_tokens', current_tokens)
    total_time = sum(turn_times)

    logger.info(f"\n✓ Completed {turn_count} turns")
    logger.info(f"✓ Final context: {final_tokens} tokens")
    logger.info(f"✓ Total time: {total_time:.2f}s")
    logger.info(f"✓ Avg per turn: {total_time / turn_count:.2f}s")

    return {
        "num_turns": turn_count,
        "final_context_tokens": final_tokens,
        "turn_times_sec": turn_times,
        "total_time_sec": total_time,
        "avg_turn_sec": total_time / turn_count,
        "note": "Multi-turn with incremental context growth",
        "messages": messages  # Return messages for resume test
    }


def benchmark_lmstudio_resume(base_url: str, previous_messages: list):
    """Benchmark LM Studio 'resuming' conversation (re-processing all context)."""
    print_section(f"LM Studio: Resume Session with Full Context")

    # LM Studio doesn't persist cache, so we need to re-send all previous messages
    # This simulates what happens when resuming a conversation
    messages = previous_messages.copy()
    messages.append({"role": "user", "content": "Based on our entire discussion, what's the key insight about persistent KV caching?"})

    estimated_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    logger.info(f"Re-sending {len(messages)} messages (~{estimated_tokens} tokens)")

    start_time = time.time()
    response = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": "local-model",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0
        },
        timeout=300.0
    )
    total_time = time.time() - start_time

    result = response.json()
    actual_tokens = result['usage'].get('prompt_tokens', 0)

    logger.info(f"✓ Re-processed {actual_tokens} tokens")
    logger.info(f"✓ Total time: {total_time:.2f}s")

    return {
        "context_tokens": actual_tokens,
        "total_time_sec": total_time,
        "note": "Full context re-processing (no persistent cache)"
    }


def benchmark_poc_multiturn(mlx_model: str, target_tokens: int = 4096):
    """Benchmark POC with multi-turn conversation to target tokens."""
    print_section(f"This POC: Multi-Turn to {target_tokens} tokens")

    manager = PersistentAgentManager(model_name=mlx_model, max_agents=1)

    # Create agent with system prompt
    manager.create_agent(
        agent_id="multiturn_agent",
        agent_type="technical",
        system_prompt=LONG_SYSTEM_PROMPT
    )

    current_tokens = estimate_tokens(LONG_SYSTEM_PROMPT)
    logger.info(f"Starting with system prompt: ~{current_tokens} tokens")

    turn_times = []
    turn_count = 0

    # Execute turns
    for question in CONVERSATION_QUESTIONS:
        if current_tokens >= target_tokens:
            break

        turn_count += 1
        current_tokens += estimate_tokens(question)

        # Generate response
        start_time = time.time()
        response = manager.generate(
            "multiturn_agent",
            question,
            max_tokens=150
        )
        turn_time = time.time() - start_time
        turn_times.append(turn_time)

        current_tokens += estimate_tokens(response)
        logger.info(f"  Turn {turn_count}: {turn_time:.2f}s (context: ~{current_tokens} tokens)")

    # Save the agent with full context
    manager.save_agent("multiturn_agent")

    total_time = sum(turn_times)

    logger.info(f"\n✓ Completed {turn_count} turns")
    logger.info(f"✓ Final context: ~{current_tokens} tokens")
    logger.info(f"✓ Total time: {total_time:.2f}s")
    logger.info(f"✓ Avg per turn: {total_time / turn_count:.2f}s")
    logger.info(f"✓ Agent saved with full cache")

    return {
        "num_turns": turn_count,
        "estimated_context_tokens": current_tokens,
        "turn_times_sec": turn_times,
        "total_time_sec": total_time,
        "avg_turn_sec": total_time / turn_count,
        "note": "Multi-turn with cache building"
    }


def benchmark_poc_resume(mlx_model: str):
    """Benchmark POC resuming from persistent cache."""
    print_section("This POC: Resume from Persistent Cache")

    manager = PersistentAgentManager(model_name=mlx_model, max_agents=1)

    # Load agent from cache
    start_time = time.time()
    manager.load_agent("multiturn_agent")
    load_time = time.time() - start_time

    logger.info(f"✓ Cache loaded in {load_time*1000:.1f}ms")

    # Generate with loaded context
    start_time = time.time()
    response = manager.generate(
        "multiturn_agent",
        "Based on our previous discussion, what's the key insight about persistent KV caching?",
        max_tokens=100
    )
    gen_time = time.time() - start_time

    total_time = load_time + gen_time

    logger.info(f"✓ Generated: {response[:80]}...")
    logger.info(f"✓ Generation time: {gen_time:.2f}s")
    logger.info(f"✓ Total time: {total_time:.2f}s")

    return {
        "load_time_sec": load_time,
        "generation_time_sec": gen_time,
        "total_time_sec": total_time,
        "output": response[:200]
    }


def main():
    """Run long context multi-turn benchmarks."""
    base_url = "http://localhost:1234/v1"
    mlx_model = "mlx-community/gemma-3-12b-it-4bit"
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("LONG CONTEXT MULTI-TURN BENCHMARK")
    print("Realistic engine: 500 tokens → 4096 tokens via multi-turn")

    results = {
        "scenario": "multi_turn_4096_tokens",
        "note": "Real multi-turn execution from 500 to 4096 tokens"
    }

    # Check if LM Studio is available
    try:
        response = httpx.get(f"{base_url}/models", timeout=2.0)
        lmstudio_available = response.status_code == 200
    except:
        lmstudio_available = False

    if lmstudio_available:
        logger.info("\n✓ LM Studio detected - running multi-turn test")

        # Warmup
        print_section("Warmup: LM Studio")
        httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "warmup"}],
                "max_tokens": 10
            },
            timeout=300.0
        )
        logger.info("✓ Warmup complete")

        # Multi-turn benchmark
        lms_multiturn = benchmark_lmstudio_multiturn(base_url, target_tokens=4096)

        # Save for results (without messages)
        lms_multiturn_for_save = {k: v for k, v in lms_multiturn.items() if k != "messages"}
        results["lmstudio_multiturn"] = lms_multiturn_for_save

        # Resume benchmark (using the actual messages from multi-turn)
        lms_resume = benchmark_lmstudio_resume(base_url, lms_multiturn["messages"])
        results["lmstudio_resume"] = lms_resume

        # Save intermediate results
        with open(output_dir / "long_context_lmstudio.json", 'w') as f:
            json.dump({
                "multiturn": lms_multiturn,
                "resume": lms_resume
            }, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("  LM Studio benchmark complete!")
        logger.info("  Please shut down LM Studio, then run:")
        logger.info("  python -m benchmarks.long_context_benchmark --poc-only")
        logger.info("=" * 70)

    else:
        logger.info("\n✓ No LM Studio detected - running POC tests\n")

        # Check if we have LM Studio results
        lms_results_path = output_dir / "long_context_lmstudio.json"
        if lms_results_path.exists():
            with open(lms_results_path) as f:
                lms_data = json.load(f)
                results["lmstudio_multiturn"] = lms_data["multiturn"]
                results["lmstudio_resume"] = lms_data["resume"]

        # POC multi-turn
        poc_multiturn = benchmark_poc_multiturn(mlx_model, target_tokens=4096)
        results["poc_multiturn"] = poc_multiturn

        # POC resume
        poc_resume = benchmark_poc_resume(mlx_model)
        results["poc_resume"] = poc_resume

        # Calculate comparison if we have LM Studio data
        if "lmstudio_resume" in results:
            print_section("COMPARISON RESULTS")

            lms_tokens = results["lmstudio_resume"]["context_tokens"]
            lms_reprocess = results["lmstudio_resume"]["total_time_sec"]
            poc_cache_load = poc_resume["load_time_sec"]
            poc_total = poc_resume["total_time_sec"]

            advantage = ((lms_reprocess - poc_total) / lms_reprocess) * 100
            speedup = lms_reprocess / poc_cache_load

            print(f"\nScenario: Resume {lms_tokens}-token context session")
            print(f"  LM Studio:      {lms_reprocess:.2f}s (re-process all {lms_tokens} tokens)")
            print(f"  POC Resume:     {poc_total:.2f}s ({poc_cache_load*1000:.1f}ms cache load)")
            print(f"  Advantage:      {advantage:.1f}% faster")
            print(f"  Cache speedup:  {speedup:.0f}x faster than re-processing")

            # Per-turn comparison
            lms_avg_turn = results["lmstudio_multiturn"]["avg_turn_sec"]
            poc_avg_turn = results["poc_multiturn"]["avg_turn_sec"]
            print(f"\nMulti-turn performance:")
            print(f"  LM Studio avg:  {lms_avg_turn:.2f}s per turn")
            print(f"  POC avg:        {poc_avg_turn:.2f}s per turn")

            results["summary"] = {
                "context_size_tokens": lms_tokens,
                "lmstudio_reprocess_sec": lms_reprocess,
                "poc_cache_load_sec": poc_cache_load,
                "poc_total_sec": poc_total,
                "advantage_percent": advantage,
                "cache_speedup_factor": speedup,
                "lms_avg_turn_sec": lms_avg_turn,
                "poc_avg_turn_sec": poc_avg_turn
            }

        # Save final results
        output_path = output_dir / "long_context_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Long Context Multi-Turn Benchmark")
    parser.add_argument("--poc-only", action="store_true", help="Run POC tests only")
    args = parser.parse_args()

    main()
