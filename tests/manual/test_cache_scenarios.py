#!/usr/bin/env python3
"""End-to-end cache scenario testing with memory monitoring.

Scenarios:
1. Request ~10000 tokens → generate 100 tokens (cold start)
2. Request with same prefix → cache hit, continue generating
3. Request new ~5000 tokens → cache miss, generate
4. Repeat same 5000 tokens → cache hit, generate
5. Return to scenario 2's context → stale hit, generate more
"""

import json
import time
import requests
from typing import Any

BASE_URL = "http://localhost:8000"

def make_request(
    prompt: str,
    max_tokens: int = 50,
    stream: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Make a request to the Anthropic Messages API."""
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-ID"] = session_id

    payload = {
        "model": "deepseek-coder-v2",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }

    start = time.time()

    if stream:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            headers=headers,
            json=payload,
            stream=True,
        )

        text = ""
        tokens = 0
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if "text" in delta:
                                text += delta["text"]
                                tokens += 1
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - start
        return {
            "text": text,
            "tokens": tokens,
            "elapsed": elapsed,
            "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0,
        }
    else:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            headers=headers,
            json=payload,
        )
        elapsed = time.time() - start

        data = response.json()
        text = ""
        if "content" in data and data["content"]:
            text = data["content"][0].get("text", "")

        usage = data.get("usage", {})
        output_tokens = usage.get("output_tokens", len(text.split()))

        return {
            "text": text,
            "tokens": output_tokens,
            "elapsed": elapsed,
            "tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0,
            "input_tokens": usage.get("input_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
            "cache_creation": usage.get("cache_creation_input_tokens", 0),
        }


def get_server_stats() -> dict[str, Any]:
    """Get server health and stats."""
    try:
        response = requests.get(f"{BASE_URL}/health/detailed", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def generate_prompt(word_count: int, unique_prefix: str = "") -> str:
    """Generate a prompt with approximately word_count words."""
    # Each word is roughly 1.3 tokens on average
    base_text = "The quick brown fox jumps over the lazy dog. "
    words_per_repeat = 9
    repeats = word_count // words_per_repeat

    prompt = unique_prefix + (base_text * repeats)
    return prompt[:word_count * 6]  # Approximate character count


def run_scenario(
    name: str,
    prompt: str,
    max_tokens: int,
    session_id: str | None = None,
    expected_hit: bool = False,
) -> dict[str, Any]:
    """Run a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"Prompt length: {len(prompt)} chars (~{len(prompt.split())} words)")
    print(f"Max tokens: {max_tokens}")
    print(f"Session ID: {session_id or 'None'}")
    print(f"Expected cache: {'HIT' if expected_hit else 'MISS'}")
    print("-" * 60)

    # Get stats before
    stats_before = get_server_stats()

    # Make request
    result = make_request(prompt, max_tokens=max_tokens, session_id=session_id)

    # Get stats after
    stats_after = get_server_stats()

    # Print results
    print(f"Elapsed: {result['elapsed']:.2f}s")
    print(f"Output tokens: {result['tokens']}")
    print(f"Tokens/sec: {result['tokens_per_sec']:.1f}")

    if "cache_read" in result:
        print(f"Cache read tokens: {result['cache_read']}")
        print(f"Cache creation tokens: {result['cache_creation']}")
        actual_hit = result['cache_read'] > 0
        print(f"Actual cache: {'HIT' if actual_hit else 'MISS'}")
        if actual_hit != expected_hit:
            print(f"⚠️  UNEXPECTED: Expected {'HIT' if expected_hit else 'MISS'}")

    print(f"\nGenerated text (first 200 chars):")
    print(f"  {result['text'][:200]}...")

    # Memory stats
    if "memory" in stats_after:
        mem = stats_after["memory"]
        print(f"\nMemory after:")
        print(f"  Active: {mem.get('active_gb', 'N/A')} GB")
        print(f"  Peak: {mem.get('peak_gb', 'N/A')} GB")
        print(f"  Cache: {mem.get('cache_gb', 'N/A')} GB")

    return result


def main():
    """Run all test scenarios."""
    print("=" * 60)
    print("SEMANTIC CACHE END-TO-END TEST")
    print("=" * 60)

    # Check server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server status: {health.json()}")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        print("Start the server with: semantic serve")
        return

    # Generate prompts
    # ~10000 tokens ≈ 7500 words
    prompt_10k = generate_prompt(7500, "SESSION_A: ")
    # ~5000 tokens ≈ 3750 words
    prompt_5k = generate_prompt(3750, "SESSION_B: ")

    print(f"\nGenerated prompts:")
    print(f"  10K prompt: {len(prompt_10k)} chars")
    print(f"  5K prompt: {len(prompt_5k)} chars")

    results = {}

    # Scenario 1: Cold start with 10K tokens
    results["s1"] = run_scenario(
        name="1. Cold start - 10K tokens → 100 tokens",
        prompt=prompt_10k,
        max_tokens=100,
        session_id="session_10k",
        expected_hit=False,
    )

    time.sleep(2)  # Let cache settle

    # Scenario 2: Cache hit - same prefix
    results["s2"] = run_scenario(
        name="2. Cache HIT - same 10K prefix → 50 more tokens",
        prompt=prompt_10k,
        max_tokens=50,
        session_id="session_10k",
        expected_hit=True,
    )

    time.sleep(2)

    # Scenario 3: Cache miss - new 5K tokens
    results["s3"] = run_scenario(
        name="3. Cache MISS - new 5K tokens → 50 tokens",
        prompt=prompt_5k,
        max_tokens=50,
        session_id="session_5k",
        expected_hit=False,
    )

    time.sleep(2)

    # Scenario 4: Cache hit - same 5K prefix
    results["s4"] = run_scenario(
        name="4. Cache HIT - same 5K prefix → 50 more tokens",
        prompt=prompt_5k,
        max_tokens=50,
        session_id="session_5k",
        expected_hit=True,
    )

    time.sleep(2)

    # Scenario 5: Stale hit - back to 10K context
    results["s5"] = run_scenario(
        name="5. Stale HIT - back to 10K context → 50 more tokens",
        prompt=prompt_10k,
        max_tokens=50,
        session_id="session_10k",
        expected_hit=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<40} {:>10} {:>10} {:>10}".format(
        "Scenario", "Time (s)", "Tokens", "Tok/s"
    ))
    print("-" * 70)

    for key, result in results.items():
        name = {
            "s1": "1. Cold 10K→100",
            "s2": "2. Hit 10K→50",
            "s3": "3. Miss 5K→50",
            "s4": "4. Hit 5K→50",
            "s5": "5. Stale 10K→50",
        }[key]
        print("{:<40} {:>10.2f} {:>10} {:>10.1f}".format(
            name,
            result["elapsed"],
            result["tokens"],
            result["tokens_per_sec"],
        ))

    # Get final memory stats
    final_stats = get_server_stats()
    print("\nFinal server stats:")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
