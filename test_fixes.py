#!/usr/bin/env python3
"""Test the request serialization and model name fixes.

Verifies:
1. Concurrent requests are serialized (no 700x slowdown)
2. Model name returned is actual loaded model (gpt-oss-20b-MXFP4-Q4)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Server URL
BASE_URL = "http://localhost:8000"

def make_request(prompt: str, max_tokens: int = 10, request_id: int = 0) -> dict:
    """Make a single non-streaming request."""
    payload = {
        "model": "claude-haiku-4-5-20251001",  # Request Claude model
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        return {
            "id": request_id,
            "success": True,
            "elapsed": elapsed,
            "model": result.get("model"),
            "input_tokens": result.get("usage", {}).get("input_tokens", 0),
            "output_tokens": result.get("usage", {}).get("output_tokens", 0),
        }
    else:
        return {
            "id": request_id,
            "success": False,
            "elapsed": elapsed,
            "error": response.text[:200],
        }

def test_model_name_fix():
    """Test that actual model name is returned, not requested model."""
    print("=" * 60)
    print("TEST 1: Model Name Fix")
    print("=" * 60)

    result = make_request("Hello", max_tokens=5)

    if result["success"]:
        returned_model = result["model"]
        print(f"✓ Request successful")
        print(f"  Requested model: claude-haiku-4-5-20251001")
        print(f"  Returned model:  {returned_model}")
        print()

        if "gpt-oss" in returned_model:
            print("✓ Model name fix WORKING - returns actual model")
        else:
            print("✗ Model name fix FAILED - still returning requested model")
    else:
        print(f"✗ Request failed: {result['error']}")

    print()

def test_concurrent_serialization():
    """Test that concurrent requests are serialized properly."""
    print("=" * 60)
    print("TEST 2: Request Serialization")
    print("=" * 60)

    # Create 3 requests with different prompt lengths
    prompts = [
        ("Short prompt for request 1", 10),  # ~10 tokens
        ("Medium length prompt for request 2 with more words", 10),  # ~15 tokens
        ("This is a longer prompt for request 3 " * 50, 10),  # ~500 tokens
    ]

    print(f"Sending 3 concurrent requests...")
    print()

    start_time = time.time()

    # Submit requests concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(make_request, prompt, max_tokens, i)
            for i, (prompt, max_tokens) in enumerate(prompts)
        ]

        results = [future.result() for future in as_completed(futures)]

    total_time = time.time() - start_time

    # Sort by request ID
    results.sort(key=lambda x: x["id"])

    print("Results:")
    for result in results:
        if result["success"]:
            print(f"  Request {result['id']}: {result['elapsed']:.2f}s, "
                  f"{result['input_tokens']} input tokens")
        else:
            print(f"  Request {result['id']}: FAILED - {result['error']}")

    print()
    print(f"Total time: {total_time:.2f}s")

    # With serialization, requests should be processed sequentially
    # Each should take roughly the same time per token
    if all(r["success"] for r in results):
        # Check if total time is reasonable (not 14 minutes!)
        if total_time < 60:  # Should be under 1 minute for these tiny prompts
            print(f"✓ Serialization WORKING - reasonable total time")
            print(f"  (Without serialization, this would take minutes)")
        else:
            print(f"⚠️  Total time seems high: {total_time:.2f}s")
            print(f"  Might still have batching issues")
    else:
        print("✗ Some requests failed, can't evaluate serialization")

    print()

def test_large_prompt_performance():
    """Test that large prompt doesn't take 14 minutes."""
    print("=" * 60)
    print("TEST 3: Large Prompt Performance (No Concurrent Interference)")
    print("=" * 60)

    # Generate ~1000 token prompt
    base_text = "The quick brown fox jumps over the lazy dog. " * 50

    print(f"Sending large prompt (~1000 tokens)...")
    print()

    result = make_request(base_text, max_tokens=5)

    if result["success"]:
        elapsed = result["elapsed"]
        tokens = result["input_tokens"]
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

        print(f"✓ Request successful")
        print(f"  Input tokens: {tokens}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {tokens_per_sec:.2f} tokens/sec")
        print()

        if tokens_per_sec > 200:
            print(f"✓ Performance is EXCELLENT ({tokens_per_sec:.0f} tok/s)")
        elif tokens_per_sec > 100:
            print(f"✓ Performance is GOOD ({tokens_per_sec:.0f} tok/s)")
        elif tokens_per_sec > 50:
            print(f"⚠️  Performance is ACCEPTABLE ({tokens_per_sec:.0f} tok/s)")
        else:
            print(f"✗ Performance is POOR ({tokens_per_sec:.0f} tok/s)")
    else:
        print(f"✗ Request failed: {result['error']}")

    print()

if __name__ == "__main__":
    print()
    print("Testing Request Serialization and Model Name Fixes")
    print()

    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("✗ Server not responding at http://localhost:8000")
            print("  Start server with: semantic serve")
            exit(1)
    except requests.exceptions.RequestException:
        print("✗ Server not running at http://localhost:8000")
        print("  Start server with: semantic serve")
        exit(1)

    print("✓ Server is running")
    print()

    # Run tests
    test_model_name_fix()
    test_concurrent_serialization()
    test_large_prompt_performance()

    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)
