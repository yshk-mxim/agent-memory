#!/usr/bin/env python3
"""Test prefill speed with a large context prompt.

Sends a ~10,000 token prompt directly to the semantic server
to measure actual inference performance without CLI overhead.
"""

import json
import time

import requests

# Generate a ~10,000 token prompt
# Rough estimate: 1 token ≈ 4 characters for English text
# So 10,000 tokens ≈ 40,000 characters
base_text = """In the realm of artificial intelligence and machine learning, the development of large language models has revolutionized natural language processing. These models, trained on vast corpora of text data, demonstrate remarkable capabilities in understanding context, generating coherent responses, and performing complex reasoning tasks. The architecture underlying these systems typically involves transformer networks with attention mechanisms that allow the model to weigh the importance of different parts of the input sequence when generating outputs. """

# Repeat to get approximately 10,000 tokens (each repetition adds ~100 tokens)
prompt = (base_text * 150)  # ~15,000 tokens to be safe

print("=" * 60)
print("Prefill Speed Test")
print("=" * 60)
print(f"Prompt length (chars): {len(prompt):,}")
print(f"Estimated tokens: ~{len(prompt) // 4:,}")
print()

# Prepare request
url = "http://localhost:8000/v1/messages"
payload = {
    "model": "gpt-oss-20b",
    "max_tokens": 1,  # Just need 1 token to measure prefill time
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
}

print("Sending request to server...")
print(f"URL: {url}")
print(f"Max tokens: 1 (testing prefill only)")
print()

# Time the request
start_time = time.time()

try:
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=900,  # 15 minute timeout
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if response.status_code == 200:
        result = response.json()

        # Extract token counts
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        print(f"✓ Request successful")
        print(f"Status: {response.status_code}")
        print()
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print()
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print()

        if input_tokens > 0:
            tokens_per_sec = input_tokens / elapsed
            print(f"Prefill speed: {tokens_per_sec:.2f} tokens/sec")
            print()

            # Compare with expected performance
            print("Expected Performance:")
            print("  M3 Max (64GB):  ~421 tokens/sec")
            print("  M3 Ultra:       ~1000 tokens/sec")
            print("  MLX typical:    ~230 tokens/sec")
            print("  llama.cpp:      ~32 tokens/sec")
            print()

            if tokens_per_sec < 50:
                print("⚠️  WARNING: Prefill speed is EXTREMELY SLOW!")
                print("   Expected: 200-400 tokens/sec for MLX")
                print(f"   Actual:   {tokens_per_sec:.2f} tokens/sec")
                print()
                print("Possible issues:")
                print("  1. prefill_step_size too small (should be 8192)")
                print("  2. Memory pressure / swapping")
                print("  3. CPU throttling")
                print("  4. Model loading issue")
            elif tokens_per_sec < 150:
                print("⚠️  Performance is below expected for MLX")
                print(f"   Expected: 200-400 tokens/sec")
                print(f"   Actual:   {tokens_per_sec:.2f} tokens/sec")
            else:
                print("✓ Performance is within expected range for MLX")

        # Show response content
        content = result.get("content", [])
        if content:
            text = content[0].get("text", "")
            print(f"Response: {text[:100]}...")
    else:
        print(f"✗ Request failed")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except requests.exceptions.Timeout:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✗ Request timed out after {elapsed:.2f} seconds")

except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✗ Error: {e}")
    print(f"Time elapsed before error: {elapsed:.2f} seconds")

print("=" * 60)
