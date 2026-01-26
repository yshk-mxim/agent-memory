#!/usr/bin/env python3
"""Test with exactly ~17,616 tokens to match user's scenario."""

import json
import time

import requests

# Generate a ~17,616 token prompt (need ~70,464 characters)
base_text = """In the realm of artificial intelligence and machine learning, the development of large language models has revolutionized natural language processing. These models, trained on vast corpora of text data, demonstrate remarkable capabilities in understanding context, generating coherent responses, and performing complex reasoning tasks. The architecture underlying these systems typically involves transformer networks with attention mechanisms that allow the model to weigh the importance of different parts of the input sequence when generating outputs. """

# Repeat to get approximately 17,616 tokens
prompt = (base_text * 260)  # ~26,000 tokens to ensure we exceed 17K

print("=" * 60)
print("17K Token Prefill Test")
print("=" * 60)
print(f"Prompt length (chars): {len(prompt):,}")
print(f"Estimated tokens: ~{len(prompt) // 4:,}")
print()

# Prepare request
url = "http://localhost:8000/v1/messages"
payload = {
    "model": "gpt-oss-20b",
    "max_tokens": 1,
    "messages": [{"role": "user", "content": prompt}]
}

print("Sending request to server...")
print()

start_time = time.time()

try:
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=900,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if response.status_code == 200:
        result = response.json()
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        print(f"✓ Request successful")
        print()
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print()
        print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        print()

        if input_tokens > 0:
            tokens_per_sec = input_tokens / elapsed
            print(f"Prefill speed: {tokens_per_sec:.2f} tokens/sec")
            print()

            if elapsed > 120:
                print(f"⚠️  WARNING: This took {elapsed:.0f} seconds!")
                print(f"   At 566 tokens/sec, {input_tokens} tokens should take ~{input_tokens/566:.0f} seconds")
                print(f"   Something is wrong!")
            else:
                print(f"✓ Performance is good!")

        content = result.get("content", [])
        if content:
            text = content[0].get("text", "")
            print(f"Response: {text[:100]}")
    else:
        print(f"✗ Request failed")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✗ Error: {e}")
    print(f"Time: {elapsed:.2f} seconds")

print("=" * 60)
