#!/usr/bin/env python3
"""Test prefill speed with STREAMING enabled (like Claude Code CLI)."""

import json
import time

import requests

# Generate ~17K token prompt
base_text = """In the realm of artificial intelligence and machine learning, the development of large language models has revolutionized natural language processing. These models, trained on vast corpora of text data, demonstrate remarkable capabilities in understanding context, generating coherent responses, and performing complex reasoning tasks. The architecture underlying these systems typically involves transformer networks with attention mechanisms that allow the model to weigh the importance of different parts of the input sequence when generating outputs. """

prompt = (base_text * 260)  # ~21K tokens

print("=" * 60)
print("Streaming Prefill Test (17K+ tokens)")
print("=" * 60)
print(f"Prompt length (chars): {len(prompt):,}")
print(f"Estimated tokens: ~{len(prompt) // 4:,}")
print()

# Prepare request with STREAMING
url = "http://localhost:8000/v1/messages"
payload = {
    "model": "gpt-oss-20b",
    "max_tokens": 50,  # Generate a few tokens to test streaming
    "stream": True,  # ENABLE STREAMING
    "messages": [{"role": "user", "content": prompt}]
}

print("Sending STREAMING request to server...")
print()

start_time = time.time()
first_chunk_time = None
last_chunk_time = None
chunk_count = 0
input_tokens = 0
output_tokens = 0

try:
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,  # Enable streaming response
        timeout=900,
    )

    if response.status_code == 200:
        print("Receiving streaming response...")
        print()

        # Process SSE stream
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # SSE format: "data: {...}"
            if line.startswith('data: '):
                chunk_count += 1
                now = time.time()

                if first_chunk_time is None:
                    first_chunk_time = now
                    ttft = first_chunk_time - start_time
                    print(f"⏱️  Time to First Token (TTFT): {ttft:.2f} seconds")
                    print()

                last_chunk_time = now

                # Parse JSON data
                data_str = line[6:]  # Remove "data: " prefix

                if data_str.strip() == '[DONE]':
                    break

                try:
                    event = json.loads(data_str)
                    event_type = event.get('type')

                    if event_type == 'message_start':
                        # Extract input token count from message_start
                        msg = event.get('message', {})
                        usage = msg.get('usage', {})
                        input_tokens = usage.get('input_tokens', 0)
                        print(f"📊 Input tokens: {input_tokens:,}")

                    elif event_type == 'content_block_delta':
                        # Count output tokens (each delta = 1 token typically)
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            output_tokens += 1
                            if output_tokens <= 5:
                                print(f"   Token {output_tokens}: {delta.get('text', '')}")

                    elif event_type == 'message_delta':
                        # Final token counts
                        delta_data = event.get('delta', {})
                        usage = event.get('usage', {})
                        output_tokens = usage.get('output_tokens', output_tokens)

                except json.JSONDecodeError:
                    pass

        end_time = time.time()
        total_elapsed = end_time - start_time

        print()
        print("=" * 60)
        print("STREAMING RESULTS")
        print("=" * 60)
        print(f"✓ Streaming complete")
        print()
        print(f"Total chunks: {chunk_count}")
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens}")
        print()
        print(f"TTFT (prefill time): {ttft:.2f} seconds")
        print(f"Total time: {total_elapsed:.2f} seconds")
        print()

        if input_tokens > 0 and ttft > 0:
            prefill_speed = input_tokens / ttft
            print(f"Prefill speed: {prefill_speed:.2f} tokens/sec")
            print()

            if ttft > 60:
                print(f"⚠️  WARNING: TTFT is {ttft:.0f} seconds ({ttft/60:.1f} minutes)!")
                print(f"   Expected: ~{input_tokens/450:.0f} seconds at 450 tokens/sec")
                print()
                print("   This suggests STREAMING has overhead issues!")
            else:
                print(f"✓ TTFT is good!")

        if output_tokens > 0:
            generation_time = total_elapsed - ttft
            if generation_time > 0:
                gen_speed = output_tokens / generation_time
                print(f"Generation speed: {gen_speed:.2f} tokens/sec")

    else:
        print(f"✗ Request failed")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✗ Error: {e}")
    print(f"Time: {elapsed:.2f} seconds")
    import traceback
    traceback.print_exc()

print("=" * 60)
