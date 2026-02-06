#!/usr/bin/env python3
"""Directly test DeepSeek tokenizer to see if it has correct vocab."""
from mlx_lm import load

print("Loading DeepSeek model...")
model, tokenizer = load("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

# Test 1: Simple encoding/decoding
test_text = "Hello, I am a prison warden."
print(f"\nTest 1: Simple encode/decode")
print(f"  Input: {test_text}")
tokens = tokenizer.encode(test_text)
print(f"  Tokens: {tokens[:20]}")
decoded = tokenizer.decode(tokens)
print(f"  Decoded: {decoded}")
print(f"  Match: {decoded.strip() == test_text.strip()}")

# Test 2: Chat template
print(f"\n\nTest 2: Chat template")
messages = [
    {"role": "system", "content": "You are a prison warden."},
    {"role": "user", "content": "Introduce yourself."}
]

if hasattr(tokenizer, "apply_chat_template"):
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"  Formatted prompt (first 300 chars):")
    print(f"  {repr(formatted[:300])}")

    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    print(f"\n  Template tokens (first 20): {tokens[:20]}")

    # Decode the tokens to see what they say
    decoded_prompt = tokenizer.decode(tokens)
    print(f"\n  Decoded template:")
    print(f"  {repr(decoded_prompt[:300])}")
else:
    print("  No apply_chat_template method")

# Test 3: The problematic tokens from logs
print(f"\n\nTest 3: Decode problematic tokens from logs")
prob_tokens = [2285, 63328, 4056, 318, 7936, 32097, 2871, 2796, 94703, 325]
prob_decoded = tokenizer.decode(prob_tokens)
print(f"  Tokens: {prob_tokens}")
print(f"  Decoded: {repr(prob_decoded)}")
print(f"  Is Russian/Cyrillic: {any(ord(c) > 1024 for c in prob_decoded)}")
