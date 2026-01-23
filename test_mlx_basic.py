#!/usr/bin/env python3
"""
Basic MLX test - verify generation works with Gemma model.
"""

import time
import mlx.core as mx
from mlx_lm import load, generate

def test_basic_generation():
    """Test basic MLX generation."""
    print("=" * 60)
    print("MLX Basic Generation Test")
    print("=" * 60)

    # Load model (will download first time)
    print("\n1. Loading Gemma 2 9B (4-bit)...")
    start = time.time()
    model, tokenizer = load("mlx-community/gemma-2-9b-it-4bit")
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.2f}s")

    # Test simple generation
    print("\n2. Testing simple generation...")
    prompt = "The capital of France is"

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=20,
        verbose=False
    )
    gen_time = time.time() - start

    print(f"   Prompt: {prompt}")
    print(f"   Response: {response}")
    print(f"   Time: {gen_time:.2f}s")

    # Test chat format
    print("\n3. Testing chat format...")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=50,
        verbose=False
    )
    gen_time = time.time() - start

    print(f"   Prompt: {messages[0]['content']}")
    print(f"   Response: {response}")
    print(f"   Time: {gen_time:.2f}s")

    print("\n" + "=" * 60)
    print("✓ MLX generation working!")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_generation()
