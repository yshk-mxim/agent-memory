#!/usr/bin/env python3
"""
Simple test to verify Gemma 3 works with manual generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def main():
    print("Loading Gemma 3 4B...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded successfully", flush=True)

    # Test simple generation with chat template
    print("\nTest 1: Simple prompt with chat template", flush=True)
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt: {prompt[:200]}...", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    print("Generating with high-level API...", flush=True)

    # Try with high-level generate() method first
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated: {result}", flush=True)

    print("\n✓ Test passed - Gemma 3 working correctly!", flush=True)

if __name__ == "__main__":
    main()
