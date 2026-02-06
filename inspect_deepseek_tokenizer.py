#!/usr/bin/env python3
"""Inspect DeepSeek tokenizer chat template."""
from transformers import AutoTokenizer

model_id = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("="*80)
print("DeepSeek Tokenizer Inspection")
print("="*80)
print(f"\nModel: {model_id}")
print(f"\nHas chat_template: {hasattr(tokenizer, 'chat_template')}")
print(f"Chat template: {tokenizer.chat_template if hasattr(tokenizer, 'chat_template') else 'None'}")

# Test tokenization
messages = [
    {"role": "system", "content": "You are Marco, a suspect."},
    {"role": "user", "content": "What do you have to say?"},
]

print("\n" + "="*80)
print("Test Messages:")
print("="*80)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

if hasattr(tokenizer, "apply_chat_template"):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    text = tokenizer.decode(tokens)
    print("\n" + "="*80)
    print("Tokenized Prompt:")
    print("="*80)
    print(repr(text))
    print("\n" + "="*80)
    print("Decoded (visible):")
    print("="*80)
    print(text)

    # Test generation
    print("\n" + "="*80)
    print("Simulated Response:")
    print("="*80)
    response_text = "I didn't do anything wrong."
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    decoded_response = tokenizer.decode(response_tokens)
    print(f"Original: {repr(response_text)}")
    print(f"Encoded: {response_tokens}")
    print(f"Decoded: {repr(decoded_response)}")
else:
    print("\nNo apply_chat_template method found")
