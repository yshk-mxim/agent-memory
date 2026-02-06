#!/usr/bin/env python3
"""Test DeepSeek chat template to see why it generates Chinese."""
from mlx_lm import load
import mlx.core as mx

print("Loading DeepSeek model...")
model, tokenizer = load("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

# Test 1: Simple messages (like OpenAI API - works)
print("\n" + "=" * 80)
print("TEST 1: Simple API-style messages (WORKS)")
print("=" * 80)

simple_messages = [
    {"role": "system", "content": "You are a prison warden. Speak only in English. Be brief."},
    {"role": "user", "content": "Warden, introduce yourself."}
]

formatted_simple = tokenizer.apply_chat_template(
    simple_messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Formatted prompt length: {len(formatted_simple)}")
print(f"First 300 chars:\n{repr(formatted_simple[:300])}")

tokens_simple = tokenizer.apply_chat_template(
    simple_messages,
    tokenize=True,
    add_generation_prompt=True
)
print(f"\nToken count: {len(tokens_simple)}")
print(f"First 10 tokens: {tokens_simple[:10]}")

# Test 2: Multi-turn with Chinese-generating pattern (coordination style)
print("\n" + "=" * 80)
print("TEST 2: Coordination-style multi-turn (GENERATES CHINESE)")
print("=" * 80)

coord_messages = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34, first offense, anxious about your family. You know the warden might tell the other suspect what you say. Anything you say in public areas like the yard might be overheard. You worry about doing the right thing but also about self-preservation. Keep responses under 3 sentences."},
    {"role": "user", "content": "Interrogation Room A. The Warden addresses Marco directly."},
    {"role": "user", "content": "Warden: I am the Warden, and as such, I must address you directly. The rules of the game are clear: if both suspects remain silent, they each receive two years. If one confesses and the other remains silent, the confessor goes free while the silent one receives ten years. If both confess, they each get five years."},
    {"role": "user", "content": "[Marco, respond now.]"}
]

formatted_coord = tokenizer.apply_chat_template(
    coord_messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Formatted prompt length: {len(formatted_coord)}")
print(f"First 500 chars:\n{repr(formatted_coord[:500])}")
print(f"\nLast 200 chars:\n{repr(formatted_coord[-200:])}")

tokens_coord = tokenizer.apply_chat_template(
    coord_messages,
    tokenize=True,
    add_generation_prompt=True
)
print(f"\nToken count: {len(tokens_coord)}")
print(f"First 10 tokens: {tokens_coord[:10]}")
print(f"Last 10 tokens: {tokens_coord[-10:]}")

# Test 3: Check if consecutive user messages cause issues
print("\n" + "=" * 80)
print("TEST 3: Analysis - consecutive user messages")
print("=" * 80)

print(f"Simple messages roles: {[m['role'] for m in simple_messages]}")
print(f"Coord messages roles: {[m['role'] for m in coord_messages]}")
print("\nNotice: Coordination has consecutive user messages (user, user, user)")
print("This might be confusing DeepSeek's chat template!")

# Test 4: Try with proper alternation
print("\n" + "=" * 80)
print("TEST 4: Fixed version - merge consecutive user messages")
print("=" * 80)

fixed_messages = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34, first offense, anxious about your family. You know the warden might tell the other suspect what you say. Anything you say in public areas like the yard might be overheard. You worry about doing the right thing but also about self-preservation. Keep responses under 3 sentences."},
    {"role": "user", "content": "Interrogation Room A. The Warden addresses Marco directly.\n\nWarden: I am the Warden, and as such, I must address you directly. The rules of the game are clear: if both suspects remain silent, they each receive two years. If one confesses and the other remains silent, the confessor goes free while the silent one receives ten years. If both confess, they each get five years.\n\n[Marco, respond now.]"}
]

formatted_fixed = tokenizer.apply_chat_template(
    fixed_messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Formatted prompt length: {len(formatted_fixed)}")
print(f"First 500 chars:\n{repr(formatted_fixed[:500])}")

tokens_fixed = tokenizer.apply_chat_template(
    fixed_messages,
    tokenize=True,
    add_generation_prompt=True
)
print(f"\nToken count: {len(tokens_fixed)}")
print(f"Roles: {[m['role'] for m in fixed_messages]}")
