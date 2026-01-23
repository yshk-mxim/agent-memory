#!/usr/bin/env python3
"""Debug MLX context building."""

import json

# Load validation example
with open('data/3cluster_examples/validation_001_software_eng.json', 'r') as f:
    example = json.load(f)

print("Example ID:", example['id'])
print("\nTurns in example:")

# Check what turns exist
for i in range(1, 16):
    turn_key = f"turn_{i}"
    if turn_key in example:
        turn = example[turn_key]
        text = turn.get("text", "")
        print(f"\n{turn_key}: {text[:100]}...")
    else:
        print(f"\n{turn_key}: NOT FOUND")

# Test context building
all_turns = []
for i in range(1, 16):
    turn_key = f"turn_{i}"
    if turn_key in example:
        all_turns.append(example[turn_key])

full_context = "\n\n".join([t.get("text", "") for t in all_turns if t.get("text")])

print("\n" + "="*60)
print("FULL CONTEXT:")
print("="*60)
print(full_context[:500])
print("...")
print(f"\nTotal context length: {len(full_context)} characters")
