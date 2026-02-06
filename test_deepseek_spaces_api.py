#!/usr/bin/env python3
"""Test if simple API also has space stripping with same prompt."""
import httpx

base_url = "http://localhost:8000"

# Use the EXACT same system prompt as coordination
messages = [
    {"role": "system", "content": "Your name is Warden. Pragmatic corrections officer. You know the punishment rules: If both suspects keep silent, they each get 2 years. If one confesses and the other keeps silent, the confessor goes free and the silent one gets 10 years. If both confess, they each get 5 years. You use this information strategically to pressure suspects. Speak plainly. Under 3 sentences.\n\nRULES: You are Warden and nobody else. Respond in first person as yourself. Never generate dialogue for other characters. Never prefix your response with any name or label."},
    {"role": "user", "content": "Interrogation Room A. The Warden addresses Marco directly."},
    {"role": "user", "content": "Warden, what do you say?"}
]

print("Testing simple API with coordination-style prompt...")
print(f"Message count: {len(messages)}")
print(f"Roles: {[m['role'] for m in messages]}")
print()

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
    )

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]

        print(f"Tokens: {tokens}")
        print(f"Length: {len(content)}")
        print(f"\nContent (repr):")
        print(repr(content))

        print(f"\nContent (readable):")
        print(content)

        # Check for spaces
        word_count = len(content.split())
        char_count_no_spaces = len(content.replace(" ", ""))
        space_ratio = (len(content) - char_count_no_spaces) / len(content) if len(content) > 0 else 0

        print(f"\nSpace analysis:")
        print(f"  Word count: {word_count}")
        print(f"  Characters (no spaces): {char_count_no_spaces}")
        print(f"  Space ratio: {space_ratio:.2%}")

        if space_ratio < 0.10:  # Less than 10% spaces is suspicious
            print(f"  ⚠️ WARNING: Very few spaces! Space stripping detected.")
        else:
            print(f"  ✓ Normal spacing")
    else:
        print(f"ERROR: {resp.status_code}")
        print(resp.text)

print("\n" + "=" * 80)
print("Check server logs for [RAW_DECODED] to see if spaces are missing at decode time")
print("=" * 80)
