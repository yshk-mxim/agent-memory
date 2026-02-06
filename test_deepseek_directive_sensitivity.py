#!/usr/bin/env python3
"""Test if DeepSeek is sensitive to [respond now.] directives."""
import httpx

base_url = "http://localhost:8000"

# Test Case 1: With [respond now.] directive (coordination style)
print("=" * 80)
print("TEST 1: WITH [respond now.] directive (coordination style)")
print("=" * 80)

messages_with_directive = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34. Under 3 sentences.\n\nRULES: You are Marco and nobody else. Respond in first person as yourself. Never generate dialogue for other characters."},
    {"role": "user", "content": "Interrogation Room A."},
    {"role": "user", "content": "Warden: If both keep silent, 2 years each. If one confesses, confessor goes free."},
    {"role": "user", "content": "[Marco, respond now.]"}
]

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": messages_with_directive,
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
        print(f"Content: {repr(content[:200])}")

        is_refusal = "sorry" in content.lower() and ("can't" in content.lower() or "cannot" in content.lower())
        print(f"Is refusal: {is_refusal}")

        if content:
            print(f"\nReadable:\n{content[:200]}")
    else:
        print(f"ERROR: {resp.status_code}")

# Test Case 2: WITHOUT [respond now.] directive (simple)
print("\n" + "=" * 80)
print("TEST 2: WITHOUT [respond now.] directive (simple)")
print("=" * 80)

messages_without_directive = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34. Under 3 sentences.\n\nRULES: You are Marco and nobody else. Respond in first person as yourself. Never generate dialogue for other characters."},
    {"role": "user", "content": "Interrogation Room A."},
    {"role": "user", "content": "Warden: If both keep silent, 2 years each. If one confesses, confessor goes free."},
    {"role": "user", "content": "Marco, what do you say?"}
]

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": messages_without_directive,
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
        print(f"Content: {repr(content[:200])}")

        is_refusal = "sorry" in content.lower() and ("can't" in content.lower() or "cannot" in content.lower())
        print(f"Is refusal: {is_refusal}")

        if content:
            print(f"\nReadable:\n{content[:200]}")
    else:
        print(f"ERROR: {resp.status_code}")

# Test Case 3: WITHOUT RULES and WITHOUT directive (minimal)
print("\n" + "=" * 80)
print("TEST 3: WITHOUT RULES and WITHOUT [respond now.] (minimal)")
print("=" * 80)

messages_minimal = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34. Under 3 sentences."},
    {"role": "user", "content": "Interrogation Room A. The Warden says: If both keep silent, 2 years each. If one confesses, confessor goes free. Marco, what do you say?"}
]

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": messages_minimal,
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
        print(f"Content: {repr(content[:200])}")

        is_refusal = "sorry" in content.lower() and ("can't" in content.lower() or "cannot" in content.lower())
        print(f"Is refusal: {is_refusal}")

        if content:
            print(f"\nReadable:\n{content[:200]}")
    else:
        print(f"ERROR: {resp.status_code}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Compare token counts and refusal rates:")
print("  Test 1 (WITH directive): Check if it triggers refusal or empty output")
print("  Test 2 (WITHOUT directive): Check if it works better")
print("  Test 3 (Minimal): Baseline to see clean performance")
print("=" * 80)
