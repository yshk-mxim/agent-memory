#!/usr/bin/env python3
"""Debug what prompt is being sent to DeepSeek - compare simple API vs coordination."""
import httpx
import json
import sys

base_url = "http://localhost:8000"

# Test 1: Simple API call (WORKS)
print("=" * 80)
print("TEST 1: Simple OpenAI API call (known to work)")
print("=" * 80)

simple_messages = [
    {"role": "system", "content": "You are a prison warden. Speak only in English. Be brief."},
    {"role": "user", "content": "Warden, introduce yourself."}
]

with httpx.Client(timeout=30.0) as client:
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": simple_messages,
            "max_tokens": 50,
            "temperature": 0.7
        }
    )

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\nSimple API Response:")
        print(f"  Content: {repr(content[:200])}")
        print(f"  Content (raw): {content[:200]}")
        print(f"  Tokens: {result['usage']['completion_tokens']}")
    else:
        print(f"ERROR: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

# Test 2: Coordination service (BROKEN)
print("\n" + "=" * 80)
print("TEST 2: Coordination service (generates Russian)")
print("=" * 80)

session_data = {
    "topology": "round_robin",
    "debate_format": "free_form",
    "decision_mode": "none",
    "agents": [
        {
            "agent_id": "debug_warden",
            "display_name": "Warden",
            "role": "moderator",
            "system_prompt": "You are a prison warden. Speak only in English. Be brief.",
            "lifecycle": "ephemeral"
        }
    ],
    "initial_prompt": "Warden, introduce yourself.",
    "max_turns": 1
}

with httpx.Client(timeout=30.0) as client:
    resp = client.post(f"{base_url}/v1/coordination/sessions", json=session_data)
    if resp.status_code not in (200, 201):
        print(f"ERROR creating session: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    session_id = resp.json()["session_id"]
    print(f"Session ID: {session_id}")

    # Execute one turn
    print("\nExecuting turn 1...")
    resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")
    if resp.status_code != 200:
        print(f"ERROR executing turn: {resp.status_code}")
        print(resp.text)

    # Get messages
    resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
    messages = resp.json()["messages"]

    for msg in messages:
        sender = msg.get("sender_name", "?")
        content = msg.get("content", "")
        print(f"\n{sender}:")
        print(f"  Content: {repr(content[:200])}")
        print(f"  Content (raw): {content[:200]}")

        # Check for Cyrillic
        has_cyrillic = any(ord(c) > 1024 for c in content)
        print(f"  Has Cyrillic: {has_cyrillic}")

print("\n" + "=" * 80)
print("Check server logs for [DEBUG] and [RAW_DECODED] messages to see:")
print("  - Exact prompt text sent to tokenizer")
print("  - Token IDs generated")
print("  - Decoded text before cleaning")
print("=" * 80)
