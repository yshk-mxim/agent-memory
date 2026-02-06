#!/usr/bin/env python3
"""Compare DeepSeek behavior: Simple OpenAI API vs Coordination Service."""
import httpx
import json

base_url = "http://localhost:8000"

print("=" * 80)
print("TEST 1: Simple OpenAI API - Multi-turn conversation")
print("=" * 80)

# Test multi-turn via simple API
simple_messages = [
    {"role": "system", "content": "Your name is Marco. Blue-collar worker, 34, first offense, anxious about your family. Keep responses under 3 sentences."},
    {"role": "user", "content": "Interrogation Room A. The Warden addresses Marco directly."},
    {"role": "user", "content": "Warden: I am the Warden. If both suspects keep silent, you each get 2 years. If one confesses and the other keeps silent, the confessor goes free and the silent one gets 10 years. If both confess, you each get 5 years."},
    {"role": "user", "content": "Marco, what do you say?"}
]

with httpx.Client(timeout=30.0) as client:
    print("\nSending to /v1/chat/completions...")
    print(f"Message count: {len(simple_messages)}")
    print(f"Roles: {[m['role'] for m in simple_messages]}")

    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek",
            "messages": simple_messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
    )

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]

        print(f"\n✓ Response received:")
        print(f"  Tokens: {tokens}")
        print(f"  Content length: {len(content)}")
        print(f"  Content: {repr(content)}")
        print(f"\n  Readable:\n  {content}")

        # Check for issues
        has_spaces = " " in content
        has_chinese = any(ord(c) > 1024 for c in content)
        is_refusal = "sorry" in content.lower() and "can't" in content.lower()

        print(f"\n  Analysis:")
        print(f"    Has spaces: {has_spaces}")
        print(f"    Has Chinese/Cyrillic: {has_chinese}")
        print(f"    Is refusal: {is_refusal}")
    else:
        print(f"✗ ERROR: {resp.status_code}")
        print(resp.text)

print("\n" + "=" * 80)
print("TEST 2: Coordination Service - Same scenario")
print("=" * 80)

session_data = {
    "topology": "round_robin",
    "debate_format": "free_form",
    "decision_mode": "none",
    "agents": [
        {
            "agent_id": "test_marco",
            "display_name": "Marco",
            "role": "participant",
            "system_prompt": "Your name is Marco. Blue-collar worker, 34, first offense, anxious about your family. Keep responses under 3 sentences.",
            "lifecycle": "ephemeral"
        }
    ],
    "initial_prompt": "Interrogation Room A. The Warden addresses Marco directly.\n\nWarden: I am the Warden. If both suspects keep silent, you each get 2 years. If one confesses and the other keeps silent, the confessor goes free and the silent one gets 10 years. If both confess, you each get 5 years.\n\nMarco, what do you say?",
    "max_turns": 1
}

with httpx.Client(timeout=30.0) as client:
    print("\nCreating coordination session...")
    resp = client.post(f"{base_url}/v1/coordination/sessions", json=session_data)

    if resp.status_code not in (200, 201):
        print(f"✗ ERROR creating session: {resp.status_code}")
        print(resp.text)
    else:
        session_id = resp.json()["session_id"]
        print(f"Session ID: {session_id}")

        print("\nExecuting turn...")
        resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")

        if resp.status_code != 200:
            print(f"✗ ERROR executing turn: {resp.status_code}")
            print(resp.text)

        # Get messages
        resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
        messages = resp.json()["messages"]

        print(f"\n✓ Response received:")
        print(f"  Message count: {len(messages)}")

        for msg in messages:
            sender = msg.get("sender_name", "?")
            content = msg.get("content", "")

            print(f"\n  {sender}:")
            print(f"    Length: {len(content)}")
            print(f"    Content: {repr(content[:200])}")

            if content:
                has_spaces = " " in content
                has_chinese = any(ord(c) > 1024 for c in content)
                is_refusal = "sorry" in content.lower() and "can't" in content.lower()

                print(f"    Has spaces: {has_spaces}")
                print(f"    Has Chinese/Cyrillic: {has_chinese}")
                print(f"    Is refusal: {is_refusal}")

                if len(content) > 0:
                    print(f"\n    Readable:\n    {content[:300]}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Check the server logs for:")
print("  - [TOKENIZE] and [TEMPLATE_MESSAGES] entries")
print("  - [FORMATTED_PROMPT] to see actual prompt text")
print("  - [RAW_DECODED] to see decoded tokens")
print("=" * 80)
