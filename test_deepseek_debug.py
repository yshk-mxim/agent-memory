#!/usr/bin/env python3
"""Quick DeepSeek coordination test to check raw output."""
import httpx
import time

base_url = "http://localhost:8000"

# Create a simple 2-agent session
session_data = {
    "topology": "round_robin",
    "debate_format": "free_form",
    "decision_mode": "none",
    "agents": [
        {
            "agent_id": "test_warden",
            "display_name": "Warden",
            "role": "moderator",
            "system_prompt": "You are a prison warden. Be brief (1-2 sentences).",
            "lifecycle": "ephemeral"
        },
        {
            "agent_id": "test_marco",
            "display_name": "Marco",
            "role": "participant",
            "system_prompt": "You are Marco, a suspect. Be brief (1-2 sentences).",
            "lifecycle": "ephemeral"
        }
    ],
    "initial_prompt": "Interrogation begins. Warden speaks first.",
    "max_turns": 10
}

print("Creating session...")
with httpx.Client(timeout=30.0) as client:
    resp = client.post(f"{base_url}/v1/coordination/sessions", json=session_data)
    session_id = resp.json()["session_id"]
    print(f"Session ID: {session_id}")

    # Execute 4 turns
    for i in range(4):
        print(f"\nTurn {i+1}...")
        resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")
        if resp.status_code == 200:
            print(f"  Status: OK")
        else:
            print(f"  Error: {resp.status_code}")
            break
        time.sleep(0.5)

    # Get messages
    resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
    messages = resp.json()["messages"]

    print(f"\n{'='*80}")
    print(f"Messages ({len(messages)} total):")
    print(f"{'='*80}")
    for msg in messages:
        sender = msg.get("sender_name", "?")
        content = msg.get("content", "")
        print(f"{sender}: {repr(content)}")

print("\nCheck /tmp/deepseek_debug.log for raw_generation_output logs")
