#!/usr/bin/env python3
"""Test if cache replay causes space stripping."""
import httpx
import json

base_url = "http://localhost:8000"

print("Creating two-agent conversation to test cache replay...")
print()

# Phase 1: Warden speaks (fresh, no cache)
session_data = {
    "topology": "round_robin",
    "debate_format": "free_form",
    "decision_mode": "none",
    "agents": [
        {
            "agent_id": "test_warden",
            "display_name": "Warden",
            "role": "moderator",
            "system_prompt": "You are a prison warden. Under 2 sentences.",
            "lifecycle": "permanent"  # Will cache KV
        },
        {
            "agent_id": "test_marco",
            "display_name": "Marco",
            "role": "participant",
            "system_prompt": "You are an inmate named Marco. Under 2 sentences.",
            "lifecycle": "permanent"  # Will cache KV
        }
    ],
    "initial_prompt": "Warden, introduce yourself.",
    "max_turns": 4  # Warden, Marco, Warden, Marco
}

with httpx.Client(timeout=30.0) as client:
    # Create session
    resp = client.post(f"{base_url}/v1/coordination/sessions", json=session_data)
    session_id = resp.json()["session_id"]
    print(f"Session ID: {session_id}\n")

    # Turn 1: Warden speaks (FRESH)
    print("=" * 80)
    print("TURN 1: Warden speaks (fresh, no cache)")
    print("=" * 80)
    resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")
    resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
    messages = resp.json()["messages"]
    warden_turn1 = [m for m in messages if m.get("sender_name") == "Warden"][-1]["content"]
    print(f"Warden: {repr(warden_turn1[:100])}")
    print(f"Has spaces: {' ' in warden_turn1}")
    print()

    # Turn 2: Marco speaks (sees Warden's response in context, uses cache?)
    print("=" * 80)
    print("TURN 2: Marco speaks (Warden's response in context)")
    print("=" * 80)
    resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")
    resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
    messages = resp.json()["messages"]
    marco_turn2 = [m for m in messages if m.get("sender_name") == "Marco"][-1]["content"]
    print(f"Marco: {repr(marco_turn2[:100])}")
    print(f"Has spaces: {' ' in marco_turn2}")
    print()

    # Turn 3: Warden speaks AGAIN (uses cache from turn 1, sees Marco's response)
    print("=" * 80)
    print("TURN 3: Warden speaks AGAIN (cache replay)")
    print("=" * 80)
    resp = client.post(f"{base_url}/v1/coordination/sessions/{session_id}/turn")
    resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}/messages")
    messages = resp.json()["messages"]
    warden_turn3 = [m for m in messages if m.get("sender_name") == "Warden" and m.get("turn") == 2]
    if warden_turn3:
        warden_turn3 = warden_turn3[0]["content"]
        print(f"Warden: {repr(warden_turn3[:100])}")
        print(f"Has spaces: {' ' in warden_turn3}")
        print()

        # Compare
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"Turn 1 (fresh):  spaces={' ' in warden_turn1}")
        print(f"Turn 3 (cached): spaces={' ' in warden_turn3}")

        if (' ' in warden_turn1) != (' ' in warden_turn3):
            print("\n⚠️  INCONSISTENCY DETECTED!")
            print("Cache replay changes space handling!")
        else:
            print("\n✓ Consistent behavior")
    else:
        print("No Warden response in turn 3")

print("\n" + "=" * 80)
print("Check server logs for cache_reuse events and [RAW_DECODED] outputs")
print("=" * 80)
