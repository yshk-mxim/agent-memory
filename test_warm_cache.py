#!/usr/bin/env python3
"""Test warm cache save/load functionality."""

import asyncio
import httpx

BASE_URL = "http://localhost:8399"


async def test_warm_cache():
    """Test the warm cache save/load flow."""
    print("Starting warm cache test...")

    # Step 1: Prime the cache
    print("\n1. Prime request (create cache)...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Hello! " * 100}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            headers={"X-Session-ID": "test_warm_123"},
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()['choices'][0]['message']['content'][:50]}...")

    # Step 2: Check if agent exists
    print("\n2. Check agent status...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{BASE_URL}/v1/agents/oai_test_warm_123")
        if response.status_code == 200:
            data = response.json()
            print(f"   Agent exists: {data['agent_id']}, {data['cache_size_tokens']} tokens")
        else:
            print(f"   Agent not found: {response.status_code}")

    # Step 3: Evict (keep disk file)
    print("\n3. Evict from hot tier (keep_disk=True)...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.delete(
            f"{BASE_URL}/v1/agents/oai_test_warm_123?evict_only=true"
        )
        print(f"   Status: {response.status_code}")

    # Check if cache file exists
    import os
    cache_path = os.path.expanduser("~/.semantic/caches/oai_test_warm_123.safetensors")
    print(f"\n4. Check if cache file exists on disk...")
    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"   ✓ Cache file exists: {size_mb:.2f} MB")
    else:
        print(f"   ✗ Cache file NOT found at {cache_path}")
        print(f"   This is the bug! File should have been created during eviction.")

    # Step 4: Wait a moment
    await asyncio.sleep(1.0)

    # Step 5: Warm reload test
    print("\n5. Warm reload test (should load from disk)...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Hello! " * 100}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            headers={"X-Session-ID": "test_warm_123"},
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()['choices'][0]['message']['content'][:50]}...")

    # Clean up
    print("\n6. Cleanup...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.delete(f"{BASE_URL}/v1/agents/oai_test_warm_123")
    print("   Done")


if __name__ == "__main__":
    asyncio.run(test_warm_cache())
