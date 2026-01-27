#!/usr/bin/env python3
"""Test Q4 direct injection via API calls to semantic server.

Simpler approach: Use semantic server's API to test Q4 injection.
"""

import requests
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_q4_injection_via_api():
    """Test Q4 injection by making API calls and observing server logs."""

    base_url = "http://localhost:8000"
    agent_id = "test_q4_1k"

    # Create a 1K token prompt
    prompt = "Explain what machine learning is in detail. " * 100  # ~1K tokens

    logger.info("="*60)
    logger.info("Q4 DIRECT INJECTION TEST VIA API (1K tokens)")
    logger.info("="*60)

    # Request 1: Generate cache
    logger.info("\n[REQUEST 1] Generating cache...")
    payload = {
        "agent_id": agent_id,
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.0,
    }

    response1 = requests.post(f"{base_url}/v1/agents", json=payload)

    if response1.status_code != 200:
        logger.error(f"❌ Request 1 failed: {response1.status_code} {response1.text}")
        return False

    result1 = response1.json()
    logger.info(f"✅ Response 1: {result1['completion'][:100]}...")

    # Wait a bit for cache to be saved
    time.sleep(2)

    # Request 2: Load from cache (Q4 injection should happen here!)
    logger.info("\n[REQUEST 2] Loading from cache (Q4 injection)...")

    response2 = requests.post(f"{base_url}/v1/agents", json=payload)

    if response2.status_code != 200:
        logger.error(f"❌ Request 2 failed: {response2.status_code} {response2.text}")
        return False

    result2 = response2.json()
    logger.info(f"✅ Response 2: {result2['completion'][:100]}...")

    logger.info("\n" + "="*60)
    logger.info("CHECK SERVER LOGS FOR:")
    logger.info("  - [Q4 INJECT L##] messages (should see Q4 injection, not dequantization)")
    logger.info("  - Memory spike should be minimal")
    logger.info("="*60)

    return True


def main():
    """Run API-based Q4 injection test."""

    logger.info("Starting Q4 injection test via API...")
    logger.info("Make sure semantic server is running on http://localhost:8000")

    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ Server is not responding correctly")
            return 1
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to server. Is it running?")
        logger.error("   Start with: semantic serve")
        return 1

    logger.info("✅ Server is running")

    # Run test
    success = test_q4_injection_via_api()

    if success:
        logger.info("\n🎉 TEST COMPLETED - Check server logs for Q4 injection messages!")
        return 0
    else:
        logger.error("\n❌ TEST FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
