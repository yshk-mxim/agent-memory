"""
Test script for API connections.

Run this once network connectivity is restored to verify all APIs work.

Usage:
    python -m tests.test_apis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import APIClients, print_section


def test_claude_api():
    """Test Claude API connection"""
    print_section("Testing Claude API (Haiku 4.5)")

    try:
        clients = APIClients()
        response = clients.call_claude(
            "Say 'Hello' in exactly 3 words.",
            model="claude-haiku-4-5-20251001",
            max_tokens=50
        )
        print(f"✓ Claude Haiku 4.5 API works!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Claude Haiku 4.5 API failed: {e}")
        return False


def test_claude_sonnet_api():
    """Test Claude Sonnet 4.5 API connection"""
    print_section("Testing Claude API (Sonnet 4.5)")

    try:
        clients = APIClients()
        response = clients.call_claude(
            "Say 'Hello' in exactly 3 words.",
            model="claude-sonnet-4-5-20250929",
            max_tokens=50
        )
        print(f"✓ Claude Sonnet 4.5 API works!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Claude Sonnet 4.5 API failed: {e}")
        return False


def test_deepseek_api():
    """Test DeepSeek R1 API connection"""
    print_section("Testing DeepSeek R1 API")

    try:
        clients = APIClients()
        response = clients.call_deepseek_r1(
            "What is 2+2? Answer in one short sentence.",
            model="deepseek-reasoner",
            max_tokens=100,
            return_reasoning=True
        )
        print(f"✓ DeepSeek R1 API works!")
        print(f"Response: {response['content']}")
        if response.get('reasoning'):
            print(f"Reasoning trace: {response['reasoning'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ DeepSeek R1 API failed: {e}")
        return False


def test_llama_inference(model_path: str = None):
    """Test local Llama inference"""
    print_section("Testing Local Llama Inference")

    try:
        import time
        clients = APIClients()

        print("Testing Llama generation speed...")
        start = time.time()
        response = clients.call_llama(
            "Say 'Hello' in exactly 3 words.",
            model_path=model_path,
            max_tokens=50
        )
        elapsed = time.time() - start

        print(f"✓ Llama inference works!")
        print(f"Response: {response}")
        print(f"Time: {elapsed:.2f}s")

        if elapsed < 30:
            print(f"✓ Generation under 30s threshold")
            return True
        else:
            print(f"⚠ Generation took {elapsed:.2f}s (target: <30s)")
            return False

    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        print("Download model from:")
        print("  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        return False
    except Exception as e:
        print(f"✗ Llama inference failed: {e}")
        return False


def main():
    """Run all API tests"""
    print_section("Day 1 API Verification Tests")

    results = {
        "claude-haiku-4.5": test_claude_api(),
        "claude-sonnet-4.5": test_claude_sonnet_api(),
        "deepseek": test_deepseek_api(),
        "llama": test_llama_inference()
    }

    # Summary
    print_section("Test Summary")
    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All Day 1 requirements met!")
        print("Ready to proceed to Day 2")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
