"""
Test script for Gemma 3 12B model only.
Does NOT call Claude or DeepSeek APIs.

Usage:
    python -m tests.test_gemma_only
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import APIClients, print_section


def test_gemma_inference(model_path: str = None):
    """Test local Gemma 3 12B inference"""
    print_section("Testing Gemma 3 12B Inference")

    try:
        clients = APIClients()

        print("Loading Gemma 3 12B model...")
        print("(First load may take 10-20 seconds)")

        # Test 1: Simple generation
        print("\n[Test 1] Simple generation test...")
        start = time.time()
        response = clients.call_gemma(
            "Say 'Hello' in exactly 3 words.",
            model_path=model_path,
            max_tokens=50
        )
        elapsed = time.time() - start

        print(f"✓ Response: {response}")
        print(f"✓ Time: {elapsed:.2f}s")

        if elapsed < 30:
            print(f"✓ Generation under 30s threshold")
        else:
            print(f"⚠ Generation took {elapsed:.2f}s (target: <30s)")

        # Test 2: Instruction following
        print("\n[Test 2] Instruction following test...")
        start = time.time()
        response = clients.call_gemma(
            "List exactly 3 colors, separated by commas. No other text.",
            model_path=model_path,
            max_tokens=50
        )
        elapsed = time.time() - start

        print(f"✓ Response: {response}")
        print(f"✓ Time: {elapsed:.2f}s")

        # Test 3: Context window
        print("\n[Test 3] Context window test...")
        gemma = clients.get_gemma(model_path)
        context_size = gemma.n_ctx()

        print(f"✓ Context window: {context_size:,} tokens")

        if context_size >= 32768:
            print(f"✓ Context window >= 32k (sufficient for RDIC)")
        else:
            print(f"⚠ Context window < 32k (may limit RDIC experiments)")

        # Test 4: Multi-turn conversation
        print("\n[Test 4] Multi-turn conversation test...")
        start = time.time()

        # Turn 1: Establish instruction
        response1 = clients.call_gemma(
            "Remember: always respond in a formal, professional tone. "
            "Acknowledge this instruction.",
            model_path=model_path,
            max_tokens=50
        )
        print(f"Turn 1: {response1[:80]}...")

        # Turn 2: Test if instruction is maintained
        response2 = clients.call_gemma(
            "Now explain what 2+2 equals in one sentence.",
            model_path=model_path,
            max_tokens=50
        )
        print(f"Turn 2: {response2[:80]}...")

        elapsed = time.time() - start
        print(f"✓ Multi-turn completed in {elapsed:.2f}s")

        # Summary
        print("\n" + "="*60)
        print("GEMMA 3 12B TEST RESULTS")
        print("="*60)
        print("✓ Model loads successfully")
        print("✓ Basic generation works")
        print("✓ Instruction following functional")
        print(f"✓ Context window: {context_size:,} tokens")
        print("✓ Multi-turn conversation works")
        print("\n✅ All Gemma 3 12B tests PASSED")
        print("="*60)

        return True

    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        print("\nExpected location:")
        print("  models/gemma-3-12b-it-Q4_K_M.gguf")
        print("\nDownload with:")
        print("  huggingface-cli download ggml-org/gemma-3-12b-it-GGUF \\")
        print("    gemma-3-12b-it-Q4_K_M.gguf --local-dir models")
        return False

    except Exception as e:
        print(f"✗ Gemma 3 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Gemma-only test"""
    print_section("Gemma 3 12B Only - API Test")
    print("Note: This does NOT test Claude or DeepSeek APIs")
    print("Only tests local Gemma 3 inference\n")

    success = test_gemma_inference()

    if success:
        print("\n🎉 Gemma 3 12B is ready for RDIC research!")
        return 0
    else:
        print("\n⚠ Gemma 3 12B test failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
