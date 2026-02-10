# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Verify tokenizer accepts 100K tokens after fix.

This test verifies that the tokenizer override in mlx_model_loader.py
successfully extends the context limit from 16K to 100K tokens.

Run:
    python tests/manual/test_tokenizer_fix.py

Expected:
    ✓ Tokenizer accepts 100K tokens
    ✓ Can encode sequences >16K tokens
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_memory.adapters.outbound.mlx_model_loader import MLXModelLoader


def test_tokenizer_max_length():
    """Verify tokenizer model_max_length is 100K."""
    print("=" * 60)
    print("Test 1: Verify tokenizer max_length")
    print("=" * 60)

    loader = MLXModelLoader()
    model, tokenizer = loader.load_model("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

    print("✓ Model loaded successfully")
    print(f"✓ Tokenizer max length: {tokenizer.model_max_length:,} tokens")

    assert tokenizer.model_max_length == 100000, f"Expected 100K, got {tokenizer.model_max_length}"

    print("✓ Tokenizer accepts 100K tokens!")


def test_long_sequence():
    """Test tokenizer with 20K token sequence (exceeds old 16K limit)."""
    print("\n" + "=" * 60)
    print("Test 2: Verify long sequence handling (>16K tokens)")
    print("=" * 60)

    loader = MLXModelLoader()
    model, tokenizer = loader.load_model("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

    # Generate text that will exceed 16K tokens
    # Repeat a sentence many times to create a very long input
    long_text = (
        "This is a test sentence with many words that should tokenize to approximately one hundred tokens when encoded by the DeepSeek tokenizer model. "
        * 1000
    )
    tokens = tokenizer.encode(long_text)

    print(f"✓ Encoded {len(tokens):,} tokens successfully")
    assert len(tokens) > 16384, f"Should accept >16K tokens, got {len(tokens):,}"
    print(f"✓ Tokenizer accepts sequences >16K tokens (got {len(tokens):,} tokens)!")


if __name__ == "__main__":
    try:
        test_tokenizer_max_length()
        test_long_sequence()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nTokenizer fix verified:")
        print("  • Max length: 100,000 tokens")
        print("  • Can process sequences >16K tokens")
        print("  • Ready for Claude Code CLI (18K+ prompts)")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
