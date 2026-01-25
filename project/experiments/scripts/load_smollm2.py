#!/usr/bin/env python3
"""SmolLM2-135M fixture for Sprint 2 experiments.

This script loads SmolLM2-135M-Instruct for fast testing in experiments.
The model is cached after first load, making subsequent loads instant.

Usage:
    python load_smollm2.py                    # Load and print model info
    python -c "from load_smollm2 import get_smollm2; get_smollm2()"
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


def get_smollm2():
    """Load SmolLM2-135M-Instruct for testing.

    This model is small (135M parameters) and fast, making it ideal for
    experiment validation before testing on production models (Gemma 3, etc.).

    Returns:
        tuple: (model, tokenizer) from mlx_lm.load()

    Note:
        Model is cached in ~/.cache/huggingface/hub/ after first download.
        Subsequent loads are near-instant (no network required).
    """
    try:
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    model_id = "mlx-community/SmolLM2-135M-Instruct"

    print(f"Loading {model_id}...")
    model, tokenizer = load(model_id)
    print(f"✅ Loaded successfully!")

    return model, tokenizer


def print_model_info(model, tokenizer):
    """Print model configuration and tokenizer info.

    Args:
        model: MLX model object
        tokenizer: MLX tokenizer object
    """
    args = model.args

    print("\n📊 Model Info:")
    print(f"  Model type: {getattr(args, 'model_type', 'unknown')}")
    print(f"  Layers: {getattr(args, 'num_hidden_layers', 'N/A')}")
    print(f"  KV heads: {getattr(args, 'num_key_value_heads', 'N/A')}")
    print(f"  Attention heads: {getattr(args, 'num_attention_heads', 'N/A')}")
    print(f"  Hidden size: {getattr(args, 'hidden_size', 'N/A')}")
    print(f"  Head dim: {getattr(args, 'hidden_size', 0) // getattr(args, 'num_attention_heads', 1)}")
    print(f"  Sliding window: {getattr(args, 'sliding_window', 'None (full attention)')}")

    print("\n📊 Tokenizer Info:")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  EOS token ID: {tokenizer.eos_token_ids}")
    print(f"  BOS token: {getattr(tokenizer, 'bos_token', 'N/A')}")

    print("\n💾 Cache Location:")
    print(f"  ~/.cache/huggingface/hub/models--mlx-community--SmolLM2-135M-Instruct")


def test_generation(model, tokenizer):
    """Test basic generation to verify model works.

    Args:
        model: MLX model object
        tokenizer: MLX tokenizer object
    """
    from mlx_lm import generate

    prompt = "The quick brown fox"
    print(f"\n🧪 Test Generation:")
    print(f"  Prompt: '{prompt}'")

    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=20,
        temp=0.0,  # Greedy
        verbose=False,
    )

    print(f"  Output: '{result}'")
    print(f"  ✅ Generation works!")


if __name__ == "__main__":
    # Load model
    model, tokenizer = get_smollm2()

    # Print info
    print_model_info(model, tokenizer)

    # Test generation
    test_generation(model, tokenizer)

    print("\n✅ SmolLM2-135M fixture ready for experiments!")
    print("   Use: from load_smollm2 import get_smollm2")
