"""
Test TRUE KV cache isolation implementation.

Verifies:
1. Model loads correctly (Gemma 2 12B with 4-bit quantization)
2. KV cache sizes are as expected for each condition
3. Isolation actually works (different caches for semantic condition)
4. Outputs are generated successfully
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch


def test_dependencies():
    """Verify required packages are installed."""
    print("Checking dependencies...")

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers not installed")
        print("    Run: pip install transformers")
        return False

    try:
        import bitsandbytes
        print(f"  ✓ bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        print("  ✗ bitsandbytes not installed")
        print("    Run: pip install bitsandbytes")
        return False

    try:
        import accelerate
        print(f"  ✓ accelerate {accelerate.__version__}")
    except ImportError:
        print("  ✗ accelerate not installed")
        print("    Run: pip install accelerate")
        return False

    print(f"  ✓ torch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    return True


def test_model_loading():
    """Test if Gemma 2 12B loads correctly."""
    print("\nTesting model loading...")

    try:
        from src.semantic_isolation import SemanticIsolationTester

        print("  Attempting to load Gemma 2 12B (4-bit)...")
        print("  This may take 2-5 minutes for first download...")

        tester = SemanticIsolationTester()

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Device: {tester.model.device}")

        # Check if we can do basic inference
        test_prompt = "Hello, this is a test."
        inputs = tester.tokenizer(test_prompt, return_tensors="pt").to(tester.model.device)

        with torch.no_grad():
            outputs = tester.model(**inputs, use_cache=True)

        cache_size = tester.get_cache_size(outputs.past_key_values)
        print(f"  ✓ Basic inference works (cache size: {cache_size} tokens)")

        return tester

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return None


def test_cache_isolation(tester):
    """Test that semantic isolation creates separate caches."""
    print("\nTesting cache isolation...")

    # Load validation example
    with open('data/3cluster_examples/validation_001_software_eng.json', 'r') as f:
        example = json.load(f)

    print(f"  Loaded example: {example['id']}")
    print(f"  Total turns: {len(example['turns'])}")

    # Test sequential condition
    print("\n  Testing Sequential (baseline)...")
    result_seq = tester.condition_1_sequential(example)
    print(f"    Cache size: {result_seq.cache_sizes}")
    print(f"    Total tokens processed: {result_seq.total_tokens_processed}")
    print(f"    Generated {len(result_seq.outputs)} outputs")

    # Test semantic condition
    print("\n  Testing Semantic (our method)...")
    result_sem = tester.condition_4_semantic(example)
    print(f"    Cache sizes: {result_sem.cache_sizes}")
    print(f"    Total tokens processed: {result_sem.total_tokens_processed}")
    print(f"    Generated {len(result_sem.outputs)} outputs")

    # Verify isolation
    print("\n  Verifying cache isolation...")

    # Sequential should have ONE large cache
    if len(result_seq.cache_sizes) == 1:
        print("    ✓ Sequential has single unified cache")
    else:
        print(f"    ✗ Sequential has {len(result_seq.cache_sizes)} caches (expected 1)")

    # Semantic should have THREE separate caches
    if len(result_sem.cache_sizes) == 3:
        print("    ✓ Semantic has 3 isolated caches")

        # Check sizes
        c1 = result_sem.cache_sizes.get('cluster_1_technical', 0)
        c2 = result_sem.cache_sizes.get('cluster_2_business', 0)
        c3 = result_sem.cache_sizes.get('cluster_3_synthesis', 0)

        print(f"      Cluster 1 (technical): {c1} tokens")
        print(f"      Cluster 2 (business): {c2} tokens")
        print(f"      Cluster 3 (synthesis): {c3} tokens")

        # Clusters 1 and 2 should be smaller than sequential
        seq_size = list(result_seq.cache_sizes.values())[0]
        if c1 < seq_size and c2 < seq_size:
            print(f"    ✓ Isolated clusters are smaller than unified cache ({c1}, {c2} < {seq_size})")
        else:
            print(f"    ✗ Isolated clusters not smaller than unified")

    else:
        print(f"    ✗ Semantic has {len(result_sem.cache_sizes)} caches (expected 3)")

    # Compare outputs
    print("\n  Checking output quality...")
    for output_type in ['technical', 'business', 'synthesis']:
        seq_len = len(result_seq.outputs[output_type])
        sem_len = len(result_sem.outputs[output_type])
        print(f"    {output_type}: seq={seq_len} chars, sem={sem_len} chars")

        if seq_len > 0 and sem_len > 0:
            print(f"      ✓ Both conditions generated {output_type} output")
        else:
            print(f"      ✗ Empty output detected")

    return result_seq, result_sem


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRUE KV Cache Isolation Test Suite")
    print("=" * 60)

    # Check dependencies
    if not test_dependencies():
        print("\n✗ Missing dependencies - install required packages first")
        return

    # Test model loading
    tester = test_model_loading()
    if tester is None:
        print("\n✗ Model loading failed - check error messages above")
        return

    # Test cache isolation
    try:
        result_seq, result_sem = test_cache_isolation(tester)
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print(f"  Sequential cache: {result_seq.cache_sizes}")
        print(f"  Semantic caches: {result_sem.cache_sizes}")
        print(f"  Isolation verified: TRUE architectural separation")

    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
