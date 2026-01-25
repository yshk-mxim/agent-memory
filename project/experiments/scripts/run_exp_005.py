#!/usr/bin/env python3
"""EXP-005: BlockPoolBatchEngine Correctness Validation

Validates that BlockPoolBatchEngine produces byte-identical output to
reference mlx_lm.generate() implementation.

Sprint 2, Day 8
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


def setup_test_environment():
    """Load model and create test infrastructure.

    Returns:
        tuple: (model, tokenizer, pool, spec, engine)
    """
    # TODO: Day 8 implementation
    # from mlx_lm import load
    # from semantic.domain.services import BlockPool
    # from semantic.domain.value_objects import ModelCacheSpec
    # from semantic.application.batch_engine import BlockPoolBatchEngine

    # model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    # spec = ModelCacheSpec.from_model(model)
    # pool = BlockPool(spec, total_blocks=100)
    # engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)

    # return model, tokenizer, pool, spec, engine
    raise NotImplementedError("Setup pending BlockPoolBatchEngine implementation")


def generate_reference(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Generate reference output using mlx_lm.generate().

    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text (ground truth)
    """
    # TODO: Day 8 implementation
    # from mlx_lm import generate

    # result = generate(
    #     model,
    #     tokenizer,
    #     prompt=prompt,
    #     max_tokens=max_tokens,
    #     temp=0.0,  # Greedy (deterministic)
    #     verbose=False,
    # )
    # return result
    raise NotImplementedError("Reference generation pending mlx_lm integration")


def generate_test(engine, prompt: str, max_tokens: int = 100) -> tuple[str, int]:
    """Generate output using BlockPoolBatchEngine.

    Args:
        engine: BlockPoolBatchEngine instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        tuple: (generated_text, token_count)
    """
    # TODO: Day 8 implementation
    # uid = engine.submit(
    #     agent_id="test_agent",
    #     prompt=prompt,
    #     cache=None,  # No cache (fresh generation)
    #     max_tokens=max_tokens,
    # )

    # # Poll for completion
    # for completion in engine.step():
    #     if completion.uid == uid:
    #         return completion.text, completion.token_count

    # raise RuntimeError(f"Generation {uid} did not complete")
    raise NotImplementedError("Test generation pending BlockPoolBatchEngine.submit()")


def validate_output(actual: str, expected: str, prompt_label: str) -> bool:
    """Validate output matches reference byte-for-byte.

    Args:
        actual: Test output
        expected: Reference output
        prompt_label: Test case label

    Returns:
        True if match, False otherwise
    """
    if actual != expected:
        print(f"❌ {prompt_label}: Output mismatch")
        print(f"   Expected: '{expected[:50]}...'")
        print(f"   Actual:   '{actual[:50]}...'")

        # Show divergence point
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e != a:
                print(f"   Divergence at position {i}: expected='{e}', actual='{a}'")
                break

        return False

    print(f"✅ {prompt_label}: Output matches exactly")
    return True


def validate_token_count(
    token_count: int, prompt: str, tokenizer, max_tokens: int
) -> bool:
    """Validate token count is within expected range.

    Args:
        token_count: Total tokens (prompt + generated)
        prompt: Input prompt
        tokenizer: Tokenizer instance
        max_tokens: Maximum allowed generated tokens

    Returns:
        True if valid, False otherwise
    """
    prompt_tokens = len(tokenizer.encode(prompt))
    generated_tokens = token_count - prompt_tokens

    if generated_tokens <= 0:
        print(f"❌ Token count validation failed: No tokens generated")
        return False

    if generated_tokens > max_tokens:
        print(f"❌ Token count validation failed: {generated_tokens} > {max_tokens}")
        return False

    print(f"✅ Token count valid: {generated_tokens} tokens generated")
    return True


def run_exp_005():
    """Execute EXP-005 validation.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=" * 70)
    print("EXP-005: BLOCKPOOLBATCHENGINE CORRECTNESS VALIDATION")
    print("=" * 70)
    print()

    # Test prompts (3 lengths)
    test_cases = [
        ("short", "The quick brown fox", 100),
        ("medium", "Write a story about a robot", 100),
        ("long", "Explain quantum computing in detail", 100),
    ]

    print("🔧 Setting up test environment...")
    try:
        model, tokenizer, pool, spec, engine = setup_test_environment()
        print("✅ Test environment ready")
        print()
    except NotImplementedError as e:
        print(f"⏳ Skipping EXP-005: {e}")
        return False

    # Generate references
    print("📝 Generating reference outputs...")
    references = {}
    for label, prompt, max_tokens in test_cases:
        print(f"  Generating reference for '{label}'...")
        start = time.perf_counter()
        ref_output = generate_reference(model, tokenizer, prompt, max_tokens)
        end = time.perf_counter()
        references[label] = {
            "output": ref_output,
            "time": end - start,
        }
        print(f"    ✅ Done ({end - start:.2f}s)")
    print()

    # Save references
    ref_path = Path(__file__).parent.parent / "data" / "exp_005_references.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(references, f, indent=2)
    print(f"💾 References saved to: {ref_path}")
    print()

    # Run tests
    print("🧪 Running correctness tests...")
    results = []

    for label, prompt, max_tokens in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {label}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 70}")

        # Generate test output
        start = time.perf_counter()
        test_output, token_count = generate_test(engine, prompt, max_tokens)
        end = time.perf_counter()
        test_time = end - start

        # Validate
        ref_output = references[label]["output"]
        ref_time = references[label]["time"]

        output_match = validate_output(test_output, ref_output, label)
        token_valid = validate_token_count(token_count, prompt, tokenizer, max_tokens)

        # Performance check (secondary)
        perf_ratio = test_time / ref_time
        perf_ok = perf_ratio <= 1.2  # Within 20%

        if perf_ok:
            print(f"✅ Performance within 20% of reference ({perf_ratio:.1%})")
        else:
            print(f"⚠️  Performance slower than reference ({perf_ratio:.1%})")

        # Record results
        results.append({
            "prompt": label,
            "output_match": output_match,
            "token_valid": token_valid,
            "perf_ok": perf_ok,
            "test_time": test_time,
            "ref_time": ref_time,
            "perf_ratio": perf_ratio,
        })

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["output_match"] and r["token_valid"])
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print()

    for r in results:
        status = "✅ PASS" if (r["output_match"] and r["token_valid"]) else "❌ FAIL"
        print(f"{status} {r['prompt']}: match={r['output_match']}, "
              f"tokens={r['token_valid']}, perf={r['perf_ratio']:.1%}")

    print()

    if passed == total:
        print("🎉 EXP-005 PASSED: All outputs match reference")
        print()
        print("BlockPoolBatchEngine generates byte-identical output to mlx_lm.generate().")
        print("Cache reconstruction and extraction logic is correct.")
        return True
    else:
        print("❌ EXP-005 FAILED: Output mismatch detected")
        print()
        print("NEXT STEPS:")
        print("1. Compare token-by-token to identify divergence point")
        print("2. Validate cache reconstruction logic (blocks → KVCache)")
        print("3. Check for floating-point precision issues")
        print("4. Escalate to PM if mismatch > 5 tokens (BLOCKING)")
        return False


def main():
    """Main entry point."""
    success = run_exp_005()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
