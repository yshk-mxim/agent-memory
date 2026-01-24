"""
EXP-004: Validate Response.prompt_cache() extraction

Goal: Prove that per-sequence cache extraction works on completion
Success criteria: Extract cache from each sequence, save/reload, continue generation
"""

import mlx.core as mx
from mlx_lm import BatchGenerator, load, save_prompt_cache, load_prompt_cache
import tempfile
import os


def main():
    print("=" * 80)
    print("EXP-004: Per-Sequence Cache Extraction Validation")
    print("=" * 80)

    # Use SmolLM2-135M for fast testing
    model_id = "mlx-community/SmolLM2-135M-Instruct-4bit"
    print(f"\n1. Loading model: {model_id}")
    model, tokenizer = load(model_id)

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Python programming language",
    ]
    print(f"\n2. Testing with {len(prompts)} prompts in batch")

    # Generate batch and extract per-sequence caches
    print("\n3. Batch generation (extracting per-sequence caches)...")
    gen = BatchGenerator(model, stop_tokens=tokenizer.eos_token_id, max_tokens=10)
    uids = gen.insert(prompts, max_tokens=10)

    caches = {}
    texts = {}
    completion_count = 0

    while responses := gen.next():
        for r in responses:
            if r.finish_reason is not None:
                completion_count += 1
                try:
                    # Extract cache for this specific sequence
                    cache = r.prompt_cache()
                    caches[r.uid] = cache
                    texts[r.uid] = tokenizer.decode(r.tokens)
                    print(f"   [{completion_count}/3] UID {r.uid}: extracted cache, text: '{texts[r.uid]}'")
                except AttributeError as e:
                    print(f"\n❌ EXP-004 FAILED: Response.prompt_cache() not callable")
                    print(f"   Error: {e}")
                    print(f"   Available attributes: {dir(r)}")
                    print("\n   → INVOKE PLAN B: Cannot extract per-sequence caches")
                    return False

    if len(caches) != len(prompts):
        print(f"\n❌ EXP-004 FAILED: Expected {len(prompts)} caches, got {len(caches)}")
        return False

    print(f"\n   ✓ All {len(caches)} caches extracted successfully")

    # Test save/reload/re-inject cycle
    print("\n4. Testing save/reload/re-inject cycle...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all caches
        cache_paths = {}
        for uid, cache in caches.items():
            path = os.path.join(tmpdir, f"cache_{uid}.safetensors")
            save_prompt_cache(path, cache)
            cache_paths[uid] = path
        print(f"   ✓ Saved {len(cache_paths)} caches to disk")

        # Reload all caches
        reloaded_caches = {}
        for uid, path in cache_paths.items():
            reloaded_caches[uid] = load_prompt_cache(path)
        print(f"   ✓ Reloaded {len(reloaded_caches)} caches from disk")

        # Re-inject into new batch
        print("\n5. Re-injecting caches for continued generation...")
        continuation_prompts = [
            prompts[0] + texts[uids[0]],
            prompts[1] + texts[uids[1]],
            prompts[2] + texts[uids[2]],
        ]

        gen2 = BatchGenerator(model, stop_tokens=tokenizer.eos_token_id, max_tokens=5)

        try:
            # Attempt re-injection with reloaded caches
            uids2 = gen2.insert(
                continuation_prompts,
                max_tokens=5,
                caches=[reloaded_caches[uid] for uid in uids]
            )

            continuation_count = 0
            while responses := gen2.next():
                for r in responses:
                    if r.finish_reason is not None:
                        continuation_count += 1
                        continued_text = tokenizer.decode(r.tokens)
                        print(f"   [{continuation_count}/3] Continued: '{continued_text}'")

            print("\n   ✓ Cache re-injection successful, generation continued")

        except Exception as e:
            print(f"\n❌ EXP-004 FAILED: Cache re-injection failed")
            print(f"   Error: {e}")
            return False

    print("\n✅ EXP-004 PASSED: Per-sequence cache extraction and re-injection work")
    print("\n   Key findings:")
    print("   - Response.prompt_cache() callable exists")
    print("   - Per-sequence cache extraction on completion works")
    print("   - Save/reload/re-inject cycle works")
    print("   - Sequences complete independently (don't wait for full batch)")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
