#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""
Perplexity benchmark: FP16 vs actual Q4 KV cache quality evaluation.

Measures perplexity on WikiText-2 test set to quantify Q4 KV cache
quality loss. Uses actual QuantizedKVCache objects during model forward
passes — K/V tensors are quantized to 4-bit at every attention layer,
stored in the cache, and dequantized for attention computation. This is
the same code path used in production inference.

Compares:
  - FP16 KV cache (KVCache from mlx-lm, no quantization)
  - Q4 KV cache (QuantizedKVCache, group size 64, 4-bit)

Both paths create fresh cache objects per window and pass them to the
model's forward method. The only difference is quantization of K/V.

Usage:
    python benchmarks/perplexity_benchmark.py --model gemma
    python benchmarks/perplexity_benchmark.py --model deepseek
    python benchmarks/perplexity_benchmark.py --model all

Requirements:
    - mlx, mlx-lm, transformers
    - No server running (standalone model load)
    - Run with sandbox disabled (needs Metal GPU)
"""

import argparse
import json
import math
import gc
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Model IDs
GEMMA_MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
DEEPSEEK_MODEL_ID = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

# Config — sized for 24 GB device
WINDOW_SIZE = 512    # Tokens per evaluation window
STRIDE = 256         # Non-overlapping portion per window
MAX_TOKENS = 8192    # ~8K tokens total (safe for memory)
GROUP_SIZE = 64      # Q4 quantization group size
BITS = 4             # Quantization bits


def load_corpus():
    """Load evaluation corpus. Try local first, then WikiText-2."""
    corpus_path = Path(__file__).parent / "data" / "prefill_corpus.txt"
    if corpus_path.exists():
        text = corpus_path.read_text()
        print(f"Loaded local corpus: {len(text)} chars")
        return text

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join([line for line in dataset["text"] if line.strip()])
        print(f"Loaded WikiText-2: {len(text)} chars")
        return text
    except Exception as e:
        raise RuntimeError(f"No corpus available: {e}")


def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer via mlx-lm."""
    from mlx_lm import load
    print(f"Loading model: {model_id}")
    t0 = time.time()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    # DeepSeek model lacks make_cache(); fall back to counting model.layers
    if hasattr(model, "make_cache"):
        n_layers = len(model.make_cache())
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        n_layers = len(model.layers)
    else:
        raise RuntimeError("Cannot determine number of layers")
    print(f"Model loaded in {time.time() - t0:.1f}s ({n_layers} layers)")
    return model, tokenizer, n_layers


def evaluate_window_fp16(model, input_ids, n_layers):
    """Forward pass with FP16 KV cache (baseline).

    Creates standard KVCache objects — K/V stored in full precision.
    This is the reference for measuring Q4 degradation.
    """
    from mlx_lm.models.cache import KVCache
    caches = [KVCache() for _ in range(n_layers)]
    logits = model(input_ids, cache=caches)
    mx.eval(logits)
    return logits


def evaluate_window_q4(model, input_ids, n_layers, group_size=64, bits=4):
    """Forward pass with actual Q4 KV cache.

    Creates QuantizedKVCache objects — K/V tensors are quantized to 4-bit
    at each attention layer via cache.update_and_fetch(), stored as
    (uint32 packed data, float16 scales, float16 biases), and dequantized
    on-the-fly during attention computation.

    This is the exact code path used in production. The quantization error
    propagates through all subsequent attention layers, matching real
    inference behavior.
    """
    from mlx_lm.models.cache import QuantizedKVCache
    caches = [QuantizedKVCache(group_size=group_size, bits=bits) for _ in range(n_layers)]
    logits = model(input_ids, cache=caches)
    mx.eval(logits)
    return logits


def compute_ppl_from_logits(logits, tokens, count_start, count_end):
    """Compute log probability sum for a token range within the window."""
    log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
    target_ids = mx.array(tokens[1:]).reshape(1, -1)
    gathered = mx.take_along_axis(
        log_probs,
        mx.expand_dims(target_ids, axis=-1),
        axis=-1
    ).squeeze(-1)
    mx.eval(gathered)

    window_lp = gathered[0, count_start:count_end]
    mx.eval(window_lp)
    return window_lp.sum().item(), count_end - count_start


def run_perplexity(model, n_layers, tokens, mode="fp16"):
    """Run perplexity evaluation over sliding windows."""
    n_tokens = min(len(tokens), MAX_TOKENS)
    total_log_prob = 0.0
    total_counted = 0
    n_windows = 0

    print(f"  Evaluating {n_tokens} tokens, window={WINDOW_SIZE}, stride={STRIDE}, mode={mode}")
    t0 = time.time()

    for i in range(0, n_tokens - WINDOW_SIZE, STRIDE):
        window = tokens[i:i + WINDOW_SIZE]
        input_ids = mx.array(window).reshape(1, -1)

        if mode == "fp16":
            logits = evaluate_window_fp16(model, input_ids, n_layers)
        else:
            logits = evaluate_window_q4(model, input_ids, n_layers, GROUP_SIZE, BITS)

        # Count only non-overlapping tokens (except first window)
        if i == 0:
            cs, ce = 0, WINDOW_SIZE - 1
        else:
            cs, ce = WINDOW_SIZE - STRIDE - 1, WINDOW_SIZE - 1

        lp, n = compute_ppl_from_logits(logits, window, cs, ce)
        total_log_prob += lp
        total_counted += n
        n_windows += 1

        if n_windows % 5 == 0:
            ppl_so_far = math.exp(-total_log_prob / max(total_counted, 1))
            elapsed = time.time() - t0
            print(f"    Window {n_windows}, pos {i}/{n_tokens}, "
                  f"PPL={ppl_so_far:.4f}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    ppl = math.exp(-total_log_prob / total_counted) if total_counted > 0 else float("inf")
    print(f"  {mode.upper()} PPL = {ppl:.4f} ({total_counted} tokens, {n_windows} windows, {elapsed:.1f}s)")
    return ppl, total_counted, elapsed


def run_benchmark(model_id: str, model_name: str):
    """Run full FP16 vs Q4 comparison for one model."""
    print(f"\n{'='*60}")
    print(f"Perplexity Benchmark: {model_name}")
    print(f"{'='*60}")

    # Load corpus and tokenize
    text = load_corpus()
    model, tokenizer, n_layers = load_model_and_tokenizer(model_id)
    tokens = tokenizer.encode(text)
    print(f"Tokenized: {len(tokens)} tokens, evaluating first {min(len(tokens), MAX_TOKENS)}")

    # FP16 baseline
    print(f"\n--- FP16 KV Cache (baseline) ---")
    fp16_ppl, fp16_n, fp16_t = run_perplexity(model, n_layers, tokens, mode="fp16")

    # Q4 evaluation — actual QuantizedKVCache
    print(f"\n--- Q4 KV Cache (actual, group={GROUP_SIZE}, bits={BITS}) ---")
    q4_ppl, q4_n, q4_t = run_perplexity(model, n_layers, tokens, mode="q4")

    # Results
    delta = q4_ppl - fp16_ppl
    pct = (delta / fp16_ppl) * 100 if fp16_ppl > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  FP16 PPL:    {fp16_ppl:.4f}")
    print(f"  Q4 PPL:      {q4_ppl:.4f}")
    print(f"  Delta:       {delta:+.4f} ({pct:+.2f}%)")
    print(f"  Tokens:      {fp16_n}")
    print(f"  Method:      Actual QuantizedKVCache (not logit proxy)")

    results = {
        "model_id": model_id,
        "model_name": model_name,
        "method": "actual_q4_kv_cache",
        "corpus": "local_prefill_corpus",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_tokens": MAX_TOKENS,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "fp16_ppl": round(fp16_ppl, 4),
        "q4_ppl": round(q4_ppl, 4),
        "delta_ppl": round(delta, 4),
        "delta_pct": round(pct, 4),
        "tokens_evaluated": fp16_n,
        "fp16_time_s": round(fp16_t, 1),
        "q4_time_s": round(q4_t, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    slug = model_name.lower().replace(" ", "_").replace("-", "_")
    out_path = results_dir / f"perplexity_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    # Unload model
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    time.sleep(2)

    return results


def main():
    global MAX_TOKENS
    parser = argparse.ArgumentParser(description="FP16 vs Q4 KV cache perplexity benchmark")
    parser.add_argument("--model", choices=["gemma", "deepseek", "all"], default="gemma",
                        help="Which model to benchmark (default: gemma)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max tokens (default: 8192)")
    args = parser.parse_args()

    if args.max_tokens:
        MAX_TOKENS = args.max_tokens

    all_results = []

    if args.model in ("gemma", "all"):
        r = run_benchmark(GEMMA_MODEL_ID, "Gemma 3 12B")
        all_results.append(r)

    if args.model in ("deepseek", "all"):
        r = run_benchmark(DEEPSEEK_MODEL_ID, "DeepSeek V2 Lite 16B")
        all_results.append(r)

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'FP16':>8} {'Q4':>8} {'Delta':>8} Method")
        print("-" * 65)
        for r in all_results:
            print(f"{r['model_name']:<25} {r['fp16_ppl']:>8.4f} {r['q4_ppl']:>8.4f} "
                  f"{r['delta_ppl']:>+8.4f} {r['method']}")


if __name__ == "__main__":
    main()
