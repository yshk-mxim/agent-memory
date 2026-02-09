#!/usr/bin/env python3
"""Targeted verification for BUG 1: batch=2 + cached + short context = empty output.

Usage:
    # Start server first:
    SEMANTIC_MLX_MAX_BATCH_SIZE=2 SEMANTIC_MLX_SCHEDULER_ENABLED=true \
        SEMANTIC_ADMIN_KEY=benchmark \
        python -m semantic.entrypoints.cli serve --port 8000

    # Then run:
    python benchmarks/verify_bug1.py [--port 8000]

Tests the exact failing configs from the Gemma benchmark:
  - batch=2, warm, streaming, 1024 tokens
  - batch=2, warm, non-streaming, 1024 tokens
  - batch=2, hot, streaming, 1024 tokens
  - batch=2, hot, non-streaming, 1024 tokens
  - batch=2, hot, non-streaming, 2048 tokens

Also runs controls that should pass:
  - batch=2, cold, streaming, 1024 tokens
  - batch=2, warm, streaming, 2048 tokens
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Reuse existing benchmark infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from openai_benchmark import OpenAIStreamingClient, OpenAIRequestClient
from colm_full_benchmark import (
    build_messages,
    clear_all_caches,
    clean_cache_files,
    check_structural,
    TEMPERATURE,
)
from streaming_benchmark import _delete_agent


CORPUS_PATH = Path(__file__).parent / "data" / "prefill_corpus.txt"
OUTPUT_TOKENS = 64
CONCURRENT_TIMEOUT = 300


async def run_pair(
    base_url: str,
    context: int,
    mode: str,
    cache_state: str,
    corpus: str,
    label: str,
) -> dict:
    """Run a single batch=2 test pair and return results."""
    await clear_all_caches(base_url)
    clean_cache_files()
    await asyncio.sleep(2)

    streaming = mode == "streaming"
    messages_a = build_messages(corpus, context, 0)
    messages_b = build_messages(corpus, context, 5000)

    body_a = {
        "model": "default",
        "messages": messages_a,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": streaming,
    }
    body_b = {
        "model": "default",
        "messages": messages_b,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": streaming,
    }

    sid_a = f"verify_a_{label}"
    sid_b = f"verify_b_{label}"

    # Prime for warm/hot
    if cache_state in ("warm", "hot"):
        prime = OpenAIRequestClient(base_url)
        nb_a = {**body_a, "stream": False}
        nb_b = {**body_b, "stream": False}
        await asyncio.wait_for(
            asyncio.gather(
                prime.send_and_measure(nb_a, session_id=sid_a),
                prime.send_and_measure(nb_b, session_id=sid_b),
            ),
            timeout=CONCURRENT_TIMEOUT,
        )
        await prime.close()

        if cache_state == "warm":
            await _delete_agent(base_url, f"oai_{sid_a}", evict_only=True)
            await _delete_agent(base_url, f"oai_{sid_b}", evict_only=True)
            await asyncio.sleep(1.5)

    # Measurement
    if streaming:
        clients = [OpenAIStreamingClient(base_url) for _ in range(2)]
    else:
        clients = [OpenAIRequestClient(base_url) for _ in range(2)]

    t_start = time.perf_counter()
    results = await asyncio.wait_for(
        asyncio.gather(
            clients[0].send_and_measure(body_a, session_id=sid_a),
            clients[1].send_and_measure(body_b, session_id=sid_b),
        ),
        timeout=CONCURRENT_TIMEOUT,
    )
    wall_ms = (time.perf_counter() - t_start) * 1000

    for c in clients:
        if hasattr(c, "close"):
            await c.close()

    total_output = sum(r.output_tokens for r in results)
    raw_a = results[0].raw_output
    raw_b = results[1].raw_output
    struct_issues = check_structural(f"{raw_a} {raw_b}", total_output)

    await _delete_agent(base_url, f"oai_{sid_a}")
    await _delete_agent(base_url, f"oai_{sid_b}")

    return {
        "label": label,
        "context": context,
        "cache_state": cache_state,
        "mode": mode,
        "wall_ms": round(wall_ms, 1),
        "total_tokens": total_output,
        "tokens_a": results[0].output_tokens,
        "tokens_b": results[1].output_tokens,
        "output_a": raw_a[:100],
        "output_b": raw_b[:100],
        "quality_ok": len(struct_issues) == 0,
        "issues": struct_issues,
    }


async def main():
    parser = argparse.ArgumentParser(description="BUG 1 verification")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    corpus = CORPUS_PATH.read_text()

    # Define test cases: (context, mode, cache_state, expect_pass)
    tests = [
        # Controls (should pass)
        (1024, "streaming", "cold", True),
        (2048, "streaming", "warm", True),
        # Failing configs from benchmark
        (1024, "streaming", "warm", False),
        (1024, "non-streaming", "warm", False),
        (1024, "streaming", "hot", False),
        (1024, "non-streaming", "hot", False),
        (2048, "non-streaming", "hot", False),
    ]

    print(f"\n{'='*70}")
    print("BUG 1 VERIFICATION: batch=2 + cached + short context")
    print(f"{'='*70}")
    print(f"Server: {base_url}")
    print(f"Output tokens: {OUTPUT_TOKENS}, Temperature: {TEMPERATURE}")
    print()

    results = []
    for context, mode, cache_state, expected_pass in tests:
        label = f"{cache_state}_{mode}_{context}"
        tag = "CONTROL" if expected_pass else "BUG"
        print(f"[{tag:>7}] {label} ... ", end="", flush=True)

        try:
            result = await run_pair(
                base_url, context, mode, cache_state, corpus, label
            )
            passed = result["quality_ok"] and result["total_tokens"] > 0
            symbol = "PASS" if passed else "FAIL"
            fixed = ""
            if not expected_pass and passed:
                fixed = " (FIXED!)"
            elif expected_pass and not passed:
                fixed = " (REGRESSION!)"
            print(
                f"{symbol}{fixed} — "
                f"tokens={result['total_tokens']} "
                f"({result['tokens_a']}+{result['tokens_b']}), "
                f"wall={result['wall_ms']:.0f}ms"
            )
            if not passed:
                print(f"         output_a: {result['output_a']!r}")
                print(f"         output_b: {result['output_b']!r}")
                print(f"         issues: {result['issues']}")
            result["passed"] = passed
            result["expected_pass"] = expected_pass
        except Exception as e:
            print(f"ERROR — {e}")
            result = {"label": label, "passed": False, "expected_pass": expected_pass, "error": str(e)}

        results.append(result)
        await asyncio.sleep(5)  # Brief cooldown between tests

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    controls = [r for r in results if r.get("expected_pass")]
    bugs = [r for r in results if not r.get("expected_pass")]

    controls_passed = sum(1 for r in controls if r.get("passed"))
    bugs_fixed = sum(1 for r in bugs if r.get("passed"))

    print(f"Controls: {controls_passed}/{len(controls)} passed")
    print(f"Bug cases: {bugs_fixed}/{len(bugs)} fixed")

    if bugs_fixed == len(bugs) and controls_passed == len(controls):
        print("\nAll bug cases FIXED, all controls pass. H1 fix confirmed!")
        return 0
    elif controls_passed < len(controls):
        print("\nREGRESSION: some controls failed!")
        return 2
    else:
        remaining = len(bugs) - bugs_fixed
        print(f"\n{remaining} bug case(s) still failing. May need additional fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
