#!/usr/bin/env python3
"""Comprehensive parameter sweep benchmark via OpenAI API.

Tests all combinations of:
- Batch sizes: 1, 2, 3
- Context lengths: 2K, 5K, 10K, 25K, 50K tokens
- Output lengths: 100, 500, 1000, 5000 tokens
- Cache states: cold, warm, hot
- Concurrent requests (for batch >= 2)

Preallocates 64K token KV cache via warmup before benchmarking.

Usage:
    # Run full sweep (our server, manages lifecycle)
    python benchmarks/comprehensive_sweep.py

    # Run against LM Studio (external, cold only)
    python benchmarks/comprehensive_sweep.py --external --base-url http://127.0.0.1:1234 \
        --model deepseek-coder-v2-lite-instruct-mlx

    # Single batch size
    python benchmarks/comprehensive_sweep.py --batch-sizes 2

    # Quick subset
    python benchmarks/comprehensive_sweep.py --contexts 2000 10000 --outputs 100 500
"""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Reuse clients and helpers from openai_benchmark
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from openai_benchmark import (
    OpenAIStreamingClient,
    OpenAIRequestClient,
    OpenAIPromptFactory,
    ScenarioResult,
    ServerManager,
    compute_stats,
    OPENAI_BENCH_ENV,
    PORT,
    PADDING_TEXT,
)

RESULTS_DIR = Path(__file__).parent / "results"

# Default parameter grid
DEFAULT_CONTEXTS = [50000, 25000, 10000, 5000, 2000]
DEFAULT_OUTPUTS = [5000, 1000, 500, 100]
DEFAULT_BATCH_SIZES = [1, 2, 3]

PREALLOC_TOKENS = 64000


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Preallocation: send a 64K token request to force GPU memory allocation
# ---------------------------------------------------------------------------

async def preallocate_cache(
    base_url: str,
    model: str,
    prealloc_tokens: int = PREALLOC_TOKENS,
) -> None:
    """Send a large warmup request to preallocate KV cache on GPU.

    First sends a small warmup to ensure the model is loaded and ready,
    then sends a large request to force GPU memory allocation.
    """
    factory = OpenAIPromptFactory(model=model)

    # Step 1: small warmup to ensure model is loaded
    print("\n  [PREALLOC] Step 1: Model warmup (small request)...")
    small_body = factory.build_request(100, max_tokens=1)
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(
                f"{base_url}/v1/chat/completions", json=small_body,
            )
            if r.status_code == 200:
                print("  [PREALLOC] Model loaded OK")
            else:
                print(f"  [PREALLOC] Warmup warning: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [PREALLOC] Warmup failed: {e}")
            return

    await asyncio.sleep(1)

    # Step 2: large request to preallocate KV cache
    print(f"  [PREALLOC] Step 2: Sending {prealloc_tokens} token request to preallocate cache...")
    body = factory.build_request(prealloc_tokens, max_tokens=1)
    headers = {"X-Session-ID": "prealloc_warmup"}

    async with httpx.AsyncClient(timeout=600.0) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            t1 = time.perf_counter()
            if resp.status_code == 200:
                print(f"  [PREALLOC] Done in {(t1-t0)*1000:.0f}ms ({prealloc_tokens} tokens prefilled)")
            else:
                print(f"  [PREALLOC] Warning: status {resp.status_code} — {resp.text[:200]}")
        except Exception as e:
            print(f"  [PREALLOC] Warning: {e}")

    # Cleanup the prealloc agent and free GPU memory
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await client.delete(f"{base_url}/v1/agents/oai_prealloc_warmup")
        except Exception:
            pass

    # Longer delay for GPU memory to be reclaimed after 64K cache cleanup
    await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Individual test runners
# ---------------------------------------------------------------------------

async def run_cold(
    base_url: str,
    model: str,
    ctx: int,
    out: int,
    run_id: str,
) -> dict[str, Any]:
    """Cold request: unique session, no prior cache."""
    factory = OpenAIPromptFactory(model=model)
    client = OpenAIStreamingClient(base_url)
    body = factory.build_request(ctx, out)

    sid = f"cold_{ctx}_{out}_{run_id}"
    try:
        r = await client.send_and_measure(body, session_id=sid)
        result = asdict(r)
        result["cache_state"] = "cold"
    except Exception as e:
        result = {"error": str(e), "cache_state": "cold"}

    # Cleanup
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            await c.delete(f"{base_url}/v1/agents/oai_{sid}")
        except Exception:
            pass

    return result


async def run_warm(
    base_url: str,
    model: str,
    ctx: int,
    out: int,
    run_id: str,
) -> dict[str, Any]:
    """Warm request: prime cache, then measure second identical request."""
    factory = OpenAIPromptFactory(model=model)
    body = factory.build_request(ctx, out)

    sid = f"warm_{ctx}_{out}_{run_id}"

    # Prime: send first request (cold) to build cache
    prime_client = OpenAIRequestClient(base_url)
    try:
        prime_r = await prime_client.send_and_measure(body, session_id=sid)
        if prime_r.error:
            return {"error": f"prime failed: {prime_r.error}", "cache_state": "warm"}
    except Exception as e:
        return {"error": f"prime failed: {e}", "cache_state": "warm"}
    finally:
        await prime_client.close()

    await asyncio.sleep(0.3)

    # Measure: send same request again (should hit warm cache)
    measure_client = OpenAIStreamingClient(base_url)
    try:
        r = await measure_client.send_and_measure(body, session_id=sid)
        result = asdict(r)
        result["cache_state"] = "warm"
    except Exception as e:
        result = {"error": str(e), "cache_state": "warm"}

    # Cleanup
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            await c.delete(f"{base_url}/v1/agents/oai_{sid}")
        except Exception:
            pass

    return result


async def run_hot(
    base_url: str,
    model: str,
    ctx: int,
    out: int,
    run_id: str,
) -> dict[str, Any]:
    """Hot request: 3-turn conversation, measure 3rd turn.

    Each turn extends the context (multi-turn), exercising the EXTEND cache path.
    """
    factory = OpenAIPromptFactory(model=model)
    sid = f"hot_{ctx}_{out}_{run_id}"

    # Build initial prompt
    messages = factory.build_messages(ctx)

    # Turn 1 (cold prime)
    req_client = OpenAIRequestClient(base_url)
    body: dict[str, Any] = {
        "model": factory._model,
        "messages": messages,
        "max_tokens": min(out, 100),  # shorter output for priming turns
        "temperature": 0.0,
        "stream": False,
    }

    try:
        r1 = await req_client.send_and_measure(body, session_id=sid)
        if r1.error:
            return {"error": f"turn1 failed: {r1.error}", "cache_state": "hot"}

        # Turn 2 (extend)
        assistant_text = r1.raw_output or "Understood."
        messages = messages + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": "Continue explaining in more detail."},
        ]
        body["messages"] = messages
        r2 = await req_client.send_and_measure(body, session_id=sid)
        if r2.error:
            return {"error": f"turn2 failed: {r2.error}", "cache_state": "hot"}

        # Turn 3 (hot — measured)
        assistant_text2 = r2.raw_output or "Understood."
        messages = messages + [
            {"role": "assistant", "content": assistant_text2},
            {"role": "user", "content": "Now provide a comprehensive summary."},
        ]
    except Exception as e:
        return {"error": f"priming failed: {e}", "cache_state": "hot"}
    finally:
        await req_client.close()

    await asyncio.sleep(0.3)

    # Measure turn 3 with streaming for TTFT
    measure_client = OpenAIStreamingClient(base_url)
    body_stream: dict[str, Any] = {
        "model": factory._model,
        "messages": messages,
        "max_tokens": out,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        r = await measure_client.send_and_measure(body_stream, session_id=sid)
        result = asdict(r)
        result["cache_state"] = "hot"
    except Exception as e:
        result = {"error": str(e), "cache_state": "hot"}

    # Cleanup
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            await c.delete(f"{base_url}/v1/agents/oai_{sid}")
        except Exception:
            pass

    return result


async def run_concurrent(
    base_url: str,
    model: str,
    ctx: int,
    out: int,
    n_concurrent: int,
    run_id: str,
) -> dict[str, Any]:
    """N concurrent cold requests — measures system throughput."""
    factory = OpenAIPromptFactory(model=model)
    body = factory.build_request(ctx, out)

    clients = [OpenAIRequestClient(base_url) for _ in range(n_concurrent)]
    sids = [f"conc_{ctx}_{out}_{run_id}_c{j}" for j in range(n_concurrent)]

    t_wall_start = time.perf_counter()
    try:
        results = await asyncio.gather(*(
            clients[j].send_and_measure(body, session_id=sids[j])
            for j in range(n_concurrent)
        ))
    except Exception as e:
        for c in clients:
            await c.close()
        return {"error": str(e), "cache_state": "concurrent"}
    t_wall_end = time.perf_counter()

    for c in clients:
        await c.close()

    wall_ms = (t_wall_end - t_wall_start) * 1000
    total_out = sum(r.output_tokens for r in results)
    wall_s = wall_ms / 1000
    system_tps = total_out / wall_s if wall_s > 0 else 0

    # Collect per-request errors
    req_errors = [r.error for r in results if r.error]

    # Cleanup
    async with httpx.AsyncClient(timeout=10.0) as c:
        for sid in sids:
            try:
                await c.delete(f"{base_url}/v1/agents/oai_{sid}")
            except Exception:
                pass

    result: dict[str, Any] = {
        "cache_state": "concurrent",
        "n_concurrent": n_concurrent,
        "wall_ms": wall_ms,
        "system_tps": system_tps,
        "total_output_tokens": total_out,
        "per_request": [asdict(r) for r in results],
    }
    if req_errors and total_out == 0:
        result["error"] = req_errors[0]
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def run_sweep(
    base_url: str,
    model: str,
    batch_sizes: list[int],
    contexts: list[int],
    outputs: list[int],
    external: bool = False,
    output_path: str | None = None,
    label: str = "semantic",
    concurrent_only: bool = False,
    prealloc_first_only: bool = False,
) -> None:
    all_results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": label,
            "base_url": base_url,
            "model": model,
            "machine": {
                "os": platform.system(),
                "os_version": platform.release(),
                "chip": platform.machine(),
            },
            "git_sha": _git_sha(),
            "contexts": contexts,
            "outputs": outputs,
            "batch_sizes": batch_sizes,
            "prealloc_tokens": PREALLOC_TOKENS,
        },
        "sweeps": {},
    }

    out_path = output_path or str(
        RESULTS_DIR / f"sweep_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # In concurrent-only mode, load existing results to merge with
    if concurrent_only and output_path and Path(output_path).exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results from {output_path}")

    def save():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    cache_states = ["cold", "warm", "hot"]
    if external:
        # External servers (LM Studio) don't support session-based caching
        cache_states = ["cold"]

    prealloc_done = False
    for batch_size in batch_sizes:
        batch_key = f"batch_{batch_size}"
        print(f"\n{'='*80}")
        print(f"  BATCH SIZE = {batch_size}")
        print(f"{'='*80}")

        server = None
        if not external:
            server = ServerManager(port=PORT)
            env = {**OPENAI_BENCH_ENV}
            env["SEMANTIC_MLX_MAX_BATCH_SIZE"] = str(batch_size)
            # Scheduler MUST be enabled for all batch sizes — OpenAI streaming
            # depends on it for per-token SSE deltas
            env["SEMANTIC_MLX_SCHEDULER_ENABLED"] = "true"
            print(f"\n[SERVER] Starting with batch_size={batch_size}...")
            server.start(env)

        try:
            # Preallocate 64K cache — skip when:
            #  - concurrent-only mode (restart fresh to avoid GPU OOM)
            #  - prealloc_first_only and already done once
            skip_prealloc = concurrent_only or (prealloc_first_only and prealloc_done)
            if not skip_prealloc:
                await preallocate_cache(base_url, model, PREALLOC_TOKENS)
                prealloc_done = True
            else:
                # Small warmup to load model weights only
                print("\n  [WARMUP] Small model warmup (skipping 64K prealloc)...")
                factory_wu = OpenAIPromptFactory(model=model)
                async with httpx.AsyncClient(timeout=120.0) as c:
                    small = factory_wu.build_request(100, max_tokens=5)
                    small["stream"] = False
                    try:
                        await c.post(
                            f"{base_url}/v1/chat/completions",
                            json=small,
                            headers={"X-Session-ID": "warmup_only"},
                        )
                    except Exception:
                        pass
                    try:
                        await c.delete(f"{base_url}/v1/agents/oai_warmup_only")
                    except Exception:
                        pass
                print("  [WARMUP] Done")

            batch_results: dict[str, Any] = (
                all_results.get("sweeps", {}).get(batch_key, {})
                if concurrent_only else {}
            )

            # --- Single-request tests: cold, warm, hot ---
            if concurrent_only:
                print("\n  (Skipping single-request tests — concurrent-only mode)")
            total = len(contexts) * len(outputs) * len(cache_states)
            done = 0

            for ctx in contexts if not concurrent_only else []:
                for out in outputs:
                    for state in cache_states:
                        done += 1
                        key = f"{state}_{ctx}in_{out}out"
                        print(f"\n  [{done}/{total}] {key}", end=" ", flush=True)

                        t0 = time.perf_counter()
                        if state == "cold":
                            result = await run_cold(base_url, model, ctx, out, f"b{batch_size}")
                        elif state == "warm":
                            result = await run_warm(base_url, model, ctx, out, f"b{batch_size}")
                        elif state == "hot":
                            result = await run_hot(base_url, model, ctx, out, f"b{batch_size}")
                        else:
                            continue
                        elapsed = time.perf_counter() - t0

                        err = result.get("error")
                        if err:
                            print(f"ERROR: {str(err)[:80]} ({elapsed:.1f}s)")
                        else:
                            ttft = result.get("ttft_ms", 0)
                            e2e = result.get("e2e_ms", 0)
                            tps = result.get("decode_tps", 0)
                            out_tok = result.get("output_tokens", 0)
                            print(
                                f"TTFT={ttft:.0f}ms E2E={e2e:.0f}ms "
                                f"TPS={tps:.1f} out={out_tok} ({elapsed:.1f}s)"
                            )

                        result["context_tokens"] = ctx
                        result["max_output_tokens"] = out
                        batch_results[key] = result
                        save()  # Save after each test

            # --- Concurrent tests (batch >= 2) ---
            if batch_size >= 2:
                print(f"\n  --- Concurrent tests (n={batch_size}) ---")

                # Restart server WITHOUT 64K prealloc for concurrent tests.
                # The 64K prealloc fills GPU memory, leaving no room for
                # multiple concurrent KV caches. A small warmup is sufficient.
                if server is not None:
                    server.stop()
                    print("  [SERVER] Restarting for concurrent tests (no 64K prealloc)...")
                    server.start(env)
                    # Small warmup only — load model without filling GPU memory
                    factory_warmup = OpenAIPromptFactory(model=model)
                    async with httpx.AsyncClient(timeout=120.0) as c:
                        small = factory_warmup.build_request(100, max_tokens=5)
                        small["stream"] = False
                        try:
                            await c.post(
                                f"{base_url}/v1/chat/completions",
                                json=small,
                                headers={"X-Session-ID": "conc_warmup"},
                            )
                        except Exception:
                            pass
                        try:
                            await c.delete(f"{base_url}/v1/agents/oai_conc_warmup")
                        except Exception:
                            pass
                    await asyncio.sleep(3)

                # Shortest context first so we get data before long contexts
                # potentially crash the server
                conc_contexts = sorted(contexts)
                for ctx in conc_contexts:
                    for out in [100, 500]:  # Representative output lengths
                        key = f"concurrent_{batch_size}x_{ctx}in_{out}out"

                        # Check server health before each test
                        if server is not None and not server.is_alive():
                            print("\n  [SERVER] Crashed — restarting...")
                            server.stop()
                            server.start(env)
                            await asyncio.sleep(3)

                        print(f"\n  {key}", end=" ", flush=True)

                        t0 = time.perf_counter()
                        result = await run_concurrent(
                            base_url, model, ctx, out, batch_size, f"b{batch_size}"
                        )
                        elapsed = time.perf_counter() - t0

                        err = result.get("error")
                        if err:
                            print(f"ERROR: {str(err)[:80]} ({elapsed:.1f}s)")
                        else:
                            wall = result.get("wall_ms", 0)
                            sys_tps = result.get("system_tps", 0)
                            total_out = result.get("total_output_tokens", 0)
                            print(
                                f"wall={wall:.0f}ms sysTPS={sys_tps:.1f} "
                                f"out={total_out} ({elapsed:.1f}s)"
                            )

                        result["context_tokens"] = ctx
                        result["max_output_tokens"] = out
                        batch_results[key] = result
                        save()

            all_results["sweeps"][batch_key] = batch_results
            save()

        finally:
            if server is not None:
                server.stop()
                print("\n[SERVER] Stopped.")

    # --- Final summary ---
    save()
    print(f"\n{'='*80}")
    print(f"  Sweep complete. Results saved to: {out_path}")
    print(f"{'='*80}")
    _print_summary(all_results)


def _print_summary(results: dict[str, Any]) -> None:
    """Print a concise summary table."""
    print(f"\n{'='*100}")
    print(f"  COMPREHENSIVE SWEEP SUMMARY")
    print(f"{'='*100}")

    for batch_key, batch_data in sorted(results.get("sweeps", {}).items()):
        print(f"\n  --- {batch_key} ---")
        print(f"  {'Scenario':<35} {'TTFT':>8} {'E2E':>9} {'TPS':>8} {'Out':>6}")
        print(f"  {'─'*35} {'─'*8} {'─'*9} {'─'*8} {'─'*6}")

        for key, data in sorted(batch_data.items()):
            if data.get("error"):
                print(f"  {key:<35} {'ERROR':>8}")
                continue

            if "wall_ms" in data:
                # Concurrent result
                wall = data.get("wall_ms", 0)
                sys_tps = data.get("system_tps", 0)
                total_out = data.get("total_output_tokens", 0)
                print(f"  {key:<35} {'':>8} {wall:>8.0f}ms {sys_tps:>7.1f} {total_out:>6}")
            else:
                ttft = data.get("ttft_ms", 0)
                e2e = data.get("e2e_ms", 0)
                tps = data.get("decode_tps", 0)
                out_tok = data.get("output_tokens", 0)
                print(f"  {key:<35} {ttft:>7.0f}ms {e2e:>8.0f}ms {tps:>7.1f} {out_tok:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive OpenAI benchmark sweep")
    parser.add_argument("--base-url", type=str, default=f"http://127.0.0.1:{PORT}")
    parser.add_argument("--external", action="store_true", help="External server (no lifecycle)")
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to test (default: 1 2 3)"
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS,
        help="Context token counts (default: 2000 5000 10000 25000 50000)"
    )
    parser.add_argument(
        "--outputs", nargs="+", type=int, default=DEFAULT_OUTPUTS,
        help="Output token counts (default: 100 500 1000 5000)"
    )
    parser.add_argument(
        "--concurrent-only", action="store_true",
        help="Only run concurrent tests (skip single-request tests)"
    )
    parser.add_argument(
        "--prealloc-first-only", action="store_true",
        help="64K prealloc only for first batch size, small warmup for rest"
    )

    args = parser.parse_args()

    label = args.label
    if label is None:
        label = "lmstudio" if args.external and "1234" in args.base_url else "semantic"

    asyncio.run(run_sweep(
        base_url=args.base_url,
        model=args.model,
        batch_sizes=args.batch_sizes,
        contexts=args.contexts,
        outputs=args.outputs,
        external=args.external,
        output_path=args.output,
        label=label,
        concurrent_only=args.concurrent_only,
        prealloc_first_only=args.prealloc_first_only,
    ))


if __name__ == "__main__":
    main()
