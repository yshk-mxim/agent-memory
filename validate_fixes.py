#!/usr/bin/env python3
"""Comprehensive real-model validation for the 5 bug fixes.

Tests with actual Gemma 3 (port 8000) and DeepSeek (port 8001):
- 4K+ token inputs with 1K additions
- Hot / Warm / Cold cache paths
- Memory pressure (block pool utilization)
- Timing (TTFT, decode TPS, E2E)
- Semantic quality (coherent output, instruction following)
- Cache preservation across evict/reload
- Prisoner's dilemma full scenario

Must be run AFTER 'semantic serve --port 8000' is active.
"""

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
GEMMA_URL = "http://localhost:8000"
DEEPSEEK_URL = "http://localhost:8001"
TIMEOUT = 120.0  # seconds per request
CACHE_DIR = Path.home() / ".semantic" / "caches"

PADDING = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models process long context efficiently. "
    "Transformer architectures use attention to weigh input tokens. "
    "KV caching avoids recomputing attention for prior tokens. "
) * 20  # ~80 words * 20 = ~1600 words ≈ 2400 tokens per repeat

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class TestResult:
    name: str
    status: str
    details: str = ""
    timing_ms: float = 0.0
    tokens_generated: int = 0
    memory_mb: float = 0.0


results: list[TestResult] = []


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def log(msg: str):
    print(f"  {msg}")


def section(title: str):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


async def server_ready(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url}/health")
            return r.status_code == 200
    except Exception:
        return False


async def chat(base_url: str, messages: list[dict], session_id: str,
               max_tokens: int = 64, temperature: float = 0.0) -> dict:
    """Send a chat completion and return full response dict."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={"X-Session-ID": session_id},
        )
        r.raise_for_status()
        return r.json()


async def stream_chat(base_url: str, messages: list[dict], session_id: str,
                      max_tokens: int = 64, temperature: float = 0.0) -> dict:
    """Send streaming chat completion, measure TTFT and full response."""
    start = time.time()
    ttft = None
    full_text = ""
    token_count = 0

    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        async with c.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            headers={"X-Session-ID": session_id},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft is None:
                        ttft = (time.time() - start) * 1000
                    if content:
                        full_text += content
                        token_count += 1
                except json.JSONDecodeError:
                    continue

    e2e = (time.time() - start) * 1000
    return {
        "text": full_text,
        "ttft_ms": ttft or e2e,
        "e2e_ms": e2e,
        "token_count": token_count,
    }


def get_cache_files() -> list[tuple[str, float]]:
    """Return list of (filename, size_mb) in cache dir."""
    if not CACHE_DIR.exists():
        return []
    files = []
    for f in CACHE_DIR.glob("*.safetensors"):
        files.append((f.name, f.stat().st_size / (1024 * 1024)))
    return files


async def get_agent_info(base_url: str, agent_id: str) -> dict | None:
    """Get agent cache info from API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base_url}/v1/agents/{agent_id}")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def evict_agent(base_url: str, agent_id: str, keep_disk: bool = True):
    """Evict agent from hot tier."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        url = f"{base_url}/v1/agents/{agent_id}"
        if keep_disk:
            url += "?evict_only=true"
        await c.delete(url)


def get_process_memory_mb(port: int) -> float:
    """Get RSS memory of the server process."""
    try:
        pid_out = subprocess.run(
            ["lsof", f"-ti:{port}"], capture_output=True, text=True
        )
        pids = pid_out.stdout.strip().split("\n")
        if not pids or not pids[0]:
            return 0.0
        pid = pids[0]
        rss_out = subprocess.run(
            ["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True
        )
        return int(rss_out.stdout.strip()) / 1024  # KB -> MB
    except Exception:
        return 0.0


def make_4k_messages(extra_tokens: int = 0) -> list[dict]:
    """Build messages totaling ~4K tokens (+ optional extra)."""
    # Approx 1.3 tokens per word, PADDING is ~1600 words ≈ 2400 tokens per copy
    # Two copies ≈ 4800 tokens
    context = PADDING * 2
    if extra_tokens > 0:
        # ~0.75 words per token → each word ≈ 1.3 tokens
        extra_words = int(extra_tokens / 1.3)
        extra = " ".join(["additional"] * extra_words)
        context += "\n\n" + extra
    return [
        {"role": "system", "content": "You are a helpful assistant. Always respond concisely."},
        {"role": "user", "content": f"Context:\n\n{context}\n\nSummarize the above in 2 sentences."},
    ]


# -------------------------------------------------------------------
# Test suites
# -------------------------------------------------------------------

async def test_cold_cache_4k(base_url: str, label: str):
    """TEST 1: Cold cache with ~4K token input."""
    test_name = f"{label}: Cold cache 4K input"
    section(test_name)
    session_id = f"test_cold_4k_{label.lower().replace(' ', '_')}"

    try:
        msgs = make_4k_messages()
        start = time.time()
        resp = await stream_chat(base_url, msgs, session_id, max_tokens=128)
        elapsed = time.time() - start

        text = resp["text"]
        ttft = resp["ttft_ms"]
        tps = resp["token_count"] / (elapsed if elapsed > 0 else 1)

        log(f"TTFT:      {ttft:.0f} ms")
        log(f"E2E:       {resp['e2e_ms']:.0f} ms")
        log(f"Tokens:    {resp['token_count']}")
        log(f"Decode:    {tps:.1f} tok/s")
        log(f"Output:    {text[:120]}...")

        # Quality checks
        ok = True
        if not text.strip():
            log("FAIL: Empty output!")
            ok = False
        if len(text.strip()) < 20:
            log(f"FAIL: Output too short ({len(text)} chars)")
            ok = False
        if resp["token_count"] < 5:
            log(f"FAIL: Too few tokens ({resp['token_count']})")
            ok = False

        # Check cache was created
        agent_id = f"oai_{session_id}"
        info = await get_agent_info(base_url, agent_id)
        if info:
            log(f"Cache:     {info.get('cache_size_tokens', '?')} tokens cached")
        else:
            log("WARN: Agent info not available")

        results.append(TestResult(
            name=test_name,
            status=PASS if ok else FAIL,
            details=f"TTFT={ttft:.0f}ms, {resp['token_count']} tokens, {tps:.1f} tok/s",
            timing_ms=resp["e2e_ms"],
            tokens_generated=resp["token_count"],
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_hot_cache_continuation(base_url: str, label: str):
    """TEST 2: Hot cache — add ~1K tokens to existing 4K conversation."""
    test_name = f"{label}: Hot cache +1K continuation"
    section(test_name)
    session_id = f"test_cold_4k_{label.lower().replace(' ', '_')}"  # Reuse from test 1

    try:
        # Second turn with additional context
        msgs = make_4k_messages(extra_tokens=1000)
        # Add a follow-up that extends the conversation
        msgs.append({"role": "assistant", "content": "The context discusses natural language processing, transformer models, and KV caching."})
        msgs.append({"role": "user", "content": "Now explain what KV caching is and why it matters, in 2 sentences."})

        start = time.time()
        resp = await stream_chat(base_url, msgs, session_id, max_tokens=128)
        elapsed = time.time() - start

        text = resp["text"]
        ttft = resp["ttft_ms"]
        tps = resp["token_count"] / (elapsed if elapsed > 0 else 1)

        log(f"TTFT:      {ttft:.0f} ms (should be faster than cold)")
        log(f"E2E:       {resp['e2e_ms']:.0f} ms")
        log(f"Tokens:    {resp['token_count']}")
        log(f"Decode:    {tps:.1f} tok/s")
        log(f"Output:    {text[:120]}...")

        ok = True
        if not text.strip():
            log("FAIL: Empty output!")
            ok = False
        if resp["token_count"] < 5:
            log(f"FAIL: Too few tokens ({resp['token_count']})")
            ok = False

        results.append(TestResult(
            name=test_name,
            status=PASS if ok else FAIL,
            details=f"TTFT={ttft:.0f}ms, {resp['token_count']} tokens, {tps:.1f} tok/s",
            timing_ms=resp["e2e_ms"],
            tokens_generated=resp["token_count"],
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_warm_cache_roundtrip(base_url: str, label: str):
    """TEST 3: Warm cache — evict from hot, reload from disk, verify output."""
    test_name = f"{label}: Warm cache roundtrip"
    section(test_name)
    session_id = f"test_cold_4k_{label.lower().replace(' ', '_')}"  # Reuse
    agent_id = f"oai_{session_id}"

    try:
        # Check cache file exists before eviction
        pre_files = get_cache_files()
        pre_agent = [f for f in pre_files if agent_id in f[0]]
        log(f"Pre-evict: {len(pre_files)} cache files, agent file exists: {len(pre_agent) > 0}")

        # Evict from hot tier (keep disk)
        await evict_agent(base_url, agent_id, keep_disk=True)
        await asyncio.sleep(1.0)

        # Verify disk file still exists
        post_files = get_cache_files()
        post_agent = [f for f in post_files if agent_id in f[0]]
        if post_agent:
            log(f"Disk file: {post_agent[0][0]} ({post_agent[0][1]:.2f} MB)")
        else:
            log("FAIL: Cache file missing after eviction!")
            results.append(TestResult(name=test_name, status=FAIL, details="Cache file missing after eviction"))
            return

        # Warm reload — same prompt, should reload from disk
        msgs = make_4k_messages()
        start = time.time()
        resp = await stream_chat(base_url, msgs, session_id, max_tokens=64)
        elapsed = time.time() - start

        text = resp["text"]
        ttft = resp["ttft_ms"]

        log(f"TTFT:      {ttft:.0f} ms (warm reload)")
        log(f"E2E:       {resp['e2e_ms']:.0f} ms")
        log(f"Tokens:    {resp['token_count']}")
        log(f"Output:    {text[:120]}...")

        ok = True
        if not text.strip():
            log("FAIL: Empty output after warm reload!")
            ok = False
        if resp["token_count"] < 3:
            log(f"FAIL: Too few tokens ({resp['token_count']})")
            ok = False

        results.append(TestResult(
            name=test_name,
            status=PASS if ok else FAIL,
            details=f"TTFT={ttft:.0f}ms warm, disk file {post_agent[0][1]:.2f}MB",
            timing_ms=resp["e2e_ms"],
            tokens_generated=resp["token_count"],
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_memory_pressure(base_url: str, label: str, port: int):
    """TEST 4: Memory pressure — submit multiple agents, check pool doesn't leak."""
    test_name = f"{label}: Memory pressure (5 agents)"
    section(test_name)

    try:
        mem_before = get_process_memory_mb(port)
        log(f"Memory before: {mem_before:.0f} MB")

        # Create 5 agents with ~2K token contexts each
        agent_sessions = []
        for i in range(5):
            sid = f"test_mempress_{label.lower().replace(' ', '_')}_{i}"
            msgs = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": PADDING + f"\n\nSummarize in one sentence. Agent {i}."},
            ]
            resp = await chat(base_url, msgs, sid, max_tokens=32)
            text = resp["choices"][0]["message"]["content"]
            agent_sessions.append((sid, text))
            log(f"  Agent {i}: {text[:60]}...")

        mem_after = get_process_memory_mb(port)
        mem_delta = mem_after - mem_before
        log(f"Memory after:  {mem_after:.0f} MB (delta: {mem_delta:+.0f} MB)")

        # Verify all agents produced output
        all_ok = all(len(text.strip()) > 0 for _, text in agent_sessions)
        if not all_ok:
            log("FAIL: Some agents produced empty output!")

        # Check memory didn't explode (allow 500MB growth for 5 agents)
        mem_ok = mem_delta < 500
        if not mem_ok:
            log(f"WARN: Memory grew by {mem_delta:.0f} MB (>500 MB threshold)")

        results.append(TestResult(
            name=test_name,
            status=PASS if (all_ok and mem_ok) else FAIL,
            details=f"5 agents OK, mem delta={mem_delta:+.0f}MB",
            memory_mb=mem_after,
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_semantic_quality(base_url: str, label: str):
    """TEST 5: Semantic quality — verify coherent, on-topic responses."""
    test_name = f"{label}: Semantic quality"
    section(test_name)

    try:
        checks = []

        # Check 1: Instruction following
        resp1 = await chat(base_url, [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ], f"test_quality_1_{label}", max_tokens=16)
        text1 = resp1["choices"][0]["message"]["content"].strip()
        has_four = "4" in text1
        log(f"Math: '{text1}' — {'PASS' if has_four else 'FAIL'}")
        checks.append(has_four)

        # Check 2: Non-empty, non-garbled response to a real question
        resp2 = await chat(base_url, [
            {"role": "user", "content": "Explain what a KV cache is in one sentence."}
        ], f"test_quality_2_{label}", max_tokens=64)
        text2 = resp2["choices"][0]["message"]["content"].strip()
        has_real_words = len(text2.split()) >= 5 and any(
            w in text2.lower() for w in ["cache", "key", "value", "attention", "store", "memory"]
        )
        log(f"KV cache: '{text2[:80]}...' — {'PASS' if has_real_words else 'FAIL'}")
        checks.append(has_real_words)

        # Check 3: Long context doesn't produce garbage
        resp3 = await chat(base_url, make_4k_messages(),
                           f"test_quality_3_{label}", max_tokens=64)
        text3 = resp3["choices"][0]["message"]["content"].strip()
        not_garbage = (
            len(text3) > 10 and
            len(text3.split()) >= 3 and
            not all(c == text3[0] for c in text3[:20])  # Not all same char
        )
        log(f"4K summary: '{text3[:80]}...' — {'PASS' if not_garbage else 'FAIL'}")
        checks.append(not_garbage)

        all_ok = all(checks)
        results.append(TestResult(
            name=test_name,
            status=PASS if all_ok else FAIL,
            details=f"{sum(checks)}/{len(checks)} quality checks passed",
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_cache_preservation(base_url: str, label: str):
    """TEST 6: Verify cache is preserved correctly across save/load cycle."""
    test_name = f"{label}: Cache preservation"
    section(test_name)
    session_id = f"test_cachepreserve_{label.lower().replace(' ', '_')}"
    agent_id = f"oai_{session_id}"

    try:
        # Step 1: Prime with a known prompt
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Remember this number: 42. What number did I say?"},
        ]
        resp1 = await chat(base_url, msgs, session_id, max_tokens=32)
        text1 = resp1["choices"][0]["message"]["content"]
        log(f"Turn 1: {text1[:80]}")

        # Check agent exists and has cache
        info1 = await get_agent_info(base_url, agent_id)
        cached_tokens_1 = info1.get("cache_size_tokens", 0) if info1 else 0
        log(f"Cached tokens after turn 1: {cached_tokens_1}")

        # Step 2: Follow-up turn (hot cache hit)
        msgs.append({"role": "assistant", "content": text1})
        msgs.append({"role": "user", "content": "What was the number again? Just the number."})
        resp2 = await chat(base_url, msgs, session_id, max_tokens=16)
        text2 = resp2["choices"][0]["message"]["content"]
        log(f"Turn 2 (hot): {text2[:80]}")

        info2 = await get_agent_info(base_url, agent_id)
        cached_tokens_2 = info2.get("cache_size_tokens", 0) if info2 else 0
        log(f"Cached tokens after turn 2: {cached_tokens_2}")

        # Cache should have grown
        cache_grew = cached_tokens_2 > cached_tokens_1
        log(f"Cache grew: {cache_grew} ({cached_tokens_1} -> {cached_tokens_2})")

        # Step 3: Evict + reload + third turn
        await evict_agent(base_url, agent_id, keep_disk=True)
        await asyncio.sleep(1.0)

        msgs.append({"role": "assistant", "content": text2})
        msgs.append({"role": "user", "content": "Multiply that number by 2. Just the result."})
        resp3 = await chat(base_url, msgs, session_id, max_tokens=16)
        text3 = resp3["choices"][0]["message"]["content"]
        log(f"Turn 3 (warm): {text3[:80]}")

        info3 = await get_agent_info(base_url, agent_id)
        cached_tokens_3 = info3.get("cache_size_tokens", 0) if info3 else 0
        log(f"Cached tokens after turn 3: {cached_tokens_3}")

        ok = cache_grew and cached_tokens_3 > 0 and len(text3.strip()) > 0
        results.append(TestResult(
            name=test_name,
            status=PASS if ok else FAIL,
            details=f"Tokens: {cached_tokens_1}->{cached_tokens_2}->{cached_tokens_3}",
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_prisoners_dilemma(base_url: str, label: str):
    """TEST 7: Prisoner's dilemma end-to-end scenario."""
    test_name = f"{label}: Prisoner's dilemma"
    section(test_name)

    scenario_path = Path(__file__).parent / "demo" / "scenarios" / "prisoners_dilemma.yaml"
    if not scenario_path.exists():
        log(f"SKIP: Scenario file not found at {scenario_path}")
        results.append(TestResult(name=test_name, status=SKIP, details="Scenario file missing"))
        return

    try:
        # Import the scenario runner
        sys.path.insert(0, str(Path(__file__).parent))
        from demo.lib import api_client
        from semantic.adapters.config.scenario_loader import load_scenario
        from test_prisoners_dilemma import build_agent_configs, collect_prior_messages

        spec = load_scenario(scenario_path)
        log(f"Scenario: {spec.title} ({len(spec.phases)} phases)")

        phase_results_list = []
        total_start = time.time()

        for phase_idx, phase in enumerate(spec.phases):
            log(f"\nPhase {phase_idx+1}/{len(spec.phases)}: {phase.label}")

            agent_configs = build_agent_configs(spec, phase)
            prior_agent_messages = None
            if phase_idx > 0:
                prior_agent_messages = collect_prior_messages(spec, phase_idx, phase_results_list)

            phase_start = time.time()
            result = api_client.create_session(
                base_url,
                topology=phase.topology,
                debate_format=phase.debate_format,
                decision_mode=phase.decision_mode,
                agents=agent_configs,
                initial_prompt=phase.initial_prompt,
                max_turns=100,
                persistent_cache_prefix=spec.id,
                prior_agent_messages=prior_agent_messages,
            )

            if not result:
                log(f"  FAIL: Session creation failed")
                results.append(TestResult(name=test_name, status=FAIL, details=f"Phase {phase_idx+1} session failed"))
                return

            session_id = result["session_id"]
            turn_count = phase.auto_rounds * len(phase.agents)

            success = api_client.execute_turns(base_url, session_id, turn_count)
            if not success:
                log(f"  FAIL: Turn execution failed")
                results.append(TestResult(name=test_name, status=FAIL, details=f"Phase {phase_idx+1} turns failed"))
                return

            messages = api_client.get_session_messages(base_url, session_id)
            phase_time = time.time() - phase_start

            log(f"  Done: {len(messages)} messages in {phase_time:.1f}s")
            phase_results_list.append({
                "phase": phase.label,
                "session_id": session_id,
                "messages": messages,
                "time_seconds": phase_time,
            })

        total_time = time.time() - total_start
        log(f"\nTotal: {total_time:.1f}s across {len(spec.phases)} phases")

        # Verify all phases completed
        all_ok = len(phase_results_list) == len(spec.phases)
        for pr in phase_results_list:
            if len(pr["messages"]) == 0:
                all_ok = False
                log(f"  FAIL: Phase '{pr['phase']}' has no messages!")

        results.append(TestResult(
            name=test_name,
            status=PASS if all_ok else FAIL,
            details=f"{len(spec.phases)} phases, {total_time:.1f}s total",
            timing_ms=total_time * 1000,
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


async def test_drain_timeout_safety(base_url: str, label: str):
    """TEST 8: Verify step() safety limit doesn't break normal generation."""
    test_name = f"{label}: Drain/step safety (normal gen works)"
    section(test_name)

    try:
        # This is a smoke test — normal generation should still work perfectly
        # after the step() max_iterations change
        resp = await chat(base_url, [
            {"role": "user", "content": "Count from 1 to 10 with commas."}
        ], f"test_drain_{label}", max_tokens=64)
        text = resp["choices"][0]["message"]["content"].strip()
        log(f"Output: {text[:80]}")

        # Should contain numbers
        has_numbers = any(str(n) in text for n in range(1, 11))
        ok = len(text) > 5 and has_numbers
        results.append(TestResult(
            name=test_name,
            status=PASS if ok else FAIL,
            details=f"Normal gen OK: {text[:50]}",
        ))
    except Exception as e:
        log(f"FAIL: {e}")
        results.append(TestResult(name=test_name, status=FAIL, details=str(e)))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

async def run_model_tests(base_url: str, label: str, port: int):
    """Run all tests for one model."""
    if not await server_ready(base_url):
        log(f"Server not reachable at {base_url} — skipping {label}")
        results.append(TestResult(name=f"{label}: Server check", status=SKIP))
        return

    log(f"Server ready at {base_url}")
    mem_start = get_process_memory_mb(port)
    log(f"Initial memory: {mem_start:.0f} MB\n")

    # Run tests in order — some depend on prior state
    await test_cold_cache_4k(base_url, label)
    await test_hot_cache_continuation(base_url, label)
    await test_warm_cache_roundtrip(base_url, label)
    await test_memory_pressure(base_url, label, port)
    await test_semantic_quality(base_url, label)
    await test_cache_preservation(base_url, label)
    await test_drain_timeout_safety(base_url, label)
    await test_prisoners_dilemma(base_url, label)

    mem_end = get_process_memory_mb(port)
    log(f"\nFinal memory: {mem_end:.0f} MB (delta: {mem_end - mem_start:+.0f} MB)")


async def main():
    print("\n" + "=" * 78)
    print("  COMPREHENSIVE REAL-MODEL VALIDATION")
    print("  Tests: 4K+ inputs, hot/warm/cold cache, memory, timing, quality")
    print("=" * 78)

    # Clean caches for reproducibility
    cache_files = get_cache_files()
    if cache_files:
        log(f"Clearing {len(cache_files)} existing cache files...")
        for f, _ in cache_files:
            (CACHE_DIR / f).unlink(missing_ok=True)

    # Test Gemma 3
    print("\n" + "#" * 78)
    print("  MODEL 1: Gemma 3 (port 8000)")
    print("#" * 78)
    await run_model_tests(GEMMA_URL, "Gemma3", 8000)

    # Test DeepSeek
    print("\n" + "#" * 78)
    print("  MODEL 2: DeepSeek (port 8001)")
    print("#" * 78)
    await run_model_tests(DEEPSEEK_URL, "DeepSeek", 8001)

    # Summary
    print("\n" + "=" * 78)
    print("  VALIDATION SUMMARY")
    print("=" * 78)

    passed = sum(1 for r in results if r.status == PASS)
    failed = sum(1 for r in results if r.status == FAIL)
    skipped = sum(1 for r in results if r.status == SKIP)

    for r in results:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[r.status]
        timing = f" ({r.timing_ms:.0f}ms)" if r.timing_ms > 0 else ""
        print(f"  [{icon}] {r.name}{timing}")
        if r.details:
            print(f"         {r.details}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 78)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
