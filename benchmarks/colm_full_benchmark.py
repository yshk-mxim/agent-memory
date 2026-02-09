#!/usr/bin/env python3
"""COLM 2026 full benchmark suite for semantic server.

Runs the complete measurement matrix for the paper:
- Models: gemma-3-12b-it-4bit, DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
- Contexts: 1K, 2K, 4K, 8K, 16K, 32K
- Cache states: cold, warm, hot
- Modes: streaming, non-streaming
- Batch sizes: 1 (all contexts), 2 (1K-16K, skip 32K)
- Staggered arrivals: sequential vs batched
- 3 passes per config, T=0.0 greedy (deterministic, via request body)

Server lifecycle:
- ONE server per model (scheduler=on, batch_size=2) — stays running for all phases
- Single requests route through the scheduler without measurable overhead
- Between measurements: caches flushed via admin API + filesystem cleanup (no restart)
- Between models: graceful stop (offloads model + GPU memory), then start next model

Temperature: T=0.0 is set per-request in the JSON body. The env var
SEMANTIC_MLX_DEFAULT_TEMPERATURE is intentionally NOT set because it has
no effect: the OpenAI adapter uses request_body.temperature directly, and
the coordination service hardcodes T=0.3 (ignoring the env var).

Usage:
    python benchmarks/colm_full_benchmark.py              # Full run (~35 hours)
    python benchmarks/colm_full_benchmark.py --quick       # 1 pass, 1K/4K/16K only
    python benchmarks/colm_full_benchmark.py --models gemma
    python benchmarks/colm_full_benchmark.py --models deepseek
    python benchmarks/colm_full_benchmark.py --resume FILE
    python benchmarks/colm_full_benchmark.py --port 8399
    python benchmarks/colm_full_benchmark.py --passes 1
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Reuse existing infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from openai_benchmark import (
    OPENAI_BENCH_ENV,
    PADDING_TEXT,
    OpenAIPromptFactory,
    OpenAIRequestClient,
    OpenAIStreamingClient,
)
from capability_benchmark import (
    ScenarioResult,
    ServerManager,
    compute_stats,
)
from streaming_benchmark import (
    _delete_agent,
    _wait_for_server,
)
from staggered_benchmark import StaggeredResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
CORPUS_PATH = Path(__file__).parent / "data" / "prefill_corpus.txt"

MODELS = {
    "gemma": "mlx-community/gemma-3-12b-it-4bit",
    "deepseek": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
}

ALL_CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768]
QUICK_CONTEXTS = [1024, 4096, 16384]

CACHE_STATES = ["cold", "warm", "hot"]
MODES = ["streaming", "non-streaming"]

OUTPUT_TOKENS = 64
DEFAULT_PASSES = 3
DEFAULT_PORT = 8399
MAX_COOLDOWN_SECONDS = 240  # Safety cap
MIN_COOLDOWN_SECONDS = 10   # Minimum to let caches flush
COOLDOWN_POLL_INTERVAL = 5  # Seconds between thermal checks
WARMUP_SETTLE_SECONDS = 30
THROTTLE_TPS_TOLERANCE = 0.20  # 20% tolerance for TPS recovery (sustained inference drops ~15% from peak)
STAGGER_DELAY = 2.0
STAGGER_CONTEXT = 4096  # Paper uses 4K context for staggered arrivals (Figure 3)
# T=0.0 greedy (argmax): fully deterministic, no seed needed.
# At T=0, make_sampler() uses mx.argmax — no random variable involved.
# At T>0, logits/T + Uniform(0,1) → non-deterministic even with seed
# because MLX categorical sampling is not seed-stable across runs.
# Echo loops possible in multi-turn (hot) — detected by structural checks.
# Cold/warm are single-turn summarization: no echo risk.
TEMPERATURE = 0.0

ADMIN_KEY = "benchmark"

# Model-specific cache budgets (MB) — smaller budget leaves more headroom
# for intermediate tensors.  DeepSeek-Coder-V2-Lite has 64-expert MoE layers
# that need significant intermediate memory during forward pass.
MODEL_CACHE_BUDGET: dict[str, int] = {
    "gemma": 8192,      # default
    "deepseek": 4096,   # reduced — MoE intermediates need headroom
}

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus() -> str:
    """Load diverse prefill corpus from disk, fallback to PADDING_TEXT."""
    if CORPUS_PATH.exists():
        text = CORPUS_PATH.read_text(encoding="utf-8")
        if len(text) > 10000:
            return text
        print(f"  WARNING: Corpus file too small ({len(text)} bytes), using padding")
    else:
        print(f"  WARNING: Corpus file not found at {CORPUS_PATH}, using padding")
    # Fallback: repeat PADDING_TEXT enough for 32K tokens
    return PADDING_TEXT * 5000


def build_messages(
    corpus: str, target_tokens: int, offset: int = 0
) -> list[dict[str, str]]:
    """Build prompt from diverse corpus text at the target token count.

    Each measurement uses a different offset into the corpus to prevent
    attention caching or prefix-matching artifacts across measurements.
    If the corpus is too small (or missing), falls back to PADDING_TEXT.

    Previously, >4K contexts used repeated PADDING_TEXT because Gemma 3
    generated early EOS with diverse text at long contexts.  This was
    caused by the chunked prefill sliding window mask bug (ee24513) —
    QuantizedKVCache.make_mask() ignored window_size, corrupting KV
    entries for Gemma 3's 41/46 sliding window layers.  With that fix,
    diverse corpus produces correct output at all context lengths up to
    16K+ tokens (validated on both Gemma 3 and DeepSeek).
    """
    chars_needed = target_tokens * 4  # ~4 chars/token

    if len(corpus) >= chars_needed + 1000:
        # Diverse corpus: slice from offset, wrap around if needed
        max_start = len(corpus) - chars_needed
        start = offset % max_start if max_start > 0 else 0
        content = corpus[start : start + chars_needed]
    else:
        # Fallback: repeat PADDING_TEXT (only if corpus file is missing/tiny)
        content = (PADDING_TEXT + " ") * (chars_needed // len(PADDING_TEXT) + 1)
        content = content[:chars_needed]
    return [
        {
            "role": "user",
            "content": (
                f"Here is some text:\n\n{content}\n\n"
                "What are the main topics and themes discussed above?"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Memory pressure check
# ---------------------------------------------------------------------------


def get_system_memory_free_pct() -> int:
    """Return system-wide memory free percentage (0-100), or -1 on error."""
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            if "free percentage" in line:
                # "System-wide memory free percentage: 17%"
                pct = int(line.strip().rstrip("%").split(":")[-1].strip())
                return pct
    except Exception:
        pass
    return -1


def check_memory_pressure(min_free_pct: int = 20, max_wait: int = 300) -> bool:
    """Wait until memory pressure drops below threshold.

    Returns True if memory is OK, False if timeout reached.
    """
    pct = get_system_memory_free_pct()
    if pct < 0:
        print("  WARNING: Cannot read memory pressure, proceeding anyway")
        return True
    if pct >= min_free_pct:
        return True

    print(f"  Memory pressure HIGH: {pct}% free (need {min_free_pct}%)")
    print(f"  Waiting up to {max_wait}s for memory recovery...")

    t_start = time.time()
    while (time.time() - t_start) < max_wait:
        time.sleep(15)
        gc.collect()
        pct = get_system_memory_free_pct()
        elapsed = int(time.time() - t_start)
        if pct >= min_free_pct:
            print(f"  Memory OK: {pct}% free (waited {elapsed}s)")
            return True
        print(f"  Still {pct}% free ({elapsed}s/{max_wait}s)...")

    pct = get_system_memory_free_pct()
    print(f"  WARNING: Memory still {pct}% free after {max_wait}s wait")
    print(f"  Consider rebooting if model loading fails (Metal OOM)")
    return False


# ---------------------------------------------------------------------------
# Server environment
# ---------------------------------------------------------------------------


def build_server_env(model_id: str, model_key: str = "") -> dict[str, str]:
    """Build env dict for server startup.

    Always starts with scheduler=on, batch_size=2.  Single requests route
    through the scheduler without measurable overhead, and this avoids a
    server restart between batch=1 and batch=2 phases.

    Temperature is set per-request in the JSON body (T=0.0 greedy).
    SEMANTIC_MLX_DEFAULT_TEMPERATURE is NOT set here because:
      - The OpenAI adapter uses request_body.temperature directly
      - The coordination service hardcodes T=0.3 and ignores the env var
    reasoning_extra_tokens defaults to 0 in settings.py — no override needed.
    """
    env = dict(OPENAI_BENCH_ENV)
    cache_budget = MODEL_CACHE_BUDGET.get(model_key, 8192)
    env.update({
        "SEMANTIC_MLX_MODEL_ID": model_id,
        "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
        "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
        "SEMANTIC_ADMIN_KEY": ADMIN_KEY,
        "SEMANTIC_MLX_CACHE_BUDGET_MB": str(cache_budget),
    })
    return env


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def check_structural(
    raw_output: str, output_tokens: int, expected_tokens: int = OUTPUT_TOKENS,
) -> list[str]:
    """Fast structural checks on generated output."""
    issues = []
    if not raw_output.strip():
        issues.append("empty_output")
    if output_tokens < 5:
        issues.append("too_few_tokens")
    # Early EOS: model stopped well before max_tokens.
    # Only flag when output is less than half expected — the model legitimately
    # finishing a coherent response (e.g. 45/64 tokens) is not a quality issue.
    if 0 < output_tokens < expected_tokens // 2:
        issues.append(f"early_eos_{output_tokens}_of_{expected_tokens}")
    # 4-gram repetition loop
    words = raw_output.split()
    for i in range(len(words) - 3):
        gram = " ".join(words[i : i + 4])
        if raw_output.count(gram) > 5:
            issues.append("repetition_loop")
            break
    # Spacing corruption (words concatenated without spaces)
    if len(words) > 0:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 20:
            issues.append("spacing_corruption")
    return issues


def check_semantic(raw_output: str, prompt_context: str) -> dict[str, Any]:
    """Heuristic semantic checks for manual review."""
    prompt_words = set(prompt_context.lower().split()[:200])
    output_words_list = raw_output.lower().split()
    output_words = set(output_words_list)
    overlap = len(prompt_words & output_words)
    return {
        "relevance_score": round(overlap / max(len(output_words), 1), 3),
        "output_word_count": len(output_words_list),
        "unique_word_ratio": round(
            len(output_words) / max(len(output_words_list), 1), 3
        ),
        "has_punctuation": any(c in raw_output for c in ".!?,;:"),
        "starts_with_capital": (
            raw_output.strip()[:1].isupper() if raw_output.strip() else False
        ),
    }


# ---------------------------------------------------------------------------
# Admin API helpers
# ---------------------------------------------------------------------------


async def clear_all_caches(base_url: str) -> dict[str, Any]:
    """Clear all caches via admin API."""
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            r = await c.delete(
                f"{base_url}/admin/caches",
                headers={"X-Admin-Key": ADMIN_KEY},
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"    WARN: clear_all_caches failed: {e}")
    return {}



async def get_agent_stats(base_url: str) -> dict[str, Any]:
    """Fetch pool utilization and cache size from /v1/agents/stats."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base_url}/v1/agents/stats")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}


async def get_memory_stats(base_url: str) -> dict[str, float]:
    """Fetch MLX memory stats from /debug/memory."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base_url}/debug/memory")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}


async def check_server_health(base_url: str) -> bool:
    """Check if server is responding."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


def clean_cache_files() -> None:
    """Belt-and-suspenders filesystem cleanup."""
    cache_dir = Path.home() / ".semantic" / "caches"
    if cache_dir.exists():
        for f in cache_dir.glob("*.safetensors"):
            try:
                f.unlink()
            except OSError:
                pass
        for f in cache_dir.glob("*.tmp"):
            try:
                f.unlink()
            except OSError:
                pass


def kill_all_servers() -> None:
    """Kill any running semantic servers and streamlit processes.

    Kills by name pattern AND by port to catch stragglers.
    Uses SIGKILL (-9) after a grace period.
    Excludes the current process (and its parent) to avoid self-kill.
    """
    my_pid = str(os.getpid())
    my_ppid = str(os.getppid())
    safe_pids = {my_pid, my_ppid}

    def _kill_by_pattern(pattern: str, signal: str = "") -> None:
        """Kill processes matching pattern, excluding self."""
        try:
            # Use pgrep to get PIDs, then kill selectively
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid.isdigit() and pid not in safe_pids:
                        cmd = ["kill", pid] if not signal else ["kill", signal, pid]
                        subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception:
            pass

    # 1. Graceful SIGTERM — lets the server run its shutdown path
    #    (release model tensors, clear Metal cache, gc.collect)
    for pattern in ["semantic.*serve", "semantic.*cli", "streamlit"]:
        _kill_by_pattern(pattern)
    time.sleep(10)  # Give server time to release GPU memory

    # 2. Force kill stragglers (SIGKILL)
    for pattern in ["semantic.*serve", "semantic.*cli", "streamlit"]:
        _kill_by_pattern(pattern, signal="-9")

    # 3. Kill by port — catch anything still listening
    for port in [8000, 8001, 8005, 8399]:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid.isdigit() and pid not in safe_pids:
                        subprocess.run(
                            ["kill", "-9", pid],
                            capture_output=True,
                            timeout=5,
                        )
        except Exception:
            pass
    time.sleep(2)


# ---------------------------------------------------------------------------
# Thermal monitoring & adaptive cooldown
# ---------------------------------------------------------------------------


def get_thermal_state() -> int:
    """Read macOS thermal pressure via NSProcessInfo.thermalState.

    Returns: 0=nominal, 1=fair, 2=serious, 3=critical, -1=unavailable
    """
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        NSProcessInfo = objc.objc_getClass(b"NSProcessInfo")
        info = objc.objc_msgSend(
            NSProcessInfo, objc.sel_registerName(b"processInfo")
        )

        send_long = objc.objc_msgSend
        send_long.restype = ctypes.c_long
        send_long.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        return send_long(info, objc.sel_registerName(b"thermalState"))
    except Exception:
        return -1


_THERMAL_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical", -1: "unknown"}


async def tps_probe(base_url: str, corpus: str) -> float:
    """Run a tiny inference request and return avg TPS (tokens/total_time).

    Uses 256-token context, 16 output tokens — non-streaming for accurate TPS.
    (Streaming decode TPS is meaningless: tokens arrive near-instantly after TTFT.)
    """
    messages = build_messages(corpus, 256, offset=99999)
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": 16,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    sid = f"tps_probe_{int(time.time())}"
    try:
        client = OpenAIRequestClient(base_url)
        r = await client.send_and_measure(body, session_id=sid)
        await client.close()
        await _delete_agent(base_url, f"oai_{sid}")
        # avg_tps = output_tokens / e2e_seconds
        if r.e2e_ms > 0 and r.output_tokens > 0:
            return r.output_tokens / (r.e2e_ms / 1000)
        return 0.0
    except Exception:
        return 0.0


async def calibrate_baseline_tps(base_url: str, corpus: str) -> float:
    """Run 3 probes at startup (after warmup) to establish baseline TPS."""
    probes = []
    for _ in range(3):
        tps = await tps_probe(base_url, corpus)
        if tps > 0:
            probes.append(tps)
        await asyncio.sleep(1)
    if probes:
        baseline = statistics.median(probes)
        print(f"  Baseline TPS: {baseline:.1f} (from {len(probes)} probes)")
        return baseline
    print("  WARNING: Could not calibrate baseline TPS, using fixed cooldown")
    return 0.0


async def adaptive_cooldown(
    base_url: str,
    corpus: str,
    baseline_tps: float,
) -> float:
    """Wait until thermal state is nominal AND TPS recovers to baseline.

    Returns actual seconds waited.
    """
    t_start = time.time()

    # Always wait minimum cooldown
    await asyncio.sleep(MIN_COOLDOWN_SECONDS)

    # If no baseline, fall back to fixed cooldown
    if baseline_tps <= 0:
        remaining = MAX_COOLDOWN_SECONDS - MIN_COOLDOWN_SECONDS
        if remaining > 0:
            await asyncio.sleep(remaining)
        return MAX_COOLDOWN_SECONDS

    while (time.time() - t_start) < MAX_COOLDOWN_SECONDS:
        thermal = get_thermal_state()
        thermal_name = _THERMAL_NAMES.get(thermal, "unknown")

        # If thermal state is serious/critical, keep waiting
        if thermal >= 2:
            print(f"    Thermal: {thermal_name}, waiting...", flush=True)
            await asyncio.sleep(COOLDOWN_POLL_INTERVAL)
            continue

        # Thermal is nominal or fair — check TPS recovery
        current_tps = await tps_probe(base_url, corpus)
        if current_tps <= 0:
            await asyncio.sleep(COOLDOWN_POLL_INTERVAL)
            continue

        ratio = current_tps / baseline_tps
        if ratio >= (1.0 - THROTTLE_TPS_TOLERANCE):
            elapsed = time.time() - t_start
            print(
                f"    Ready: thermal={thermal_name}, "
                f"TPS={current_tps:.1f}/{baseline_tps:.1f} "
                f"({ratio:.0%}), waited {elapsed:.0f}s",
                flush=True,
            )
            return elapsed

        print(
            f"    Throttled: thermal={thermal_name}, "
            f"TPS={current_tps:.1f}/{baseline_tps:.1f} "
            f"({ratio:.0%}), waiting...",
            flush=True,
        )
        await asyncio.sleep(COOLDOWN_POLL_INTERVAL)

    elapsed = time.time() - t_start
    print(f"    Max cooldown reached ({elapsed:.0f}s)", flush=True)
    return elapsed


# ---------------------------------------------------------------------------
# Server crash recovery
# ---------------------------------------------------------------------------


class ManagedServer:
    """ServerManager wrapper with crash detection and auto-restart.

    The server stays running for the entire model benchmark (all phases).
    Between measurements, caches are flushed via admin API — no restarts.
    Only restarts on crash recovery or when switching models.
    """

    def __init__(self, port: int, env: dict[str, str], max_restarts: int = 3):
        self.port = port
        self.env = env
        self.max_restarts = max_restarts
        self._restart_count = 0
        self._server = ServerManager(port=port)
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self) -> None:
        """Start server (stops any previous instance on same port first)."""
        self._restart_count = 0
        self._do_start()

    def _do_start(self) -> None:
        """Internal start — graceful, no aggressive kill-all."""
        # Stop our own server if running
        try:
            if self._server and self._server.is_alive():
                self._server.stop()
                time.sleep(2)
        except Exception:
            pass
        clean_cache_files()
        self._server = ServerManager(port=self.port)
        self._server.start(env_overrides=self.env)

    def stop(self) -> None:
        """Stop server gracefully, allowing GPU memory release."""
        if self._server:
            self._server.stop()
            # Wait for OS to reclaim Metal/GPU memory after process exits
            time.sleep(5)

    def is_alive(self) -> bool:
        """Check if server process is running."""
        return self._server is not None and self._server.is_alive()

    async def ensure_alive_async(self) -> bool:
        """Async version of ensure_alive."""
        if self.is_alive():
            # Also check HTTP health
            healthy = await check_server_health(self.base_url)
            if healthy:
                return True

        if self._restart_count >= self.max_restarts:
            pct = get_system_memory_free_pct()
            print(f"  FATAL: Server died {self._restart_count} times, giving up")
            if pct >= 0 and pct < 20:
                print(f"  System memory: {pct}% free — likely Metal OOM.")
                print(f"  Reboot recommended to reclaim wired GPU memory.")
            return False

        self._restart_count += 1
        print(f"  SERVER CRASHED — restarting (attempt {self._restart_count}/{self.max_restarts})...")

        try:
            self._server.stop()
        except Exception:
            pass

        try:
            # Crash recovery is the one place we kill aggressively
            kill_all_servers()
            clean_cache_files()
            time.sleep(2)
            self._server = ServerManager(port=self.port)
            self._server.start(env_overrides=self.env)

            if not await _wait_for_server(self.base_url, timeout=180):
                print("  FATAL: Server failed to restart")
                return False

            print("  Server restarted successfully")
            self._restart_count = 0
            return True
        except Exception as e:
            print(f"  FATAL: Restart failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------


async def measure_cold(
    base_url: str,
    context: int,
    output_tokens: int,
    mode: str,
    run_id: str,
    corpus: str,
    corpus_offset: int,
) -> dict[str, Any]:
    """Cold start measurement — no prior cache."""
    await clear_all_caches(base_url)
    clean_cache_files()
    await asyncio.sleep(2)

    messages = build_messages(corpus, context, corpus_offset)
    prompt_text = messages[-1]["content"][:500]  # For semantic check

    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": mode == "streaming",
    }
    sid = f"cold_{context}_{mode}_{run_id}"

    try:
        if mode == "streaming":
            client = OpenAIStreamingClient(base_url)
        else:
            client = OpenAIRequestClient(base_url)
        r = await client.send_and_measure(body, session_id=sid)
        if hasattr(client, "close"):
            await client.close()
    except Exception as e:
        return _error_record("cold", context, mode, run_id, str(e))

    stats = await get_agent_stats(base_url)
    mem = await get_memory_stats(base_url)
    result = _build_record("cold", context, mode, run_id, r, prompt_text)
    result["pool_utilization_pct"] = stats.get("pool_utilization_pct", 0)
    result["cache_size_mb"] = stats.get("total_cache_size_mb", 0)
    result["peak_memory_mb"] = mem.get("peak_memory_mb", 0)
    await _delete_agent(base_url, f"oai_{sid}")
    return result


async def measure_warm(
    base_url: str,
    context: int,
    output_tokens: int,
    mode: str,
    run_id: str,
    corpus: str,
    corpus_offset: int,
) -> dict[str, Any]:
    """Warm cache measurement — prime, evict to disk, reload."""
    await clear_all_caches(base_url)
    clean_cache_files()
    await asyncio.sleep(2)

    messages = build_messages(corpus, context, corpus_offset)
    prompt_text = messages[-1]["content"][:500]

    prime_body = {
        "model": "default",
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    sid = f"warm_{context}_{mode}_{run_id}"

    try:
        # Prime: cold request to populate cache
        prime_client = OpenAIRequestClient(base_url)
        await prime_client.send_and_measure(prime_body, session_id=sid)
        await prime_client.close()

        # Evict from hot tier but keep disk file
        await _delete_agent(base_url, f"oai_{sid}", evict_only=True)
        await asyncio.sleep(1.5)  # Allow disk write

        # Measure: reload from disk
        measure_body = {**prime_body, "stream": mode == "streaming"}
        if mode == "streaming":
            client = OpenAIStreamingClient(base_url)
        else:
            client = OpenAIRequestClient(base_url)
        r = await client.send_and_measure(measure_body, session_id=sid)
        if hasattr(client, "close"):
            await client.close()
    except Exception as e:
        return _error_record("warm", context, mode, run_id, str(e))

    stats = await get_agent_stats(base_url)
    mem = await get_memory_stats(base_url)
    result = _build_record("warm", context, mode, run_id, r, prompt_text)
    result["pool_utilization_pct"] = stats.get("pool_utilization_pct", 0)
    result["cache_size_mb"] = stats.get("total_cache_size_mb", 0)
    result["peak_memory_mb"] = mem.get("peak_memory_mb", 0)
    await _delete_agent(base_url, f"oai_{sid}")
    return result


async def measure_hot(
    base_url: str,
    context: int,
    output_tokens: int,
    mode: str,
    run_id: str,
    corpus: str,
    corpus_offset: int,
) -> dict[str, Any]:
    """Hot cache measurement — multi-turn, measure turn 3."""
    await clear_all_caches(base_url)
    clean_cache_files()
    await asyncio.sleep(2)

    messages = build_messages(corpus, context, corpus_offset)
    prompt_text = messages[-1]["content"][:500]
    factory = OpenAIPromptFactory()

    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    sid = f"hot_{context}_{mode}_{run_id}"

    try:
        prime_client = OpenAIRequestClient(base_url)

        # Turn 1: cold
        r1 = await prime_client.send_and_measure(body, session_id=sid)
        await asyncio.sleep(0.3)

        # Turn 2: extend
        assistant_text = r1.raw_output if r1.raw_output else "I see."
        followup1 = factory.build_followup(messages, assistant_text, output_tokens)
        followup1["temperature"] = TEMPERATURE
        await prime_client.send_and_measure(followup1, session_id=sid)
        await prime_client.close()
        await asyncio.sleep(0.3)

        # Turn 3: hot measurement
        followup2 = factory.build_followup(
            followup1["messages"], "Understood.", output_tokens
        )
        followup2["stream"] = mode == "streaming"
        followup2["temperature"] = TEMPERATURE

        if mode == "streaming":
            client = OpenAIStreamingClient(base_url)
        else:
            client = OpenAIRequestClient(base_url)
        r = await client.send_and_measure(followup2, session_id=sid)
        if hasattr(client, "close"):
            await client.close()
    except Exception as e:
        return _error_record("hot", context, mode, run_id, str(e))

    stats = await get_agent_stats(base_url)
    mem = await get_memory_stats(base_url)
    result = _build_record("hot", context, mode, run_id, r, prompt_text)
    result["pool_utilization_pct"] = stats.get("pool_utilization_pct", 0)
    result["cache_size_mb"] = stats.get("total_cache_size_mb", 0)
    result["peak_memory_mb"] = mem.get("peak_memory_mb", 0)
    await _delete_agent(base_url, f"oai_{sid}")
    return result


async def measure_concurrent(
    base_url: str,
    context: int,
    output_tokens: int,
    mode: str,
    cache_state: str,
    run_id: str,
    corpus: str,
    corpus_offset: int,
) -> dict[str, Any]:
    """Concurrent pair (batch=2) measurement."""
    await clear_all_caches(base_url)
    clean_cache_files()
    await asyncio.sleep(2)

    streaming = mode == "streaming"
    messages_a = build_messages(corpus, context, corpus_offset)
    messages_b = build_messages(corpus, context, corpus_offset + 5000)
    prompt_text = messages_a[-1]["content"][:500]

    body_a = {
        "model": "default",
        "messages": messages_a,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": streaming,
    }
    body_b = {
        "model": "default",
        "messages": messages_b,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": streaming,
    }

    sid_a = f"conc_a_{context}_{mode}_{run_id}"
    sid_b = f"conc_b_{context}_{mode}_{run_id}"

    # Timeout: 2x single-request httpx timeout (300s) to cover concurrent prefill
    CONCURRENT_TIMEOUT = 600

    try:
        # For warm/hot, prime first
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
        avg_e2e = sum(r.e2e_ms for r in results) / 2
        avg_ttft = sum(r.ttft_ms for r in results) / 2 if streaming else 0
        system_tps = total_output / (wall_ms / 1000) if wall_ms > 0 else 0

        raw_outputs = [r.raw_output for r in results]
        combined_raw = " ".join(raw_outputs)
        struct_issues = check_structural(combined_raw, total_output)
        sem_check = check_semantic(combined_raw, prompt_text)

    except Exception as e:
        return _error_record(cache_state, context, mode, run_id, str(e), batch_size=2)

    stats = await get_agent_stats(base_url)
    mem = await get_memory_stats(base_url)
    await _delete_agent(base_url, f"oai_{sid_a}")
    await _delete_agent(base_url, f"oai_{sid_b}")

    return {
        "model_id": "",  # Filled by caller
        "context_tokens": context,
        "batch_size": 2,
        "cache_state": cache_state,
        "mode": mode,
        "pass_id": run_id,
        "wall_ms": round(wall_ms, 1),
        "avg_e2e_ms": round(avg_e2e, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "total_output_tokens": total_output,
        "system_tps": round(system_tps, 1),
        "per_request_tps": round(system_tps / 2, 1),
        "raw_output": combined_raw[:500],
        "quality_ok": len(struct_issues) == 0,
        "quality_structural": struct_issues,
        "quality_semantic": sem_check,
        "pool_utilization_pct": stats.get("pool_utilization_pct", 0),
        "cache_size_mb": stats.get("total_cache_size_mb", 0),
        "peak_memory_mb": mem.get("peak_memory_mb", 0),
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Staggered arrival measurements (uses build_messages for consistent prompts)
# ---------------------------------------------------------------------------


async def staggered_run_sequential(
    base_url: str,
    context_tokens: int,
    output_tokens: int,
    run_id: int,
    corpus: str,
    corpus_offset: int = 0,
) -> StaggeredResult:
    """Sequential serving: User A completes, then User B starts."""
    client = OpenAIStreamingClient(base_url)

    messages_a = build_messages(corpus, context_tokens, corpus_offset)
    messages_b = build_messages(corpus, context_tokens, corpus_offset + 5000)
    body_a = {
        "model": "default",
        "messages": messages_a,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    body_b = {
        "model": "default",
        "messages": messages_b,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    sid_a = f"stagger_seq_a_{run_id}"
    sid_b = f"stagger_seq_b_{run_id}"

    t_start_wall = time.perf_counter()
    result_a = await client.send_and_measure(body_a, session_id=sid_a)
    await _delete_agent(base_url, f"oai_{sid_a}")

    t_start_b = time.perf_counter()
    result_b = await client.send_and_measure(body_b, session_id=sid_b)
    t_end_wall = time.perf_counter()
    await _delete_agent(base_url, f"oai_{sid_b}")

    total_wall_ms = (t_end_wall - t_start_wall) * 1000
    user_b_delay_ms = (t_start_b - t_start_wall) * 1000
    total_tokens = result_a.output_tokens + result_b.output_tokens
    system_tps = (total_tokens / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0

    return StaggeredResult(
        mode="sequential",
        run_id=run_id,
        user_a_ttft_ms=result_a.ttft_ms,
        user_a_e2e_ms=result_a.e2e_ms,
        user_b_ttft_ms=result_b.ttft_ms,
        user_b_e2e_ms=result_b.e2e_ms,
        user_b_start_delay_ms=user_b_delay_ms,
        user_b_wait_ms=user_b_delay_ms + result_b.ttft_ms,
        total_wall_time_ms=total_wall_ms,
        user_a_tps=result_a.decode_tps,
        user_b_tps=result_b.decode_tps,
        system_tps=system_tps,
        error=result_a.error or result_b.error,
    )


async def staggered_run_batched(
    base_url: str,
    context_tokens: int,
    output_tokens: int,
    stagger_delay: float,
    run_id: int,
    corpus: str,
    corpus_offset: int = 0,
) -> StaggeredResult:
    """Batched serving: User B joins while User A is running."""
    client = OpenAIStreamingClient(base_url)

    messages_a = build_messages(corpus, context_tokens, corpus_offset)
    messages_b = build_messages(corpus, context_tokens, corpus_offset + 5000)
    body_a = {
        "model": "default",
        "messages": messages_a,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    body_b = {
        "model": "default",
        "messages": messages_b,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    sid_a = f"stagger_batch_a_{run_id}"
    sid_b = f"stagger_batch_b_{run_id}"

    t_start_wall = time.perf_counter()

    async def launch_a():
        return await client.send_and_measure(body_a, session_id=sid_a)

    async def launch_b():
        await asyncio.sleep(stagger_delay)
        t_b = time.perf_counter()
        result = await client.send_and_measure(body_b, session_id=sid_b)
        return result, t_b

    result_a, (result_b, t_b_start) = await asyncio.gather(launch_a(), launch_b())
    t_end_wall = time.perf_counter()

    await _delete_agent(base_url, f"oai_{sid_a}")
    await _delete_agent(base_url, f"oai_{sid_b}")

    total_wall_ms = (t_end_wall - t_start_wall) * 1000
    user_b_delay_ms = (t_b_start - t_start_wall) * 1000
    total_tokens = result_a.output_tokens + result_b.output_tokens
    system_tps = (total_tokens / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0

    return StaggeredResult(
        mode="batched",
        run_id=run_id,
        user_a_ttft_ms=result_a.ttft_ms,
        user_a_e2e_ms=result_a.e2e_ms,
        user_b_ttft_ms=result_b.ttft_ms,
        user_b_e2e_ms=result_b.e2e_ms,
        user_b_start_delay_ms=user_b_delay_ms,
        user_b_wait_ms=user_b_delay_ms + result_b.ttft_ms,
        total_wall_time_ms=total_wall_ms,
        user_a_tps=result_a.decode_tps,
        user_b_tps=result_b.decode_tps,
        system_tps=system_tps,
        error=result_a.error or result_b.error,
    )


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _build_record(
    cache_state: str,
    context: int,
    mode: str,
    run_id: str,
    r: ScenarioResult,
    prompt_text: str,
) -> dict[str, Any]:
    """Build measurement record from ScenarioResult."""
    raw = r.raw_output or ""
    out_tok = r.output_tokens
    struct_issues = check_structural(raw, out_tok)
    sem_check = check_semantic(raw, prompt_text)

    # For streaming: ttft_ms is true time-to-first-token (= prefill time).
    # For non-streaming: OpenAIRequestClient sets ttft_ms = e2e_ms (no
    # streaming granularity), so prefill_ms is only meaningful for streaming.
    is_streaming = mode == "streaming"
    ttft = r.ttft_ms if is_streaming else 0.0
    decode_ms = (r.e2e_ms - ttft) if ttft > 0 else r.e2e_ms
    tpot = (decode_ms / max(out_tok - 1, 1)) if out_tok > 0 else 0

    return {
        "model_id": "",  # Filled by caller
        "context_tokens": context,
        "batch_size": 1,
        "cache_state": cache_state,
        "mode": mode,
        "pass_id": run_id,
        "ttft_ms": round(ttft, 1),
        "e2e_ms": round(r.e2e_ms, 1),
        "decode_tps": round(r.decode_tps, 1),
        "avg_tps": round(out_tok / (r.e2e_ms / 1000), 1) if r.e2e_ms > 0 else 0,
        "tpot_ms": round(tpot, 1),
        "input_tokens": r.input_tokens,
        "output_tokens": out_tok,
        "prefill_ms": round(ttft, 1),  # Only meaningful for streaming
        "peak_memory_mb": round(r.peak_memory_mb, 1),
        "raw_output": raw[:500],
        "quality_ok": len(struct_issues) == 0,
        "quality_structural": struct_issues,
        "quality_semantic": sem_check,
        "error": r.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _error_record(
    cache_state: str,
    context: int,
    mode: str,
    run_id: str,
    error: str,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Build error measurement record."""
    return {
        "model_id": "",
        "context_tokens": context,
        "batch_size": batch_size,
        "cache_state": cache_state,
        "mode": mode,
        "pass_id": run_id,
        "ttft_ms": 0,
        "e2e_ms": 0,
        "decode_tps": 0,
        "avg_tps": 0,
        "tpot_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "prefill_ms": 0,
        "peak_memory_mb": 0,
        "raw_output": "",
        "quality_ok": False,
        "quality_structural": ["error"],
        "quality_semantic": {},
        "pool_utilization_pct": 0,
        "cache_size_mb": 0,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_result_doc(model_id: str, env: dict[str, str]) -> dict[str, Any]:
    """Create initial result document."""
    return {
        "metadata": {
            "benchmark": "colm_2026_full",
            "git_sha": _git_sha(),
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "timestamp_end": "",
            "machine": {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": platform.python_version(),
            },
            "config": {
                "passes": DEFAULT_PASSES,
                "output_tokens": OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "deterministic": "greedy (T=0, argmax)",
                "cooldown": "adaptive (thermal + TPS probe)",
                "max_cooldown_s": MAX_COOLDOWN_SECONDS,
                "min_cooldown_s": MIN_COOLDOWN_SECONDS,
                "throttle_tps_tolerance": THROTTLE_TPS_TOLERANCE,
            },
        },
        "model_id": model_id,
        "server_env": {k: v for k, v in env.items() if "KEY" not in k},
        "measurements": [],
        "staggered": [],
        "quality_summary": {"total": 0, "passed": 0, "failed": 0},
    }


def save_results(doc: dict[str, Any], path: Path) -> None:
    """Incremental save after every measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=str)


def load_resume(paths: list[Path]) -> set[str]:
    """Load already-completed measurement keys from one or more result files.

    Measurements with ``quality_ok=false`` are excluded so they get re-run
    (e.g. BUG 1 empty-output failures that are now fixed).
    """
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"  Resume file not found (skipped): {path}")
            continue
        with open(path) as f:
            doc = json.load(f)
        n_skip_quality = 0
        for m in doc.get("measurements", []):
            key = _measurement_key(m)
            if not key or m.get("error"):
                continue
            if not m.get("quality_ok", True):
                n_skip_quality += 1
                continue
            completed.add(key)
        for s in doc.get("staggered", []):
            key = f"staggered_{s.get('stagger_mode', '')}_{s.get('pass_id', '')}"
            if not s.get("error"):
                completed.add(key)
        n_total = len(doc.get("measurements", [])) + len(doc.get("staggered", []))
        print(f"  Loaded {path.name}: {n_total} records, "
              f"{n_skip_quality} quality failures will be re-run")
    return completed


def _measurement_key(m: dict) -> str:
    """Unique key for deduplication."""
    return (
        f"{m.get('cache_state', '')}_{m.get('context_tokens', '')}_"
        f"{m.get('mode', '')}_{m.get('batch_size', '')}_{m.get('pass_id', '')}"
    )


def merge_result_files(paths: list[Path], output: Path) -> None:
    """Merge multiple result JSON files into one, keeping best per key.

    When duplicate keys exist, prefers quality_ok=true over quality_ok=false.
    For staggered, prefers entries without errors.
    """
    all_measurements: dict[str, dict] = {}  # key -> best record
    all_staggered: dict[str, dict] = {}
    metadata = None

    for path in paths:
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue
        with open(path) as f:
            doc = json.load(f)
        if metadata is None:
            metadata = doc.get("metadata", {})

        for m in doc.get("measurements", []):
            key = _measurement_key(m)
            if not key:
                continue
            existing = all_measurements.get(key)
            if existing is None:
                all_measurements[key] = m
            else:
                # Prefer quality_ok=true over false, and no-error over error
                new_ok = m.get("quality_ok", False) and not m.get("error")
                old_ok = existing.get("quality_ok", False) and not existing.get("error")
                if new_ok and not old_ok:
                    all_measurements[key] = m

        for s in doc.get("staggered", []):
            key = f"staggered_{s.get('stagger_mode', '')}_{s.get('pass_id', '')}"
            existing = all_staggered.get(key)
            if existing is None:
                all_staggered[key] = s
            elif not s.get("error") and existing.get("error"):
                all_staggered[key] = s

    merged = {
        "metadata": metadata or {},
        "measurements": list(all_measurements.values()),
        "staggered": list(all_staggered.values()),
        "quality_summary": {
            "total": len(all_measurements) + len(all_staggered),
            "passed": sum(1 for m in all_measurements.values()
                         if m.get("quality_ok") and not m.get("error")),
            "failed": sum(1 for m in all_measurements.values()
                         if not m.get("quality_ok") or m.get("error")),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    n_meas = len(merged["measurements"])
    n_stag = len(merged["staggered"])
    n_ok = merged["quality_summary"]["passed"]
    print(f"  Merged {len(paths)} files -> {output.name}: "
          f"{n_meas} measurements + {n_stag} staggered ({n_ok} quality_ok)")


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


async def warmup(base_url: str, corpus: str) -> float:
    """Warmup: single 2K cold request, then calibrate baseline TPS.

    Returns baseline TPS for adaptive cooldown.
    """
    print("  Running warmup (2K cold, discarded)...")

    messages = build_messages(corpus, 2048, offset=0)
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": 16,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    sid = "warmup_discard"
    try:
        client = OpenAIRequestClient(base_url)
        await client.send_and_measure(body, session_id=sid)
        await client.close()
        await _delete_agent(base_url, f"oai_{sid}")
    except Exception as e:
        print(f"  Warmup failed (non-fatal): {e}")
    await clear_all_caches(base_url)
    clean_cache_files()
    print(f"  Stabilizing {WARMUP_SETTLE_SECONDS}s...")
    await asyncio.sleep(WARMUP_SETTLE_SECONDS)

    # Calibrate baseline TPS for adaptive cooldown
    baseline = await calibrate_baseline_tps(base_url, corpus)
    return baseline


# ---------------------------------------------------------------------------
# Paper table output
# ---------------------------------------------------------------------------


def print_table1(measurements: list[dict], model_id: str) -> None:
    """Print Table 1: TTFT (ms) — streaming, batch=1."""
    print(f"\n{'='*70}")
    print(f"Table 1: TTFT (ms) — streaming, batch=1 — {model_id}")
    print(f"{'='*70}")

    # Filter: streaming, batch=1
    data = [
        m for m in measurements
        if m.get("mode") == "streaming"
        and m.get("batch_size") == 1
        and not m.get("error")
    ]

    contexts = sorted(set(m["context_tokens"] for m in data))
    header = f"{'Cache':<8}" + "".join(f"{c:>8}" for c in contexts)
    print(header)
    print("-" * len(header))

    for state in CACHE_STATES:
        row = f"{state:<8}"
        for ctx in contexts:
            vals = [
                m["ttft_ms"]
                for m in data
                if m["cache_state"] == state and m["context_tokens"] == ctx
            ]
            med = statistics.median(vals) if vals else 0
            row += f"{med:>8.0f}"
        print(row)

    # Speedup rows
    for base_state, label in [("warm", "Warm x"), ("hot", "Hot x")]:
        row = f"{label:<8}"
        for ctx in contexts:
            cold_vals = [
                m["ttft_ms"]
                for m in data
                if m["cache_state"] == "cold" and m["context_tokens"] == ctx
            ]
            state_vals = [
                m["ttft_ms"]
                for m in data
                if m["cache_state"] == base_state and m["context_tokens"] == ctx
            ]
            if cold_vals and state_vals:
                ratio = statistics.median(cold_vals) / max(statistics.median(state_vals), 0.1)
                row += f"{ratio:>8.1f}"
            else:
                row += f"{'N/A':>8}"
        print(row)


def print_table2(measurements: list[dict], model_id: str) -> None:
    """Print Table 2: Single request vs concurrent pair — 1K context.

    Both run on the SAME server (scheduler=on, batch_size=2).  "Single"
    means one request at a time; "Concurrent" means two simultaneous
    requests.  This measures the batching benefit without confounding
    server configuration differences.
    """
    print(f"\n{'='*70}")
    print(f"Table 2: Single vs concurrent — 1K cold — {model_id}")
    print(f"{'='*70}")

    b1 = [
        m for m in measurements
        if m.get("batch_size") == 1
        and m.get("context_tokens") == 1024
        and m.get("cache_state") == "cold"
        and m.get("mode") == "non-streaming"
        and not m.get("error")
    ]
    b2 = [
        m for m in measurements
        if m.get("batch_size") == 2
        and m.get("context_tokens") == 1024
        and m.get("cache_state") == "cold"
        and not m.get("error")
    ]

    b1_tps = statistics.median([m["avg_tps"] for m in b1]) if b1 else 0
    b2_sys_tps = statistics.median([m["system_tps"] for m in b2]) if b2 else 0
    b2_per_tps = statistics.median([m["per_request_tps"] for m in b2]) if b2 else 0
    b1_e2e = statistics.median([m["e2e_ms"] for m in b1]) if b1 else 0
    b2_wall = statistics.median([m["wall_ms"] for m in b2]) if b2 else 0

    print(f"{'Metric':<24} {'Single (1)':>12} {'Concurrent (2)':>14}")
    print("-" * 52)
    print(f"{'Per-agent TPS':<24} {b1_tps:>12.1f} {b2_per_tps:>14.1f}")
    print(f"{'System TPS':<24} {b1_tps:>12.1f} {b2_sys_tps:>14.1f}")
    print(f"{'Total time':<24} {b1_e2e/1000:>11.2f}s {b2_wall/1000:>13.2f}s")
    speedup = b1_e2e / b2_wall if b2_wall > 0 else 0
    print(f"{'Speedup':<24} {'1.0x':>12} {speedup:>13.2f}x")


def print_figure3(staggered: list[dict], model_id: str) -> None:
    """Print Figure 3: Staggered arrivals.

    Uses wall-start-relative wait time for User B (``user_b_wait_ms``) to
    provide an apples-to-apples comparison.  In the sequential scenario User B
    cannot start until User A finishes, so its real wait from the scenario
    start is ``A_e2e + B_ttft``.  In batched mode User B arrives after a short
    stagger delay and overlaps with A, so its wait is ``delay + B_ttft``.
    """
    print(f"\n{'='*70}")
    print(f"Figure 3: Staggered arrivals — 4K context — {model_id}")
    print(f"{'='*70}")

    seq = [s for s in staggered if s.get("stagger_mode") == "sequential" and not s.get("error")]
    bat = [s for s in staggered if s.get("stagger_mode") == "batched" and not s.get("error")]

    # Derive user_b_wait_ms from existing fields if not present (backwards compat)
    for group in (seq, bat):
        for s in group:
            if "user_b_wait_ms" not in s:
                delay = s.get("user_b_start_delay_ms", 0)
                ttft = s.get("user_b_ttft_ms", 0)
                s["user_b_wait_ms"] = delay + ttft

    if not seq or not bat:
        print("  (insufficient data)")
        return

    print(f"{'Metric':<30} {'Sequential':>12} {'Batched':>12} {'Speedup':>10}")
    print("-" * 66)

    for label, key in [
        ("User A TTFT", "user_a_ttft_ms"),
        ("User A E2E", "user_a_e2e_ms"),
        ("User B TTFT (own clock)", "user_b_ttft_ms"),
        ("User B wait (wall start)", "user_b_wait_ms"),
        ("User B E2E", "user_b_e2e_ms"),
        ("Total wall", "total_wall_time_ms"),
    ]:
        seq_vals = [s.get(key, 0) for s in seq]
        bat_vals = [s.get(key, 0) for s in bat]
        if not any(seq_vals) and not any(bat_vals):
            # Backwards compat: skip if field missing from old data
            continue
        seq_med = statistics.median(seq_vals)
        bat_med = statistics.median(bat_vals)
        speedup = seq_med / bat_med if bat_med > 0 else 0
        sp_str = f"{speedup:.2f}x" if speedup > 0 else ""
        print(f"{label:<30} {seq_med/1000:>11.1f}s {bat_med/1000:>11.1f}s {sp_str:>10}")

    seq_sys = statistics.median([s["system_tps"] for s in seq])
    bat_sys = statistics.median([s["system_tps"] for s in bat])
    speedup = bat_sys / seq_sys if seq_sys > 0 else 0
    sp_str = f"{speedup:.2f}x" if speedup > 0 else ""
    print(f"{'System TPS':<30} {seq_sys:>11.1f} {bat_sys:>11.1f} {sp_str:>10}")

    print()
    print("Note: 'User B wait (wall start)' = delay + B_TTFT — the fair comparison.")
    print("      Sequential: B must wait for A to finish before starting.")
    print("      Batched: B starts after stagger delay, overlaps with A.")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_model(
    model_key: str,
    model_id: str,
    port: int,
    passes: int,
    contexts: list[int],
    corpus: str,
    resume_keys: set[str],
    result_path: Path,
) -> dict[str, Any]:
    """Run complete benchmark for one model.

    Starts ONE server per model (scheduler on, batch_size=2) and keeps it
    running for all phases.  Between measurements the server stays up — only
    caches are flushed via admin API + filesystem cleanup.  The server is
    stopped at the end so the model is offloaded before the next model starts.
    """

    print(f"\n{'#'*70}")
    print(f"  MODEL: {model_id}")
    print(f"  Passes: {passes}, Contexts: {contexts}")
    print(f"{'#'*70}")

    # Pre-flight memory pressure check
    check_memory_pressure(min_free_pct=15, max_wait=300)

    measurement_index = 0  # For corpus offset diversity

    # -----------------------------------------------------------------------
    # Start ONE server for the entire model benchmark
    # -----------------------------------------------------------------------
    env = build_server_env(model_id, model_key=model_key)
    doc = make_result_doc(model_id, env)

    print(f"\n--- Starting server (scheduler=on, max_batch=2) ---")
    server = ManagedServer(port, env)
    server.start()

    if not await _wait_for_server(server.base_url, timeout=180):
        print("FATAL: Server failed to start")
        server.stop()
        return doc

    baseline_tps = await warmup(server.base_url, corpus)

    measure_funcs = {
        "cold": measure_cold,
        "warm": measure_warm,
        "hot": measure_hot,
    }

    # -----------------------------------------------------------------------
    # BATCH=1 PHASE — single requests on the shared scheduler-enabled server.
    # The scheduler handles one request at a time with no measurable overhead.
    # -----------------------------------------------------------------------
    print(f"\n--- BATCH=1 PHASE (single requests, same server) ---")

    for cache_state in CACHE_STATES:
        for mode in MODES:
            for ctx in sorted(contexts):
                for pass_id in range(passes):
                    run_id = f"p{pass_id}_{int(time.time())}"

                    # Check resume
                    resume_check = f"{cache_state}_{ctx}_{mode}_1_p{pass_id}"
                    if any(k.startswith(resume_check) for k in resume_keys):
                        print(f"  SKIP (resume): {cache_state}/{mode}/{ctx}/pass{pass_id}")
                        continue

                    # Check server health
                    if not await server.ensure_alive_async():
                        print(f"  SKIP (server dead): {cache_state}/{mode}/{ctx}/pass{pass_id}")
                        continue

                    # Log thermal state before measurement
                    thermal_before = get_thermal_state()
                    thermal_name = _THERMAL_NAMES.get(thermal_before, "unknown")
                    if thermal_before >= 1:
                        print(f"  WARNING: thermal={thermal_name} before measurement")

                    label = f"{cache_state}/{mode}/{ctx}tok/pass{pass_id}"
                    print(f"  {label}...", end=" ", flush=True)

                    corpus_offset = measurement_index * 1000
                    measurement_index += 1

                    record = await measure_funcs[cache_state](
                        server.base_url, ctx, OUTPUT_TOKENS, mode,
                        run_id, corpus, corpus_offset,
                    )
                    record["model_id"] = model_id
                    record["thermal_state_before"] = thermal_name

                    # Log thermal state after measurement
                    thermal_after = get_thermal_state()
                    thermal_after_name = _THERMAL_NAMES.get(thermal_after, "unknown")
                    record["thermal_state_after"] = thermal_after_name

                    if record.get("error"):
                        print(f"ERROR: {record['error'][:80]}")
                        doc["quality_summary"]["failed"] += 1

                        # Check if error looks like server crash
                        err = record["error"].lower()
                        if any(w in err for w in ("connect", "refused", "reset", "broken", "peer closed", "incomplete")):
                            if not await server.ensure_alive_async():
                                print("  Server unrecoverable, skipping remaining batch=1")
                                break
                    else:
                        ttft = record.get("ttft_ms", 0)
                        tps = record.get("decode_tps", 0)
                        qok = record.get("quality_ok", False)
                        therm_tag = f" T:{thermal_after_name}" if thermal_after >= 1 else ""
                        print(f"TTFT={ttft:.0f}ms TPS={tps:.1f} Q={'OK' if qok else 'WARN'}{therm_tag}")
                        doc["quality_summary"]["passed"] += 1

                    doc["quality_summary"]["total"] += 1
                    doc["measurements"].append(record)
                    save_results(doc, result_path)

                    # Adaptive cooldown
                    cd_secs = await adaptive_cooldown(
                        server.base_url, corpus, baseline_tps,
                    )
                    record["cooldown_actual_s"] = round(cd_secs, 1)
                    gc.collect()

    # -----------------------------------------------------------------------
    # BATCH=2 PHASE (concurrent requests — same server, no restart)
    # -----------------------------------------------------------------------
    # Skip 32K for batch=2: two simultaneous 32K decode KV caches exceed
    # the memory budget on 24GB M4 Pro.  Chunked interleaved prefill
    # (threshold 2048, default) handles all contexts correctly — Metal
    # crashes fixed by commits 04c814d, 50a4388, b2d5617, 320d25e.
    batch2_contexts = [c for c in contexts if c <= 16384]

    print(f"\n--- BATCH=2 PHASE (concurrent requests, same server) ---")

    for cache_state in CACHE_STATES:
        for mode in MODES:
            for ctx in sorted(batch2_contexts):
                for pass_id in range(passes):
                    run_id = f"p{pass_id}_{int(time.time())}"

                    resume_check = f"{cache_state}_{ctx}_{mode}_2_p{pass_id}"
                    if any(k.startswith(resume_check) for k in resume_keys):
                        print(f"  SKIP (resume): {cache_state}/{mode}/{ctx}/batch2/pass{pass_id}")
                        continue

                    if not await server.ensure_alive_async():
                        print(f"  SKIP (server dead): batch2/{cache_state}/{mode}/{ctx}/pass{pass_id}")
                        continue

                    thermal_before = get_thermal_state()
                    thermal_name = _THERMAL_NAMES.get(thermal_before, "unknown")
                    if thermal_before >= 1:
                        print(f"  WARNING: thermal={thermal_name} before measurement")

                    label = f"batch2/{cache_state}/{mode}/{ctx}tok/pass{pass_id}"
                    print(f"  {label}...", end=" ", flush=True)

                    corpus_offset = measurement_index * 1000
                    measurement_index += 1

                    record = await measure_concurrent(
                        server.base_url, ctx, OUTPUT_TOKENS, mode,
                        cache_state, run_id, corpus, corpus_offset,
                    )
                    record["model_id"] = model_id
                    record["thermal_state_before"] = thermal_name
                    thermal_after = get_thermal_state()
                    thermal_after_name = _THERMAL_NAMES.get(thermal_after, "unknown")
                    record["thermal_state_after"] = thermal_after_name

                    if record.get("error"):
                        print(f"ERROR: {record['error'][:80]}")
                        doc["quality_summary"]["failed"] += 1
                        err = record["error"].lower()
                        if any(w in err for w in ("connect", "refused", "reset", "broken", "peer closed", "incomplete")):
                            if not await server.ensure_alive_async():
                                break
                    else:
                        sys_tps = record.get("system_tps", 0)
                        qok = record.get("quality_ok", False)
                        therm_tag = f" T:{thermal_after_name}" if thermal_after >= 1 else ""
                        print(f"SysTPS={sys_tps:.1f} Q={'OK' if qok else 'WARN'}{therm_tag}")
                        doc["quality_summary"]["passed"] += 1

                    doc["quality_summary"]["total"] += 1
                    doc["measurements"].append(record)
                    save_results(doc, result_path)

                    cd_secs = await adaptive_cooldown(
                        server.base_url, corpus, baseline_tps,
                    )
                    record["cooldown_actual_s"] = round(cd_secs, 1)
                    gc.collect()

    # -----------------------------------------------------------------------
    # STAGGERED PHASE (same server — no restart)
    # -----------------------------------------------------------------------
    print(f"\n--- STAGGERED PHASE (same server) ---")

    for stagger_mode in ["sequential", "batched"]:
        for pass_id in range(passes):
            resume_check = f"staggered_{stagger_mode}_p{pass_id}"
            if any(k.startswith(resume_check) for k in resume_keys):
                print(f"  SKIP (resume): staggered/{stagger_mode}/pass{pass_id}")
                continue

            if not await server.ensure_alive_async():
                print(f"  SKIP (server dead): staggered/{stagger_mode}/pass{pass_id}")
                continue

            await clear_all_caches(server.base_url)
            clean_cache_files()
            await asyncio.sleep(2)

            thermal_before = get_thermal_state()
            thermal_name = _THERMAL_NAMES.get(thermal_before, "unknown")
            if thermal_before >= 1:
                print(f"  WARNING: thermal={thermal_name} before measurement")

            label = f"staggered/{stagger_mode}/pass{pass_id}"
            print(f"  {label}...", end=" ", flush=True)

            try:
                STAGGER_TIMEOUT = 600  # 10 min safety timeout
                stagger_offset = (pass_id + 1) * 10000
                if stagger_mode == "sequential":
                    sr = await asyncio.wait_for(
                        staggered_run_sequential(
                            server.base_url, STAGGER_CONTEXT, OUTPUT_TOKENS, pass_id,
                            corpus, stagger_offset,
                        ),
                        timeout=STAGGER_TIMEOUT,
                    )
                else:
                    sr = await asyncio.wait_for(
                        staggered_run_batched(
                            server.base_url, STAGGER_CONTEXT, OUTPUT_TOKENS,
                            STAGGER_DELAY, pass_id,
                            corpus, stagger_offset,
                        ),
                        timeout=STAGGER_TIMEOUT,
                    )

                stag_record = {
                    "model_id": model_id,
                    "pass_id": f"p{pass_id}",
                    "stagger_mode": stagger_mode,
                    "user_a_ttft_ms": round(sr.user_a_ttft_ms, 1),
                    "user_a_e2e_ms": round(sr.user_a_e2e_ms, 1),
                    "user_b_ttft_ms": round(sr.user_b_ttft_ms, 1),
                    "user_b_e2e_ms": round(sr.user_b_e2e_ms, 1),
                    "user_b_start_delay_ms": round(sr.user_b_start_delay_ms, 1),
                    "user_b_wait_ms": round(sr.user_b_wait_ms, 1),
                    "total_wall_time_ms": round(sr.total_wall_time_ms, 1),
                    "user_a_tps": round(sr.user_a_tps, 1),
                    "user_b_tps": round(sr.user_b_tps, 1),
                    "system_tps": round(sr.system_tps, 1),
                    "thermal_state_before": thermal_name,
                    "error": sr.error or None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                thermal_after = get_thermal_state()
                thermal_after_name = _THERMAL_NAMES.get(thermal_after, "unknown")
                stag_record["thermal_state_after"] = thermal_after_name

                if sr.error:
                    print(f"ERROR: {sr.error[:80]}")
                else:
                    wall_s = sr.total_wall_time_ms / 1000
                    therm_tag = f" T:{thermal_after_name}" if thermal_after >= 1 else ""
                    print(f"Wall={wall_s:.1f}s SysTPS={sr.system_tps:.1f}{therm_tag}")

            except Exception as e:
                print(f"ERROR: {e}")
                stag_record = {
                    "model_id": model_id,
                    "pass_id": f"p{pass_id}",
                    "stagger_mode": stagger_mode,
                    "thermal_state_before": thermal_name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                err_str = str(e).lower()
                if any(w in err_str for w in ("connect", "refused", "reset", "broken", "peer closed", "incomplete")):
                    if not await server.ensure_alive_async():
                        break

            doc["staggered"].append(stag_record)
            save_results(doc, result_path)

            cd_secs = await adaptive_cooldown(
                server.base_url, corpus, baseline_tps,
            )
            stag_record["cooldown_actual_s"] = round(cd_secs, 1)
            gc.collect()

    # -----------------------------------------------------------------------
    # Stop server — offloads model, releases GPU memory for next model
    # -----------------------------------------------------------------------
    print(f"\n  Stopping server (offloading {model_id})...")
    server.stop()

    # -----------------------------------------------------------------------
    # Print paper tables
    # -----------------------------------------------------------------------
    doc["metadata"]["timestamp_end"] = datetime.now(timezone.utc).isoformat()
    save_results(doc, result_path)

    print_table1(doc["measurements"], model_id)
    print_table2(doc["measurements"], model_id)
    print_figure3(doc["staggered"], model_id)

    qs = doc["quality_summary"]
    print(f"\nQuality: {qs['passed']}/{qs['total']} passed, {qs['failed']} failed")

    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> int:
    global MAX_COOLDOWN_SECONDS

    parser = argparse.ArgumentParser(
        description="COLM 2026 full benchmark suite"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 pass, contexts 1K/4K/16K",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        choices=["all", "gemma", "deepseek"],
        help="Which model(s) to benchmark",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint JSON file(s), comma-separated",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--passes", type=int, default=None,
        help=f"Override pass count (default: {DEFAULT_PASSES})",
    )
    parser.add_argument(
        "--cooldown", type=int, default=None,
        help=f"Override max cooldown seconds (default: {MAX_COOLDOWN_SECONDS})",
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=None,
        help="Override context sizes (e.g., --contexts 4096)",
    )
    parser.add_argument(
        "--merge", type=str, default=None,
        help="Merge result files (comma-separated) into one. "
             "Use with --output to set destination path.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for --merge",
    )
    args = parser.parse_args()

    # Handle --merge mode (no server needed)
    if args.merge:
        merge_paths = [Path(p.strip()) for p in args.merge.split(",")]
        out = Path(args.output) if args.output else RESULTS_DIR / "colm_full_merged.json"
        print(f"Merging {len(merge_paths)} result files...")
        merge_result_files(merge_paths, out)
        print(f"Output: {out}")
        return 0

    # Apply overrides
    if args.cooldown is not None:
        MAX_COOLDOWN_SECONDS = args.cooldown
    passes = args.passes if args.passes is not None else (1 if args.quick else DEFAULT_PASSES)
    if args.contexts:
        contexts = sorted(args.contexts)
    elif args.quick:
        contexts = QUICK_CONTEXTS
    else:
        contexts = ALL_CONTEXTS

    # Select models
    if args.models == "all":
        model_list = list(MODELS.items())
    else:
        model_list = [(args.models, MODELS[args.models])]

    # Load resume state (accepts comma-separated paths)
    resume_keys: set[str] = set()
    if args.resume:
        resume_paths = [Path(p.strip()) for p in args.resume.split(",")]
        resume_keys = load_resume(resume_paths)
        print(f"Resuming: {len(resume_keys)} measurements already completed")

    # Load corpus
    print("Loading prefill corpus...")
    corpus = load_corpus()
    print(f"  Corpus size: {len(corpus):,} chars ({len(corpus)//4:,} est. tokens)")

    # Phase 0: preparation
    print("\n--- PHASE 0: PREPARATION ---")
    kill_all_servers()
    clean_cache_files()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_start = time.time()

    for model_key, model_id in model_list:
        model_slug = model_key
        result_path = RESULTS_DIR / f"colm_full_{model_slug}_{timestamp}.json"

        await run_model(
            model_key, model_id, args.port, passes, contexts,
            corpus, resume_keys, result_path,
        )

        print(f"\nResults saved: {result_path}")

        # Between models: server.stop() already offloaded the model via the
        # shutdown path (batch_engine.shutdown + model_registry.unload_model).
        # Brief cooldown + memory check before loading next model.
        if len(model_list) > 1 and model_key != model_list[-1][0]:
            print(f"\nModel offloaded. Cooldown 60s before next model...")
            clean_cache_files()
            gc.collect()
            await asyncio.sleep(60)
            check_memory_pressure(min_free_pct=15, max_wait=120)

    elapsed = time.time() - total_start
    hours = elapsed / 3600
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE — {hours:.1f} hours total")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        kill_all_servers()
        sys.exit(1)
