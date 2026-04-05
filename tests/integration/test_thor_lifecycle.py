#!/usr/bin/env python3
"""Thor lifecycle test: cold start, warm cache, model swap 26B→31B→26B.

Run manually: python tests/integration/test_thor_lifecycle.py
Requires: Thor server running at http://main4.local:8000
"""

import json
import sys
import time

import httpx

BASE = "http://main4.local:8000"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "test-key",
    "anthropic-version": "2023-06-01",
}

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. You help users "
    "with coding tasks including debugging, refactoring, writing tests, "
    "and implementing new features.\n\n"
    "# Tool Quick Reference\n\n"
    "## File Operations\n"
    "- Read: file_path (required, absolute), offset (start line), limit (max lines)\n"
    "- Write: file_path (required, absolute), content (required, full file)\n"
    "- Edit: file_path (required), old_string (required), new_string (required)\n"
    "- Glob: pattern (required, e.g. \"**/*.py\"), path (optional directory)\n"
    "- Grep: pattern (required, regex), path, glob, output_mode\n\n"
    "## Execution\n"
    "- Bash: command (required), timeout (ms, max 600000)\n"
    "- Agent: prompt (required), description (required, 3-5 words)\n\n"
    "# Project Context\n"
    "This is a Python project using FastAPI, Pydantic, and pytest.\n"
    "Follow hexagonal architecture: ports in application layer.\n\n"
) * 5  # ~2500 tokens of system prompt


def send_msg(
    user_msg: str,
    session_id: str,
    model: str = "gemma-4-26b-a4b",
    stream: bool = False,
    max_tokens: int = 64,
) -> tuple[float, float, str]:
    """Send message, return (ttft_ms, total_ms, response_text)."""
    hdrs = {**HEADERS, "X-Session-ID": session_id}
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
        "stream": stream,
    }

    t0 = time.perf_counter()

    if stream:
        ttft_ms = None
        full_text = ""
        with httpx.stream(
            "POST", f"{BASE}/v1/messages", json=body, headers=hdrs, timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        parsed = json.loads(data)
                        if parsed.get("type") == "content_block_delta":
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            full_text += parsed.get("delta", {}).get("text", "")
        total_ms = (time.perf_counter() - t0) * 1000
        return ttft_ms or total_ms, total_ms, full_text
    else:
        r = httpx.post(
            f"{BASE}/v1/messages", json=body, headers=hdrs, timeout=120.0,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        resp_json = r.json()
        text = ""
        for block in resp_json.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return total_ms, total_ms, text


def swap_model(model_id: str) -> None:
    """Swap to a different model via the admin API."""
    print(f"\n  Swapping to {model_id}...")
    t0 = time.perf_counter()
    admin_headers = {
        **HEADERS,
        "X-Admin-Key": "thor-admin-key",
    }
    r = httpx.post(
        f"{BASE}/admin/models/swap",
        json={"model_id": model_id},
        headers=admin_headers,
        timeout=300.0,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    print(f"  Swap complete in {elapsed:.1f}s: {r.json()}")


def test_health():
    print("\n=== 1. Health check ===")
    r = httpx.get(f"{BASE}/health/live", timeout=5)
    assert r.status_code == 200
    print(f"  OK: {r.status_code}")

    r = httpx.get(f"{BASE}/v1/models", timeout=5)
    models = r.json()["data"]
    active = [m for m in models if m.get("active")]
    print(f"  Active model: {active[0]['id'] if active else 'none'}")
    print(f"  Available: {[m['id'] for m in models]}")


def test_cold_vs_warm_same_session():
    print("\n=== 2. Cold vs warm (same session, non-streaming) ===")
    session = "thor-same-session-ns"

    ttft1, total1, text1 = send_msg("What is 2+2?", session)
    print(f"  Cold: {total1:.0f}ms — {text1[:80]}")

    ttft2, total2, text2 = send_msg("What is 3+3?", session)
    print(f"  Warm: {total2:.0f}ms — {text2[:80]}")

    ttft3, total3, text3 = send_msg("What is 4+4?", session)
    print(f"  Warm2: {total3:.0f}ms — {text3[:80]}")

    avg_warm = (total2 + total3) / 2
    speedup = total1 / avg_warm if avg_warm > 0 else 0
    print(f"  Speedup: {speedup:.2f}x (cold/avg_warm)")
    return speedup


def test_cold_vs_warm_streaming():
    print("\n=== 3. Cold vs warm (same session, streaming TTFT) ===")
    session = "thor-same-session-st"

    ttft1, total1, text1 = send_msg("What is 2+2?", session, stream=True)
    print(f"  Cold TTFT: {ttft1:.0f}ms  total: {total1:.0f}ms")

    ttft2, total2, text2 = send_msg("What is 3+3?", session, stream=True)
    print(f"  Warm TTFT: {ttft2:.0f}ms  total: {total2:.0f}ms")

    ttft3, total3, text3 = send_msg("What is 4+4?", session, stream=True)
    print(f"  Warm TTFT: {ttft3:.0f}ms  total: {total3:.0f}ms")

    avg_ttft = (ttft2 + ttft3) / 2
    speedup = ttft1 / avg_ttft if avg_ttft > 0 else 0
    print(f"  TTFT speedup: {speedup:.2f}x")
    return speedup


def test_cross_session():
    print("\n=== 4. Cross-session prefix cache ===")

    ttft_a, _, text_a = send_msg("What is 2+2?", "thor-cross-a", stream=True)
    print(f"  Session A (cold) TTFT: {ttft_a:.0f}ms")

    ttft_b, _, text_b = send_msg("What is 3+3?", "thor-cross-b", stream=True)
    print(f"  Session B (prefix hit) TTFT: {ttft_b:.0f}ms")

    ttft_c, _, text_c = send_msg("What is 4+4?", "thor-cross-c", stream=True)
    print(f"  Session C (prefix hit) TTFT: {ttft_c:.0f}ms")

    avg_warm = (ttft_b + ttft_c) / 2
    speedup = ttft_a / avg_warm if avg_warm > 0 else 0
    print(f"  Cross-session TTFT speedup: {speedup:.2f}x")
    return speedup


def test_model_swap_26b_to_31b():
    print("\n=== 5. Model swap: 26B → 31B ===")
    swap_model("gemma-4-31b")

    # Verify model swapped
    r = httpx.get(f"{BASE}/v1/models", timeout=5)
    active = [m for m in r.json()["data"] if m.get("active")]
    assert active[0]["id"] == "gemma-4-31b", f"Expected gemma-4-31b, got {active}"
    print(f"  Active: {active[0]['id']}")

    # Cold request on 31B
    ttft1, total1, text1 = send_msg(
        "What is 2+2?", "thor-31b-session", model="gemma-4-31b", stream=True,
    )
    print(f"  31B cold TTFT: {ttft1:.0f}ms  total: {total1:.0f}ms")
    print(f"  31B response: {text1[:100]}")

    # Warm on 31B
    ttft2, total2, text2 = send_msg(
        "What is 3+3?", "thor-31b-session", model="gemma-4-31b", stream=True,
    )
    print(f"  31B warm TTFT: {ttft2:.0f}ms  total: {total2:.0f}ms")
    speedup = ttft1 / ttft2 if ttft2 > 0 else 0
    print(f"  31B TTFT speedup: {speedup:.2f}x")
    return speedup


def test_model_swap_back_to_26b():
    print("\n=== 6. Model swap back: 31B → 26B ===")
    swap_model("gemma-4-26b-a4b")

    r = httpx.get(f"{BASE}/v1/models", timeout=5)
    active = [m for m in r.json()["data"] if m.get("active")]
    assert active[0]["id"] == "gemma-4-26b-a4b"
    print(f"  Active: {active[0]['id']}")

    # Cold after swap back (slot restore should help)
    ttft1, total1, text1 = send_msg(
        "What is 2+2?", "thor-26b-restored", stream=True,
    )
    print(f"  26B post-swap cold TTFT: {ttft1:.0f}ms  total: {total1:.0f}ms")

    # Warm
    ttft2, total2, text2 = send_msg(
        "What is 3+3?", "thor-26b-restored", stream=True,
    )
    print(f"  26B post-swap warm TTFT: {ttft2:.0f}ms  total: {total2:.0f}ms")
    speedup = ttft1 / ttft2 if ttft2 > 0 else 0
    print(f"  26B post-swap speedup: {speedup:.2f}x")
    return speedup


def test_slot_files_on_disk():
    """Check that slot files were saved after swap."""
    print("\n=== 7. Slot file persistence check ===")
    import subprocess
    result = subprocess.run(
        ["ssh", "localhost", "ls -la ~/.agent_memory/llamacpp_slots/"],
        capture_output=True, text=True, timeout=10,
    )
    print(f"  {result.stdout.strip()}")
    return result.stdout


if __name__ == "__main__":
    try:
        test_health()
        ns_speedup = test_cold_vs_warm_same_session()
        st_speedup = test_cold_vs_warm_streaming()
        cross_speedup = test_cross_session()

        swap_26b_speedup = test_model_swap_26b_to_31b()
        swap_back_speedup = test_model_swap_back_to_26b()
        slot_files = test_slot_files_on_disk()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Same-session (non-streaming): {ns_speedup:.2f}x")
        print(f"  Same-session (streaming TTFT): {st_speedup:.2f}x")
        print(f"  Cross-session (streaming TTFT): {cross_speedup:.2f}x")
        print(f"  31B cold→warm: {swap_26b_speedup:.2f}x")
        print(f"  26B restored cold→warm: {swap_back_speedup:.2f}x")
        print(f"  Slot files on disk: {'yes' if '.bin' in slot_files else 'no'}")
        print("=" * 60)

    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
