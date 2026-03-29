#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Fake llm_inference binary for unit tests.

Standalone script mimicking the TRT llm_inference NDJSON protocol.
Used by unit tests as a subprocess replacement — no CUDA/TRT needed.

Protocol:
    1. On start, prints: {"status": "ready"}
    2. Reads NDJSON commands from stdin, writes responses to stdout.
    3. Supports: generate, get_model_spec, shutdown
"""

import json
import os
import sys
import tempfile

import numpy as np

# Model geometry — overridable via env vars for integration tests
N_LAYERS = int(os.environ.get("FAKE_N_LAYERS", "30"))
N_KV_HEADS = int(os.environ.get("FAKE_N_KV_HEADS", "3"))
HEAD_DIM = int(os.environ.get("FAKE_HEAD_DIM", "64"))


def _make_fake_cache(seq_len: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create fake FP16 KV cache."""
    rng = np.random.default_rng(0)
    cache = []
    for _ in range(N_LAYERS):
        k = rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)).astype(np.float16)
        cache.append((k, v))
    return cache


def _save_cache_to_shm(cache: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Save cache to temp safetensors file and return path."""
    from safetensors.numpy import save_file

    tensors = {}
    for layer_idx, (k, v) in enumerate(cache):
        tensors[f"L{layer_idx}_K"] = k
        tensors[f"L{layer_idx}_V"] = v

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file(tensors, path)
    return path


def handle_command(cmd: dict) -> dict:
    """Process a single command and return response."""
    action = cmd.get("cmd")

    if action == "generate":
        tokens = cmd.get("tokens", [])
        max_tokens = cmd.get("max_tokens", 10)

        # Fake generation: echo back some tokens
        gen_tokens = list(range(100, 100 + max_tokens))
        text = f"[fake output for {len(tokens)} input tokens]"

        # Create and save a fake updated cache
        seq_len = len(tokens) + max_tokens
        cache = _make_fake_cache(seq_len)
        cache_path = _save_cache_to_shm(cache)

        return {
            "text": text,
            "tokens": gen_tokens,
            "finish_reason": "stop",
            "kv_cache_path": cache_path,
        }

    if action == "inject_cache":
        # Fake inject: just acknowledge with seq_len from the file
        input_path = cmd.get("input_path", "")
        seq_len = 64  # Default fake seq_len
        if input_path:
            try:
                from safetensors.numpy import load_file

                tensors = load_file(input_path)
                if "L0_K" in tensors:
                    seq_len = tensors["L0_K"].shape[1]
            except Exception:  # noqa: S110
                pass
        return {"status": "ok", "seq_len": seq_len}

    if action == "extract_cache":
        output_path = cmd.get("output_path", "/tmp/fake_kv.safetensors")
        cache = _make_fake_cache(64)
        from safetensors.numpy import save_file as sf

        tensors = {}
        for layer_idx, (k, v) in enumerate(cache):
            tensors[f"L{layer_idx}_K"] = k
            tensors[f"L{layer_idx}_V"] = v
        sf(tensors, output_path)
        return {"kv_cache_path": output_path, "seq_len": 64}

    if action == "get_model_spec":
        return {
            "n_layers": N_LAYERS,
            "n_kv_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_tokens": 256,
        }

    if action == "shutdown":
        return {"status": "shutdown"}

    return {"error": f"Unknown command: {action}"}


def main() -> None:
    """Main loop: emit ready, then process commands."""
    # Signal ready
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            resp = {"error": f"Invalid JSON: {line}"}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_command(cmd)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

        if cmd.get("cmd") == "shutdown":
            break


if __name__ == "__main__":
    main()
