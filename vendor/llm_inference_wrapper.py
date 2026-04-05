#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Interactive NDJSON wrapper around stock TRT Edge-LLM llm_inference binary.

Bridges agent-memory's TRTSubprocessAdapter (NDJSON over stdin/stdout)
and the stock llm_inference binary (JSON file I/O).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def send_json(obj: dict) -> None:
    """Write NDJSON line to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def get_model_spec(engine_dir: Path) -> dict:
    """Read model spec from engine config.json."""
    config = json.loads((engine_dir / "config.json").read_text())
    return {
        "n_layers": config.get("num_hidden_layers", 0),
        "n_kv_heads": config.get("num_key_value_heads", 0),
        "head_dim": config.get("head_dim", 0),
        "block_tokens": 256,
        "vocab_size": config.get("vocab_size", 0),
    }


def run_generate(binary: Path, engine_dir: Path, cmd: dict, work: Path) -> dict:
    """Run stock llm_inference for a generate command."""
    messages = cmd.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": cmd.get("text", "Hello")}]

    input_data = {
        "batch_size": 1,
        "temperature": cmd.get("temperature", 0.7),
        "top_p": cmd.get("top_p", 0.95),
        "top_k": cmd.get("top_k", 40),
        "max_generate_length": cmd.get("max_tokens", 256),
        "requests": [{"messages": messages}],
    }

    input_path = work / "input.json"
    output_path = work / "output.json"
    input_path.write_text(json.dumps(input_data))

    try:
        result = subprocess.run(  # noqa: S603
            [
                str(binary),
                "--engineDir",
                str(engine_dir),
                "--inputFile",
                str(input_path),
                "--outputFile",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Generation timed out"}

    if result.returncode != 0:
        return {"error": f"Binary failed (rc={result.returncode}): {result.stderr[-500:]}"}

    try:
        output = json.loads(output_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": f"Failed to read output: {e}"}

    responses = output.get("responses", [])
    if not responses:
        return {"error": "No responses in output"}

    resp = responses[0]
    return {
        "text": resp.get("output_text", ""),
        "tokens": resp.get("output_token_ids", []),
        "finish_reason": "stop",
    }


def find_binary(script_dir: Path) -> Path | None:
    """Locate stock llm_inference binary relative to script."""
    edgellm_bin = "TensorRT-Edge-LLM" / Path("build/examples/llm/llm_inference")
    candidates = [
        script_dir / edgellm_bin,
        script_dir.parent / "vendor" / edgellm_bin,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def handle_commands(binary: Path, engine_dir: Path) -> None:
    """Process NDJSON commands from stdin."""
    work = Path(tempfile.mkdtemp(prefix="edgellm_"))
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                send_json({"error": f"Invalid JSON: {line}"})
                continue

            action = cmd.get("cmd", "")
            if action == "get_model_spec":
                send_json(get_model_spec(engine_dir))
            elif action == "generate":
                send_json(run_generate(binary, engine_dir, cmd, work))
            elif action == "shutdown":
                send_json({"status": "shutdown"})
                break
            else:
                send_json({"error": f"Unknown command: {action}"})
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    """Entry point: parse args, find binary, run command loop."""
    parser = argparse.ArgumentParser(description="NDJSON wrapper for llm_inference")
    parser.add_argument("--engineDir", "--engine-path", required=True, dest="engine_dir")
    parser.add_argument("--binaryPath", default=None)
    parser.add_argument("--mode", default="interactive")
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)
    script_dir = Path(__file__).resolve().parent
    binary = Path(args.binaryPath) if args.binaryPath else find_binary(script_dir)

    if not binary or not binary.is_file():
        send_json({"status": "error", "error": f"Binary not found: {binary}"})
        return

    send_json({"status": "ready"})
    handle_commands(binary, engine_dir)


if __name__ == "__main__":
    main()
