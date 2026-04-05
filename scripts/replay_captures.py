#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay captured llama.cpp traffic through the parser chain.

Reads JSONL capture files (produced by SEMANTIC_LLAMACPP_CAPTURE_TRAFFIC)
and runs each raw model output through the full parsing pipeline:
  1. strip_thinking_tags (channel markers, thinking tags)
  2. tool call parser chain (Gemma native, JSON, ReAct)
  3. parameter normalization

Reports parsing failures and generates pytest fixture code for
entries that contain tool calls.

Usage:
    python scripts/replay_captures.py captures.jsonl
    python scripts/replay_captures.py captures.jsonl --gen-fixtures
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory.adapters.outbound.tool_call_parsers import create_parser_for_model
from agent_memory.application.text_cleaning import strip_thinking_tags


def replay(capture_path: str, gen_fixtures: bool = False) -> None:
    entries = []
    with open(capture_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} captures from {capture_path}\n")

    failures = 0
    fixture_code = []

    for i, entry in enumerate(entries):
        model_id = entry.get("model_id", "unknown")
        raw = entry.get("raw_content", "")
        raw_tool_calls = entry.get("raw_tool_calls")

        # Step 1: strip tags
        cleaned = strip_thinking_tags(raw)

        # Step 2: parse tool calls
        parser = create_parser_for_model(model_id)
        remaining, parsed = parser.parse(cleaned)

        # Check for channel marker leaks
        has_channel_leak = "<|channel>" in cleaned or "<channel|>" in cleaned
        has_thought_leak = any(
            tag in cleaned
            for tag in ("<start_of_thought>", "<end_of_thought>", "<think>", "</think>")
        )

        status = "OK"
        if has_channel_leak:
            status = "CHANNEL_LEAK"
            failures += 1
        elif has_thought_leak:
            status = "THINKING_LEAK"
            failures += 1

        # Report
        tc_summary = ", ".join(tc.name for tc in parsed) if parsed else "none"
        native = "native" if raw_tool_calls else "text"
        print(
            f"[{i:3d}] {status:15s} | model={model_id:20s} | "
            f"tool_calls({native})=[{tc_summary}] | "
            f"raw={len(raw):5d} chars | cleaned={len(cleaned):5d} chars"
        )

        if has_channel_leak or has_thought_leak:
            print(f"      RAW: {raw[:200]!r}")
            print(f"      CLEANED: {cleaned[:200]!r}")

        # Generate fixture
        if gen_fixtures and (parsed or raw_tool_calls):
            safe_raw = json.dumps(raw)
            fixture_code.append(
                f"def test_capture_{i}():\n"
                f'    """Auto-generated from capture {i} ({model_id})."""\n'
                f"    raw = {safe_raw}\n"
                f"    cleaned = strip_thinking_tags(raw)\n"
                f'    assert "<|channel>" not in cleaned\n'
                f'    assert "<channel|>" not in cleaned\n'
                f"    parser = create_parser_for_model({model_id!r})\n"
                f"    remaining, calls = parser.parse(cleaned)\n"
                f"    assert len(calls) == {len(parsed)}\n"
            )
            for j, tc in enumerate(parsed):
                fixture_code.append(
                    f'    assert calls[{j}].name == {tc.name!r}\n'
                )
            fixture_code.append("\n")

    print(f"\n{'='*60}")
    print(f"Total: {len(entries)} | Passed: {len(entries) - failures} | Failed: {failures}")

    if gen_fixtures and fixture_code:
        out = Path(capture_path).with_suffix(".test.py")
        with open(out, "w") as f:
            f.write("# Auto-generated parser regression tests from traffic captures\n")
            f.write("from agent_memory.adapters.outbound.tool_call_parsers import create_parser_for_model\n")
            f.write("from agent_memory.application.text_cleaning import strip_thinking_tags\n\n\n")
            f.write("".join(fixture_code))
        print(f"\nFixtures written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture_file", help="JSONL capture file")
    ap.add_argument("--gen-fixtures", action="store_true", help="Generate pytest fixtures")
    args = ap.parse_args()
    replay(args.capture_file, args.gen_fixtures)
