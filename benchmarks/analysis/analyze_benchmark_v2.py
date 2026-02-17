#!/usr/bin/env python3
"""Analyze benchmark results and produce comprehensive statistics."""

import json
import statistics
from collections import defaultdict
from typing import Dict, List, Any

def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute median, min, max for a list of values."""
    if not values:
        return {"median": 0, "min": 0, "max": 0}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

def analyze_measurements(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze all measurements and compute comprehensive statistics."""
    measurements = data.get("measurements", [])

    # Group measurements by (context_tokens, cache_state, mode, batch_size)
    grouped = defaultdict(lambda: {
        "ttft": [],
        "decode_tps": [],
        "errors": [],
        "quality_failures": [],
        "e2e": [],  # For batch=2
        "per_request_tps": []  # For batch=2
    })

    total_measurements = len(measurements)
    quality_pass_count = 0
    error_count = 0

    for m in measurements:
        key = (
            m.get("context_tokens"),
            m.get("cache_state"),
            m.get("mode"),
            m.get("batch_size")
        )

        # Track quality and errors
        if m.get("quality_ok"):
            quality_pass_count += 1
        else:
            grouped[key]["quality_failures"].append(m)

        if m.get("error"):
            error_count += 1
            grouped[key]["errors"].append(m)
            continue

        # Handle batch=1 format (individual request)
        if m.get("batch_size") == 1:
            ttft = m.get("ttft_ms")
            if ttft is not None and ttft > 0:  # Only include streaming (TTFT > 0)
                grouped[key]["ttft"].append(ttft)

            decode_tps = m.get("decode_tps")
            if decode_tps is not None:
                grouped[key]["decode_tps"].append(decode_tps)

        # Handle batch=2 format (aggregated metrics)
        elif m.get("batch_size") == 2:
            avg_ttft = m.get("avg_ttft_ms")
            if avg_ttft is not None and avg_ttft > 0:  # streaming
                grouped[key]["ttft"].append(avg_ttft)

            avg_e2e = m.get("avg_e2e_ms")
            if avg_e2e is not None:
                grouped[key]["e2e"].append(avg_e2e)

            # For batch=2, use per_request_tps (tokens per second per request)
            per_req_tps = m.get("per_request_tps")
            if per_req_tps is not None:
                grouped[key]["decode_tps"].append(per_req_tps)

    # Compute statistics for each group
    stats = {}
    for key, values in grouped.items():
        context_tokens, cache_state, mode, batch_size = key
        stats[key] = {
            "ttft": compute_stats(values["ttft"]),
            "decode_tps": compute_stats(values["decode_tps"]),
            "count": len(values["ttft"]) if values["ttft"] else len(values["decode_tps"]),
            "errors": values["errors"],
            "quality_failures": values["quality_failures"]
        }

    # Compute speedup ratios (warm/cold, hot/cold)
    speedup_data = defaultdict(dict)
    for (context, cache, mode, batch), stat in stats.items():
        if cache == "cold":
            cold_key = (context, mode, batch)
            speedup_data[cold_key]["cold_ttft"] = stat["ttft"]["median"]
            speedup_data[cold_key]["cold_decode_tps"] = stat["decode_tps"]["median"]
        elif cache == "warm":
            warm_key = (context, mode, batch)
            speedup_data[warm_key]["warm_ttft"] = stat["ttft"]["median"]
        elif cache == "hot":
            hot_key = (context, mode, batch)
            speedup_data[hot_key]["hot_ttft"] = stat["ttft"]["median"]

    # Calculate speedup ratios
    for key, data in speedup_data.items():
        if "cold_ttft" in data and data["cold_ttft"] > 0:
            if "warm_ttft" in data and data["warm_ttft"] > 0:
                data["warm_speedup"] = data["cold_ttft"] / data["warm_ttft"]
            if "hot_ttft" in data and data["hot_ttft"] > 0:
                data["hot_speedup"] = data["cold_ttft"] / data["hot_ttft"]

    return {
        "total_measurements": total_measurements,
        "quality_pass_count": quality_pass_count,
        "quality_pass_rate": quality_pass_count / total_measurements if total_measurements > 0 else 0,
        "error_count": error_count,
        "stats": stats,
        "speedup_data": speedup_data,
    }

def format_table_streaming(stats: Dict, speedup_data: Dict, batch_size: int) -> str:
    """Format streaming results as a table for given batch_size."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append(f"BATCH SIZE = {batch_size}, STREAMING")
    lines.append("="*100)

    contexts = [1024, 2048, 4096]
    cache_states = ["cold", "warm", "hot"]

    lines.append("\nTTFT (ms):")
    lines.append("-" * 100)
    lines.append(f"{'Context':<12} {'Cold Median':<15} {'Cold Min-Max':<20} {'Warm Median':<15} {'Warm Min-Max':<20}")
    lines.append(f"{'Tokens':<12} {'(ms)':<15} {'(ms)':<20} {'(ms)':<15} {'(ms)':<20}")
    lines.append("-" * 100)

    for context in contexts:
        row_data = {}
        for cache in cache_states:
            key = (context, cache, "streaming", batch_size)
            if key in stats:
                stat = stats[key]
                row_data[cache] = stat

        cold = row_data.get("cold")
        warm = row_data.get("warm")

        if cold:
            cold_med = cold["ttft"]["median"]
            cold_range = f"{cold['ttft']['min']:.1f}-{cold['ttft']['max']:.1f}"

            if warm:
                warm_med = warm["ttft"]["median"]
                warm_range = f"{warm['ttft']['min']:.1f}-{warm['ttft']['max']:.1f}"
            else:
                warm_med = 0
                warm_range = "N/A"

            lines.append(f"{context:<12} {cold_med:<15.1f} {cold_range:<20} {warm_med:<15.1f} {warm_range:<20}")

    lines.append("\n")
    lines.append(f"{'Context':<12} {'Hot Median':<15} {'Hot Min-Max':<20} {'Speedup':<30}")
    lines.append(f"{'Tokens':<12} {'(ms)':<15} {'(ms)':<20} {'(warm/cold, hot/cold)':<30}")
    lines.append("-" * 100)

    for context in contexts:
        speedup_key = (context, "streaming", batch_size)
        speedup = speedup_data.get(speedup_key, {})

        hot_key = (context, "hot", "streaming", batch_size)
        hot = stats.get(hot_key)

        if hot:
            hot_med = hot["ttft"]["median"]
            hot_range = f"{hot['ttft']['min']:.1f}-{hot['ttft']['max']:.1f}"

            warm_sp = speedup.get("warm_speedup", 0)
            hot_sp = speedup.get("hot_speedup", 0)
            speedup_str = f"{warm_sp:.2f}x, {hot_sp:.2f}x" if warm_sp > 0 and hot_sp > 0 else "N/A"

            lines.append(f"{context:<12} {hot_med:<15.1f} {hot_range:<20} {speedup_str:<30}")

    # Decode TPS section
    lines.append("\n\nDecode TPS (tokens/sec):")
    if batch_size == 2:
        lines.append("(Per-request TPS: system_tps / 2)")
    lines.append("-" * 100)
    lines.append(f"{'Context':<12} {'Cold Median':<15} {'Cold Min-Max':<20} {'Warm Median':<15} {'Warm Min-Max':<20}")
    lines.append(f"{'Tokens':<12} {'(tps)':<15} {'(tps)':<20} {'(tps)':<15} {'(tps)':<20}")
    lines.append("-" * 100)

    for context in contexts:
        row_data = {}
        for cache in cache_states:
            key = (context, cache, "streaming", batch_size)
            if key in stats:
                stat = stats[key]
                row_data[cache] = stat

        cold = row_data.get("cold")
        warm = row_data.get("warm")

        if cold:
            cold_med = cold["decode_tps"]["median"]
            cold_range = f"{cold['decode_tps']['min']:.1f}-{cold['decode_tps']['max']:.1f}"

            warm_med = warm["decode_tps"]["median"] if warm else 0
            warm_range = f"{warm['decode_tps']['min']:.1f}-{warm['decode_tps']['max']:.1f}" if warm else "N/A"

            lines.append(f"{context:<12} {cold_med:<15.1f} {cold_range:<20} {warm_med:<15.1f} {warm_range:<20}")

    lines.append("\n")
    lines.append(f"{'Context':<12} {'Hot Median':<15} {'Hot Min-Max':<20}")
    lines.append(f"{'Tokens':<12} {'(tps)':<15} {'(tps)':<20}")
    lines.append("-" * 100)

    for context in contexts:
        hot_key = (context, "hot", "streaming", batch_size)
        hot = stats.get(hot_key)

        if hot:
            hot_med = hot["decode_tps"]["median"]
            hot_range = f"{hot['decode_tps']['min']:.1f}-{hot['decode_tps']['max']:.1f}"
            lines.append(f"{context:<12} {hot_med:<15.1f} {hot_range:<20}")

    return "\n".join(lines)

def format_non_streaming_tables(stats: Dict) -> str:
    """Format non-streaming results for both batch sizes."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("NON-STREAMING RESULTS (TTFT=0, only E2E and decode TPS)")
    lines.append("="*100)

    for batch in [1, 2]:
        lines.append(f"\nBATCH SIZE = {batch}, NON-STREAMING")
        if batch == 2:
            lines.append("(Per-request TPS: system_tps / 2)")
        lines.append("-" * 100)
        lines.append(f"{'Context':<12} {'Cache':<10} {'Decode TPS Median':<20} {'Decode TPS Min-Max':<25} {'Count':<10}")
        lines.append(f"{'Tokens':<12} {'State':<10} {'(tps)':<20} {'(tps)':<25} {'(n)':<10}")
        lines.append("-" * 100)

        contexts = [1024, 2048, 4096]
        cache_states = ["cold", "warm", "hot"]

        for context in contexts:
            for cache in cache_states:
                key = (context, cache, "non-streaming", batch)
                if key in stats:
                    stat = stats[key]
                    med = stat["decode_tps"]["median"]
                    range_str = f"{stat['decode_tps']['min']:.1f}-{stat['decode_tps']['max']:.1f}"
                    count = stat["count"]
                    lines.append(f"{context:<12} {cache:<10} {med:<20.1f} {range_str:<25} {count:<10}")

    return "\n".join(lines)

def format_errors_and_failures(stats: Dict) -> str:
    """Format errors and quality failures."""
    lines = []

    # Collect all errors and failures
    all_errors = []
    all_failures = []

    for key, stat in stats.items():
        all_errors.extend(stat["errors"])
        all_failures.extend(stat["quality_failures"])

    if all_errors:
        lines.append("\n" + "="*100)
        lines.append("ERRORS")
        lines.append("="*100)
        for err in all_errors:
            lines.append(f"\nContext: {err.get('context_tokens')}, Cache: {err.get('cache_state')}, "
                        f"Batch: {err.get('batch_size')}, Mode: {err.get('mode')}")
            lines.append(f"Error: {err.get('error')}")
    else:
        lines.append("\n(No errors found)")

    if all_failures:
        lines.append("\n" + "="*100)
        lines.append("QUALITY FAILURES")
        lines.append("="*100)
        for fail in all_failures:
            lines.append(f"\nContext: {fail.get('context_tokens')}, Cache: {fail.get('cache_state')}, "
                        f"Batch: {fail.get('batch_size')}, Mode: {fail.get('mode')}")
            lines.append(f"Quality issues: {fail.get('quality_structural', [])} / {fail.get('quality_semantic', {})}")
            lines.append(f"Output: {fail.get('raw_output', '')[:100]}...")
    else:
        lines.append("\n(No quality failures found)")

    return "\n".join(lines)

def main():
    filepath = "/Users/dev_user/agent-memory/benchmarks/results/full_gemma_20260215_232610.json"

    print("Loading benchmark results...")
    data = load_results(filepath)

    print("Analyzing measurements...")
    analysis = analyze_measurements(data)

    # Print summary
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"Model: {data.get('model_id', 'Unknown')}")
    print(f"Git SHA: {data['metadata']['git_sha']}")
    print(f"Timestamp: {data['metadata']['timestamp_start']} to {data['metadata']['timestamp_end']}")
    print(f"Total measurements: {analysis['total_measurements']}")
    print(f"Quality pass count: {analysis['quality_pass_count']}")
    print(f"Quality pass rate: {analysis['quality_pass_rate']*100:.2f}%")
    print(f"Error count: {analysis['error_count']}")

    # Print tables
    print(format_table_streaming(analysis['stats'], analysis['speedup_data'], batch_size=1))
    print(format_table_streaming(analysis['stats'], analysis['speedup_data'], batch_size=2))
    print(format_non_streaming_tables(analysis['stats']))
    print(format_errors_and_failures(analysis['stats']))

    print("\n" + "="*100)
    print("Analysis complete.")
    print("="*100)

if __name__ == "__main__":
    main()
