#!/usr/bin/env python3
"""Generate paper-ready statistics from benchmark JSON file."""

import json
import statistics
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def load_and_analyze(filepath: str) -> Tuple[Dict, Dict]:
    """Load benchmark results and compute all statistics."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    measurements = data.get("measurements", [])

    # Group by (context, cache, mode, batch)
    grouped = defaultdict(lambda: {"ttft": [], "decode_tps": []})

    for m in measurements:
        if m.get("error"):
            continue

        key = (
            m.get("context_tokens"),
            m.get("cache_state"),
            m.get("mode"),
            m.get("batch_size")
        )

        if m.get("batch_size") == 1:
            ttft = m.get("ttft_ms")
            if ttft and ttft > 0:
                grouped[key]["ttft"].append(ttft)
            tps = m.get("decode_tps")
            if tps:
                grouped[key]["decode_tps"].append(tps)
        elif m.get("batch_size") == 2:
            avg_ttft = m.get("avg_ttft_ms")
            if avg_ttft and avg_ttft > 0:
                grouped[key]["ttft"].append(avg_ttft)
            per_req_tps = m.get("per_request_tps")
            if per_req_tps:
                grouped[key]["decode_tps"].append(per_req_tps)

    # Compute statistics
    stats = {}
    for key, values in grouped.items():
        stats[key] = {
            "ttft_median": statistics.median(values["ttft"]) if values["ttft"] else 0,
            "ttft_min": min(values["ttft"]) if values["ttft"] else 0,
            "ttft_max": max(values["ttft"]) if values["ttft"] else 0,
            "tps_median": statistics.median(values["decode_tps"]) if values["decode_tps"] else 0,
            "tps_min": min(values["decode_tps"]) if values["decode_tps"] else 0,
            "tps_max": max(values["decode_tps"]) if values["decode_tps"] else 0,
            "count": len(values["ttft"]) if values["ttft"] else len(values["decode_tps"])
        }

    # Compute speedups
    speedups = {}
    for context in [1024, 2048, 4096]:
        for mode in ["streaming"]:
            for batch in [1, 2]:
                cold_key = (context, "cold", mode, batch)
                warm_key = (context, "warm", mode, batch)
                hot_key = (context, "hot", mode, batch)

                if cold_key in stats:
                    cold_ttft = stats[cold_key]["ttft_median"]
                    warm_ttft = stats.get(warm_key, {}).get("ttft_median", 0)
                    hot_ttft = stats.get(hot_key, {}).get("ttft_median", 0)

                    speedup_key = (context, mode, batch)
                    speedups[speedup_key] = {
                        "warm_speedup": cold_ttft / warm_ttft if warm_ttft > 0 else 0,
                        "hot_speedup": cold_ttft / hot_ttft if hot_ttft > 0 else 0
                    }

    return stats, speedups

def print_paper_table1(stats: Dict, speedups: Dict):
    """Table 1: Single Request TTFT (Batch=1, Streaming)."""
    print("\n" + "="*80)
    print("TABLE 1: Single Request TTFT Performance (Batch=1, Streaming)")
    print("="*80)
    print(f"{'Context':<10} {'Cold':<12} {'Warm':<12} {'Hot':<12} {'Warm':<12} {'Hot':<12}")
    print(f"{'(tokens)':<10} {'TTFT (ms)':<12} {'TTFT (ms)':<12} {'TTFT (ms)':<12} {'Speedup':<12} {'Speedup':<12}")
    print("-"*80)

    for context in [1024, 2048, 4096]:
        cold = stats[(context, "cold", "streaming", 1)]
        warm = stats[(context, "warm", "streaming", 1)]
        hot = stats[(context, "hot", "streaming", 1)]
        sp = speedups[(context, "streaming", 1)]

        print(f"{context:<10} {cold['ttft_median']:<12.0f} {warm['ttft_median']:<12.0f} "
              f"{hot['ttft_median']:<12.0f} {sp['warm_speedup']:<12.1f}x {sp['hot_speedup']:<12.1f}x")

def print_paper_table2(stats: Dict, speedups: Dict):
    """Table 2: Concurrent Request TTFT (Batch=2, Streaming)."""
    print("\n" + "="*80)
    print("TABLE 2: Concurrent Request TTFT Performance (Batch=2, Streaming)")
    print("="*80)
    print(f"{'Context':<10} {'Cold':<12} {'Warm':<12} {'Hot':<12} {'Warm':<12} {'Hot':<12}")
    print(f"{'(tokens)':<10} {'TTFT (ms)':<12} {'TTFT (ms)':<12} {'TTFT (ms)':<12} {'Speedup':<12} {'Speedup':<12}")
    print("-"*80)

    for context in [1024, 2048, 4096]:
        cold = stats[(context, "cold", "streaming", 2)]
        warm = stats[(context, "warm", "streaming", 2)]
        hot = stats[(context, "hot", "streaming", 2)]
        sp = speedups[(context, "streaming", 2)]

        print(f"{context:<10} {cold['ttft_median']:<12.0f} {warm['ttft_median']:<12.0f} "
              f"{hot['ttft_median']:<12.0f} {sp['warm_speedup']:<12.1f}x {sp['hot_speedup']:<12.1f}x")

def print_paper_table3(stats: Dict):
    """Table 3: Decode Throughput (Batch=1, Streaming)."""
    print("\n" + "="*80)
    print("TABLE 3: Decode Throughput (Batch=1, Streaming)")
    print("="*80)
    print(f"{'Context':<10} {'Cold':<12} {'Warm':<12} {'Hot':<12}")
    print(f"{'(tokens)':<10} {'TPS':<12} {'TPS':<12} {'TPS':<12}")
    print("-"*80)

    for context in [1024, 2048, 4096]:
        cold = stats[(context, "cold", "streaming", 1)]
        warm = stats[(context, "warm", "streaming", 1)]
        hot = stats[(context, "hot", "streaming", 1)]

        print(f"{context:<10} {cold['tps_median']:<12.1f} {warm['tps_median']:<12.1f} {hot['tps_median']:<12.1f}")

def print_latex_values(stats: Dict, speedups: Dict):
    """Print individual LaTeX-ready values for inline citation."""
    print("\n" + "="*80)
    print("LATEX INLINE VALUES (for \\newcommand definitions)")
    print("="*80)

    # Single request speedups
    sp_1k = speedups[(1024, "streaming", 1)]
    sp_2k = speedups[(2048, "streaming", 1)]
    sp_4k = speedups[(4096, "streaming", 1)]

    print("\n% Single request (batch=1) warm cache speedups")
    print(f"\\newcommand{{\\warmSpeedupOneK}}{{{sp_1k['warm_speedup']:.1f}}}  % 1K tokens")
    print(f"\\newcommand{{\\warmSpeedupTwoK}}{{{sp_2k['warm_speedup']:.1f}}}  % 2K tokens")
    print(f"\\newcommand{{\\warmSpeedupFourK}}{{{sp_4k['warm_speedup']:.1f}}} % 4K tokens")

    # Concurrent request speedups
    sp_1k_b2 = speedups[(1024, "streaming", 2)]
    sp_2k_b2 = speedups[(2048, "streaming", 2)]
    sp_4k_b2 = speedups[(4096, "streaming", 2)]

    print("\n% Concurrent requests (batch=2) warm cache speedups")
    print(f"\\newcommand{{\\warmSpeedupConcOneK}}{{{sp_1k_b2['warm_speedup']:.1f}}}  % 1K tokens")
    print(f"\\newcommand{{\\warmSpeedupConcTwoK}}{{{sp_2k_b2['warm_speedup']:.1f}}}  % 2K tokens")
    print(f"\\newcommand{{\\warmSpeedupConcFourK}}{{{sp_4k_b2['warm_speedup']:.1f}}} % 4K tokens")

    # TTFT absolute values
    cold_1k = stats[(1024, "cold", "streaming", 1)]["ttft_median"]
    warm_1k = stats[(1024, "warm", "streaming", 1)]["ttft_median"]
    cold_4k = stats[(4096, "cold", "streaming", 1)]["ttft_median"]
    warm_4k = stats[(4096, "warm", "streaming", 1)]["ttft_median"]

    print("\n% Absolute TTFT values (ms)")
    print(f"\\newcommand{{\\ttftColdOneK}}{{{cold_1k:.0f}}}   % 1K cold")
    print(f"\\newcommand{{\\ttftWarmOneK}}{{{warm_1k:.0f}}}   % 1K warm")
    print(f"\\newcommand{{\\ttftColdFourK}}{{{cold_4k:.0f}}}  % 4K cold")
    print(f"\\newcommand{{\\ttftWarmFourK}}{{{warm_4k:.0f}}}  % 4K warm")

    # Decode TPS values
    cold_tps = stats[(1024, "cold", "streaming", 1)]["tps_median"]
    warm_tps = stats[(1024, "warm", "streaming", 1)]["tps_median"]

    print("\n% Decode throughput (tps)")
    print(f"\\newcommand{{\\tpsCold}}{{{cold_tps:.1f}}}  % Cold cache")
    print(f"\\newcommand{{\\tpsWarm}}{{{warm_tps:.1f}}}  % Warm cache")

def print_summary_stats(filepath: str):
    """Print high-level summary statistics."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    measurements = data.get("measurements", [])
    quality_pass = sum(1 for m in measurements if m.get("quality_ok"))
    errors = sum(1 for m in measurements if m.get("error"))

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Model: {data.get('model_id', 'Unknown')}")
    print(f"Git SHA: {data['metadata']['git_sha']}")
    print(f"Start: {data['metadata']['timestamp_start']}")
    print(f"End: {data['metadata']['timestamp_end']}")
    print(f"Total measurements: {len(measurements)}")
    print(f"Quality pass: {quality_pass} ({quality_pass/len(measurements)*100:.1f}%)")
    print(f"Errors: {errors} ({errors/len(measurements)*100:.1f}%)")

def main():
    filepath = "./benchmarks/results/full_gemma_20260215_232610.json"

    print_summary_stats(filepath)

    print("\nAnalyzing measurements...")
    stats, speedups = load_and_analyze(filepath)

    print_paper_table1(stats, speedups)
    print_paper_table2(stats, speedups)
    print_paper_table3(stats)
    print_latex_values(stats, speedups)

    print("\n" + "="*80)
    print("COMPLETE - Ready for paper integration")
    print("="*80)

if __name__ == "__main__":
    main()
