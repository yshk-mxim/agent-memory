#!/usr/bin/env python3
"""Warm cache validation script.

Run after benchmarks to ensure warm cache is actually working.
Fails CI if warm TTFT ≈ cold TTFT (indicating broken warm cache test).

Usage:
    python benchmarks/validate_warm_cache.py benchmarks/results/streaming_*.json
    python benchmarks/validate_warm_cache.py benchmarks/results/streaming_*.json --min-speedup 1.5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class WarmCacheValidator:
    """Validates that warm cache provides speedup over cold start."""

    def __init__(self, min_speedup: float = 1.5):
        """Initialize validator.

        Args:
            min_speedup: Minimum required speedup (warm TTFT / cold TTFT).
                        Default 1.5x means warm must be at least 50% faster.
        """
        self.min_speedup = min_speedup
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def validate_file(self, path: Path) -> bool:
        """Validate a single benchmark file.

        Returns:
            True if validation passed, False otherwise
        """
        with open(path) as f:
            data = json.load(f)

        benchmark = data.get("benchmark", "unknown")
        results = data.get("results", [])

        if not results:
            self.warnings.append(f"{path.name}: No results found")
            return True

        # Group by context length and cache state
        by_context: dict[int, dict[str, list[float]]] = {}
        for r in results:
            if "error" in r:
                continue
            ctx = r.get("context_tokens", 0)
            state = r.get("cache_state", "unknown")
            ttft = r.get("ttft_ms", 0)

            if ctx not in by_context:
                by_context[ctx] = {}
            if state not in by_context[ctx]:
                by_context[ctx][state] = []
            by_context[ctx][state].append(ttft)

        # Validate each context length
        all_passed = True
        for ctx in sorted(by_context.keys()):
            states = by_context[ctx]
            if "cold" not in states or "warm" not in states:
                continue

            cold_avg = sum(states["cold"]) / len(states["cold"])
            warm_avg = sum(states["warm"]) / len(states["warm"])
            speedup = cold_avg / warm_avg if warm_avg > 0 else 0

            if speedup < self.min_speedup:
                self.failures.append(
                    f"{path.name}: {ctx} tokens - warm speedup {speedup:.2f}x "
                    f"< {self.min_speedup}x (cold={cold_avg:.0f}ms, warm={warm_avg:.0f}ms)"
                )
                all_passed = False

            # Warn if warm > cold (worse than cold start)
            if warm_avg > cold_avg * 1.1:
                self.warnings.append(
                    f"{path.name}: {ctx} tokens - warm SLOWER than cold "
                    f"(cold={cold_avg:.0f}ms, warm={warm_avg:.0f}ms)"
                )

        return all_passed

    def report(self) -> int:
        """Print validation report and return exit code.

        Returns:
            0 if all validations passed, 1 if any failed
        """
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.failures:
            print("\n❌ VALIDATION FAILED:")
            for failure in self.failures:
                print(f"  {failure}")
            print(
                f"\nWarm cache MUST be at least {self.min_speedup}x faster than cold start."
            )
            print("\nPossible causes:")
            print("  1. Benchmark uses same session_id for prime and measure (hits hot cache)")
            print("  2. Agent not evicted between prime and measure (stays in memory)")
            print("  3. Disk write not flushed before measure (cache not persisted)")
            print("\nFix: Use evict_only=true in DELETE between prime and measure")
            return 1

        print("\n✅ WARM CACHE VALIDATION PASSED")
        print(f"   All warm cache speedups >= {self.min_speedup}x")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate warm cache benchmark results"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Benchmark result JSON files to validate",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.5,
        help="Minimum required speedup (default: 1.5x)",
    )
    args = parser.parse_args()

    validator = WarmCacheValidator(min_speedup=args.min_speedup)

    for path in args.files:
        if not path.exists():
            print(f"⚠️  File not found: {path}")
            continue

        print(f"Validating {path.name}...")
        validator.validate_file(path)

    return validator.report()


if __name__ == "__main__":
    sys.exit(main())
