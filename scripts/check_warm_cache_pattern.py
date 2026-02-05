#!/usr/bin/env python3
"""Pre-commit hook to detect broken warm cache test patterns.

Checks for common anti-patterns that break warm cache testing:
1. Using same session_id for prime and measure without eviction
2. Missing evict_only parameter in warm test DELETE calls
3. No sleep/wait between evict and measure

Usage:
    python scripts/check_warm_cache_pattern.py benchmarks/streaming_benchmark.py
"""

import argparse
import re
import sys
from pathlib import Path


class WarmCachePatternChecker:
    """Detects anti-patterns in warm cache test code."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_file(self, path: Path) -> bool:
        """Check file for warm cache anti-patterns.

        Returns:
            True if no errors found, False otherwise
        """
        if not path.exists():
            self.errors.append(f"{path}: File not found")
            return False

        content = path.read_text()
        lines = content.split("\n")

        # Check for warm test functions
        in_warm_function = False
        function_start_line = 0
        session_id_var = None
        has_evict_only = False
        has_sleep_after_evict = False

        for i, line in enumerate(lines, 1):
            # Detect warm test function
            if re.search(r"async def \w+_warm\(", line):
                in_warm_function = True
                function_start_line = i
                session_id_var = None
                has_evict_only = False
                has_sleep_after_evict = False
                continue

            if not in_warm_function:
                continue

            # End of function
            if line and not line[0].isspace() and i > function_start_line:
                # Validate function before reset
                if in_warm_function and session_id_var:
                    if not has_evict_only:
                        self.errors.append(
                            f"{path}:{function_start_line}: Warm test missing "
                            f"evict_only=True in DELETE call"
                        )
                    if not has_sleep_after_evict:
                        self.warnings.append(
                            f"{path}:{function_start_line}: Warm test missing "
                            f"sleep after evict (may cause race condition)"
                        )

                in_warm_function = False
                continue

            # Check for session ID assignment
            sid_match = re.search(r'sid = ["\'](\w+)', line)
            if sid_match:
                session_id_var = sid_match.group(1)

            # Check for DELETE with evict_only
            if re.search(r"_delete_agent.*evict_only\s*=\s*True", line):
                has_evict_only = True

            # Check for sleep after evict
            if has_evict_only and re.search(r"await asyncio\.sleep\(", line):
                has_sleep_after_evict = True

        return len(self.errors) == 0

    def report(self) -> int:
        """Print report and return exit code.

        Returns:
            0 if no errors, 1 if errors found
        """
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  {error}")
            print("\nWarm cache anti-patterns detected!")
            print("\nCommon fixes:")
            print("  1. Use evict_only=True in DELETE between prime and measure")
            print("  2. Add await asyncio.sleep(1.0) after evict")
            print("  3. Ensure prime and measure use same session_id")
            return 1

        print("✅ No warm cache anti-patterns detected")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check for warm cache anti-patterns"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Files to check",
    )
    args = parser.parse_args()

    checker = WarmCachePatternChecker()
    for path in args.files:
        checker.check_file(path)

    return checker.report()


if __name__ == "__main__":
    sys.exit(main())
