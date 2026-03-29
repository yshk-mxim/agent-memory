#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Remove debug fprintf statements added during development."""

import sys
from pathlib import Path


def remove_debug(filepath: str) -> None:
    path = Path(filepath)
    src = path.read_text()
    lines = src.split("\n")
    cleaned = [line for line in lines if "[PLUGIN] FMHA:" not in line and "[DEBUG]" not in line]
    if len(cleaned) < len(lines):
        path.write_text("\n".join(cleaned))
        print(f"Removed {len(lines) - len(cleaned)} debug lines from {filepath}")
    else:
        print(f"No debug lines found in {filepath}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        remove_debug(f)
