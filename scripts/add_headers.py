#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Add SPDX copyright headers to all Python files in the repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER_LINES = [
    "# SPDX-License-Identifier: MIT\n",
    "# Copyright (c) 2026 Yakov Shkolnikov and contributors\n",
]

SPDX_MARKER = "SPDX-License-Identifier"

DEFAULT_DIRS = ["src/", "tests/", "benchmarks/", "demo/", "scripts/"]


def _has_header(path: Path) -> bool:
    """Return True if the file already contains the SPDX marker in the first 5 lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                if SPDX_MARKER in line:
                    return True
    except (UnicodeDecodeError, OSError):
        return True  # skip files we cannot read
    return False


def _add_header(path: Path) -> None:
    """Insert the SPDX header into *path*, respecting a shebang if present."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines(keepends=True)

    # Determine insertion point: after shebang line if present.
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    new_lines = lines[:insert_at] + HEADER_LINES + lines[insert_at:]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _collect_py_files(directories: list[str]) -> list[Path]:
    """Recursively collect all .py files under the given directories."""
    repo_root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    for d in directories:
        dir_path = repo_root / d
        if not dir_path.is_dir():
            continue
        files.extend(sorted(dir_path.rglob("*.py")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add SPDX copyright headers to Python files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Check which files are missing headers (exit 1 if any).",
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="Add headers to files that are missing them.",
    )
    parser.add_argument(
        "directories",
        nargs="*",
        default=DEFAULT_DIRS,
        help="Directories to scan (default: src/ tests/ benchmarks/ demo/ scripts/).",
    )

    args = parser.parse_args()

    py_files = _collect_py_files(args.directories)
    missing: list[Path] = [p for p in py_files if not _has_header(p)]

    if args.check:
        if missing:
            print(f"Files missing SPDX header ({len(missing)}):")
            for p in missing:
                print(f"  {p}")
        print(f"\n{len(py_files)} files checked, {len(missing)} missing headers.")
        return 1 if missing else 0

    # --apply mode
    for p in missing:
        _add_header(p)

    print(f"{len(py_files)} files checked, {len(missing)} headers added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
