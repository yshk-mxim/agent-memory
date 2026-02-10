#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Automated code quality checks for the agent-memory codebase.

Scans Python files for patterns that indicate code quality issues,
stale branding, placeholder code, and other common problems.

Usage:
    python scripts/code_quality_check.py [directories...]

Default directory: src/agent_memory/

Output: JSON report to stdout, summary table to stderr.
Exit code: 1 if any CRITICAL findings, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator


class Severity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Finding:
    file: str
    line: int
    severity: str
    check: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        file: str,
        line: int,
        severity: Severity,
        check: str,
        message: str,
    ) -> None:
        self.findings.append(
            Finding(
                file=file,
                line=line,
                severity=severity.value,
                check=check,
                message=message,
            )
        )

    def to_dict(self) -> dict:
        summary = {s.value: 0 for s in Severity}
        for f in self.findings:
            summary[f.severity] += 1
        return {
            "summary": summary,
            "total": len(self.findings),
            "findings": [
                {
                    "file": f.file,
                    "line": f.line,
                    "severity": f.severity,
                    "check": f.check,
                    "message": f.message,
                }
                for f in self.findings
            ],
        }

    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL.value for f in self.findings)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_python_files(directories: list[str]) -> list[Path]:
    """Recursively collect all .py files from the given directories."""
    files: list[Path] = []
    for directory in directories:
        root = Path(directory)
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        if not root.is_dir():
            print(f"Warning: {directory} is not a valid directory", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_file():
                files.append(path)
    return files


# ---------------------------------------------------------------------------
# Check 1: Numbered comments
# ---------------------------------------------------------------------------

_NUMBERED_COMMENT_RE = re.compile(
    r"#\s*(?:"
    r"\d+\.\s"          # "# 1. ", "# 2. "
    r"|Step\s+\d+"      # "# Step 1", "# Step 2"
    r")",
    re.IGNORECASE,
)


def check_numbered_comments(filepath: str, lines: list[str], report: Report) -> None:
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#") and _NUMBERED_COMMENT_RE.search(stripped):
            report.add(
                file=filepath,
                line=i,
                severity=Severity.WARNING,
                check="numbered-comment",
                message=f"Numbered comment: {stripped.rstrip()}",
            )


# ---------------------------------------------------------------------------
# Check 2: Docstring longer than body
# ---------------------------------------------------------------------------

def _count_docstring_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return the number of lines in the function's docstring, or 0."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        ds_node = node.body[0]
        # Use end_lineno if available (Python 3.8+), else estimate from string
        if hasattr(ds_node, "end_lineno") and ds_node.end_lineno is not None:
            return ds_node.end_lineno - ds_node.lineno + 1
        # Fallback: count newlines in the string value
        text = ds_node.value.value
        return text.count("\n") + 1
    return 0


def _function_body_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count non-docstring body lines of a function (from AST line range)."""
    if not node.body:
        return 0
    end = getattr(node, "end_lineno", None)
    if end is None:
        return 0

    # Total lines of the function body (excluding the def line itself)
    first_body = node.body[0]
    total_body_lines = end - first_body.lineno + 1

    ds_lines = _count_docstring_lines(node)
    return max(total_body_lines - ds_lines, 0)


def check_docstring_longer_than_body(
    filepath: str, tree: ast.Module, report: Report
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ds_lines = _count_docstring_lines(node)
        if ds_lines == 0:
            continue
        body_lines = _function_body_lines(node)
        if body_lines == 0:
            continue
        if ds_lines > body_lines:
            report.add(
                file=filepath,
                line=node.lineno,
                severity=Severity.INFO,
                check="docstring-longer-than-body",
                message=(
                    f"Function '{node.name}' has docstring ({ds_lines} lines) "
                    f"longer than body ({body_lines} lines)"
                ),
            )


# ---------------------------------------------------------------------------
# Check 3: Sprint/ticket references
# ---------------------------------------------------------------------------

_SPRINT_TICKET_RE = re.compile(
    r"(?:"
    r"Sprint\s+\d+"
    r"|Day\s+\d+"
    r"|NEW-\d+"
    r"|CRITICAL-\d+"
    r")"
)


def check_sprint_ticket_refs(filepath: str, lines: list[str], report: Report) -> None:
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # Only check comments and docstrings (lines starting with # or inside strings)
        if stripped.startswith("#"):
            match = _SPRINT_TICKET_RE.search(stripped)
            if match:
                report.add(
                    file=filepath,
                    line=i,
                    severity=Severity.WARNING,
                    check="sprint-ticket-ref",
                    message=f"Sprint/ticket reference in comment: {match.group()}",
                )
        elif '"""' in stripped or "'''" in stripped:
            match = _SPRINT_TICKET_RE.search(stripped)
            if match:
                report.add(
                    file=filepath,
                    line=i,
                    severity=Severity.WARNING,
                    check="sprint-ticket-ref",
                    message=f"Sprint/ticket reference in docstring: {match.group()}",
                )


# ---------------------------------------------------------------------------
# Check 4: Generic variable names
# ---------------------------------------------------------------------------

_GENERIC_NAMES = {"data", "result", "temp", "item", "obj"}

# Matches standalone assignments like "data = ..." but not "self.data =",
# "result_list =", "my_data =", etc.
_GENERIC_VAR_RE = re.compile(
    r"(?<![.\w])"           # Not preceded by dot or word char
    r"(?P<name>"
    + "|".join(_GENERIC_NAMES)
    + r")"
    r"\s*=[^=]"             # Assignment (not ==)
)


def check_generic_variable_names(
    filepath: str, lines: list[str], report: Report
) -> None:
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # Skip comments and blank lines
        if stripped.startswith("#") or not stripped:
            continue
        # Skip lines that are purely string literals (docstrings)
        if stripped.startswith(('"""', "'''", '"', "'")):
            continue
        match = _GENERIC_VAR_RE.search(stripped)
        if match:
            name = match.group("name")
            report.add(
                file=filepath,
                line=i,
                severity=Severity.INFO,
                check="generic-variable-name",
                message=f"Generic variable name '{name}' used as standalone assignment",
            )


# ---------------------------------------------------------------------------
# Check 5: Placeholder code
# ---------------------------------------------------------------------------

def check_placeholder_code(
    filepath: str, tree: ast.Module, lines: list[str], report: Report
) -> None:
    # 5a: `pass` in non-abstract, non-Protocol methods
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Check if the function is inside a Protocol or is abstract
        is_abstract = any(
            (isinstance(d, ast.Name) and d.id in ("abstractmethod", "abstractproperty"))
            or (isinstance(d, ast.Attribute) and d.attr in ("abstractmethod", "abstractproperty"))
            for d in node.decorator_list
        )
        if is_abstract:
            continue

        # Check if the function body is just `pass` (or docstring + pass)
        body_stmts = node.body
        non_docstring = body_stmts
        if (
            body_stmts
            and isinstance(body_stmts[0], ast.Expr)
            and isinstance(body_stmts[0].value, ast.Constant)
            and isinstance(body_stmts[0].value.value, str)
        ):
            non_docstring = body_stmts[1:]

        if len(non_docstring) == 1 and isinstance(non_docstring[0], ast.Pass):
            # Check if the containing class is a Protocol
            is_protocol = _is_inside_protocol(tree, node)
            if not is_protocol:
                report.add(
                    file=filepath,
                    line=non_docstring[0].lineno,
                    severity=Severity.WARNING,
                    check="placeholder-pass",
                    message=f"'pass' in non-abstract method '{node.name}'",
                )

    # 5b: NotImplementedError("Sprint...")
    for i, line in enumerate(lines, start=1):
        if 'NotImplementedError' in line and 'Sprint' in line:
            report.add(
                file=filepath,
                line=i,
                severity=Severity.WARNING,
                check="placeholder-not-implemented",
                message="NotImplementedError with Sprint reference",
            )


def _is_inside_protocol(tree: ast.Module, func_node: ast.AST) -> bool:
    """Check if a function node is defined inside a Protocol class."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if the class inherits from Protocol
        is_protocol = any(
            (isinstance(base, ast.Name) and base.id == "Protocol")
            or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
            for base in node.bases
        )
        if is_protocol:
            for child in ast.walk(node):
                if child is func_node:
                    return True
    return False


# ---------------------------------------------------------------------------
# Check 6: Silent exception swallowing
# ---------------------------------------------------------------------------

def check_silent_exception_swallowing(
    filepath: str, tree: ast.Module, report: Report
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Bare except (type is None) or except Exception
        is_bare = node.type is None
        is_exception = (
            isinstance(node.type, ast.Name) and node.type.id == "Exception"
        )
        if not (is_bare or is_exception):
            continue

        # Check if the handler body is just pass or continue
        body = node.body
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                kind = "bare except" if is_bare else "except Exception"
                report.add(
                    file=filepath,
                    line=node.lineno,
                    severity=Severity.WARNING,
                    check="silent-exception-swallow",
                    message=f"Silent exception swallowing: {kind} followed by pass",
                )
            elif isinstance(stmt, ast.Continue):
                kind = "bare except" if is_bare else "except Exception"
                report.add(
                    file=filepath,
                    line=node.lineno,
                    severity=Severity.WARNING,
                    check="silent-exception-swallow",
                    message=f"Silent exception swallowing: {kind} followed by continue",
                )


# ---------------------------------------------------------------------------
# Check 7: Runtime imports (imports inside function bodies)
# ---------------------------------------------------------------------------

def check_runtime_imports(filepath: str, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                modules = []
                if isinstance(child, ast.Import):
                    modules = [alias.name for alias in child.names]
                else:
                    modules = [child.module or ""]
                report.add(
                    file=filepath,
                    line=child.lineno,
                    severity=Severity.INFO,
                    check="runtime-import",
                    message=(
                        f"Import inside function '{node.name}': "
                        f"{', '.join(modules)}"
                    ),
                )


# ---------------------------------------------------------------------------
# Check 8: God methods (functions > 50 lines)
# ---------------------------------------------------------------------------

_GOD_METHOD_THRESHOLD = 50


def check_god_methods(filepath: str, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        length = end - node.lineno + 1
        if length > _GOD_METHOD_THRESHOLD:
            report.add(
                file=filepath,
                line=node.lineno,
                severity=Severity.INFO,
                check="god-method",
                message=(
                    f"Function '{node.name}' is {length} lines "
                    f"(threshold: {_GOD_METHOD_THRESHOLD})"
                ),
            )


# ---------------------------------------------------------------------------
# Check 9: Stale branding
# ---------------------------------------------------------------------------

# Patterns that indicate stale "semantic" branding that should be "agent_memory"
_STALE_BRANDING_PATTERNS = [
    (re.compile(r"\bfrom\s+semantic\."), "from semantic."),
    (re.compile(r"\bimport\s+semantic\."), "import semantic."),
    (re.compile(r"Semantic\s+Team\b"), "Semantic Team"),
    (re.compile(r"\bsemantic-server\b"), "semantic-server"),
    (re.compile(r"\bsrc/semantic/"), "src/semantic/"),
]

# Patterns to EXCLUDE from stale branding (env vars, etc.)
_STALE_BRANDING_EXCLUDE = re.compile(
    r"(?:"
    r"SEMANTIC_MLX_"
    r"|SEMANTIC_ADMIN_"
    r"|SEMANTIC_ENABLE_"
    r"|SEMANTIC_LOG_"
    r"|SEMANTIC_DEBUG"
    r"|['\"]SEMANTIC_"       # Quoted env var names
    r"|os\.environ.*SEMANTIC_"
    r"|os\.getenv.*SEMANTIC_"
    r"|settings\.semantic"   # Settings attribute access
    r")"
)


def check_stale_branding(filepath: str, lines: list[str], report: Report) -> None:
    for i, line in enumerate(lines, start=1):
        # Skip lines that reference env vars
        if _STALE_BRANDING_EXCLUDE.search(line):
            continue
        for pattern, label in _STALE_BRANDING_PATTERNS:
            if pattern.search(line):
                report.add(
                    file=filepath,
                    line=i,
                    severity=Severity.CRITICAL,
                    check="stale-branding",
                    message=f"Stale branding '{label}' found: {line.strip()[:120]}",
                )
                break  # One finding per line is enough


# ---------------------------------------------------------------------------
# Main analysis driver
# ---------------------------------------------------------------------------

def analyze_file(filepath: Path, report: Report) -> None:
    """Run all checks on a single Python file."""
    rel = str(filepath)
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Warning: cannot read {filepath}: {e}", file=sys.stderr)
        return

    lines = source.splitlines()

    # Parse AST (some checks need it)
    tree = None
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        # If we can't parse, skip AST-based checks but still run line-based ones
        pass

    # Line-based checks (always run)
    check_numbered_comments(rel, lines, report)
    check_sprint_ticket_refs(rel, lines, report)
    check_generic_variable_names(rel, lines, report)
    check_stale_branding(rel, lines, report)

    # AST-based checks (only if parsing succeeded)
    if tree is not None:
        check_docstring_longer_than_body(rel, tree, report)
        check_placeholder_code(rel, tree, lines, report)
        check_silent_exception_swallowing(rel, tree, report)
        check_runtime_imports(rel, tree, report)
        check_god_methods(rel, tree, report)


def print_summary_table(report_dict: dict) -> None:
    """Print a human-readable summary table to stderr."""
    summary = report_dict["summary"]
    total = report_dict["total"]
    findings = report_dict["findings"]

    print("\n" + "=" * 70, file=sys.stderr)
    print("  CODE QUALITY CHECK REPORT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(file=sys.stderr)

    # Summary counts
    for sev in ("CRITICAL", "WARNING", "INFO"):
        count = summary.get(sev, 0)
        marker = ">>>" if sev == "CRITICAL" and count > 0 else "   "
        print(f"  {marker} {sev:10s}: {count}", file=sys.stderr)
    print(f"  {'':3s} {'TOTAL':10s}: {total}", file=sys.stderr)
    print(file=sys.stderr)

    if not findings:
        print("  No findings.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return

    # Group findings by check
    by_check: dict[str, list[dict]] = {}
    for f in findings:
        by_check.setdefault(f["check"], []).append(f)

    for check, items in sorted(by_check.items()):
        sev = items[0]["severity"]
        print(f"  [{sev}] {check} ({len(items)} finding(s))", file=sys.stderr)
        for item in items[:10]:  # Show up to 10 per check
            print(
                f"    {item['file']}:{item['line']} - {item['message'][:80]}",
                file=sys.stderr,
            )
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more", file=sys.stderr)
        print(file=sys.stderr)

    print("=" * 70, file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated code quality checks for the agent-memory codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Checks performed:
              numbered-comment          (WARNING)  # 1., # Step 1 patterns
              docstring-longer-than-body (INFO)    docstring > body line count
              sprint-ticket-ref         (WARNING)  Sprint N, Day N, NEW-N, CRITICAL-N
              generic-variable-name     (INFO)     data=, result=, temp=, item=, obj=
              placeholder-pass          (WARNING)  pass in non-abstract methods
              placeholder-not-implemented (WARNING) NotImplementedError("Sprint...")
              silent-exception-swallow  (WARNING)  bare except/Exception + pass/continue
              runtime-import            (INFO)     import inside function body
              god-method                (INFO)     functions > 50 lines
              stale-branding            (CRITICAL) from semantic., Semantic Team, etc.

            Exit code: 1 if any CRITICAL findings, 0 otherwise.
        """),
    )
    parser.add_argument(
        "directories",
        nargs="*",
        default=["src/agent_memory/"],
        help="Directories to scan (default: src/agent_memory/)",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)",
    )

    args = parser.parse_args()

    report = Report()
    files = collect_python_files(args.directories)

    if not files:
        print("Warning: no Python files found.", file=sys.stderr)

    for filepath in files:
        analyze_file(filepath, report)

    report_dict = report.to_dict()

    # JSON report to stdout
    json.dump(report_dict, sys.stdout, indent=args.json_indent)
    sys.stdout.write("\n")

    # Summary table to stderr
    print_summary_table(report_dict)

    # Exit code
    if report.has_critical():
        print(
            "\nExit code 1: CRITICAL findings detected.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
