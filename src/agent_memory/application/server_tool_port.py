# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Port for server-side tool execution (WebSearch, WebFetch).

When the local model calls WebSearch or WebFetch, the server executes them
against SearXNG / Jina Reader and feeds results back to the model in the
same generation loop.

Architecture layer: application (port definition).
Implemented by: adapters/outbound/server_tool_adapter.py
"""

from __future__ import annotations

from typing import Any, Protocol

# Tools the server handles instead of returning to the client
SERVER_TOOL_NAMES = frozenset({"WebSearch", "WebFetch"})

# Max rounds of server-side tool execution per request
MAX_TOOL_ROUNDS = 3


class ServerToolPort(Protocol):
    """Port for executing tools server-side."""

    def can_execute(self, tool_name: str) -> bool:
        """Check if this tool can be executed server-side."""
        ...

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a server-side tool and return the result as text."""
        ...


def _normalize_query(q: str) -> str:
    """Normalize a search query for dedup comparison."""
    return " ".join(q.lower().split())


def _dedup_search_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate WebSearch calls — exact query match only.

    Keeps first occurrence of each unique normalized query, drops duplicates.
    """
    if not calls:
        return calls

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()

    for tc in calls:
        query = _normalize_query(tc.get("input", {}).get("query", ""))
        if not query or query in seen:
            continue
        seen.add(query)
        kept.append(tc)

    return kept


def split_tool_calls(
    tool_calls: list[dict[str, Any]],
    executor: ServerToolPort | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split tool calls into server-executable and client-side.

    Server-side WebSearch calls are deduplicated to remove the model's
    repetitive query generation pattern.

    Returns:
        (server_calls, client_calls)
    """
    if not executor or not tool_calls:
        return [], tool_calls

    server_calls = []
    client_calls = []
    for tc in tool_calls:
        if executor.can_execute(tc.get("name", "")):
            server_calls.append(tc)
        else:
            client_calls.append(tc)

    # Deduplicate WebSearch calls
    search_calls = [tc for tc in server_calls if tc.get("name") == "WebSearch"]
    other_server = [tc for tc in server_calls if tc.get("name") != "WebSearch"]
    if len(search_calls) > 1:
        search_calls = _dedup_search_calls(search_calls)
    server_calls = other_server + search_calls

    return server_calls, client_calls
