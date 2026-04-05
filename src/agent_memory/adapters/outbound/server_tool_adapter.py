# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Outbound adapter: server-side tool execution via SearXNG / Jina Reader.

Implements ServerToolPort by making HTTP calls to locally-hosted search
(SearXNG) and fetch (Jina Reader) services. This keeps outbound HTTP
concerns in the adapter layer, not in the application or inbound layers.

Architecture layer: outbound adapter.
Port: application/server_tool_port.py (ServerToolPort protocol).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class ServerToolAdapter:
    """Executes WebSearch and WebFetch against local infrastructure.

    Satisfies ServerToolPort protocol.
    Injected from api_server.py where the URLs are configured.
    """

    def __init__(
        self,
        searxng_url: str = "",
        jina_reader_url: str = "",
        search_timeout: int = 30,
        fetch_timeout: int = 30,
    ) -> None:
        self._searxng_url = searxng_url
        self._jina_reader_url = jina_reader_url
        self._search_timeout = search_timeout
        self._fetch_timeout = fetch_timeout

    def can_execute(self, tool_name: str) -> bool:
        """Check if this tool can be executed server-side."""
        if tool_name == "WebSearch" and self._searxng_url:
            return True
        if tool_name == "WebFetch" and self._jina_reader_url:
            return True
        return False

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a server-side tool and return the result as text."""
        if tool_name == "WebSearch":
            return self._web_search(tool_input)
        if tool_name == "WebFetch":
            return self._web_fetch(tool_input)
        return f"Unknown server tool: {tool_name}"

    def _web_search(self, tool_input: dict[str, Any]) -> str:
        """Execute web search via SearXNG."""
        query = tool_input.get("query", "")
        if not query:
            return "Error: query parameter is required"
        if not self._searxng_url:
            return "Error: search not configured (SEMANTIC_SERVER_SEARXNG_URL not set)"

        num = tool_input.get("num", 10)
        params = f"q={quote_plus(query)}&format=json&pageno=1"
        url = f"{self._searxng_url}/search?{params}"

        try:
            with urlopen(url, timeout=self._search_timeout) as resp:  # noqa: S310
                data = json.loads(resp.read())
            results = data.get("results", [])[:num]
            lines = [f"Web search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                link = r.get("url", "")
                snippet = r.get("content", "")
                lines.append(f"{i}. {title}")
                lines.append(f"   {link}")
                if snippet:
                    lines.append(f"   {snippet}")
                lines.append("")
            return "\n".join(lines)
        except (URLError, TimeoutError) as e:
            logger.warning("WebSearch failed: %s", e)
            return f"Search failed: {e}"
        except Exception as e:
            logger.warning("WebSearch error: %s", e)
            return f"Search error: {e}"

    def _web_fetch(self, tool_input: dict[str, Any]) -> str:
        """Fetch URL content via Jina Reader."""
        url = tool_input.get("url", "")
        if not url:
            return "Error: url parameter is required"
        if not self._jina_reader_url:
            return "Error: fetch not configured (SEMANTIC_SERVER_JINA_READER_URL not set)"

        reader_url = f"{self._jina_reader_url}/{url}"
        try:
            req = Request(reader_url)  # noqa: S310
            with urlopen(req, timeout=self._fetch_timeout) as resp:  # noqa: S310
                content = resp.read().decode("utf-8", errors="replace")
            # Truncate to avoid blowing context
            max_chars = 8000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated]"
            return content
        except (URLError, TimeoutError) as e:
            logger.warning("WebFetch failed for %s: %s", url, e)
            return f"Fetch failed: {e}"
        except Exception as e:
            logger.warning("WebFetch error for %s: %s", url, e)
            return f"Fetch error: {e}"
