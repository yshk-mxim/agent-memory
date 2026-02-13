# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Shared HTTP client for coordination API calls.

Wraps all coordination and agent management endpoints.
Each function is stateless — no Streamlit dependency.
All HTTP calls are wrapped in try/except to handle connection errors gracefully.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)

_HTTP_OK = 200
_HTTP_CREATED = 201
_HTTP_NO_CONTENT = 204


def create_session(
    base_url: str,
    *,
    topology: str,
    debate_format: str,
    decision_mode: str,
    agents: list[dict[str, Any]],
    initial_prompt: str,
    max_turns: int,
    per_agent_prompts: dict[str, str] | None = None,
    persistent_cache_prefix: str = "",
    prior_agent_messages: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any] | None:
    """Create a coordination session. Returns response dict or None on failure."""
    try:
        payload: dict[str, Any] = {
            "topology": topology,
            "debate_format": debate_format,
            "decision_mode": decision_mode,
            "agents": agents,
            "initial_prompt": initial_prompt,
            "max_turns": max_turns,
        }
        if per_agent_prompts:
            payload["per_agent_prompts"] = per_agent_prompts
        if persistent_cache_prefix:
            payload["persistent_cache_prefix"] = persistent_cache_prefix
        if prior_agent_messages:
            payload["prior_agent_messages"] = prior_agent_messages
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{base_url}/v1/coordination/sessions",
                json=payload,
            )
            if resp.status_code == _HTTP_CREATED:
                return cast(dict[str, Any], resp.json())
            logger.warning("Create session failed: HTTP %d", resp.status_code)
            return None
    except httpx.HTTPError:
        logger.debug("Create session connection error", exc_info=True)
        return None


def get_session_status(base_url: str, session_id: str) -> dict[str, Any] | None:
    """Get session status. Returns status dict or None on failure."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/v1/coordination/sessions/{session_id}")
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            return None
    except httpx.HTTPError:
        logger.debug("Get session status connection error", exc_info=True)
        return None


def get_session_messages(base_url: str, session_id: str) -> list[dict[str, Any]]:
    """Get all messages in a session. Returns list of message dicts."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base_url}/v1/coordination/sessions/{session_id}/messages",
            )
            if resp.status_code == _HTTP_OK:
                data = cast(dict[str, Any], resp.json())
                return cast(list[dict[str, Any]], data.get("messages", []))
            return []
    except httpx.HTTPError:
        logger.debug("Get messages connection error", exc_info=True)
        return []


def execute_turn(base_url: str, session_id: str) -> dict[str, Any] | None:
    """Execute a single turn. Returns response dict or None on failure."""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{base_url}/v1/coordination/sessions/{session_id}/turn",
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            return None
    except httpx.HTTPError:
        logger.debug("Execute turn connection error", exc_info=True)
        return None


def execute_round(base_url: str, session_id: str) -> dict[str, Any] | None:
    """Execute a full round. Returns response dict or None on failure."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{base_url}/v1/coordination/sessions/{session_id}/round",
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            return None
    except httpx.HTTPError:
        logger.debug("Execute round connection error", exc_info=True)
        return None


def execute_turns(base_url: str, session_id: str, count: int) -> bool:
    """Execute multiple sequential turns. Returns True on success."""
    for _ in range(count):
        result = execute_turn(base_url, session_id)
        if result is None:
            return False
    return True


def stream_turn(
    base_url: str,
    session_id: str,
) -> Iterator[tuple[str, Any]]:
    """Stream a single turn via SSE. Yields (event_type, text_chunk)."""
    try:
        with (
            httpx.Client(timeout=120.0) as client,
            client.stream(
                "POST",
                f"{base_url}/v1/coordination/sessions/{session_id}/turn/stream",
            ) as resp,
        ):
            if resp.status_code != _HTTP_OK:
                return
            event_type = ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    yield event_type, data
    except httpx.HTTPError:
        logger.debug("Stream turn connection error", exc_info=True)


def stream_round(
    base_url: str,
    session_id: str,
) -> Iterator[tuple[str, str, str]]:
    """Stream a round via SSE. Yields (event_type, agent_name, data)."""
    try:
        with (
            httpx.Client(timeout=300.0) as client,
            client.stream(
                "POST",
                f"{base_url}/v1/coordination/sessions/{session_id}/round/stream",
            ) as resp,
        ):
            if resp.status_code != _HTTP_OK:
                return
            event_type = ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    agent_name = data.get("agent_name", "")
                    yield event_type, agent_name, json.dumps(data)
    except httpx.HTTPError:
        logger.debug("Stream round connection error", exc_info=True)


def delete_session(base_url: str, session_id: str) -> bool:
    """Delete a coordination session. Returns True on success."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                f"{base_url}/v1/coordination/sessions/{session_id}",
            )
            return resp.status_code == _HTTP_NO_CONTENT
    except httpx.HTTPError:
        logger.debug("Delete session connection error", exc_info=True)
        return False


def list_sessions(base_url: str) -> list[dict[str, Any]]:
    """List all active sessions. Returns list of status dicts."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/coordination/sessions")
            if resp.status_code == _HTTP_OK:
                data = cast(dict[str, Any], resp.json())
                return cast(list[dict[str, Any]], data.get("sessions", []))
            return []
    except httpx.HTTPError:
        logger.debug("List sessions connection error", exc_info=True)
        return []


def get_agent_stats(base_url: str) -> dict[str, Any] | None:
    """Get aggregate cache statistics. Returns stats dict or None on failure."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/agents/stats")
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            return None
    except httpx.HTTPError:
        logger.debug("Get agent stats connection error", exc_info=True)
        return None


def get_agent_list(base_url: str) -> list[dict[str, Any]]:
    """Get list of all cached agents. Returns list of agent dicts."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/agents/list")
            if resp.status_code == _HTTP_OK:
                data = cast(dict[str, Any], resp.json())
                return cast(list[dict[str, Any]], data.get("agents", []))
            return []
    except httpx.HTTPError:
        logger.debug("Get agent list connection error", exc_info=True)
        return []


def delete_persistent_caches(base_url: str, prefix: str) -> bool:
    """Delete all persistent caches for a scenario. Returns True on success."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                f"{base_url}/v1/coordination/caches/{prefix}",
            )
            return resp.status_code == _HTTP_NO_CONTENT
    except httpx.HTTPError:
        logger.debug("Delete persistent caches connection error", exc_info=True)
        return False


def delete_agent(base_url: str, agent_id: str) -> bool:
    """Delete an agent cache. Returns True on success."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{base_url}/v1/agents/{agent_id}")
            return resp.status_code == _HTTP_NO_CONTENT
    except httpx.HTTPError:
        logger.debug("Delete agent connection error", exc_info=True)
        return False


# --- Admin API functions for model management ---


def get_available_models(base_url: str, admin_key: str) -> list[str]:
    """Get list of available models. Returns list of model IDs."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base_url}/admin/models/available",
                headers={"X-Admin-Key": admin_key},
            )
            if resp.status_code == _HTTP_OK:
                data = cast(dict[str, Any], resp.json())
                return cast(list[str], data.get("models", []))
            return []
    except httpx.HTTPError:
        logger.debug("Get available models connection error", exc_info=True)
        return []


def get_current_model(base_url: str, admin_key: str) -> dict[str, Any] | None:
    """Get current model info. Returns model info dict or None."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base_url}/admin/models/current",
                headers={"X-Admin-Key": admin_key},
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            return None
    except httpx.HTTPError:
        logger.debug("Get current model connection error", exc_info=True)
        return None


def swap_model(
    base_url: str,
    admin_key: str,
    model_id: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any] | None:
    """Swap to a new model. Returns swap result or None on failure."""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{base_url}/admin/models/swap",
                headers={"X-Admin-Key": admin_key},
                json={"model_id": model_id, "timeout_seconds": timeout_seconds},
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            logger.warning("Swap model failed: HTTP %d - %s", resp.status_code, resp.text)
            return None
    except httpx.HTTPError:
        logger.debug("Swap model connection error", exc_info=True)
        return None


def offload_model(base_url: str, admin_key: str) -> dict[str, Any] | None:
    """Offload current model to free memory. Returns result or None on failure."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{base_url}/admin/models/offload",
                headers={"X-Admin-Key": admin_key},
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            logger.warning("Offload model failed: HTTP %d - %s", resp.status_code, resp.text)
            return None
    except httpx.HTTPError:
        logger.debug("Offload model connection error", exc_info=True)
        return None


def clear_all_caches(base_url: str, admin_key: str) -> dict[str, Any] | None:
    """Clear all caches (memory and disk). Returns result or None on failure."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(
                f"{base_url}/admin/caches",
                headers={"X-Admin-Key": admin_key},
            )
            if resp.status_code == _HTTP_OK:
                return cast(dict[str, Any], resp.json())
            logger.warning("Clear caches failed: HTTP %d - %s", resp.status_code, resp.text)
            return None
    except httpx.HTTPError:
        logger.debug("Clear caches connection error", exc_info=True)
        return None
