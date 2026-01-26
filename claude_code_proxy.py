#!/usr/bin/env python3
"""Proxy server for Claude Code CLI that limits token requests to Gemma 3 capacity.

This proxy sits between Claude Code CLI and the Semantic Cache server,
capping max_tokens and thinking.budget_tokens to the model's actual limits.

Usage:
    python claude_code_proxy.py

Then configure Claude Code CLI:
    export ANTHROPIC_BASE_URL=http://localhost:8001
    claude
"""

import json
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

# Actual Gemma 3 12B limits
MAX_TOKENS_LIMIT = 8192
THINKING_BUDGET_LIMIT = 10000

# Backend semantic cache server
BACKEND_URL = "http://localhost:8000"


async def limit_request_tokens(body: dict[str, Any]) -> dict[str, Any]:
    """Cap token requests to model's actual capacity."""

    # Limit max_tokens
    if "max_tokens" in body:
        original = body["max_tokens"]
        body["max_tokens"] = min(original, MAX_TOKENS_LIMIT)
        if original > MAX_TOKENS_LIMIT:
            print(f"⚠️  Capped max_tokens: {original} → {body['max_tokens']}")

    # Limit thinking.budget_tokens
    if "thinking" in body and isinstance(body["thinking"], dict):
        if "budget_tokens" in body["thinking"]:
            original = body["thinking"]["budget_tokens"]
            body["thinking"]["budget_tokens"] = min(original, THINKING_BUDGET_LIMIT)
            if original > THINKING_BUDGET_LIMIT:
                print(f"⚠️  Capped thinking.budget_tokens: {original} → {body['thinking']['budget_tokens']}")

    return body


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward all requests to backend with token limiting."""

    # Build backend URL
    backend_path = f"{BACKEND_URL}/{path}"

    # Get request body
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
            # Limit tokens before forwarding
            body = await limit_request_tokens(body)
        except json.JSONDecodeError:
            body = await request.body()

    # Forward request
    async with httpx.AsyncClient() as client:
        # Handle streaming responses
        if body and isinstance(body, dict) and body.get("stream"):
            async def stream_response():
                async with client.stream(
                    request.method,
                    backend_path,
                    json=body,
                    headers=dict(request.headers),
                    params=request.query_params,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )

        # Non-streaming request
        response = await client.request(
            request.method,
            backend_path,
            json=body if isinstance(body, dict) else None,
            content=body if isinstance(body, bytes) else None,
            headers=dict(request.headers),
            params=request.query_params,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Claude Code Proxy Server")
    print("=" * 60)
    print(f"Proxy: http://localhost:8001")
    print(f"Backend: {BACKEND_URL}")
    print(f"Max tokens limit: {MAX_TOKENS_LIMIT}")
    print(f"Thinking budget limit: {THINKING_BUDGET_LIMIT}")
    print("=" * 60)
    print()
    print("Configure Claude Code CLI:")
    print("  export ANTHROPIC_BASE_URL=http://localhost:8001")
    print("  export ANTHROPIC_API_KEY=sk-ant-local-dev")
    print("  claude")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
