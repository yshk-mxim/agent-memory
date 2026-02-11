# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Agent Memory Multi-Agent Demo.

Demonstrates multi-agent conversation with per-agent KV cache persistence.
Each column represents an independent agent with its own session, showing
cache state transitions (COLD -> WARM -> HOT) and real-time performance metrics.

Usage:
    pip install -r demo/requirements.txt
    agent-memory serve  # start the server on :8000
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so page files can import demo.lib
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import contextlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from uuid import uuid4

import httpx
import streamlit as st

SERVER_URL = "http://127.0.0.1:8000"
NUM_AGENTS = 4
_WARM_TURN_THRESHOLD = 2
AGENT_NAMES = ["Alpha", "Beta", "Gamma", "Delta"]
AGENT_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
SUGGESTED_PROMPTS = [
    "Explain quantum computing basics",
    "Write a Python linked list implementation",
    "What caused World War I?",
    "Describe the water cycle step by step",
]

# Default inference parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_CONTEXT = 32768
_CHAT_CONTAINER_HEIGHT = 350
_THINKING_SYSTEM_PROMPT = "Think step by step before answering."


def init_session_state() -> None:
    """Initialize per-agent session state on first load."""
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.server_status = "unknown"
    st.session_state.model_info = None

    # ThreadPoolExecutor for concurrent HTTP requests
    st.session_state.executor = ThreadPoolExecutor(max_workers=NUM_AGENTS)

    for i in range(NUM_AGENTS):
        prefix = f"agent_{i}"
        st.session_state[f"{prefix}_sid"] = f"{AGENT_NAMES[i].lower()}_{uuid4().hex[:8]}"
        st.session_state[f"{prefix}_messages"] = []
        st.session_state[f"{prefix}_turn"] = 0
        st.session_state[f"{prefix}_metrics"] = []
        st.session_state[f"{prefix}_generating"] = False
        st.session_state[f"{prefix}_future"] = None
        st.session_state[f"{prefix}_pending_input"] = None
        # Per-agent inference settings
        st.session_state[f"{prefix}_temperature"] = DEFAULT_TEMPERATURE
        st.session_state[f"{prefix}_top_p"] = DEFAULT_TOP_P
        st.session_state[f"{prefix}_max_tokens"] = DEFAULT_MAX_TOKENS
        st.session_state[f"{prefix}_max_context"] = DEFAULT_MAX_CONTEXT
        st.session_state[f"{prefix}_thinking"] = False


def get_cache_state(turn: int) -> tuple[str, str]:
    """Return (label, color) based on turn count."""
    if turn == 0:
        return "COLD", "#3b82f6"
    if turn <= _WARM_TURN_THRESHOLD:
        return "WARM", "#f59e0b"
    return "HOT", "#ef4444"


def check_server() -> dict | None:
    """Check server health and return memory info."""
    try:
        with httpx.Client(timeout=3.0) as client:
            health = client.get(f"{SERVER_URL}/health/ready")
            if health.status_code != HTTPStatus.OK:
                return None
            mem = client.get(f"{SERVER_URL}/debug/memory")
            if mem.status_code == HTTPStatus.OK:
                return mem.json()
            return {}
    except httpx.HTTPError:
        return None


def fetch_model_info() -> dict | None:
    """Fetch model information from the server."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/models")
            if resp.status_code == HTTPStatus.OK:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    return models[0]
    except httpx.HTTPError:
        return None
    return None


def get_model_short_name(model_id: str) -> str:
    """Extract a short display name from a HuggingFace model ID."""
    return model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id


def stream_response(
    messages: list[dict],
    session_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """Send a streaming chat request and yield tokens.

    Returns (full_text, usage_dict).
    """
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    headers = {"X-Session-ID": session_id}

    full_text = ""
    usage: dict = {}
    first_token_time: float | None = None
    start = time.perf_counter()

    with (
        httpx.Client(timeout=120.0) as client,
        client.stream(
            "POST", f"{SERVER_URL}/v1/chat/completions", json=body, headers=headers
        ) as resp,
    ):
        if resp.status_code != HTTPStatus.OK:
            return f"[Error: HTTP {resp.status_code}]", {}
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "usage" in chunk:
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += token

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    out_tokens = usage.get("completion_tokens", 0)
    tps = out_tokens / elapsed if elapsed > 0 and out_tokens > 0 else 0

    metrics = {
        "ttft_ms": round(ttft),
        "e2e_ms": round(elapsed * 1000),
        "tps": round(tps, 1),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": out_tokens,
    }
    return full_text, metrics


def non_stream_response(
    messages: list[dict],
    session_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """Send a non-streaming chat request."""
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    headers = {"X-Session-ID": session_id}

    start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{SERVER_URL}/v1/chat/completions", json=body, headers=headers)
        elapsed = time.perf_counter() - start

    if resp.status_code != HTTPStatus.OK:
        return f"[Error: HTTP {resp.status_code}: {resp.text[:200]}]", {}

    data = resp.json()
    usage = data.get("usage", {})
    text = data["choices"][0]["message"]["content"] or ""
    out_tokens = usage.get("completion_tokens", 0)
    tps = out_tokens / elapsed if elapsed > 0 and out_tokens > 0 else 0

    return text, {
        "ttft_ms": round(elapsed * 1000),
        "e2e_ms": round(elapsed * 1000),
        "tps": round(tps, 1),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": out_tokens,
    }


def estimate_token_count(messages: list[dict]) -> int:
    """Rough estimate of token count from messages (4 chars per token)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


def render_agent_settings(agent_idx: int) -> None:
    """Render per-agent inference settings in an expander."""
    prefix = f"agent_{agent_idx}"

    with st.expander("Settings", expanded=False):
        st.session_state[f"{prefix}_temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get(f"{prefix}_temperature", DEFAULT_TEMPERATURE),
            step=0.1,
            key=f"{prefix}_temp_slider",
            help="0.0 = deterministic, 2.0 = very creative",
        )

        st.session_state[f"{prefix}_top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(f"{prefix}_top_p", DEFAULT_TOP_P),
            step=0.05,
            key=f"{prefix}_topp_slider",
            help="Nucleus sampling: 1.0 = all tokens, 0.1 = top 10%",
        )

        st.session_state[f"{prefix}_max_tokens"] = st.number_input(
            "Max tokens",
            min_value=1,
            max_value=4096,
            value=st.session_state.get(f"{prefix}_max_tokens", DEFAULT_MAX_TOKENS),
            step=64,
            key=f"{prefix}_maxtok_input",
            help="Maximum tokens to generate per response",
        )

        # Context length limit
        model_info = st.session_state.get("model_info")
        max_ctx_limit = DEFAULT_MAX_CONTEXT
        if model_info and "spec" in model_info:
            max_ctx_limit = model_info["spec"].get("max_context_length", DEFAULT_MAX_CONTEXT)

        st.session_state[f"{prefix}_max_context"] = st.number_input(
            "Max context length",
            min_value=256,
            max_value=max_ctx_limit,
            value=min(
                st.session_state.get(f"{prefix}_max_context", DEFAULT_MAX_CONTEXT),
                max_ctx_limit,
            ),
            step=1024,
            key=f"{prefix}_maxctx_input",
            help="Stop inference if conversation exceeds this token count",
        )

        # Thinking toggle — only meaningful for models that support it
        st.session_state[f"{prefix}_thinking"] = st.toggle(
            "Extended thinking",
            value=st.session_state.get(f"{prefix}_thinking", False),
            key=f"{prefix}_thinking_toggle",
            help="Enable chain-of-thought reasoning (model must support it)",
        )


def _submit_agent_message(prefix: str, sid: str, messages: list[dict], user_input: str) -> None:
    """Validate context, add user message, and submit generation request."""
    max_context = st.session_state.get(f"{prefix}_max_context", DEFAULT_MAX_CONTEXT)
    est_tokens = estimate_token_count([*messages, {"role": "user", "content": user_input}])
    if est_tokens > max_context:
        st.warning(
            f"Context length (~{est_tokens} tokens) exceeds limit ({max_context}). "
            "Clear history or increase the limit."
        )
        return

    messages.append({"role": "user", "content": user_input})
    st.session_state[f"{prefix}_messages"] = messages

    temperature = st.session_state.get(f"{prefix}_temperature", DEFAULT_TEMPERATURE)
    top_p = st.session_state.get(f"{prefix}_top_p", DEFAULT_TOP_P)
    max_tokens = st.session_state.get(f"{prefix}_max_tokens", DEFAULT_MAX_TOKENS)
    thinking = st.session_state.get(f"{prefix}_thinking", False)

    api_messages = list(messages)
    if thinking and api_messages:
        api_messages = [
            {"role": "system", "content": _THINKING_SYSTEM_PROMPT},
            *api_messages,
        ]

    future = st.session_state.executor.submit(
        non_stream_response, api_messages, sid, temperature, top_p, max_tokens
    )
    st.session_state[f"{prefix}_future"] = future
    st.session_state[f"{prefix}_generating"] = True
    st.rerun()


@st.fragment
def render_agent_column(agent_idx: int) -> None:
    """Render a single agent's conversation column.

    Decorated with @st.fragment so each agent reruns independently —
    clicking a button in one column won't interrupt an in-progress
    HTTP request in another column.
    """
    prefix = f"agent_{agent_idx}"
    name = AGENT_NAMES[agent_idx]
    sid = st.session_state[f"{prefix}_sid"]
    messages = st.session_state[f"{prefix}_messages"]
    turn = st.session_state[f"{prefix}_turn"]
    all_metrics = st.session_state[f"{prefix}_metrics"]

    cache_label, cache_color = get_cache_state(turn)
    st.markdown(
        f"### {name} "
        f'<span style="background:{cache_color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.7em;">{cache_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Session: `{sid}` | Turns: {turn}")

    render_agent_settings(agent_idx)

    chat_container = st.container(height=_CHAT_CONTAINER_HEIGHT)
    with chat_container:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if all_metrics:
        m = all_metrics[-1]
        cols = st.columns(3)
        cols[0].metric("TTFT", f"{m['ttft_ms']}ms")
        cols[1].metric("TPS", f"{m['tps']}")
        cols[2].metric("Tokens", f"{m['input_tokens']}+{m['output_tokens']}")

    user_input = st.chat_input(f"Message {name}...", key=f"{prefix}_input")

    if not messages and st.button(
        f'Try: "{SUGGESTED_PROMPTS[agent_idx][:40]}..."',
        key=f"{prefix}_suggest",
        use_container_width=True,
    ):
        user_input = SUGGESTED_PROMPTS[agent_idx]

    if st.session_state.get(f"{prefix}_generating", False):
        st.info(f"{name} is generating... (concurrent request)", icon="⏳")

    if user_input:
        _submit_agent_message(prefix, sid, messages, user_input)


@st.fragment(run_every="0.5s")
def check_agent_completions() -> None:
    """Poll for completed futures and update session state.

    This fragment runs every 0.5s and checks if any agent's background
    HTTP request has finished. When a future completes, extracts the
    result, updates messages/metrics, and clears the generating flag.
    """
    for i in range(NUM_AGENTS):
        prefix = f"agent_{i}"
        future = st.session_state.get(f"{prefix}_future")

        if future and future.done():
            # Extract result from completed future
            try:
                text, metrics = future.result()

                # Update session state with response
                messages = st.session_state[f"{prefix}_messages"]
                messages.append({"role": "assistant", "content": text})
                st.session_state[f"{prefix}_messages"] = messages

                turn = st.session_state[f"{prefix}_turn"]
                st.session_state[f"{prefix}_turn"] = turn + 1

                if metrics:
                    all_metrics = st.session_state[f"{prefix}_metrics"]
                    all_metrics.append(metrics)
                    st.session_state[f"{prefix}_metrics"] = all_metrics

                # Clear future and generating flag
                st.session_state[f"{prefix}_future"] = None
                st.session_state[f"{prefix}_generating"] = False

                # Trigger rerun to display new message
                st.rerun()

            except Exception as e:
                # Handle errors from HTTP request
                messages = st.session_state[f"{prefix}_messages"]
                messages.append({"role": "assistant", "content": f"[Error: {e}]"})
                st.session_state[f"{prefix}_messages"] = messages
                st.session_state[f"{prefix}_future"] = None
                st.session_state[f"{prefix}_generating"] = False
                st.rerun()


def _render_server_status() -> None:
    """Render server status section in sidebar."""
    st.subheader("Server Status")
    mem_info = check_server()
    if mem_info is None:
        st.error("Server offline")
        st.caption(f"Expected at {SERVER_URL}")
        st.markdown("Start with: `agent-memory serve`")
        return

    st.success("Server online")
    if not mem_info:
        return

    c1, c2 = st.columns(2)
    active = mem_info.get("active_memory_mb", 0)
    peak = mem_info.get("peak_memory_mb", 0)
    used = mem_info.get("pool_used_blocks", 0)
    total = mem_info.get("pool_total_blocks", 0)
    c1.metric("GPU Memory", f"{active:.0f} MB")
    c2.metric("Peak", f"{peak:.0f} MB")
    if total > 0:
        pct = used / total * 100
        st.progress(pct / 100, text=f"Pool: {used}/{total} blocks ({pct:.0f}%)")


def _render_model_info() -> None:
    """Render model information section in sidebar."""
    st.subheader("Model")
    model_info = fetch_model_info()
    st.session_state.model_info = model_info

    # Check if model is loaded
    model_id = model_info.get("id", "") if model_info else ""

    if not model_info or not model_id:
        st.markdown("**No model loaded**")
        st.caption("Use Switch Model to load one")
        # Still show model switch UI even when no model loaded
        _render_model_switch("")
        return

    short_name = get_model_short_name(model_id)
    st.markdown(f"**{short_name}**")
    st.caption(model_id)
    spec = model_info.get("spec", {})
    if spec:
        s1, s2 = st.columns(2)
        s1.markdown(f"Layers: **{spec.get('n_layers', '?')}**")
        s2.markdown(f"KV heads: **{spec.get('n_kv_heads', '?')}**")
        s3, s4 = st.columns(2)
        kv_bits = spec.get("kv_bits")
        quant_label = f"Q{kv_bits}" if kv_bits else "FP16"
        s3.markdown(f"Quantization: **{quant_label}**")
        s4.markdown(f"Head dim: **{spec.get('head_dim', '?')}**")
        max_ctx = spec.get("max_context_length", "?")
        ctx_text = (
            f"Max context: **{max_ctx:,}** tokens"
            if isinstance(max_ctx, int)
            else f"Max context: **{max_ctx}**"
        )
        st.markdown(ctx_text)

    # Model switch section (requires SEMANTIC_ADMIN_KEY)
    _render_model_switch(model_id)


def _render_model_switch(current_model_id: str) -> None:
    """Render model switch controls (requires admin key)."""
    import os

    from demo.lib import api_client

    admin_key = os.getenv("SEMANTIC_ADMIN_KEY", "")
    if not admin_key:
        st.caption("Set SEMANTIC_ADMIN_KEY to enable model switching")
        return

    with st.expander("Switch Model", expanded=False):
        available = api_client.get_available_models(SERVER_URL, admin_key)
        if not available:
            st.warning("Could not fetch available models")
            return

        # Show all models, mark current one
        def format_model(m):
            name = get_model_short_name(m)
            if m == current_model_id:
                return f"{name} (current)"
            return name

        selected = st.selectbox(
            "Target model",
            options=available,
            format_func=format_model,
            key="model_switch_selector",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬇️ Offload", use_container_width=True):
                with st.spinner("Offloading model..."):
                    result = api_client.offload_model(SERVER_URL, admin_key)
                    if result and result.get("status") == "success":
                        st.success("Model offloaded - memory freed")
                        st.rerun()
                    else:
                        st.error("Offload failed")
        with col2:
            if st.button("🔄 Load", use_container_width=True, type="primary"):
                with st.spinner(f"Loading {get_model_short_name(selected)}..."):
                    result = api_client.swap_model(SERVER_URL, admin_key, selected)
                    if result and result.get("status") == "success":
                        st.success(f"Loaded {get_model_short_name(selected)}")
                        st.rerun()
                    else:
                        st.error("Model swap failed")

        # Clear all caches button
        if st.button("🗑️ Clear All Caches", use_container_width=True):
            with st.spinner("Clearing caches..."):
                result = api_client.clear_all_caches(SERVER_URL, admin_key)
                if result and result.get("status") == "success":
                    st.success(result.get("message", "Caches cleared"))
                    st.rerun()
                else:
                    st.error("Failed to clear caches")


def _render_agent_metrics() -> None:
    """Render per-agent metrics summary in sidebar."""
    st.subheader("Agent Metrics")
    for i in range(NUM_AGENTS):
        prefix = f"agent_{i}"
        turn = st.session_state.get(f"{prefix}_turn", 0)
        metrics = st.session_state.get(f"{prefix}_metrics", [])
        label, color = get_cache_state(turn)
        badge = (
            f'<span style="background:{color};color:white;padding:1px 6px;'
            f'border-radius:8px;font-size:0.7em;">{label}</span>'
        )
        if metrics:
            avg_tps = sum(m["tps"] for m in metrics) / len(metrics)
            avg_ttft = sum(m["ttft_ms"] for m in metrics) / len(metrics)
            detail = f"| {turn} turns | {avg_tps:.0f} TPS | {avg_ttft:.0f}ms TTFT"
        else:
            detail = "| No messages yet"
        st.markdown(
            f"**{AGENT_NAMES[i]}** {badge} {detail}",
            unsafe_allow_html=True,
        )


def _reset_all_agents() -> None:
    """Delete all agents from server and clear session state."""
    try:
        with httpx.Client(timeout=5.0) as client:
            for i in range(NUM_AGENTS):
                sid = st.session_state.get(f"agent_{i}_sid", "")
                if sid:
                    with contextlib.suppress(Exception):
                        client.delete(f"{SERVER_URL}/v1/agents/oai_{sid}")
    except httpx.HTTPError:
        pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def render_sidebar() -> None:
    """Render the sidebar with server status, model info, and global stats."""
    with st.sidebar:
        st.title("Agent Memory Demo")
        st.markdown("Multi-agent KV cache persistence on Apple Silicon")
        st.divider()

        _render_server_status()
        st.divider()

        _render_model_info()
        st.divider()

        _render_agent_metrics()
        st.divider()

        if st.button("Reset All Agents", use_container_width=True):
            _reset_all_agents()

        st.divider()
        st.caption(
            "Built with [Streamlit](https://streamlit.io) | "
            "[agent-memory](https://github.com/yshk-mxim/agent-memory)"
        )


def main() -> None:
    st.set_page_config(
        page_title="Agent Memory - Multi-Agent Demo",
        page_icon="$",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    check_agent_completions()  # Polling fragment for concurrent requests
    render_sidebar()

    # Title with model badge
    model_info = st.session_state.get("model_info")
    model_badge = ""
    if model_info:
        short_name = get_model_short_name(model_info.get("id", ""))
        spec = model_info.get("spec", {})
        kv_bits = spec.get("kv_bits")
        quant_label = f"Q{kv_bits}" if kv_bits else "FP16"
        model_badge = (
            f' <span style="background:#6366f1;color:white;padding:2px 10px;'
            f'border-radius:10px;font-size:0.5em;vertical-align:middle;">'
            f"{short_name} ({quant_label})</span>"
        )

    st.markdown(
        f"## Multi-Agent Conversation Demo{model_badge}\n"
        "Each agent maintains an independent conversation with **persistent KV cache**. "
        "Watch cache states transition from **COLD** (first message) to **WARM** (cache hit) "
        "to **HOT** (deep conversation) as turn count increases. "
        "Expand **Settings** under each agent to tune inference parameters.",
        unsafe_allow_html=True,
    )

    # Send All button for concurrent batch requests
    _btn_col1, btn_col2, _btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        if st.button("🚀 Send All (Concurrent Test)", use_container_width=True, type="primary"):
            # Submit all agents with suggested prompts simultaneously
            for i in range(NUM_AGENTS):
                prefix = f"agent_{i}"
                messages = st.session_state[f"{prefix}_messages"]
                sid = st.session_state[f"{prefix}_sid"]

                # Skip if already generating
                if st.session_state.get(f"{prefix}_generating", False):
                    continue

                # Use suggested prompt for this agent
                user_input = SUGGESTED_PROMPTS[i]
                messages.append({"role": "user", "content": user_input})
                st.session_state[f"{prefix}_messages"] = messages

                # Read per-agent settings
                temperature = st.session_state.get(f"{prefix}_temperature", DEFAULT_TEMPERATURE)
                top_p = st.session_state.get(f"{prefix}_top_p", DEFAULT_TOP_P)
                max_tokens = st.session_state.get(f"{prefix}_max_tokens", DEFAULT_MAX_TOKENS)
                thinking = st.session_state.get(f"{prefix}_thinking", False)

                # Prepend thinking instruction if enabled
                api_messages = list(messages)
                if thinking and api_messages:
                    api_messages = [
                        {"role": "system", "content": _THINKING_SYSTEM_PROMPT},
                        *api_messages,
                    ]

                # Submit to executor (non-blocking)
                future = st.session_state.executor.submit(
                    non_stream_response, api_messages, sid, temperature, top_p, max_tokens
                )
                st.session_state[f"{prefix}_future"] = future
                st.session_state[f"{prefix}_generating"] = True

            st.rerun()

    st.divider()

    # 4 agent columns
    cols = st.columns(NUM_AGENTS, gap="medium")
    for i, col in enumerate(cols):
        with col:
            render_agent_column(i)


if __name__ == "__main__":
    main()
