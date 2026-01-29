#!/usr/bin/env python3
"""Minimal semantic caching server for Claude Code CLI testing.

This is a simplified extraction of the core semantic caching functionality
designed to validate the basic flow without the complexity of the full server.

Components included:
1. Model loading (MLX with Q4)
2. Chunked prefill (memory-efficient long context processing)
3. Simple generation (using BatchGenerator)
4. Q4 cache save/load (safetensors format)

Components EXCLUDED (intentionally):
- Native generation path (too buggy)
- Complex prefix matching (simplified to: extend or miss)
- Block pool (using simple dicts)
- Multiple API protocols (Anthropic only)
- LRU eviction (simple dict cache)

Usage:
    python test_claude_server.py

Then configure Claude Code CLI to use http://localhost:8002
"""

import gc
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.mlx import save_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
CACHE_DIR = Path("/tmp/semantic_test_cache")
PORT = 8002
N_LAYERS = 27
KV_BITS = 4
KV_GROUP_SIZE = 64
CHUNKED_PREFILL_THRESHOLD = 2048
MAX_CHUNK_SIZE = 4096
MIN_CHUNK_SIZE = 512

# Absolute safety cap: if processor always says "continue", stop here
ABSOLUTE_SAFETY_CAP = 15

# Two-way translator system prompts (cached in separate DeepSeek contexts).
#
# Reverse translator: CLI tool results → DeepSeek-native conversation text.
# Forward translator: Turn management via constrained decoding.

REVERSE_TRANSLATOR_SYSTEM = """You translate tool execution results into conversation text.
Write as if the AI assistant performed the action. Be concise and factual.

Input format:
TOOL: <name>
ARGS: <key arguments>
RESULT: <execution output>
ERROR: <true/false>

Output: One or two sentences describing what happened. Include file paths, command output, or error messages verbatim.

Examples:

TOOL: Bash
ARGS: command=echo $((1+2))
RESULT: 3
ERROR: false
→ I ran `echo $((1+2))`. Output: 3

TOOL: Write
ARGS: file_path=/Users/dev/project/stack.py
RESULT: File created successfully at /Users/dev/project/stack.py
ERROR: false
→ I created /Users/dev/project/stack.py with the requested content.

TOOL: Read
ARGS: file_path=/Users/dev/config.py
RESULT: class Config:\\n    debug = True
ERROR: false
→ I read /Users/dev/config.py:
class Config:
    debug = True

TOOL: Bash
ARGS: command=pytest tests/
RESULT: FAILED test_stack.py::test_push - AssertionError
ERROR: true
→ Error running tests: FAILED test_stack.py::test_push - AssertionError

TOOL: TodoWrite
ARGS: todos=[Create stack.py (pending), Add tests (pending)]
RESULT: Todos have been modified.
ERROR: false
→ Updated task list: Create stack.py, Add tests (both pending).

TOOL: Write
ARGS: file_path=/readonly/hello.py
RESULT: Permission denied: /readonly/hello.py
ERROR: true
→ Error: Permission denied writing to /readonly/hello.py

TOOL: Bash
ARGS: command=cd /Users/dev/other_project && ls
RESULT: README.md src/ tests/ setup.py
ERROR: false
→ I ran `cd /Users/dev/other_project && ls`. Output: README.md src/ tests/ setup.py"""

FORWARD_TRANSLATOR_SYSTEM = """Output exactly one word: end, complete_and_end, or continue.

Decision rules (first match wins):
1. CONSECUTIVE_ERRORS >= 3 → end (stop retrying, report failure)
2. CONSECUTIVE_ERRORS 1-2 → continue (let model try a different approach)
3. PENDING_TODOS not none AND FILES_WRITTEN not none → complete_and_end
4. PENDING_TODOS not none AND FILES_WRITTEN = none → continue
5. RESULT_COUNT >= 8 AND PENDING_TODOS = none → end (long chain, wrap up)
6. Otherwise → end"""

# In-memory cache for reverse translations (deterministic: same input → same output)
_reverse_translation_cache: dict[str, str] = {}

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamped test_set filename (set once at module load)
TEST_SET_FILENAME = f"test_set_{time.strftime('%Y%m%d_%H%M%S')}.json"
TEST_SET_PATH = Path("/tmp/claude") / TEST_SET_FILENAME

# FastAPI app
app = FastAPI(title="Minimal Semantic Cache Server")

# Global state
model = None
tokenizer = None
cache_index: dict[str, dict[str, Any]] = {}  # agent_id -> {path, total_tokens, token_sequence}


# Request/Response models - Full Anthropic Messages API compatibility
class Message(BaseModel):
    role: str
    content: Any  # Can be string or list of content blocks

    model_config = {"extra": "ignore"}


class Tool(BaseModel):
    name: str
    description: str = ""
    input_schema: dict = {}

    model_config = {"extra": "ignore"}


class ThinkingConfig(BaseModel):
    type: str = "enabled"
    budget_tokens: int = 1000

    model_config = {"extra": "ignore"}


class ToolChoice(BaseModel):
    """Tool choice configuration."""
    type: str = "auto"  # "auto", "any", "tool", "none"
    name: str | None = None  # For type="tool"
    disable_parallel_tool_use: bool = False

    model_config = {"extra": "ignore"}


class OutputConfig(BaseModel):
    """Output configuration (beta)."""
    effort: str | None = None  # "low", "medium", "high"

    model_config = {"extra": "ignore"}


class OutputFormat(BaseModel):
    """JSON output format (beta)."""
    type: str = "json_schema"
    json_schema: dict | None = None

    model_config = {"extra": "ignore"}


class ContextManagement(BaseModel):
    """Context management config (beta)."""
    clear_tool_results: bool = False

    model_config = {"extra": "ignore"}


class MCPServer(BaseModel):
    """MCP server definition (beta)."""
    url: str
    name: str | None = None

    model_config = {"extra": "ignore"}


class MessagesRequest(BaseModel):
    """Full Anthropic Messages API request model.

    Supports all parameters from:
    - Standard Messages API
    - Beta Messages API
    - Claude Code CLI extensions
    """
    # Required
    model: str = "local"
    messages: list[Message]
    max_tokens: int = 100

    # Generation parameters
    temperature: float | None = None  # None = model default (usually 1.0)
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] = []

    # System prompt (string or list of blocks with cache_control)
    system: Any = ""

    # Tool use
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | dict | None = None  # dict for {"type": "tool", "name": "..."}

    # Extended thinking
    thinking: ThinkingConfig | None = None

    # Metadata
    metadata: dict | None = None

    # Service tier
    service_tier: str | None = None  # "auto" or "standard_only"

    # Beta features (from anthropic-beta header)
    betas: list[str] | None = None

    # Beta: Output configuration
    output_config: OutputConfig | dict | None = None
    output_format: OutputFormat | dict | None = None

    # Beta: Context management
    context_management: ContextManagement | dict | None = None

    # Beta: MCP servers
    mcp_servers: list[MCPServer] | list[dict] | None = None

    # Beta: Container for reuse
    container: str | None = None

    model_config = {"extra": "ignore"}


def extract_content_blocks(content: Any) -> list[dict]:
    """Extract all content blocks from content (string or list of blocks).

    Returns list of dicts with type and relevant content.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(block)
            elif isinstance(block, str):
                blocks.append({"type": "text", "text": block})
        return blocks
    return [{"type": "text", "text": str(content)}]


def extract_text(content: Any, role: str = "") -> str:
    """Extract text from content that may be string or list of blocks.

    For assistant messages: brief tool_use summaries, drops post-tool narration.
    Tool_result handling is done by the reverse translator in messages_to_prompt(),
    so this function only provides a fallback for tool_results.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        tool_texts: list[str] = []
        user_texts: list[str] = []
        seen_tool_use = False
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if role == "assistant" and seen_tool_use:
                        continue
                    if "SUGGESTION MODE" in text:
                        continue
                    user_texts.append(text)
                elif block.get("type") == "thinking":
                    pass
                elif block.get("type") == "tool_use":
                    seen_tool_use = True
                    name = block.get("name", "")
                    tool_texts.append(f"(Called {name})")
                elif block.get("type") == "tool_result":
                    # Fallback — normally handled by reverse translator
                    seen_tool_use = True
                    rc = block.get("content", "")
                    result_text = str(rc) if not isinstance(rc, list) else "\n".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in rc
                    )
                    if block.get("is_error"):
                        tool_texts.append(f"Error: {result_text}")
                    elif result_text.strip():
                        tool_texts.append(result_text)
            elif isinstance(block, str):
                user_texts.append(block)
        parts: list[str] = []
        if tool_texts:
            parts.extend(tool_texts)
        if user_texts:
            if tool_texts:
                parts.append("\n--- New message ---")
            parts.extend(user_texts)
        return "\n".join(parts)
    return str(content)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ContentBlock(BaseModel):
    """Content block that can be text or tool_use."""
    type: str = "text"
    text: str | None = None
    # For tool_use blocks
    id: str | None = None
    name: str | None = None
    input: dict | None = None

    model_config = {"extra": "ignore", "exclude_none": True}


class MessagesResponse(BaseModel):
    id: str
    type: str = "message"
    content: list[ContentBlock | dict]  # Allow dicts for flexibility
    model: str
    stop_reason: str
    stop_sequence: str | None = None
    usage: Usage
    role: str = "assistant"


# ============================================================
# CORE FUNCTIONS
# ============================================================


def log_memory(label: str) -> tuple[float, float]:
    """Log current MLX memory state."""
    active = mx.get_active_memory() / (1024**3)
    peak = mx.get_peak_memory() / (1024**3)
    logger.info(f"[MEMORY {label}] Active: {active:.2f}GB, Peak: {peak:.2f}GB")
    return active, peak


def adaptive_chunk_size(cache_pos: int) -> int:
    """Calculate chunk size based on current cache position.

    Larger chunks early (cache small), smaller chunks late (cache large).
    Memory for attention: O(chunk_size × cache_size)
    """
    if cache_pos < 2000:
        return MAX_CHUNK_SIZE  # 4096
    elif cache_pos < 8000:
        return MAX_CHUNK_SIZE // 2  # 2048
    elif cache_pos < 20000:
        return MAX_CHUNK_SIZE // 4  # 1024
    else:
        return MIN_CHUNK_SIZE  # 512


def chunked_prefill(tokens: list[int]) -> list[QuantizedKVCache]:
    """Process tokens in adaptive chunks for memory efficiency.

    Instead of materializing N×N attention matrix, we process in chunks
    and materialize only chunk×cache attention at each step.

    Memory savings: 38-65% peak reduction for long sequences.
    """
    logger.info(f"[CHUNKED PREFILL] Processing {len(tokens)} tokens")
    log_memory("PREFILL_START")

    # Create Q4 cache for each layer
    kv_caches = [
        QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)
        for _ in range(N_LAYERS)
    ]

    tokens_array = mx.array([tokens])
    pos = 0
    chunk_count = 0

    while pos < len(tokens):
        chunk_size = adaptive_chunk_size(pos)
        end = min(pos + chunk_size, len(tokens))

        chunk = tokens_array[:, pos:end]
        chunk_count += 1

        logger.debug(f"[CHUNK {chunk_count}] pos={pos}, size={end-pos}")

        # Forward pass updates kv_caches in-place
        y = model(chunk, cache=kv_caches)

        # CRITICAL: Force evaluation to materialize tensors
        mx.eval(y)

        # CRITICAL: Clear MLX cache to release intermediates
        mx.clear_cache()

        pos = end

    log_memory("PREFILL_END")
    logger.info(f"[CHUNKED PREFILL] Complete: {chunk_count} chunks, cache offset={kv_caches[0].offset}")

    return kv_caches


def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> int:
    """Sample next token from logits with temperature, top_p, top_k.

    Args:
        logits: Logits array of shape (1, vocab_size)
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)

    Returns:
        Sampled token ID
    """
    logits = logits.squeeze(0)  # (vocab_size,)

    # Temperature scaling
    if temperature == 0.0:
        return mx.argmax(logits).item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices = mx.argpartition(-logits, top_k - 1)[:top_k]
        mask = mx.zeros_like(logits)
        mask = mask.at[indices].add(1.0)
        logits = mx.where(mask > 0, logits, mx.array(float("-inf")))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        probs = mx.softmax(sorted_logits)
        cumsum = mx.cumsum(probs)
        cutoff_idx = mx.sum(cumsum < top_p).item() + 1
        cutoff_idx = min(cutoff_idx, logits.shape[-1])
        top_indices = sorted_indices[:cutoff_idx]
        mask = mx.zeros_like(logits)
        mask = mask.at[top_indices].add(1.0)
        logits = mx.where(mask > 0, logits, mx.array(float("-inf")))

    # Sample from distribution
    probs = mx.softmax(logits)
    return mx.random.categorical(probs).item()


def generate_tokens(
    kv_caches: list[QuantizedKVCache],
    tokens_to_process: list[int],
    max_tokens: int,
    last_input_token: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    stop_sequences: list[str] | None = None,
) -> tuple[list[int], str]:
    """Generate tokens with configurable sampling.

    Args:
        kv_caches: List of Q4 KV caches (one per layer)
        tokens_to_process: New tokens to process before generation (may be empty)
        max_tokens: Maximum tokens to generate
        last_input_token: For exact cache hit case
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)
        stop_sequences: Stop generation if any of these strings appear

    Returns:
        (generated_token_ids, decoded_text)
    """
    logger.info(f"[GENERATE] Processing {len(tokens_to_process)} new tokens, generating up to {max_tokens}")
    logger.info(f"[GENERATE] temp={temperature}, top_p={top_p}, top_k={top_k}")

    # Process any new tokens first
    if tokens_to_process:
        y = model(mx.array([tokens_to_process]), cache=kv_caches)
        mx.eval(y)
    elif last_input_token is not None:
        # Exact hit - reprocess last input token to get logits
        for c in kv_caches:
            if c.offset > 0:
                c.offset -= 1
        y = model(mx.array([[last_input_token]]), cache=kv_caches)
        mx.eval(y)
    else:
        raise ValueError("No tokens to process and no last_input_token provided")

    # Generation loop with sampling
    generated = []
    eos_id = tokenizer.eos_token_id
    stop_seqs = stop_sequences or []

    for i in range(max_tokens):
        # Get logits for last position
        logits = y[:, -1, :]

        # Sample next token
        next_token = sample_token(logits, temperature, top_p, top_k)

        # Check for EOS
        if next_token == eos_id:
            logger.info(f"[GENERATE] EOS at token {i+1}")
            break

        generated.append(next_token)

        # Check for stop sequences
        if stop_seqs:
            current_text = tokenizer.decode(generated)
            for seq in stop_seqs:
                if seq in current_text:
                    logger.info(f"[GENERATE] Stop sequence '{seq}' at token {i+1}")
                    # Remove the stop sequence from output
                    idx = current_text.find(seq)
                    if idx >= 0:
                        current_text = current_text[:idx]
                    return generated, current_text

        # Forward pass for next token
        y = model(mx.array([[next_token]]), cache=kv_caches)
        mx.eval(y)

        # Periodic cache clear
        if i % 50 == 0 and i > 0:
            mx.clear_cache()

    # Decode generated tokens
    text = tokenizer.decode(generated)
    logger.info(f"[GENERATE] Complete: {len(generated)} tokens")

    return generated, text


def save_cache_to_disk(
    agent_id: str,
    kv_caches: list[QuantizedKVCache],
    input_tokens: list[int],
    generated_tokens: list[int],
    request_type: str = "main",
) -> Path:
    """Save Q4 cache to safetensors file.

    File format:
    - Tensors: layer_{i}_k_weights, layer_{i}_k_scales, layer_{i}_k_biases (same for v)
    - Metadata:
      - input_token_count: number of input tokens (for prefix matching)
      - total_tokens: full cache size (input + output)
      - input_sequence: input tokens only (for comparison - tokenization is consistent)

    IMPORTANT: We save the FULL cache (input + output). But for prefix matching,
    we only compare INPUT tokens because output token re-tokenization isn't consistent.
    """
    total_tokens = len(input_tokens) + len(generated_tokens)
    path = CACHE_DIR / f"{agent_id}.safetensors"
    logger.info(f"[SAVE] Saving cache: input={len(input_tokens)}, output={len(generated_tokens)}, total={total_tokens}")

    tensors = {}
    for layer_id, cache in enumerate(kv_caches):
        if cache.keys is not None:
            k_w, k_s, k_b = cache.keys
            v_w, v_s, v_b = cache.values

            # Save FULL cache using cache.offset (includes input + generated output)
            offset = cache.offset
            tensors[f"layer_{layer_id}_k_weights"] = k_w[..., :offset, :]
            tensors[f"layer_{layer_id}_k_scales"] = k_s[..., :offset, :]
            tensors[f"layer_{layer_id}_k_biases"] = k_b[..., :offset, :]
            tensors[f"layer_{layer_id}_v_weights"] = v_w[..., :offset, :]
            tensors[f"layer_{layer_id}_v_scales"] = v_s[..., :offset, :]
            tensors[f"layer_{layer_id}_v_biases"] = v_b[..., :offset, :]

    # Metadata - store both input count (for comparison) and total (for offset)
    metadata = {
        "input_token_count": str(len(input_tokens)),
        "total_tokens": str(total_tokens),
        "input_sequence": json.dumps(input_tokens),  # Only input for comparison
    }

    save_file(tensors, str(path), metadata=metadata)

    # Update index
    cache_index[agent_id] = {
        "path": str(path),
        "input_token_count": len(input_tokens),
        "total_tokens": total_tokens,
        "input_sequence": input_tokens,
        "request_type": request_type,  # Track request type for better matching
    }

    logger.info(f"[SAVE] Complete: {len(tensors)} tensors")
    return path


def load_cache_from_disk(agent_id: str) -> tuple[list[QuantizedKVCache] | None, int, int, list[int]]:
    """Load Q4 cache from safetensors file.

    Returns:
        (kv_caches, total_tokens, input_token_count, input_sequence) or (None, 0, 0, []) if not found

    Note: total_tokens is the full cache size (input + output).
          input_token_count and input_sequence are for prefix matching.
    """
    if agent_id not in cache_index:
        return None, 0, 0, []

    info = cache_index[agent_id]
    path = Path(info["path"])

    if not path.exists():
        del cache_index[agent_id]
        return None, 0, 0, []

    logger.info(f"[LOAD] Loading cache from {path}")
    log_memory("LOAD_START")

    kv_caches = []

    with safe_open(str(path), framework="mlx") as f:
        metadata = f.metadata()
        total_tokens = int(metadata.get("total_tokens", "0"))
        input_token_count = int(metadata.get("input_token_count", "0"))

        # Read input sequence from metadata (for prefix comparison)
        input_seq_json = metadata.get("input_sequence", "[]")
        input_sequence = json.loads(input_seq_json)

        for layer_id in range(N_LAYERS):
            cache = QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)

            try:
                k_w = f.get_tensor(f"layer_{layer_id}_k_weights")
                k_s = f.get_tensor(f"layer_{layer_id}_k_scales")
                k_b = f.get_tensor(f"layer_{layer_id}_k_biases")
                v_w = f.get_tensor(f"layer_{layer_id}_v_weights")
                v_s = f.get_tensor(f"layer_{layer_id}_v_scales")
                v_b = f.get_tensor(f"layer_{layer_id}_v_biases")

                cache.keys = (k_w, k_s, k_b)
                cache.values = (v_w, v_s, v_b)
                cache.offset = total_tokens  # Full cache size (input + output)

            except Exception as e:
                logger.warning(f"[LOAD] Layer {layer_id} missing: {e}")

            kv_caches.append(cache)

    # Force evaluation
    for cache in kv_caches:
        if cache.keys is not None:
            mx.eval(cache.keys[0])

    log_memory("LOAD_END")
    logger.info(f"[LOAD] Complete: total={total_tokens}, input={input_token_count}, {N_LAYERS} layers")

    return kv_caches, total_tokens, input_token_count, input_sequence


def generate_agent_id(tokens: list[int]) -> str:
    """Generate deterministic agent ID from full token sequence."""
    # Use ALL tokens for hash - Claude CLI system prompts are 2000+ tokens
    # so first 100 is not enough to distinguish different user queries
    hash_val = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
    return f"msg_{hash_val}"


def find_best_prefix_match(tokens: list[int], request_type: str = "main") -> tuple[str | None, int]:
    """Scan cached agents of same type to find one whose input shares a prefix with tokens.

    Now supports PARTIAL prefix matching - if cached and new tokens share a common
    prefix (e.g., system prompt + tools), we can use the cache up to that point.

    Returns:
        (agent_id, common_prefix_length) or (None, 0) if no useful match
    """
    best_agent = None
    best_length = 0

    # Minimum useful prefix length
    # Main requests: system prompt + tools header is ~100-200 tokens, full tools ~6500 tokens
    # We want to match on at least the system prompt + tools header to save computation
    MIN_USEFUL_PREFIX = 100 if request_type == "main" else 50

    for agent_id, info in cache_index.items():
        # Only match same request type
        if info.get("request_type", "main") != request_type:
            continue

        cached_seq = info.get("input_sequence", [])
        if not cached_seq:
            continue

        # Find common prefix length
        cached_len = len(cached_seq)
        common_len = 0
        for i in range(min(cached_len, len(tokens))):
            if tokens[i] == cached_seq[i]:
                common_len += 1
            else:
                break

        # Check if this is a useful match
        if common_len >= MIN_USEFUL_PREFIX and common_len > best_length:
            best_length = common_len
            best_agent = agent_id
            logger.debug(f"[PREFIX MATCH] {agent_id}: {common_len} tokens common prefix")

    # Count same-type caches
    same_type_count = sum(1 for info in cache_index.values() if info.get("request_type") == request_type)

    if best_agent:
        logger.info(f"[PREFIX SCAN] Best match: {best_agent} with {best_length} tokens (type={request_type})")
    else:
        logger.info(f"[PREFIX SCAN] No prefix match in {same_type_count} {request_type} caches (total: {len(cache_index)})")
        # Debug: find where divergence happens for same-type caches
        if same_type_count > 0 and len(tokens) > 10:
            for agent_id, info in list(cache_index.items())[:3]:
                if info.get("request_type") != request_type:
                    continue
                cached_seq = info.get("input_sequence", [])
                if not cached_seq:
                    continue
                cached_len = len(cached_seq)
                # Find first divergence point
                diverge_at = -1
                for i in range(min(cached_len, len(tokens))):
                    if tokens[i] != cached_seq[i]:
                        diverge_at = i
                        break
                if diverge_at >= 0:
                    logger.info(f"[DIVERGE] {agent_id} (len={cached_len}): diverges at token {diverge_at}")
                    logger.info(f"[DIVERGE] new[{diverge_at}]={tokens[diverge_at]}, cached[{diverge_at}]={cached_seq[diverge_at]}")
                elif cached_len <= len(tokens):
                    logger.info(f"[DIVERGE] {agent_id}: cached_len={cached_len} but should have matched!")

    return best_agent, best_length


def _extract_pending_todos(messages: list) -> list[str]:
    """Find pending todo items from the most recent TodoWrite in the conversation."""
    latest_todos: list[dict] = []
    for msg in messages:
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "TodoWrite":
                    latest_todos = block.get("input", {}).get("todos", [])
    return [
        t["content"] for t in latest_todos
        if isinstance(t, dict) and t.get("status") != "completed" and t.get("content")
    ]


def _build_reverse_input(tool_use_block: dict, tool_result_block: dict) -> str:
    """Build input for the reverse translator from a tool_use + tool_result pair."""
    name = tool_use_block.get("name", "unknown")
    args = tool_use_block.get("input", {})
    is_error = tool_result_block.get("is_error", False)

    result_content = tool_result_block.get("content", "")
    if isinstance(result_content, list):
        result_text = "\n".join(
            b.get("text", str(b)) if isinstance(b, dict) else str(b)
            for b in result_content
        )
    else:
        result_text = str(result_content)

    if name == "Bash":
        args_str = f"command={args.get('command', '')}"
    elif name in ("Write", "Read", "Edit"):
        args_str = f"file_path={args.get('file_path', '')}"
    elif name == "TodoWrite":
        todos = args.get("todos", [])
        items = [f"{t.get('content', '')} ({t.get('status', '')})" for t in todos if isinstance(t, dict)]
        args_str = f"todos=[{', '.join(items)}]"
    elif name in ("Glob", "Grep"):
        args_str = f"pattern={args.get('pattern', '')}"
    else:
        args_str = str(args)[:200]

    if len(result_text) > 500:
        result_text = result_text[:500] + "...(truncated)"

    return (
        f"TOOL: {name}\n"
        f"ARGS: {args_str}\n"
        f"RESULT: {result_text}\n"
        f"ERROR: {str(is_error).lower()}"
    )


def _call_reverse_translator(input_text: str) -> str:
    """Translate a tool result into DeepSeek-native conversation text.

    Uses a separate cached DeepSeek context (slot "rev"). System prompt +
    few-shot example are cached; only the input varies per call.
    Results are also cached in-memory for determinism (same input → same tokens
    in main context, preserving prefix matching).
    """
    cache_key_hash = hashlib.md5(input_text.encode()).hexdigest()
    if cache_key_hash in _reverse_translation_cache:
        logger.info("[REV] In-memory cache hit")
        return _reverse_translation_cache[cache_key_hash]

    conv = [
        {"role": "system", "content": REVERSE_TRANSLATOR_SYSTEM},
        {"role": "user", "content": "TOOL: Bash\nARGS: command=echo hello\nRESULT: hello\nERROR: false"},
        {"role": "assistant", "content": "I ran `echo hello`. Output: hello"},
        {"role": "user", "content": input_text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{REVERSE_TRANSLATOR_SYSTEM}\n\n{input_text}\n\nTranslation:"

    tokens = tokenizer.encode(prompt)

    cache_key = "rev"
    kv_caches = None
    tokens_to_process = tokens

    if cache_key in cache_index:
        loaded = load_cache_from_disk(cache_key)
        if loaded[0] is not None:
            cached_kv, cached_total, _, cached_seq = loaded
            common = 0
            for a, b in zip(cached_seq, tokens):
                if a != b:
                    break
                common += 1
            if common >= 50:
                usable = min(cached_total, common)
                for c in cached_kv:
                    c.offset = usable
                kv_caches = cached_kv
                tokens_to_process = tokens[usable:]
                logger.info(f"[REV] Cache hit: {usable} cached, {len(tokens_to_process)} new")

    if kv_caches is None:
        kv_caches = [
            QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)
            for _ in range(N_LAYERS)
        ]
        tokens_to_process = tokens
        logger.info(f"[REV] Cold start: {len(tokens)} tokens")

    generated_ids, output_text = generate_tokens(
        kv_caches, tokens_to_process, max_tokens=200,
        temperature=0.0, top_p=1.0, top_k=0,
    )

    save_cache_to_disk(cache_key, kv_caches, tokens, generated_ids, "rev")
    del kv_caches
    gc.collect()
    mx.clear_cache()

    result = output_text.strip()
    if "\n\n" in result:
        result = result.split("\n\n")[0]

    logger.info(f"[REV] Translated: {result[:200]}")
    _reverse_translation_cache[cache_key_hash] = result
    return result


def _is_tool_roundtrip(messages: list) -> bool:
    """Stateless: does the last user message have tool_results?

    If yes, forward translator fires to manage the turn.
    User text may also be present (CLI bundles them) — the 'end' action
    only suppresses text, never tool calls, so model can still respond
    to new instructions via tool calls.
    """
    if not messages:
        return False
    last_msg = messages[-1]
    if last_msg.role != "user":
        return False
    if not isinstance(last_msg.content, list):
        return False
    has_tool_result = any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in last_msg.content
    )
    logger.info(f"[GATE] tool_roundtrip={has_tool_result}")
    return has_tool_result


def _last_message_has_user_text(messages: list) -> bool:
    """Stateless: does the last user message contain real user text?

    CLI bundles tool_results + user text + system-reminders in the same message.
    Returns True only if there's non-system-reminder text content.
    """
    if not messages:
        return False
    last_msg = messages[-1]
    if last_msg.role != "user":
        return False
    if isinstance(last_msg.content, str):
        return bool(last_msg.content.strip())
    if not isinstance(last_msg.content, list):
        return False
    for b in last_msg.content:
        if isinstance(b, dict) and b.get("type") == "text":
            text = b.get("text", "")
            cleaned = re.sub(
                r"<system-reminder>.*?</system-reminder>", "",
                text, flags=re.DOTALL,
            ).strip()
            if cleaned:
                return True
    return False


def _build_forward_input(
    tool_uses: list[dict], messages: list, output_text: str
) -> str:
    """Build input for the forward translator (turn management)."""
    pending_todos: list[str] = []
    written_files: list[str] = []
    tool_results: list[dict] = []

    for msg in messages:
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") == "TodoWrite":
                        for t in block.get("input", {}).get("todos", []):
                            if isinstance(t, dict) and t.get("status") != "completed":
                                pending_todos.append(t.get("content", ""))
                    if block.get("name") == "Write":
                        fp = block.get("input", {}).get("file_path", "")
                        if fp:
                            written_files.append(fp)
        if msg.role == "user" and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_results.append({"error": block.get("is_error", False)})

    for tu in tool_uses:
        if tu.get("name") == "Write":
            fp = tu.get("input", {}).get("file_path", "")
            if fp:
                written_files.append(fp)
        if tu.get("name") == "TodoWrite":
            for t in tu.get("input", {}).get("todos", []):
                if isinstance(t, dict) and t.get("status") != "completed":
                    pending_todos.append(t.get("content", ""))

    has_error = any(r.get("error") for r in tool_results)
    tools_called = [tu.get("name", "") for tu in tool_uses]
    has_tool_call = len(tool_uses) > 0

    # Count consecutive errors from the end of the result list
    consecutive_errors = 0
    for r in reversed(tool_results):
        if r.get("error"):
            consecutive_errors += 1
        else:
            break

    return (
        f"TOOLS_CALLED: {', '.join(tools_called) if tools_called else 'none'}\n"
        f"FILES_WRITTEN: {', '.join(written_files) if written_files else 'none'}\n"
        f"PENDING_TODOS: {', '.join(pending_todos) if pending_todos else 'none'}\n"
        f"HAS_ERROR: {str(has_error).lower()}\n"
        f"CONSECUTIVE_ERRORS: {consecutive_errors}\n"
        f"WORKING_DIR: {os.getcwd()}\n"
        f"LATEST_OUTPUT: {'tool_use' if has_tool_call else 'text'}\n"
        f"RESULT_COUNT: {len(tool_results)}"
    )


def _call_forward_translator(input_text: str) -> dict:
    """Forward translator: validate tool calls and decide turn management.

    Uses constrained decoding (logit comparison) for reliable classification.
    Cache slot: "fwd".
    """
    conv = [
        {"role": "system", "content": FORWARD_TRANSLATOR_SYSTEM},
        # Example 1: first error → continue (let model try different approach)
        {"role": "user", "content": "TOOLS_CALLED: Bash\nFILES_WRITTEN: none\nPENDING_TODOS: none\nHAS_ERROR: true\nCONSECUTIVE_ERRORS: 1\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 1"},
        {"role": "assistant", "content": "continue"},
        # Example 2: 3 consecutive errors → end (stop retrying)
        {"role": "user", "content": "TOOLS_CALLED: Bash\nFILES_WRITTEN: none\nPENDING_TODOS: none\nHAS_ERROR: true\nCONSECUTIVE_ERRORS: 3\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 3"},
        {"role": "assistant", "content": "end"},
        # Example 3: files written + pending todos → complete_and_end
        {"role": "user", "content": "TOOLS_CALLED: Write\nFILES_WRITTEN: stack.py\nPENDING_TODOS: Create stack.py\nHAS_ERROR: false\nCONSECUTIVE_ERRORS: 0\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 2"},
        {"role": "assistant", "content": "complete_and_end"},
        # Example 4: pending todos, no files → continue
        {"role": "user", "content": "TOOLS_CALLED: TodoWrite\nFILES_WRITTEN: none\nPENDING_TODOS: Create stack.py, Add tests\nHAS_ERROR: false\nCONSECUTIVE_ERRORS: 0\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: tool_use\nRESULT_COUNT: 1"},
        {"role": "assistant", "content": "continue"},
        # Example 5: no todos, no error → end
        {"role": "user", "content": "TOOLS_CALLED: Bash\nFILES_WRITTEN: none\nPENDING_TODOS: none\nHAS_ERROR: false\nCONSECUTIVE_ERRORS: 0\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 1"},
        {"role": "assistant", "content": "end"},
        # Example 6: long chain done → end
        {"role": "user", "content": "TOOLS_CALLED: Bash\nFILES_WRITTEN: hello.py\nPENDING_TODOS: none\nHAS_ERROR: false\nCONSECUTIVE_ERRORS: 0\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 8"},
        {"role": "assistant", "content": "end"},
        # Example 7: multiple files + todos → complete_and_end
        {"role": "user", "content": "TOOLS_CALLED: Write\nFILES_WRITTEN: stack.py, test_stack.py\nPENDING_TODOS: Create stack.py, Add tests\nHAS_ERROR: false\nCONSECUTIVE_ERRORS: 0\nWORKING_DIR: /Users/dev\nLATEST_OUTPUT: text\nRESULT_COUNT: 4"},
        {"role": "assistant", "content": "complete_and_end"},
        {"role": "user", "content": input_text},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{FORWARD_TRANSLATOR_SYSTEM}\n\n{input_text}\n\nDecision:"

    tokens = tokenizer.encode(prompt)

    cache_key = "fwd"
    kv_caches = None
    tokens_to_process = tokens

    if cache_key in cache_index:
        loaded = load_cache_from_disk(cache_key)
        if loaded[0] is not None:
            cached_kv, cached_total, _, cached_seq = loaded
            common = 0
            for a, b in zip(cached_seq, tokens):
                if a != b:
                    break
                common += 1
            if common >= 50:
                usable = min(cached_total, common)
                for c in cached_kv:
                    c.offset = usable
                kv_caches = cached_kv
                tokens_to_process = tokens[usable:]

    if kv_caches is None:
        kv_caches = [
            QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)
            for _ in range(N_LAYERS)
        ]
        tokens_to_process = tokens

    if tokens_to_process:
        y = model(mx.array([tokens_to_process]), cache=kv_caches)
        mx.eval(y)
    else:
        for c in kv_caches:
            if c.offset > 0:
                c.offset -= 1
        y = model(mx.array([[tokens[-1]]]), cache=kv_caches)
        mx.eval(y)

    logits = y[:, -1, :].squeeze(0)
    answer_tokens = {
        "complete_and_end": tokenizer.encode("complete")[1],
        "end": tokenizer.encode("end")[1],
        "continue": tokenizer.encode("continue")[1],
    }
    scores = {action: logits[tid].item() for action, tid in answer_tokens.items()}
    model_action = max(scores, key=scores.get)

    save_cache_to_disk(cache_key, kv_caches, tokens, [], "fwd")
    del kv_caches
    gc.collect()
    mx.clear_cache()

    # Post-processing: deterministic overrides for edge cases the model
    # gets wrong due to token-level bias (AND conditions are hard for
    # single-token logit comparison).
    best_action = model_action
    fields = {}
    for line in input_text.strip().split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            fields[k.strip()] = v.strip()

    has_error = fields.get("HAS_ERROR", "false") == "true"
    pending_todos = fields.get("PENDING_TODOS", "none")
    has_pending = pending_todos != "none" and pending_todos != ""
    files_written = fields.get("FILES_WRITTEN", "none")
    has_files = files_written != "none" and files_written != ""

    # Minimal guard rail: complete_and_end is meaningless without pending todos
    # (logit comparison can't reliably handle AND conditions)
    if model_action == "complete_and_end" and not has_pending:
        best_action = "end"

    override = "" if best_action == model_action else f" (override: {model_action}→{best_action})"
    logger.info(
        f"[FWD] Decision: {best_action}{override} "
        f"(complete={scores['complete_and_end']:.1f}, "
        f"end={scores['end']:.1f}, "
        f"continue={scores['continue']:.1f})"
    )
    return {"action": best_action}


def messages_to_prompt(messages: list[Message], system: Any = "", tools: list[Tool] | None = None) -> tuple[str, str]:
    """Convert messages to prompt string. Returns (prompt, prefill)."""
    lines = []
    system_text = extract_text(system)

    # Split system text into static and variable parts for better cache prefix matching.
    # The <env> section contains working directory which changes per session.
    # Put static system text first, then tools (static), then variable env, then messages.
    env_marker = "<env>"
    static_system = system_text
    variable_system = ""
    if env_marker in system_text:
        env_idx = system_text.index(env_marker)
        # Find the text before <env> - may include intro line
        intro_end = system_text.rfind("\n", 0, env_idx)
        if intro_end > 0:
            static_system = system_text[:intro_end]
            variable_system = system_text[intro_end:]
        else:
            static_system = system_text[:env_idx]
            variable_system = system_text[env_idx:]

    if static_system:
        lines.append(f"System: {static_system}\n")

    # Add tool definitions in a format DeepSeek understands
    if tools:
        lines.append("\n## Available Functions\n")
        lines.append("You can call these functions when needed. To call a function, use this exact JSON format:")
        lines.append('```json')
        lines.append('{"name": "function_name", "arguments": {"param1": "value1"}}')
        lines.append('```\n')
        lines.append("IMPORTANT: To create or modify files, you MUST use the Write function. Use the actual file path relative to the working directory:")
        lines.append('```json')
        lines.append('{"name": "Write", "arguments": {"file_path": "src/example.py", "content": "file contents here"}}')
        lines.append('```')
        lines.append("Do NOT output file contents in markdown code blocks. Always use Write with a real file path.\n")
        lines.append("To launch a background task or subagent, use the Task function:")
        lines.append('```json')
        lines.append('{"name": "Task", "arguments": {"description": "short desc", "prompt": "what to do", "subagent_type": "general-purpose"}}')
        lines.append('```')
        lines.append("To track COMPLEX multi-step tasks (3+ steps), use TodoWrite. For simple tasks (single file, single command), skip TodoWrite and call Write or Bash directly:")
        lines.append('```json')
        lines.append('{"name": "TodoWrite", "arguments": {"todos": [{"content": "task name", "status": "pending", "activeForm": "Doing task"}]}}')
        lines.append('```')
        lines.append("After calling TodoWrite, you MUST call Write or Bash next. Never call TodoWrite twice in a row.\n")
        lines.append("Available functions:\n")
        for tool in tools:
            lines.append(f"### {tool.name}")
            if tool.description:
                lines.append(f"{tool.description[:2000]}")
            if tool.input_schema and tool.input_schema.get("properties"):
                props = tool.input_schema["properties"]
                required = tool.input_schema.get("required", [])
                lines.append("Parameters:")
                for name, spec in props.items():
                    req_marker = " (required)" if name in required else ""
                    desc = spec.get("description", "")[:300]
                    param_type = spec.get('type', 'any')
                    lines.append(f"  - {name}: {param_type}{req_marker} - {desc}")
                    # Show nested structure for arrays/objects
                    if param_type == "array" and "items" in spec:
                        items = spec["items"]
                        if items.get("type") == "object" and "properties" in items:
                            lines.append(f"    Array items must have: {list(items['properties'].keys())}")
                    elif param_type == "object" and "properties" in spec:
                        lines.append(f"    Object must have: {list(spec['properties'].keys())}")
            lines.append("")

    # Append variable system parts (env section) after static prefix
    if variable_system:
        lines.append(variable_system.strip())

    total_tool_results = 0

    for i, msg in enumerate(messages):
        # For user messages with tool_results: use the reverse translator
        # to produce DeepSeek-native text instead of hardcoded templates.
        if msg.role == "user" and isinstance(msg.content, list):
            tool_result_blocks = [
                b for b in msg.content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_result_blocks:
                # Find preceding assistant message to match tool_use blocks
                prev_assistant = messages[i - 1] if i > 0 and messages[i - 1].role == "assistant" else None
                tool_use_map: dict[str, dict] = {}
                if prev_assistant and isinstance(prev_assistant.content, list):
                    for b in prev_assistant.content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            tool_use_map[b.get("id", "")] = b

                translated_parts: list[str] = []
                for b in msg.content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        matched = tool_use_map.get(b.get("tool_use_id", ""))
                        if matched:
                            rev_input = _build_reverse_input(matched, b)
                            translated = _call_reverse_translator(rev_input)
                            translated_parts.append(translated)
                        else:
                            rc = b.get("content", "")
                            translated_parts.append(str(rc) if not isinstance(rc, list) else str(rc))
                    elif isinstance(b, dict) and b.get("type") == "text":
                        text = b.get("text", "")
                        if text.strip():
                            translated_parts.append(text)

                content_text = "\n".join(translated_parts)
                lines.append(f"User: {content_text}")
                logger.info(f"[MSG {i} REV-TRANSLATED] user: {content_text[:300]}")
                total_tool_results += len(tool_result_blocks)
                continue

        # Non-tool-result messages: use extract_text as before
        content_text = extract_text(msg.content, role=msg.role)
        lines.append(f"{msg.role.capitalize()}: {content_text}")
        logger.info(f"[MSG {i} CONVERTED] {msg.role}: {content_text[:300]}")

    logger.info(f"[PROMPT] total_tool_results={total_tool_results}")

    lines.append("Assistant:")
    return "\n".join(lines), "", total_tool_results


# ============================================================
# API ENDPOINTS
# ============================================================

# DeepSeek tool call markers - stop after FIRST tool call completes
DEEPSEEK_TOOL_STOP_SEQUENCES = [
    "<｜tool▁call▁end｜>",   # Stop after first tool call (singular)
    "<|tool▁call▁end|>",    # Alternative encoding
]

# DeepSeek tool call markers for parsing
TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_SEP = "<｜tool▁sep｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"


def sanitize_terminal_output(text: str) -> str:
    """Remove ANSI escape sequences and control characters that can freeze terminals.

    This prevents model output from containing sequences that could put zsh
    in alternate screen mode, trigger vi mode, or cause other terminal issues.
    """
    # Remove ANSI escape sequences (CSI sequences like \x1b[...)
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # Remove OSC sequences (like \x1b]...BEL)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    # Remove other escape sequences
    text = re.sub(r'\x1b[PX^_][^\x1b]*\x1b\\', '', text)
    # Remove remaining bare escapes
    text = re.sub(r'\x1b.', '', text)
    # Remove control characters except tab, newline, carriage return
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def parse_tool_calls(text: str, available_tools: list[Tool] | None = None) -> tuple[str, list[dict]]:
    """Parse DeepSeek tool call format and extract tool_use blocks.

    DeepSeek format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name
    ... code/args ...
    <｜tool▁call▁end｜>

    Also tries to parse JSON format: {"name": "...", "arguments": {...}}

    Returns:
        tuple of (remaining_text, list of tool_use dicts)
    """
    tool_uses = []
    remaining_text = text.strip()

    # Build case-insensitive tool name lookup
    tool_name_map = {}
    if available_tools:
        for t in available_tools:
            tool_name_map[t.name.lower()] = t.name

    # First, try JSON format (preferred) - handles nested objects/arrays
    try:
        # Find start of JSON with "name" key
        name_match = re.search(r'\{\s*"name"\s*:', text)
        if name_match:
            start_idx = name_match.start()
            # Use bracket counting to find matching closing brace
            depth = 0
            end_idx = start_idx
            for i, c in enumerate(text[start_idx:]):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = start_idx + i + 1
                        break

            # If bracket counting didn't reach depth 0, model may have
            # omitted closing braces (common with forced prefill).
            # Try appending missing braces.
            if end_idx <= start_idx and 0 < depth <= 2:
                json_str = text[start_idx:]
                # Strip trailing markdown fences
                json_str = re.sub(r'\s*```\s*$', '', json_str)
                json_str = json_str.rstrip() + '}' * depth
                logger.info(f"[TOOL PARSE] Bracket repair: added {depth} closing brace(s)")
                try:
                    obj = json.loads(json_str)
                    if "name" in obj and "arguments" in obj:
                        end_idx = len(text)  # Consumed everything
                except json.JSONDecodeError:
                    obj = None
            elif end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    obj = json.loads(json_str)
                except json.JSONDecodeError as e:
                    obj = None
                    logger.debug(f"[TOOL PARSE] JSON decode failed: {e}")
            else:
                obj = None

            if obj and "name" in obj and "arguments" in obj:
                parsed_name = obj["name"]
                args = obj["arguments"] if isinstance(obj["arguments"], dict) else {}

                # Case-insensitive tool name lookup
                canonical_name = tool_name_map.get(parsed_name.lower())
                if canonical_name:
                    tool_id = f"toolu_{hashlib.md5(f'{canonical_name}{time.time()}'.encode()).hexdigest()[:24]}"
                    tool_uses.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": canonical_name,
                        "input": args,
                    })
                    # Remove the JSON and surrounding markdown fences from remaining text
                    remaining_text = text[:start_idx] + text[end_idx:]
                    remaining_text = re.sub(r'```json\s*```', '', remaining_text)
                    remaining_text = re.sub(r'```\s*```', '', remaining_text)
                    remaining_text = remaining_text.strip()
                    logger.info(f"[TOOL PARSE] Found JSON tool call: {parsed_name} -> {canonical_name}")
                    return remaining_text, tool_uses
                else:
                    logger.warning(f"[TOOL PARSE] Unknown tool: {parsed_name}, available: {list(tool_name_map.keys())}")

    except Exception as e:
        logger.debug(f"[TOOL PARSE] JSON parse failed: {e}")

    # Second, try DeepSeek native format
    MAX_TOOL_CALLS = 10  # Cap to prevent degenerate output loops
    if TOOL_CALL_BEGIN in text:
        logger.info(f"[TOOL PARSE] Found DeepSeek markers in output")

        # Split on tool call begin markers
        parts = text.split(TOOL_CALL_BEGIN)
        remaining_parts = [parts[0]]  # Keep text before first marker

        for part in parts[1:]:
            if len(tool_uses) >= MAX_TOOL_CALLS:
                logger.warning(f"[TOOL PARSE] Hit max tool call limit ({MAX_TOOL_CALLS}), ignoring rest")
                break

            # Each part should be: function<｜tool▁sep｜>name\ncode...<｜tool▁call▁end｜>
            if TOOL_SEP in part:
                sep_parts = part.split(TOOL_SEP, 1)
                if len(sep_parts) >= 2:
                    call_type = sep_parts[0].strip()  # Usually "function"
                    rest = sep_parts[1]

                    # Extract function name (first word/identifier after sep)
                    name_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)', rest)
                    if name_match:
                        parsed_name = name_match.group(1)

                        # Case-insensitive tool name lookup
                        canonical_name = tool_name_map.get(parsed_name.lower())
                        if canonical_name:
                            # Try to extract arguments
                            args = {}
                            args_text = rest[name_match.end():]
                            if TOOL_CALL_END in args_text:
                                args_text = args_text.split(TOOL_CALL_END)[0]

                            # Strip markdown code fences from args
                            args_text = re.sub(r'```(?:json)?\s*', '', args_text)
                            args_text = args_text.strip()

                            # Use bracket counting for nested JSON (not flat regex)
                            json_start = args_text.find('{')
                            if json_start >= 0:
                                depth = 0
                                json_end = json_start
                                for i, c in enumerate(args_text[json_start:]):
                                    if c == '{':
                                        depth += 1
                                    elif c == '}':
                                        depth -= 1
                                        if depth == 0:
                                            json_end = json_start + i + 1
                                            break
                                if json_end > json_start:
                                    try:
                                        args = json.loads(args_text[json_start:json_end])
                                    except json.JSONDecodeError:
                                        pass

                            # Fallback: regex extract for Write with malformed JSON
                            if not args and canonical_name == "Write":
                                fp = re.search(r'"file_path"\s*:\s*"([^"]*)"', args_text)
                                ct = re.search(r'"content"\s*:\s*"(.*?)(?:"\s*[},]|$)', args_text, re.DOTALL)
                                if fp and ct:
                                    content_val = ct.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                                    args = {"file_path": fp.group(1), "content": content_val}
                                    logger.info(f"[TOOL PARSE] Regex fallback for Write: {fp.group(1)}")

                            # Skip empty tool calls (degenerate output)
                            if not args:
                                logger.warning(f"[TOOL PARSE] Skipping empty tool call: {parsed_name}")
                                continue

                            tool_id = f"toolu_{hashlib.md5(f'{canonical_name}{time.time()}'.encode()).hexdigest()[:24]}"
                            tool_uses.append({
                                "type": "tool_use",
                                "id": tool_id,
                                "name": canonical_name,
                                "input": args,
                            })
                            logger.info(f"[TOOL PARSE] Extracted tool call: {parsed_name} -> {canonical_name}")
                        else:
                            logger.warning(f"[TOOL PARSE] Unknown tool: {parsed_name}, available: {list(tool_name_map.keys())}")
                            remaining_parts.append(part)  # Keep unrecognized parts
                    else:
                        remaining_parts.append(part)
                else:
                    remaining_parts.append(part)
            else:
                remaining_parts.append(part)

        # Reconstruct remaining text without tool markers
        remaining_text = "".join(remaining_parts)
        # Clean up any remaining markers
        remaining_text = remaining_text.replace(TOOL_CALLS_BEGIN, "")
        remaining_text = remaining_text.replace(TOOL_CALLS_END, "")
        remaining_text = remaining_text.replace(TOOL_CALL_END, "")
        remaining_text = remaining_text.strip()

    # Fallback: model outputs just arguments without "name" wrapper (last resort)
    # e.g., {"command": "..."} instead of {"name": "Bash", "arguments": {"command": "..."}}
    if not tool_uses:
        try:
            # Find start of any JSON object using bracket counting
            json_start = remaining_text.find('{')
            if json_start >= 0:
                depth = 0
                json_end = json_start
                for i, c in enumerate(remaining_text[json_start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = json_start + i + 1
                            break
                if json_end > json_start:
                    json_str = remaining_text[json_start:json_end]
                    try:
                        obj = json.loads(json_str)
                        # Infer tool from argument keys
                        inferred_name = None
                        if "command" in obj:
                            inferred_name = "Bash"
                        elif "prompt" in obj and "subagent_type" in obj:
                            inferred_name = "Task"
                        elif "description" in obj and "subagent_type" in obj:
                            inferred_name = "Task"
                        elif "file_path" in obj and "content" not in obj:
                            inferred_name = "Read"
                        elif "file_path" in obj and "content" in obj:
                            inferred_name = "Write"
                        elif "pattern" in obj:
                            inferred_name = "Grep"
                        elif "query" in obj:
                            inferred_name = "WebSearch"
                        elif "todos" in obj:
                            inferred_name = "TodoWrite"

                        if inferred_name:
                            canonical_name = tool_name_map.get(inferred_name.lower())
                            if canonical_name:
                                tool_id = f"toolu_{hashlib.md5(f'{canonical_name}{time.time()}'.encode()).hexdigest()[:24]}"
                                tool_uses.append({
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": canonical_name,
                                    "input": obj,
                                })
                                remaining_text = remaining_text[:json_start] + remaining_text[json_end:]
                                remaining_text = remaining_text.strip()
                                logger.info(f"[TOOL PARSE] Inferred tool from args: {inferred_name} -> {canonical_name}")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.debug(f"[TOOL PARSE] Fallback parse failed: {e}")

    # Convert markdown code blocks to tool_use for CLI
    # This is safe because line 237 converts tool_use back to markdown when sending
    # history to the model, so model always sees consistent markdown format.
    if not tool_uses:
        # Match ```bash or ```shell code blocks
        bash_match = re.search(r'```(?:bash|shell|sh)\n(.*?)```', remaining_text, re.DOTALL)
        if bash_match and tool_name_map.get('bash'):
            command = bash_match.group(1).strip()
            if command:
                tool_id = f"toolu_{hashlib.md5(f'Bash{time.time()}'.encode()).hexdigest()[:24]}"
                tool_uses.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name_map['bash'],
                    "input": {"command": command},
                })
                # Keep explanation text, just remove the code block
                remaining_text = re.sub(r'```(?:bash|shell|sh)\n.*?```', '', remaining_text, flags=re.DOTALL).strip()
                logger.info(f"[TOOL PARSE] Converted bash markdown to tool_use: {command[:50]}...")

        # Match ```python code blocks -> run via python3 -c
        if not tool_uses:
            python_match = re.search(r'```python\n(.*?)```', remaining_text, re.DOTALL)
            if python_match and tool_name_map.get('bash'):
                code = python_match.group(1).strip()
                if code:
                    escaped_code = code.replace("'", "'\"'\"'")
                    tool_id = f"toolu_{hashlib.md5(f'Bash{time.time()}'.encode()).hexdigest()[:24]}"
                    tool_uses.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name_map['bash'],
                        "input": {"command": f"python3 -c '{escaped_code}'"},
                    })
                    remaining_text = re.sub(r'```python\n.*?```', '', remaining_text, flags=re.DOTALL).strip()
                    logger.info(f"[TOOL PARSE] Converted python markdown to bash tool_use")

    # Always clean up any DeepSeek markers from remaining text
    remaining_text = remaining_text.replace(TOOL_CALLS_BEGIN, "")
    remaining_text = remaining_text.replace(TOOL_CALLS_END, "")
    remaining_text = remaining_text.replace(TOOL_CALL_BEGIN, "")
    remaining_text = remaining_text.replace(TOOL_CALL_END, "")
    remaining_text = remaining_text.replace(TOOL_SEP, "")
    # Clean up "function" keyword that appears after tool_call_begin
    remaining_text = re.sub(r'\bfunction\b\s*', '', remaining_text)
    remaining_text = remaining_text.strip()

    return remaining_text, tool_uses


def _get_stop_sequences(request: MessagesRequest) -> list[str]:
    """Get stop sequences including DeepSeek tool markers if tools are provided."""
    stop_seqs = list(request.stop_sequences) if request.stop_sequences else []
    if request.tools:
        # Add DeepSeek tool call end markers to stop generation after tool call
        for marker in DEEPSEEK_TOOL_STOP_SEQUENCES:
            if marker not in stop_seqs:
                stop_seqs.append(marker)
    return stop_seqs


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    global model, tokenizer

    logger.info(f"Loading model: {MODEL_ID}")
    log_memory("STARTUP")

    model, tokenizer = load(MODEL_ID)

    # Fix tokenizer max length
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < 100000:
            tokenizer.model_max_length = 100000
            logger.info("Set tokenizer.model_max_length = 100000")

    log_memory("MODEL_LOADED")
    logger.info("Model loaded successfully")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


async def _generate_sse_events(response: MessagesResponse):
    """Convert a complete MessagesResponse into Anthropic SSE stream events.

    Emits: message_start, content_block_start/delta/stop for each block,
    message_delta (with stop_reason), message_stop.
    """
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # message_start - initial message with empty content
    msg_data = response.model_dump(exclude_none=True)
    msg_data["content"] = []
    msg_data["stop_reason"] = None
    yield sse("message_start", {"type": "message_start", "message": msg_data})

    # Emit each content block
    for idx, block in enumerate(response.content):
        block_data = block if isinstance(block, dict) else block.model_dump(exclude_none=True)
        block_type = block_data.get("type", "text")

        if block_type == "text":
            text_content = block_data.get("text", "")
            # content_block_start with empty text
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            # Send text in one delta
            if text_content:
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text_content},
                })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

        elif block_type == "tool_use":
            # content_block_start with tool info (empty input)
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block_data.get("id", ""),
                    "name": block_data.get("name", ""),
                    "input": {},
                },
            })
            # Send input as JSON delta
            input_data = block_data.get("input", {})
            if input_data:
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(input_data)},
                })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    # message_delta with stop_reason
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": response.stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": response.usage.output_tokens},
    })

    # message_stop
    yield sse("message_stop", {"type": "message_stop"})


@app.post("/v1/messages")
async def create_message(request_body: MessagesRequest, request: Request) -> JSONResponse:
    """Anthropic Messages API endpoint."""
    start_time = time.time()
    request_id = int(time.time() * 1000)  # Unique request ID

    # Initialize request log entry (will be completed and saved at end)
    # Capture FULL data for semantic analysis and optimization
    system_text = str(request_body.system) if request_body.system else None
    request_log = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}",
        "request": {
            "model": request_body.model,
            "max_tokens": request_body.max_tokens,
            "temperature": request_body.temperature,
            "top_p": request_body.top_p,
            "top_k": request_body.top_k,
            "stream": request_body.stream,
            # Full system prompt for semantic analysis
            "system_text": system_text,
            "system_length": len(system_text) if system_text else 0,
            # Full messages
            "messages": [{"role": m.role, "content": str(m.content)} for m in request_body.messages],
            "n_messages": len(request_body.messages),
            # Full tool definitions with schemas for commonality analysis
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in request_body.tools
            ] if request_body.tools else [],
            "n_tools": len(request_body.tools) if request_body.tools else 0,
            "tool_names": [t.name for t in request_body.tools] if request_body.tools else [],
            "stop_sequences": request_body.stop_sequences,
            # Additional API params
            "tool_choice": str(request_body.tool_choice) if request_body.tool_choice else None,
            "thinking": {"type": request_body.thinking.type, "budget": request_body.thinking.budget_tokens} if request_body.thinking else None,
        },
        "headers": {
            "anthropic_beta": request.headers.get("anthropic-beta", ""),
            "user_agent": request.headers.get("user-agent", ""),
            "session_id": request.headers.get("X-Session-ID", ""),
        },
        "timing": {},      # Timing breakdown
        "tokens": {},      # Token analysis
        "cache": {},       # Cache analysis
        "processing": {},  # Processing stats
        "response": {},    # Response data
    }

    # Cap max_tokens to prevent runaway generation (context window is 100K)
    max_tokens = min(request_body.max_tokens, 16000)

    # Log headers and request parameters for debugging
    session_id = request.headers.get("X-Session-ID")
    anthropic_beta = request.headers.get("anthropic-beta", "")
    n_tools = len(request_body.tools) if request_body.tools else 0

    # Resolve generation parameters
    # ALWAYS use greedy (temp=0) for this DeepSeek model - random sampling produces garbage
    # Claude Code CLI sends temp=1.0 explicitly, but this model can't handle it
    temperature = 0.0  # Override any client value
    top_p = request_body.top_p if request_body.top_p is not None else 1.0
    top_k = request_body.top_k if request_body.top_k is not None else 0

    logger.info(f"[REQUEST] msgs={len(request_body.messages)}, max_tokens={max_tokens}, tools={n_tools}")
    logger.info(f"[PARAMS] temp={temperature}, top_p={top_p}, top_k={top_k}, stream={request_body.stream}")
    if anthropic_beta:
        logger.info(f"[BETA] {anthropic_beta}")
    if request_body.betas:
        logger.info(f"[BETAS] {request_body.betas}")
    if request_body.tools:
        logger.info(f"[TOOLS] {[t.name for t in request_body.tools]}")
    if request_body.tool_choice:
        logger.info(f"[TOOL_CHOICE] {request_body.tool_choice}")
    if request_body.thinking:
        logger.info(f"[THINKING] type={request_body.thinking.type}, budget={request_body.thinking.budget_tokens}")
    if request_body.output_format:
        logger.info(f"[OUTPUT_FORMAT] {request_body.output_format}")
    for i, msg in enumerate(request_body.messages):
        logger.info(f"[RAW MSG {i}] role={msg.role}, content={str(msg.content)[:200]}")
    log_memory("REQUEST_START")

    # 1. Convert to prompt and tokenize
    tokenize_start = time.time()
    prompt, prefill, total_tool_results = messages_to_prompt(
        request_body.messages, request_body.system, request_body.tools
    )

    # Cap max_tokens during tool roundtrips — model only needs enough for
    # 1-2 tool calls, not 10K tokens of rambling. Checked before generation.
    if _is_tool_roundtrip(request_body.messages):
        max_tokens = min(max_tokens, 2000)
        logger.info(f"[TOOL RT] Capped max_tokens to {max_tokens}")

    # Absolute safety cap: if processor keeps saying "continue", hard-stop here
    if total_tool_results >= ABSOLUTE_SAFETY_CAP:
        logger.info(f"[ABSOLUTE SAFETY CAP] {total_tool_results} results, forcing end_turn")
        cap_response = MessagesResponse(
            id=f"msg_{int(time.time())}",
            content=[{"type": "text", "text": "Done."}],
            model=request_body.model,
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=1),
        )
        if request_body.stream:
            return StreamingResponse(
                _generate_sse_events(cap_response),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return JSONResponse(content=cap_response.model_dump(exclude_none=True))

    tokens = tokenizer.encode(prompt)
    tokenize_time = time.time() - tokenize_start
    logger.info(f"[TOKENIZE] {len(tokens)} tokens")
    logger.info(f"[PROMPT] {prompt[:500]}...")  # Log first 500 chars of prompt

    # Capture token data for semantic analysis
    request_log["tokens"] = {
        "total_input": len(tokens),
        "prompt_length_chars": len(prompt),
        # Store first/last tokens for prefix analysis (full sequence too large)
        "first_100_tokens": tokens[:100],
        "last_50_tokens": tokens[-50:] if len(tokens) > 50 else tokens,
    }
    request_log["timing"]["tokenize_ms"] = round(tokenize_time * 1000, 2)

    # Determine request type for cache segregation
    system_len = len(str(request_body.system)) if request_body.system else 0
    if system_len > 10000 and n_tools > 0:
        request_type = "main"  # Full conversation with tools
    elif system_len > 0 and n_tools == 0:
        request_type = "title"  # Title generation
    elif n_tools > 0 and system_len == 0:
        request_type = "count"  # Token counting
    else:
        request_type = "other"
    logger.info(f"[REQUEST TYPE] {request_type} (system={system_len}, tools={n_tools})")

    # 2. Find best prefix match among cached agents OF SAME TYPE
    session_id = request.headers.get("X-Session-ID")
    prefix_length = 0
    if session_id:
        agent_id = f"sess_{session_id}"
        matched_agent = agent_id
        prefix_length = len(tokens)  # Assume full match for session-based
    else:
        # Scan caches of same type to find one whose input shares a prefix
        matched_agent, prefix_length = find_best_prefix_match(tokens, request_type)
        if matched_agent:
            agent_id = matched_agent  # Reuse the matched agent's cache
        else:
            agent_id = generate_agent_id(tokens)  # New agent

    # 3. Load existing cache (if we found a match)
    if matched_agent:
        kv_caches, total_cached, input_cached, input_sequence = load_cache_from_disk(matched_agent)
    else:
        kv_caches, total_cached, input_cached, input_sequence = None, 0, 0, []

    cache_hit = kv_caches is not None
    tokens_to_process = []
    cache_lookup_time = time.time() - start_time - tokenize_time

    # Capture cache analysis data
    request_log["cache"] = {
        "cache_hit": cache_hit,
        "matched_agent": matched_agent,
        "prefix_length": prefix_length,
        "total_cached": total_cached if cache_hit else 0,
        "input_cached": input_cached if cache_hit else 0,
        "n_cached_agents": len(cache_index),
        "cached_agent_types": list(set(info.get("request_type", "unknown") for info in cache_index.values())),
    }
    request_log["timing"]["cache_lookup_ms"] = round(cache_lookup_time * 1000, 2)

    if cache_hit:
        logger.info(f"[CACHE HIT] agent={agent_id}, total_cached={total_cached}, input_cached={input_cached}, prefix_match={prefix_length}")

        # Use the smaller of: total_cached (full cache), or prefix_length (common prefix)
        # This handles partial prefix matches where cached and new tokens diverge
        usable_cache_len = min(total_cached, prefix_length) if prefix_length > 0 else total_cached

        # Truncate KV cache to usable length
        for c in kv_caches:
            if c.offset > usable_cache_len:
                c.offset = usable_cache_len

        # Process all tokens after the usable cache
        if len(tokens) > usable_cache_len:
            tokens_to_process = tokens[usable_cache_len:]
            logger.info(f"[PARTIAL PREFIX] Using {usable_cache_len} cached tokens, processing {len(tokens_to_process)} new")
        else:
            # Exact match or shorter - reprocess last token to get logits
            tokens_to_process = [tokens[-1]]
            for c in kv_caches:
                if c.offset > 0:
                    c.offset = min(c.offset, len(tokens)) - 1
            logger.info(f"[EXACT/SHORT] Reprocessing last token for logits")

    if not cache_hit:
        logger.info(f"[CACHE MISS] agent={agent_id}")

        # Cold start: use chunked prefill for long prompts
        if len(tokens) >= CHUNKED_PREFILL_THRESHOLD:
            kv_caches = chunked_prefill(tokens)
            tokens_to_process = []  # All tokens already processed
        else:
            # Short prompt: process directly
            kv_caches = [
                QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)
                for _ in range(N_LAYERS)
            ]
            tokens_to_process = tokens

    # 4. Generate
    # Pass last_input_token for cases where tokens_to_process is empty (chunked prefill)
    last_token = tokens[-1] if not tokens_to_process else None
    generated_ids, output_text = generate_tokens(
        kv_caches,
        tokens_to_process,
        max_tokens,
        last_input_token=last_token,
        temperature=temperature,  # Use resolved value (None -> 1.0)
        top_p=top_p,
        top_k=top_k,
        stop_sequences=_get_stop_sequences(request_body),
    )

    # 5. Save FULL cache (input + output) so next request doesn't reprocess
    save_cache_to_disk(agent_id, kv_caches, tokens, generated_ids, request_type)

    # 6. Cleanup
    del kv_caches
    gc.collect()
    mx.clear_cache()

    # 7. Build response
    elapsed = time.time() - start_time
    log_memory("REQUEST_END")
    logger.info(f"[DONE] {len(generated_ids)} tokens in {elapsed:.2f}s ({len(generated_ids)/elapsed:.1f} tok/s)")

    # Sanitize output to prevent terminal control sequence issues
    output_text = sanitize_terminal_output(output_text)

    # Parse tool calls from output
    remaining_text, tool_uses = parse_tool_calls(output_text, request_body.tools)

    # Build content blocks
    content_blocks: list[dict] = []

    # When tool_uses exist, return ONLY the tool_use - no text
    # Text was causing confusion and infinite loops
    if tool_uses:
        for tool_use in tool_uses:
            content_blocks.append(tool_use)
    else:
        # No tool uses - include text normally
        if remaining_text:
            clean_text = sanitize_terminal_output(remaining_text.strip())
            # Filter out empty JSON code blocks
            clean_text = re.sub(r'```json\s*```', '', clean_text)
            clean_text = re.sub(r'```\s*```', '', clean_text)
            clean_text = clean_text.strip()
            if clean_text:
                content_blocks.append({"type": "text", "text": clean_text})

    # Determine stop reason
    if tool_uses:
        stop_reason = "tool_use"
        logger.info(f"[TOOL USE] Detected {len(tool_uses)} tool call(s)")
    elif len(generated_ids) < max_tokens:
        stop_reason = "end_turn"
    else:
        stop_reason = "max_tokens"

    # Forward translator: stateless turn management.
    # Fires whenever the last user message has tool_results.
    # "end" only suppresses text — tool calls always pass through
    # (the model is actively working). Safety cap handles runaways.
    is_tool_rt = _is_tool_roundtrip(request_body.messages)
    if is_tool_rt:
        fwd_input = _build_forward_input(
            tool_uses, request_body.messages, output_text
        )
        logger.info(f"[FWD] State: {fwd_input}")
        decision = _call_forward_translator(fwd_input)
        action = decision.get("action", "continue")

        if action == "complete_and_end":
            pending = _extract_pending_todos(request_body.messages)
            if pending:
                completed = [
                    {"content": item, "status": "completed", "activeForm": "Completed"}
                    for item in pending
                ]
                tool_id = f"toolu_{hashlib.md5(f'proc{time.time()}'.encode()).hexdigest()[:24]}"
                content_blocks = [{
                    "type": "tool_use", "id": tool_id,
                    "name": "TodoWrite", "input": {"todos": completed},
                }]
                stop_reason = "tool_use"
                logger.info(f"[FWD] complete_and_end: TodoWrite(completed) for {len(completed)} items")
            else:
                content_blocks = [{"type": "text", "text": "Done."}]
                stop_reason = "end_turn"
                logger.info("[FWD] complete_and_end (no pending todos)")

        elif action == "end":
            if tool_uses and _last_message_has_user_text(request_body.messages):
                # Model responding to bundled user instruction — let tool calls through
                logger.info("[FWD] end but user text present → passing through tool calls")
            else:
                content_blocks = [{"type": "text", "text": "Done."}]
                stop_reason = "end_turn"
                logger.info("[FWD] end → Done.")

        # "continue" → use model's output as-is (already in content_blocks)

    response = MessagesResponse(
        id=f"msg_{agent_id}",
        content=content_blocks if content_blocks else [{"type": "text", "text": ""}],
        model=request_body.model,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=len(tokens),
            output_tokens=len(generated_ids),
            cache_creation_input_tokens=len(tokens) if not cache_hit else 0,
            cache_read_input_tokens=prefix_length if cache_hit else 0,
        ),
    )

    # 8. Save complete request/response log to timestamped test_set file
    request_log["processing"] = {
        "request_type": request_type,
        "agent_id": agent_id,
        "total_input_tokens": len(tokens),
        "tokens_processed": len(tokens_to_process),
        "tokens_from_cache": prefix_length if cache_hit else 0,
        "output_tokens": len(generated_ids),
        "cache_savings_pct": round((prefix_length / len(tokens)) * 100, 1) if cache_hit and len(tokens) > 0 else 0,
    }
    request_log["timing"]["total_ms"] = round(elapsed * 1000, 2)
    request_log["timing"]["generation_ms"] = round((elapsed - cache_lookup_time - tokenize_time) * 1000, 2)
    request_log["timing"]["tokens_per_second"] = round(len(generated_ids) / elapsed, 1) if elapsed > 0 else 0

    request_log["response"] = {
        "output_text": output_text,
        "remaining_text": remaining_text,
        "output_tokens": len(generated_ids),
        "stop_reason": stop_reason,
        "tool_uses": [{"name": t.get("name"), "input": t.get("input")} for t in tool_uses] if tool_uses else [],
        "n_tool_uses": len(tool_uses),
        "content_blocks": len(content_blocks),
    }

    # Append to timestamped test_set file (won't overwrite previous sessions)
    try:
        # Read existing entries from this session's file
        if TEST_SET_PATH.exists():
            with open(TEST_SET_PATH, "r") as f:
                test_set = json.load(f)
        else:
            test_set = {
                "session": {
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": MODEL_ID,
                    "filename": TEST_SET_FILENAME,
                },
                "requests": [],
                "summary": {},
            }

        test_set["requests"].append(request_log)

        # Update running summary statistics
        reqs = test_set["requests"]
        cache_hits = sum(1 for r in reqs if r.get("cache", {}).get("cache_hit", False))
        by_type = {}
        for r in reqs:
            rt = r.get("processing", {}).get("request_type", "unknown")
            if rt not in by_type:
                by_type[rt] = {"count": 0, "cache_hits": 0, "total_tokens": 0, "total_ms": 0}
            by_type[rt]["count"] += 1
            if r.get("cache", {}).get("cache_hit", False):
                by_type[rt]["cache_hits"] += 1
            by_type[rt]["total_tokens"] += r.get("processing", {}).get("total_input_tokens", 0)
            by_type[rt]["total_ms"] += r.get("timing", {}).get("total_ms", 0)

        test_set["summary"] = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_requests": len(reqs),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / len(reqs) * 100, 1) if reqs else 0,
            "by_request_type": by_type,
            "total_tokens_processed": sum(r.get("processing", {}).get("total_input_tokens", 0) for r in reqs),
            "total_time_ms": sum(r.get("timing", {}).get("total_ms", 0) for r in reqs),
        }

        with open(TEST_SET_PATH, "w") as f:
            json.dump(test_set, f, indent=2)
        logger.info(f"[TEST SET] Logged request #{len(reqs)} to {TEST_SET_PATH}")
    except Exception as e:
        logger.warning(f"[TEST SET] Failed to save: {e}")

    # Return response in the format the client expects
    if request_body.stream:
        return StreamingResponse(
            _generate_sse_events(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return JSONResponse(content=response.model_dump(exclude_none=True))


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    logger.info(f"Starting minimal semantic cache server on port {PORT}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Test set log: {TEST_SET_PATH}")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
