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


def extract_text(content: Any) -> str:
    """Extract text from content that may be string or list of blocks.

    Converts tool_use/tool_result to formats DeepSeek understands:
    - tool_use: JSON function call format
    - tool_result: Function output format
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    # Skip thinking blocks in prompt (internal reasoning)
                    pass
                elif block.get("type") == "tool_use":
                    # Format as JSON function call that DeepSeek understands
                    name = block.get("name", "")
                    args = block.get("input", {})
                    # Use clean JSON format the model was trained on
                    texts.append(f'```json\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n```')
                elif block.get("type") == "tool_result":
                    # Format tool results clearly
                    tool_use_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # Handle list of content blocks
                        result_text = "\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in result_content
                        )
                    else:
                        result_text = str(result_content)
                    if block.get("is_error"):
                        texts.append(f"Error: {result_text}")
                    else:
                        texts.append(f"Result: {result_text}")
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
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
    content: list[ContentBlock | dict]  # Allow dicts for flexibility
    model: str
    stop_reason: str
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


def messages_to_prompt(messages: list[Message], system: Any = "", tools: list[Tool] | None = None) -> str:
    """Convert messages to prompt string (matching main anthropic adapter)."""
    lines = []
    system_text = extract_text(system)
    if system_text:
        lines.append(f"System: {system_text}\n")

    # Add tool definitions in a format DeepSeek understands
    if tools:
        lines.append("\n## Available Functions\n")
        lines.append("You can call these functions when needed. To call a function, use this exact JSON format:")
        lines.append('```json')
        lines.append('{"name": "function_name", "arguments": {"param1": "value1"}}')
        lines.append('```\n')
        lines.append("Available functions:\n")
        for tool in tools:
            lines.append(f"### {tool.name}")
            if tool.description:
                lines.append(f"{tool.description[:500]}")
            if tool.input_schema and tool.input_schema.get("properties"):
                props = tool.input_schema["properties"]
                required = tool.input_schema.get("required", [])
                lines.append("Parameters:")
                for name, spec in props.items():
                    req_marker = " (required)" if name in required else ""
                    desc = spec.get("description", "")[:100]
                    lines.append(f"  - {name}: {spec.get('type', 'any')}{req_marker} - {desc}")
            lines.append("")

    for msg in messages:
        content_text = extract_text(msg.content)
        lines.append(f"{msg.role.capitalize()}: {content_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


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

            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    obj = json.loads(json_str)
                    if "name" in obj and "arguments" in obj:
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
                            # Remove the JSON from remaining text
                            remaining_text = text[:start_idx] + text[end_idx:]
                            remaining_text = remaining_text.strip()
                            logger.info(f"[TOOL PARSE] Found JSON tool call: {parsed_name} -> {canonical_name}")
                            return remaining_text, tool_uses
                        else:
                            logger.warning(f"[TOOL PARSE] Unknown tool: {parsed_name}, available: {list(tool_name_map.keys())}")
                except json.JSONDecodeError as e:
                    logger.debug(f"[TOOL PARSE] JSON decode failed: {e}")

    except Exception as e:
        logger.debug(f"[TOOL PARSE] JSON parse failed: {e}")

    # Try fallback: model outputs just arguments without "name" wrapper
    # e.g., {"command": "..."} instead of {"name": "Bash", "arguments": {"command": "..."}}
    try:
        # Find any JSON object
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            json_str = json_match.group()
            try:
                obj = json.loads(json_str)
                # Infer tool from argument keys
                inferred_name = None
                if "command" in obj:
                    inferred_name = "Bash"
                elif "file_path" in obj and "content" not in obj:
                    inferred_name = "Read"
                elif "file_path" in obj and "content" in obj:
                    inferred_name = "Write"
                elif "pattern" in obj:
                    inferred_name = "Grep"
                elif "query" in obj:
                    inferred_name = "WebSearch"

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
                        remaining_text = text[:json_match.start()] + text[json_match.end():]
                        remaining_text = remaining_text.strip()
                        logger.info(f"[TOOL PARSE] Inferred tool from args: {inferred_name} -> {canonical_name}")
                        return remaining_text, tool_uses
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.debug(f"[TOOL PARSE] Fallback parse failed: {e}")

    # Second, try DeepSeek native format
    if TOOL_CALL_BEGIN in text:
        logger.info(f"[TOOL PARSE] Found DeepSeek markers in output")

        # Split on tool call begin markers
        parts = text.split(TOOL_CALL_BEGIN)
        remaining_parts = [parts[0]]  # Keep text before first marker

        for part in parts[1:]:
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

                            # Try to parse as JSON
                            json_match = re.search(r'\{[^{}]*\}', args_text)
                            if json_match:
                                try:
                                    args = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    pass

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

    # Cap max_tokens to prevent runaway generation
    max_tokens = min(request_body.max_tokens, 500)

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
    prompt = messages_to_prompt(request_body.messages, request_body.system, request_body.tools)
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

    # Parse tool calls from output
    remaining_text, tool_uses = parse_tool_calls(output_text, request_body.tools)

    # Build content blocks
    content_blocks: list[dict] = []
    # Only add text block if it has meaningful content (not empty or just code fences)
    if remaining_text:
        clean_text = remaining_text.strip()
        # Filter out empty JSON code blocks or other meaningless content
        if clean_text and clean_text not in ["```json\n\n```", "```\n```", "```json```", "```"]:
            content_blocks.append({"type": "text", "text": remaining_text})
    for tool_use in tool_uses:
        content_blocks.append(tool_use)

    # Determine stop reason
    if tool_uses:
        stop_reason = "tool_use"
        logger.info(f"[TOOL USE] Detected {len(tool_uses)} tool call(s)")
    elif generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
        stop_reason = "end_turn"
    else:
        stop_reason = "max_tokens"

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

    # Return JSON response with None fields excluded for Claude Code CLI compatibility
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
