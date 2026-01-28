# Expert Debate: Semantic Caching Server Issues

**Date**: 2026-01-27
**Participants**: MLX Memory Expert, System Architecture Expert, Inference Pipeline Expert, ML Performance Expert
**Moderator**: Technical Lead

---

## Context

The semantic caching server is experiencing issues during Claude Code CLI testing:
- OOM crashes after 3rd request
- Cache corruption (19K tokens → 324 tokens)
- 503 errors (block pool exhaustion)
- Wrong prompt processing

---

## Turn 1: Initial Problem Statement

**Technical Lead**: We've been iterating on this semantic caching server and keep hitting issues. The core goal is simple: cache KV states for multi-turn conversations so Claude Code CLI doesn't re-compute 19K tokens of context every request. What's going wrong?

**MLX Memory Expert**: Let me start with the memory analysis. The original OOM came from `BatchQuantizedKVCache.update_and_fetch()` at line 222:

```python
# ORIGINAL CODE:
headroom = max(num_steps * 10, 16384)  # 16K tokens minimum headroom!
```

For single token generation, this allocates:
- 16,384 tokens × 55,296 bytes/token = 905 MB per layer
- × 27 layers = 24.4 GB total
- On a 24GB Mac with 8GB model = instant OOM

The fix reduced this to 1K max headroom, which is ~1.5 GB - manageable.

**System Architecture Expert**: But the headroom fix didn't solve everything. After that fix, a native generation path was added to "bypass BatchGenerator for cache hits". This introduced multiple bugs. Looking at the conversation history:

1. **EOS token in output** - Native path included EOS in returned text
2. **Wrong prompt processing** - Divergent prompts handled incorrectly
3. **Cache corruption** - 19K tokens became 324 tokens
4. **Block pool exhaustion** - 503 errors from not freeing blocks

**Inference Pipeline Expert**: The native generation path was a premature optimization. BatchGenerator already works - it just needed the headroom fix. By adding a parallel path, we introduced code that:

1. Wasn't tested against experiments
2. Had different semantics for token tracking
3. Duplicated logic with subtle differences
4. Broke the working chunked prefill → BatchGenerator handoff

---

## Turn 2: Root Cause Analysis

**Technical Lead**: Let's identify the actual root causes vs. symptoms.

**MLX Memory Expert**:

**ROOT CAUSE 1: Headroom allocation was unbounded**

MLX's lazy evaluation means you can build computation graphs indefinitely. When `update_and_fetch()` allocated 16K slots, MLX didn't immediately allocate memory - but when `mx.eval()` was called, ALL 27 layers materialized simultaneously:

```
Layer 0: mx.zeros((1, 16, 16384, 16), dtype=uint32)  # Q4 weights
Layer 0: mx.zeros((1, 16, 16384, 2), dtype=float16)  # scales
Layer 0: mx.zeros((1, 16, 16384, 2), dtype=float16)  # biases
× 6 tensors per layer × 27 layers = 162 tensor allocations at once
```

The fix caps headroom at 1K tokens, which means max ~1.5 GB allocation.

**System Architecture Expert**:

**ROOT CAUSE 2: Complex state machine with multiple paths**

The submit() flow has too many cases:

```
submit()
├── Cache hit path
│   ├── Reconstruct Q4 cache
│   ├── Prefix matching
│   │   ├── Exact match → treat as miss?
│   │   ├── Partial match → extend
│   │   └── Divergent → clear and miss
│   └── Insert into BatchGenerator
├── Cache miss, long prompt
│   ├── Chunked prefill
│   └── Insert into BatchGenerator (with empty tokens?)
└── Cache miss, short prompt
    └── Insert into BatchGenerator
```

The native generation path added ANOTHER branch:
```
├── Native generation path (if kv_cache and small delta)
│   └── Direct model calls (buggy!)
```

This complexity led to bugs:
- After chunked prefill, `cache` is None but `kv_cache` is populated
- Prefix matching code expected both to be set
- Native path had different token tracking semantics

**Inference Pipeline Expert**:

**ROOT CAUSE 3: BPE boundary assumptions**

The system stores `token_sequence` for prefix matching. But:

```python
tokenize("Hello world") + tokenize(" how are you")
    ≠ tokenize("Hello world how are you")
```

BPE tokenizers produce different token boundaries depending on context. The code tries to handle this by only storing PROMPT tokens, not generated tokens. But this creates issues:

1. What counts as "prompt" in multi-turn?
2. When cache diverges at token N, can we trust tokens 0..N-1?

**ML Performance Expert**:

**ROOT CAUSE 4: No experiment validation**

Looking at the experiments directory, there were comprehensive tests:
- EXP-003: Cache injection validation ✓
- EXP-004: Per-sequence cache extraction ✓
- EXP-011: Memory reclamation validation ✓

But the native generation path was added without a corresponding experiment. The changes to submit() weren't validated against the existing experiments.

---

## Turn 3: What Should Work vs. What Doesn't

**Technical Lead**: Let's be specific about what components are working.

**System Architecture Expert**:

**WORKING**:
1. **Model loading** - Model loads correctly, extracts spec
2. **BlockPool** - Allocation/deallocation works (when used correctly)
3. **Chunked prefill** - Adaptive chunks, memory-efficient
4. **BatchGenerator** - MLX's batching works correctly
5. **Q4 Direct Injection** - Keeps Q4 format end-to-end
6. **Safetensors persistence** - Save/load works
7. **API layer** - FastAPI endpoints work

**NOT WORKING / BUGGY**:
1. **Native generation path** - Multiple bugs, now disabled
2. **Prefix matching edge cases** - Exact match, divergence handling
3. **Cache invalidation** - Hot cache has stale references
4. **Memory cleanup on divergence** - KV cache not freed

**MLX Memory Expert**:

**Memory flow issues**:

```
EXPECTED:
1. Load cache from disk → ~0 GB (mmap)
2. Reconstruct KV cache → +1 GB
3. Clear AgentBlocks → -0 GB (data moved, not copied)
4. Generation → peak +2 GB
5. Extract cache → +1 GB
6. Save and clear → -1 GB

ACTUAL (with bugs):
1. Load cache → ~0 GB
2. Reconstruct → +1 GB
3. Clear AgentBlocks → blocks dict cleared but tensors still referenced!
4. Divergence detected → reconstruct again → +1 GB (now at +2 GB)
5. Native generation → allocates again → +1 GB
6. OOM or corrupted state
```

**Inference Pipeline Expert**:

**BatchGenerator contract**:

BatchGenerator expects:
- `prompts`: List of token lists to process
- `caches`: List of KVCache objects (optional)
- If cache provided, cache.size() tells it how many tokens are cached

When we:
1. Do chunked prefill
2. Pass the resulting kv_cache to BatchGenerator
3. Pass EMPTY prompt tokens (since prefill handled them)

BatchGenerator calls `cache.size()` to determine what's cached. If size() returns wrong value, it reprocesses tokens incorrectly.

The QuantizedKVCache.size() patch was added because MLX's version returned 0, but we need to verify it's actually returning the right value after all our manipulations.

---

## Turn 4: What Needs to be Done

**Technical Lead**: What's the path forward?

**System Architecture Expert**:

**IMMEDIATE ACTIONS**:

1. **Keep native generation disabled** - It's too buggy
2. **Verify headroom fix** - Confirm it's capped at 1K
3. **Test chunked prefill → BatchGenerator handoff** - Pass empty tokens
4. **Add logging for cache.size()** - Verify values at each step

**SHORT TERM**:

1. **Simplify submit() flow** - Remove dead code, reduce branches
2. **Add integration tests** - Capture expected request sequences
3. **Memory instrumentation** - Log memory at key points
4. **Validate against experiments** - Run EXP-003, EXP-004 again

**MLX Memory Expert**:

**Memory-specific actions**:

1. **Track tensor references** - Ensure AgentBlocks.layer_data actually releases memory
2. **Explicit mx.clear_cache()** - Call after freeing tensors
3. **Verify Q4 format preserved** - No accidental FP16 conversion
4. **Check concatenation** - Are we creating new tensors or views?

```python
# Add this after clearing blocks:
import gc
gc.collect()
mx.clear_cache()
active = mx.get_active_memory()
logger.info(f"Memory after clear: {active / 1e9:.2f} GB")
```

**Inference Pipeline Expert**:

**Pipeline-specific actions**:

1. **Trace token flow** - Log tokens at each step
2. **Verify offset values** - Cache.offset must equal actual tokens
3. **Test prefix scenarios**:
   - Exact match (same prompt again)
   - Extension (more tokens added)
   - Divergence (different continuation)

**ML Performance Expert**:

**Create minimal test server**:

Extract ONLY what's needed for Claude Code:
1. Model loading
2. Chunked prefill (for cold start)
3. BatchGenerator (for generation)
4. Q4 cache save/load

Remove:
- Native generation path
- Complex prefix matching
- Multiple API protocols

Test with captured logs to validate behavior.

---

## Turn 5: Specific Code Issues

**Technical Lead**: Let's identify specific code that needs attention.

**System Architecture Expert**:

**batch_engine.py issues**:

```python
# Line 561 - DISABLED, KEEP DISABLED
if False:  # Native generation path
    ...

# Line 772-782 - CASE 1 for chunked prefill handoff
if kv_cache is not None and cache is None:
    tokens_to_process = []  # CORRECT - all tokens in cache

# Line 798-880 - Prefix matching
# This logic is complex and error-prone
# Consider simplifying to just: "if not exact match, treat as miss"
```

**MLX Memory Expert**:

**mlx_quantized_extensions.py issues**:

```python
# Line 222 - HEADROOM FIX (CORRECT)
headroom = min(max(num_steps + 256, 512), 1024)

# Line 143-147 - mx.eval() after merge
mx.eval(keys_quant, keys_scales, keys_zeros,
        values_quant, values_scales, values_zeros)
# VERIFY this actually materializes tensors

# Line 250-251 - mx.eval() after expansion
mx.eval(self.keys[0], self.keys[1], self.keys[2],
       self.values[0], self.values[1], self.values[2])
# VERIFY this releases old tensors
```

**Inference Pipeline Expert**:

**agent_cache_store.py issues**:

```python
# Line 257-267 - Check for cleared blocks
if layer_blocks and not any(
    block.layer_data is not None
    for layer_blocks in entry.blocks.blocks.values()
    for block in layer_blocks
):
    # Blocks cleared, remove from hot cache
    del self._hot_cache[agent_id]

# This check happens AFTER we already returned entry.blocks
# The caller might have a stale reference
```

---

## Turn 6: Recommendations Summary

**Technical Lead**: Let's summarize the recommendations.

**ALL EXPERTS AGREE**:

1. **Native generation path should stay disabled** - Too many bugs, not worth fixing
2. **Headroom fix is correct** - 1K cap prevents OOM
3. **Chunked prefill works** - Just need correct handoff to BatchGenerator
4. **BatchGenerator is reliable** - Use it for all generation
5. **Need better instrumentation** - Memory logging, token tracing
6. **Need integration tests** - Capture expected request sequences
7. **Consider simplifying** - Remove complex prefix matching for now

**PRIORITY ORDER**:

1. **P0**: Verify headroom fix active (1 hour)
2. **P0**: Verify chunked prefill → BatchGenerator handoff (2 hours)
3. **P1**: Add memory instrumentation (2 hours)
4. **P1**: Create minimal test_claude_server.py (4 hours)
5. **P2**: Simplify submit() flow (4 hours)
6. **P2**: Add integration tests (8 hours)

**SUCCESS CRITERIA**:

1. 3 consecutive Claude Code CLI requests succeed
2. 19K token cache hit without OOM
3. Cache reused on follow-up requests (not re-computed)
4. Peak memory < 15 GB during generation

---

## Turn 7: Minimal Server Design

**Technical Lead**: What would a minimal test_claude_server.py look like?

**ML Performance Expert**:

```python
"""
Minimal semantic caching server for Claude Code CLI testing.

Components:
1. Model loading (MLX)
2. Chunked prefill (memory-efficient)
3. BatchGenerator (single-sequence)
4. Q4 cache save/load (safetensors)

NO:
- Native generation path
- Complex prefix matching
- Multiple API protocols
- Block pool (just use dicts)
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache
from fastapi import FastAPI
from safetensors import save_file, safe_open
import hashlib
import json

app = FastAPI()

# Global state
model = None
tokenizer = None
cache_store = {}  # agent_id → (kv_cache_path, total_tokens)

@app.on_event("startup")
async def startup():
    global model, tokenizer
    model, tokenizer = load("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

def chunked_prefill(tokens: list[int], n_layers: int) -> list[QuantizedKVCache]:
    """Process tokens in chunks, return Q4 cache."""
    kv_caches = [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]

    pos = 0
    while pos < len(tokens):
        chunk_size = 2048 if pos < 8000 else 512
        end = min(pos + chunk_size, len(tokens))
        chunk = mx.array([tokens[pos:end]])
        y = model(chunk, cache=kv_caches)
        mx.eval(y)
        mx.clear_cache()
        pos = end

    return kv_caches

def generate(kv_caches: list, tokens_to_process: list, max_tokens: int) -> str:
    """Generate using BatchGenerator."""
    from mlx_lm.utils import make_sampler
    from mlx_lm.sample_utils import make_sampler

    # Simple greedy generation
    generated = []
    if tokens_to_process:
        y = model(mx.array([tokens_to_process]), cache=kv_caches)
        mx.eval(y)

    for _ in range(max_tokens):
        logits = y[:, -1, :]
        token = mx.argmax(logits, axis=-1).item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        y = model(mx.array([[token]]), cache=kv_caches)
        mx.eval(y)

    return tokenizer.decode(generated)

def save_cache(agent_id: str, kv_caches: list, total_tokens: int):
    """Save Q4 cache to disk."""
    tensors = {}
    for layer_id, cache in enumerate(kv_caches):
        if cache.keys is not None:
            k_w, k_s, k_b = cache.keys
            v_w, v_s, v_b = cache.values
            tensors[f"l{layer_id}_k_w"] = k_w
            tensors[f"l{layer_id}_k_s"] = k_s
            tensors[f"l{layer_id}_k_b"] = k_b
            tensors[f"l{layer_id}_v_w"] = v_w
            tensors[f"l{layer_id}_v_s"] = v_s
            tensors[f"l{layer_id}_v_b"] = v_b

    path = f"/tmp/{agent_id}.safetensors"
    metadata = {"total_tokens": str(total_tokens)}
    save_file(tensors, path, metadata=metadata)
    cache_store[agent_id] = (path, total_tokens)

def load_cache(agent_id: str) -> tuple[list, int]:
    """Load Q4 cache from disk."""
    path, total_tokens = cache_store.get(agent_id, (None, 0))
    if path is None:
        return None, 0

    kv_caches = []
    with safe_open(path, framework="mlx") as f:
        n_layers = len([k for k in f.keys() if k.endswith("_k_w")])
        for layer_id in range(n_layers):
            cache = QuantizedKVCache(group_size=64, bits=4)
            cache.keys = (
                f.get_tensor(f"l{layer_id}_k_w"),
                f.get_tensor(f"l{layer_id}_k_s"),
                f.get_tensor(f"l{layer_id}_k_b"),
            )
            cache.values = (
                f.get_tensor(f"l{layer_id}_v_w"),
                f.get_tensor(f"l{layer_id}_v_s"),
                f.get_tensor(f"l{layer_id}_v_b"),
            )
            cache.offset = total_tokens
            kv_caches.append(cache)

    return kv_caches, total_tokens

@app.post("/v1/messages")
async def create_message(request: dict):
    # 1. Extract prompt
    messages = request.get("messages", [])
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # 2. Tokenize
    tokens = tokenizer.encode(prompt)

    # 3. Agent ID from first 100 tokens
    agent_id = "msg_" + hashlib.sha256(str(tokens[:100]).encode()).hexdigest()[:16]

    # 4. Load or create cache
    kv_caches, cached_tokens = load_cache(agent_id)

    if kv_caches is None:
        # Cold start - chunked prefill
        kv_caches = chunked_prefill(tokens, n_layers=27)
        tokens_to_process = []
    elif cached_tokens < len(tokens):
        # Cache extension
        tokens_to_process = tokens[cached_tokens:]
    else:
        # Full cache hit
        tokens_to_process = []

    # 5. Generate
    output = generate(kv_caches, tokens_to_process, max_tokens=request.get("max_tokens", 100))

    # 6. Save updated cache
    new_total = len(tokens) + len(tokenizer.encode(output))
    save_cache(agent_id, kv_caches, new_total)

    # 7. Return response
    return {
        "id": f"msg_{agent_id}",
        "content": [{"type": "text", "text": output}],
        "model": request.get("model", "local"),
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": len(tokens),
            "output_tokens": len(tokenizer.encode(output)),
        }
    }
```

This is ~150 lines vs 1585 lines in batch_engine.py. It demonstrates the core flow without the complexity.

---

## Turn 8: Final Conclusions

**Technical Lead**: What are our conclusions?

**ALL EXPERTS**:

1. **The original architecture is sound** - Q4 injection, chunked prefill, BatchGenerator are all correct

2. **Complexity introduced bugs** - Native generation path, prefix matching edge cases, multiple code paths

3. **The headroom fix was the key** - 16K → 1K tokens solved the OOM

4. **Testing was insufficient** - Native path wasn't validated against experiments

5. **Simplification is needed** - Remove dead code, reduce branches, add instrumentation

**NEXT STEPS**:

1. Keep current code with native path disabled
2. Add memory instrumentation
3. Create minimal test server for validation
4. Test against captured Claude Code CLI logs
5. If minimal server works, backport fixes to full server

---

*Debate concluded: 2026-01-27*
*Action items assigned to implementation team*
