# Character-Level Prefix Matching for KV Cache Reuse

**Date**: 2026-01-30
**Status**: Implementation complete, pending benchmark validation

## Executive Summary

Replaced token-level prefix matching with **character-level text comparison** to fix a fundamental BPE tokenization boundary problem that made warm cache hits as slow as cold starts. By comparing raw prompt text (characters) instead of token IDs, only the truly new portion of a conversation is tokenized and processed — eliminating false cache misses caused by tokenizer non-compositionality.

**Key Achievement**: Eliminates the BPE boundary mismatch that caused warm TTFT to equal or exceed cold TTFT.

**Core Insight**: `tokenize(A + B) ≠ tokenize(A) + tokenize(B)` — BPE tokenization is non-compositional across concatenation boundaries. Token-level prefix comparison fails at these boundaries, discarding valid caches. Character-level comparison is immune to this because text concatenation IS compositional.

**Problem Severity** (pre-fix benchmark data):

| Scenario | Cold TTFT | Warm TTFT | Expected Warm |
|----------|-----------|-----------|---------------|
| short (153 tok) | 1461ms | 1661ms | ~5ms |
| medium (1449 tok) | 3020ms | 65861ms | ~50ms |
| long (5780 tok) | 9704ms | CRASH | ~100ms |

Warm was slower than cold because the cache was loaded, compared token-by-token, diverged at a BPE boundary, and then discarded — wasting time on both cache loading AND full re-prefill.

---

## Background

### The BPE Boundary Problem

BPE tokenization is context-dependent. The tokenizer produces different token IDs depending on what text surrounds a given substring:

```
tokenize("Hello world")           = [1234, 5678]
tokenize("Hello world How are you") = [1234, 5679, 9012, ...]
                                            ^^^^
                                    Different token ID for "world"!
```

The space between "world" and "How" shifts the BPE merge boundary, changing the token ID for "world" even though the text is identical up to that point.

### How This Breaks Token-Level Prefix Matching

In a multi-turn conversation, the client sends progressively longer prompts:

```
Turn 1: system + user1
Turn 2: system + user1 + assistant1 + user2
Turn 3: system + user1 + assistant1 + user2 + assistant2 + user3
```

Each turn's prompt is tokenized as a single string. The tokens for "system + user1" in Turn 2 differ from the tokens saved from Turn 1 because the tokenizer now sees additional text after the original content.

The old token-level matching:
1. Load cached tokens: `[1234, 5678, 9012, ...]` (from Turn 1)
2. Tokenize new prompt: `[1234, 5679, 9012, ...]` (Turn 2, full re-tokenization)
3. Compare token-by-token → **DIVERGE at position 2**
4. Discard entire cache → full re-prefill (cold start behavior)

### Why This Is Unique to Persistent KV Caches

Cloud LLM services (e.g., vLLM, TGI) avoid this problem because they either:
- Re-tokenize from scratch every request (no persistent cache)
- Use prompt caching within a single session (tokens stay consistent)

Persistent per-agent caches that survive across sessions and conversation turns hit this boundary problem because the tokenizer input changes between save and load.

---

## Solution: Character-Level Prefix Matching

### Core Design

Instead of comparing token sequences, compare the raw prompt text character-by-character:

```python
stored_text  = "Hello world"           # Saved with cache
new_prompt   = "Hello world How are you"
char_match   = 11                       # "Hello world" matches exactly
new_text     = " How are you"          # Only this needs tokenizing
```

Characters don't suffer from BPE boundaries — `"Hello world"` is always `"Hello world"` regardless of what follows.

### Three Match Outcomes

| Outcome | Condition | Action |
|---------|-----------|--------|
| **EXACT** | `char_match == len(stored) == len(new)` | Cache miss — same prompt would produce biased output |
| **EXTEND** | `char_match == len(stored) < len(new)` | Reuse full cache, tokenize only `new_prompt[char_match:]` |
| **DIVERGE** | `char_match < len(stored)` | If ≥80% matched: trim cache, process remainder. Otherwise: discard cache |

### EXTEND Path (Primary Hot Path)

This is the common case for multi-turn conversations:

```
Stored:  "system + user1 + assistant1"          (1449 chars)
New:     "system + user1 + assistant1 + user2"  (1800 chars)
Match:   1449 chars (100% of stored)
Action:  Reuse cache, tokenize only "+ user2" (351 chars → ~80 tokens)
```

Instead of re-prefilling 1449 tokens (3 seconds), we process ~80 new tokens (~50ms).

### DIVERGE Partial Reuse

When the user edits an earlier message, the prompt diverges:

```
Stored:  "system + user1 + assistant1"          (1449 chars)
New:     "system + user1_edited + assistant1"   (1460 chars)
Match:   400 chars (28% of stored → below 80% threshold)
Action:  Discard cache, full re-prefill
```

For high overlap (≥80%), the cache is trimmed to the matching token boundary and the remainder is processed. This handles minor edits efficiently.

### Integration with Chunked Prefill

The character-level matching integrates with adaptive chunked prefill for memory-safe processing of long extensions:

```python
if len(new_tokens) >= chunked_prefill_threshold:
    # Long extension: use chunked prefill ON TOP of existing cache
    kv_cache = self._chunked_prefill(
        new_tokens, agent_id, initial_cache=kv_cache
    )
    tokens_to_process = [new_tokens[-1]]
else:
    # Short extension: pass directly to BatchGenerator
    tokens_to_process = new_tokens
```

This ensures the same peak memory protection (38-65% reduction) applies whether processing a cold start or extending a warm cache.

---

## What Makes This Novel?

### 1. Character-Level Comparison for Persistent KV Caches

**Standard approaches** (vLLM, TGI, SGLang):
- Use token-level prefix matching (RadixAttention, prefix trees)
- Works within a single session where tokenization is consistent
- Fails across sessions when prompt text changes around existing content

**Our approach**:
- Compare raw text at the character level before any tokenization
- Only tokenize the truly NEW portion after the match point
- Immune to BPE boundary shifts across sessions

This is specifically relevant to **persistent** cache systems where the cache outlives the tokenization context that created it.

### 2. Hybrid Character → Token Pipeline

The character match determines WHAT to reuse; tokenization only applies to the new content:

```
Character domain: find match point (O(n) string comparison, ~microseconds)
Token domain:     tokenize only new text (one encode() call)
Cache domain:     trim to token boundary, extend with new tokens
```

This avoids the fundamental problem of mapping between character and token domains for the cached portion — we never re-tokenize text that's already in the cache.

### 3. Three-Outcome Decision with Partial Reuse

Most prefix caching systems are binary: match or miss. The DIVERGE partial reuse path with a configurable threshold (80%) handles real-world conversation edits where most of the cache is still valid:

```
Stored: 10000 chars of conversation
Edited: User changed one word at position 8500
Match:  8500 chars (85% → above threshold)
Action: Trim cache to 8500-char token boundary, process remainder
Saved:  ~85% of prefill compute
```

### 4. End-to-End Q4 Preservation

Character-level matching doesn't affect the cache format. The entire pipeline stays Q4:
- **Storage**: Q4 blocks in safetensors
- **Reconstruction**: Q4 tuple → QuantizedKVCache (no dequantization)
- **Slicing**: `_slice_cache_to_length` operates on Q4 tuple components
- **Extension**: `_chunked_prefill(initial_cache=...)` extends Q4 cache in-place
- **Extraction**: Float output quantized back to Q4

No Q4→FP16 conversion anywhere in the cache matching pipeline.

---

## Implementation

### Files Changed

| File | Change |
|------|--------|
| `domain/entities.py` | Added `prompt_text: str` field and `common_prefix_chars()` to `AgentBlocks` |
| `application/batch_engine.py` | Replaced CASE 2 token matching with character-level EXACT/EXTEND/DIVERGE; added `initial_cache` to `_chunked_prefill()`; fixed pool block leak, memory leaks, dead code |
| `application/agent_cache_store.py` | Persist `prompt_text` in safetensors metadata; fixed `token_sequence` serialization roundtrip |
| `application/scheduler.py` | Thread `prompt_text` through `SchedulerRequest`, `submit_and_wait`, `_submit_direct`, `_promote_to_decode` |
| `adapters/inbound/anthropic_adapter.py` | Pass `prompt_text=prompt` to scheduler |

### prompt_text Flow

```
HTTP request
  → anthropic_adapter: messages_to_prompt() → prompt string
    → scheduler.submit_and_wait(prompt_text=prompt)
      → engine.submit(prompt=prompt_text)
        → _active_requests[uid] = (..., prompt_text)
          → step() → _extract_cache(prompt_text=prompt_text)
            → AgentBlocks(prompt_text=prompt_text)
              → cache_store.save() → safetensors metadata
                → cache_store.load() → AgentBlocks with prompt_text
                  → next submit() → character comparison
```

### Cache Store Persistence

`prompt_text` is stored as a string in safetensors metadata alongside existing fields:

```python
metadata = {
    "agent_id": agent_id,
    "total_tokens": entry.blocks.total_tokens,
    "token_sequence": entry.blocks.token_sequence,  # json serialized
    "prompt_text": entry.blocks.prompt_text,         # raw string
}
```

Survives the three-tier cache lifecycle:
- **Hot**: in-memory field on AgentBlocks
- **Warm**: string in safetensors metadata header
- **Cold**: lost (regenerated on next cold start)

---

## Bugs Found and Fixed During Implementation

### CRITICAL: Pool Block Leak

`cache.blocks.clear()` after reconstruction emptied the blocks dict. When `cache` was the same Python reference as `_agent_blocks[agent_id]` (hot cache hit), `step()` later found an empty dict and skipped `pool.free()` — permanently leaking pool blocks.

**Fix**: Remove `.clear()`. Setting `block.layer_data = None` is sufficient to free GPU memory; the dict structure must remain for pool accounting.

### CRITICAL: token_sequence Serialization Roundtrip

`str([1, 2, 3])` → `"[1, 2, 3]"` in safetensors metadata (all values are strings). On load, the string was used directly instead of being deserialized back to a list.

**Fix**: `json.loads()` with type validation in `_load_from_disk()`.

### CRITICAL: Scheduler Didn't Thread prompt_text

`SchedulerRequest` lacked `prompt_text` field. `_submit_direct()` passed `prompt=None`. Character matching was completely broken for the scheduler path.

**Fix**: Added `prompt_text` to SchedulerRequest, submit_and_wait, _submit_direct, _promote_to_decode.

### HIGH: Memory Leaks

Two GPU memory leak paths:
1. Chunked prefill succeeded but pool allocation failed → kv_cache not freed
2. Batch insert failed with warm cache → kv_cache not freed

**Fix**: Added `del kv_cache; gc.collect()` in both exception handlers.

---

## Architecture Compliance

### Hexagonal Layer Boundaries

- **Domain** (`entities.py`): `prompt_text` field and `common_prefix_chars()` — pure Python, no imports
- **Application** (`batch_engine.py`, `scheduler.py`): Character matching logic, cache flow orchestration
- **Adapter** (`anthropic_adapter.py`): HTTP→prompt_text conversion, passes to application layer

Cache format details (Q4 tensors, safetensors) stay in adapters. Application layer sees opaque `AgentBlocks` with text fields.

### Code Quality

- Module-level imports only (removed 7 runtime `import gc` instances)
- Dead code removed (~95 lines of disabled native path)
- Division by zero guard in chunked prefill logging
- Type annotation corrected on `_active_requests` (5-tuple, not 2-tuple)

---

## Expected Impact

| Scenario | Before (Token Matching) | After (Character Matching) |
|----------|------------------------|---------------------------|
| Multi-turn extend | Full re-prefill (DIVERGE at BPE boundary) | Process only new tokens (EXTEND) |
| Short warm hit (153 tok) | ~1661ms (worse than cold) | ~5ms expected |
| Medium warm hit (1449 tok) | ~4000-65000ms | ~50ms expected |
| Long warm hit (5780 tok) | CRASH | ~100ms expected |

**Theoretical speedup**: For a medium conversation (1449 tokens) where the user adds ~100 new tokens, the warm path processes ~100 tokens instead of ~1449. Expected TTFT reduction: **~97%** (from 3-4 seconds to ~50ms).

---

## Relationship to Other Novelties

| Novelty | Relationship |
|---------|-------------|
| **Q4 Direct Injection** | Character matching preserves Q4 end-to-end — no format conversion during match/extend/trim |
| **Adaptive Chunked Prefill** | Extended to support `initial_cache` parameter for warm cache extensions, providing same memory protection |
| **Continuous Batching** | Character matching applies per-request within the batch; batch=2 flows work correctly with independent prompt_text per request |
| **Persistent Per-Agent Caches** | Character matching is specifically designed for the persistence use case where BPE boundaries shift between save and load |

---

**Last Updated**: January 30, 2026
