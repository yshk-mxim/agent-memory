# Character-Level Prefix Matching for KV Cache Reuse

## Problem

Warm TTFT equals or exceeds cold TTFT. Benchmark data:

| Scenario | Cold TTFT | Warm TTFT | Expected |
|----------|-----------|-----------|----------|
| short (153 tok) | 1461ms | 1661ms | ~5ms |
| medium (1449 tok) | 3020ms | 65861/3968/4943ms | ~50ms |
| long (5780 tok) | 9704ms | CRASH | ~100ms |

**Root cause**: Token-level prefix matching fails at BPE boundaries.

When a multi-turn conversation grows (system + user1 + assistant1 + user2), re-tokenizing the full text produces different token IDs at message boundaries than `tokenize(original) + tokenize(extension)`. The current `common_prefix_length()` compares token-by-token, hits a BPE mismatch early in the sequence, falls into DIVERGE path → entire cache discarded → full re-prefill.

```
Cached tokens:  [1234, 5678, 9012, ...]  ← tokenize("Hello world")
Re-tokenized:   [1234, 5679, 9012, ...]  ← tokenize("Hello world How are you")[0:3]
                        ^^^^
                BPE boundary shifted — DIVERGE at token 2
```

**Fix**: Compare raw prompt text (characters), not tokens. Only tokenize the NEW portion after the text divergence point.

---

## Design

### Core Idea

1. Store `prompt_text: str` alongside existing `token_sequence` in AgentBlocks
2. On cache lookup, compare stored prompt_text against new prompt at the character level
3. Find the longest common character prefix
4. Map that character position to a token boundary (conservative: round DOWN to last complete token within the matched character span)
5. Trim cache to that token count
6. Tokenize only the remaining text (after the character match point)
7. Process only those new tokens through the model

### Agent IDs Are Cache Lookup Keys (NOT In Text Stream)

Agent IDs (`X-Session-ID` → `agent_id`) serve as direct cache lookup keys:
- `cache_store.load(agent_id)` → finds the cached AgentBlocks
- Character comparison validates whether the cached KV is reusable for the new prompt
- Agent ID is NEVER embedded in the prompt text — doing so would cause immediate divergence since the ID changes per session

Flow: `agent_id → find cache → character-compare prompt_text → decide reuse/discard`

### Character-to-Token Boundary Mapping

The tokenizer's `encode()` with offset mapping gives us character→token correspondence:

```python
# Encode the MATCHED character prefix to find the token boundary
matched_text = new_prompt[:char_match_len]
matched_tokens = tokenizer.encode(matched_text)
# These tokens correspond to the matched text — safe to reuse this many cache positions
usable_cache_tokens = len(matched_tokens)
```

Conservative approach: re-tokenize the matched prefix to get the exact token count. This avoids any token boundary ambiguity — the re-tokenized prefix gives us the number of KV positions that are valid.

### Three Prefix Match Outcomes

| Outcome | Condition | Action |
|---------|-----------|--------|
| EXACT | `char_match == len(stored) == len(new)` | Treat as cache miss (same prompt → biased output) |
| EXTEND | `char_match == len(stored) < len(new)` | Reuse full cache, tokenize only `new_prompt[char_match:]` |
| DIVERGE | `char_match < len(stored)` | If `char_match` covers ≥80% of stored text, trim cache to that point and process remainder. Otherwise, discard cache entirely. |

The 80% threshold prevents wasting time reconstructing a cache that provides minimal benefit. Configurable.

---

## Batch=2 Cache Flow

### Extraction (per-stream)
When `response.finish_reason is not None` in `step()`:
1. `response.prompt_cache` → list of per-layer QuantizedKVCache (Q4 format)
2. `_extract_cache()` → slices into 256-token KVBlock objects per layer
3. Creates `AgentBlocks` with `prompt_text` (the prompt string used for this generation) and `token_sequence` (prompt tokens only)
4. Stored in `_agent_blocks[agent_id]` and saved to `AgentCacheStore`

**Q4 preserved end-to-end**: `_extract_cache()` quantizes any FP16 output to Q4. No Q4→FP16 conversion anywhere.

### Storage
`AgentCacheStore.save()` → `_save_to_disk()`:
- Hot tier: in-memory reference in `_hot_cache[agent_id]`
- Warm tier: safetensors file with metadata including `prompt_text` (string in header)
- `prompt_text` stored as string in safetensors metadata dict (already supports arbitrary string values)

### Lookup
`AgentCacheStore.load(agent_id)` → returns `AgentBlocks` with `prompt_text` field populated from either hot or warm tier.

### Reassembly
`_reconstruct_cache()` in batch_engine:
- Reads Q4 block data from AgentBlocks
- Concatenates per-layer into full sequence tensors
- Creates `QuantizedKVCache` objects with correct offset
- ALL stays Q4 — no dequantization

### Injection (batch=2)
`BatchQuantizedKVCache.merge()`:
- Takes list of individual QuantizedKVCache objects
- Left-pads shorter cache to match longest (attention mask handles padding)
- Returns batched Q4 cache ready for BatchGenerator
- When only 1 request arrives with batch_size=2: BatchGenerator processes batch of size 1 — no special handling needed, merge() receives a single-element list

### Attention Masking
`BatchQuantizedKVCache.make_mask()` creates per-sequence attention masks that:
- Block attention to left-padding positions
- Each sequence only attends to its own valid KV positions
- Ensures padding doesn't corrupt output

---

## Adaptive Chunked Prefill Integration

Chunked prefill applies to ANY long token sequence — cold start OR warm cache extension:

### Cold Start (cache is None)
- Long prompts split into adaptive chunks (512-2048 tokens based on position)
- Each chunk runs a forward pass, building KV cache incrementally from scratch
- Peak memory bounded: only one chunk's activations in memory at a time

### Warm Cache Extension (cache is not None, EXTEND/DIVERGE)
- Character matching determines reusable cache length and new tokens to process
- If new tokens > chunked prefill threshold: use chunked prefill for the extension ON TOP OF the existing cache
- `_chunked_prefill()` modified to accept optional `initial_cache` parameter:

```python
def _chunked_prefill(self, tokens, agent_id, initial_cache=None):
    if initial_cache is not None:
        kv_caches = initial_cache  # Start from existing warm cache
    else:
        kv_caches = [QuantizedKVCache(...) for _ in range(n_layers)]  # Fresh
    # ... rest of chunked processing is identical
```

- In CASE 2 EXTEND, after determining new tokens:

```python
new_tokens = self._tokenizer.encode(new_text)
if enabled and len(new_tokens) >= threshold:
    # Chunked prefill for extension (memory-safe)
    kv_cache = self._chunked_prefill(new_tokens, agent_id, initial_cache=kv_cache)
    tokens_to_process = [new_tokens[-1]]  # Last token for BatchGenerator
else:
    tokens_to_process = new_tokens  # Short extension, direct to BatchGenerator
```

This ensures memory protection whether the long sequence is a cold start or a warm cache extension. The chunked prefill mechanism is the same — the only difference is the starting cache state (empty vs pre-populated).

**No excess overhead**: Character comparison is O(n) in prompt length (microseconds). Re-tokenizing the matched prefix is one tokenizer call. Chunked prefill adds no overhead for short extensions (<threshold). For long extensions, it provides the same 38-65% peak memory reduction as cold starts.

---

## Three-Tier Cache Architecture (Preserved)

| Tier | Location | Format | Lifecycle |
|------|----------|--------|-----------|
| Hot | `_hot_cache` dict in AgentCacheStore | AgentBlocks with Q4 block references | Active agents, LRU eviction |
| Warm | `~/.semantic/caches/*.safetensors` | Q4 tensors + metadata (incl. prompt_text) | Recently evicted, disk load on access |
| Cold | Evicted from warm tier | Must regenerate | Old agents, model mismatch |

`prompt_text` persists across all tiers:
- Hot: in-memory field on AgentBlocks
- Warm: string in safetensors metadata header
- Cold: lost (regenerated on next cold start)

---

## Novelty Considerations

From `backend_plan.md` and `semantic-cache-architecture-v2.md`:

1. **Q4 Direct Injection**: Cache stays quantized end-to-end. Character-level matching doesn't change this — it only changes HOW we decide what to reuse, not the format.

2. **Block Pool Management**: 256-token blocks allocated from shared pool. Character matching determines how many blocks to reuse (trim to token boundary that falls within matched character span).

3. **Hybrid Generation Path**: Native path for single-agent cache hits, batch path for multi-agent. Character matching applies to both — it's the prefix validation step before either path.

4. **Model-Agnostic Design**: `ModelCacheSpec` abstraction means character matching works regardless of model architecture (DeepSeek MLA, Gemma hybrid attention, uniform attention).

---

## Files Changed

### 1. `src/semantic/domain/entities.py` — Add `prompt_text` field

```python
@dataclass
class AgentBlocks:
    agent_id: str
    blocks: dict[int, list[KVBlock]]
    total_tokens: int
    token_sequence: list[int] = field(default_factory=list)
    prompt_text: str = ""  # NEW: Raw prompt text for character-level matching
    metadata: dict[str, Any] = field(default_factory=dict)

    def common_prefix_chars(self, query_text: str) -> int:
        """Find length of common character prefix between stored and query text."""
        if not self.prompt_text or not query_text:
            return 0
        prefix_len = 0
        for c1, c2 in zip(self.prompt_text, query_text):
            if c1 != c2:
                break
            prefix_len += 1
        return prefix_len
```

Keep existing `common_prefix_length()` for backward compatibility (legacy path).

### 2. `src/semantic/application/batch_engine.py` — Character-level CASE 2 + Chunked Prefill Extension

**`_chunked_prefill()` (line 222)**: Add `initial_cache` parameter:
```python
def _chunked_prefill(self, tokens, agent_id, initial_cache=None):
    # ...
    if initial_cache is not None:
        kv_caches = initial_cache
    else:
        kv_caches = [QuantizedKVCache(...) for _ in range(self._spec.n_layers)]
    # ... rest unchanged
```

**CASE 2 replacement** (lines 807-930): Replace token comparison with character comparison + chunked prefill for long extensions:

```python
# CASE 2: kv_cache from loaded cache
elif kv_cache is not None and cache is not None:
    stored_text = getattr(cache, 'prompt_text', '')

    if stored_text:
        char_match = cache.common_prefix_chars(prompt)

        if char_match == len(stored_text) == len(prompt):
            # EXACT MATCH — treat as cache miss
            ...
        elif char_match == len(stored_text) < len(prompt):
            # EXTEND — full cache reusable, tokenize only new text
            new_text = prompt[char_match:]
            new_tokens = self._tokenizer.encode(new_text)
            stored_prompt_token_count = len(cache.token_sequence)
            if cache.total_tokens > stored_prompt_token_count:
                self._slice_cache_to_length(kv_cache, stored_prompt_token_count)

            # Use chunked prefill for long extensions (memory protection)
            enabled, threshold, _, _ = self._get_chunked_prefill_settings()
            if enabled and len(new_tokens) >= threshold:
                kv_cache = self._chunked_prefill(
                    new_tokens, agent_id, initial_cache=kv_cache
                )
                tokens_to_process = [new_tokens[-1]]
            else:
                tokens_to_process = new_tokens

        elif char_match < len(stored_text):
            # DIVERGE — check if enough matched to be worth trimming
            match_ratio = char_match / len(stored_text) if stored_text else 0
            if match_ratio >= 0.8 and char_match > 100:
                matched_tokens = self._tokenizer.encode(prompt[:char_match])
                usable_tokens = len(matched_tokens)
                self._slice_cache_to_length(kv_cache, usable_tokens)
                remaining_text = prompt[char_match:]
                remaining_tokens = self._tokenizer.encode(remaining_text)

                # Chunked prefill for long remaining portion
                enabled, threshold, _, _ = self._get_chunked_prefill_settings()
                if enabled and len(remaining_tokens) >= threshold:
                    kv_cache = self._chunked_prefill(
                        remaining_tokens, agent_id, initial_cache=kv_cache
                    )
                    tokens_to_process = [remaining_tokens[-1]]
                else:
                    tokens_to_process = remaining_tokens
            else:
                # Not enough overlap — discard cache
                ... (free kv_cache, set cache = None)
    else:
        # No prompt_text stored — fall back to token comparison (legacy)
        ...existing token-level logic...
```

**In `_active_requests`** (line 960): Store prompt text alongside prompt tokens:
```python
# Store: (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
self._active_requests[actual_uid] = (agent_id, [], detokenizer, list(prompt_tokens), prompt)
```

**In `step()`** (line 1171): Pass prompt_text to _extract_cache:
```python
agent_id, tokens, detokenizer, prompt_tokens, prompt_text = self._active_requests[uid]
# ...
full_token_sequence = list(prompt_tokens)
blocks = self._extract_cache(uid, cache, full_token_sequence, prompt_text=prompt_text)
```

**In `_extract_cache()`**: Pass prompt_text through to AgentBlocks:
```python
def _extract_cache(self, uid, cache, token_sequence, prompt_text=""):
    # ... existing logic ...
    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks_dict,
        total_tokens=seq_len,
        token_sequence=token_sequence or [],
        prompt_text=prompt_text,  # NEW
    )
```

### 3. `src/semantic/application/agent_cache_store.py` — Persist prompt_text

**`_save_to_disk()`** (line 481): Add prompt_text to metadata:
```python
metadata = {
    ...existing fields...,
    "prompt_text": entry.blocks.prompt_text,  # NEW
}
```

**`_load_from_disk()`** (line 522): Load prompt_text:
```python
prompt_text = str(metadata.get("prompt_text", ""))
blocks = AgentBlocks(
    agent_id=agent_id,
    blocks=blocks_dict,
    total_tokens=total_tokens,
    token_sequence=token_sequence,
    prompt_text=prompt_text,  # NEW
)
```

### 4. `src/semantic/adapters/inbound/anthropic_adapter.py` — Already passes prompt

The adapter already constructs the prompt string via `messages_to_prompt()` and passes it to `batch_engine.submit(prompt=...)`. No changes needed — `submit()` already receives the prompt text as `prompt` parameter.

### 5. `src/semantic/application/batch_engine.py` — submit_with_cache()

Add `prompt_text` parameter to `submit_with_cache()` (used by ConcurrentScheduler after chunked prefill):
```python
def submit_with_cache(self, agent_id, prompt_tokens, kv_caches,
                      max_tokens=256, prompt_text="", ...):
    # Store prompt_text in _active_requests
```

---

## Implementation Order

### Step 0: Copy plan to `plans/character_prefix_matching.md`

### Step 1: Add `prompt_text` field to AgentBlocks (`entities.py`)
- Add field with default empty string
- Add `common_prefix_chars()` method
- No breaking changes (default value)

### Step 2: Thread `prompt_text` through batch_engine
- Add to `_active_requests` tuple
- Pass through `_extract_cache()` → `AgentBlocks`
- Add to `submit_with_cache()`
- Update tuple unpacking in `step()`

### Step 3: Persist `prompt_text` in agent_cache_store
- Add to `_save_to_disk()` metadata
- Load in `_load_from_disk()`

### Step 4: Modify `_chunked_prefill()` to accept `initial_cache`
- Add `initial_cache: list[Any] | None = None` parameter
- Use `initial_cache` if provided, else create fresh QuantizedKVCache list
- No other logic changes needed

### Step 5: Replace CASE 2 with character-level matching + chunked extension
- Replace token comparison with character comparison
- Handle EXACT, EXTEND, DIVERGE with character semantics
- In EXTEND/DIVERGE partial-reuse: use chunked prefill when new tokens >= threshold
- Keep legacy token fallback for caches without prompt_text

### Step 6: Run benchmarks
- Start server, run capability_benchmark.py with warm scenarios
- Verify warm TTFT << cold TTFT
- Verify no regressions on cold scenarios

---

## Verification

1. **Unit test**: Create AgentBlocks with prompt_text, verify `common_prefix_chars()` returns correct lengths for exact, extend, and diverge cases.

2. **Integration test**:
   - Send request A (creates cache with prompt_text stored)
   - Send request B extending A's conversation (same agent_id)
   - Verify CASE 2 hits EXTEND path (check logs for `[CACHE EXTEND]`)
   - Verify warm TTFT << cold TTFT

3. **Batch=2 test**: Submit 2 concurrent requests with different agent_ids and cache states. Verify both complete correctly, caches saved with prompt_text.

4. **Benchmark**: Run `python benchmarks/capability_benchmark.py --config single` and verify warm scenarios show significant TTFT improvement.

5. **Edge cases**:
   - First request for new agent (no cache) → cold start, chunked prefill
   - Exact same prompt resent → EXACT MATCH → cache miss (fresh generation)
   - Prompt with shared prefix but divergent suffix → DIVERGE with threshold check
   - Cache loaded from disk (warm tier) → prompt_text loaded from metadata → character matching works

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Re-tokenizing matched prefix gives different token count than cached | Cache trim to wrong position | Conservative: trim to re-tokenized count, which is guaranteed correct for the matched text |
| prompt_text exceeds safetensors metadata limit | Cache save fails | safetensors metadata is string-valued dict with no practical size limit; prompts are <100KB |
| Legacy caches without prompt_text | Falls back to token matching | Graceful degradation via `if stored_text:` check |
| Memory overhead of storing prompt_text in AgentBlocks | Negligible | Prompt strings are ~1-50KB vs Q4 cache tensors at ~1-7GB |
