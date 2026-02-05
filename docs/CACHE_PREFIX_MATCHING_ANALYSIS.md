# Cache Prefix Matching Analysis

## Executive Summary

The warm cache investigation revealed that cache prefix matching is working but only achieving **83-87% character match** instead of the expected **100%**. This results in **~550ms gaps** between agent turns instead of the expected **10-100ms** with full cache reuse.

**Root Cause**: The cache stores the templated prompt from Turn N, but Turn N+1's templated prompt has a fundamentally different structure due to conversation history being inserted mid-prompt.

---

## Complete Call Graph

### Entry Point: Streaming Coordination Turn

```
POST /v1/coordination/sessions/{id}/round/stream
    │
    └─► coordination_adapter.py: execute_turn_stream()
            │
            └─► coordination_service.py: execute_turn_stream()
                    │
                    ├─► 1. build_agent_prompt(directive, agent_role)
                    │       Returns: list[dict] messages
                    │
                    ├─► 2. _tokenize_chat_messages(messages)
                    │       │
                    │       ├─► _format_messages_as_text() → fallback_prompt_text
                    │       ├─► chat_template.needs_message_merging() → True for Gemma
                    │       ├─► _merge_consecutive_messages() → merged_messages
                    │       ├─► tokenizer.apply_chat_template(tokenize=False) → templated_text
                    │       └─► tokenizer.apply_chat_template(tokenize=True) → tokens
                    │       Returns: (tokens, templated_text)
                    │
                    ├─► 3. cache_store.load(agent_id) → cached_blocks (or None)
                    │
                    └─► 4. scheduler.submit_and_stream(
                            agent_id, prompt_tokens, cache, prompt_text=templated_text
                        )
                            │
                            └─► scheduler._submit_direct(request)
                                    │
                                    └─► batch_engine.submit(
                                            agent_id, prompt=templated_text, cache, prompt_tokens
                                        )
```

### Inside batch_engine.submit() - Cache Matching Logic

```python
# Line 639-650: EARLY EXACT detection
stored_text = cache.prompt_text  # From Turn N
if stored_text == prompt:  # EXACT match only
    # Optimization: Limit reconstruction to prompt tokens only
    max_reconstruct = len(prompt_tokens)

# Line 680: Reconstruct KV cache from blocks
kv_cache = _reconstruct_cache(cache)

# Line 938-1070: Character-level prefix matching
stored_text = cache.prompt_text
char_match = cache.common_prefix_chars(prompt)  # Compare character-by-character

if char_match == len(stored_text) == len(prompt):
    # EXACT MATCH: Reuse entire prompt cache

elif char_match == len(stored_text) < len(prompt):
    # EXTEND: Cache is prefix of new prompt, process only new text

elif char_match < len(stored_text):
    # DIVERGE: Prompt diverges from stored text
    if match_ratio >= 0.5 and char_match > 100:
        # Partial reuse: trim cache to match point, process rest
    else:
        # Cache miss: process entire prompt
```

### After Generation: Cache Saving

```python
# Line 1469: Extract and save cache
blocks = _extract_cache(uid, cache, full_token_sequence, prompt_text=prompt_text)

# Line 1897-1902: AgentBlocks created with prompt_text
return AgentBlocks(
    agent_id=agent_id,
    blocks=...,
    total_tokens=...,
    token_sequence=...,
    prompt_text=prompt_text,  # ← STORED for next turn's prefix matching
)
```

---

## Text Transformations at Each Step

### Step 1: build_agent_prompt() Output

For Alice Turn 1:
```json
[
  {"role": "system", "content": "Your name is Alice. You are Alice. Be concise.\n\nRULES: ..."},
  {"role": "user", "content": "Say hello briefly."},
  {"role": "user", "content": "[Alice, respond now.]"}
]
```

For Alice Turn 2:
```json
[
  {"role": "system", "content": "Your name is Alice. You are Alice. Be concise.\n\nRULES: ..."},
  {"role": "user", "content": "Say hello briefly."},
  {"role": "assistant", "content": "Hello!"},  // ← Alice's Turn 1 response
  {"role": "user", "content": "[Alice, respond now.]"}
]
```

### Step 2: After Message Merging (Gemma)

For Alice Turn 1 (merged):
```json
[
  {"role": "user", "content": "Your name is Alice...\n\nRULES: ...\nSay hello briefly.\n[Alice, respond now.]"},
]
```

For Alice Turn 2 (no consecutive user messages to merge):
```json
[
  {"role": "user", "content": "Your name is Alice...\n\nRULES: ..."},
  {"role": "user", "content": "Say hello briefly."},  // Actually NO - these are NOT consecutive
  {"role": "assistant", "content": "Hello!"},
  {"role": "user", "content": "[Alice, respond now.]"}
]
```

**WAIT** - I need to re-examine. The system message becomes first user message for Gemma. Let me trace more carefully.

### Step 3: After apply_chat_template()

For Alice Turn 1:
```
<bos><start_of_turn>user
Your name is Alice. You are Alice. Be concise.

RULES: You are Alice and nobody else...

Say hello briefly.
[Alice, respond now.]<end_of_turn>
<start_of_turn>model
```

For Alice Turn 2:
```
<bos><start_of_turn>user
Your name is Alice. You are Alice. Be concise.

RULES: You are Alice and nobody else...

Say hello briefly.<end_of_turn>
<start_of_turn>model
Hello!<end_of_turn>
<start_of_turn>user
[Alice, respond now.]<end_of_turn>
<start_of_turn>model
```

---

## The Fundamental Mismatch

### Turn 1 Stored Text (348 chars)
```
<bos><start_of_turn>user
Your name is Alice...Say hello briefly.
[Alice, respond now.]<end_of_turn>
<start_of_turn>model
```
Position 290-320: `[Alice, respond now.]<end_of_turn>`

### Turn 2 Current Prompt (431 chars)
```
<bos><start_of_turn>user
Your name is Alice...Say hello briefly.<end_of_turn>
<start_of_turn>model
Hello!<end_of_turn>
<start_of_turn>user
[Alice, respond now.]<end_of_turn>
<start_of_turn>model
```
Position 290-320: `<end_of_turn>\n<start_of_turn>model`

### Result
- **char_match = 291**: Characters match until position 291 (approximately where "[Alice..." starts in Turn 1)
- **Divergence point**: Turn 1 has `[Alice` but Turn 2 has `<end_of_turn>` at this position
- **Match ratio**: 291/348 = 83.6%

---

## Why 550ms Gaps?

The DIVERGE code path is triggered (line 1026-1070):

1. **Partial cache reuse**: 291 chars match → ~60 tokens usable
2. **Cache trimmed** to 60 tokens
3. **Remaining text tokenized**: `prompt[291:]` → ~30 new tokens
4. **But wait**: The entire KV cache needs reconstruction from disk (288 Q4 tensors)
5. **Plus**: Even partial prefill of 30 tokens takes time

### The Real Problem

Even though we're reusing 60 tokens of cache, we still:
1. **Reconstruct 48 layers × 6 tensors = 288 tensor operations** from disk
2. **Trim the cache** to the match point
3. **Process remaining tokens** through forward pass

The reconstruction cost (~500ms) dominates, even for partial reuse.

---

## Key Insight

**The cache matching is working correctly.** The 83% match is expected because the conversation structure genuinely changes between turns.

**The problem is**: Cache reconstruction from disk takes ~500ms regardless of how much we reuse. The fix should focus on:

1. **Hot cache in memory**: Keep recently-used caches in GPU memory (not on disk)
2. **Structural cache design**: Store cache at conversation turn boundaries, not at prompt boundaries
3. **Or**: Accept that multi-agent conversations can't perfectly reuse cache because the prompt structure changes each turn

---

## Files Involved

| File | Function | Role |
|------|----------|------|
| `coordination_service.py` | `execute_turn_stream()` | Entry point, builds prompt |
| `coordination_service.py` | `build_agent_prompt()` | Creates message list |
| `coordination_service.py` | `_tokenize_chat_messages()` | Merges messages, applies template |
| `chat_template_adapter.py` | `needs_message_merging()` | Detects Gemma/DeepSeek |
| `scheduler.py` | `submit_and_stream()` | Queues request |
| `batch_engine.py` | `submit()` | Cache matching logic |
| `batch_engine.py` | `_extract_cache()` | Saves cache with prompt_text |
| `entities.py` | `AgentBlocks` | Stores prompt_text field |

---

## Detailed Message Flow Analysis

### Alice Turn 1: First Turn (No History)
```
build_agent_prompt() output:
  [0] system: "You are Alice. Be concise.\n\nRULES: ..."
  [1] user: "Say hello briefly."
  [2] user: "[Alice, respond now.]"

After Gemma merging (consecutive users → one user):
  [0] user: "You are Alice...\nSay hello briefly.\n[Alice, respond now.]"

After apply_chat_template():
  <bos><start_of_turn>user
  You are Alice...Say hello briefly.
  [Alice, respond now.]<end_of_turn>
  <start_of_turn>model
```

### Alice Turn 2: With Conversation History
```
build_agent_prompt() output:
  [0] system: "You are Alice. Be concise.\n\nRULES: ..."
  [1] user: "Say hello briefly."
  [2] assistant: "Hello!"                    ← Alice's own Turn 1 response
  [3] user: "Bob: Hi there!"                 ← Bob's response (prefixed)
  [4] user: "[Alice, respond now.]"

After Gemma merging (only consecutive users [3]+[4] merge):
  [0] system → becomes user for Gemma
  [1] user: "Say hello briefly."
  [2] assistant: "Hello!"
  [3] user: "Bob: Hi there!\n[Alice, respond now.]"  ← Merged

After apply_chat_template():
  <bos><start_of_turn>user
  You are Alice...
  Say hello briefly.<end_of_turn>
  <start_of_turn>model
  Hello!<end_of_turn>
  <start_of_turn>user
  Bob: Hi there!
  [Alice, respond now.]<end_of_turn>
  <start_of_turn>model
```

### The Structural Problem

| Turn 1 Position ~290 | Turn 2 Position ~290 |
|---------------------|---------------------|
| `[Alice, respond now.]` | `<end_of_turn>\n<start_of_turn>model\n` |

The **conversation history insertion** happens at position ~290, causing the prompts to diverge. This is **expected and correct behavior** - the cache can only reuse the prefix before the divergence point.

---

## Why Reconstruction Takes 500ms

Even with 83% character match (291/348 chars), the code still:

1. **Loads cache from disk**: Reads safetensors file
2. **Reconstructs 288 Q4 tensors**: 48 layers × 6 tensors each (K,V × data,scales,biases)
3. **Trims cache to match point**: Slices to ~60 usable tokens
4. **Tokenizes remaining text**: `prompt[291:]` → ~30 tokens
5. **Processes remaining tokens**: Forward pass through 48 layers

**The reconstruction happens BEFORE knowing how much cache is usable.** We load the full cache, then trim.

---

## Implementation Plan

### Phase 1: Diagnose the 500ms Source (Immediate)

Add timing instrumentation to identify where time is spent:
```python
# In batch_engine.submit():
t0 = time.perf_counter()
kv_cache = self._reconstruct_cache(cache)
t1 = time.perf_counter()
# ... prefix matching ...
t2 = time.perf_counter()
# ... token processing ...
t3 = time.perf_counter()
logger.info(f"[TIMING] reconstruct={t1-t0:.3f}s, match={t2-t1:.3f}s, process={t3-t2:.3f}s")
```

### Phase 2: Hot Cache Pool (Medium-term)

Keep recently-used caches in GPU memory instead of always loading from disk.

**Changes needed:**
1. **batch_engine.py**: Add `_hot_caches: dict[str, list[KVCache]]`
2. **submit()**: Check hot cache before disk load
3. **_finalize_sequence()**: Keep cache in hot pool after generation
4. **Eviction policy**: LRU with memory limit

### Phase 3: Structural Cache Design (Long-term)

Store cache at **conversation turn boundaries** rather than at prompt boundaries.

**Concept:**
- Cache A: System prompt only (never changes)
- Cache B: System + history up to last message from this agent
- On new turn: Load Cache B, extend with new messages

This requires refactoring the cache storage to be incremental.

---

## Immediate Next Steps

1. **Add timing instrumentation** to confirm where 500ms is spent
2. **Test hypothesis**: If reconstruction is the bottleneck, implement hot cache pool
3. **If prefill is the bottleneck**: The partial reuse is working correctly, just ~30 tokens of new prefill

### Expected Outcome

With hot cache pool:
- Turn 1: ~500ms (cold, full prefill)
- Turn 2+: ~50-100ms (hot cache, only new tokens prefilled)

