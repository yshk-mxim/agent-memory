# EXP-003: Cache Injection Validation

**Date**: 2026-01-24
**Status**: ✅ PASSED
**Sprint**: 0 (Foundation)

## Objective

Prove that `caches=[loaded_cache]` parameter works on `BatchGenerator.insert()` to enable cache injection for continued generation.

## Hypothesis

mlx_lm's `BatchGenerator.insert()` supports a `caches` parameter that allows pre-built KV caches to be injected, enabling generation to continue from a previous state without re-processing the prompt.

## Method

1. Load Gemma 3 12B model
2. Generate text from prompt "The quick brown fox jumps over the lazy"
3. Extract KV cache via `Response.prompt_cache` attribute
4. Save cache to disk via `save_prompt_cache()`
5. Reload cache via `load_prompt_cache()`
6. Create new `BatchGenerator` and inject reloaded cache via `caches=[loaded_cache]` parameter
7. Generate again and compare output

## Results

✅ **SUCCESS** - Cache injection works correctly

### Key Findings

1. **Correct API**: `BatchGenerator.insert([prompt_tokens], caches=[loaded_cache])`
   - `prompt_tokens` must be pre-tokenized (list of ints), NOT raw strings
   - `caches` parameter is a list (one cache per prompt in the batch)

2. **Cache Extraction**: `Response.prompt_cache` is an **attribute**, not a method
   - Returns a list of 48 `RotatingKVCache` objects (one per layer for Gemma 3)
   - Available on `Response` when `finish_reason is not None`

3. **Token Accumulation**: `Response.token` is singular
   - Tokens must be accumulated: `tokens.append(r.token)` during generation
   - Final text via `tokenizer.decode(accumulated_tokens)`

4. **Output Consistency**: With deterministic generation (default settings), outputs match exactly
   - Without cache: ' dog.\\n\\nThis is a pangram,'
   - With cache:    ' dog.\\n\\nThis is a pangram,'

## Corrected API (vs Production Plan)

| Production Plan Assumption | Actual API (v0.30.4) |
|---------------------------|----------------------|
| `BatchGenerator(model, tokenizer=tok)` | `BatchGenerator(model)` — no tokenizer param |
| `gen.insert([prompt_string])` | `gen.insert([tokenizer.encode(prompt)])` — must tokenize |
| `cache = r.prompt_cache()` | `cache = r.prompt_cache` — attribute, not callable |
| `r.tokens` | `r.token` — singular, accumulate manually |

## Implications for Sprint 1+

1. **Continuous batching IS feasible** — cache injection works as needed
2. **API wrapper required** — must handle tokenization before `insert()`
3. **Token accumulation** — application layer must track tokens per sequence
4. **Update production plan** — correct API references in §2 (Corrected mlx_lm API)

## Recommendation

✅ **PROCEED** with continuous batching architecture (NOT Plan B)

## Files

- Experiment script: `experiments/exp_003_cache_injection.py`
- Output: Successfully injected cache and matched outputs
