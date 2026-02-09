# Semantic Project Memory

## Key Architecture
- 3-tier cache: hot (in-memory), warm (metadata), cold (disk safetensors)
- BlockPoolBatchEngine with step()/step_once()/drain() patterns
- ConcurrentScheduler with interleaved prefill/decode
- Chat template handling in `adapter_helpers.py` and `coordination_service.py`

## Critical Learnings

### DeepSeek Chat Template EOS Bug (2026-02-06)
- DeepSeek template closes EVERY assistant message with `<EOS>`, including the last one
- `add_generation_prompt=True` then appends fresh `Assistant:` after the EOS
- So `{"role": "assistant", "content": "Name:"}` becomes: `Assistant: Name:<EOS>Assistant:` — the name cue is dead
- **Fix**: Option 3 — inject the name directly into the token stream after template application, bypassing the EOS
- See `generation_prefix` parameter in `generate_chat_completion()` and `_tokenize_chat_messages()`

### DeepSeek Temperature (CORRECTED 2026-02-08)
- T=0 causes deterministic echo loops in multi-turn dialogue
- **Spacing corruption was the transformers v5 tokenizer bug, NOT temperature** (fixed by pinning <5.0.0)
- T=0.3 works fine post-tokenizer-fix — coordination_service.py hardcodes T=0.3 (line 617)
- Stale comment on line 614 FIXED — now correctly explains spacing was tokenizer bug, not temperature
- Coordination service ignores `SEMANTIC_MLX_DEFAULT_TEMPERATURE` env var

### DeepSeek V2 MLA Spec Mismatch (2026-02-06, FIXED)
- MLA caches K at dim=192 (128 nope + 64 rope), V at dim=128
- Spec extractor fell back to hidden_size//num_heads=128 for both
- Added `v_head_dim` field to ModelCacheSpec (None = symmetric default)
- Extractor now detects MLA via `qk_nope_head_dim` + `qk_rope_head_dim` on attn module
- Was causing 20% memory undercount in block pool allocation

### MLX Attention Dispatch & Fused Q4 Patch
- `base.py:scaled_dot_product_attention` checks `hasattr(cache, "bits")`
- Q4 path: `mx.quantized_matmul` (3 dispatches: Q@K^T, softmax, scores@V)
- FP16 path: `mx.fast.scaled_dot_product_attention` (single fused flash kernel)
- **mlx_sink_compat.py** patches `scaled_dot_product_attention` — captures qsdpa via `from` import
- **mlx_fused_attention.py** patches `quantized_scaled_dot_product_attention`:
  - GQA (n_repeats>1, Gemma): uncompiled path with mask broadcast fix
  - Non-GQA (n_repeats=1, DeepSeek): `mx.compile(shapeless=True)` wrapper
  - Metal decode kernel disabled by default (set `SEMANTIC_ENABLE_METAL_DECODE=1`)
- Gemma 3 Q4 scales/biases are **bfloat16** not float16
- Server entry point: `python -m semantic.entrypoints.cli serve --port 8005 --model ...`

### reasoning_extra_tokens Bug (2026-02-07, FIXED)
- `settings.mlx.reasoning_extra_tokens` was 300 by default, UNCONDITIONALLY added to max_tokens
- max_tokens=16 became 316, max_tokens=32 became 332 — wasted GPU memory + prolonged generation
- **Fix**: default changed to 0 in settings.py and coordination_service.py
- Only set explicitly for models that support structured reasoning

### Native MLX Safetensors I/O (2026-02-06)
- Replaced numpy-based `safetensors.numpy.save_file`/`load_file` with native `mx.save_safetensors`/`mx.load`
- **`mx.save_safetensors(path, tensors, metadata)`** auto-appends `.safetensors` to `path` — don't include the extension
- **`mx.load(path)`** returns `dict[str, mx.array]` — tensors already in correct dtype
- **`mx.load(path, return_metadata=True)`** returns `tuple[dict, dict]` (tensors, metadata) — NOT a dict with `__metadata__` key
- Q4 packed format: `uint32` holds 8 × 4-bit values. dim=256 → 32 uint32s. `mx.dequantize()` unpacks at inference.
- Bfloat16 scales/biases preserved natively (no `ml_dtypes` workaround needed)
- Metadata-only reads still use raw safetensors header bytes (8-byte LE uint64 + JSON)

### Cache Integrity Fixes (2026-02-06)
- ModelTag now includes `kv_bits` and `kv_group_size` for quantization compat checking
- `is_tag_compatible()` method for direct tag-to-tag comparison (avoids ModelCacheSpec default value mismatch)
- Block pool leak fix: `_extract_cache()` failure in `_finalize_sequence()` now cleans up via `free_agent_blocks()`
- `force_clear_all_allocations()` now nulls `block.layer_data` to release tensor memory
- Hot cache eviction after warm→hot promotion prevents unbounded growth
- Orphan `.tmp.safetensors` files cleaned up on adapter init

### Transformers v5.0.0rc1 Tokenizer Regression (2026-02-06, FIXED)
- `transformers==5.0.0rc1` breaks SentencePiece decode for DeepSeek (and likely all SP models)
- `▁` (space marker) stripped from token pieces → round-trip decode loses ALL spaces
- GitHub issue: huggingface/transformers#43066, fix in PR #42894
- **Fix**: pin `transformers>=4.47.0,<5.0.0` in pyproject.toml and requirements.txt
- `mlx-lm==0.30.4` declares `transformers==5.0.0rc1` dependency — pip warning but works with 4.57.6
- Root cause: v5 changed to "manually generate tokenizers" instead of trusting saved config files

### Full Threading & Batch Analysis (2026-02-08)
- **See `docs/analysis.md`** for comprehensive 6-section document
- MLX is NOT thread-safe (Issues #2067, #2133, #3078)
- `mx.synchronize()` syncs ONLY current default stream, NOT all streams
- `StreamContext` (`mx.stream()`) is NOT thread-safe — writes to unprotected `unordered_map`
- In-place `a[i] = b` creates lazy `SliceUpdate` graph, NOT true mutation
- MLX slicing creates copies, NOT views (unlike NumPy)
- No upstream `BatchQuantizedKVCache` — our code is entirely custom
- **Architecture is sound when scheduler enabled**: single thread for all MLX inference + `mlx_io_lock` for cross-thread I/O
- **Fixed**: misleading `mx.synchronize()` comment (said "all streams", actually "generation_stream")
- **Fixed**: `_chunked_prefill()` now acquires `mlx_io_lock` (was missing, unsafe for non-scheduler path)

### GQA Mask Shape for Batch>1 + mx.compile Crash (2026-02-07, TESTED & FIXED)
- Fused Q4 attention GQA path reshapes Q to 5D `(B, n_kv_heads, n_repeats, L, D)`
- Mask from model is 4D `(B, 1, L, S)` — can't broadcast against 5D for B>1
- **Fix**: `mx.expand_dims(mask, axis=2)` for GQA mask broadcast
- Fused SDPA is ESSENTIAL — without it, mlx_lm's original SDPA crashes with GQA broadcast error
- **B=1 split**: Compiled function always sees B=1, results concatenated. Both GQA and non-GQA use `mx.compile(shapeless=True)`
- **clip_residual monkeypatch**: Removed mx.compile from gemma3_text.clip_residual (52 calls/fwd pass)
- **mx.async_eval→mx.eval**: Replaced globally to prevent Metal "Completed handler after commit call"
- **filter() mx.synchronize()**: KEY FIX — mx.eval() only syncs current stream, but mlx_lm runs decode in a separate `generation_stream`. mx.synchronize() syncs ALL streams, preventing "uncommitted encoder" Metal assertion
  - Two distinct Metal assertions: "Completed handler after commit" (fixed by async→sync) and "commit command buffer with uncommitted encoder" (fixed by synchronize)
- **merge()/extend() mx.eval**: Materialize lazy tensors from slice assignments and concatenations
- **Commits**: b2d5617, 04c814d, 320d25e, 50a4388
- **Validated**: 25/25 rounds across 5 invocations (cold + warm cache, varied prompts/max_tokens)

### Scheduler Decode Error Infinite Loop (2026-02-07, FIXED)
- `_run_decode_step()` caught exceptions, logged, returned — but futures never resolved
- `has_active_batch()` stayed True → infinite loop of failing decode steps
- **Fix**: reject all in-flight requests on exception in decode step

### Metal GPU Memory Wiring Issue (2026-02-07)
- Repeated model load/kill/crash cycles accumulate wired kernel memory
- On 24GB M4 Pro: wired reaches 17-19 GB (70-80%), leaving <5 GB
- Even a fresh reboot shows 17.4 GB wired — baseline is high on this machine
- Failed Metal allocations + zombie python processes make it worse
- Gemma 3 12B Q4 (~6.5 GB) needs ~8-10 GB free to load + run inference
- **Only fix**: system reboot + ensure no background apps consume GPU memory
- **Prevention**: memory pressure checks in benchmark; avoid rapid server restart cycles
- **Sandbox note**: Claude Code sandbox blocks Metal GPU access — use dangerouslyDisableSandbox=true
- Model-specific cache budgets: DeepSeek gets 4096 MB (MoE needs intermediate headroom)

### MLX Upstream References (2026-02-07)
- MLX Issue #3078: Metal assertion "Scheduled handler after commit" with concurrent eval — validates async→sync fix
- MLX Issue #2067: MLX is NOT thread-safe (`unordered_map` without sync)
- MLX Issue #2607: shapeless matmul hardcodes initial dimension — validates B=1 split
- No upstream issue for our specific B=2→B=1 compile crash — should file one
- mlx-lm v0.30.4-v0.30.6 fixed many batch generation bugs — batch is still evolving

### Chunked Prefill Sliding Window Corruption (2026-02-08, FIXED)
- `_chunked_prefill()` creates `QuantizedKVCache` for ALL layers
- Gemma 3 uses hybrid attention: sliding window (512 tokens) for 41/46 layers, global for 5/46
- **BUG**: `QuantizedKVCache.make_mask()` ignores `window_size` — returns `"causal"` for N>1
- Sliding window layers get FULL causal attention during chunked prefill → corrupt KV cache
- `BatchQuantizedKVCache.make_mask()` was already correct (always calls `create_causal_mask`)
- **Fix**: Patched `QuantizedKVCache.make_mask` in `mlx_quantized_extensions.py` to create
  explicit mask array when `window_size` is specified
- Validated: 10/10 prompts (1018-2482 tokens) produce correct output with chunking enabled
- Root cause chain: `QuantizedKVCache.make_mask` → `cache.create_attention_mask` →
  returns `"causal"` when `return_array=False`, ignoring `window_size` parameter

### Concurrent Prefill / Interleaved Scheduler (2026-02-08, UPDATED)
- Interleaved chunked prefill NOW ENABLED — Metal crashes fixed by:
  mx.eval (04c814d), mx.synchronize (50a4388), B=1 split (b2d5617),
  clip_residual patch (b2d5617), merge/extend/extract eval (320d25e)
- Default threshold 2048 applies: prompts >2K get chunked interleaved prefill
- Essential for staggered benchmark (Figure 3: User B TTFT reduction)
- OOM note: 4K×2 *simultaneous* prefills still OOM, but interleaved mode processes
  one 256-token chunk at a time, so memory is not an issue

### Benchmark Audit Cleanup (2026-02-08, FIXED)
- PADDING_TEXT fallback for >4K was workaround for sliding window mask bug, now removed
- `SEMANTIC_MLX_DEFAULT_TEMPERATURE` env var does NOTHING: OpenAI adapter uses
  request_body.temperature; coordination service hardcodes T=0.3. Removed from benchmark env.
- `reasoning_extra_tokens` defaults to 0 in settings.py — no env override needed
- `build_server_env` always uses scheduler=on, batch=2 (dead batch_size param removed)
- Non-streaming `ttft_ms` was set to `e2e_ms` by OpenAIRequestClient — now `_build_record`
  correctly sets ttft_ms=0 for non-streaming (TTFT only meaningful for streaming)
- Table 2 relabeled: "Single vs concurrent" (both on same server config)

### Server Lifecycle
- Start: `python -m semantic.entrypoints.cli serve --port 8000`
- Env vars: `SEMANTIC_MLX_MODEL_ID`, `SEMANTIC_MLX_MAX_BATCH_SIZE`, `SEMANTIC_MLX_SCHEDULER_ENABLED`, `SEMANTIC_ADMIN_KEY`, `SEMANTIC_MLX_CACHE_BUDGET_MB`
- Gemma 3 (default): no model env var needed
- DeepSeek: `SEMANTIC_MLX_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"` + `SEMANTIC_MLX_CACHE_BUDGET_MB=4096`
- **Graceful shutdown**: `kill -TERM <pid>` then wait 5s. Check: `lsof -ti:<port>` returns empty
- If SIGTERM doesn't work after 5s: `kill -9 <pid>` then wait 2s
- Find PID: `lsof -ti:8000`
- Readiness check: `curl -sf http://localhost:8000/health/ready`
- Coordination service **hardcodes T=0.3** (line 617) — ignores `SEMANTIC_MLX_DEFAULT_TEMPERATURE` env var
- Sandbox note: Metal GPU requires `dangerouslyDisableSandbox=true`

### Test Infrastructure
- Unit tests: `python -m pytest tests/unit -x -q --timeout=30` (792 tests)
- Integration tests have import issues (ModelTag)
- Smoke tests have conftest.py issues
- Real model tests require starting server on port 8000/8001

## Files to Know
- `src/semantic/application/coordination_service.py` — multi-agent orchestration, prompt building
- `src/semantic/application/chat_completion_service.py` — shared generation logic
- `src/semantic/adapters/inbound/adapter_helpers.py` — tokenization, message merging
- `src/semantic/application/batch_engine.py` — core engine with step/drain
- `demo/scenarios/prisoners_dilemma.yaml` — test scenario
