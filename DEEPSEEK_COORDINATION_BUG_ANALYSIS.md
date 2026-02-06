# DeepSeek Coordination Bug Analysis

## Status: CRITICAL BUG - Blocks Production Benchmarks

## Summary

DeepSeek generates perfectly in simple API calls but produces empty/malformed responses in multi-agent coordination scenarios. This is a production-critical bug that must be fixed before running benchmarks.

## Evidence

### Working: Simple API Test
```bash
curl http://localhost:8000/v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Count from 1 to 5"}],"max_tokens":50}'
```
**Result**: ✅ Perfect response with proper formatting and content

### Broken: Multi-Agent Coordination
```python
# Prisoner's Dilemma scenario with 5 phases
```
**Results**:
- ❌ Missing spaces: "Iwillconfess" instead of "I will confess"
- ❌ Empty messages: Many agents produce blank responses
- ❌ Truncated/incoherent output

## Root Cause Investigation

### Key Finding 1: Tokens Generated But Detokenization Fails

**Server logs show**:
```
Turn 3 (Warden):
  - Cache update: _idx: 83->117 (generated 34 tokens) ✓
  - raw_text_length: 0 ❌

Turn 4 (Marco):
  - Cache update: _idx: 95->120 (generated 25 tokens) ✓
  - raw_text_length: 0 ❌
```

**Conclusion**: The model IS generating tokens (KV cache is being updated), but `tokenizer.decode(tokens)` returns empty string.

### Key Finding 2: Issue is in Detokenization, Not Generation

- Added logging to coordination service showing `result.text` is empty BEFORE cleaning
- Added logging to batch engine showing tokens list exists but detokenizes to ""
- This rules out the response cleaning function as the cause

### Key Finding 3: DeepSeek-Specific Issue

- Gemma 3 works perfectly in the same coordination scenario
- Simple DeepSeek API calls work perfectly
- Only breaks in multi-agent/multi-turn coordination

## Hypothesis

**Most Likely**: DeepSeek's tokenizer is generating special tokens (like EOS, padding, or chat template markers) that decode to empty strings when certain prompt patterns are used in multi-turn conversations.

**Possible causes**:
1. DeepSeek's chat template not being applied correctly for multi-agent scenarios
2. Stop tokens being included in the generated tokens list
3. Special tokens (BOS/EOS/padding) decode to empty in certain contexts
4. Prompt format causing model to generate only special tokens

## Code Locations

### Where Tokens Are Decoded
**File**: `src/semantic/application/batch_engine.py:1431`
```python
text = self._tokenizer.decode(tokens)
```

### Where Chat Template is Applied
**File**: `src/semantic/application/coordination_service.py:1022-1073`
```python
def _tokenize_chat_messages(self, messages: list[dict])
```

### Chat Template Adapter
**File**: `src/semantic/adapters/outbound/chat_template_adapter.py`
- Currently has no DeepSeek-specific handling
- Only handles Llama 3.1, ChatML, and GPT-OSS Harmony formats

## Next Steps

1. **Inspect DeepSeek's chat template** - Check what template markers it uses
2. **Add DeepSeek-specific chat template handling** if needed
3. **Log actual token IDs** being generated (not just count)
4. **Decode individual tokens** to see which ones produce empty strings
5. **Compare prompt formatting** between simple API call vs coordination scenario
6. **Check if DeepSeek requires specific tokenizer settings** for multi-turn conversations

## Files Modified (Debug Logging Added)

1. `src/semantic/application/coordination_service.py:579-587` - Log raw generation output
2. `src/semantic/application/batch_engine.py:1431-1438` - Log token count and preview

## Test Commands

```bash
# Start DeepSeek server
semantic serve --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx --port 8000

# Simple test (works ✓)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'

# Coordination test (fails ✗)
python3 test_deepseek_debug.py

# Check logs
grep "raw_generation_output" /tmp/deepseek_debug.log
grep "finalize_tokens" /tmp/deepseek_debug2.log
```

## Impact on Benchmarks

**BLOCKS ALL DEEPSEEK BENCHMARKS** until fixed. Cannot proceed with:
- DeepSeek batch=1 full sweep
- DeepSeek batch=2 concurrent
- DeepSeek staggered arrivals
- Paper validation with DeepSeek

**Gemma 3 benchmarks can proceed** - this issue does not affect Gemma 3.

## Recommendation

Fix this coordination bug before running any production benchmarks. The current implementation would produce invalid/incomplete benchmark data for DeepSeek.
