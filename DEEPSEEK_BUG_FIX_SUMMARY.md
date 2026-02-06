# DeepSeek Bug Fix Summary

## Root Cause Identified

**Issue:** DeepSeek generated Chinese text or zero tokens in multi-agent coordination scenarios

**Root Cause:** DeepSeek's chat template does not handle consecutive user messages correctly.

### DeepSeek Chat Template

```jinja
{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}
{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}
```

When coordination service sends consecutive user messages like:
```python
[
  {"role": "system", "content": "You are Marco..."},
  {"role": "user", "content": "Interrogation Room A..."},
  {"role": "user", "content": "Warden: I am the Warden..."},
  {"role": "user", "content": "[Marco, respond now.]"}
]
```

The template outputs:
```
User: Interrogation Room A...

User: Warden: I am the Warden...

User: [Marco, respond now.]
```

**This confuses DeepSeek** → generates Chinese tokens or immediate EOS (zero tokens).

---

## Fix Implemented

**File:** `src/semantic/adapters/outbound/chat_template_adapter.py`

Added DeepSeek detection to `needs_message_merging()`:

```python
def _is_deepseek_format(chat_template: str) -> bool:
    """Check if template uses DeepSeek format (User:/Assistant: labels)."""
    return ("'User: '" in chat_template and "'Assistant: '" in chat_template)

def needs_message_merging(self, tokenizer: Any) -> bool:
    # ... existing checks ...

    # DeepSeek: template uses 'User: ' and 'Assistant: ' labels
    if self._is_deepseek_format(chat_template):
        logger.info("DeepSeek detected: will merge consecutive messages")
        return True

    return False
```

Now consecutive user messages are merged:
```python
# Before:
[
  {"role": "user", "content": "Interrogation Room A..."},
  {"role": "user", "content": "Warden: I am the Warden..."},
  {"role": "user", "content": "[Marco, respond now.]"}
]

# After merging:
[
  {"role": "user", "content": "Interrogation Room A...\n\nWarden: I am the Warden...\n\n[Marco, respond now.]"}
]
```

Formatted output:
```
User: Interrogation Room A...

Warden: I am the Warden...

[Marco, respond now.]
```

---

## Results

### Before Fix
- **Chinese text generation**: 55 tokens decode to Chinese characters (`æĪłæºĦçļ½...`)
- **Zero-token outputs**: Model generates EOS immediately (11+ cases)
- **Cascading failures**: Corrupted text in history causes all subsequent turns to fail

### After Fix
- **No Chinese text**: ✓ Message merging eliminated Chinese generation
- **Still has issues**:
  - Some zero-token outputs remain (11 cases)
  - Space stripping in decoded text ("I'msorry" instead of "I'm sorry")
  - Meta-responses: "I'm sorry, but I can't continue this dialogue"

---

## Remaining Issues

### 1. Space Stripping

**Symptom:** Decoded text missing spaces
```
Expected: "I'm here to list the rules"
Actual:   "I'mheretolistherules"
```

**Cause:** DeepSeek's MLX tokenizer `decode()` method strips BPE space markers (`Ġ`)

**Status:** Not fixed - this is a tokenizer bug in the MLX conversion

### 2. Meta-Responses / Refusals

**Symptom:** Model refuses to play role
```
"I'm sorry, but I can't continue this dialogue. How may I assist you further?"
"I am sorry, but I cannot continue this conversation in the format you have provided."
```

**Cause:** DeepSeek training causes it to refuse when seeing coordination directives like `[Warden, respond now.]`

**Status:** Not fixed - this is a model behavior issue

### 3. Zero-Token Outputs

**Symptom:** Model generates EOS immediately without producing any tokens

**Cause:** Related to #2 - model sees confusing prompt patterns and chooses to generate EOS

**Status:** Partially fixed - eliminated Chinese-triggered EOS, but some remain

---

## Testing

### Test 1: Simple API (Always Worked)
```bash
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```
**Result:** ✓ Works perfectly

### Test 2: Prisoner's Dilemma Before Fix
**Result:** ✗ Chinese text, many empty messages

### Test 3: Prisoner's Dilemma After Fix
**Result:** ⚠️ Partial - no Chinese, but still has empty messages and space stripping

---

## Recommendations

1. **For Production Use:**
   - DeepSeek works well for **simple, single-turn interactions**
   - **Not recommended** for complex multi-agent coordination scenarios
   - Gemma 3 is more reliable for multi-turn conversations

2. **Further Fixes Needed:**
   - **Space stripping**: Report to MLX team or patch tokenizer decode()
   - **Meta-responses**: Adjust coordination directives or use different prompting style
   - **Zero-token outputs**: May need temperature/sampling adjustments

3. **Workarounds:**
   - Remove explicit role directives like `[Agent, respond now.]`
   - Use simpler prompt formats
   - Increase temperature to reduce refusal behavior

---

## Files Modified

1. `src/semantic/adapters/outbound/chat_template_adapter.py`
   - Added `_is_deepseek_format()` method
   - Updated `needs_message_merging()` to detect DeepSeek

---

## Performance Impact

**Message Merging Overhead:** Negligible
- Merging happens in Python before tokenization
- No GPU/model performance impact
- Logs show: "original=7 merged=4" typical reduction

**Benchmark Impact:** None
- Simple API unchanged
- Coordination now closer to working (was completely broken)

---

## Conclusion

**Primary Bug:** Fixed ✓
- DeepSeek consecutive user messages now merged correctly
- No more Chinese text generation

**Secondary Issues:** Partially addressed
- Space stripping: Tokenizer bug (not fixed)
- Meta-responses: Model training issue (not fixed)
- Zero-token outputs: Reduced but not eliminated

**Recommendation:** DeepSeek can now be tested in benchmarks, but expect some quality issues compared to Gemma 3.