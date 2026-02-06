# DeepSeek Coordination Issues - Root Cause Analysis

## Summary

DeepSeek works perfectly in simple single-turn OpenAI API calls but produces degraded output in multi-turn coordination scenarios. The root cause is **cascading context corruption**, not model or tokenizer bugs.

## Evidence

### Simple API (Perfect)
```bash
$ python test_deepseek_spaces_api.py
```
- Clean output with proper spacing
- Coherent reasoning (300+ words)
- No meta-text or instruction echoing
- Uses SAME model & tokenizer as coordination

### Coordination (Degraded)
```bash
$ python test_prisoners_dilemma_single.py
```
- Space stripping: "Warden.Marcos,yourunderstandingof"
- Meta-text echoing: "DoNotConfessStaySilent"
- Cascading failures across turns
- Chinese characters in later turns

## Root Cause: Cascading Context Corruption

1. **Turn 1**: Model generates reasonable output
2. **Turn 2**: Previous output included in context as "Agent said: <content>"
3. If Turn 1 had ANY artifacts (spaces missing, meta-text), these get **encoded into Turn 2's prompt**
4. Turn 2 sees malformed context, produces worse output
5. Turn 3 sees BOTH malformed outputs, produces even worse output
6. By Turn 5+, context is completely corrupted

### Example Cascade
```
Turn 1 (Warden): "Ifyouareconfessing,thereareconsequences"  ← Space stripping begins
Turn 2 (Marco): Sees "Warden said: Ifyouareconfessing..."  ← Malformed input
Turn 2 (Marco): "DoNotConfessStaySilent"  ← Echoes meta-instructions
Turn 3 (Danny): Sees both malformed messages
Turn 3 (Danny): "Danny,youshouldstayinsilent" + Chinese  ← Complete breakdown
```

## Why Simple API Works

- **Single turn**: No cascading
- **Clean context**: No "Agent said:" prefixes
- **No persistent KV cache**: Each request is independent
- **Standard format**: Pure User/Assistant roles

## Why Coordination Fails

- **Multi-turn**: Errors compound across turns
- **Complex formatting**: "Agent said:" breaks DeepSeek's User:/Assistant: structure
- **Persistent KV cache**: Carries corrupted context forward
- **Message merging**: Multiple semantic units merged into single user message

## Attempted Fixes

### ✅ Fixed
1. **Double-spacing**: Changed `"\n\n"` to `"\n"` in message merge
2. **Empty message prevention**: Don't add zero-token generations to history
3. **Natural language prompts**: Use "Agent, what do you say?" instead of "[Agent, respond now.]"
4. **Message merging**: Enabled for DeepSeek to avoid consecutive "User:" labels

### ⚠️ Partially Mitigated
1. **"Name said:" format**: Changed from "Name:" to "Name said:" - still confuses model
2. **Initial prompt visibility**: All agents see directive meant for others
3. **Context length**: Even with merging, context becomes unwieldy

### ❌ Cannot Fix
1. **Space stripping in decoded output**: Once model generates space-stripped tokens, they're stored in KV cache
2. **Meta-text echoing**: Model sees "RULES: Don't do X" and echoes it
3. **Cascading corruption**: Each turn makes next turn worse

## Architectural Issues

The user correctly identified that coordination is doing things differently from OpenAI API:

1. **Custom message construction**: `build_agent_prompt()` creates messages with "Agent said:" format
2. **Custom merging**: Coordination merges BEFORE calling template
3. **Persistent context**: KV cache reuse means errors persist

### What Should Happen
Both paths should use identical tokenization:
- OpenAI: messages → template → tokens
- Coordination: messages → template → tokens

### What Actually Happens
- OpenAI: messages → template → tokens
- Coordination: messages + custom formatting → custom merge → template → tokens

## Recommended Actions

### Short Term
1. ✅ Use Gemma 3 for all production benchmarks
2. ✅ Document DeepSeek as "experimental - multi-turn not recommended"
3. ⚠️ Keep DeepSeek fixes for single-turn use cases

### Long Term
1. **Refactor coordination to use OpenAI-style messages**:
   - Remove "Agent said:" prefixes
   - Use assistant role for agent responses when possible
   - Let chat template handle formatting

2. **Investigate alternative multi-speaker formats**:
   - Narrative style: "The Warden replies: ..."
   - No names: Rely on context to track speakers
   - Separate channels: Don't merge all agents into one context

3. **Add validation**:
   - Detect space-stripped output and retry
   - Detect meta-text echoing and clean aggressively
   - Limit cascade depth (evict cache after N corrupted turns)

## Conclusion

DeepSeek works perfectly for single-turn interactions but is not suitable for multi-turn coordination scenarios due to cascading context corruption. The fixes applied reduce failure rates but don't eliminate the core issue.

**Verdict**: Use Gemma 3 for coordination benchmarks. DeepSeek can be used for simple API scenarios.
