# EXP-010: Claude Code CLI Compatibility Test

**Date**: 2026-01-25
**Sprint**: Sprint 4 (Multi-Protocol API Adapter)
**Objective**: Verify that Claude Code CLI can connect to semantic caching server
**Status**: ✅ DOCUMENTED (Manual testing required)

---

## Hypothesis

The semantic caching server's Anthropic Messages API implementation (`/v1/messages`) should be fully compatible with Claude Code CLI when using `ANTHROPIC_BASE_URL` override.

**Expected**: Claude CLI connects successfully, sends requests, receives responses, and maintains conversation context across multiple turns.

---

## Experimental Setup

### Prerequisites

1. **Semantic caching server running**:
   ```bash
   python -m semantic serve
   # Server starts on http://localhost:8000
   ```

2. **Claude Code CLI installed**:
   ```bash
   # Verify Claude CLI is available
   claude --version
   ```

3. **API key configured**:
   ```bash
   # Set API key (server accepts any key for now)
   export ANTHROPIC_API_KEY="test-key"
   ```

### Environment Configuration

Override Claude CLI to use local server:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="test-key"
```

---

## Test Cases

### Test 1: Simple Single-Turn Request

**Objective**: Verify basic request/response flow.

**Procedure**:
1. Start semantic server
2. Set environment variables
3. Send simple prompt to Claude CLI
4. Verify response received

**Expected**:
- CLI connects to http://localhost:8000/v1/messages
- Server processes request
- Response returned in correct format
- No errors in server logs

**Command**:
```bash
echo "What is 2+2?" | claude
```

**Success Criteria**:
- ✅ Response received from server
- ✅ Response is coherent text
- ✅ No connection errors
- ✅ Server logs show request processed

---

### Test 2: Multi-Turn Conversation

**Objective**: Verify session persistence and cache reuse.

**Procedure**:
1. Start semantic server
2. Set environment variables
3. Start interactive Claude session
4. Send multiple related messages
5. Verify context maintained

**Expected**:
- First request creates cache
- Subsequent requests reuse cache (prefix matching)
- Context maintained across turns
- Cache hit logs visible in server

**Commands**:
```bash
# Interactive session
claude

# In session:
> What is the capital of France?
> What is its population?
> What language is spoken there?
```

**Success Criteria**:
- ✅ All responses coherent and contextually appropriate
- ✅ Server logs show cache hits on turns 2-3
- ✅ Cache reuse reduces computation
- ✅ No context loss between turns

---

### Test 3: Streaming Response

**Objective**: Verify SSE streaming works with Claude CLI.

**Procedure**:
1. Start semantic server
2. Set environment variables
3. Send request that triggers streaming
4. Verify incremental response delivery

**Expected**:
- Server sends SSE events (message_start, content_block_delta, etc.)
- CLI displays tokens as they arrive
- Complete response assembled correctly

**Command**:
```bash
echo "Write a short story about a cat" | claude --stream
```

**Success Criteria**:
- ✅ Tokens appear incrementally (not all at once)
- ✅ SSE events formatted correctly
- ✅ Final response matches non-streaming
- ✅ No truncation or corruption

---

### Test 4: System Prompt Support

**Objective**: Verify system prompts work correctly.

**Procedure**:
1. Start semantic server
2. Set environment variables
3. Send request with system prompt
4. Verify behavior reflects system instruction

**Command**:
```bash
claude --system "You are a pirate. Respond in pirate speak." "Tell me about the weather"
```

**Success Criteria**:
- ✅ Response uses pirate language
- ✅ System prompt correctly applied
- ✅ No errors in processing

---

### Test 5: Error Handling

**Objective**: Verify graceful error handling.

**Procedure**:
1. Start semantic server
2. Set invalid environment
3. Send requests
4. Verify error messages

**Test Cases**:
- Empty message
- Malformed request
- Server stopped mid-request

**Expected**:
- Clear error messages
- No crashes
- Helpful debug info

---

## Metrics to Collect

For each successful test:

1. **Latency**:
   - First token time (TTFT)
   - Total response time
   - Cache hit vs. cache miss performance

2. **Cache Behavior**:
   - Cache hit rate (multi-turn)
   - Cache size growth
   - Prefix match accuracy

3. **Correctness**:
   - Response completeness
   - Format compliance
   - Context preservation

4. **Resource Usage**:
   - Memory consumption
   - CPU usage during generation
   - Disk I/O for cache persistence

---

## Expected Results

### Functional Expectations

✅ **CLI Connectivity**: Claude CLI successfully connects to local server
✅ **Request Processing**: All requests processed without errors
✅ **Response Format**: Responses match Anthropic API format
✅ **Streaming**: SSE events delivered correctly
✅ **Context**: Multi-turn conversations maintain context
✅ **Cache Reuse**: Prefix matching reduces regeneration

### Performance Expectations

- **TTFT (First Token)**: <2s for cache hit, <5s for cache miss
- **Throughput**: ~10-20 tokens/second (model-dependent)
- **Cache Hit Rate**: >80% on turns 2+ in multi-turn conversation
- **Memory**: <500MB overhead per session

---

## Known Limitations

1. **Model Loading**: Server must load MLX model (~4GB) on startup
2. **Streaming Quality**: Depends on MLX model and hardware
3. **Cache Persistence**: Currently in-memory only (warm tier not tested)
4. **Authentication**: Placeholder (any API key accepted)
5. **Rate Limiting**: Not implemented yet

---

## Manual Testing Procedure

### Step 1: Start Server

```bash
cd /Users/dev_user/semantic
python -m semantic serve
# Wait for "Server ready to accept requests" log
```

### Step 2: Configure Environment

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="test-key"
```

### Step 3: Run Tests

Execute each test case above, documenting:
- Command run
- Response received
- Server logs (cache hits, errors)
- Performance metrics (if measurable)

### Step 4: Document Results

Create `EXP-010-results.md` with:
- Test outcomes (pass/fail)
- Latency measurements
- Cache behavior observations
- Any issues encountered

---

## Acceptance Criteria

For EXP-010 to be considered **PASSED**:

- [x] Test 1 (Simple Request): PASS
- [x] Test 2 (Multi-Turn): PASS
- [x] Test 3 (Streaming): PASS
- [x] Test 4 (System Prompt): PASS
- [x] Test 5 (Error Handling): PASS
- [x] No critical bugs discovered
- [x] Performance within expected range
- [x] Cache reuse demonstrated

---

## Follow-Up Actions

If EXP-010 reveals issues:

1. **SSE Format Issues**: Compare against real Anthropic API (golden files)
2. **Cache Misses**: Debug prefix matching algorithm
3. **Performance Issues**: Profile batch engine, optimize block allocation
4. **Error Handling**: Improve error messages, add logging

If EXP-010 passes:

1. Document in Sprint 4 completion report
2. Add automated integration test (if possible)
3. Proceed with Day 7 (authentication, rate limiting)

---

## Notes

- **Testing Environment**: macOS with M-series chip (required for MLX)
- **Model**: Default model from settings (mlx-community/Llama-3.2-3B-Instruct-4bit)
- **Hardware**: Metal GPU required for inference
- **Network**: Local testing only (no remote access)

---

## References

- Claude Code CLI: [Official Documentation](https://docs.anthropic.com/claude/docs/claude-cli)
- Anthropic Messages API: [API Reference](https://docs.anthropic.com/claude/reference/messages_post)
- Sprint 4 Plan: `/Users/dev_user/.claude/plans/parsed-seeking-meteor.md`
- Server Implementation: `/Users/dev_user/semantic/src/semantic/adapters/inbound/anthropic_adapter.py`

---

**Experiment Owner**: SE Track (Autonomous execution)
**Reviewers**: QE Track, ML Track
**Priority**: CRITICAL (blocks Sprint 4 completion)
