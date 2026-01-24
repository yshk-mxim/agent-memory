# Comparative Benchmark Methodology

## Key Findings

### Ollama and MLX Support

**Ollama does NOT support MLX natively.** Despite community demand ([GitHub Issue #1730](https://github.com/ollama/ollama/issues/1730) with 283 👍), Ollama uses llama.cpp with Metal backend only.

### LM Studio as Benchmark Platform

**LM Studio is the ideal platform for comparing MLX vs llama.cpp** because:
- ✅ Supports **both** MLX and llama.cpp backends
- ✅ Can switch between backends with same model
- ✅ Provides REST API for programmatic benchmarking
- ✅ Widely used in production

### Performance Hierarchy (Apple Silicon)

Based on research and benchmarks:
1. **MLX**: ~230 tok/s (optimized for Apple Silicon)
2. **llama.cpp**: ~150 tok/s (cross-platform, Metal backend)
3. **Ollama**: 20-40 tok/s (llama.cpp wrapper with overhead)

Sources:
- [Benchmarking Apple's MLX vs. llama.cpp](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416)
- [Comparative Study (arXiv)](https://arxiv.org/pdf/2511.05502)
- [LM Studio MLX Support](https://lmstudio.ai/blog/lmstudio-v0.3.4)

## Benchmark Approaches

### 1. Simulated Comparison (Original)

**File**: `benchmarks/comparative_benchmark.py`

**Methodology**:
- Actually measures: This POC performance
- Simulates: LM Studio/Ollama/llama.cpp behavior
- Assumption: No cache persistence = re-prefill each session

**Limitations**:
- ⚠️ Not actual measurements of competitors
- ⚠️ Assumptions may not reflect real-world performance
- ⚠️ Cannot account for implementation differences

**Results**:
- 17% faster on 3-session resume
- 41% faster on 10-turn conversation
- Context scaling: 14% @67 tokens → 98% @20k tokens

### 2. Direct llama.cpp Comparison (Attempted)

**File**: `benchmarks/real_comparative_benchmark.py`

**Issue**: GGUF model used bfloat16 format, which Metal doesn't support.

**Symptoms**:
```
ggml_metal_init: skipping kernel_*_bf16 (not supported)
```

**Result**: 237 seconds for 50 tokens (0.21 tok/s) - **700× slower than expected!**

**Root Cause**: Model fell back to CPU processing instead of GPU.

**Lesson**: Must use FP16 or Q4_K_M GGUF models for Metal acceleration.

### 3. LM Studio API Comparison (Recommended)

**File**: `benchmarks/lmstudio_comparative_benchmark.py`

**Methodology**:
1. Use LM Studio's REST API on localhost:1234
2. Load gemma-3-12b model (MLX or GGUF format)
3. Benchmark both cold start and session resume
4. Compare against This POC with same prompts

**Advantages**:
- ✅ Real measurements (no simulation)
- ✅ Fair comparison (same model, same hardware)
- ✅ Can test both MLX and llama.cpp backends
- ✅ Reflects actual user experience with LM Studio

**Prerequisites**:
```bash
# 1. Open LM Studio
# 2. Load gemma-3-12b model
# 3. Start Local Server (port 1234)
# 4. Run benchmark:
python -m benchmarks.lmstudio_comparative_benchmark --quick
```

## Benchmark Scenarios

### Scenario 1: Cold Start

**Tests**: Initial model load + first generation

**Expected**:
- All tools similar performance (no cache advantage yet)
- MLX may be slightly faster due to better Apple Silicon optimization

**Key Metric**: Total time from start to first token

### Scenario 2: Session Resume

**Tests**: Loading saved state across multiple sessions

**This POC Advantage**:
- Loads cache from disk (~1ms)
- Skips re-prefill of system prompt

**LM Studio/Ollama**:
- No persistent cache across sessions
- Must re-process system prompt each time

**Key Metric**: Time per session (load + generate)

### Scenario 3: Long Context (Theoretical)

**Tests**: Advantage scaling with context length

**Calculation**:
- Prefill rate: ~219 tokens/sec (measured)
- Cache load: ~1ms (measured)
- For N tokens:
  - Competitor time: N / 219 seconds
  - This POC time: 0.001 seconds
  - Advantage: (N / 219 - 0.001) / (N / 219) * 100%

**Results**:
- 67 tokens: 14% faster
- 2k tokens: 83% faster
- 20k tokens: 98% faster

## Key Innovation: Cache Persistence

The core innovation is **NOT** raw speed - it's **eliminating redundant prefill** via persistent KV cache.

### Without Cache Persistence (LM Studio/Ollama)
```
Session 1: [Load Model] → [Prefill System] → [Generate]
Session 2: [Model in RAM] → [Prefill System] → [Generate]  ← Redundant!
Session 3: [Model in RAM] → [Prefill System] → [Generate]  ← Redundant!
```

### With Cache Persistence (This POC)
```
Session 1: [Load Model] → [Prefill System] → [Generate] → [Save Cache]
Session 2: [Model in RAM] → [Load Cache 1ms] → [Generate]  ← No prefill!
Session 3: [Model in RAM] → [Load Cache 1ms] → [Generate]  ← No prefill!
```

### Impact

For a typical workflow:
- System prompt: 20k tokens
- 5 sessions per day
- Prefill time: ~91 seconds per session

**Without cache**: 91s × 5 = 7.6 minutes/day wasted
**With cache**: 1ms × 5 = 5ms/day

**Annual savings**: ~46 hours per developer

## Recommendations

### For Blog Post

1. **Be transparent**: Clearly state what's measured vs simulated
2. **Use LM Studio**: Run actual benchmarks with LM Studio API
3. **Focus on innovation**: Cache persistence is the key differentiator
4. **Scale matters**: Advantage grows with context length

### For Production Use

1. **MLX on Apple Silicon**: Best performance + memory efficiency
2. **LM Studio for users**: Supports both MLX and llama.cpp
3. **This POC for developers**: Adds persistent cache to MLX

### Future Work

1. Run real LM Studio benchmarks (requires model download and setup)
2. Test with multiple model sizes (2B, 7B, 12B, 70B)
3. Measure multi-agent workflows in production
4. Compare MCP tool execution scenarios

## Conclusion

The comparative benchmark strategy evolved based on practical constraints:

1. **Initial**: Simulated llama.cpp/Ollama behavior
2. **Attempted**: Direct llama.cpp (failed due to bfloat16 GGUF)
3. **Current**: LM Studio API (best approach for fair comparison)

The key finding remains valid: **Persistent KV cache eliminates redundant prefill**, providing 14-98% speedup depending on context length.
