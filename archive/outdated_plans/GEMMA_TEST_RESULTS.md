# Gemma 3 12B Test Results

**Date**: 2026-01-22
**Model**: gemma-3-12b-it-Q4_K_M.gguf (6.8GB)
**Status**: ✅ PASSED

---

## Test Summary

All Gemma 3 12B tests passed successfully:

- ✅ Model loads successfully (1.78s)
- ✅ Basic generation works (2.28s)
- ✅ Instruction following functional
- ✅ Multi-turn conversation works (2.15s)
- ⚠️  Context window: 8,192 tokens (limited from full 128k)

---

## Test Output

```
[Test 1] Simple generation test...
✓ Response: Hello there friend!
✓ Time: 2.28s
✓ Generation under 30s threshold

[Test 2] Instruction following test...
✓ Response: Red, Blue, Green
✓ Time: 0.39s

[Test 3] Context window test...
✓ Context window: 8,192 tokens
⚠ Context window < 32k (may limit RDIC experiments)

[Test 4] Multi-turn conversation test...
Turn 1: Understood. I acknowledge that I am to respond in a formal and professional tone...
Turn 2: 2 + 2 equals 4.
✓ Multi-turn completed in 2.15s
```

---

## Context Window Limitation

**Issue**: The model was trained with 131k context window but currently configured to use 8k.

**Reason**: Using larger context (32k, 128k) causes `llama_decode returned -3` error, likely due to:
- VRAM limitations on the current system
- Batch size or cache size incompatibility
- Model quantization (Q4_K_M) may affect maximum usable context

**Workaround**: Use 8k context for now. This is still sufficient for:
- Initial RDIC experiments (multi-turn conversations)
- Dataset generation validation
- Proof-of-concept testing

**Future**: Can increase context window to 16k, 32k, or 128k by:
1. Using a higher-end GPU with more VRAM
2. Adjusting batch size parameters
3. Using FP16 or higher precision (but larger model size)
4. Testing with newer llama-cpp-python versions

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Model Load Time | 1.78s |
| Simple Generation | 2.28s |
| Instruction Following | 0.39s |
| Multi-turn (2 exchanges) | 2.15s |
| Context Window | 8,192 tokens |
| Generation Speed | ~5-10 tokens/sec (estimated) |

---

## Model Details

- **Source**: [ggml-org/gemma-3-12b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF)
- **Quantization**: Q4_K_M
- **File Size**: 6.8GB
- **Parameters**: 12B
- **Architecture**: Gemma 3 (multimodal capable)
- **Training Context**: 131,072 tokens
- **Usable Context**: 8,192 tokens (current configuration)

---

## Recommendations

1. **For RDIC Development**: 8k context is sufficient for initial experiments
2. **For Production**: Consider:
   - Upgrading to larger VRAM GPU
   - Using FP16 model for full 128k context (but 13-14GB file)
   - Testing on cloud GPU with more VRAM
3. **Alternative**: Gemma 3 4B Q4_K_M might support larger context with less VRAM

---

## Next Steps

- [x] Gemma 3 12B installed and tested
- [x] Basic inference verified
- [x] Instruction following confirmed
- [ ] Test with longer conversations (within 8k limit)
- [ ] Benchmark KV cache behavior
- [ ] Implement RDIC context isolation with Gemma 3

---

**Status**: Gemma 3 12B is ready for RDIC research with 8k context window.
