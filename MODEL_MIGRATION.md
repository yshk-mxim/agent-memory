# Model Migration: Llama 3.1 8B → Gemma 3 12B

**Date**: 2026-01-22
**Reason**: Better KV cache management, larger context window, multimodal support

---

## Changes Summary

### Model Specifications

| Feature | Llama 3.1 8B (OLD) | Gemma 3 12B (NEW) |
|---------|-------------------|-------------------|
| Parameters | 8B | 12B |
| Context Window | 128K | 128K |
| Quantized Size (Q4_K_M) | ~4.9GB | ~7-8GB |
| KV Cache @ 32k | ~6-8GB | ~8-10GB |
| KV Cache @ 128k | ~24-30GB | ~25-30GB |
| Multimodal | No | Yes (text + images) |
| Release | July 2024 | January 2026 |
| Benchmark (MMLU) | 66.7% | ~72-75% (estimated) |

### Why Gemma 3 12B?

1. **Better for KV Cache Research**:
   - More manageable cache size than 27B
   - Still has full 128k context window
   - Better performance per parameter

2. **Latest Technology**:
   - Released January 2026 (most recent)
   - Multimodal capabilities for future work
   - Better instruction following

3. **Development Efficiency**:
   - Faster iteration cycles
   - Can run multiple contexts in parallel
   - Good balance of quality vs resource usage

---

## Updated Files

### Code Files
- ✅ `requirements.txt` - Updated comment
- ✅ `src/utils.py` - Renamed `get_llama()` → `get_gemma()`, `call_llama()` → `call_gemma()`
- ✅ `tests/test_apis.py` - Renamed `test_llama_inference()` → `test_gemma_inference()`
- ✅ `README.md` - Updated acknowledgments
- ✅ `data/batch_001.json` - Replaced with restructured version

### Model Path Changes

**Old**:
```python
model_path = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
download_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
```

**New**:
```python
model_path = "models/gemma-3-12b-instruct-Q4_K_M.gguf"
download_url = "https://huggingface.co/google/gemma-3-12b-it-GGUF"
```

---

## Installation Instructions

### Download Gemma 3 12B

**Option 1: Via Ollama (Recommended)**
```bash
ollama pull gemma3:12b-instruct-q4_K_M
```

**Option 2: Direct Download**
```bash
# Search for Gemma 3 12B Q4_K_M GGUF on Hugging Face
# Example (URL may vary):
wget https://huggingface.co/google/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-instruct-Q4_K_M.gguf \
  -O models/gemma-3-12b-instruct-Q4_K_M.gguf
```

**Option 3: Via llama.cpp**
```bash
# Clone llama.cpp and convert from original Gemma 3 checkpoint
# (Advanced - only if you need custom quantization)
```

### Verify Installation

```bash
python -m tests.test_apis
```

Expected output:
```
✓ PASS: claude-haiku-4.5
✓ PASS: claude-sonnet-4.5
✓ PASS: deepseek-r1
✓ PASS: gemma-3-12b

🎉 All API requirements met!
```

---

## API Changes

### Python API

**Old**:
```python
from src.utils import APIClients

clients = APIClients()
response = clients.call_llama("Hello", model_path="...")
```

**New**:
```python
from src.utils import APIClients

clients = APIClients()
response = clients.call_gemma("Hello", model_path="...")
```

### Default Context Window

**Old**: `n_ctx=4096` (4k default)
**New**: `n_ctx=32768` (32k default, max 128k)

---

## Historical Note

The original day plans referenced Llama 3.1 8B as it was state-of-the-art in July 2024. As of January 2026, Gemma 3 12B represents the latest open-source model technology and is better suited for RDIC research goals.

Original plan documents in `plans/` directory have been preserved for historical reference but should be considered outdated regarding model choice.

---

## Performance Expectations

### Inference Speed
- **Gemma 3 12B Q4_K_M**: ~5-10 tokens/sec on M1/M2 Max
- **Target**: <30s for typical responses (same as Llama 3.1)

### Memory Usage
- **Model weights**: ~7-8GB (Q4_K_M)
- **KV cache @ 32k**: ~8-10GB
- **Total VRAM needed**: ~15-18GB recommended
- **Safe for**: M1/M2 Max (32GB+), M3 Max/Ultra

---

## Migration Checklist

- [x] Update requirements.txt
- [x] Update src/utils.py
- [x] Update tests/test_apis.py
- [x] Update README.md
- [x] Restructure dataset (batch_001.json)
- [ ] Download Gemma 3 12B model
- [ ] Run API tests
- [ ] Update any custom scripts using old API

---

**Status**: Code migration complete. Model download pending.
