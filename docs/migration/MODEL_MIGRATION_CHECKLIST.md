# Model Migration Checklist

**Date**: 2026-01-26
**From**: `mlx-community/gemma-3-12b-it-4bit`
**To**: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`

**Reason**: Gemma 3 MLX implementation has 8K context limit, but Claude Code CLI requires 64K+ context.

---

## Files Requiring Updates

### Priority 1: Configuration Files (CRITICAL)

These files control the default model used by the server:

1. **`config/default.toml`** ⚠️ CRITICAL
   - Line 9: `model_id = "mlx-community/gemma-3-12b-it-4bit"`
   - **Change to**: `model_id = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"`
   - **Impact**: Sets default model for all deployments

2. **`src/semantic/adapters/config/settings.py`** ⚠️ CRITICAL
   - Line 25: `default="mlx-community/gemma-3-12b-it-4bit"`
   - **Change to**: `default="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"`
   - **Impact**: Pydantic settings default value

3. **`config/.env.example`** (Documentation)
   - Line 8: `# SEMANTIC_MLX__MODEL_ID=mlx-community/Llama-3.1-8B-Instruct-4bit`
   - **Change to**: `# SEMANTIC_MLX__MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`
   - **Impact**: Example for users copying .env

---

### Priority 2: User Documentation (HIGH)

User-facing documentation that references the default model:

4. **`README.md`**
   - Line 31: "**Gemma 3** (12B 4-bit) - Default production model"
   - Line 78: `"model": "gemma-3-12b-it-4bit"`
   - Line 92: `"model": "gemma-3-12b-it-4bit"`
   - Line 106: `"model": "gemma-3-12b-it-4bit"`
   - Line 200: `SEMANTIC_MLX_MODEL_ID=mlx-community/gemma-3-12b-it-4bit`
   - Line 250: `'{"model": "gemma-3-12b-it-4bit", "messages": [...]}'`
   - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - **Impact**: First documentation users see

5. **`docs/configuration.md`**
   - Line 18: `SEMANTIC_MLX_MODEL_ID=mlx-community/gemma-3-12b-it-4bit`
   - Line 44: `| \`SEMANTIC_MLX_MODEL_ID\` | string | \`mlx-community/gemma-3-12b-it-4bit\``
   - Line 56: `- \`mlx-community/gemma-3-12b-it-4bit\` (default, 12GB model, ~6GB quantized)`
   - Line 60: "- Gemma 3: 16GB+ RAM recommended"
   - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - **Update memory**: "DeepSeek-Coder-V2-Lite: 16GB+ RAM recommended"
   - **Impact**: Configuration reference guide

6. **`docs/deployment.md`**
   - Line 25: "- Gemma 3: 16GB+ recommended"
   - Line 93: `SEMANTIC_MLX_MODEL_ID=mlx-community/gemma-3-12b-it-4bit`
   - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - **Impact**: Deployment instructions

7. **`docs/installation.md`**
   - Line 114: `model_id = "mlx-community/gemma-3-12b-it-4bit"`
   - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - **Impact**: Installation guide

8. **`docs/model-onboarding.md`**
   - Likely has references to Gemma 3 as example
   - **Action**: Read and update as needed

9. **`docs/user-guide.md`**
   - Likely has references to Gemma 3 in examples
   - **Action**: Read and update as needed

10. **`docs/faq.md`**
    - Likely has references to supported models
    - **Action**: Read and update as needed

11. **`docs/model-hot-swap.md`**
    - Line 84: `"old_model_id": "mlx-community/gemma-3-12b-it-4bit"`
    - Line 95: `- \`mlx-community/gemma-3-12b-it-4bit\``
    - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx

---

### Priority 3: Test Files (MEDIUM)

Test assertions that expect specific model names:

12. **`tests/unit/test_settings.py`**
    - Line 26: `assert settings.model_id == "mlx-community/gemma-3-12b-it-4bit"`
    - Line 168: `assert settings.mlx.model_id == "mlx-community/gemma-3-12b-it-4bit"`
    - **Change to**: `assert settings.model_id == "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"`
    - **Impact**: Unit tests will fail without this change

**NOTE**: Tests using `SmolLM2-135M-Instruct` should NOT be changed - that's the lightweight test model.

---

### Priority 4: Code Documentation (LOW)

Docstrings and code examples:

13. **`src/semantic/application/model_registry.py`**
    - Line 40: `>>> registry.load_model("mlx-community/gemma-3-12b-it-4bit")`
    - Line 61: `model_id: HuggingFace model ID (e.g., "mlx-community/gemma-3-12b-it-4bit")`
    - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
    - **Impact**: Code examples in docstrings

14. **`src/semantic/application/ports.py`**
    - Line 26: `model_id: HuggingFace model ID (e.g., "mlx-community/gemma-3-12b-it-4bit")`
    - **Update to**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
    - **Impact**: Protocol docstrings

---

### NOT Changing (Historical/Legacy)

These files contain historical information or legacy experiments - DO NOT UPDATE:

- ❌ `LIVE_OBSERVATION_ANALYSIS.md` - Historical test results
- ❌ `MODEL_RECOMMENDATION.md` - Analysis document
- ❌ `experiments/exp_004_cache_extraction.py` - Legacy experiment
- ❌ `experiments/exp_003_cache_injection.py` - Legacy experiment
- ❌ `benchmarks/long_context_benchmark.py` - Historical benchmark
- ❌ `benchmarks/lmstudio_comparative_benchmark.py` - Historical benchmark
- ❌ Archive and project documentation - Historical records

---

## Update Summary

**Total files to update**: 14

**Breakdown**:
- Configuration: 3 files (CRITICAL)
- User docs: 8 files (HIGH)
- Tests: 1 file (MEDIUM)
- Code docs: 2 files (LOW)

**Model Changes**:
- **From**: `mlx-community/gemma-3-12b-it-4bit`
- **To**: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`

**Context Support**:
- Gemma 3: 8K tokens (MLX implementation) ❌
- DeepSeek-Coder-V2-Lite: 163K tokens ✅

**Memory Requirements**:
- Gemma 3: ~12GB (6GB quantized)
- DeepSeek-Coder-V2-Lite: ~20GB (10GB quantized, estimated)

---

## Verification Checklist

After updates, verify:

- [ ] `semantic serve` starts successfully
- [ ] Model loads: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`
- [ ] Test with 49K token cache (should work without memory explosion)
- [ ] Generation speed: 50-100+ tok/s (not 25 tok/s)
- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] Settings tests pass: `pytest tests/unit/test_settings.py -v`
- [ ] Documentation builds: `make docs-build` (if applicable)

---

## Notes

1. **Model name is long**: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` (56 characters)
2. **Memory increase**: DeepSeek-Coder-V2-Lite uses ~20GB vs Gemma 3's ~12GB
3. **Context advantage**: 163K vs 8K (20x improvement)
4. **Coding focus**: DeepSeek-Coder is optimized for code generation
5. **Claude Code CLI**: Will work properly with 163K context support

---

**Status**: Ready for implementation
**Next Step**: Update all 14 files in priority order
