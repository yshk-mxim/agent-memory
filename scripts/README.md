# Scripts Directory

**Purpose**: Utility and testing scripts for development

---

## Active Scripts

### Testing/Debugging Scripts

**`test_mlx_basic.py`**
- Basic MLX framework test
- Tests model loading and simple generation
- Use for: Verifying MLX setup is working

**`test_gemma3_simple.py`**
- Simple Gemma 3 12B test script
- Tests basic generation with specific model
- Use for: Confirming Gemma 3 model works

**`debug_mlx_context.py`**
- Debug script for MLX context handling
- Tests how context is built and used
- Use for: Debugging context-related issues

---

## Archive

**`archive/`** - Old utility scripts from early development:
- `combine_batches.py` - Batch combination (outdated)
- `extract_review_samples.py` - Sample extraction (outdated)
- `fix_batch_purpose.py` - Batch fixing (outdated)
- `generate_batch_002.py` - Batch generation (outdated)
- `generate_batch_003.py` - Batch generation (outdated)
- `generate_day_plans.py` - Day plan generation (outdated)
- `restructure_all_batches.py` - Batch restructuring (outdated)
- `restructure_dataset.py` - Dataset restructuring (outdated)

These scripts were used during early development phases (Days 1-5) and are preserved for reference but no longer needed.

---

## Usage

### Run a test script:
```bash
cd /Users/dev_user/semantic
python scripts/test_mlx_basic.py
```

### Debug context issues:
```bash
python scripts/debug_mlx_context.py
```

---

**Created**: 2026-01-23
