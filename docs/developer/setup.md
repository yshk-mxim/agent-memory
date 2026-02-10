# Development Environment Setup

## Prerequisites

- Python 3.11 or 3.12
- Apple Silicon Mac (M-series) for MLX inference
- At least 24 GB unified memory recommended (Gemma 3 12B Q4 requires ~8-10 GB free)
- macOS with Metal support

## Clone and Install

```bash
git clone https://github.com/yshk-mxim/agent-memory.git
cd agent-memory

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Pinned Dependencies

The project pins exact versions for MLX due to API instability:

- `mlx==0.30.3`
- `mlx-lm==0.30.4`
- `transformers>=4.47.0,<5.0.0` (v5.0.0rc1 breaks SentencePiece decode; see [huggingface/transformers#43066](https://github.com/huggingface/transformers/issues/43066))

`mlx-lm==0.30.4` declares a dependency on `transformers==5.0.0rc1`. pip will warn about the conflict, but the pinned `<5.0.0` constraint works correctly at runtime.

## Running Tests

### Unit Tests

Unit tests run without GPU access and mock all MLX boundaries. This is the primary test suite (792 tests).

```bash
python -m pytest tests/unit -x -q --timeout=30
```

### With Coverage

```bash
python -m pytest tests/unit --cov=src --cov-report=html --timeout=30
```

Coverage target is 80% (configured in `pyproject.toml`). Coverage excludes MLX adapters, entrypoints, and the batch engine (these require Apple Silicon integration tests).

### Test Markers

The project defines several test markers in `pyproject.toml`:

| Marker | Description |
|---|---|
| `unit` | Fast unit tests with mocked boundaries |
| `integration` | Tests with real MLX and disk I/O (Apple Silicon only) |
| `smoke` | Basic server lifecycle tests |
| `e2e` | Full-stack multi-agent tests (slow, Apple Silicon only) |
| `stress` | Load and concurrency stress tests |
| `benchmark` | Performance benchmarks |
| `property` | Hypothesis property-based tests |
| `live` | Tests requiring a running server on `localhost:8000` |

Run a specific marker:

```bash
python -m pytest tests/ -m unit -x -q --timeout=30
```

### Integration Tests

Integration tests require a running server on port 8000:

```bash
# Terminal 1: start the server
python -m agent_memory.entrypoints.cli serve --port 8000

# Terminal 2: run integration tests
python -m pytest tests/ -m integration
```

## Pre-commit Hooks

The project uses pre-commit for code quality enforcement. Install hooks after cloning:

```bash
pre-commit install
```

Hooks run automatically on `git commit`. To run manually against all files:

```bash
pre-commit run --all-files
```

### Configured Hooks

| Hook | Purpose |
|---|---|
| `trailing-whitespace` | Strip trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `check-yaml` / `check-toml` | Validate config file syntax |
| `check-merge-conflict` | Detect unresolved merge markers |
| `debug-statements` | Catch leftover `breakpoint()` / `pdb` |
| `check-added-large-files` | Block files >500 KB (prevents model weight commits) |
| `ruff-format` | Auto-format code |
| `ruff` | Lint with auto-fix (`--fix --exit-non-zero-on-fix`) |
| `gitleaks` | Secret detection (API keys, tokens) |
| `mypy` | Static type checking (`--strict`, excludes tests) |
| `codespell` | Spell checker |
| `warm-cache-pattern-check` | Custom check for benchmark warm cache patterns |

### Linter Configuration

Ruff is the all-in-one linter (replaces flake8, pylint, bandit, isort, etc.). Configuration is in `pyproject.toml` under `[tool.ruff]`:

- Line length: 100
- Target: Python 3.11
- Max cyclomatic complexity: 10
- Docstring convention: Google-style

### Type Checking

Mypy runs in strict mode. MLX, mlx-lm, and transformers imports are configured to ignore missing stubs:

```bash
mypy src/agent_memory/
```

## Starting the Server Locally

```bash
# Default model (Gemma 3 12B Q4)
python -m agent_memory.entrypoints.cli serve --port 8000

# DeepSeek-Coder-V2-Lite
SEMANTIC_MLX_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" \
SEMANTIC_MLX_CACHE_BUDGET_MB=4096 \
python -m agent_memory.entrypoints.cli serve --port 8000

# Batch mode with scheduler
SEMANTIC_MLX_MAX_BATCH_SIZE=2 \
SEMANTIC_MLX_SCHEDULER_ENABLED=true \
python -m agent_memory.entrypoints.cli serve --port 8000
```

Readiness check:

```bash
curl -sf http://localhost:8000/health/ready
```

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_MLX_MODEL_ID` | `mlx-community/gemma-3-12b-it-4bit` | HuggingFace model ID or local path |
| `SEMANTIC_MLX_MAX_BATCH_SIZE` | `2` | Max concurrent sequences |
| `SEMANTIC_MLX_SCHEDULER_ENABLED` | `true` | Enable interleaved prefill/decode |
| `SEMANTIC_MLX_CACHE_BUDGET_MB` | `8192` | Max cache memory budget in MB |
| `SEMANTIC_ADMIN_KEY` | (empty) | Optional API key for auth |

All MLX settings use the `SEMANTIC_MLX_` prefix. Agent settings use `SEMANTIC_AGENT_`. Server settings use `SEMANTIC_SERVER_`. See `src/agent_memory/adapters/config/settings.py` for the full list.

## Project Layout

```
src/agent_memory/
  entrypoints/cli.py          -- CLI entry point (typer)
  entrypoints/api_server.py   -- FastAPI app, lifespan manager
  application/batch_engine.py -- Core inference engine (step/drain/shutdown)
  application/coordination_service.py -- Multi-agent orchestration
  application/chat_completion_service.py -- Shared generation logic
  application/agent_cache_store.py -- 3-tier cache (hot/warm/cold)
  adapters/config/settings.py -- Pydantic Settings configuration
  adapters/outbound/mlx_quantized_extensions.py -- Q4 KV cache, batch decode
  adapters/outbound/mlx_fused_attention.py -- Fused SDPA patches
  adapters/outbound/mlx_spec_extractor.py -- Model spec extraction
  adapters/outbound/chat_template_adapter.py -- Model-specific template handling
  adapters/inbound/adapter_helpers.py -- Tokenization, message merging
  adapters/inbound/openai_adapter.py -- OpenAI-compatible API
  domain/value_objects.py     -- ModelCacheSpec, ModelTag, etc.
  domain/entities.py          -- BLOCK_SIZE_TOKENS, core domain entities
benchmarks/                   -- Performance benchmarks
tests/unit/                   -- Unit tests (792 tests)
config/models/                -- Model-specific TOML configs
```
