# Tests

## Test categories

| Category | Location | Count | Requires GPU | Description |
|----------|----------|-------|-------------|-------------|
| Unit | `tests/unit/` | 792 | No | Fast tests with mocked MLX boundaries |
| Integration | `tests/integration/` | ~40 | Yes | Real server + API tests |
| MLX | `tests/mlx/` | ~15 | Yes | Real MLX model + cache round-trip tests |
| E2E | `tests/e2e/` | ~10 | Yes | Full server lifecycle, cache persistence |
| Stress | `tests/stress/` | ~5 | Yes | Concurrent load, memory pressure |
| Benchmarks | `tests/benchmarks/` | ~5 | Yes | Performance regression tests |

## Running tests

```bash
# Unit tests only (fast, no GPU needed, ~3 seconds)
python -m pytest tests/unit -x -q --timeout=30

# With verbose output
python -m pytest tests/unit -v --timeout=30

# Single test file
python -m pytest tests/unit/test_batch_engine.py -v

# Integration tests (requires running server on port 8000)
python -m pytest tests/integration -v

# E2E tests (starts/stops server automatically)
python -m pytest tests/e2e -v

# All tests with coverage
python -m pytest tests/ --cov=src/agent_memory --cov-report=term-missing
```

## Known issues

- Integration tests require `pip install -e .` (editable install) for import resolution
- E2E tests start their own server processes and need port 8000 free
- MLX tests require Apple Silicon hardware
- Stress tests can consume significant GPU memory; check `memory_pressure` before running

## Test structure

Tests mirror the source layout:

```
tests/
  unit/
    adapters/          — adapter layer tests (request parsing, admin API, etc.)
    application/       — application service tests (scheduler, batch engine, etc.)
    test_*.py          — domain and cross-cutting tests
  integration/         — real HTTP API tests with mocked or real MLX
  mlx/                 — tests that load real MLX models
  e2e/                 — full server lifecycle tests
  stress/              — concurrent load tests
  benchmarks/          — performance regression tests
  conftest.py          — shared fixtures
```
