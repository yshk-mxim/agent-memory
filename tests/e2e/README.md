# End-to-End (E2E) Tests

## Overview

E2E tests validate the semantic caching system from end to end with real components:
- Real MLX models loaded
- Real HTTP server running
- Real file system persistence
- Real concurrent sessions

**Requirements**: Apple Silicon (M1/M2/M3/M4) with MLX support

---

## Running E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run specific E2E test
pytest tests/e2e/test_multi_agent_sessions.py::test_five_concurrent_claude_code_sessions -v

# Skip E2E tests (for CI or non-MLX systems)
pytest tests/ -v -m "not e2e"
```

---

## Test Categories

### Multi-Agent Sessions (`test_multi_agent_sessions.py`)
Tests concurrent agent isolation and cache independence:
- 5 concurrent Claude Code sessions
- Independent cache storage
- No cache leakage between agents
- All agents generate correctly

### Cache Persistence (`test_cache_persistence.py`)
Tests cache save/load across server restarts:
- Cache persists to disk
- Agent resumes from saved cache
- Cache load time <500ms
- Model tag compatibility validation

### Model Hot-Swap (`test_model_hot_swap_e2e.py`)
Tests model swapping with active agents:
- Swap model mid-session
- Active agents drain successfully
- New model loads and serves requests
- Rollback on swap failure

---

## Test Infrastructure

### Fixtures (`conftest.py`)

**`live_server`**:
- Starts FastAPI server in subprocess
- Yields server URL (http://localhost:PORT)
- Automatically tears down on test completion
- Captures server logs for debugging

**`test_client`**:
- HTTP client configured for E2E testing
- Handles authentication headers
- Supports both sync and async requests

**`cleanup_caches`**:
- Clears test cache directories after each test
- Prevents test pollution
- Automatic teardown

### Threading Patterns

E2E tests use `threading.Thread` for concurrent scenarios:
- `threading.Barrier` for synchronized start
- Result collection via thread-safe lists
- Exception handling in worker threads
- Pattern reference: `tests/integration/test_concurrent.py`

---

## Best Practices

1. **Cleanup**: Always use `cleanup_caches` fixture to prevent test pollution
2. **Timeouts**: Set reasonable timeouts for server startup (60s) and operations
3. **Isolation**: Each test should be independent and not rely on previous test state
4. **Real Components**: Use real MLX models, not mocks (validates actual behavior)
5. **Performance**: Measure actual timings, not mocked delays

---

## Debugging Failed Tests

### Server Won't Start
```bash
# Check if port is already in use
lsof -i :8000

# View server logs (captured in test output)
pytest tests/e2e/test_foo.py -v -s
```

### Cache Issues
```bash
# Inspect cache directory
ls -la ~/.cache/semantic/test/

# Clean up manually if needed
rm -rf ~/.cache/semantic/test/
```

### MLX Errors
```bash
# Verify MLX installed and working
python -c "import mlx.core as mx; print(mx.metal.get_active_memory())"

# Check Apple Silicon
uname -m  # Should show: arm64
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Server startup | <60s | Reasonable model load time |
| Cache load | <500ms | Sprint 6 requirement |
| Model swap | <30s | EXP-012 validated |
| 5 concurrent agents | <2s p95 latency | Production ready |

---

## Adding New E2E Tests

1. Import fixtures from `conftest.py`
2. Mark test with `@pytest.mark.e2e`
3. Use `live_server` fixture for server URL
4. Use `cleanup_caches` for automatic cleanup
5. Document expected behavior and performance targets
6. Handle errors gracefully and provide clear failure messages

Example:
```python
import pytest

@pytest.mark.e2e
def test_my_e2e_scenario(live_server, cleanup_caches):
    \"\"\"Test description and expected behavior.\"\"\"
    # Arrange
    server_url = live_server

    # Act
    # ... test logic ...

    # Assert
    assert result_matches_expectations
```

---

**Last Updated**: 2026-01-25 (Sprint 6 Day 0)
