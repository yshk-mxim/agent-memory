# Testing Guide

Comprehensive testing strategy for agent-memory.

## Table of Contents

- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Model-Specific Tests](#model-specific-tests)
- [Tool Calling Tests](#tool-calling-tests)
- [CI/CD Integration](#cicd-integration)
- [Coverage Requirements](#coverage-requirements)

## Test Categories

agent-memory uses a multi-tier testing strategy:

| Category | Location | Purpose | Speed | Model Loading |
|----------|----------|---------|-------|---------------|
| **Unit** | `tests/unit/` | Isolated component testing | Fast (< 1s) | No |
| **Integration** | `tests/integration/` | API endpoint testing | Medium (1-5s) | No |
| **Integration (WithModel)** | `tests/integration/` | Full MLX model testing | Slow (10-60s) | Yes |
| **Smoke** | Manual | Quick health checks | Fast | Optional |
| **Benchmark** | `tests/benchmarks/` | Performance validation | Slow | Yes |

## Running Tests

### Quick Start

Run all unit tests:

```bash
pytest tests/unit/ -v
```

Run all integration tests (no model loading):

```bash
pytest tests/integration/ -k "not WithModel" -v
```

Run all tests including model loading:

```bash
pytest tests/ -v
```

### Test Markers

Tests are marked by category:

```python
@pytest.mark.unit
def test_unit_example():
    """Unit test - no external dependencies"""

@pytest.mark.integration
def test_integration_example():
    """Integration test - uses TestClient, no model"""

@pytest.mark.WithModel
def test_with_model_example():
    """Integration test - loads real MLX model"""
```

### Running by Marker

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only model tests
pytest -m WithModel -v

# Run integration tests excluding model tests
pytest -m "integration and not WithModel" -v
```

### Verbose Output

```bash
# Show test names and results
pytest tests/ -v

# Show full output (print statements)
pytest tests/ -v -s

# Show coverage
pytest tests/ -v --cov=src/agent_memory --cov-report=term-missing
```

### Specific Test Files

```bash
# Run single test file
pytest tests/unit/test_batch_engine.py -v

# Run specific test function
pytest tests/unit/test_batch_engine.py::test_submit_generation -v

# Run specific test class
pytest tests/integration/test_anthropic_tool_calling.py::TestAnthropicToolCalling -v
```

## Unit Tests

**Location**: `tests/unit/`

**Purpose**: Test individual components in isolation

**Examples**:

```bash
# Core domain tests
pytest tests/unit/test_batch_engine.py -v
pytest tests/unit/test_agent_cache_store.py -v
pytest tests/unit/test_model_cache_spec.py -v

# Adapter tests
pytest tests/unit/test_request_models.py -v
pytest tests/unit/test_health.py -v
```

**Characteristics**:
- No external dependencies (no network, no file I/O)
- Fast execution (< 1 second per test)
- Use mocks/stubs for dependencies
- Test single responsibility

**Example Unit Test**:

```python
@pytest.mark.unit
def test_model_cache_spec_extraction():
    """Test ModelCacheSpec extraction from model config."""
    config = {"num_hidden_layers": 42, "num_key_value_heads": 16, "head_dim": 256}
    spec = ModelCacheSpec.from_model_config(config, block_size=256)

    assert spec.num_layers == 42
    assert spec.num_kv_heads == 16
    assert spec.head_dim == 256
```

## Integration Tests

**Location**: `tests/integration/`

**Purpose**: Test API endpoints end-to-end

### Without Model Loading

Tests that use `TestClient` with mocked model responses:

```bash
pytest tests/integration/test_health.py -v
pytest tests/integration/test_anthropic_api.py -k "not WithModel" -v
pytest tests/integration/test_openai_api.py -k "not WithModel" -v
```

**Characteristics**:
- Use FastAPI `TestClient`
- Mock batch engine responses
- Fast execution (1-5 seconds)
- Test request/response formats

### With Model Loading

Tests that load real MLX models:

```bash
pytest tests/integration/test_anthropic_api.py -k "WithModel" -v
pytest tests/integration/test_openai_api.py -k "WithModel" -v
pytest tests/integration/test_gemma3_model.py -v
```

**Characteristics**:
- Load full MLX model (SmolLM2 or Gemma 3)
- Slow execution (10-60 seconds per test)
- Test real inference pipeline
- Validate cache persistence
- Require Metal GPU (Apple Silicon)

**Example Integration Test**:

```python
@pytest.mark.integration
class TestAnthropicAPI:
    def test_create_message(self):
        """Test POST /v1/messages endpoint."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "content" in data
            assert data["stop_reason"] in ["end_turn", "max_tokens"]
```

## Model-Specific Tests

Test specific model architectures and configurations.

### Gemma 3 Tests

**File**: `tests/integration/test_gemma3_model.py`

```bash
# Run all Gemma 3 tests
pytest tests/integration/test_gemma3_model.py -v

# Specific test
pytest tests/integration/test_gemma3_model.py::TestGemma3AnthropicAPI::test_gemma3_anthropic_api -v
```

**Tests**:
1. `test_gemma3_anthropic_api` - Anthropic Messages API
2. `test_gemma3_openai_api` - OpenAI Chat Completions
3. `test_gemma3_direct_agent_api` - Direct Agent API
4. `test_gemma3_cache_grows_across_requests` - Cache persistence
5. `test_gemma3_model_spec_extraction` - ModelCacheSpec validation

### SmolLM2 Tests

SmolLM2 is used in unit tests as a lightweight model:

```bash
# Tests using SmolLM2
pytest tests/integration/ -k "WithModel" -v
```

## Tool Calling Tests

Test tool calling functionality for both API formats.

### Anthropic Tool Use Tests

**File**: `tests/integration/test_anthropic_tool_calling.py`

```bash
# Run all Anthropic tool tests
pytest tests/integration/test_anthropic_tool_calling.py -v

# Specific test
pytest tests/integration/test_anthropic_tool_calling.py::TestAnthropicToolCalling::test_tool_use_single_call -v
```

**Tests**:
1. `test_tool_use_single_call` - Single tool invocation with result
2. `test_tool_use_without_tools_defined` - Normal operation without tools
3. `test_tool_result_error_handling` - Error handling
4. `test_tool_use_streaming` - Streaming with tools
5. `test_streaming_without_tools` - Streaming without tools

### OpenAI Function Calling Tests

**File**: `tests/integration/test_openai_function_calling.py`

```bash
# Run all OpenAI function calling tests
pytest tests/integration/test_openai_function_calling.py -v

# Specific test
pytest tests/integration/test_openai_function_calling.py::TestOpenAIFunctionCalling::test_function_calling_single -v
```

**Tests**:
1. `test_function_calling_single` - Single function call
2. `test_function_calling_without_tools` - Normal operation
3. `test_function_calling_parallel` - Multiple parallel calls
4. `test_tool_choice_parameter` - tool_choice validation
5. `test_function_calling_streaming` - Streaming function calls
6. `test_streaming_without_tools` - Streaming without tools

## CI/CD Integration

### GitHub Actions

Example workflow configuration:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest  # Apple Silicon required

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/agent_memory

      - name: Run integration tests (no model)
        run: pytest tests/integration/ -k "not WithModel" -v

      - name: Run model tests (if Metal GPU available)
        run: pytest tests/integration/ -k "WithModel" -v
        continue-on-error: true  # Optional for CI without GPU
```

### Pre-Commit Hooks

Run tests before committing:

```bash
# .git/hooks/pre-commit
#!/bin/bash
set -e

echo "Running unit tests..."
pytest tests/unit/ -q

echo "Running linting..."
ruff check src/ tests/

echo "All checks passed!"
```

### Make Targets

Add to `Makefile`:

```makefile
.PHONY: test test-unit test-integration test-all

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -k "not WithModel" -v

test-model:
	pytest tests/integration/ -k "WithModel" -v

test-all:
	pytest tests/ -v --cov=src/agent_memory --cov-report=term-missing

test-tool-calling:
	pytest tests/integration/test_anthropic_tool_calling.py -v
	pytest tests/integration/test_openai_function_calling.py -v
```

Usage:

```bash
make test-unit
make test-integration
make test-model
make test-all
```

## Coverage Requirements

### Target Coverage

- **Overall**: 85%+
- **Core domain**: 90%+
- **Adapters**: 80%+
- **Critical paths**: 100%

### Measuring Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src/agent_memory --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src/agent_memory --cov-report=html

# Open HTML report
open htmlcov/index.html
```

### Coverage Configuration

`.coveragerc`:

```ini
[run]
source = src/agent_memory
omit =
    */tests/*
    */migrations/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

### Quality Gates

All PRs must pass:

```bash
# Zero linting errors
ruff check src/ tests/
# Exit code: 0

# All unit tests pass
pytest tests/unit/ -v
# Exit code: 0

# All integration tests pass (no model)
pytest tests/integration/ -k "not WithModel" -v
# Exit code: 0

# Coverage above threshold
pytest tests/ --cov=src/agent_memory --cov-fail-under=85
# Exit code: 0
```

## Writing New Tests

### Unit Test Template

```python
"""Unit tests for MyComponent."""

import pytest

from agent_memory.domain.my_component import MyComponent


@pytest.mark.unit
class TestMyComponent:
    """Test MyComponent functionality."""

    def test_basic_operation(self):
        """Test basic operation."""
        component = MyComponent()
        result = component.do_something()
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case handling."""
        component = MyComponent()
        with pytest.raises(ValueError):
            component.do_something_invalid()
```

### Integration Test Template

```python
"""Integration tests for /v1/my-endpoint."""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.mark.integration
class TestMyEndpoint:
    """Test /v1/my-endpoint."""

    def test_successful_request(self):
        """Test successful request."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/my-endpoint",
                json={"param": "value"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "result" in data
```

## Troubleshooting

### Issue: Tests Fail with Model Loading Error

**Solution**: Check MLX installation and model availability

```bash
pip show mlx-lm
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Issue: Slow Test Execution

**Solution**: Run tests without model loading

```bash
pytest tests/ -k "not WithModel" -v
```

### Issue: Coverage Not Measured

**Solution**: Install pytest-cov

```bash
pip install pytest-cov
```

### Issue: Tests Pass Locally but Fail in CI

**Solution**: Check for:
1. Missing dependencies in CI environment
2. Model loading requiring Metal GPU
3. Environment variable differences
4. File path issues

## Best Practices

1. **Write tests first**: TDD approach for new features
2. **Keep tests isolated**: No dependencies between tests
3. **Use fixtures**: Share setup code with pytest fixtures
4. **Mock external dependencies**: Don't rely on network/file I/O
5. **Test edge cases**: Error handling, empty inputs, limits
6. **Maintain coverage**: Add tests for all new code
7. **Fast feedback**: Run unit tests frequently during development

## See Also

- [Developer Guide](developer-guide.md) - Development workflow
- [Architecture Overview](architecture.md) - System architecture
- [Configuration Guide](configuration.md) - Test configuration
