# Contributing to agent-memory

Thank you for your interest in contributing to agent-memory! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [License](#license)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment. We expect all contributors to:

- Be respectful and constructive in feedback
- Welcome newcomers and help them get started
- Focus on what is best for the project and community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or 3.12
- **Apple Silicon**: M1/M2/M3/M4 chip (MLX requires Apple Silicon)
- **macOS**: 13.0 or later (for MLX support)
- **Git**: For version control
- **pip**: Python package installer

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yshk-mxim/agent-memory.git
   cd agent-memory
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Run tests to verify setup**:
   ```bash
   pytest tests/unit/ -v
   ```

---

## Development Setup

### Install Development Dependencies

```bash
# Install with all dev dependencies
pip install -e .[dev]

# This includes:
# - pytest (testing framework)
# - ruff (linting and formatting)
# - mypy (type checking)
# - pre-commit (git hooks)
# - All other dev tools
```

### Set Up Pre-Commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files (optional)
pre-commit run --all-files
```

---

## Project Structure

```
agent-memory/
├── src/agent_memory/            # Source code
│   ├── domain/              # Domain models (pure business logic)
│   ├── application/         # Application services (use cases)
│   ├── adapters/            # Infrastructure adapters
│   │   ├── inbound/         # FastAPI endpoints, middleware
│   │   ├── outbound/        # MLX, safetensors, external services
│   │   └── config/          # Settings, logging configuration
│   ├── ports/               # Protocol definitions (interfaces)
│   └── entrypoints/         # CLI, API server entry points
├── tests/                   # Test suites
│   ├── unit/                # Unit tests (fast, mocked)
│   ├── integration/         # Integration tests (real MLX, I/O)
│   ├── e2e/                 # End-to-end tests
│   ├── smoke/               # Basic server lifecycle tests
│   └── stress/              # Load and performance tests
├── config/                  # Configuration files
│   ├── prometheus/          # Alert rules
│   └── logging/             # Log retention policies
├── docs/                    # Documentation
├── project/                 # Project management
│   ├── architecture/        # Architecture Decision Records (ADRs)
│   └── sprints/             # Sprint planning and reports
├── pyproject.toml           # Package metadata and dependencies
├── LICENSE                  # MIT License
├── NOTICE                   # Dependency attributions
└── CHANGELOG.md             # Release notes
```

### Architecture

agent-memory follows **Hexagonal Architecture** (Ports & Adapters):

- **Domain Layer**: Pure business logic (no external dependencies)
- **Application Layer**: Use cases and orchestration
- **Adapters Layer**: Infrastructure bindings (FastAPI, MLX, disk I/O)
- **Ports Layer**: Protocol definitions for dependency inversion

See `project/architecture/ADR-001-hexagonal-architecture.md` for details.

---

## Development Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/issue-123
```

### 2. Make Changes

- Write code following the project's coding standards
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run unit tests (fast)
pytest tests/unit/ -v

# Run integration tests (requires MLX)
pytest tests/integration/ -v -m integration

# Run all tests
pytest tests/ -v
```

### 4. Check Code Quality

```bash
# Run ruff linter
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Run type checker
mypy --strict src/

# Run all checks
pre-commit run --all-files
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add XYZ feature

- Implemented ABC
- Added tests for DEF
- Updated documentation"
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Build/tooling changes

### 6. Push and Create PR

```bash
# Push branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# Follow the PR template
```

---

## Code Quality Standards

### Linting and Formatting

The project uses **Ruff** for linting and formatting:

```bash
# Check code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Code Quality Rules

- **Line length**: 100 characters max
- **Complexity**: McCabe complexity ≤ 10
- **Imports**: Sorted with isort
- **Docstrings**: Google-style docstrings for public APIs
- **Type hints**: All functions must have type annotations

### Type Checking

Use **mypy** for static type checking:

```bash
# Run type checker
mypy --strict src/

# Check specific file
mypy --strict src/agent_memory/application/batch_engine.py
```

---

## Testing

### Test Categories

The project has multiple test suites with different markers:

- **unit**: Fast unit tests with mocked boundaries
- **integration**: Tests with real MLX and disk I/O (Apple Silicon only)
- **smoke**: Basic server lifecycle tests
- **e2e**: Full-stack multi-agent tests (slow, Apple Silicon only)
- **stress**: Load and concurrency stress tests (very slow, Apple Silicon only)

### Running Tests

```bash
# Run unit tests only (fast)
pytest tests/unit/ -v

# Run integration tests (requires MLX)
pytest tests/integration/ -v -m integration

# Run smoke tests
pytest tests/smoke/ -v -m smoke

# Run all tests (slow)
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_block_pool.py -v

# Run specific test
pytest tests/unit/test_block_pool.py::test_allocate_blocks -v
```

### Coverage

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Writing Tests

**Test Structure**:
```python
"""Tests for XYZ module."""

import pytest


@pytest.mark.unit
def test_feature_description():
    """Test that feature works as expected."""
    # Arrange
    input_data = create_test_input()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value
```

---

## Pull Request Process

### Before Submitting

1. **Run all checks**:
   ```bash
   # Pre-commit checks
   pre-commit run --all-files

   # All tests
   pytest tests/ -v

   # Type checking
   mypy --strict src/
   ```

2. **Update documentation**:
   - Add/update docstrings for new functions
   - Update README.md if adding new features
   - Add ADRs for significant architectural decisions

3. **Add changelog entry**:
   - Update CHANGELOG.md under "Unreleased" section

### Code Review

Your PR will be reviewed for:

- **Correctness**: Does it work as intended?
- **Testing**: Are tests comprehensive?
- **Code Quality**: Follows style guidelines?
- **Documentation**: Is it well-documented?
- **Architecture**: Fits with project design?

---

## Development Tips

### Running the Server

```bash
# Start server (development mode)
python -m agent_memory.entrypoints.cli serve --log-level DEBUG

# Start with custom settings
python -m agent_memory.entrypoints.cli serve --host 0.0.0.0 --port 8080

# View configuration
python -m agent_memory.entrypoints.cli config
```

---

## Getting Help

- **Issues**: https://github.com/yshk-mxim/agent-memory/issues
- **Documentation**: `docs/` directory

---

## License

By contributing to agent-memory, you agree that your contributions will be licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

**Thank you for contributing! 🎉**
