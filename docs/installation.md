# Installation Guide

Complete installation instructions for Semantic.

## System Requirements

### Hardware

- **Apple Silicon Required** — M1, M2, M3, or M4 chip
- **Minimum RAM**: 16GB (24GB recommended for 12B models)
- **Disk Space**: ~20GB (10GB for models + caches)

**Note**: MLX framework requires Apple Silicon. This project will not run on Intel Macs or other platforms.

### Software

- **macOS**: 14+ (Sonoma or later)
- **Python**: 3.11 or 3.12
- **Git**: For cloning repository

## Installation Methods

### Method 1: Development Install (Recommended)

For contributors and active development:

```bash
# Clone repository
git clone https://github.com/yshk-mxim/rdic.git
cd rdic

# Create virtual environment (recommended)
python3.12 -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**Includes**:
- Core dependencies (mlx, mlx-lm, safetensors)
- Development tools (pytest, hypothesis, ruff, mypy)
- Pre-commit hooks (automated quality checks)

### Method 2: User Install

For running the server without development tools:

```bash
# Clone repository
git clone https://github.com/yshk-mxim/rdic.git
cd rdic

# Install core dependencies only
pip install -e .
```

### Method 3: From Source

For building from source:

```bash
git clone https://github.com/yshk-mxim/rdic.git
cd rdic
python -m build
pip install dist/semantic-*.whl
```

## Verify Installation

### Check Python Version

```bash
python --version
# Should be Python 3.11.x or 3.12.x
```

### Check MLX Installation

```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
# Should print MLX version without errors
```

### Check Semantic Installation

```bash
python -c "from semantic.domain.services import BlockPool; print('✓ Semantic installed')"
# Should print success message
```

### Run Unit Tests

```bash
make test-unit
# Should see: 112 tests passed, 95.07% coverage
```

## Configuration

### Default Configuration

Semantic uses sensible defaults. No configuration needed for basic usage.

### Custom Configuration (Advanced)

Create `~/.semantic/config.toml`:

```toml
[model]
model_id = "mlx-community/gemma-3-12b-it-4bit"
max_batch_size = 5
block_tokens = 256

[cache]
cache_dir = "~/.semantic/caches"
max_agents_in_memory = 5
batch_window_ms = 10

[server]
host = "0.0.0.0"
port = 8000
```

### Environment Variables

```bash
export SEMANTIC_CACHE_DIR="~/.semantic/caches"
export SEMANTIC_MAX_BATCH_SIZE=5
```

Environment variables override config file settings.

## Dependencies

### Core Dependencies

Installed automatically with `pip install -e .`:

| Package | Version | Purpose |
|---------|---------|---------|
| mlx | >=0.30.0 | Apple ML framework |
| mlx-lm | >=0.30.0 | Language model utilities |
| safetensors | >=0.7.0 | Tensor serialization |
| pydantic | >=2.0.0 | Data validation |
| fastapi | >=0.115.0 | API framework (future) |

### Development Dependencies

Installed with `pip install -e ".[dev]"`:

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=8.3.0 | Test runner |
| pytest-asyncio | >=0.25.0 | Async test support |
| pytest-cov | >=6.0.0 | Coverage reporting |
| hypothesis | >=6.122.0 | Property-based testing |
| ruff | >=0.8.0 | Linting + formatting |
| mypy | >=1.14.0 | Type checking |
| pre-commit | >=4.0.0 | Git hooks |

## Platform-Specific Notes

### macOS Sonoma (14.x) or Later

Fully supported. No special configuration needed.

### macOS Ventura (13.x)

May work but untested. MLX requires macOS 14+ officially.

### Non-Apple Silicon

**Not supported**. MLX framework requires Apple Silicon (Metal GPU).

**Alternatives for Intel/Linux**:
- Use vLLM (supports CUDA/ROCm)
- Use llama.cpp (CPU inference)
- Use Ollama (CPU/CUDA)

## Troubleshooting

### ImportError: No module named 'mlx'

**Solution**: MLX requires Apple Silicon. Verify hardware:
```bash
uname -m
# Should output: arm64
```

### ModuleNotFoundError: No module named 'semantic'

**Solution**: Install in editable mode:
```bash
pip install -e .
```

### Tests fail with "hypothesis not found"

**Solution**: Install dev dependencies:
```bash
pip install -e ".[dev]"
```

### Pre-commit hooks fail

**Solution**: Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

### Permission denied on ~/.semantic/

**Solution**: Create directory with correct permissions:
```bash
mkdir -p ~/.semantic/caches
chmod 755 ~/.semantic
```

## Uninstallation

```bash
# Remove package
pip uninstall semantic

# Remove cache directory (optional)
rm -rf ~/.semantic
```

## Next Steps

- **Quick Start**: [5-minute guide](quick-start.md)
- **User Guide**: [Usage and configuration](user-guide.md)
- **Developer Guide**: [Contributing](developer-guide.md)

## Getting Help

- **Documentation**: Browse this site
- **Issues**: [GitHub Issues](https://github.com/yshk-mxim/rdic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yshk-mxim/rdic/discussions)
