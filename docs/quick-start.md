# Quick Start

Get Semantic running in 5 minutes.

## Prerequisites

- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- 16GB RAM minimum

## Installation

```bash
# Clone repository
git clone https://github.com/yshk-mxim/rdic.git
cd rdic

# Install in development mode
pip install -e .
```

## Verify Installation

```bash
# Check that all imports work
python -c "from semantic.domain.services import BlockPool; print('✓ Semantic installed successfully')"
```

## Run Tests

```bash
# Run unit tests
make test-unit

# Expected output:
# 112 tests passed, 95.07% coverage
```

## Next Steps

- **Configuration**: See [Installation Guide](installation.md) for detailed setup
- **Usage**: See [User Guide](user-guide.md) for multi-agent workflows
- **Development**: See [Developer Guide](developer-guide.md) for contributing

## Project Structure

```
semantic/
├── src/semantic/
│   ├── domain/          # Core business logic (zero dependencies)
│   │   ├── entities.py      # KVBlock, AgentBlocks
│   │   ├── value_objects.py # ModelCacheSpec, CacheKey
│   │   └── services.py      # BlockPool
│   ├── ports/           # Interface definitions
│   ├── application/     # Orchestration services
│   └── adapters/        # External integrations
├── tests/
│   ├── unit/            # Pure domain tests (mocked boundaries)
│   ├── integration/     # Real MLX + disk tests
│   └── smoke/           # Server lifecycle tests
└── docs/                # Documentation (you are here!)
```

## Quick Concepts

### Block-Pool Memory

- Fixed **256-token blocks** for all models
- O(1) allocation via LIFO free list
- Efficient memory reuse across agents

### Persistent KV Cache

- Agents save their cache to disk using **safetensors**
- Resume conversations instantly without re-computing context
- **40-60% faster** on session resume

### Hexagonal Architecture

- **Domain core** has zero external dependencies
- All infrastructure concerns in **adapters**
- Easy to test, swap implementations, evolve independently

## Troubleshooting

**Import errors**: Ensure you installed with `pip install -e .`

**Test failures**: Check Python version is 3.11+ with `python --version`

**Coverage too low**: This is expected — integration tests require Apple Silicon

## Help

For issues, see:

- [Installation Guide](installation.md) — Detailed setup
- [Developer Guide](developer-guide.md) — Contributing guidelines
- [GitHub Issues](https://github.com/yshk-mxim/rdic/issues) — Report bugs
