# Semantic Caching API

> Production-ready multi-agent LLM inference server with persistent KV cache for Apple Silicon

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)](https://ml-explore.github.io/mlx/)

**Version**: 1.0.0
**Architecture**: Hexagonal (Ports & Adapters) with Domain-Driven Design
**Status**: Production Ready

## What is Semantic Caching API?

A high-performance HTTP server for running MLX language models with **persistent KV cache** across sessions. Enables true multi-agent workflows with intelligent cache management and multiple API protocols.

### Key Features

- **Persistent KV Cache**: Conversations resume instantly with cached context intact
- **Multi-Agent Support**: Manage multiple independent conversation caches simultaneously
- **Tool Calling**: Anthropic tool_use and OpenAI function calling support
- **3 API Formats**: Anthropic Messages, OpenAI Chat Completions, and Direct Agent APIs
- **SSE Streaming**: Real-time token streaming for both API formats
- **Block Pool Memory**: Budget-limited cache allocation with LRU eviction
- **Model Hot-Swap**: Zero-downtime model switching via admin API
- **Prometheus Metrics**: Production-grade observability

### Supported Models

- **DeepSeek-Coder-V2-Lite** (16B 4-bit) - Default production model (163K context)
- **SmolLM2** (135M) - Lightweight testing model
- Extensible to any MLX-compatible model

## Quick Start

### Prerequisites

- **Hardware**: Apple Silicon (M1/M2/M3 or later)
- **OS**: macOS 13.0+ (Ventura or later)
- **RAM**: 16GB minimum (8GB for SmolLM2)
- **Python**: 3.10, 3.11, or 3.12

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .

# Verify
semantic version
```

### Start Server

```bash
# Start with default DeepSeek-Coder-V2-Lite model
semantic serve

# Or specify options
semantic serve --port 8080 --log-level DEBUG
```

Server starts on `http://0.0.0.0:8000`

### Make First Request

#### Anthropic Messages API

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What is semantic caching?"}
    ]
  }'
```

#### OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "messages": [
      {"role": "user", "content": "What is semantic caching?"}
    ],
    "max_tokens": 200
  }'
```

### With Tool Calling

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

## Why Semantic Caching API?

### The Problem

Local LLM tools (LM Studio, Ollama) don't persist KV cache across sessions:

- **LM Studio**: Saves conversation text only
- **Ollama**: No native cache persistence
- **llama.cpp**: Has API support but not exposed in tools

**Result**: Every new session re-computes expensive system prompts, wasting time and energy.

### The Solution

Semantic Caching API exploits Apple Silicon's unified memory for intelligent cache management:

1. **Extract** KV cache blocks during generation
2. **Persist** to `~/.semantic/caches/` with model validation
3. **Manage** hot (memory) and warm (disk) cache tiers
4. **Resume** sessions instantly by loading cached context

**Performance**: 40-60% faster session resume by avoiding re-prefill.

## Architecture

```
┌─────────────────────────────────────────────┐
│         HTTP API (FastAPI)                  │
│  Anthropic / OpenAI / Direct Agent         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Application Layer                      │
│  - AgentCacheStore (LRU eviction)          │
│  - BlockPoolBatchEngine (inference)        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Domain Layer                           │
│  - BlockPool (memory management)           │
│  - ModelCacheSpec (architecture)           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      MLX Framework (Apple Silicon)         │
└─────────────────────────────────────────────┘
```

**Hexagonal Architecture** ensures clean separation between business logic and infrastructure.

## Documentation

Comprehensive documentation available in `docs/`:

- **[Quick Start](docs/quick-start.md)** - Get started in 5 minutes
- **[User Guide](docs/user-guide.md)** - Complete API usage guide
- **[Configuration](docs/configuration.md)** - Environment variables and settings
- **[Model Onboarding](docs/model-onboarding.md)** - Adding new models
- **[Testing](docs/testing.md)** - Running and writing tests
- **[Deployment](docs/deployment.md)** - Production deployment guide
- **[Architecture](docs/architecture/)** - System design documentation
  - [Domain Layer](docs/architecture/domain.md)
  - [Application Layer](docs/architecture/application.md)
  - [Adapters Layer](docs/architecture/adapters.md)
- **[API Reference](docs/api-reference.md)** - Complete API specification
- **[FAQ](docs/faq.md)** - Frequently asked questions

## Configuration

Create a `.env` file:

```bash
# MLX Model Configuration
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5

# Agent Cache Configuration
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=5
SEMANTIC_AGENT_CACHE_DIR=~/.semantic/caches

# Server Configuration
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_LOG_LEVEL=INFO

# Security
SEMANTIC_API_KEY=your-api-key-here
```

See [Configuration Guide](docs/configuration.md) for complete reference.

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests (no model loading)
pytest tests/integration/ -k "not WithModel" -v

# Run all tests including model tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/semantic --cov-report=term-missing
```

**Test Coverage**: 370+ tests, 85%+ coverage

## Use Cases

### Multi-Agent Workflows

```bash
# Agent 1
curl -X POST http://localhost:8000/v1/messages \
  -H "X-Session-ID: agent-alice" \
  -d '{"model": "deepseek-coder-v2-lite", "messages": [...]}'

# Agent 2 (separate cache)
curl -X POST http://localhost:8000/v1/messages \
  -H "X-Session-ID: agent-bob" \
  -d '{"model": "deepseek-coder-v2-lite", "messages": [...]}'
```

### Tool-Enabled Assistants

Build agents that can invoke functions:

- Weather lookups
- Database queries
- API calls
- Code execution
- Custom tools

See [User Guide](docs/user-guide.md#tool-calling) for examples.

### Long-Running Conversations

Cache persists across server restarts:

1. Have conversation with agent
2. Stop server
3. Restart server
4. Continue conversation - context preserved!

## Performance

**DeepSeek-Coder-V2-Lite (M3 Max, 64GB RAM)**:
- Latency: ~50-100ms per token
- Throughput: 50-100 tokens/second
- Memory: ~20GB (model + cache)

**SmolLM2 (M1, 16GB RAM)**:
- Latency: ~20-40ms per token
- Throughput: 25-30 tokens/second
- Memory: ~2GB (model + cache)

## Development

### Project Structure

```
semantic/
├── src/semantic/
│   ├── domain/              # Core business logic
│   ├── application/         # Use case orchestration
│   ├── adapters/           # External interfaces
│   └── entrypoints/        # FastAPI app
├── tests/
│   ├── unit/               # Isolated component tests
│   └── integration/        # End-to-end API tests
├── docs/                   # Comprehensive documentation
└── project/               # Sprint plans and ADRs
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy --strict src/

# All quality checks
make lint test
```

**Standards**:
- Zero ruff errors
- Full type coverage with mypy --strict
- 85%+ test coverage
- Hexagonal architecture compliance

## Comparison

| Feature | LM Studio | Ollama | llama.cpp | Semantic Caching |
|---------|-----------|--------|-----------|------------------|
| **KV Cache Persistence** | ❌ | ❌ | ⚠️ API only | ✅ Full |
| **Multi-Agent Native** | ❌ | ❌ | ❌ | ✅ Yes |
| **Tool Calling** | ⚠️ Partial | ⚠️ Partial | ❌ | ✅ Anthropic + OpenAI |
| **Streaming** | ✅ | ✅ | ✅ | ✅ SSE |
| **Apple Silicon** | ✅ | ✅ | ✅ | ✅ MLX Native |
| **Block Pool Memory** | ❌ | ❌ | ❌ | ✅ Yes |
| **HTTP API** | ✅ | ✅ | ✅ | ✅ 3 formats |

## Limitations

- **Platform**: Apple Silicon only (MLX requirement)
- **Docker**: Not supported (Metal GPU passthrough limitation)
- **Multi-User**: Single-user deployment (can run multiple instances)

## Roadmap

### Post v1.0.0

- Additional models (Llama 3, Mistral, DeepSeek)
- Extended Prometheus metrics catalog
- OpenTelemetry tracing
- Performance optimizations
- Multi-modal support exploration

## Contributing

Contributions welcome! Please:

1. Read [Architecture Documentation](docs/architecture/)
2. Follow code quality standards (ruff, mypy)
3. Add tests for new features
4. Update documentation

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

- **MLX**: Apple's ML framework for Apple Silicon
- **MLX-LM**: Language model utilities for MLX
- **FastAPI**: Modern async web framework
- **Anthropic**: Tool use protocol inspiration
- **OpenAI**: Chat Completions API compatibility

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs via GitHub Issues
- **Questions**: See [FAQ](docs/faq.md)

---

**Built with ❤️ for Apple Silicon**
Version 1.0.0 | January 2026
