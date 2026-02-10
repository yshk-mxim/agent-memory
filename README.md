# agent-memory

Persistent KV cache for multi-agent LLM systems on Apple Silicon.

```
pip install -e .
python -m agent_memory.entrypoints.cli serve --port 8000
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## What this does

When multiple LLM agents share one local model, every new request re-computes the full KV cache from scratch. On a 12B model with 2K tokens of context, that costs 3-8 seconds of prefill per request.

agent-memory persists KV caches to disk as Q4-quantized safetensors files. When an agent returns with the same conversation prefix, the server reloads the cache and skips directly to decoding. The result: 30-130x faster time-to-first-token on cache hits, depending on context length.

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so existing tools (LangChain, OpenAI SDK, curl) work without changes.

## Performance

Measured on M4 Pro 24 GB with Gemma 3 12B IT (Q4) and DeepSeek-Coder-V2-Lite 16B (Q4):

| Metric | Cold (no cache) | Warm (cache hit) | Speedup |
|--------|-----------------|-------------------|---------|
| TTFT, 512 tokens | 1.8s | 0.014s | 130x |
| TTFT, 2048 tokens | 7.2s | 0.055s | 131x |
| TTFT, 4096 tokens | 15.1s | 0.48s | 31x |

Concurrent batch=2 inference with interleaved chunked prefill. Cache adds <3% perplexity impact (Q4 quantization).

See `benchmarks/` for full methodology and reproducible scripts.

## How it works

```
                  ┌─────────────────────────┐
                  │   OpenAI-compatible API  │
                  │   /v1/chat/completions   │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │  Coordination Service    │
                  │  (multi-agent routing)   │
                  └───────────┬─────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    Block Pool Batch Engine     │
              │  ┌──────────────────────────┐  │
              │  │ Concurrent Scheduler     │  │
              │  │ (interleaved prefill +   │  │
              │  │  batch decode)           │  │
              │  └──────────────────────────┘  │
              └───────────────┬───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼─────┐        ┌─────▼──────┐       ┌──────▼──────┐
   │ Hot Cache │        │ Warm Cache │       │ Cold Cache  │
   │ (memory)  │◄──────►│ (metadata) │◄─────►│ (disk .st)  │
   └──────────┘        └────────────┘       └─────────────┘
```

Three-tier cache: hot caches live in GPU memory, warm caches keep metadata for fast reloading, cold caches persist as safetensors files on disk. The block pool manages memory budgets and evicts least-recently-used caches when space runs low.

The server supports batch=2 concurrent inference with a scheduler that interleaves chunked prefill and token-by-token decode, hiding prefill latency for staggered agent arrivals.

## Supported models

| Model | Size | Architecture notes |
|-------|------|--------------------|
| Gemma 3 12B IT | Q4, 6.5 GB | Hybrid attention: 8 global + 40 sliding window layers |
| DeepSeek-Coder-V2-Lite | Q4, 8 GB | MLA with asymmetric K=192/V=128 dims |

Adding a new model requires a TOML config in `config/models/` and verifying the spec extractor detects its attention architecture. See `docs/developer-guide.md`.

**Note on model access**: Gemma 3 12B is a **gated model** — you must accept Google's license at [the model page](https://huggingface.co/google/gemma-3-12b-it) and run `huggingface-cli login` before first use. DeepSeek community models are not gated (no token needed). Run `scripts/setup.sh` for guided setup.

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_MLX_MODEL_ID` | Gemma 3 12B IT Q4 | HuggingFace model ID |
| `SEMANTIC_MLX_CACHE_BUDGET_MB` | 8192 | Max GPU memory for KV caches (MB) |
| `SEMANTIC_MLX_MAX_BATCH_SIZE` | 2 | Max concurrent sequences |
| `SEMANTIC_MLX_SCHEDULER_ENABLED` | true | Enable concurrent scheduler |
| `SEMANTIC_ADMIN_KEY` | (none) | Admin API authentication key |

```bash
# Gemma 3 12B (default, no env vars needed)
python -m agent_memory.entrypoints.cli serve --port 8000

# DeepSeek-Coder-V2-Lite
SEMANTIC_MLX_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" \
SEMANTIC_MLX_CACHE_BUDGET_MB=4096 \
python -m agent_memory.entrypoints.cli serve --port 8000
```

## Quick start (server + demo)

One script launches the inference server with Gemma 3 12B (recommended settings) and opens the Streamlit demo:

```bash
scripts/launch.sh                # server + demo UI
scripts/launch.sh --server-only  # server only, no Streamlit
scripts/launch.sh --stop         # stop running server + demo
```

This starts the server on port 8000 (scheduler on, batch=2, 8 GB cache budget, T=0.3) and the demo UI on port 8501. Press Ctrl+C to stop both with graceful shutdown (persists KV caches, releases GPU memory). If instances are already running, the script detects them and offers to shut them down first. Override ports with `SEMANTIC_SERVER_PORT` and `STREAMLIT_PORT`.

For first-time setup (venv, dependencies, HuggingFace login, model download):

```bash
scripts/setup.sh    # guided setup
scripts/launch.sh   # start everything
```

## Demos

Streamlit demos in `demo/` (launched via `scripts/launch.sh` or manually):

1. **Multi-Agent Chat** -- Per-agent KV cache persistence with real-time performance metrics (home page)
2. **Coordination** -- Dynamic multi-agent session creation with configurable agents
3. **Agent Memory Inspector** -- View and manage cached agents across hot/warm/cold tiers
4. **Gossip** -- Three friends, two private conversations, one awkward reunion (YAML-driven)
5. **Prisoner's Dilemma** -- Two suspects, one warden, iterated game theory with persistent memory

To run the demo manually (two terminals):

```bash
# Terminal 1: server
python -m agent_memory.entrypoints.cli serve --port 8000

# Terminal 2: demo UI
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+ (Ventura)
- Python 3.11+
- 16 GB RAM minimum (24 GB recommended for 12B models)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/unit -x -q --timeout=30  # ~1,100 tests, ~3s
```

Architecture: hexagonal (ports and adapters) with domain-driven design. Source in `src/agent_memory/`, tests in `tests/`, benchmarks in `benchmarks/`.

See `CLAUDE.md` for the full developer guide, including graceful shutdown procedures and MLX threading notes.

## Paper

The technical details, benchmark methodology, and cache architecture are described in:

> **Agent Memory Below the Prompt: Persistent KV Cache for Multi-Agent LLM Systems on Apple Silicon**
>
> [arXiv preprint]

## Known Limitations

- **Apple Silicon only** — requires MLX framework (M1/M2/M3/M4)
- **`stop` / `stop_sequences`** — accepted but not enforced during generation
- **`tool_choice`** — accepted but not constrained; tool calls detected via post-hoc parsing
- **Batch decode** — new requests wait until all active decodes complete
- **`mx.async_eval`** — replaced process-wide with `mx.eval` to prevent Metal crashes
- **Cache persistence** — disk save is synchronous on the event loop; ~50-100ms stall once per completed request (not per token)

## License

MIT. See [LICENSE](LICENSE).
