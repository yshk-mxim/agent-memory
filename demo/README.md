# Multi-Agent Demo

Interactive demo showing 4 concurrent agents, each with independent KV cache sessions.

## Setup

```bash
pip install -r demo/requirements.txt
```

## Run

```bash
# Terminal 1: Start the semantic server
semantic serve

# Terminal 2: Launch the demo UI
streamlit run demo/app.py
```

Open http://localhost:8501 in your browser.

## Features

- **4 independent agent conversations** with per-agent cache persistence
- **Cache state indicators**: COLD (first message) -> WARM (cache hit) -> HOT (deep conversation)
- **Real-time metrics**: TTFT, tokens/sec, input/output token counts
- **Server monitoring**: GPU memory, pool utilization, active agents
- **Suggested prompts** for quick demonstration
