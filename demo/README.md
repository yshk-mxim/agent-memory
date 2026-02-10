# Demos

Streamlit multi-page app with two demos showing agent-memory in action.

## Setup

```bash
pip install streamlit
```

## Running

```bash
# Terminal 1: Start the server
python -m agent_memory.entrypoints.cli serve --port 8000

# Terminal 2: Launch the demo UI
streamlit run demo/app.py
```

Open http://localhost:8501 in your browser.

## Pages

### Prisoner's Dilemma

Two LLM agents play iterated prisoner's dilemma. Each agent maintains persistent memory across rounds through the KV cache -- they remember previous moves and adapt their strategy over time.

Demonstrates:
- Independent per-agent cache isolation
- Multi-turn conversation with persistent context
- Cache state transitions (cold -> warm -> hot)

### Agent Memory (Wikipedia routing)

Multiple agents route Wikipedia article queries through the server. Shows cache persistence across server restarts: start agents, kill the server, restart it, and watch agents resume with cached context intact.

Demonstrates:
- Cache persistence to disk (safetensors files)
- Warm cache reload on server restart
- TTFT speedup on cache hits vs cold starts

## Additional pages

- **Coordination** -- Multi-agent orchestration with agent identity routing
- **Gossip Demo** -- Agents sharing information through structured dialogue
