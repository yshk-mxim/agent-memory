# Claude Code Setup with agent-memory

How to configure Claude Code CLI to use agent-memory as a local LLM backend.

## Prerequisites

- agent-memory running on your server (see [deployment.md](deployment.md))
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- A model loaded (see [current_models.md](current_models.md))

## Project settings

Create `.claude/settings.json` in your project directory:

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://<your-server>:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "gemma-4-26b-a4b",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "100000"
    }
}
```

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_BASE_URL` | Points Claude Code to agent-memory instead of Anthropic API |
| `ANTHROPIC_AUTH_TOKEN` | Any non-empty string (agent-memory accepts `local`) |
| `ANTHROPIC_MODEL` | Model ID matching a `config/models/*.toml` name |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | Disable attribution header (not needed locally) |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Prevent telemetry/update checks to Anthropic |
| `MAX_THINKING_TOKENS` | Set to `0` — local models don't use extended thinking |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Must be ≤ server's per-slot context size |

## Recommended CLAUDE.md

Create a `CLAUDE.md` in your project root with guidance tailored to local model behavior.

### Web search (if SearXNG proxy enabled)

```markdown
# Web Search

DO NOT use the WebSearch tool — it requires an API key and returns 0 results.

Use the `/search` proxy on agent-memory (SearXNG backend, returns JSON).
Replace spaces with `+` in the query:

    Bash: curl -s "http://<your-server>:8000/search?q=your+query+here"

Response has `results` array with `title`, `url`, `content` per result.
```

### Web fetch (if Jina Reader proxy enabled)

```markdown
# Fetching Web Pages

DO NOT use WebFetch directly for public URLs — it returns raw HTML (slow, token-heavy).

Use the `/fetch` proxy on agent-memory (Jina Reader backend, returns clean markdown):

    Bash: curl -s "http://<your-server>:8000/fetch?url=https://example.com/page"

Note: the reader has no JavaScript engine — JS-rendered pages return nav chrome
instead of content. For GitHub files, use raw URLs.
```

### Math rendering

```markdown
# Math & Formula Rendering

When writing math in conversational output (displayed in terminal),
use plain-text / ASCII notation — LaTeX does not render in the CLI.

Example: write `0.9 / (1 - 0.1) = 1` not `$$\frac{0.9}{1-0.1} = 1$$`.

When writing to files (.md, .tex, .html), use proper LaTeX math.
```

### Model info and swap instructions

```markdown
# Local LLM Backend

This workspace uses agent-memory as a local LLM proxy.
Models are swapped manually (auto-swap is disabled to prevent ping-pong
from Claude Code's parallel title-generation requests).

Swap via admin API:

    curl -X POST http://<your-server>:8000/admin/models/swap \
      -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
      -H "Content-Type: application/json" \
      -d '{"model_id": "gemma-4-31b"}'
```

### Execution strategy (important for local models)

```markdown
# Execution Strategy

Use agents and tool calling for reliable execution — don't try to hold everything
in your head. Break complex tasks into focused sub-tasks using the Agent tool:

- Reduce context per task: Each agent gets a fresh context window. Delegate
  research, file exploration, and independent sub-problems to agents.
- Return findings to main context: Summarize agent findings in your response
  so the main conversation has information for decisions.
- Rely on tool calls: Use Read, Grep, Glob, and Bash to verify state rather
  than assuming. Grounding every step in tool output prevents drift.
- Parallelize independent work: Launch multiple agents concurrently when
  tasks don't depend on each other.
- Keep main thread for synthesis: Coordinate agents, make decisions, and
  talk to the user — not raw exploration.
```

## Context and compaction

Claude Code automatically compacts conversations when approaching the context limit.
Key settings:

| Variable | Default | Notes |
|----------|---------|-------|
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | 200000 | Set to your server's per-slot ctx size (with margin) |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | ~95 | Lower this (e.g. 65) to trigger compaction earlier |

For a server with 131K context per slot, `CLAUDE_CODE_MAX_CONTEXT_TOKENS=100000`
is a safe value — it gives ~30K tokens of buffer.

## Why auto-swap is disabled

Claude Code sends **two parallel requests** per turn:
1. Title generation (mapped to the "haiku" tier)
2. Main response (mapped to the selected model)

If model aliases map different tiers to different models, auto-swap causes
destructive ping-pong — the server flips between models on every turn.
Use manual swap via the admin API or `scripts/thor/swap_model.sh` instead.

## Troubleshooting

**Tool calls return "Invalid tool parameters"**
- Ensure `jinja2` is installed in agent-memory's Python environment
- Verify the model's TOML config has `chat_template_file` pointing to a valid Jinja2 template
- Check `config/chat_templates/` for model-specific templates

**Model not responding / connection refused**
- Verify agent-memory is running: `curl http://<server>:8000/health`
- Check if llama-server started: look for the process or check agent-memory logs

**Context too long errors**
- Lower `CLAUDE_CODE_MAX_CONTEXT_TOKENS` in settings
- Switch to a model with larger context (e.g., MoE with 262K vs dense with 131K)
