# llama.cpp + Thor — Full Stack Example

> Copy-paste reference for the complete stack:
> llama-server → agent-memory → Claude Code CLI (with tools).

## Architecture

```
Mac terminal                  Thor (main4.local)
────────────                  ──────────────────
Claude Code CLI  ──HTTP──►  agent-memory :8000   ──HTTP──►  llama-server :8001  ──►  GPU
                             (session mgmt,                   (GGUF inference,
                              cache persistence,               slot KV cache,
                              Anthropic API,                   Qwen3-Coder-Next
                              tool translation)                44.5 tok/s)
```

## 1. Start llama-server (on Thor)

```bash
ssh yshkolni@main4.local

mkdir -p ~/.agent_memory/llamacpp_slots

/tmp/llama-cpp-build/build/bin/llama-server \
    -m ~/models/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_M.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    -ngl 999 \
    --ctx-size 131072 \
    -np 4 \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --cache-prompt \
    -b 4096 \
    -ub 1024
```

> **Context sizing:** llama-server divides `--ctx-size` equally among `-np` slots.
> `131072 ÷ 4 = 32768` tokens per slot. Claude Code's system prompt is ~23K tokens,
> so each slot needs ≥32K to fit it plus conversation history.
> Using fewer slots (e.g. `-np 2 --ctx-size 65536`) gives the same per-slot budget
> with lower total memory usage.

Wait for: `llama server listening at http://0.0.0.0:8001`

Verify:
```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

**For Qwen3.5-27B-Opus-Distilled instead:**
```bash
/tmp/llama-cpp-build/build/bin/llama-server \
    -m ~/models/qwen35-opus-distilled/Qwen3.5-27B-Claude-4.6-Opus-Distilled-Q4_K_M.gguf \
    --port 8001 --host 0.0.0.0 \
    -ngl 999 --ctx-size 65536 -np 4 \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q4_0 --cache-type-v q4_0 --cache-prompt \
    -b 4096 -ub 1024
```

## 2. Start agent-memory (on Thor)

```bash
cd ~/agent-memory
source ~/vllm-env/bin/activate

SEMANTIC_BACKEND=llamacpp \
SEMANTIC_LLAMACPP_BASE_URL=http://localhost:8001 \
SEMANTIC_LLAMACPP_MODEL_ID=qwen3-coder-next \
SEMANTIC_LLAMACPP_TOKENIZER_ID=Qwen/Qwen2.5-Coder-32B-Instruct \
SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH=262144 \
SEMANTIC_LLAMACPP_N_SLOTS=2 \
SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

> **`TOKENIZER_ID` note:** `unsloth/*-GGUF` repos contain only model weights, no
> HuggingFace tokenizer files. Set `TOKENIZER_ID` to the base model (e.g.
> `Qwen/Qwen2.5-Coder-32B-Instruct`) to load a compatible tokenizer.
> `MODEL_ID` can be any short name — it's only what agent-memory sends in API
> responses, not a HuggingFace path.

> **`SEARXNG_URL` note:** Enables the `/search?q=...` proxy endpoint on port 8000,
> routing to SearXNG's JSON API. Omit if SearXNG is not running. See
> [`searxng_thor_setup.md`](searxng_thor_setup.md) for setup instructions.

For Qwen3.5-27B-Opus-Distilled:
```bash
SEMANTIC_BACKEND=llamacpp \
SEMANTIC_LLAMACPP_BASE_URL=http://localhost:8001 \
SEMANTIC_LLAMACPP_MODEL_ID=qwen35-opus-distilled \
SEMANTIC_LLAMACPP_TOKENIZER_ID=Qwen/Qwen2.5-72B-Instruct \
SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH=65536 \
SEMANTIC_LLAMACPP_N_SLOTS=4 \
SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

Verify from Thor (or Mac):
```bash
curl http://main4.local:8000/health/live
# {"status":"alive"}
```

## 3. Test the stack (from Mac)

```bash
# Plain generation
curl -s http://main4.local:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    }' | python3 -m json.tool

# With system prompt
curl -s http://main4.local:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 128,
        "system": "You are a coding assistant. Be concise.",
        "messages": [{"role": "user", "content": "Write a Python hello world."}]
    }' | python3 -m json.tool

# With tool definitions (Claude Code sends these on every turn)
curl -s http://main4.local:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 256,
        "system": "You are a coding assistant. Use tools to help the user.",
        "tools": [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command"}
                    },
                    "required": ["command"]
                }
            }
        ],
        "messages": [{"role": "user", "content": "Read the file README.md and summarize it."}]
    }' | python3 -m json.tool
```

A tool call response looks like:
```json
{
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01abc",
            "name": "read_file",
            "input": {"path": "README.md"}
        }
    ],
    "stop_reason": "tool_use"
}
```

## 4. Connect Claude Code CLI

Claude Code loads settings in priority order:
1. **Project-level** `.claude/settings.json` (directory you `cd` into) — highest priority
2. User-level `~/.claude/settings.json` — fallback
3. Env vars at launch — always override both

Use a **project-level `.claude/` directory** to scope these settings to one
project without touching your global Claude Code config.

### Setup (project-scoped, from Mac)

```bash
cd /path/to/your/project

mkdir -p .claude

cat > .claude/settings.json << 'EOF'
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://main4.local:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "qwen3-coder-next",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0"
    },
    "permissions": {
        "allow": [
            "Bash(npm*)", "Bash(node*)", "Bash(python*)", "Bash(pip*)",
            "Bash(git*)", "Bash(ls*)", "Bash(cat*)", "Bash(find*)",
            "Bash(grep*)", "Bash(rg*)", "Bash(mkdir*)", "Bash(touch*)",
            "Read", "Write", "Edit", "Glob", "Grep"
        ]
    }
}
EOF
```

**Critical:** `CLAUDE_CODE_ATTRIBUTION_HEADER=0` must be in `settings.json`, not
passed as a shell env var — it prevents a per-request header that invalidates
the llama.cpp `--cache-prompt` prefix match on every single turn.

Add `.claude/settings.json` to `.gitignore` if you don't want it committed
(it contains the base URL which may change per machine):

```bash
echo '.claude/settings.json' >> .gitignore
```

Or commit it if the whole team uses Thor — it contains no secrets (`local` is
not a real API key).

### Interactive session

```bash
cd /path/to/your/project
claude
# picks up .claude/settings.json automatically
```

### Headless (scripted / CI)

```bash
cd /path/to/your/project
claude --bare \
    -p "List the files in this directory and tell me what this project does." \
    --output-format json \
    --max-turns 5
```

### One-shot env var override (no settings file needed)

```bash
ANTHROPIC_BASE_URL=http://main4.local:8000 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=qwen3-coder-next \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
MAX_THINKING_TOKENS=0 \
claude --bare -p "What does this project do?" --max-turns 3
```

### First-time login bypass

If Claude Code has never been run on this machine, it will prompt for login.
Create `~/.claude.json` once to skip it:

```bash
echo '{"hasCompletedOnboarding": true, "primaryApiKey": "sk-local"}' \
    > ~/.claude.json
```

This does not affect which server Claude Code connects to — that comes from
`ANTHROPIC_BASE_URL` in `.claude/settings.json`.

## 5. Claude Code on Thor itself

Run Claude Code directly on Thor, scoped to the project you're working in:

```bash
ssh yshkolni@main4.local
cd ~/your-project

mkdir -p .claude
cat > .claude/settings.json << 'EOF'
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "qwen3-coder-next",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0"
    }
}
EOF

claude
```

`localhost:8000` instead of `main4.local:8000` — no network hop, lowest latency.

## 6. NemoClaw / OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json
{
    "models": {
        "providers": {
            "thor-local": {
                "baseUrl": "http://main4.local:8000/v1",
                "apiKey": "local",
                "api": "anthropic-messages",
                "models": [
                    {
                        "id": "qwen3-coder-next",
                        "name": "Qwen3-Coder-Next (Thor local)"
                    },
                    {
                        "id": "qwen35-opus-distilled",
                        "name": "Qwen3.5-27B Opus Distilled (Thor local)"
                    }
                ]
            }
        }
    },
    "agents": {
        "defaults": {
            "model": {"primary": "thor-local/qwen3-coder-next"}
        }
    }
}
```

## 7. Session persistence across turns

Pass `X-Session-ID` to keep conversation context and reuse KV cache:

```bash
SESSION="my-coding-session-$(date +%s)"

# Turn 1
curl -s http://main4.local:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: $SESSION" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "My project is called Falcon. Remember that."}]
    }'

# Turn 2 — same slot, KV cache reused, system prompt prefix skipped
curl -s http://main4.local:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: $SESSION" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is my project called?"}]
    }'
```

Claude Code sends `X-Claude-Code-Session-Id` automatically — agent-memory maps
this to a llama.cpp slot via `hash(session_id) % n_slots`.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ConnectionRefused` from Mac | agent-memory not running or wrong host binding | Use `--host 0.0.0.0`; verify with `ss -tlnp \| grep 8000` |
| `ConnectionRefused` from Mac | Port 8000 blocked by firewall | `ssh main4.local "sudo ufw allow 8000"` |
| `{"status":"loading"}` from llama-server | Model still loading | Wait for `{"status":"ok"}` before starting agent-memory |
| `500` on `/v1/messages` | llama-server not running | Check `curl http://main4.local:8001/health` |
| Slow every request (no cache speedup) | `CLAUDE_CODE_ATTRIBUTION_HEADER` not `0` | Confirm it's set in `settings.json`, not just as env var |
| `<think>` tags in response | Thinking not suppressed | Set `MAX_THINKING_TOKENS=0` |
| `slot save failed` | `--slot-save-path` missing from llama-server cmd | Add `--slot-save-path ~/.agent_memory/llamacpp_slots` |
| `no choices` in response | Context overflow | Shorten prompt or increase `--ctx-size` |
| `OOM killed` | Model too large | Switch to Q4_K_M or smaller model |
| DNS `main4.local` not resolving | mDNS not working on Mac | Use IP directly: `ssh yshkolni@main4.local "hostname -I"` |

## Cache Management

Two cache directories on Thor:

| Directory | Contents | Managed by |
|-----------|----------|------------|
| `~/.agent_memory/caches/` | Session KV cache (safetensors per agent ID) | agent-memory |
| `~/.agent_memory/llamacpp_slots/` | Slot KV cache saves (llama-server slot save/restore) | llama-server |

**Clear all cache (start fresh):**
```bash
rm -rf ~/.agent_memory/caches/* ~/.agent_memory/llamacpp_slots/*
```

No restart needed — both are loaded on demand. Do this when:
- Switching models
- Context corruption suspected
- Freeing disk space
- Starting a clean session after major prompt changes

**Inspect cache:**
```bash
# Session caches (one dir per agent ID)
ls -lh ~/.agent_memory/caches/

# Slot saves (one file per slot)
ls -lh ~/.agent_memory/llamacpp_slots/
```

---

## Diagnostics

```bash
# Is agent-memory alive?
curl http://main4.local:8000/health/live

# Is llama-server alive?
curl http://main4.local:8001/health

# Is port 8000 reachable from Mac?
nc -zv main4.local 8000

# What is agent-memory bound to?
ssh yshkolni@main4.local "ss -tlnp | grep 8000"

# Is agent-memory running?
ssh yshkolni@main4.local "ps aux | grep uvicorn | grep -v grep"

# Is llama-server running?
ssh yshkolni@main4.local "ps aux | grep llama-server | grep -v grep"

# Slot cache files on Thor
ssh yshkolni@main4.local "ls -lh ~/.agent_memory/llamacpp_slots/"
```

## Quick reference: environment variables

| Variable | Value | Where |
|----------|-------|-------|
| `ANTHROPIC_BASE_URL` | `http://main4.local:8000` | project `.claude/settings.json` |
| `ANTHROPIC_AUTH_TOKEN` | `local` | project `.claude/settings.json` |
| `ANTHROPIC_MODEL` | `qwen3-coder-next` | project `.claude/settings.json` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` | project `.claude/settings.json` (must be here, not shell) |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` | project `.claude/settings.json` |
| `MAX_THINKING_TOKENS` | `0` | project `.claude/settings.json` |
| `SEMANTIC_BACKEND` | `llamacpp` | Thor, agent-memory startup |
| `SEMANTIC_LLAMACPP_BASE_URL` | `http://localhost:8001` | Thor, agent-memory startup |
| `SEMANTIC_LLAMACPP_MODEL_ID` | `unsloth/Qwen3-Coder-Next-GGUF` | Thor, agent-memory startup |
| `SEMANTIC_LLAMACPP_N_SLOTS` | `4` | Thor, agent-memory startup (match `-np`) |
