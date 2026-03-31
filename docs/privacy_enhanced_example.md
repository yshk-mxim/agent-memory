# Privacy-Enhanced Setup — Fully Local AI Stack

This guide documents a complete private AI coding assistant stack where **no query,
no page content, and no conversation ever leaves your local network.**

| Component | Role | External traffic |
|-----------|------|-----------------|
| llama-server | LLM inference | None — weights on disk |
| agent-memory | API proxy + session cache | None |
| SearXNG | Web search | Search queries to Google/DDG/Bing only |
| Jina Reader (local) | URL→markdown conversion | Outbound fetches to target URLs only |
| Claude Code | IDE/CLI | `ANTHROPIC_BASE_URL` pointed at Thor — no Anthropic cloud |

The only data that leaves your machine is: (1) search queries to public search engines via SearXNG,
and (2) HTTP GETs to pages you explicitly ask to fetch. No model queries, no conversation history,
no code context reaches any external API.

---

## Architecture

```
Mac (Claude Code CLI)
  │
  │  Anthropic Messages API (HTTP, private LAN)
  ▼
Thor — agent-memory :8000
  ├── /v1/messages ──────────────────────► llama-server :8001 ──► GPU (Qwen3-Coder-Next)
  ├── /v1/chat/completions ──────────────► llama-server :8001
  ├── /search?q=... ─────────────────────► SearXNG :8080 ──► Google + DDG + Bing
  └── (model traffic only, never leaves Thor)

Thor — Jina Reader :3000
  └── /<url> ─────────────────────────────► target website (fetch on demand)

~/fetch.py  (on Thor, called by Claude via Bash)
  └── http://localhost:3000/<url> ─────────► Jina Reader :3000

~/search.py (on Thor, called by Claude via Bash)
  └── http://localhost:8000/search?q=... ──► agent-memory /search proxy
```

All inference, caching, and tool execution runs on Thor. The Mac only runs the
Claude Code CLI.

---

## Prerequisites

- **Thor** (or any Linux server with NVIDIA GPU + Docker)
- **Mac** with Claude Code CLI installed
- llama-server built for your GPU (see [`llamacpp_thor_example.md`](llamacpp_thor_example.md))
- Docker on Thor

---

## Step 1 — LLM Inference: llama-server

```bash
# On Thor
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

Wait for: `llama server listening at http://0.0.0.0:8001`

For full llama-server options and model download instructions see
[`llamacpp_thor_example.md`](llamacpp_thor_example.md).

---

## Step 2 — API Proxy: agent-memory

```bash
# On Thor (new terminal or nohup)
SEMANTIC_BACKEND=llamacpp \
SEMANTIC_LLAMACPP_BASE_URL=http://localhost:8001 \
SEMANTIC_LLAMACPP_MODEL_ID=qwen3-coder-next \
SEMANTIC_LLAMACPP_TOKENIZER_ID=Qwen/Qwen2.5-Coder-32B-Instruct \
SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH=131072 \
SEMANTIC_LLAMACPP_N_SLOTS=4 \
SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host '::' --port 8000
```

`SEMANTIC_SERVER_SEARXNG_URL` enables the `/search` proxy endpoint.
Omit it if SearXNG is not running yet.

Verify:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Step 3 — Web Search: SearXNG

```bash
# On Thor
mkdir -p ~/searxng
cat > ~/searxng/settings.yml << 'EOF'
use_default_settings: true

server:
  bind_address: "0.0.0.0"
  port: 8080
  secret_key: "REPLACE_WITH_$(openssl rand -hex 32)"
  limiter: false
  image_proxy: false

search:
  safe_search: 0
  formats:
    - html
    - json

engines:
  - name: google
    engine: google
    shortcut: g
    weight: 2
  - name: duckduckgo
    engine: duckduckgo
    shortcut: d
    weight: 1
  - name: bing
    engine: bing
    shortcut: b
    weight: 1
EOF

docker run -d --restart always \
  --name searxng \
  -p 8080:8080 \
  -v ~/searxng:/etc/searxng \
  searxng/searxng
```

Verify:
```bash
python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8080/search?q=test&format=json')
print('results:', len(json.loads(r.read()).get('results', [])))
"
# results: 39
```

For more details see [`searxng_thor_setup.md`](searxng_thor_setup.md).

---

## Step 4 — Page Fetching: Jina Reader

Converts any public URL to clean markdown. Runs locally — page content never
passes through Jina AI's cloud service.

```bash
# On Thor
docker run -d \
  --name jina-reader \
  --restart always \
  -p 3000:3000 \
  ghcr.io/intergalacticalvariable/reader:latest
```

Verify:
```bash
curl "http://localhost:3000/https://example.com" | head -10
```

For more details see [`jina_reader_thor_setup.md`](jina_reader_thor_setup.md).

---

## Step 5 — Helper Scripts on Thor

These scripts let Claude Code call search and fetch via Bash without hitting
Claude Code's WebFetch private-IP restriction.

**`~/search.py`** — web search:
```python
#!/usr/bin/env python3
"""Web search via SearXNG on Thor. Usage: python3 ~/search.py "your query" """
import sys, json, urllib.request, urllib.parse

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
if not query:
    print("Usage: python3 ~/search.py \"your query\""); sys.exit(1)

url = "http://192.168.184.150:8000/search?q=" + urllib.parse.quote_plus(query)
try:
    data = json.loads(urllib.request.urlopen(url, timeout=15).read())
except Exception as e:
    print(f"Search failed: {e}", file=sys.stderr); sys.exit(1)

results = data.get("results", [])
print(f"Search: {query!r}  ({len(results)} results)\n")
for i, r in enumerate(results, 1):
    print(f"{i}. [{r.get('engine','?')}] {r.get('title','')}")
    print(f"   {r.get('url','')}")
    if snippet := r.get("content","").strip():
        print(f"   {snippet[:200]}")
    print()
```

**`~/fetch.py`** — URL to markdown:
```python
#!/usr/bin/env python3
"""Fetch a URL as clean markdown via local Jina Reader. Usage: python3 ~/fetch.py <url>"""
import sys, urllib.request

url = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
if not url:
    print("Usage: python3 ~/fetch.py <url>"); sys.exit(1)
if not url.startswith("http"):
    url = "https://" + url

req = urllib.request.Request("http://localhost:3000/" + url, headers={"Accept": "text/plain"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.read().decode(errors="replace"))
except Exception as e:
    print(f"Fetch failed: {e}", file=sys.stderr); sys.exit(1)
```

---

## Step 6 — Claude Code Configuration

### Mac: `~/.claude/projects/<project>/settings.json`

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.184.150:8000",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "qwen3-coder-next",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "MAX_THINKING_TOKENS": "0"
  }
}
```

- `ANTHROPIC_BASE_URL` — points Claude Code at agent-memory instead of Anthropic cloud
- `ANTHROPIC_AUTH_TOKEN` — any non-empty string; agent-memory doesn't validate it
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` — disables telemetry, update checks, etc.

### Project: `CLAUDE.md`

Add to your project's `CLAUDE.md` to instruct Claude how to search and fetch:

```markdown
# Web Search

DO NOT use the WebSearch tool — it requires an API key and returns 0 results.

For web search, use Bash with curl (no URL restrictions in Bash):
    Bash: curl -s "http://192.168.184.150:8000/search?q=your+query+here"

Response is JSON with a `results` array — each entry has `url`, `title`, `content`, `engine`.
Alternatively: python3 ~/search.py "your query here"

# Fetching Web Pages

DO NOT use WebFetch directly for public URLs — it returns raw HTML (slow, token-heavy).

Use Bash to fetch via the local Jina Reader on Thor (returns clean markdown, 5-10x smaller):
    Bash: curl -s "http://192.168.184.150:3000/https://example.com/page"

Alternatively: python3 ~/fetch.py https://example.com/page

For compact content that doesn't need conversion (JSON APIs like api.github.com, search
snippets), WebFetch is fine without the proxy.
```

> **Why Bash instead of WebFetch?** Claude Code blocks WebFetch to private IPs
> (SSRF protection). Bash/curl has no such restriction. The Python helper scripts
> (`~/search.py`, `~/fetch.py`) do the same thing and are useful for other users or
> non-Claude tooling.

---

## Thinking Mode

Claude Code's `/effort` command controls thinking:

| Command | API field | Effect on Qwen3-Coder-Next |
|---------|-----------|---------------------------|
| `/effort low` | `thinking: {type: "disabled"}` | `/no_think` prepended to system prompt — fast responses |
| `/effort high` | `thinking: {type: "enabled"}` | Full chain-of-thought |
| (default) | no `thinking` field | `/no_think` applied (agent-memory default) |

agent-memory maps `thinking.type` → `disable_thinking` → `_apply_no_think()` in
`LlamaCppBackendAdapter`. For models finetuned from thinking models (e.g. Qwen3.5-27B
Opus Distilled), `/no_think` has no effect — the model always thinks.

---

## Startup Order

Services must start in this order (each depends on the previous being healthy):

```
1. llama-server   (model loads in ~30s)
2. agent-memory   (connects to llama-server on startup for model spec)
3. SearXNG        (independent — can start any time)
4. Jina Reader    (independent — can start any time)
```

Verify the full stack from the Mac:
```bash
# Model inference
curl http://192.168.184.150:8000/v1/models

# Search proxy
curl "http://192.168.184.150:8000/search?q=test" | python3 -m json.tool | head -20

# Page fetch (must run on Thor — localhost:3000)
ssh user@main4.local "python3 ~/fetch.py https://example.com | head -10"
```

---

## What Still Leaves the Network

| Data | Destination | Opt-out |
|------|-------------|---------|
| Search query strings | Google, DuckDuckGo, Bing (via SearXNG) | Remove engines from `settings.yml` |
| HTTP GETs to fetched URLs | Target website | Don't call `python3 ~/fetch.py` |
| DNS lookups for fetched URLs | Your DNS resolver | Use a local DNS resolver |

**What does NOT leave the network:** model prompts, code context, conversation history,
file contents, tool call arguments, API keys.

---

## See Also

- [`llamacpp_thor_example.md`](llamacpp_thor_example.md) — llama-server setup, model download, context sizing
- [`searxng_thor_setup.md`](searxng_thor_setup.md) — SearXNG configuration details
- [`jina_reader_thor_setup.md`](jina_reader_thor_setup.md) — Jina Reader Docker details
- [`configuration.md`](configuration.md) — all `SEMANTIC_*` env vars
