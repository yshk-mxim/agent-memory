# SearXNG on Thor — Web Search for Local LLMs

SearXNG is a self-hosted meta-search engine that queries Google, DuckDuckGo,
and Bing simultaneously and returns clean JSON. No API keys, no rate limits,
no external accounts.

## Architecture

```
Claude Code (Mac)
  → WebFetch http://main4.local:8080/search?q=...&format=json
    → SearXNG (Thor :8080, Docker)
      → Google + DuckDuckGo + Bing
```

## Start SearXNG (one-time setup on Thor)

```bash
ssh <user>@<thor-hostname>

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

`--restart always` means it comes back up after Thor reboots.

Verify:
```bash
python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8080/search?q=test&format=json')
d = json.loads(r.read())
print('results:', len(d.get('results', [])))
"
# results: 39
```

## Usage from Claude Code

Add a `CLAUDE.md` to your project:

```markdown
## Web Search

Use WebFetch to search: `http://main4.local:8080/search?q=YOUR+QUERY&format=json`

Response JSON has a `results` array with `url`, `title`, `content`, `engine` per result.
Fetch the actual page with a second WebFetch when the snippet isn't enough.
```

### Example search response

```json
{
  "query": "llama.cpp context size",
  "results": [
    {
      "url": "https://github.com/ggml-org/llama.cpp/...",
      "title": "Context size options · ggml-org/llama.cpp",
      "content": "Use --ctx-size to set the maximum context ...",
      "engine": "google"
    },
    ...
  ]
}
```

## Management

```bash
# Status
docker ps --filter name=searxng

# Logs
docker logs searxng --tail 20

# Restart
docker restart searxng

# Stop
docker stop searxng

# Update to latest image
docker pull searxng/searxng && docker restart searxng
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 0 results | Google/DDG bot detection — results vary; try a different query |
| Port 8080 blocked | `sudo ufw allow 8080` on Thor |
| Container not starting | Check `docker logs searxng` for settings.yml parse errors |
| `format=json` returns HTML | Confirm `formats: [html, json]` is in settings.yml, restart container |
