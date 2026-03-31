# Local Jina Reader on Thor

Jina Reader converts any public URL into clean markdown — 5-10x fewer tokens than raw HTML.
Running it locally keeps page content private and removes the external dependency on `r.jina.ai`.

Source: [github.com/jina-ai/reader](https://github.com/jina-ai/reader) (TypeScript, Puppeteer + Mozilla Readability)  
Self-hosted fork: [github.com/intergalacticalvariable/reader](https://github.com/intergalacticalvariable/reader)

---

## Start on Thor

```bash
docker run -d \
  --name jina-reader \
  --restart always \
  -p 3000:3000 \
  ghcr.io/intergalacticalvariable/reader:latest
```

`--restart always` ensures it comes back after reboots.

Verify:
```bash
curl "http://localhost:3000/https://example.com" | head -20
```

---

## Usage

### From Thor (Bash, no URL restrictions)

```bash
# Fetch any public URL as markdown
python3 ~/fetch.py https://docs.python.org/3/library/urllib.html
```

`~/fetch.py` hits `http://localhost:3000/<url>` and prints the markdown response.

### Direct curl

```bash
curl "http://localhost:3000/https://example.com/page"
```

---

## Agent Test Integration

`~/agent_test/CLAUDE.md` instructs Claude Code to use `python3 ~/fetch.py <url>` via Bash
for all page fetches. This bypasses Claude Code's WebFetch private-IP restriction
(the HTTP request originates from Thor's loopback, not the Mac).

The companion search script `~/search.py` handles web search via SearXNG (port 8080).
See [`searxng_thor_setup.md`](searxng_thor_setup.md) for SearXNG setup.

---

## Management

```bash
# Check status
docker ps --filter name=jina-reader

# View logs
docker logs jina-reader --tail 20

# Stop / remove
docker stop jina-reader && docker rm jina-reader

# Update to latest image
docker pull ghcr.io/intergalacticalvariable/reader:latest
docker stop jina-reader && docker rm jina-reader
# re-run the docker run command above
```

---

## Memory Budget

The container uses Puppeteer (headless Chromium) — allocate ~500 MB RAM.
Thor has 128 GB unified memory, so this is negligible alongside llama-server.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused :3000` | Container not running | `docker start jina-reader` |
| `Timeout` on fetch | Target site slow or blocked | Increase timeout in `fetch.py` or skip |
| `Error: net::ERR_NAME_NOT_RESOLVED` | DNS unavailable in container | `docker run --dns 8.8.8.8 ...` |
| Port 3000 already in use | Another service | Change `-p 3001:3000` and update `fetch.py` |
