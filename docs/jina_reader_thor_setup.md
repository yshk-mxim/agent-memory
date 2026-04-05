# Local URL-to-Markdown Reader on Thor

Converts any public URL into clean markdown — 5-10x fewer tokens than raw HTML.
Running locally keeps page content private (no traffic to `r.jina.ai`).

Implemented as a lightweight Python HTTP server using `html2text`.
No Docker required — runs natively on Thor's ARM64 (Jetson).

**Source:** `~/reader_server.py` on Thor  
**Port:** 3000  
**Interface:** `GET http://<thor-ip>:3000/<url>`

---

## Setup (one-time on Thor)

```bash
# Install html2text
pip3 install html2text --break-system-packages

# Copy reader_server.py to home directory (already done)

# Install as persistent systemd user service
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/reader.service << 'SVCEOF'
[Unit]
Description=Local URL-to-markdown reader server
After=network.target

[Service]
ExecStart=/usr/bin/python3 $HOME/reader_server.py
Restart=always
RestartSec=5
StandardOutput=append:$HOME/reader_server.log
StandardError=append:$HOME/reader_server.log

[Install]
WantedBy=default.target
SVCEOF

systemctl --user daemon-reload
systemctl --user enable reader
systemctl --user start reader
```

Verify:
```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:3000/https://example.com').read().decode()[:200])"
# Example Domain
# This domain is for use in documentation examples...
```

---

## Usage

### From Mac (Bash/Python, no URL restrictions)

```bash
# Via fetch.py helper
python3 ~/fetch.py https://docs.python.org/3/library/urllib.html

# Direct via Python
python3 -c "import urllib.request; print(urllib.request.urlopen('http://192.168.184.150:3000/https://example.com').read().decode())"
```

### In CLAUDE.md (instructs Claude Code)

```markdown
Use Bash to fetch pages as clean markdown:
    Bash: curl -s "http://192.168.184.150:3000/https://example.com/page"
Or:
    Bash: python3 ~/fetch.py https://example.com/page
```

---

## How It Works

`reader_server.py` is a minimal `http.server`-based HTTP server:
1. Receives `GET /<url>`
2. Fetches the target URL with a browser-like User-Agent
3. Converts HTML → markdown via `html2text`
4. Returns `text/plain` response

Non-HTML responses (JSON, plain text) are passed through unchanged.

Compared to the Jina AI cloud service, this version:
- Has no JavaScript rendering (no Puppeteer/Chromium) — works for most docs/blogs
- Is ~2 MB RAM vs ~500 MB for Puppeteer-based Docker image
- Runs natively on ARM64 (no cross-architecture emulation)
- Pages requiring JS to render (SPAs) may return empty or partial content

---

## Management

```bash
# Status
systemctl --user status reader

# Logs
tail -f ~/reader_server.log

# Restart
systemctl --user restart reader

# Stop
systemctl --user stop reader
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused :3000` | Service not running | `systemctl --user start reader` |
| Empty/partial content | Page requires JavaScript | Use SearXNG snippet instead, or skip |
| `502` error | Target site unreachable from Thor | Verify network connectivity |
| Port 3000 in use | Another process | `ss -tlnp | grep 3000` to find it |

---

## See Also

- [`privacy_enhanced_example.md`](privacy_enhanced_example.md) — full private stack overview
- [`searxng_thor_setup.md`](searxng_thor_setup.md) — web search companion service
