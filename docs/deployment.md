# Deployment Guide

Guide for deploying Semantic Caching API on Apple Silicon.

**Important**: This application is optimized for Apple Silicon (M1/M2/M3) with MLX framework. Docker deployment is not supported due to MLX Metal GPU requirements.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [Background Process Management](#background-process-management)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **Apple Silicon**: M1, M2, M3, or later
- **RAM**: Minimum 20GB recommended
  - SmolLM2: 4GB+ sufficient
  - DeepSeek-Coder-V2-Lite: 20GB+ recommended (163K context)
- **Storage**: 10GB+ free space for models and cache

### Software Requirements

- **macOS**: 13.0+ (Ventura or later)
- **Python**: 3.10, 3.11, or 3.12
- **pip**: Latest version
- **Xcode Command Line Tools**:
  ```bash
  xcode-select --install
  ```

### Verify Apple Silicon

```bash
# Check architecture
uname -m
# Expected: arm64

# Check Metal support
python3 -c "import mlx.core as mx; print('Metal available:', mx.metal.is_available())"
# Expected: Metal available: True
```

## Installation

### Step 1: Install via pip

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install semantic-caching-api
pip install -e .

# Verify installation
semantic version
# Expected: semantic-caching-api v1.0.0
```

### Step 2: Verify Installation

```bash
# Check MLX installation
pip show mlx-lm
# Should show version info

# Test Metal GPU
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
# Should print: True

# Check semantic CLI
semantic --help
# Should show available commands
```

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```bash
# .env

# MLX Model Configuration
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5
SEMANTIC_MLX_PREFILL_STEP_SIZE=512
SEMANTIC_MLX_KV_BITS=null
SEMANTIC_MLX_DEFAULT_MAX_TOKENS=256
SEMANTIC_MLX_DEFAULT_TEMPERATURE=0.7

# Agent Cache Configuration
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=5
SEMANTIC_AGENT_CACHE_DIR=~/.agent_memory/caches
SEMANTIC_AGENT_BATCH_WINDOW_MS=10
SEMANTIC_AGENT_LRU_EVICTION_ENABLED=true
SEMANTIC_AGENT_EVICT_TO_DISK=true

# Server Configuration
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_LOG_LEVEL=INFO
SEMANTIC_SERVER_WORKERS=1
SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT=60
SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=1000
SEMANTIC_SERVER_CORS_ORIGINS=http://localhost:3000

# Security (optional but recommended)
SEMANTIC_API_KEY=your-secure-api-key-here
SEMANTIC_ADMIN_KEY=your-secure-admin-key-here
```

### Cache Directory Setup

```bash
# Create cache directory
mkdir -p ~/.agent_memory/caches

# Verify permissions
ls -la ~/.agent_memory/caches

# Optional: Custom location
mkdir -p /path/to/custom/caches
echo "SEMANTIC_AGENT_CACHE_DIR=/path/to/custom/caches" >> .env
```

### Example Configurations

**Development** (.env.development):
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
SEMANTIC_MLX_CACHE_BUDGET_MB=1024
SEMANTIC_MLX_MAX_BATCH_SIZE=2
SEMANTIC_SERVER_LOG_LEVEL=DEBUG
SEMANTIC_SERVER_CORS_ORIGINS=*
```

**Production** (.env.production):
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5
SEMANTIC_SERVER_LOG_LEVEL=INFO
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_API_KEY=your-production-api-key
SEMANTIC_ADMIN_KEY=your-admin-key
```

## Running the Server

### Development Mode

Start server in foreground:

```bash
# Activate virtual environment
source venv/bin/activate

# Start server (uses .env)
semantic serve

# Or specify options
semantic serve --port 8080 --log-level DEBUG
```

**Expected output**:
```
INFO: Loading model: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
INFO: Model loaded successfully
INFO: Started server on http://0.0.0.0:8000
INFO: Prometheus metrics on /metrics
```

### Verify Server

```bash
# Check health
curl http://localhost:8000/health
# {"status":"healthy"}

# Make test request
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Production Mode

For production, run server in background with logging:

```bash
# Start in background
nohup semantic serve > semantic.log 2>&1 &

# Save PID
echo $! > semantic.pid

# Check logs
tail -f semantic.log

# Stop server
kill $(cat semantic.pid)
```

## Background Process Management

### macOS launchd (Recommended)

Create a launchd plist for automatic startup:

**File**: `~/Library/LaunchAgents/com.semantic.server.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.semantic.server</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/venv/bin/semantic</string>
        <string>serve</string>
        <string>--port</string>
        <string>8000</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/semantic</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>SEMANTIC_MLX_MODEL_ID</key>
        <string>mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx</string>
        <key>SEMANTIC_MLX_CACHE_BUDGET_MB</key>
        <string>4096</string>
        <key>SEMANTIC_SERVER_LOG_LEVEL</key>
        <string>INFO</string>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/.semantic/logs/semantic.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/.semantic/logs/semantic.error.log</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

**Setup**:

```bash
# Replace YOUR_USERNAME
sed -i '' 's/YOUR_USERNAME/'"$(whoami)"'/g' ~/Library/LaunchAgents/com.semantic.server.plist

# Create log directory
mkdir -p ~/.agent_memory/logs

# Load service
launchctl load ~/Library/LaunchAgents/com.semantic.server.plist

# Check status
launchctl list | grep semantic

# View logs
tail -f ~/.agent_memory/logs/semantic.log

# Stop service
launchctl unload ~/Library/LaunchAgents/com.semantic.server.plist

# Start service
launchctl load ~/Library/LaunchAgents/com.semantic.server.plist
```

### Systemd (Linux - if needed)

For Linux systems (not typical for Apple Silicon):

**File**: `/etc/systemd/system/semantic.service`

```ini
[Unit]
Description=Semantic Caching API Server
After=network.target

[Service]
Type=simple
User=semantic
WorkingDirectory=/home/semantic/semantic-caching-api
Environment="PATH=/home/semantic/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/semantic/semantic-caching-api/.env
ExecStart=/home/semantic/venv/bin/semantic serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Commands**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable semantic
sudo systemctl start semantic
sudo systemctl status semantic
```

## Monitoring

### Health Checks

Monitor server health:

```bash
# Basic health check
curl http://localhost:8000/health

# Automated monitoring (every 30s)
while true; do
  curl -s http://localhost:8000/health | jq .
  sleep 30
done
```

### Prometheus Metrics

Expose metrics for monitoring:

```bash
# View metrics
curl http://localhost:8000/metrics

# Example output:
# semantic_requests_total{endpoint="/v1/messages"} 1523
# semantic_active_agents 3
# semantic_cache_hits_total 892
```

### Prometheus Configuration

Add to Prometheus config (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'semantic-caching-api'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### Logging

Configure logging levels:

```bash
# Development (verbose)
SEMANTIC_SERVER_LOG_LEVEL=DEBUG semantic serve

# Production (standard)
SEMANTIC_SERVER_LOG_LEVEL=INFO semantic serve

# Errors only
SEMANTIC_SERVER_LOG_LEVEL=ERROR semantic serve
```

**Log rotation** (using logrotate on Linux):

```bash
# /etc/logrotate.d/semantic
/home/semantic/.semantic/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 semantic semantic
    sharedscripts
}
```

### Performance Monitoring

Monitor system resources:

```bash
# Check memory usage
ps aux | grep semantic

# Monitor with Activity Monitor (macOS)
open -a "Activity Monitor"

# Real-time monitoring
top -pid $(pgrep -f "semantic serve")
```

## Security

### API Authentication

Enable API key authentication:

```bash
# Generate secure API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
echo "SEMANTIC_API_KEY=your-generated-key" >> .env

# Restart server
kill $(cat semantic.pid)
nohup semantic serve > semantic.log 2>&1 &
```

**Usage**:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "X-API-Key: your-generated-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-12b-it-4bit", "max_tokens": 50, "messages": [...]}'
```

### Admin Key

Protect admin endpoints:

```bash
# Generate admin key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
echo "SEMANTIC_ADMIN_KEY=your-admin-key" >> .env
```

**Usage**:
```bash
curl -X POST http://localhost:8000/admin/swap \
  -H "X-Admin-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/SmolLM2-135M-Instruct"}'
```

### CORS Configuration

Restrict origins:

```bash
# Single origin
SEMANTIC_SERVER_CORS_ORIGINS=https://yourdomain.com

# Multiple origins (comma-separated)
SEMANTIC_SERVER_CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com

# Development (allow all)
SEMANTIC_SERVER_CORS_ORIGINS=*
```

### Firewall

Restrict network access:

```bash
# macOS firewall (GUI)
System Settings → Network → Firewall

# Or use pf (packet filter)
# /etc/pf.conf
pass in proto tcp from 192.168.1.0/24 to any port 8000
block in proto tcp to any port 8000
```

## Troubleshooting

### Issue: Server Won't Start

**Symptom**: Server fails to start or crashes immediately

**Solutions**:
1. Check logs:
   ```bash
   tail -f semantic.log
   ```
2. Verify Python and MLX:
   ```bash
   python3 --version
   pip show mlx-lm
   ```
3. Check port availability:
   ```bash
   lsof -i :8000
   ```
4. Test model loading:
   ```bash
   python3 -c "from mlx_lm import load; load('mlx-community/SmolLM2-135M-Instruct')"
   ```

### Issue: High Memory Usage

**Symptom**: System runs out of memory

**Solutions**:
1. Reduce cache budget:
   ```bash
   SEMANTIC_MLX_CACHE_BUDGET_MB=2048
   ```
2. Use smaller model:
   ```bash
   SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
   ```
3. Limit agents in memory:
   ```bash
   SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=3
   ```

### Issue: Slow Performance

**Symptom**: Requests take a long time

**Solutions**:
1. Verify Metal GPU:
   ```bash
   python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
   ```
2. Check system resources (Activity Monitor)
3. Reduce batch size:
   ```bash
   SEMANTIC_MLX_MAX_BATCH_SIZE=2
   ```

### Issue: launchd Service Won't Start

**Symptom**: Service fails to load

**Solutions**:
1. Check plist syntax:
   ```bash
   plutil -lint ~/Library/LaunchAgents/com.semantic.server.plist
   ```
2. Verify paths in plist (absolute paths required)
3. Check permissions:
   ```bash
   chmod 644 ~/Library/LaunchAgents/com.semantic.server.plist
   ```
4. View service logs:
   ```bash
   tail -f ~/.agent_memory/logs/semantic.error.log
   ```

## Best Practices

1. **Use Virtual Environment**: Isolate dependencies
2. **Set Resource Limits**: Configure cache budget appropriately
3. **Enable Monitoring**: Use Prometheus for metrics
4. **Rotate Logs**: Prevent disk space issues
5. **Use API Keys**: Secure production deployments
6. **Backup Caches**: Periodically backup `~/.agent_memory/caches/`
7. **Update Regularly**: Keep dependencies current
8. **Test Before Deploy**: Validate with SmolLM2 first

## Performance Benchmarks

**DeepSeek-Coder-V2-Lite (M3 Max, 64GB RAM)**:
- Latency: ~50-100ms per token
- Throughput: 50-100 tokens/second
- Memory: ~20GB (model + cache)

**SmolLM2 (M1, 16GB RAM)**:
- Latency: ~20-40ms per token
- Throughput: 25-30 tokens/second
- Memory: ~2GB (model + cache)

## See Also

- [Configuration Guide](configuration.md) - Complete configuration reference
- [User Guide](user-guide.md) - API usage and examples
- [Testing Guide](testing.md) - Testing strategy and commands
- [Model Onboarding](model-onboarding.md) - Adding new models
