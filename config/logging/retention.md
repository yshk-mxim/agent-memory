# Log Retention Policy - agent-memory

**Version**: 1.0.0
**Effective Date**: 2026-01-25

---

## Log Retention Overview

### Local Logs (Development)

**Location**: `/tmp/claude/e2e_logs/` (E2E tests)
**Retention**: Deleted after test completion
**Purpose**: Testing and debugging

### Application Logs (Production)

**Location**: Configurable via deployment (stdout by default)
**Format**: JSON (structured logging via structlog)
**Retention**: Depends on log aggregator configuration

---

## Recommended Retention Periods

### Hot Storage (Fast Access)

**Duration**: 7 days
**Purpose**: Recent troubleshooting, active investigations
**Storage**: SSD, low-latency access
**Cost**: Higher per GB

**Example Configuration** (Kubernetes/ELK):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*agent-memory*.log

    output.elasticsearch:
      hosts: ["elasticsearch:9200"]
      index: "agent-memory-%{+yyyy.MM.dd}"

    setup.ilm:
      enabled: true
      policy_name: "agent-memory-hot-7days"
      rollover_alias: "agent-memory"
```

---

### Warm Storage (Occasional Access)

**Duration**: 30 days total (23 days in warm)
**Purpose**: Historical analysis, compliance, auditing
**Storage**: Standard storage, compressed
**Cost**: Medium per GB

**Elasticsearch ILM Policy Example**:
```json
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_age": "7d",
            "max_size": "50gb"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "readonly": {},
          "shrink": {
            "number_of_shards": 1
          },
          "forcemerge": {
            "max_num_segments": 1
          }
        }
      },
      "delete": {
        "min_age": "30d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

---

### Cold Storage (Archive)

**Duration**: 90-365 days (optional)
**Purpose**: Compliance, long-term auditing
**Storage**: Object storage (S3, GCS), compressed
**Cost**: Low per GB, high retrieval cost

**Not Required for v0.2.0**: Can be added based on compliance needs.

---

## Log Rotation Configuration

### File-based Logging (if using log files)

**Rotation Policy**:
- **Size**: Rotate when file reaches 100 MB
- **Time**: Rotate daily at midnight UTC
- **Compression**: gzip compressed after rotation
- **Retention**: Keep 7 rotated files (7 days)

**Example** (logrotate config):
```
/var/log/agent-memory/app.log {
    daily
    rotate 7
    size 100M
    compress
    delaycompress
    notifempty
    create 0644 agent-memory agent-memory
    sharedscripts
    postrotate
        # Signal application to reopen log file (if needed)
        kill -USR1 $(cat /var/run/agent-memory.pid) 2>/dev/null || true
    endscript
}
```

---

### Stdout Logging (Recommended for Kubernetes)

**Configuration**: Logs written to stdout, captured by container runtime
**Rotation**: Handled by Kubernetes/Docker
**Retention**: Configured via log aggregator (Filebeat, Fluentd, etc.)

**Kubernetes Log Rotation** (Docker):
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: agent-memory
spec:
  containers:
  - name: api
    image: agent-memory:latest
    env:
    - name: SEMANTIC_SERVER_LOG_LEVEL
      value: "PRODUCTION"  # JSON logging
    # Docker log rotation
    # Configured in /etc/docker/daemon.json on host:
    # {
    #   "log-driver": "json-file",
    #   "log-opts": {
    #     "max-size": "10m",
    #     "max-file": "3"
    #   }
    # }
```

---

## Log Levels and Retention

### DEBUG Logs

**When**: Development only
**Retention**: Local only, not shipped to production
**Volume**: High (includes request/response details)

**Environment**:
```bash
export SEMANTIC_SERVER_LOG_LEVEL=DEBUG
```

---

### INFO Logs

**When**: Development and production
**Retention**: Full retention period (7-30 days)
**Volume**: Medium (request lifecycle, metrics, health checks)

**Environment**:
```bash
export SEMANTIC_SERVER_LOG_LEVEL=INFO
```

---

### WARNING Logs

**When**: All environments
**Retention**: Full retention period + archive (90 days)
**Volume**: Low (errors, degraded states)

**Queries**:
```json
{
  "query": {
    "term": {
      "level": "warning"
    }
  }
}
```

---

### ERROR Logs

**When**: All environments
**Retention**: Full retention period + archive (365 days for compliance)
**Volume**: Very low (failures, exceptions)

**Queries**:
```json
{
  "query": {
    "term": {
      "level": "error"
    }
  }
}
```

---

## Structured Logging Fields

### Always Captured

- `timestamp` (ISO 8601)
- `level` (debug, info, warning, error)
- `event` (event name, e.g., "request_complete")
- `request_id` (correlation ID)
- `method` (HTTP method)
- `path` (request path)

### Indexed Fields (for fast queries)

- `request_id` - Trace all logs for a request
- `status_code` - Filter by response status
- `error_type` - Group errors by type
- `duration_ms` - Analyze latency

---

## Compliance and Privacy

### PII Handling

**Policy**: No PII in logs
**Implementation**:
- Request/response bodies NOT logged by default
- Only metadata logged (method, path, status, timing)
- API keys masked in logs (never logged)

**Exception**: If detailed request logging needed for debugging, enable with:
```python
# NOT recommended for production
logger.debug("request_body", body=request.json())
```

---

### GDPR Compliance

**Right to be Forgotten**:
- Request IDs linked to user can be purged on request
- Elasticsearch delete by query:
```json
POST /agent-memory-*/_delete_by_query
{
  "query": {
    "term": {
      "request_id": "user-request-id-to-delete"
    }
  }
}
```

---

## Cost Estimation

### Log Volume Estimates

**Per Request** (JSON log):
- request_start: ~200 bytes
- request_complete: ~150 bytes
- Total per request: ~350 bytes

**Daily Volume** (1000 req/min):
- Requests/day: 1,440,000
- Log size/day: ~490 MB
- Log size/month: ~14.7 GB

### Storage Costs (AWS Example)

**Hot (7 days)**:
- 7 days × 490 MB = 3.4 GB
- Elasticsearch: ~$0.13/GB/day = ~$3.10/week

**Warm (23 days)**:
- 23 days × 490 MB = 11.3 GB
- S3 Standard: ~$0.023/GB/month = ~$0.26/month

**Total**: ~$13-15/month for 30-day retention at 1000 req/min

---

## Monitoring Log Pipeline

### Health Checks

**Filebeat Monitoring**:
```bash
# Check Filebeat status
kubectl get pods -l app=filebeat

# Check logs shipped
kubectl logs -l app=filebeat | grep "events sent"
```

**Elasticsearch Monitoring**:
```bash
# Check index sizes
curl http://elasticsearch:9200/_cat/indices/agent-memory-*

# Check document count
curl http://elasticsearch:9200/agent-memory-*/_count
```

### Alerts

**Log Pipeline Down**:
```yaml
- alert: FilebeatNotShipping
  expr: rate(filebeat_events_sent[5m]) == 0
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Filebeat not shipping logs for 10 minutes"
```

**Disk Space Low**:
```yaml
- alert: ElasticsearchDiskSpaceLow
  expr: elasticsearch_filesystem_data_available_bytes / elasticsearch_filesystem_data_size_bytes < 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Elasticsearch disk space <10%"
```

---

## Log Retention Checklist

### Initial Setup

- [ ] Configure log aggregator (ELK, Splunk, Datadog)
- [ ] Set up index lifecycle management (ILM)
- [ ] Configure retention periods (hot: 7d, warm: 30d)
- [ ] Enable log compression
- [ ] Set up log rotation (if file-based)

### Ongoing Maintenance

- [ ] Monitor log volume weekly
- [ ] Review retention costs monthly
- [ ] Audit log access quarterly
- [ ] Test log retrieval from archive
- [ ] Review and update retention policy annually

---

**Policy Version**: 1.0.0
**Effective Date**: 2026-01-25
**Review Date**: 2027-01-25
**Maintainer**: [Your Team]
