# agent-memory — Production Runbook

**Version**: 0.2.0
**Last Updated**: 2026-01-25

---

## Quick Reference

### Health Endpoints

- **Liveness**: `GET /health/live` - Always 200 if process alive
- **Readiness**: `GET /health/ready` - 200 if ready, 503 if not
- **Startup**: `GET /health/startup` - 200 if initialized, 503 if starting

### Metrics Endpoint

- **Prometheus**: `GET /metrics` - Metrics in Prometheus exposition format

### Key Metrics

- `semantic_pool_utilization_ratio` - Pool usage (0.0-1.0)
- `semantic_request_total` - Total requests by method, path, status
- `semantic_request_duration_seconds` - Request latency histogram
- `semantic_agents_active` - Hot agents in memory

---

## Alert Response Procedures

### CRITICAL: SemanticPoolNearExhaustion

**Trigger**: Pool utilization >90% for 5 minutes

**Impact**: Risk of 503 Service Unavailable errors

**Response**:
1. Check current utilization:
   ```bash
   curl http://localhost:8000/health/ready
   # Look for pool_utilization value
   ```

2. Check metrics:
   ```bash
   curl http://localhost:8000/metrics | grep semantic_pool_utilization_ratio
   ```

3. **Immediate Actions**:
   - Scale horizontally: Add more pods/replicas
   - Review active agents: May have memory leak
   - Check for stuck requests

4. **Long-term**:
   - Increase cache budget (SEMANTIC_CACHE_BUDGET_MB)
   - Review agent retention policy
   - Optimize cache eviction strategy

---

### CRITICAL: SemanticHighErrorRate

**Trigger**: >5% of requests returning 5xx for 10 minutes

**Impact**: Service degradation, user-facing errors

**Response**:
1. Check error distribution:
   ```bash
   curl http://localhost:8000/metrics | grep 'semantic_request_total.*5..'
   ```

2. Review logs for errors:
   ```bash
   # If using JSON logging
   cat /var/log/semantic/app.log | jq '. | select(.level == "error")'

   # Recent errors
   tail -100 /var/log/semantic/app.log | grep '"level":"error"'
   ```

3. **Common Causes**:
   - Pool exhaustion (503 errors)
   - Model loading failure (500 errors)
   - Dependency failures (MLX, disk I/O)

4. **Immediate Actions**:
   - Check /health/ready endpoint
   - Review recent deployments (rollback if needed)
   - Verify pool capacity sufficient

---

### CRITICAL: SemanticHealthCheckFailing

**Trigger**: Health check failing for 2 minutes

**Impact**: Service is down, Kubernetes will restart pod

**Response**:
1. Check pod status:
   ```bash
   kubectl get pods -l app=semantic-caching-api
   kubectl describe pod <pod-name>
   ```

2. Check startup logs:
   ```bash
   kubectl logs <pod-name> --tail=100
   ```

3. **Common Causes**:
   - MLX model load failure
   - Out of memory
   - Missing dependencies
   - Configuration error

4. **Immediate Actions**:
   - Review pod events
   - Check resource limits (memory, CPU)
   - Verify environment variables
   - Check persistent volume (if cache persistence enabled)

---

### WARNING: SemanticHighLatency

**Trigger**: p95 latency >3 seconds for 5 minutes

**Impact**: Slow responses, poor user experience

**Response**:
1. Check current latency:
   ```bash
   curl http://localhost:8000/metrics | grep semantic_request_duration_seconds
   ```

2. **Common Causes**:
   - MLX inference slow (normal: ~1-2s per request)
   - Pool near capacity (context switching overhead)
   - Disk I/O slow (cache persistence)

3. **Investigation**:
   - Compare to performance baselines (Day 1 doc)
   - Check pool utilization
   - Review concurrent request count

4. **Actions**:
   - If pool high: Scale horizontally
   - If inference slow: Check MLX performance, verify GPU acceleration
   - If I/O slow: Check disk performance, review cache persistence settings

---

### WARNING: SemanticPoolUtilizationHigh

**Trigger**: Pool utilization >80% for 5 minutes

**Impact**: Approaching capacity, may see degradation soon

**Response**:
1. Monitor trend:
   ```bash
   # Check if rising or stable
   curl http://localhost:8000/metrics | grep semantic_pool_utilization_ratio
   ```

2. **Actions**:
   - Prepare to scale if continues rising
   - Review recent traffic patterns
   - Check for abnormal agent counts

---

## Common Issues and Resolutions

### Issue: 503 Service Unavailable

**Symptom**: Requests return 503, /health/ready returns "not_ready"

**Cause**: Pool exhaustion (utilization >90%)

**Resolution**:
1. Immediate: Scale horizontally (add pods)
2. Short-term: Increase cache budget
3. Long-term: Optimize agent retention, implement request queueing

---

### Issue: Slow Startup (>60s)

**Symptom**: /health/startup returns 503 for extended period

**Cause**: MLX model loading slow

**Resolution**:
1. Verify model cached locally (not downloading)
2. Check disk I/O performance
3. Review model size vs. available memory
4. Increase startup probe timeout if needed

---

### Issue: Memory Leak (Increasing Memory Usage)

**Symptom**: Memory usage grows over time, not released

**Cause**: Agents not being evicted properly

**Resolution**:
1. Check semantic_agents_active metric (should fluctuate)
2. Review agent eviction policy
3. Verify cache persistence working (agents evicted when needed)
4. Check for circular references in code

---

### Issue: Cache Persistence Failing

**Symptom**: Warnings in logs about cache save failures

**Cause**: Disk full, permissions, or corruption

**Resolution**:
1. Check disk space:
   ```bash
   df -h /path/to/cache/dir
   ```

2. Check permissions:
   ```bash
   ls -la /path/to/cache/dir
   ```

3. Verify cache directory writable
4. Check for corrupted cache files

---

## Log Analysis

### Finding Specific Request

Use request_id from X-Request-ID header:

```bash
# JSON logs
cat /var/log/semantic/app.log | jq '. | select(.request_id == "abc123def456")'

# Structured logs (development)
grep "request_id=abc123def456" /var/log/semantic/app.log
```

### Recent Errors

```bash
# Last 100 errors
tail -1000 /var/log/semantic/app.log | jq '. | select(.level == "error")'

# Error summary
cat /var/log/semantic/app.log | jq -r '.event' | grep error | sort | uniq -c | sort -rn
```

### Request Rate

```bash
# Requests per minute
cat /var/log/semantic/app.log | jq '. | select(.event == "request_complete")' | wc -l
```

---

## Scaling Guidelines

### Horizontal Scaling

**When to Scale Out** (add pods):
- Pool utilization >80% sustained
- Request latency p95 >3s sustained
- 503 error rate >5%

**Kubernetes HPA Example**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: semantic-caching-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: semantic-caching-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: semantic_pool_utilization_ratio
      target:
        type: AverageValue
        averageValue: "0.7"
```

### Vertical Scaling

**When to Scale Up** (increase resources):
- Increase cache budget: More concurrent requests per pod
- Increase memory: Support larger models or more agents
- Increase CPU: Faster inference (if not GPU-bound)

---

## Monitoring Dashboards

### Grafana Dashboard (Example Queries)

**Request Rate**:
```promql
rate(semantic_request_total[5m])
```

**Request Latency (p50, p95, p99)**:
```promql
histogram_quantile(0.50, rate(semantic_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(semantic_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(semantic_request_duration_seconds_bucket[5m]))
```

**Pool Utilization**:
```promql
semantic_pool_utilization_ratio
```

**Error Rate**:
```promql
rate(semantic_request_total{status_code=~"5.."}[5m])
/
rate(semantic_request_total[5m])
```

---

## Escalation Procedures

### Severity Levels

**Critical** (Immediate response required):
- Service down (health check failing)
- High error rate (>5% for 10 min)
- Pool exhausted (>90% for 5 min)

**Warning** (Response within 30 min):
- High latency (p95 >3s for 5 min)
- Pool utilization high (>80%)
- High cache eviction rate

**Info** (Review during business hours):
- Traffic patterns
- Agent counts
- Resource utilization trends

### Contact Information

**On-Call**: [Configure your on-call rotation]
**Escalation**: [Configure escalation path]
**Slack Channel**: #semantic-caching-api

---

## Configuration Reference

### Environment Variables

```bash
# Server
SEMANTIC_SERVER_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, PRODUCTION
SEMANTIC_SERVER_CORS_ORIGINS=*

# MLX
SEMANTIC_MLX_MODEL_ID=mlx-community/gemma-2-2b-it-4bit
SEMANTIC_MLX_CACHE_BUDGET_MB=2048
SEMANTIC_MLX_MAX_BATCH_SIZE=4
SEMANTIC_MLX_PREFILL_STEP_SIZE=512

# Agent
SEMANTIC_AGENT_CACHE_DIR=~/.cache/agent_memory
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=100

# Rate Limiting
SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT=60
SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=1000

# Authentication
ANTHROPIC_API_KEY=your-api-key-here  # Comma-separated for multiple keys
```

---

## Useful Commands

### Check Service Health

```bash
# All health endpoints
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/startup

# Metrics
curl http://localhost:8000/metrics

# Root endpoint (API info)
curl http://localhost:8000/
```

### Test Request

```bash
# Anthropic Messages API
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Kubernetes Debugging

```bash
# Pod logs
kubectl logs -f <pod-name>

# Exec into pod
kubectl exec -it <pod-name> -- /bin/bash

# Port forward
kubectl port-forward <pod-name> 8000:8000

# Describe pod
kubectl describe pod <pod-name>

# Events
kubectl get events --sort-by='.lastTimestamp'
```

---

**Runbook Version**: 1.0.0 (Sprint 7)
**Maintainer**: [Your Team]
**Last Review**: 2026-01-25
