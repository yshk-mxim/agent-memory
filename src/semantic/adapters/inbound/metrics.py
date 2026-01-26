"""Prometheus metrics for production monitoring.

Defines core metrics for observability:
- Request throughput and latency
- Pool utilization
- Active agents
- Cache hit rates
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Create registry (separate from default to avoid conflicts)
registry = CollectorRegistry()

# Request metrics
request_total = Counter(
    "semantic_request_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
    registry=registry
)

request_duration_seconds = Histogram(
    "semantic_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    registry=registry
)

# Pool metrics
pool_utilization_ratio = Gauge(
    "semantic_pool_utilization_ratio",
    "BlockPool utilization ratio (0.0 to 1.0)",
    registry=registry
)

# Agent metrics
agents_active = Gauge(
    "semantic_agents_active",
    "Number of hot agents currently in memory",
    registry=registry
)

# Cache metrics
cache_hit_total = Counter(
    "semantic_cache_hit_total",
    "Total number of cache operations",
    ["result"],  # "hit" or "miss"
    registry=registry
)
