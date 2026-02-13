# Benchmark Suite

**Purpose**: Measure and document performance characteristics of the semantic caching system.

This directory contains performance benchmarks that quantify:
- Sequential vs. batched throughput (tokens/sec)
- Cache save/load performance (by token count)
- Memory utilization vs. theoretical overhead
- Real-world performance metrics

---

## Test Categories

### 1. Batching Performance (`test_batching_performance.py`)

**Goal**: Quantify throughput benefits of batching multiple agents.

Tests:
- `test_sequential_1_agent_per_model` - Baseline: sequential processing
- `test_batched_3_agents_per_model` - 3-agent batching benefit
- `test_batched_5_agents_per_model` - 5-agent batching benefit
- `test_throughput_comparison` - Generate comparison table

**Pattern**:
```python
@pytest.mark.benchmark
def test_batched_throughput(benchmark_server):
    """Measure tokens/sec with N concurrent agents."""
    start_time = time.time()

    # Execute N concurrent generations
    results = execute_concurrent_generations(num_agents=3, tokens_per_agent=500)

    # Calculate throughput
    total_tokens = sum(r.tokens_generated for r in results)
    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed

    # Record result
    record_benchmark("batched_3_agents", {
        "tokens_per_second": throughput,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed
    })
```

**Expected Results**:
- Sequential (1 agent): ~20-30 tokens/sec (baseline)
- Batched (3 agents): ~50-70 tokens/sec (1.7-2.3x speedup)
- Batched (5 agents): ~80-100 tokens/sec (2.7-3.3x speedup)

### 2. Cache Resume Speed (`test_cache_resume.py`)

**Goal**: Validate cache save/load performance across different cache sizes.

Tests:
- `test_cache_save_time_2k_4k_8k_tokens` - Save performance by size
- `test_cache_load_time_2k_4k_8k_tokens` - Load performance by size
- `test_resume_generation_speed` - First-token latency after resume
- `test_cold_start_vs_cache_resume` - Speedup comparison

**Pattern**:
```python
@pytest.mark.benchmark
def test_cache_load_time_by_size(benchmark_server):
    """Measure cache load time for different sizes."""
    results = {}

    for token_count in [2000, 4000, 8000]:
        # Create cache of specified size
        agent_id = create_agent_with_cache(token_count)

        # Save cache
        save_cache(agent_id)

        # Unload from memory
        unload_agent(agent_id)

        # Measure load time
        start = time.time()
        load_cache(agent_id)
        load_time_ms = (time.time() - start) * 1000

        results[token_count] = load_time_ms

        # Verify <500ms target
        assert load_time_ms < 500, f"{token_count} tokens: {load_time_ms:.0f}ms (target <500ms)"

    record_benchmark("cache_load_by_size", results)
```

**Expected Results**:
- 2K tokens: <200ms load time
- 4K tokens: <350ms load time
- 8K tokens: <500ms load time

### 3. Memory Utilization (`test_memory_utilization.py`)

**Goal**: Measure actual memory usage vs. theoretical predictions.

Tests:
- `test_memory_per_agent_1_5_10_agents` - Memory scaling
- `test_block_padding_overhead` - Wasted memory from padding
- `test_cache_vs_model_memory_ratio` - Cache/model memory proportion
- `test_actual_vs_theoretical_memory` - Validate predictions

**Pattern**:
```python
@pytest.mark.benchmark
def test_memory_scaling(benchmark_server):
    """Measure memory usage as agents increase."""
    import psutil

    results = []

    for num_agents in [1, 5, 10]:
        # Create N agents with caches
        agent_ids = [create_agent() for _ in range(num_agents)]

        # Generate content to fill caches
        for agent_id in agent_ids:
            generate(agent_id, tokens=1000)

        # Measure memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        results.append({
            "num_agents": num_agents,
            "memory_mb": memory_mb,
            "memory_per_agent_mb": memory_mb / num_agents
        })

    record_benchmark("memory_scaling", results)
```

**Expected Results**:
- 1 agent: Model baseline + ~50-100MB cache
- 5 agents: Model + ~250-500MB caches
- 10 agents: Model + ~500-1000MB caches

### 4. Benchmark Suite Integration (`test_benchmark_suite.py`)

**Goal**: Comprehensive benchmark report (rewritten from `benchmarks/benchmark_suite.py`).

Tests:
- All metrics from original benchmark suite
- Now using pytest + new architecture

**Pattern**:
```python
@pytest.mark.benchmark
def test_comprehensive_benchmark_suite(benchmark_server):
    """Run full benchmark suite (rewrit of benchmarks/benchmark_suite.py)."""
    results = {
        "model_load_time_sec": measure_model_load(),
        "cache_save_time_ms": measure_cache_save(),
        "cache_load_time_ms": measure_cache_load(),
        "generation_with_cache_sec": measure_generation_with_cache(),
        "throughput_tokens_per_sec": measure_throughput(),
        "memory_usage_gb": measure_memory_usage()
    }

    # Generate report
    generate_benchmark_report(results)
```

---

## Fixtures (in `conftest.py`)

### `benchmark_server`

Live server instance for benchmark testing (similar to E2E `live_server` but optimized for benchmarks).

```python
@pytest.fixture(scope="module")
def benchmark_server():
    """Start server once for all benchmarks in module."""
    # Start server (shared across benchmarks for speed)
    server_url = start_server()
    yield server_url
    # Teardown
    stop_server()
```

### `benchmark_reporter`

JSON reporter for benchmark results.

```python
@pytest.fixture
def benchmark_reporter():
    """Reporter for benchmark results."""
    reporter = BenchmarkReporter(output_file="benchmark_results.json")
    yield reporter
    reporter.save()
```

---

## Running Benchmarks

### Run All Benchmarks

```bash
pytest tests/benchmarks/ -v -m benchmark
```

### Run Specific Benchmark Suite

```bash
pytest tests/benchmarks/test_batching_performance.py -v
pytest tests/benchmarks/test_cache_resume.py -v
pytest tests/benchmarks/test_memory_utilization.py -v
```

### Skip Benchmarks in Normal Test Runs

```bash
pytest tests/ -v -m "not benchmark"
```

### Generate Benchmark Report

```bash
pytest tests/benchmarks/ -v -m benchmark --benchmark-report
```

---

## Configuration

### Environment Variables

- `SEMANTIC_BENCHMARK_MODEL`: Override model for benchmarks (default: configured model)
- `SEMANTIC_BENCHMARK_OUTPUT`: Override output file (default: `benchmark_results.json`)
- `SEMANTIC_BENCHMARK_QUICK`: Use quick mode (fewer iterations)

### Pytest Markers

```python
# Mark test as benchmark
@pytest.mark.benchmark

# Mark test as slow benchmark (>5 minutes)
@pytest.mark.slow
@pytest.mark.benchmark
```

### CI Configuration

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "benchmark: Performance benchmarks (measure throughput, latency, memory)"
]

# Skip benchmarks in CI by default (too slow)
addopts = "-m 'not benchmark'"
```

---

## Interpreting Results

### Batching Performance

**Good**:
- Sequential (1 agent): 20-30 tokens/sec
- Batched (3 agents): 50-70 tokens/sec (1.7-2.3x)
- Batched (5 agents): 80-100 tokens/sec (2.7-3.3x)

**Poor**:
- No speedup from batching (<1.2x)
- Batching slower than sequential

### Cache Performance

**Good**:
- Load time: <500ms for 8K tokens
- Save time: <200ms per agent
- Speedup: 3-5x faster than cold start

**Poor**:
- Load time: >1000ms (too slow)
- High disk I/O latency
- No speedup from caching

### Memory Utilization

**Good**:
- Memory per agent: <100MB
- Overhead: <20% vs theoretical
- Linear scaling with agent count

**Poor**:
- Memory leaks (growing over time)
- Excessive padding overhead (>30%)
- Non-linear scaling

---

## Adding New Benchmarks

### Template

```python
import pytest
import time

@pytest.mark.benchmark
def test_your_benchmark(benchmark_server, benchmark_reporter):
    """Benchmark description.

    Measures:
    - What you're measuring
    - Expected performance
    - Comparison baseline
    """
    # Setup
    setup_environment()

    # Measure
    start_time = time.time()
    result = execute_operation()
    elapsed = time.time() - start_time

    # Record
    benchmark_reporter.record("metric_name", {
        "value": result,
        "elapsed_seconds": elapsed,
        "units": "tokens/sec"
    })

    # Assert (optional performance target)
    assert result > TARGET_VALUE, f"Performance below target: {result} < {TARGET_VALUE}"
```

### Best Practices

1. **Use consistent measurements** - Same model, same hardware, same conditions
2. **Warm up before measuring** - Run one iteration to load model, then measure
3. **Repeat measurements** - Average over multiple runs for stability
4. **Document baseline** - Record expected performance for comparison
5. **Test realistic scenarios** - Use real-world token counts, agent counts
6. **Profile carefully** - Use `time.perf_counter()` for precision
7. **Isolate measurements** - Minimize external factors (network, disk I/O)
8. **Record metadata** - Model, hardware, configuration in results

---

## Performance Targets (Sprint 6)

### Batching
- 3-agent batching: >1.5x speedup vs sequential
- 5-agent batching: >2.5x speedup vs sequential

### Cache Resume
- Load time: <500ms for 8K tokens
- Save time: <200ms per agent
- Speedup: >3x vs cold start

### Memory
- Per-agent memory: <100MB
- Overhead: <20% vs theoretical
- No leaks over time

### Throughput
- Sequential: 20-30 tokens/sec
- Batched (3): 50-70 tokens/sec
- Batched (5): 80-100 tokens/sec

---

**Last Updated**: 2026-01-25 (Sprint 6 Day 5)
**Framework Version**: 1.0.0
**Dependencies**: pytest, psutil, aiohttp
