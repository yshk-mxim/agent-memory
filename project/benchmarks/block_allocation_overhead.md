# EXP-002: Block Allocation Overhead Benchmark

**Date**: 2026-01-24
**Sprint**: 2 (Day 3)
**Owner**: ML (Machine Learning Engineer)
**Status**: ✅ COMPLETE

---

## Objective

Measure BlockPool allocation/free overhead to validate < 1ms per operation (p95).

## Method

- Iterations: 1000
- Block size: 5 blocks per allocation
- Model spec: Gemma 3 12B (48 layers)
- Pool size: 1000 blocks

## Results

### Allocation

- Mean: 0.0021 ms
- p50: 0.0020 ms
- **p95: 0.0025 ms** ← Target: < 1.0 ms
- p99: 0.0029 ms
- StdDev: 0.0004 ms

### Free

- Mean: 0.0005 ms
- p50: 0.0005 ms
- **p95: 0.0006 ms** ← Target: < 1.0 ms
- p99: 0.0008 ms
- StdDev: 0.0001 ms

### Combined (allocate + free)

- **p95: 0.0031 ms** ← Target: < 1.0 ms

## Conclusion

✅ **PASSED**: Both allocation and free operations complete in < 1ms (p95).

BlockPool overhead is negligible and will not impact generation latency.
