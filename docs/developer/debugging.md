# Debugging Common Issues

## Metal Memory Accumulation After Crash Cycles

**Symptom:** Server fails to load the model, Metal allocation errors, or extremely slow inference after several restart cycles.

**Root cause:** MLX allocates Metal GPU buffers that the OS is slow to reclaim from killed processes. Repeated `kill -9` or crash cycles accumulate wired kernel memory. On a 24 GB M4 Pro, wired memory has been observed reaching 17-19 GB.

**Diagnosis:**

```bash
memory_pressure | head -5
```

Look at "System-wide memory free percentage." If it reports significant pressure or wired memory exceeds 70% of total RAM, this is the issue.

You can also check with:

```bash
vm_stat | grep "Pages wired"
# Multiply the page count by 16384 (16 KB page size on ARM) to get bytes
```

**Fix:** The only reliable fix is a system reboot. After reboot:

1. Close unnecessary apps that consume GPU memory
2. Start the server before other GPU-heavy applications
3. Use graceful shutdown going forward (see below)

## Server Won't Start (Port in Use)

**Symptom:** `Address already in use` error when starting the server.

**Diagnosis:**

```bash
# Find what's using the port
lsof -ti:8000
```

**Fix:**

```bash
# If it's a previous server instance, try graceful shutdown first
PID=$(lsof -ti:8000)
kill -TERM $PID
sleep 5

# Verify it's gone
lsof -ti:8000  # Should return empty

# If still running, force kill
kill -9 $(lsof -ti:8000)
sleep 2
```

## Zombie Python Processes

**Symptom:** Port appears free but `ps` shows orphaned python processes consuming memory.

**Diagnosis:**

```bash
ps aux | grep agent_memory
ps aux | grep "python.*agent_memory"
```

**Fix:**

```bash
# Kill any orphan processes
pkill -f "agent_memory"

# Also check for Streamlit demo processes
pkill -f streamlit
```

## Graceful Shutdown Procedure

MLX allocates Metal GPU buffers that the OS is slow to reclaim. A graceful shutdown ensures all 6 cleanup stages run, preventing memory leaks.

### The shutdown sequence (triggered by SIGTERM)

When the server receives SIGTERM, the FastAPI lifespan manager (`api_server.py`) runs:

1. **Stop scheduler** -- `scheduler.stop()` halts the concurrent decode loop
2. **Drain active requests** -- `batch_engine.drain(timeout=30s)` finishes all in-flight generations via `step_once()` calls (rejects new submits)
3. **Persist caches** -- `cache_store.evict_all_to_disk()` writes all dirty hot caches to `.safetensors` files
4. **Shutdown engine** -- `batch_engine.shutdown()` clears model/tokenizer refs, active requests, agent block tracking
5. **Unload model** -- `model_registry.unload_model()` deletes model and tokenizer objects, runs `gc.collect()`
6. **Release GPU memory** -- `block_pool.force_clear_all_allocations()` nulls all `block.layer_data` tensors, then `gc.collect()` x2 + `mx.clear_cache()` releases Metal buffer cache back to the OS

### Standard shutdown

```bash
# 1. Find the server PID
PID=$(lsof -ti:8000)

# 2. Send SIGTERM (triggers graceful shutdown)
kill -TERM $PID

# 3. Wait for shutdown to complete
#    Watch for "server_shutdown_complete" in logs. Typically 5-10 seconds.
sleep 5

# 4. Verify the port is free
lsof -ti:8000  # Should return empty
```

### If SIGTERM doesn't work (server hangs)

```bash
# Wait up to 10 seconds for graceful shutdown
sleep 10

# Force kill as last resort (skips cleanup -- may leak Metal memory)
kill -9 $(lsof -ti:8000)
sleep 2

# Clean up orphan cache files
rm -f ~/.agent_memory/caches/*.tmp.safetensors
```

### Full cleanup between model switches (benchmarking)

```bash
# Kill server
PID=$(lsof -ti:8000) && kill -TERM $PID && sleep 5

# Verify dead
[ -z "$(lsof -ti:8000)" ] || { kill -9 $(lsof -ti:8000); sleep 2; }

# Remove all cached KV data (forces cold start)
rm -rf ~/.agent_memory/caches/*.safetensors ~/.agent_memory/caches/*.tmp

# If wired memory is too high (check with: memory_pressure | head -5)
# the only fix is a system reboot
```

### Common pitfalls

- **Never `kill -9` as first choice** -- skips all 6 cleanup stages, leaks Metal buffers
- **Don't restart rapidly** -- give 5-10 seconds between stop and start for Metal to reclaim
- **Streamlit demo also holds a port** -- kill it too: `pkill -f streamlit`
- **Zombie python processes** -- check with `ps aux | grep agent_memory` after shutdown
- **Wired memory accumulation** -- after many crash cycles, only a reboot frees it

## Checking Wired Memory Pressure

Use these commands to assess system memory state before and after server operations:

```bash
# Quick summary of memory pressure
memory_pressure | head -5

# Detailed page statistics
vm_stat

# Watch memory in real time (updates every 2 seconds)
vm_stat 2

# Check for Metal-related processes holding memory
ps aux | grep -E "(Metal|mlx|agent_memory)" | grep -v grep
```

On Apple Silicon, the "Pages wired down" count in `vm_stat` multiplied by 16384 gives wired bytes. If this exceeds ~70% of total RAM after server shutdown, consider rebooting before the next development cycle.

## SIGSEGV / Metal Crashes

The CLI enables `faulthandler` at startup to print Python tracebacks on SIGSEGV, SIGABRT, and SIGBUS. If you see a Metal crash:

1. Check if it is a known thread-safety issue (see [mlx-notes.md](mlx-notes.md))
2. Verify `mlx_io_lock` is held around any cross-thread MLX operations
3. Confirm `mx.eval()` and `mx.synchronize()` are called after lazy operations that cross stream boundaries
4. Check that `mx.async_eval` is not used anywhere (should be `mx.eval` only)

## Cache Corruption

**Symptom:** Model produces garbage output on cache-hit requests but works on cold starts.

**Possible causes:**

- Cache was saved by a different model or quantization config. `ModelTag` includes `kv_bits` and `kv_group_size` for compatibility checking.
- Orphan `.tmp.safetensors` files from interrupted saves. Clean up: `rm -f ~/.agent_memory/caches/*.tmp.safetensors`
- Sliding window mask bug (fixed): `QuantizedKVCache.make_mask` was ignoring `window_size` for chunked prefill. If you see this pattern, verify the patch in `mlx_quantized_extensions.py` is intact.

**Recovery:**

```bash
# Remove all cached KV data
rm -rf ~/.agent_memory/caches/*.safetensors ~/.agent_memory/caches/*.tmp
```
