## Model Hot-Swap Guide

Sprint 5 implementation - Dynamic model switching while preserving agent caches.

### Overview

The Semantic Cache Server supports hot-swapping between MLX models without restarting the server. Agent caches are preserved on disk and automatically reload when the original model returns.

**Constraint**: M4 Pro 24GB can only fit ONE model at a time (see ADR-007).

### Quick Start

```bash
# Swap to a different model
curl -X POST http://localhost:8000/admin/models/swap \
  -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen2.5-14B-Instruct-4bit", "timeout_seconds": 60}'

# Check currently loaded model
curl http://localhost:8000/admin/models/current \
  -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY"

# List available models
curl http://localhost:8000/admin/models/available \
  -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY"
```

### Hot-Swap Sequence

1. **Drain** - Wait for active requests to complete
2. **Evict** - Save all caches to disk (safetensors format)
3. **Shutdown** - Clear BatchEngine references
4. **Unload** - Free model memory (`del + gc + mx.clear_cache`)
5. **Load** - Load new model from HuggingFace
6. **Reconfigure** - Update BlockPool dimensions
7. **Update Tag** - Update cache store model tag
8. **Reinit** - Create new BatchEngine

**Total Time**: <30s (validated by EXP-012)

### Cache Compatibility

Caches are tagged with model dimensions:
- `n_layers` - Number of transformer layers
- `n_kv_heads` - Number of KV attention heads
- `head_dim` - Dimension per head
- `block_tokens` - Tokens per cache block

**Compatible**: Same dimensions, different model_id (e.g., Gemma-3-v1 → Gemma-3-v2)
**Incompatible**: Different dimensions - caches rejected on load

### Error Recovery

Failed swaps trigger automatic rollback:

```
Swap attempt → Failure → Unload failed model → Reload old model → Reconfigure back
```

If rollback also fails, server enters degraded state (requires manual intervention).

### Admin API

**Authentication**: Set `SEMANTIC_ADMIN_KEY` environment variable

**Endpoints**:
- `POST /admin/models/swap` - Trigger hot-swap
- `GET /admin/models/current` - Get loaded model info
- `GET /admin/models/available` - List supported models

**Swap Request**:
```json
{
  "model_id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
  "timeout_seconds": 60.0
}
```

**Swap Response**:
```json
{
  "status": "success",
  "old_model_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
  "new_model_id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
  "message": "Model swapped successfully"
}
```

### Supported Models

All MLX 4-bit quantized models on HuggingFace:
- `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` (default)
- `mlx-community/Qwen2.5-14B-Instruct-4bit`
- `mlx-community/Llama-3.1-8B-Instruct-4bit`
- `mlx-community/SmolLM2-135M-Instruct` (for testing)

### Best Practices

1. **Pre-download models**: Use `huggingface-cli download <model-id>` before swap to reduce latency
2. **Monitor swap time**: Target <30s; optimize if exceeding
3. **Verify admin key**: Keep `SEMANTIC_ADMIN_KEY` secret
4. **Test rollback**: Verify recovery works before production use
5. **Cache cleanup**: Old incompatible caches auto-rejected on load

### Troubleshooting

**Swap timeout**: Increase `timeout_seconds` if active requests need more time to complete

**Memory not reclaimed**: Verified by EXP-011 - should never happen. If it does, restart server.

**Incompatible cache**: Expected behavior - cache dimensions don't match new model. Agent regenerates.

**Rollback failure**: Critical error - check logs, may require server restart.

### Architecture

See `project/architecture/ADR-007-one-model-at-a-time.md` for detailed architecture rationale.

Components:
- **ModelRegistry** - Model lifecycle (load/unload)
- **ModelSwapOrchestrator** - Coordinates 8-step swap sequence
- **AgentCacheStore** - Cache persistence with model tagging
- **BlockPool** - Memory management with reconfiguration support
- **Admin API** - HTTP interface for model management

### Performance

**EXP-011 Results** (Memory Reclamation):
- 100% memory reclaimed after model unload
- No residual allocations
- Pattern: `del + gc.collect() + mx.clear_cache()`

**EXP-012 Expected** (Swap Latency):
- Small → Large: ~18-25s ✅
- Large → Large: ~23-30s ✅
- Large → Small: ~13-20s ✅
- Primary bottleneck: Model loading (80-90% of time)

### References

- ADR-007: One Model At A Time
- EXP-011: Memory Reclamation Validation (100% pass)
- EXP-012: Swap Latency Measurement (pending)
- Sprint 5 Summary: project/sprints/sprint_5_model_hot_swap.md
