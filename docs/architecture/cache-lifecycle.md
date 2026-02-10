# Cache Lifecycle

The `AgentCacheStore` manages a 3-tier cache hierarchy for per-agent KV
state. Tiers are promoted on access and evicted under memory pressure.

## Tier Definitions

| Tier | Storage | Latency | Contents |
|------|---------|---------|----------|
| Hot | In-memory dict | ~0ms | `CacheEntry` with live `AgentBlocks` |
| Warm | Disk path in dict | ~50-200ms | Safetensors file, metadata in memory |
| Cold | No reference | Full prefill | Agent must be regenerated from scratch |

## Promotion: Cold to Warm to Hot

```mermaid
sequenceDiagram
    participant Client
    participant Adapter as OpenAI Adapter
    participant Store as AgentCacheStore
    participant Disk as SafetensorsCacheAdapter
    participant Pool as BlockPool
    participant Engine as BatchEngine

    Client->>Adapter: POST /v1/chat/completions<br>agent_id=agent_1
    Adapter->>Store: load("agent_1")

    alt Hot hit
        Store-->>Adapter: CacheEntry.blocks (in-memory)
        Note over Store: metrics.hot_hits++
    else Warm hit (on disk)
        Store->>Disk: load_agent_cache("agent_1")
        Disk-->>Store: AgentBlocks from safetensors
        Store->>Store: Insert into hot cache
        Store->>Store: LRU evict if hot > max_hot_agents
        Store-->>Adapter: AgentBlocks
        Note over Store: metrics.warm_hits++<br>metrics.disk_loads++
    else Cold miss
        Store-->>Adapter: None
        Note over Store: metrics.misses++
        Adapter->>Engine: Full prefill required
        Engine->>Pool: Allocate blocks
        Pool-->>Engine: Fresh KVBlocks
    end

    Adapter->>Engine: submit(agent_id, tokens, cache)
    Engine-->>Adapter: CompletedGeneration + updated blocks
    Adapter->>Store: save("agent_1", blocks)
    Store->>Store: Upsert hot cache, mark dirty
```

## Eviction: Hot to Warm to Cold

```mermaid
sequenceDiagram
    participant Store as AgentCacheStore
    participant Disk as SafetensorsCacheAdapter
    participant Pool as BlockPool

    Note over Store: Hot cache exceeds max_hot_agents

    Store->>Store: Find LRU entry (oldest last_accessed)

    alt Entry is dirty
        Store->>Disk: save_agent_cache(agent_id, blocks)
        Disk-->>Store: Safetensors file written
        Note over Store: metrics.dirty_flushes++
    end

    Store->>Store: Remove from hot cache
    Store->>Store: Add file path to warm cache
    Note over Store: metrics.evictions++

    Note over Store: On shutdown: evict_all_to_disk()
    Store->>Disk: Flush all dirty hot entries
    Store->>Pool: Free agent blocks
```

## Prefix Matching Flow

When a new agent shares the same system prompt as an existing agent, the
`SharedPrefixCache` avoids redundant prefill for the common prefix.

```mermaid
sequenceDiagram
    participant Adapter as Inbound Adapter
    participant SPC as SharedPrefixCache
    participant Engine as BatchEngine

    Adapter->>SPC: lookup(prefix_hash)

    alt Prefix cache hit
        SPC-->>Adapter: PrefixEntry (KV state + n_tokens)
        Note over SPC: hit_count++
        Adapter->>Engine: Clone prefix KV, prefill remainder only
    else Prefix cache miss
        SPC-->>Adapter: None
        Adapter->>Engine: Full prefill
        Engine-->>Adapter: KV state for full prompt
        Adapter->>SPC: store(prefix_hash, kv_state, n_tokens)
    end
```

## Disk Format

Cache files are stored as safetensors in `~/.agent_memory/caches/`:

```
{agent_id}.safetensors       — KV cache tensors (Q4 packed uint32)
```

Metadata (stored in safetensors header):
- `model_id`, `n_layers`, `n_kv_heads`, `head_dim` (ModelTag fields)
- `kv_bits`, `kv_group_size` (quantization parameters)
- `v_head_dim` (for MLA asymmetric caches)

Atomic writes use `.tmp.safetensors` intermediary with rename-on-complete.
Orphan `.tmp` files are cleaned on adapter initialization.
