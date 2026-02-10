# Hexagonal Architecture

The codebase follows a hexagonal (ports and adapters) architecture. The
application core contains all business logic and depends only on abstract
ports. Inbound adapters translate external protocols into application calls.
Outbound adapters implement infrastructure concerns behind port interfaces.

## Layer Diagram

```mermaid
graph TB
    subgraph Inbound Adapters ["Inbound Adapters (driving)"]
        direction TB
        OA["OpenAI Adapter<br>POST /v1/chat/completions<br>Streaming + non-streaming"]
        AN["Anthropic Adapter<br>POST /v1/messages<br>(non-functioning)"]
        CO["Coordination Adapter<br>POST /v1/coordinate<br>Multi-agent sessions"]
        DA["Direct Agent Adapter<br>Lower-level agent API"]
        AD["Admin API<br>Health, metrics, model swap"]
        AH["Adapter Helpers<br>Tokenization, state access"]
    end

    subgraph Application ["Application Core (business logic)"]
        direction TB
        COS["Coordination Service<br>Session lifecycle, turn sequencing,<br>prompt building, debate/vote"]
        CCS["Chat Completion Service<br>Shared generation: tokenize,<br>cache lookup, submit, save"]
        SCH["Concurrent Scheduler<br>Worker thread, request queue,<br>interleaved prefill/decode"]
        BPE["Batch Engine<br>Block alloc, KV reconstruction,<br>submit/step/drain/shutdown"]
        ACS["Agent Cache Store<br>3-tier LRU, ModelTag validation,<br>dirty tracking, eviction"]
        SPC["Shared Prefix Cache<br>System prompt KV reuse"]
        MRG["Model Registry<br>Load/unload lifecycle"]
        MSO["Model Swap Orchestrator<br>Hot-swap coordination"]
        PS["Prefill State<br>Chunked prefill tracking"]
    end

    subgraph Domain ["Domain (entities + value objects)"]
        direction TB
        BP["BlockPool<br>Fixed-size block allocation"]
        AB["AgentBlocks<br>Per-agent KV block list"]
        KB["KVBlock<br>Layer tensors + metadata"]
        MCS["ModelCacheSpec<br>Layer count, dims, quant params"]
        CG["CompletedGeneration<br>Text + blocks + finish reason"]
        SD["StreamDelta<br>Per-token streaming unit"]
        MT["ModelTag<br>Cache compatibility check"]
    end

    subgraph Outbound Adapters ["Outbound Adapters (driven)"]
        direction TB
        MCA["MLX Cache Adapter<br>Tensor concatenation, splitting"]
        SFA["Safetensors Adapter<br>Disk persistence, atomic writes"]
        CTA["Chat Template Adapter<br>Model-specific prompt formatting"]
        QEX["Quantized Extensions<br>BatchQuantizedKVCache,<br>merge/extend/make_mask patches"]
        FAT["Fused Attention<br>Q4 SDPA monkeypatches,<br>GQA broadcast fix"]
        SEX["Spec Extractor<br>Model introspection,<br>MLA detection"]
        MLO["MLX Model Loader<br>HuggingFace download + load"]
        SNK["Sink Compat<br>SDPA capture patch"]
    end

    subgraph Infrastructure ["Infrastructure"]
        direction TB
        MLX["MLX / Metal GPU"]
        DSK["Disk (~/.agent_memory/caches/)"]
        HUB["HuggingFace Hub"]
    end

    OA & AN & CO & DA --> CCS
    CO --> COS
    AD --> MRG & MSO
    AH -.-> OA & AN & CO & DA

    COS --> CCS
    CCS --> SCH
    CCS --> ACS
    SCH --> BPE
    BPE --> BP
    BPE --> ACS
    ACS --> SFA
    MRG --> MLO
    MSO --> MRG & SCH & BPE

    MCA --> MLX
    QEX --> MLX
    FAT --> MLX
    SFA --> DSK
    MLO --> HUB
    SEX --> MLX

    BPE -.-> MCA & CTA & QEX & FAT
    ACS -.-> MCA

    style Application fill:#e8f4e8,stroke:#2d7d2d
    style Domain fill:#e8e8f4,stroke:#2d2d7d
    style Inbound Adapters fill:#f4e8e8,stroke:#7d2d2d
    style Outbound Adapters fill:#f4f4e8,stroke:#7d7d2d
    style Infrastructure fill:#f0f0f0,stroke:#666
```

## Port Interfaces

The application core defines ports (abstract interfaces) that adapters
implement. Key ports:

| Port | Direction | Implementor |
|------|-----------|------------|
| `ChatTemplatePort` | Outbound | `ChatTemplateAdapter` |
| `GenerationEnginePort` | Outbound | `BlockPoolBatchEngine` |
| Cache persistence | Outbound | `SafetensorsCacheAdapter` |
| Cache tensor ops | Outbound | `MLXCacheAdapter` |
| HTTP API | Inbound | OpenAI/Anthropic/Coordination adapters |

## Dependency Rule

Dependencies point inward: adapters depend on the application core, the
application core depends on domain entities and value objects. No domain
object imports from adapters or application services.

```
Inbound Adapters --> Application Core --> Domain <-- Outbound Adapters
                                                         |
                                                    Infrastructure
```

## Package Layout

```
src/agent_memory/
    adapters/
        inbound/        # HTTP adapters, middleware, request models
        outbound/       # MLX, safetensors, chat template adapters
        config/         # Settings, logging
    application/        # Services, scheduler, engine, cache store
    domain/             # Entities, value objects, errors, ports
    entrypoints/        # CLI, FastAPI app factory
```
