# Request Flow

End-to-end flow for a chat completion request, from HTTP to token generation
and streaming response.

## Non-Streaming Request

```mermaid
sequenceDiagram
    participant Client
    participant OA as OpenAI Adapter
    participant CCS as Chat Completion Service
    participant CTA as Chat Template Adapter
    participant Store as AgentCacheStore
    participant Sched as Concurrent Scheduler
    participant Engine as Batch Engine
    participant Pool as BlockPool
    participant MLX as MLX Runtime

    Client->>OA: POST /v1/chat/completions<br>{messages, max_tokens, agent_id}

    OA->>CCS: generate_chat_completion(messages, agent_id, ...)
    CCS->>CTA: apply_chat_template(messages)
    CTA-->>CCS: prompt_tokens (list[int])

    CCS->>Store: load(agent_id)
    Store-->>CCS: AgentBlocks | None

    CCS->>Sched: submit_and_wait(agent_id, tokens, cache, max_tokens)
    Note over Sched: Enqueues SchedulerRequest

    rect rgb(240, 240, 255)
        Note over Sched,MLX: Scheduler worker thread
        Sched->>Engine: submit(agent_id, tokens, cache)

        alt No existing cache
            Engine->>Pool: allocate(n_blocks)
            Pool-->>Engine: KVBlocks
            Engine->>MLX: Prefill (full prompt)
        else Has cache (hot/warm)
            Engine->>MLX: Prefill (new tokens only)
        end

        loop Until EOS or max_tokens
            Engine->>MLX: Decode one token
            MLX-->>Engine: next_token_id
            Engine->>Engine: Append to output
        end

        Engine->>Engine: Extract updated KV blocks
        Engine-->>Sched: CompletedGeneration
    end

    Sched-->>CCS: CompletedGeneration
    CCS->>Store: save(agent_id, updated_blocks)
    CCS-->>OA: {text, token_count, finish_reason}
    OA-->>Client: ChatCompletionsResponse (JSON)
```

## Streaming Request (SSE)

```mermaid
sequenceDiagram
    participant Client
    participant OA as OpenAI Adapter
    participant CCS as Chat Completion Service
    participant Sched as Concurrent Scheduler
    participant Engine as Batch Engine
    participant MLX as MLX Runtime

    Client->>OA: POST /v1/chat/completions<br>{stream: true}

    OA->>CCS: generate_chat_completion(...)
    CCS->>Sched: submit_and_stream(agent_id, tokens, cache)
    Note over Sched: Creates token_queue (asyncio.Queue)

    rect rgb(240, 240, 255)
        Note over Sched,MLX: Scheduler worker thread
        loop Each decode step
            Engine->>MLX: Decode one token
            MLX-->>Engine: next_token_id
            Engine->>Sched: Put StreamDelta on token_queue
        end
        Engine->>Sched: Put None (sentinel) on token_queue
    end

    loop Until sentinel
        Sched-->>OA: StreamDelta from token_queue
        OA-->>Client: SSE: data: {"choices":[{"delta":{"content":"..."}}]}
    end

    OA-->>Client: SSE: data: [DONE]
```

## Scheduler Interleaving

When a long prompt arrives while another sequence is actively decoding,
the scheduler interleaves prefill chunks with decode steps.

```mermaid
sequenceDiagram
    participant S as Scheduler Thread

    Note over S: Seq A: actively decoding<br>Seq B: 4096-token prompt arrives

    S->>S: Decode step (Seq A, 1 token)
    S->>S: Prefill chunk (Seq B, 256 tokens)
    S->>S: Decode step (Seq A, 1 token)
    S->>S: Prefill chunk (Seq B, 256 tokens)
    Note over S: ... repeat until Seq B prefill complete ...
    S->>S: Decode step (Seq A + Seq B, 1 token each)
    Note over S: Both sequences now in decode phase
```

## Chunked Prefill Sizing

The `BatchEngine` uses adaptive chunk sizes based on current cache position:

| Cache Position | Chunk Size | Rationale |
|---------------|------------|-----------|
| 0 - 2000 tokens | 4096 | Aggressive: plenty of GPU headroom |
| 2000 - 8000 tokens | 2048 | Moderate: growing memory pressure |
| 8000 - 20000 tokens | 1024 | Conservative: avoid OOM |
| 20000+ tokens | 512 | Minimal: near memory ceiling |
