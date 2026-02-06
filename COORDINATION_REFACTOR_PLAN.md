# Coordination Architecture Refactor Plan

**Status**: Partial — DeepSeek empty response fix applied (assistant prefix for turn 3+). DRY refactor below is still TODO.

## Root Cause
Coordination bypasses the OpenAI API layer and calls the engine directly, causing:
1. Space-stripped output ("Warden.Marcos,yourunderstanding")
2. Broken text stored in channel
3. Cascading corruption when other agents see the broken text

## Evidence
- **OpenAI API** (with identical messages): ✓ No space stripping, proper output
- **Coordination** (with identical messages): ❌ Space stripping, broken output

## Architectural Issue
```
Current (BROKEN):
┌─────────────┐
│  OpenAI API │──→ [Generation Logic A] ──→ ✓ Clean output
└─────────────┘

┌──────────────┐
│ Coordination │──→ [Generation Logic B] ──→ ❌ Broken output
└──────────────┘
```

Should be:
```
Fixed (DRY):
┌─────────────┐
│  OpenAI API │──→ [Shared Generation] ──→ ✓ Clean output
└─────────────┘        ↑
                       │
┌──────────────┐       │
│ Coordination │───────┘
└──────────────┘
```

## Refactor Steps

### 1. Extract Shared Generation Function
Create `src/semantic/application/generation_service.py`:

```python
async def generate_completion(
    engine: BlockPoolBatchEngine,
    cache_store: AgentCacheStore,
    messages: list[dict],
    agent_id: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
    **kwargs
) -> GenerationResult:
    """Shared generation logic used by both OpenAI API and Coordination.

    Returns:
        GenerationResult with clean decoded text (no space stripping)
    """
    # Tokenize using chat template
    tokens = tokenize_messages(messages, engine.tokenizer)

    # Load cache
    cached_blocks = cache_store.load(agent_id)

    # Generate
    result = await engine.generate(
        agent_id=agent_id,
        prompt_tokens=tokens,
        cache=cached_blocks,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )

    # Save cache
    if result.blocks:
        cache_store.save(agent_id, result.blocks)

    return result
```

### 2. Update OpenAI Adapter
`src/semantic/adapters/inbound/openai_adapter.py`:

```python
async def create_chat_completion(request_body, request):
    # ... validation ...

    result = await generate_completion(
        engine=batch_engine,
        cache_store=cache_store,
        messages=chat_dicts,
        agent_id=agent_id,
        max_tokens=request_body.max_tokens,
        temperature=request_body.temperature,
        # ...
    )

    return ChatCompletionsResponse(
        choices=[Choice(message={"content": result.text}, ...)],
        # ...
    )
```

### 3. Update Coordination Service
`src/semantic/application/coordination_service.py`:

```python
async def execute_turn(self, session_id: str) -> ChannelMessage:
    session = self.get_session(session_id)
    directive = self.get_next_turn(session_id)
    agent_role = session.agents[directive.agent_id]

    # Build messages
    messages = self.build_agent_prompt(directive, agent_role)

    # Generate using SHARED logic (same as OpenAI API)
    result = await generate_completion(
        engine=self._engine,
        cache_store=self._cache_store,
        messages=messages,
        agent_id=self._resolve_cache_key(session_id, directive.agent_id),
        max_tokens=self._get_generation_max_tokens(),
        temperature=1.0,
        top_p=0.95,
    )

    # Clean text (remove artifacts)
    clean_text = self._clean_agent_response(
        result.text,  # Now this is CLEAN (no space stripping)
        sender_name=agent_role.display_name,
        all_agent_names=self._all_known_agent_names(session_id),
    )

    # Store in channel (now with CLEAN text)
    message = public_channel.add_message(
        sender_id=directive.agent_id,
        content=clean_text,
        turn_number=session.current_turn,
    )

    return message
```

## Benefits
1. ✅ **DRY**: Single source of truth for generation logic
2. ✅ **No space stripping**: Both paths produce identical output
3. ✅ **Proper architecture**: Coordination built on top of shared logic
4. ✅ **Easier testing**: Test one function, both paths work
5. ✅ **Easier debugging**: One place to fix generation issues

## Implementation Order
1. Create `generation_service.py` with shared function
2. Update OpenAI adapter to use it (test that it still works)
3. Update Coordination to use it (test that space stripping is fixed)
4. Remove duplicate generation code

## Testing
- Run simple OpenAI API test → should still work
- Run coordination prisoner's dilemma → should have NO space stripping
- Compare outputs → should be identical for same inputs
