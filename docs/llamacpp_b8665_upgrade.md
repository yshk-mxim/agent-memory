# Upgrading llama.cpp to b8665 on Thor

> Required for Gemma 4 native tool calling and quality fixes.
> Previous build: b8660 (April 3, 2026). Target: b8665 (April 4, 2026).

## Why upgrade

b8661–b8665 are **all Gemma 4 fixes**:

| Build | PR | Fix |
|-------|----|-----|
| b8665 | [#21418](https://github.com/ggml-org/llama.cpp/pull/21418) | **Dedicated Gemma 4 tool call parser** — native `<\|tool_call\>call:fn{...}<tool_call\|>` handling, fixes infinite loops, `<unused24>` spam, array param serialization |
| b8664 | — | Server timing fix |
| b8663 | — | Respect specified tag, fallback when empty |
| b8662 | [#21390](https://github.com/ggml-org/llama.cpp/pull/21390) | Read `final_logit_softcapping` from Gemma 4 GGUF — fixes generation quality |
| b8661 | [#21406](https://github.com/ggml-org/llama.cpp/pull/21406) | Custom newline split for Gemma 4 tokenizer |

Already in b8660 (no rebuild needed for these):
- [#21326](https://github.com/ggml-org/llama.cpp/pull/21326) — Template parser fixes (array-style JSON Schema types)
- [#21343](https://github.com/ggml-org/llama.cpp/pull/21343) — Tokenizer `\n\n` split fix

## Build steps (on Thor)

```bash
ssh <user>@<thor-hostname>

# Stop running llama-server
pkill -f llama-server

# Update source
cd ~/llama.cpp-build
git fetch --tags
git checkout b8665

# Rebuild (sm_110 only)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="110" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server

# Verify
./build/bin/llama-server --version
# Expected: version b8665 or similar
```

## Launch with new flags

Key changes from b8660 launch:
- Add `--reasoning off` to suppress `<|channel>thought` tokens (replaces unreliable `chat_template_kwargs`)
- Add `--jinja` for template processing (required for native tool calling)

### Gemma 4 26B-A4B (MoE, fast)

```bash
~/llama.cpp-build/build/bin/llama-server \
    -m ~/models/gemma4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    -ngl 999 \
    --ctx-size 262144 \
    -np 4 \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-prompt \
    --jinja \
    --reasoning off \
    -fa on \
    -b 4096 \
    -ub 1024
```

### Gemma 4 31B (Dense, deep reasoning)

```bash
~/llama.cpp-build/build/bin/llama-server \
    -m ~/models/gemma4-31b/gemma-4-31B-it-Q4_K_M.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    -ngl 999 \
    --ctx-size 131072 \
    -np 2 \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-prompt \
    --jinja \
    --reasoning off \
    -fa on \
    -b 4096 \
    -ub 1024
```

## Verify

```bash
# Health check
curl http://localhost:8001/health
# {"status":"ok"}

# Test tool calling (should return native tool_calls in JSON)
curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-26b-a4b",
        "messages": [{"role": "user", "content": "What is the weather in London?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "max_tokens": 256
    }' | python3 -m json.tool
```

With b8665, tool calls should appear in `message.tool_calls` as structured JSON,
not as raw text in `message.content`. The agent-memory parser chain handles both
paths — native tool_calls are preferred when available.

## Config changes (already applied)

These changes were made in the agent-memory repo:

| File | Change |
|------|--------|
| `config/models/gemma-4-26b-a4b.toml` | `top_k` 40→64, `extra_args = ["--reasoning", "off"]` |
| `config/models/gemma-4-31b.toml` | `top_k` 40→64, `extra_args = ["--reasoning", "off"]` |
| `llamacpp_backend_adapter.py` | Removed `chat_template_kwargs`; now passes `openai_tools` to server for grammar-constrained generation |

## Known issues after upgrade

- **`<unused24>` token spam** ([#21321](https://github.com/ggml-org/llama.cpp/issues/21321)):
  Partially fixed by #21418 output constraints, but issue remains open.
  If observed, restart llama-server and clear slot caches.

- **Audio modality** ([#21325](https://github.com/ggml-org/llama.cpp/issues/21325)):
  Not yet supported in llama.cpp.

- **Native tool calling docs**: llama.cpp's `function-calling.md` does not yet
  list Gemma 4 as a supported model, even though the code support landed in b8665.
