# Comprehensive Survey of Existing Local LLM Tools
## LM Studio, Ollama, and llama.cpp Feature Comparison

**Survey Date:** January 2026
**Purpose:** Research existing capabilities for multi-agent support, KV cache management, session persistence, and Mac/Apple Silicon optimizations.

---

## Executive Summary

| Feature Category | LM Studio | Ollama | llama.cpp |
|-----------------|-----------|---------|-----------|
| **Multi-Agent Native Support** | ❌ No | ❌ No | ❌ No |
| **KV Cache Exposed** | ✅ Partial | ✅ Yes | ✅ Yes |
| **Session Persistence** | ✅ Conversations only | ❌ No native support | ✅ Via API |
| **Multi-User Support** | ⚠️ Limited | ⚠️ Queuing only | ⚠️ Shared context |
| **Apple Silicon Optimization** | ✅ Excellent (MLX+Metal) | ✅ Good (Metal) | ✅ Good (Metal) |
| **Prompt Caching** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 1. Multi-Agent Support

### LM Studio
**Capabilities:**
- ❌ No native multi-agent orchestration
- ✅ Multi-model serving: Can run "Multi Model Session" in Playground tab
- ✅ Integration support: Works with AutoGen, smolagents, CrewAI via API
- ✅ `.act()` API for agent-like behavior with tool calling and structured outputs

**Context Isolation:**
- ❌ No per-agent context isolation built-in
- ⚠️ Multi-agent capabilities only through external frameworks

**Agent Switching:**
- ✅ Can load multiple models simultaneously
- ⚠️ Limited concurrent request handling compared to Ollama
- ⚠️ API only runs while LM Studio desktop app is open

**Sources:**
- [LM Studio multi-model inference server support](https://github.com/Mintplex-Labs/anything-llm/issues/1569)
- [AutoGen integration with LM Studio](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/local-lm-studio/)
- [Building AI Agents with smolagents and LM Studio](https://www.matt-adams.co.uk/2025/03/14/smolagents-lmstudio.html)
- [CrewAI loading multiple LLMs discussion](https://community.crewai.com/t/loading-multiple-llm-for-multiple-agent/3613)

### Ollama
**Capabilities:**
- ❌ No native multi-agent orchestration
- ✅ Can run multiple models simultaneously (via multiple terminal sessions)
- ✅ Integration support: Works with LangGraph, CrewAI, MetaGPT, multiagent-orchestrator
- ✅ Better concurrent request handling than LM Studio

**Context Isolation:**
- ❌ No per-agent context isolation
- ⚠️ Each model runs independently but no built-in agent management

**Agent Switching:**
- ✅ Simple model switching via API
- ✅ Parallel processing with `OLLAMA_NUM_PARALLEL` environment variable
- ⚠️ Future updates expected to improve orchestration and context management

**Sources:**
- [Building Multi-Agent Systems with LangGraph and Ollama](https://medium.com/@diwakarkumar_18755/building-multi-agent-systems-with-langgraph-and-ollama-architectures-concepts-and-code-383d4c01e00c)
- [How to Build Multi-Agent System with CrewAI and Ollama](https://www.analyticsvidhya.com/blog/2024/09/build-multi-agent-system/)
- [How to Run Multiple Ollama Models Together](https://www.elightwalk.com/blog/run-multiple-ollama-models)
- [Multi-agent PRD automation with MetaGPT, Ollama, and DeepSeek](https://www.ibm.com/think/tutorials/multi-agent-prd-ai-automation-metagpt-ollama-deepseek)

### llama.cpp
**Capabilities:**
- ❌ No native multi-agent orchestration
- ✅ Slots system for parallel request handling
- ✅ Router mode for dynamic model loading/unloading (recent feature)
- ✅ llama-agent framework available (third-party)

**Context Isolation:**
- ⚠️ Uses shared context architecture: one `llama_context` + `llama_batch` handles multiple clients
- ✅ Per-user session isolation via cache and session management
- ⚠️ Context must be allocated for total tokens needed (e.g., 2 users × 15k = 30k ctx size)

**Agent Switching:**
- ✅ Router mode allows dynamic model switching without restart
- ✅ Can reuse same model with multiple context sizes
- ✅ Session management APIs to list/delete sessions

**Sources:**
- [Use model for multiple user sessions discussion](https://github.com/abetlen/llama-cpp-python/discussions/1730)
- [llama-agent GitHub repository](https://github.com/gary149/llama-agent)
- [New in llama.cpp: Model Management](https://huggingface.co/blog/ggml-org/model-management-in-llamacpp)
- [Tutorial: Offline Agentic coding with llama-server](https://github.com/ggml-org/llama.cpp/discussions/14758)

---

## 2. KV Cache Management

### LM Studio
**Cache Exposure:**
- ✅ KV cache quantization support (requires llama.cpp/1.9.0+)
- ✅ GPU offload toggle: "Offload KV Cache to GPU Memory"
- ⚠️ Some reported bugs with KV cache offload in recent versions (0.3.23)

**Persistence:**
- ❌ No native KV cache persistence to disk
- ⚠️ Cache persists between tasks during runtime (can cause unpredictable behavior)
- ❌ KV cache cleared when model unloaded

**Reuse:**
- ✅ In-memory reuse during active session
- ✅ Prompt caching: When large document loaded, it's cached for subsequent requests

**Eviction Policies:**
- ✅ **TTL (Time-To-Live)**: Models auto-unload after idle period (default: 60 minutes)
- ✅ **Auto-Evict**: JIT-loaded models automatically evict previous JIT-loaded model (default: ON)
- ✅ Configurable via Developer Settings

**Sources:**
- [LM Studio 0.3.7 KV Cache quantization](https://lmstudio.ai/blog/lmstudio-v0.3.7)
- [Offload KV Cache to GPU Memory issue](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/900)
- [How to Increase Context Length in LM Studio](https://localllm.in/blog/lm-studio-increase-context-length)
- [Idle TTL and Auto-Evict documentation](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict)

### Ollama
**Cache Exposure:**
- ✅ Full KV cache system with quantization support
- ✅ `OLLAMA_KV_CACHE_TYPE` environment variable (f16, q8_0, q4_0)
- ✅ q8_0: ~1/2 memory of f16; q4_0: ~1/4 memory of f16
- ✅ Flash Attention support for quantization

**Persistence:**
- ❌ No native disk persistence
- ✅ In-memory cache reuse for shared prefixes (set `cache_prompt = true`)
- ❌ Cache cleared when model unloaded

**Reuse:**
- ✅ KV cache reuse across requests with shared prefixes
- ✅ Defragmentation when cache fills up
- ✅ Limits for historical tokens per forward pass

**Eviction Policies:**
- ✅ **Default keep_alive**: 5 minutes
- ✅ Configurable via API parameter or `OLLAMA_KEEP_ALIVE` environment variable
- ✅ Values: duration string ("10m", "24h"), "0" (immediate unload), negative (keep indefinitely)
- ✅ Memory-based eviction: When limited, models queued until space available; idle models unloaded

**Sources:**
- [KV Cache System documentation](https://deepwiki.com/ollama/ollama/5.3-kv-cache-system)
- [Bringing K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [KV Cache Quantization GitHub issue](https://github.com/ollama/ollama/issues/5091)
- [Ollama FAQ - Model unloading](https://docs.ollama.com/faq)
- [Ollama Models Management: Auto-Unloading Features](https://www.arsturn.com/blog/managing-ollama-models-auto-unloading-features-explained)

### llama.cpp
**Cache Exposure:**
- ✅ Full KV cache management and configuration
- ✅ `cache_prompt` parameter for cache reuse (default: true)
- ✅ `--slot-save-path` for slot persistence
- ✅ `n_cache_reuse` parameter for KV shifting

**Persistence:**
- ✅ **Slot Persistence**: Save/restore KV cache to disk (.bin files)
- ✅ API functions: `llama_set_state_data()` / `llama_get_state_data()`
- ⚠️ Feature exists in code but not exposed in WebUI (as of Nov 2025)
- ❌ Cache discarded when model unloaded

**Reuse:**
- ✅ Excellent prefix cache reuse with `cache_prompt = true`
- ✅ Tutorial available for KV cache reuse with llama-server
- ✅ Store KV cache for large documents, restore later to reduce latency

**Eviction Policies:**
- ✅ LRU-like eviction in server logs
- ✅ Reusable blocks evicted based on LRU when memory needed
- ✅ Unified KV buffer shared across sequences (default)
- ✅ Maximum cache size setting (default: 8192 MiB)

**Sources:**
- [Feature Request: KV Cache Persistence on Disk](https://github.com/ggml-org/llama.cpp/issues/17107)
- [Tutorial: KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [Does the KV cache persist across multiple requests?](https://github.com/ggml-org/llama.cpp/discussions/8860)
- [How to cache system prompt discussion](https://github.com/ggml-org/llama.cpp/discussions/8947)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

---

## 3. Session Persistence

### LM Studio
**Conversation Saving:**
- ✅ **Native conversation storage**: JSON format
- ✅ Multiple conversation threads with folder organization
- ✅ Location: `C:\Users\USER\.cache\lm-studio\conversations` (Windows)
- ✅ Right-click to "Reveal in Finder" / "Show in File Explorer"

**History Storage:**
- ✅ Text conversation history persisted
- ❌ KV cache NOT saved (only conversation text)
- ⚠️ Doesn't handle symbolic links well

**Cross-Session Resume:**
- ✅ Conversations can be resumed (text history)
- ❌ KV cache must be regenerated
- ✅ Third-party: Long-Term Memory MCP for persistent memory

**Third-Party Solutions:**
- ✅ **Long-Term Memory MCP**: SQLite + ChromaDB hybrid
  - Cross-chat continuity
  - Cross-model continuity
  - Automatic backups (daily + every 100 memories)
  - Designed for decades-long use

**Sources:**
- [Chat Conversation folder location issue](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/553)
- [Manage chats documentation](https://lmstudio.ai/docs/app/basics/chat)
- [Robust Long-Term Memory MCP](https://github.com/Rotoslider/long-term-memory-mcp)
- [Loading chat history from LM Studio discussion](https://github.com/lmstudio-ai/lmstudio-js/issues/152)

### Ollama
**Conversation Saving:**
- ❌ **No native session persistence**
- ❌ Conversations not automatically saved between terminal sessions
- ⚠️ Must be implemented externally

**History Storage:**
- ❌ No built-in conversation history storage
- ✅ Manual implementation possible: Save to JSON files
- ✅ Web UIs (e.g., ollama-ui) use browser localStorage

**Cross-Session Resume:**
- ❌ Not supported natively
- ✅ External solution: Reload chat history from files and append to prompts
- ✅ Some web interfaces implement their own persistence

**Implementation Approaches:**
- Manual: Serialize message list to JSON
- Web UI: localStorage for chat data, context tokens, system prompts
- Recent user requests for `--save` flag (not implemented as of 2026)

**Sources:**
- [How to save chat history discussion](https://github.com/ollama/ollama/issues/8576)
- [How to save a session with local LLM via Ollama](https://tekonto.com/how-to-save-a-session-of-chat-with-local-llm-via-ollama-and-make-the-llm-remember-it-in-the-next-session-after-a-restart/)
- [Persistent chat memory issue](https://github.com/ollama/ollama-python/issues/242)
- [Chat History Management in ollama-ui](https://deepwiki.com/ollama-ui/ollama-ui/7-chat-history-management)

### llama.cpp
**Conversation Saving:**
- ✅ **Slot Persistence API**: Save/restore sessions including KV cache
- ✅ `/slots/?action=save` and `/slots/?action=restore` endpoints
- ✅ Store separate .bin files for different sessions (A.bin, B.bin, etc.)

**History Storage:**
- ✅ Can save complete inference state (KV cache + context)
- ✅ `llama_set_state_data()` / `llama_get_state_data()` functions
- ⚠️ Works in underlying code; not in WebUI (as of Nov 2025)

**Cross-Session Resume:**
- ✅ **Excellent support**: Full state restoration
- ✅ Reuse KV cache after model reload
- ✅ Save processed document state, load later for queries
- ⚠️ Only works when model stays loaded; cleared on unload

**Advanced Features:**
- ✅ System prompt caching across slots
- ✅ Session tokens save/restore
- ✅ `--slot-save-path` flag for persistence directory

**Sources:**
- [Feature Request: KV Cache Persistence on Disk](https://github.com/ggml-org/llama.cpp/issues/17107)
- [Tutorial: KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [How to cache system prompt discussion](https://github.com/ggml-org/llama.cpp/discussions/8947)
- [Does Docker Model Runner Use Token Caching?](https://www.ajeetraina.com/does-docker-model-runner-use-token-caching-a-deep-dive-into-llama-cpp-integration/)

---

## 4. Multi-Tenant / Multi-User Support

### LM Studio
**Concurrent Users:**
- ⚠️ **Limited multi-tenant support**
- ✅ Multi-user collaboration features for businesses/teams
- ❌ Desktop-focused design (not production multi-tenant server)

**Context Isolation:**
- ⚠️ Single tool call at a time by default (SDK 1.4.0+)
- ❌ No built-in per-user context isolation
- ⚠️ API only runs while desktop app is open

**Per-User Cache:**
- ❌ No dedicated per-user cache isolation
- ⚠️ Designed for single-user or low concurrency scenarios

**Production Suitability:**
- ❌ Not ideal for thousands of concurrent users
- ✅ Suitable for: internal bots, few users, batch jobs
- ⚠️ Better alternatives exist for production multi-tenant (vLLM, LiteLLM proxy)

**Sources:**
- [LM Studio Enterprise](https://lmstudio.ai/work)
- [Multi-Tenant Architecture with LiteLLM](https://docs.litellm.ai/docs/proxy/multi_tenant_architecture)
- [vLLM vs Ollama comparison](https://www.codiste.com/vllm-vs-ollama)
- [Concurrent Requests on single runtime discussion](https://github.com/lmstudio-ai/mlx-engine/issues/203)

### Ollama
**Concurrent Users:**
- ✅ **Concurrency enabled by default** (since v0.2)
- ✅ Parallel requests using shared memory
- ✅ `OLLAMA_NUM_PARALLEL` controls parallel request count
- ✅ Request batching when multiple requests arrive simultaneously

**Context Isolation:**
- ⚠️ **No built-in multi-tenant isolation**
- ✅ FIFO queuing when parallelism exceeded
- ✅ `OLLAMA_MAX_QUEUE` controls queue size (default: 512)
- ❌ 503 error when queue full

**Per-User Cache:**
- ❌ No per-user cache isolation
- ⚠️ Environment variable: `OLLAMA_MULTIUSER_CACHE:false` (as of Jan 2026)
- ⚠️ Not designed for strict resource isolation

**Production Suitability:**
- ✅ Good for: Single-user, low concurrency, internal tools
- ❌ Limited for: High throughput, thousands of concurrent users, public APIs
- ⚠️ Workaround: Multiple Ollama instances with load balancer

**Sources:**
- [How to serve multiple simultaneous requests](https://github.com/ollama/ollama/issues/1400)
- [Ollama vs. vLLM: Why Ollama is Slow for Multiple Users](https://www.arsturn.com/blog/ollama-vs-homl-the-real-reason-ollama-is-slower-for-multiple-users)
- [Enhanced Concurrency Features in Ollama](https://www.linkedin.com/pulse/enhanced-concurrency-features-ollamas-latest-update-robyn-le-sueur-bh7rf)
- [How Ollama Handles Parallel Requests](https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/)

### llama.cpp
**Concurrent Users:**
- ✅ **Shared context architecture**: One `llama_context` + `llama_batch` for multiple clients
- ✅ Slots system: `--parallel` (or `-np`) for concurrent request slots
- ✅ Continuous batching enabled by default
- ✅ Multiple users can share same model instance

**Context Isolation:**
- ⚠️ **Context shared, not isolated per user**
- ✅ Mutex protection: Requests answered sequentially
- ✅ Each user session cached separately
- ⚠️ Must allocate total context size upfront (users × context_per_user)

**Per-User Cache:**
- ✅ Context cached per user session
- ✅ Unified KV buffer shared across sequences (efficient)
- ✅ Session stickiness via `X-Session-ID` headers

**Production Suitability:**
- ✅ Efficient for multiple users with shared resources
- ⚠️ Not as robust as vLLM for multi-tenant isolation
- ✅ Suitable for: 500 users or 1 user (no difference in isolation design)

**Sources:**
- [Use model for multiple user sessions](https://github.com/abetlen/llama-cpp-python/discussions/1730)
- [Are llama_context and llama_batch shared by multiple users?](https://github.com/ggml-org/llama.cpp/discussions/8175)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Parallelization / Batching Explanation](https://github.com/ggml-org/llama.cpp/discussions/4130)

---

## 5. Mac / Apple Silicon Specific Features

### LM Studio
**UMA Optimizations:**
- ✅ **Excellent Apple Silicon support**
- ✅ MLX engine support (Apple's official framework)
- ✅ Unified memory: 3× faster model loading, 26-30% higher token speeds
- ✅ Lower RAM consumption vs competitors

**Metal Backend:**
- ✅ Full Metal acceleration via llama.cpp
- ✅ Efficient GPU offload with visual RAM usage feedback
- ✅ Simple GPU layer offload slider

**Memory Management:**
- ✅ MLX leverages unified memory (CPU/GPU access without copying)
- ✅ Eliminates major bottleneck for data transfer
- ✅ MLX models more performant and memory-efficient than others on Mac

**Platform Support:**
- ✅ M1, M2, M3, M4 chips supported
- ✅ System Requirements clearly documented

**Sources:**
- [LM Studio 0.3.4 ships with Apple MLX](https://lmstudio.ai/blog/lmstudio-v0.3.4)
- [Configure LM Studio for Apple Silicon](https://medium.com/@ingridwickstevens/configure-lm-studio-on-apple-silicon-with-87-7-faster-generations-b713cb4de6d4)
- [Local AI with MLX on Mac - practical guide](https://www.markus-schall.de/en/2025/09/mlx-on-apple-silicon-as-local-ki-compared-with-ollama-co/)
- [LM Studio System Requirements](https://lmstudio.ai/docs/app/system-requirements)

### Ollama
**UMA Optimizations:**
- ✅ **Good Apple Silicon support**
- ✅ Unified memory: 16GB Mac = 16GB for CPU+GPU shared pool
- ✅ Enables local simulation of enterprise-class GPUs (lower speed/bandwidth)

**Metal Backend:**
- ✅ Built-in Metal support via llama.cpp
- ✅ No additional steps required
- ✅ 3-5× performance improvement vs CPU-only
- ✅ 2-4× better throughput with quantized GGUF models

**Memory Management:**
- ⚠️ **Metal allocates only ~75% of system RAM** for GPU operations (macOS driver limit)
- ✅ Key environment variables:
  - `OLLAMA_MAX_RAM`: Control max RAM usage
  - `OLLAMA_NUM_PARALLEL`: Concurrent request handling
- ✅ Recommendations:
  - 8GB Macs: `OLLAMA_MAX_RAM=6GB`, `OLLAMA_NUM_PARALLEL=1`
  - 16GB+ Macs: `OLLAMA_MAX_RAM=12GB`, `OLLAMA_NUM_PARALLEL=2`

**Quantization Benefits:**
- ✅ Q4_K_M: 20B model from 40GB → 10-12GB
- ✅ 70B model in 4-bit: ~35-40GB
- ⚠️ Context length increase = linear KV cache memory growth

**Sources:**
- [Ollama on Mac: Metal Acceleration Setup](https://localaimaster.com/blog/run-llama3-on-mac)
- [Optimized Ollama LLM server for Mac Studio](https://github.com/anurmatov/mac-studio-server)
- [Apple Metal Performance Shaders: Ollama Optimization Guide](https://markaicode.com/apple-metal-performance-shaders-m1-m2-ollama-optimization/)
- [Apple silicon limitations with local LLM](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)

### llama.cpp
**UMA Optimizations:**
- ✅ **Good Apple Silicon support**
- ✅ Unified memory: GPU and CPU share same RAM pool
- ✅ Enables local enterprise-class GPU simulation

**Metal Backend:**
- ✅ Full Metal compute kernel support
- ✅ No PyTorch abstraction layer (more efficient)
- ✅ M3 Pro/Max (18+ GPU cores): 28-35 tok/s on Llama 3.1 8B
- ✅ M1 base: 12-15 tok/s

**Memory Management:**
- ⚠️ **~75% physical RAM limit** (macOS driver hard-coded)
- ✅ M2 Max: 512-bit memory bus, up to 400 GB/s bandwidth
- ✅ M2 Ultra: 2× M2 Max bandwidth
- ⚠️ Memory bandwidth constrains performance (ALUs not fully utilized)

**Performance Characteristics:**
- ✅ Can run in GPU-accelerated containers on M-series Macs
- ✅ Recent benchmarks and discussions (Jan 2025, Apr 2025)

**Sources:**
- [Performance of llama.cpp on Apple Silicon M-series](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [M-series Macs running llama.cpp in GPU-Accelerated Containers](https://github.com/ggml-org/llama.cpp/discussions/12985)
- [70B LLaMA 2 on Mac Studio M2 Ultra](https://obrienlabs.medium.com/running-the-70b-llama-2-llm-locally-on-metal-via-llama-cpp-on-mac-studio-m2-ultra-32b3179e9cbe)
- [Thoughts on Apple Silicon Performance for Local LLMs](https://medium.com/@andreask_75652/thoughts-on-apple-silicon-performance-for-local-llms-3ef0a50e08bd)

---

## 6. Agent-Specific Features

### LM Studio
**System Prompt Caching:**
- ✅ Prompt caching in server mode (similar to llama.cpp)
- ✅ Large document cached on first load (5-10 min), subsequent requests fast (15-20s)
- ⚠️ No explicit "prompt caching" API exposed

**Context Templates:**
- ✅ **Config Presets**: Save system prompts + all parameters
- ✅ Named presets for different personas/use cases
- ✅ Easy switching between configurations
- ✅ Prompt Template documentation available

**Agent Persona Switching:**
- ✅ System prompts define persona, tone, behavior, role
- ✅ Examples: helpful assistant, tech support bot, coding expert
- ✅ `.act()` API for autonomous agent behavior:
  - Step-by-step reasoning
  - Tool invocation
  - Structured JSON output
- ✅ Extensive guide with examples for system prompts

**Sources:**
- [Prompt Template documentation](https://lmstudio.ai/docs/app/advanced/prompt-template)
- [The Ultimate Guide to LM Studio System Prompts](https://gyanaangan.in/blog/the-ultimate-guide-to-lm-studio-system-prompts-a-masterclass-with-examples)
- [Config Presets documentation](https://lmstudio.ai/docs/app/presets)
- [The .act() call documentation](https://lmstudio.ai/docs/python/agent/act)

### Ollama
**System Prompt Caching:**
- ✅ Prompt caching for optimization
- ✅ `cache_prompt` parameter for KV cache reuse
- ✅ Significantly reduces inference time and costs

**Context Templates:**
- ✅ **Modelfile TEMPLATE instruction**: Define full prompt template
- ✅ May include system message, user message, model response
- ✅ Syntax may be model-specific
- ✅ Can customize via Modelfile (similar to Dockerfile)

**Agent Persona Switching:**
- ✅ Modelfile configuration:
  - `FROM`: Base model selection
  - `PARAMETER`: Temperature (0.0-1.0), context window
  - `SYSTEM`: System prompt defining behavior
- ✅ Examples: "Senior Python Backend Engineer", "Creative Writer"
- ✅ MetaGPT integration: PROMPT_TEMPLATE for LLM instructions
- ✅ In-chat adjustment: `/save` command to create model variants

**Sources:**
- [Ollama prompt templates](https://medium.com/@laurentkubaski/ollama-prompt-templates-59066e02a82e)
- [Modelfile Reference](https://docs.ollama.com/modelfile)
- [How to Customize LLM Models with Ollama's Modelfile](https://www.gpu-mart.com/blog/custom-llm-models-with-ollama-modelfile)
- [Supercharging Ollama: Mastering System Prompts](https://johnwlittle.com/supercharging-ollama-mastering-system-prompts-for-better-results/)

### llama.cpp
**System Prompt Caching:**
- ✅ **Excellent system prompt caching**
- ✅ `--system-prompt-file` flag: Load system prompt across all slots
- ✅ Common prefix cached in memory after first computation
- ✅ `cache_prompt` parameter (default: true) for KV cache reuse
- ⚠️ Some flags removed/changed in recent versions

**Context Templates:**
- ✅ Server has prompt template formatting functionality
- ✅ Converts chat messages to single string expected by chat models
- ✅ Formatted prompt returned in response
- ⚠️ `--prompt-cache` only supported by llama-cli, not llama-server

**Agent Persona Switching:**
- ⚠️ No high-level persona switching API
- ✅ System prompt shared across slots for efficiency
- ✅ Template configuration per request
- ⚠️ `--system-prompt-file` flag may have been removed in some versions
- ⚠️ Cache enabling can cause non-deterministic results (batch size differences)

**Sources:**
- [How to cache system prompt discussion](https://github.com/ggml-org/llama.cpp/discussions/8947)
- [llama.cpp server cache_prompt parameter discussion](https://github.com/ggml-org/llama.cpp/discussions/10311)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Prompt Cache discussion](https://github.com/ggml-org/llama.cpp/discussions/745)

---

## Performance Comparison (2026)

### Speed & Efficiency
| Tool | Inference Speed | Model Loading | Memory Efficiency | Concurrent Requests |
|------|----------------|---------------|-------------------|---------------------|
| **LM Studio** | Good (MLX: +26-30% on Mac) | MLX: 3× faster on Mac | Excellent (MLX on Mac) | Limited |
| **Ollama** | Excellent (10-20% faster) | Fast (no GUI overhead) | Good (quantization) | Excellent |
| **llama.cpp** | Excellent (~161 tok/s) | Fast | Good | Good (slots system) |

**Sources:**
- [LM Studio vs Ollama: Performance Showdown](https://www.arsturn.com/blog/lm-studio-vs-ollama-the-ultimate-performance-showdown)
- [Ollama vs LM Studio on macOS](https://www.chrislockard.net/posts/ollama-vs-lmstudio-macos/)
- [Ollama vs. Llama.cpp Performance Guide](https://www.arsturn.com/blog/ollama-vs-llama-cpp-which-should-you-use-for-local-llms)

### Use Case Recommendations

**LM Studio:**
- ✅ Testing and experimentation
- ✅ Desktop AI applications
- ✅ macOS users (MLX advantage)
- ✅ Users preferring GUI
- ❌ Production multi-tenant servers
- ❌ High-throughput scenarios

**Ollama:**
- ✅ Building LLM-powered applications
- ✅ Rapid prototyping
- ✅ API-first workflows
- ✅ Concurrent request handling
- ✅ Ease of use and installation
- ❌ Maximum raw performance needs
- ⚠️ Limited for thousands of concurrent users

**llama.cpp:**
- ✅ Maximum performance and customization
- ✅ Broad hardware support
- ✅ Advanced KV cache management
- ✅ Session persistence needs
- ✅ Embedded/production deployments
- ❌ Users wanting simplicity
- ❌ GUI requirements

**Sources:**
- [LM Studio vs Ollama: Which is Faster](https://www.arsturn.com/blog/lm-studio-vs-ollama-the-ultimate-performance-showdown)
- [Llama.cpp vs Ollama — Best Local LLM Tool](https://www.openxcell.com/blog/llama-cpp-vs-ollama/)
- [Local LLM Hosting: Complete 2025 Guide](https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide-ollama-vllm-localai-jan-lm-studio-more-f98136ce7e4a)

---

## Key Findings & Gaps

### What They All Do Well ✅
1. **Prompt Caching**: All three implement some form of prompt/KV caching
2. **Apple Silicon Support**: All use Metal backend with good-to-excellent performance
3. **Integration**: All work with multi-agent frameworks (AutoGen, CrewAI, LangGraph)
4. **Context Management**: All have mechanisms to handle conversation context

### Critical Gaps ❌

#### Multi-Agent Native Support
- **NONE** provide native multi-agent orchestration
- All rely on external frameworks (AutoGen, CrewAI, LangGraph, MetaGPT)
- No built-in agent switching or coordination
- No per-agent context isolation

#### Session Persistence
- **LM Studio**: Only saves text conversations, not KV cache
- **Ollama**: No native persistence at all
- **llama.cpp**: Best support but requires API usage; not in WebUI

#### Multi-Tenant Isolation
- **NONE** provide robust multi-tenant isolation
- All use shared resources (different strategies)
- No per-user cache isolation
- Not designed for production multi-tenant scenarios

#### KV Cache Disk Persistence
- **LM Studio**: No disk persistence
- **Ollama**: No disk persistence
- **llama.cpp**: Has API but not exposed in WebUI (as of Nov 2025)

### What's Missing for True Multi-Agent Systems 🎯

1. **Native Agent Management**
   - Agent creation/deletion APIs
   - Agent registry and discovery
   - Agent-to-agent communication

2. **Isolated Contexts Per Agent**
   - Dedicated KV cache per agent
   - Memory isolation between agents
   - Context switching without recomputation

3. **Persistent Agent State**
   - Save/restore agent context including KV cache
   - Cross-session agent continuity
   - Agent-specific memory systems

4. **Agent Coordination**
   - Built-in orchestration layer
   - Workflow management
   - Task routing between agents

5. **Production Multi-Tenancy**
   - Per-tenant resource limits
   - Isolation guarantees
   - Separate billing/accounting

---

## Comparison Table Summary

| Feature | LM Studio | Ollama | llama.cpp | Gap Analysis |
|---------|-----------|--------|-----------|--------------|
| **Multi-Agent Native** | ❌ | ❌ | ❌ | All require external frameworks |
| **Separate Agent Contexts** | ❌ | ❌ | ⚠️ (shared) | No true isolation |
| **Agent Switching** | ⚠️ (multi-model) | ⚠️ (multi-model) | ✅ (router mode) | Not designed for agents |
| **KV Cache Exposed** | ✅ | ✅ | ✅ | ✅ All support |
| **KV Cache Persistence** | ❌ | ❌ | ✅ (API only) | Disk persistence limited |
| **KV Cache Reuse** | ✅ | ✅ | ✅ | ✅ All support |
| **Cache Eviction Policies** | ✅ TTL/Auto-evict | ✅ keep_alive | ✅ LRU-like | ✅ All have policies |
| **Session Save/Restore** | ✅ (text only) | ❌ | ✅ (full state) | Text vs KV cache gap |
| **Cross-Session Resume** | ⚠️ (no KV cache) | ❌ | ✅ | Best: llama.cpp |
| **Multi-User Concurrent** | ⚠️ Limited | ✅ Good | ✅ Good | LM Studio weakest |
| **Context Isolation** | ❌ | ❌ | ⚠️ (shared) | No per-user isolation |
| **Per-User Cache** | ❌ | ❌ | ⚠️ (session-based) | No true isolation |
| **Apple Silicon UMA** | ✅ Excellent (MLX) | ✅ Good | ✅ Good | LM Studio best |
| **Metal Backend** | ✅ | ✅ | ✅ | ✅ All support |
| **Memory Management** | ✅ Advanced | ✅ Good | ✅ Good | LM Studio most configurable |
| **System Prompt Caching** | ✅ | ✅ | ✅ | ✅ All support |
| **Context Templates** | ✅ Presets | ✅ Modelfile | ✅ Templates | ✅ All support |
| **Agent Personas** | ✅ .act() API | ✅ Modelfile | ⚠️ Manual | LM Studio best UX |

---

## Additional Resources

### GitHub Repositories
- [LM Studio Bug Tracker](https://github.com/lmstudio-ai/lmstudio-bug-tracker)
- [LM Studio JS SDK](https://github.com/lmstudio-ai/lmstudio-js)
- [Ollama Main Repository](https://github.com/ollama/ollama)
- [Ollama Python](https://github.com/ollama/ollama-python)
- [llama.cpp Main Repository](https://github.com/ggml-org/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

### Documentation
- [LM Studio Docs](https://lmstudio.ai/docs)
- [Ollama Docs](https://docs.ollama.com/)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Community & Forums
- [LM Studio Blog](https://lmstudio.ai/blog)
- [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)
- [llama.cpp Discussions](https://github.com/ggml-org/llama.cpp/discussions)

---

## Conclusion

As of January 2026, **none of the three major local LLM tools** (LM Studio, Ollama, llama.cpp) provide **native multi-agent orchestration** or **per-agent context isolation**. All three:

1. Require external frameworks for multi-agent systems
2. Lack built-in agent switching and coordination
3. Don't persist KV cache to disk in user-friendly ways (llama.cpp has API support)
4. Provide limited or no multi-tenant isolation

**The most advanced capabilities exist in llama.cpp**, particularly for:
- KV cache persistence (via Slot Persistence API)
- Session save/restore with full state
- Router mode for dynamic model management

**LM Studio excels at:**
- Apple Silicon optimization (MLX)
- User-friendly desktop experience
- Agent-like features (.act() API)

**Ollama leads in:**
- Concurrent request handling
- Ease of use and deployment
- API-first development

**A true multi-agent local LLM system would need to build on these tools** while adding:
- Native agent lifecycle management
- Isolated contexts with efficient switching
- Persistent agent state including KV cache
- Agent coordination and orchestration layer
- Production-grade multi-tenancy

This represents a significant **opportunity for innovation** in the local LLM tooling space.
