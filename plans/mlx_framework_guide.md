# MLX Framework Comprehensive Guide

**Date**: 2026-01-25
**MLX Version**: 0.30.4
**mlx_lm Version**: 0.30.4
**Target Platform**: Apple Silicon (M-series chips)

---

## Table of Contents

1. [MLX Framework Overview](#1-mlx-framework-overview)
2. [Core Concepts](#2-core-concepts)
3. [mlx_lm Library](#3-mlx_lm-library)
4. [KV Cache Management](#4-kv-cache-management)
5. [Sampling and Generation](#5-sampling-and-generation)
6. [Performance Optimization](#6-performance-optimization)
7. [API Reference](#7-api-reference)
8. [Gotchas and Best Practices](#8-gotchas-and-best-practices)

---

## 1. MLX Framework Overview

### 1.1 What is MLX and Why It Exists

MLX is an open-source array framework developed by Apple's machine learning research team, purpose-built for Apple Silicon. It serves as a unified foundation for numerical computing and machine learning on Apple devices, from basic computations to running frontier LLM models locally.

**Key Motivations**:
- Leverage Apple Silicon's unique unified memory architecture
- Provide a familiar NumPy-like API for ML practitioners
- Enable efficient on-device ML inference and training
- Support the M-series chip ecosystem (Mac, iPhone, iPad, Vision Pro)

### 1.2 Design Philosophy

MLX is built around three core principles:

#### Unified Memory Model
Apple Silicon features a unified memory architecture where CPU and GPU share the same physical memory pool. MLX exploits this by keeping arrays in shared memory, eliminating the costly data transfers between CPU and GPU that plague discrete GPU systems.

```python
import mlx.core as mx

# Arrays live in unified memory - no explicit placement needed
a = mx.random.normal((1000, 1000))
b = mx.random.normal((1000, 1000))

# Operations specify execution device, not data location
c = mx.add(a, b, stream=mx.gpu)  # Execute on GPU
d = mx.matmul(a, b, stream=mx.cpu)  # Execute on CPU
# Both operations access the same arrays - no copy needed
```

#### Lazy Evaluation
MLX uses lazy (deferred) computation. Operations build a computation graph rather than executing immediately. This enables:
- Graph-level optimizations before execution
- Efficient function transformations (grad, vmap)
- Better memory management through fusion

```python
# No computation happens yet - just builds graph
x = mx.array([1, 2, 3])
y = mx.array([4, 5, 6])
z = mx.add(x, y)
w = mx.multiply(z, 2)

# Computation executes when needed
mx.eval(w)  # Explicit evaluation
print(w)     # Implicit evaluation (triggers compute)
```

#### Composable Transformations
MLX provides functional transformations like `grad()`, `vmap()`, and `compile()` that compose cleanly:

```python
def loss_fn(params, x, y):
    pred = model(params, x)
    return mx.mean((pred - y) ** 2)

# Compose transformations
grad_fn = mx.grad(loss_fn)
compiled_grad = mx.compile(grad_fn)
```

### 1.3 Target Platforms

MLX is optimized for Apple Silicon:

| Chip Family | Support Level | Key Features |
|-------------|---------------|--------------|
| M1/M2/M3/M4 | Full | GPU acceleration via Metal |
| M4 Pro/Max | Full | Higher memory bandwidth, more GPU cores |
| M5 | Enhanced | Neural Accelerators for matrix ops (4x speedup) |
| A-series | Supported | iPhone/iPad deployment |

**Hardware Requirements**:
- macOS 13.3+ (macOS 15+ for wired memory optimization)
- Apple Silicon (M1 or later)
- GPU memory limit: ~75% of system RAM

### 1.4 Comparison with Other Frameworks

| Feature | MLX | PyTorch (MPS) | JAX | TensorFlow |
|---------|-----|---------------|-----|------------|
| **Memory Model** | Unified (zero-copy) | Transfer-based | Transfer-based | Transfer-based |
| **Evaluation** | Lazy | Eager (default) | Lazy (JIT) | Graph/Eager |
| **Apple Optimization** | Native | MPS backend | Limited | Limited |
| **API Style** | NumPy-like | NumPy-like | NumPy-like | Mixed |
| **Transformations** | grad, vmap, compile | autograd, compile | grad, vmap, jit | GradientTape |
| **LLM Support** | mlx_lm (native) | transformers | transformers | Limited |

**Benchmark Insights** (2025-2026):
- Training: PyTorch MPS can be faster for batch training (10-14s vs 21-27s per epoch)
- Inference: MLX excels at low-latency inference due to faster GPU spinup
- LLM: MLX achieves ~50 tokens/sec on 4-bit Llama 3B (M3 Max)
- Energy: Apple Silicon uses ~80% less power than comparable NVIDIA GPUs

---

## 2. Core Concepts

### 2.1 mx.array Fundamentals

`mx.array` is MLX's central data structure, analogous to NumPy's ndarray but with lazy evaluation.

```python
import mlx.core as mx

# Creation
a = mx.array([1, 2, 3])                    # From Python list
b = mx.zeros((3, 4))                        # Zeros
c = mx.random.normal((100, 100))           # Random normal
d = mx.arange(0, 10, 0.5)                  # Range

# Properties
print(a.shape)    # (3,)
print(a.dtype)    # mlx.core.int32
print(a.nbytes)   # 12

# Operations (lazy - no computation yet)
e = a + b[0, :3]
f = mx.matmul(c, c.T)

# Common dtypes
mx.float32  # Default float
mx.float16  # Half precision
mx.bfloat16 # Brain float
mx.int32    # Default int
mx.bool_    # Boolean
```

**Key Differences from NumPy**:
1. Arrays are immutable (functional style)
2. Operations return new arrays, don't modify in-place
3. Computation is deferred until needed
4. Arrays live in unified memory (no device placement)

### 2.2 Lazy Evaluation and mx.eval()

MLX defers computation by building a computation graph:

```python
# Graph construction (no computation)
a = mx.random.normal((1000, 1000))
b = mx.random.normal((1000, 1000))
c = mx.matmul(a, b)
d = mx.sum(c)

# At this point, nothing has been computed
# a, b, c, d are "promises" of values
```

**Explicit Evaluation**:
```python
mx.eval(d)        # Evaluate single array
mx.eval(c, d)     # Evaluate multiple arrays
mx.eval([c, d])   # Evaluate from list
```

**Implicit Evaluation Triggers**:
- `print(array)` - Printing
- `array.item()` - Converting to Python scalar
- `np.array(array)` - Converting to NumPy
- `if array > 0:` - Control flow with scalar arrays
- `mx.save()` - Saving to disk
- Memory access via memoryview

**Best Practice**: Evaluate at natural boundaries (e.g., end of training iteration):

```python
for batch in dataloader:
    loss = train_step(model, batch)
    mx.eval(loss)  # Natural evaluation point
```

### 2.3 Memory Model (Unified GPU/CPU)

Unlike discrete GPU systems, MLX arrays don't need explicit device placement:

```python
# Traditional GPU framework (e.g., PyTorch)
x_cpu = torch.tensor([1, 2, 3])
x_gpu = x_cpu.to('cuda')  # Explicit transfer
result = model(x_gpu)
result_cpu = result.cpu()  # Transfer back

# MLX - no transfers needed
x = mx.array([1, 2, 3])
result = model(x)  # Same array, operations specify device
```

**Device Specification via Streams**:
```python
# Default: GPU stream
c = mx.add(a, b)  # Uses default GPU stream

# Explicit CPU execution
c = mx.add(a, b, stream=mx.cpu)

# Explicit GPU execution
c = mx.add(a, b, stream=mx.gpu)

# Custom streams for parallelism
cpu_stream = mx.new_stream(mx.cpu)
c = mx.add(a, b, stream=cpu_stream)
```

**Automatic Dependency Management**:
```python
# MLX handles cross-stream dependencies automatically
c = mx.add(a, b, stream=mx.cpu)
d = mx.add(a, c, stream=mx.gpu)  # Waits for c automatically
```

### 2.4 Metal Backend and Acceleration

MLX uses Apple's Metal framework for GPU compute:

**Metal API**:
```python
import mlx.core as mx

# Check Metal availability
print(mx.metal.is_available())  # True on Apple Silicon

# Memory management
print(mx.metal.get_active_memory())   # Current GPU memory (bytes)
print(mx.metal.get_cache_memory())    # Cached memory (bytes)
print(mx.metal.get_peak_memory())     # Peak memory (bytes)

# Clear GPU cache
mx.metal.clear_cache()

# Memory limits
mx.metal.set_memory_limit(8 * 1024**3)  # 8GB limit
mx.metal.set_cache_limit(2 * 1024**3)   # 2GB cache
```

**Wired Memory** (macOS 15+):
```python
# Get/set wired memory limit (pre-wires pages to prevent faults)
current = mx.metal.get_wired_limit()
mx.metal.set_wired_limit(16 * 1024**3)  # 16GB wired
```

**Optimized Operations** (`mx.fast`):
```python
# Highly optimized ML operations
mx.fast.scaled_dot_product_attention(q, k, v)
mx.fast.rope(x, dims, offset)  # Rotary position embeddings
mx.fast.rms_norm(x, weight, eps)  # RMS normalization
mx.fast.layer_norm(x, weight, bias, eps)  # Layer normalization
```

### 2.5 Device Management

```python
# Get default device
print(mx.default_device())  # Device(gpu, 0)

# Set default device
mx.set_default_device(mx.cpu)
mx.set_default_device(mx.gpu)

# Device context
with mx.stream(mx.cpu):
    # All operations in this block use CPU
    result = mx.matmul(a, b)

# Streams for parallel execution
gpu_stream = mx.default_stream(mx.gpu)
cpu_stream = mx.new_stream(mx.cpu)

# Synchronize stream
mx.synchronize(gpu_stream)
```

---

## 3. mlx_lm Library

### 3.1 Model Loading

The `mlx_lm` package provides utilities for loading and running LLMs:

```python
from mlx_lm import load

# Load model and tokenizer from Hugging Face Hub
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Load with specific revision
model, tokenizer = load("mlx-community/Mistral-7B-v0.3-4bit", revision="main")

# Load local model
model, tokenizer = load("/path/to/local/model")
```

**Model Configuration**:
```python
# Access model config
config = model.args
print(config.num_hidden_layers)    # 32
print(config.num_attention_heads)  # 32
print(config.hidden_size)          # 4096
print(config.num_key_value_heads)  # 8 (GQA)
```

### 3.2 Generation APIs

#### Simple Generation
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

response = generate(
    model,
    tokenizer,
    prompt="What is machine learning?",
    max_tokens=100,
    verbose=True  # Print timing stats
)
print(response)
```

#### Streaming Generation
```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

for response in stream_generate(
    model,
    tokenizer,
    prompt="Explain neural networks:",
    max_tokens=200
):
    print(response.text, end="", flush=True)
    if response.finish_reason:
        print(f"\nFinished: {response.finish_reason}")
```

**GenerationResponse Fields**:
- `text` (str): Decoded text segment
- `token` (int): Token ID
- `logprobs` (mx.array): Log probabilities
- `prompt_tokens` (int): Number of prompt tokens
- `prompt_tps` (float): Prompt processing tokens/sec
- `generation_tokens` (int): Tokens generated so far
- `generation_tps` (float): Generation tokens/sec
- `peak_memory` (float): Peak memory in GB
- `finish_reason` (Optional[str]): "length", "stop", or None

#### Batch Generation
```python
from mlx_lm import load, batch_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

prompts = [
    tokenizer.encode("What is AI?"),
    tokenizer.encode("Explain Python:"),
    tokenizer.encode("Write a haiku:")
]

response = batch_generate(
    model,
    tokenizer,
    prompts=prompts,
    max_tokens=50,
    verbose=True,
    return_prompt_caches=True,
    completion_batch_size=32,
    prefill_batch_size=8
)

for i, text in enumerate(response.texts):
    print(f"Response {i}: {text}")

print(f"Stats: {response.stats}")
```

### 3.3 BatchGenerator Class

The `BatchGenerator` is the core batching engine for efficient multi-request inference:

```python
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Create generator
generator = BatchGenerator(
    model=model,
    max_tokens=128,                    # Default max tokens per request
    stop_tokens=set([tokenizer.eos_token_id]),  # Stop token IDs
    sampler=make_sampler(temp=0.8),    # Sampling function
    completion_batch_size=32,          # Max concurrent sequences
    prefill_batch_size=8,              # Batch size for prefill
    prefill_step_size=2048,            # Tokens per prefill step
)

# Insert prompts (returns unique IDs)
prompts = [
    tokenizer.encode("Hello, how are you?"),
    tokenizer.encode("What is the capital of France?"),
]
uids = generator.insert(
    prompts,
    max_tokens=[50, 100],  # Per-prompt limits (must be list!)
)

# Generate tokens
tokens_by_uid = {uid: [] for uid in uids}

while responses := generator.next():
    for r in responses:
        if r.finish_reason != "stop":
            tokens_by_uid[r.uid].append(r.token)

        if r.finish_reason:
            text = tokenizer.decode(tokens_by_uid[r.uid])
            print(f"UID {r.uid}: {text}")
            # Cache available: r.prompt_cache (attribute, not callable)

# Get statistics
stats = generator.stats()
print(f"Prompt TPS: {stats.prompt_tps:.2f}")
print(f"Generation TPS: {stats.generation_tps:.2f}")

# Cleanup
generator.close()
```

**BatchGenerator Methods**:
- `insert(prompts, max_tokens, caches, samplers, logits_processors)` - Add prompts to queue
- `next()` - Generate next token batch, returns list of Response objects
- `remove(uids)` - Remove requests from batch
- `stats()` - Get BatchStats
- `close()` - Cleanup resources

### 3.4 Tokenization (TokenizerWrapper)

The `load()` function returns a tokenizer with extended functionality:

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Basic encoding/decoding
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Special tokens
print(tokenizer.eos_token_id)    # End of sequence
print(tokenizer.bos_token_id)    # Beginning of sequence
print(tokenizer.pad_token_id)    # Padding token

# Chat template (if supported)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 3.5 Supported Model Architectures

mlx_lm v0.30.4 supports a wide range of architectures:

| Family | Models |
|--------|--------|
| **Llama** | Llama 2, Llama 3, Llama 3.1, Llama 4 |
| **Mistral** | Mistral 7B, Mixtral MoE |
| **Qwen** | Qwen 2, Qwen 2.5, Qwen 3, Qwen MoE |
| **Gemma** | Gemma 2, Gemma 3 (hybrid attention) |
| **Phi** | Phi-2, Phi-3, Phi-3.5 MoE |
| **Cohere** | Cohere 1, Cohere 2, Command-R |
| **DeepSeek** | DeepSeek, DeepSeek MoE |
| **Starcoder** | Starcoder 2 |
| **Falcon** | Falcon H1 |
| **Others** | MiniCPM, GLM, Helium, InternLM, PLaMo, Ernie, etc. |

Full list: [mlx-community on Hugging Face](https://huggingface.co/mlx-community)

---

## 4. KV Cache Management

### 4.1 Cache Types

mlx_lm provides multiple cache implementations in `mlx_lm.models.cache`:

#### KVCache (Standard)
```python
from mlx_lm.models.cache import KVCache

cache = KVCache()

# Properties
cache.state      # Returns (keys, values)
cache.size()     # Current sequence length
cache.empty()    # True if no tokens cached

# Methods
cache.update_and_fetch(keys, values)  # Update and return full cache
cache.trim(n)    # Remove last n tokens
cache.to_quantized(group_size=64, bits=4)  # Convert to quantized

# Shape: (batch, n_kv_heads, seq_len, head_dim)
```

#### BatchKVCache
```python
from mlx_lm.models.cache import BatchKVCache

# Expects LEFT-PADDED inputs!
# Prompts: [1,3,5], [7], [2,6,8,9]
# Padded:  [0,1,3,5], [0,0,0,7], [2,6,8,9]
# left_padding = [1, 3, 0]

cache = BatchKVCache(left_padding=[1, 3, 0])

# Methods
cache.update_and_fetch(keys, values)
cache.prepare(left_padding=None, lengths=None, right_padding=None)
cache.finalize()           # Convert right-padding to left-padding
cache.filter(batch_indices)  # Keep only specified indices
cache.extend(other)         # Merge with another BatchKVCache
cache.extract(idx)          # Extract single-request cache -> KVCache
```

**Critical**: BatchKVCache uses **left-padding** convention!

#### RotatingKVCache (Sliding Window)
```python
from mlx_lm.models.cache import RotatingKVCache

cache = RotatingKVCache(
    max_size=4096,  # Maximum tokens to keep
    keep=256        # Always keep first N tokens (e.g., system prompt)
)

# Automatically evicts oldest tokens when full
# Used by models with sliding window attention (Gemma 3, etc.)
```

#### QuantizedKVCache
```python
from mlx_lm.models.cache import QuantizedKVCache

cache = QuantizedKVCache(
    group_size=64,  # Quantization group size
    bits=8          # 4 or 8 bits
)

# Reduces memory usage ~50-75% at slight quality cost
```

### 4.2 Cache Persistence

Save and load caches for prompt reuse:

```python
from mlx_lm.models.cache import (
    make_prompt_cache,
    save_prompt_cache,
    load_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache
)

# Create cache for model
cache = make_prompt_cache(model, max_kv_size=8192)

# After prefill, save to disk
save_prompt_cache(
    "my_cache.safetensors",
    cache,
    metadata={"model": "llama-3.2", "prompt_tokens": 1024}
)

# Load cache later
cache, metadata = load_prompt_cache(
    "my_cache.safetensors",
    return_metadata=True
)

# Trim cache if needed
if can_trim_prompt_cache(cache):
    trimmed = trim_prompt_cache(cache, num_tokens=100)
```

### 4.3 Quantized KV Cache

For long contexts, quantize the cache to save memory:

```python
# During generation
response = generate(
    model,
    tokenizer,
    prompt=long_prompt,
    kv_bits=4,           # 4-bit quantization
    kv_group_size=64,    # Quantization group
    quantized_kv_start=5000  # Start quantizing after N tokens
)

# Or convert existing cache
quantized = cache.to_quantized(group_size=64, bits=4)
```

**Memory Savings**:
- 4-bit: ~75% reduction
- 8-bit: ~50% reduction

### 4.4 Left-Padding Convention

BatchKVCache requires left-padded sequences for correct attention masking:

```python
# CORRECT: Left-padded
prompts = [
    [0, 1, 2, 3],    # padding=1
    [0, 0, 0, 7],    # padding=3
    [2, 3, 4, 5]     # padding=0
]
left_padding = [1, 3, 0]

# WRONG: Right-padded (will fail!)
prompts = [
    [1, 2, 3, 0],
    [7, 0, 0, 0],
    [2, 3, 4, 5]
]
```

### 4.5 Cache Extraction and Injection

Extract cache from completed sequences and inject into new generations:

```python
from mlx_lm.generate import BatchGenerator

generator = BatchGenerator(model, stop_tokens=set([eos_id]))

# Insert prompts
uids = generator.insert([prompt_tokens])

# Generate and extract cache on completion
while responses := generator.next():
    for r in responses:
        if r.finish_reason is not None:
            # Cache is an ATTRIBUTE, not callable
            cache = r.prompt_cache  # List of cache objects per layer

            # Save for later
            save_prompt_cache("agent_cache.safetensors", cache)

# Later: inject cache into new generation
loaded_cache = load_prompt_cache("agent_cache.safetensors")
uids = generator.insert(
    [continuation_tokens],
    caches=[loaded_cache]  # Inject pre-computed cache
)
```

---

## 5. Sampling and Generation

### 5.1 make_sampler() Function

Create sampling functions for token selection:

```python
from mlx_lm.sample_utils import make_sampler

# Deterministic (greedy) sampling
sampler = make_sampler(temp=0.0)

# Stochastic sampling with temperature
sampler = make_sampler(
    temp=0.8,           # Temperature (0.0 = deterministic)
    top_p=0.95,         # Nucleus sampling threshold
    top_k=50,           # Top-k sampling
    min_p=0.05,         # Minimum probability threshold
    min_tokens_to_keep=1  # Always keep at least 1 token
)

# XTC (experimental) sampling
sampler = make_sampler(
    temp=0.8,
    xtc_probability=0.1,
    xtc_threshold=0.2,
    xtc_special_tokens=[eos_id]
)

# Use sampler
token = sampler(logprobs)  # logprobs: [batch, vocab_size] -> [batch]
```

### 5.2 Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temp` | 0.0 | Temperature. 0=deterministic, higher=more random |
| `top_p` | 0.0 | Nucleus sampling. Keep tokens until cumsum >= p |
| `top_k` | 0 | Keep only top k tokens. 0=disabled |
| `min_p` | 0.0 | Remove tokens below min_p * max_prob |
| `min_tokens_to_keep` | 1 | Always keep at least N tokens |
| `repetition_penalty` | 1.0 | Penalty for repeated tokens. >1 discourages |
| `repetition_context_size` | 20 | Context window for repetition penalty |

**Recommended Settings**:
- Creative writing: `temp=0.8, top_p=0.95`
- Code generation: `temp=0.2, top_p=0.9`
- Factual Q&A: `temp=0.0` (deterministic)
- Chat: `temp=0.7, top_p=0.9, repetition_penalty=1.1`

### 5.3 Logits Processors

Apply transformations to logits before sampling:

```python
from mlx_lm.sample_utils import make_logits_processors

processors = make_logits_processors(
    logit_bias={50256: -100.0},  # Suppress specific tokens
    repetition_penalty=1.2,
    repetition_context_size=50
)

# Use with generator
generator = BatchGenerator(
    model,
    logits_processors=processors,  # Global processors
    stop_tokens=set([eos_id])
)

# Or per-request
uids = generator.insert(
    prompts,
    logits_processors=[processors_1, processors_2]  # Per-prompt
)
```

**Custom Logits Processor**:
```python
def custom_processor(tokens: mx.array, logits: mx.array) -> mx.array:
    """
    Args:
        tokens: [seq_len] - tokens generated so far
        logits: [batch, vocab_size] - current logits
    Returns:
        Modified logits: [batch, vocab_size]
    """
    # Example: Suppress specific token
    logits[:, 42] = -float('inf')
    return logits
```

### 5.4 Stop Tokens and Finish Reasons

```python
generator = BatchGenerator(
    model,
    stop_tokens=set([
        tokenizer.eos_token_id,
        tokenizer.encode("<|endoftext|>")[0],
        tokenizer.encode("###")[0]
    ]),
    max_tokens=256
)

# Response.finish_reason values:
# - "stop": Hit a stop token
# - "length": Hit max_tokens limit
# - None: Still generating
```

---

## 6. Performance Optimization

### 6.1 Batch Size Tuning

**Prefill vs Completion Batch Size**:
- `prefill_batch_size`: How many prompts to process together (memory-intensive)
- `completion_batch_size`: Max concurrent decoding sequences

```python
# High throughput (many short requests)
generator = BatchGenerator(
    model,
    completion_batch_size=64,
    prefill_batch_size=16
)

# Low latency (few long requests)
generator = BatchGenerator(
    model,
    completion_batch_size=8,
    prefill_batch_size=4
)
```

**Guidelines**:
- M4 Pro 24GB: `completion=32, prefill=8` (typical)
- M4 Max 64GB: `completion=64, prefill=16` (more headroom)
- Memory-constrained: Reduce both, prioritize `completion_batch_size`

### 6.2 Prefill Step Size

Controls how many tokens are processed per prefill step:

```python
generator = BatchGenerator(
    model,
    prefill_step_size=512,   # Conservative (less memory)
    # prefill_step_size=2048,  # Default (balanced)
    # prefill_step_size=4096,  # Aggressive (faster, more memory)
)
```

### 6.3 Memory Management on Metal

```python
import mlx.core as mx

# Monitor memory
print(f"Active: {mx.metal.get_active_memory() / 1e9:.2f} GB")
print(f"Cache: {mx.metal.get_cache_memory() / 1e9:.2f} GB")
print(f"Peak: {mx.metal.get_peak_memory() / 1e9:.2f} GB")

# Clear cache between generations
mx.metal.clear_cache()

# Set limits
mx.metal.set_memory_limit(20 * 1024**3)  # 20GB
mx.metal.set_cache_limit(4 * 1024**3)    # 4GB cache

# Wired memory (macOS 15+, prevents page faults)
mx.metal.set_wired_limit(16 * 1024**3)
```

**Memory Budget** (M4 Pro 24GB):
- Model weights (4-bit 12B): ~6-7 GB
- KV cache: ~4-12 GB (varies by context)
- MLX overhead: ~1 GB
- OS/Apps: ~4 GB

### 6.4 Quantization Strategies

**Model Quantization**:
```bash
# Convert model to 4-bit
mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct \
               -q --q-bits 4 --q-group-size 64 \
               --mlx-path ./llama-3.2-3b-4bit
```

**KV Cache Quantization**:
```python
# Generate with quantized KV cache
response = generate(
    model, tokenizer, prompt,
    kv_bits=8,              # 8-bit KV cache
    kv_group_size=64,
    quantized_kv_start=2000  # Start after 2K tokens
)
```

### 6.5 Prompt Caching Benefits

**Without Cache** (re-process 3500 tokens): ~18.9 seconds
**With Cache** (load + generate): ~0.4 seconds

```python
# First request: build and save cache
cache = make_prompt_cache(model)
# ... prefill with system prompt ...
save_prompt_cache("system_cache.safetensors", cache)

# Subsequent requests: load and reuse
for user_query in queries:
    cache = load_prompt_cache("system_cache.safetensors")
    response = generate(model, tokenizer, user_query, prompt_cache=cache)
```

---

## 7. API Reference

### 7.1 Key Classes

#### BatchGenerator
```python
class BatchGenerator:
    def __init__(
        self,
        model,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        prompt_progress_callback: Optional[Callable] = None,
    )

    def insert(
        self,
        prompts: List[List[int]],      # Tokenized prompts
        max_tokens: Union[List[int], int, None] = None,
        caches: Optional[List] = None,
        samplers: Optional[List] = None,
        logits_processors: Optional[List] = None,
    ) -> List[int]:  # Returns UIDs

    def next(self) -> List[Response]:
    def remove(self, uids: List[int]) -> None:
    def stats(self) -> BatchStats:
    def close(self) -> None:
```

#### BatchGenerator.Response
```python
@dataclass
class Response:
    uid: int                          # Request unique ID
    token: int                        # Generated token ID
    logprobs: mx.array                # Log probabilities
    finish_reason: Optional[str]      # "stop", "length", or None
    prompt_cache: List[Any]           # KV cache (attribute, not callable!)
```

#### BatchStats
```python
@dataclass
class BatchStats:
    prompt_tokens: int = 0
    prompt_tps: float = 0             # Prompt tokens per second
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0         # Generation tokens per second
    generation_time: float = 0
    peak_memory: float = 0            # Peak memory in GB
```

### 7.2 Cache Functions

```python
# mlx_lm.models.cache

def make_prompt_cache(model, max_kv_size: Optional[int] = None) -> List[Any]
def save_prompt_cache(file_name: str, cache: List, metadata: Dict = {})
def load_prompt_cache(file_name: str, return_metadata: bool = False)
def can_trim_prompt_cache(cache: List) -> bool
def trim_prompt_cache(cache: List, num_tokens: int) -> int
```

### 7.3 Sampling Functions

```python
# mlx_lm.sample_utils

def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
) -> Callable[[mx.array], mx.array]

def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
) -> List[Callable]
```

### 7.4 Common Usage Patterns

**Pattern 1: Simple Generation**
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
response = generate(model, tokenizer, "Hello!", max_tokens=100)
```

**Pattern 2: Streaming**
```python
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=200):
    print(chunk.text, end="")
```

**Pattern 3: Batch Processing**
```python
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

generator = BatchGenerator(model, stop_tokens={eos_id})
uids = generator.insert([tokenizer.encode(p) for p in prompts], max_tokens=[100]*len(prompts))

results = {uid: [] for uid in uids}
while responses := generator.next():
    for r in responses:
        if r.token and r.finish_reason != "stop":
            results[r.uid].append(r.token)
        if r.finish_reason:
            print(tokenizer.decode(results[r.uid]))

generator.close()
```

**Pattern 4: Cache Reuse**
```python
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache

# Save after prefill
save_prompt_cache("cache.safetensors", response.prompt_cache)

# Load and reuse
cache = load_prompt_cache("cache.safetensors")
generator.insert([new_tokens], caches=[cache])
```

---

## 8. Gotchas and Best Practices

### 8.1 Common Pitfalls

#### 1. Padding Convention
```python
# WRONG: Right-padded
tokens = [1, 2, 3, 0, 0]

# CORRECT: Left-padded for BatchKVCache
tokens = [0, 0, 1, 2, 3]
```

#### 2. Token Accumulation
```python
# WRONG: Expecting multiple tokens per call
for response in generator.next():
    text = tokenizer.decode(response.tokens)  # No .tokens!

# CORRECT: Accumulate single tokens
tokens = []
for response in generator.next():
    tokens.append(response.token)  # Singular .token
text = tokenizer.decode(tokens)
```

#### 3. Cache Access
```python
# WRONG: Calling prompt_cache as function
cache = response.prompt_cache()

# CORRECT: Accessing as attribute
cache = response.prompt_cache
```

#### 4. max_tokens Type
```python
# WRONG: Scalar for BatchGenerator.insert
generator.insert(prompts, max_tokens=100)

# CORRECT: Must be list
generator.insert(prompts, max_tokens=[100] * len(prompts))
```

#### 5. Sampler Parameter
```python
# WRONG: temperature parameter
sampler = make_sampler(temperature=0.8)

# CORRECT: temp parameter
sampler = make_sampler(temp=0.8)
```

#### 6. Generator Termination
```python
# WRONG: Expecting StopIteration
try:
    while True:
        responses = generator.next()
except StopIteration:
    pass

# CORRECT: Check for empty list
while responses := generator.next():
    process(responses)
# Loop exits when responses == []
```

### 8.2 Performance Tips

1. **Batch similar-length prompts**: Reduces padding waste
2. **Use quantized models**: 4-bit saves ~75% memory
3. **Enable KV cache quantization**: For long contexts
4. **Set appropriate prefill_step_size**: Balance memory vs speed
5. **Clear cache between unrelated generations**: `mx.metal.clear_cache()`
6. **Use wired memory on macOS 15+**: Prevents page faults

### 8.3 Memory Considerations

**Memory Formula**:
```
Total = Model + KV_Cache + Overhead

Model (4-bit): ~0.5 bytes/param
KV_Cache: 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
Overhead: ~1-2 GB
```

**Example** (Llama 3.2 3B, 4-bit, 4K context):
```
Model: 3B * 0.5 = ~1.5 GB
KV: 2 * 28 * 8 * 128 * 4096 * 2 = ~1.2 GB
Overhead: ~1 GB
Total: ~3.7 GB
```

### 8.4 Error Handling

```python
try:
    response = generate(model, tokenizer, prompt, max_tokens=100)
except mx.OutOfMemoryError:
    # Reduce batch size or clear cache
    mx.metal.clear_cache()
    # Retry with smaller batch
except Exception as e:
    print(f"Generation failed: {e}")
    # Log for debugging
```

### 8.5 Best Practices Summary

1. **Always tokenize prompts** before passing to BatchGenerator.insert()
2. **Accumulate tokens manually** - responses yield one token at a time
3. **Use left-padding** for batch operations
4. **Call generator.close()** to release resources
5. **Monitor memory** with `mx.metal.get_active_memory()`
6. **Evaluate at natural boundaries** - don't over-evaluate
7. **Save caches for common prefixes** - dramatic latency reduction
8. **Use quantization** for production deployments
9. **Test with small models first** before scaling up

---

## References

### Official Documentation
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [mlx-lm PyPI](https://pypi.org/project/mlx-lm/)

### Hugging Face Resources
- [MLX Community Models](https://huggingface.co/mlx-community)
- [Using MLX at Hugging Face](https://huggingface.co/docs/hub/en/mlx)

### Apple Resources
- [WWDC 2025: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [Apple ML Research: MLX and M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Apple Open Source: MLX](https://opensource.apple.com/projects/mlx/)

### Technical References
- [Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [Lazy Evaluation Documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [Metal API Reference](https://ml-explore.github.io/mlx/build/html/python/metal.html)

---

**Document Status**: Complete
**Last Updated**: 2026-01-25
**MLX Version**: 0.30.4
**mlx_lm Version**: 0.30.4
