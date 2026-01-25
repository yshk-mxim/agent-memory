# mlx_lm API Reference (v0.30.4)

**Date**: 2026-01-24
**Source**: mlx_lm v0.30.4 installed package
**Purpose**: Complete API reference for BlockPoolBatchEngine implementation (Sprint 2)
**Package Location**: `/Users/dev_user/.pyenv/versions/3.12.0/lib/python3.12/site-packages/mlx_lm/`

---

## Table of Contents

1. [BatchGenerator Class](#batchgenerator-class)
2. [Response Classes](#response-classes)
3. [Batch Class](#batch-class)
4. [Cache Classes](#cache-classes)
5. [Key Functions](#key-functions)
6. [Sampling & Logits Processing](#sampling--logits-processing)
7. [Code Examples](#code-examples)
8. [Gotchas & Notes](#gotchas--notes)

---

## BatchGenerator Class

**Location**: `mlx_lm/generate.py` (lines 920-1246)

The BatchGenerator is the core batching engine for efficient multi-request inference.

### Constructor

```python
def __init__(
    self,
    model,
    max_tokens: int = 128,
    stop_tokens: Optional[set] = None,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
    completion_batch_size: int = 32,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[
        Callable[[List[Tuple[int, int, int]]], None]
    ] = None,
)
```

**Parameters**:
- `model` (nn.Module): The language model to use for generation
- `max_tokens` (int): Default maximum tokens to generate per request. Default: 128
- `stop_tokens` (Optional[set]): Set of token IDs that stop generation. Default: None
- `sampler` (Optional[Callable]): Custom sampler function (takes logprobs, returns tokens). Default: argmax
- `logits_processors` (Optional[List]): List of logits processors for the batch. Default: []
- `completion_batch_size` (int): Maximum batch size for completion/decoding. Default: 32
- `prefill_batch_size` (int): Batch size for prefill/prompt processing. Default: 8
- `prefill_step_size` (int): Number of tokens to process per prefill step. Default: 2048
- `prompt_progress_callback` (Optional[Callable]): Callback for prompt progress: `(uid, processed, total)`. Default: None

**Internal State**:
- `unprocessed_prompts` (List): Queue of prompts waiting to be processed
- `active_batch` (Batch): Currently active batch being decoded
- `uid_count` (int): Counter for unique request IDs
- `_stats` (BatchStats): Accumulated statistics
- `_old_wired_limit` (int): Saved wired memory limit (for Metal)

**Important Notes**:
- Sets wired memory limit on Metal devices automatically
- `completion_batch_size` must be >= `prefill_batch_size`
- Manages a single active batch and a queue of unprocessed prompts

---

### Methods

#### insert()

```python
def insert(
    self,
    prompts,
    max_tokens: Union[List[int], int, None] = None,
    caches=None,
    samplers: list | None = None,
    logits_processors: list | None = None,
)
```

**Purpose**: Insert new prompts into the batch generation queue.

**Parameters**:
- `prompts` (List[List[int]]): List of tokenized prompts (each prompt is a list of token IDs)
- `max_tokens` (Union[List[int], int, None]): Max tokens per prompt. Can be:
  - `None`: Use default `self.max_tokens` for all
  - `int`: Use this value for all prompts
  - `List[int]`: Per-prompt max tokens
- `caches` (Optional[List[List[Any]]]): Pre-computed KV caches for each prompt. Default: None (creates new caches)
- `samplers` (Optional[List]): Per-prompt custom samplers. Default: None (use default sampler)
- `logits_processors` (Optional[List]): Per-prompt logits processors. Default: None (use default)

**Returns**:
- `List[int]`: List of unique IDs (UIDs) assigned to each prompt

**Behavior**:
1. Assigns unique UID to each prompt
2. Creates default caches if not provided (via `cache.make_prompt_cache(self.model)`)
3. Sorts prompts by length (ascending) for efficient batching
4. Adds to `unprocessed_prompts` queue

**Example**:
```python
prompts = [[1, 2, 3], [4, 5, 6, 7]]
uids = generator.insert(prompts, max_tokens=100)
# Returns: [0, 1]
```

---

#### next()

```python
def next(self)
```

**Purpose**: Generate the next batch of tokens for all active requests.

**Returns**:
- `List[BatchGenerator.Response]`: List of Response objects, one per active request

**Behavior**:
1. If room in `completion_batch_size`, processes prompts from `unprocessed_prompts` in batches of `prefill_batch_size`
2. For existing active batch, generates next token for each request
3. Returns responses with tokens, logprobs, finish_reason
4. Automatically removes completed requests (stop token or max_tokens reached)
5. Updates internal stats

**Response Fields** (see BatchGenerator.Response):
- `uid` (int): Request unique ID
- `token` (int): Generated token
- `logprobs` (mx.array): Log probabilities for this token
- `finish_reason` (Optional[str]): "stop", "length", or None
- `prompt_cache` (Callable): Function that returns the KV cache when called

**Key Implementation Details**:
- Uses `_process_prompts()` for prefill
- Uses `_step()` for decoding
- Manages batch filtering when requests complete
- Returns empty list when no more work

**Example**:
```python
while responses := generator.next():
    for r in responses:
        print(f"UID {r.uid}: token={r.token}, done={r.finish_reason}")
```

---

#### remove()

```python
def remove(self, uids: List[int])
```

**Purpose**: Remove requests from both active batch and unprocessed queue.

**Parameters**:
- `uids` (List[int]): List of request UIDs to remove

**Behavior**:
- Filters active batch to exclude specified UIDs
- Removes from unprocessed prompts queue
- If active batch becomes empty, sets it to None

---

#### stats()

```python
def stats(self) -> BatchStats
```

**Purpose**: Get accumulated generation statistics.

**Returns**:
- `BatchStats`: Statistics object with:
  - `prompt_tokens` (int): Total prompt tokens processed
  - `prompt_tps` (float): Prompt processing tokens/sec
  - `prompt_time` (float): Total prompt processing time (seconds)
  - `generation_tokens` (int): Total tokens generated
  - `generation_tps` (float): Generation tokens/sec
  - `generation_time` (float): Total generation time (seconds)
  - `peak_memory` (float): Peak memory usage (GB)

---

#### close()

```python
def close(self)
```

**Purpose**: Clean up resources (reset wired memory limit on Metal).

**Behavior**:
- Synchronizes generation stream
- Restores original wired limit
- Called automatically on `__del__`

---

## Response Classes

### GenerationResponse

**Location**: `mlx_lm/generate.py` (lines 259-287)

Used by `stream_generate()` for single-request streaming.

```python
@dataclass
class GenerationResponse:
    text: str                    # Decoded text segment
    token: int                   # Token ID
    logprobs: mx.array          # Log probabilities vector
    from_draft: bool            # Whether from draft model (speculative decoding)
    prompt_tokens: int          # Number of prompt tokens
    prompt_tps: float           # Prompt tokens per second
    generation_tokens: int      # Number of generated tokens so far
    generation_tps: float       # Generation tokens per second
    peak_memory: float          # Peak memory in GB
    finish_reason: Optional[str] = None  # "length", "stop", or None
```

---

### BatchGenerator.Response

**Location**: `mlx_lm/generate.py` (lines 921-928)

Used by `BatchGenerator.next()` for batched generation.

```python
@dataclass
class Response:
    uid: int                           # Unique request ID
    token: int                         # Generated token ID
    logprobs: mx.array                 # Log probabilities for this token
    finish_reason: Optional[str]       # "stop", "length", or None
    prompt_cache: Callable[[], List[Any]]  # Function returning KV cache
```

**Important**: `prompt_cache` is a **callable** that returns the cache when invoked, not the cache itself.

---

### BatchResponse

**Location**: `mlx_lm/generate.py` (lines 821-834)

Used by `batch_generate()` for batch processing.

```python
@dataclass
class BatchResponse:
    texts: List[str]                # Generated text for each prompt
    stats: BatchStats               # Generation statistics
    caches: Optional[List[List[Any]]]  # KV caches if return_prompt_caches=True
```

---

### BatchStats

**Location**: `mlx_lm/generate.py` (lines 797-819)

```python
@dataclass
class BatchStats:
    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0
```

---

## Batch Class

**Location**: `mlx_lm/generate.py` (lines 836-878)

Internal batch representation used by BatchGenerator.

```python
@dataclass
class Batch:
    uids: List[int]                     # Request UIDs
    y: mx.array                         # Current tokens (shape: [B])
    logprobs: mx.array                  # Log probabilities (list of arrays)
    max_tokens: List[int]               # Max tokens per request
    num_tokens: List[int]               # Tokens generated per request
    cache: List[Any]                    # Batch KV cache
    samplers: List[Any]                 # Per-request samplers
    logits_processors: List[Any]        # Per-request logits processors
    tokens: List[mx.array]              # Full token history per request
```

**Methods**:
- `__len__()`: Returns number of requests in batch
- `filter(keep_idx)`: Keep only specified indices (in-place)
- `extend(other)`: Extend with another batch (in-place)
- `extract_cache(idx)`: Extract KV cache for specific index

---

## Cache Classes

**Location**: `mlx_lm/models/cache.py`

### Key Cache Types

#### KVCache (Standard)

```python
class KVCache:
    step = 256  # Allocation step size

    def __init__(self)

    @property
    def state  # Returns (keys, values)

    def update_and_fetch(self, keys, values)  # Update and return cache
    def size(self) -> int  # Current cache size
    def empty(self) -> bool  # Whether cache is empty
    def is_trimmable(self) -> bool  # Can be trimmed
    def trim(self, n: int) -> int  # Trim n tokens
    def to_quantized(self, group_size=64, bits=4) -> QuantizedKVCache
    def make_mask(self, N, return_array, window_size)
```

**Shape**: `(batch, n_kv_heads, seq_len, head_dim)`

---

#### BatchKVCache

```python
class BatchKVCache:
    step = 256

    def __init__(self, left_padding: List[int])  # Expects left-padded inputs

    def update_and_fetch(self, keys, values)
    def prepare(self, *, left_padding=None, lengths=None, right_padding=None)
    def finalize(self)  # Converts right-padding to left-padding
    def filter(self, batch_indices)  # Keep only specified batch indices
    def extend(self, other)  # Merge with another BatchKVCache
    def extract(self, idx) -> KVCache  # Extract single-request cache

    @classmethod
    def merge(cls, caches: List[KVCache]) -> BatchKVCache
```

**Critical**: BatchKVCache expects **left-padded** inputs!

**Example**:
```python
# Prompts: [1,3,5], [7], [2,6,8,9]
# Padded:  [0,1,3,5], [0,0,0,7], [2,6,8,9]
# left_padding = [1, 3, 0]
cache = BatchKVCache(left_padding=[1, 3, 0])
```

---

#### RotatingKVCache

```python
class RotatingKVCache:
    def __init__(self, max_size: int, keep: int = 0)
```

**Purpose**: Sliding window cache that maintains last `max_size` tokens, preserving first `keep` tokens.

---

#### QuantizedKVCache

```python
class QuantizedKVCache:
    def __init__(self, group_size: int = 64, bits: int = 8)
```

**Purpose**: Quantized cache for memory efficiency.

---

### Cache Helper Functions

```python
def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None
) -> List[Any]:
    """Create cache for model (uses model.make_cache() if available)"""

def save_prompt_cache(
    file_name: str,
    cache: List[Any],
    metadata: Dict[str, str] = {}
)

def load_prompt_cache(
    file_name: str,
    return_metadata: bool = False
) -> Union[List[Any], Tuple[List[Any], Dict[str, str]]]

def can_trim_prompt_cache(cache: List[Any]) -> bool

def trim_prompt_cache(cache: List[Any], num_tokens: int) -> int
```

---

## Key Functions

### generate_step()

**Location**: `mlx_lm/generate.py` (lines 297-461)

```python
def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]
```

**Purpose**: Low-level token generator for single request.

**Yields**: `(token, logprobs)` tuples

---

### stream_generate()

**Location**: `mlx_lm/generate.py` (lines 640-737)

```python
def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]
```

**Purpose**: High-level streaming generation for single request.

**Yields**: `GenerationResponse` objects with decoded text and metadata

---

### generate()

**Location**: `mlx_lm/generate.py` (lines 739-782)

```python
def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str
```

**Purpose**: Complete generation (non-streaming) for single request.

**Returns**: Full generated text as string

---

### batch_generate()

**Location**: `mlx_lm/generate.py` (lines 1248-1325)

```python
def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    **kwargs,
) -> BatchResponse
```

**Purpose**: High-level batch generation (wrapper around BatchGenerator).

**Parameters**:
- `prompts`: List of tokenized prompts
- `prompt_caches`: Pre-computed caches (not updated in-place)
- `max_tokens`: Max tokens per prompt (int or list)
- `verbose`: Print progress
- `return_prompt_caches`: Include caches in response
- `logits_processors`: Per-prompt logits processors
- `**kwargs`: Passed to BatchGenerator constructor

**Returns**: `BatchResponse` with texts, stats, and optional caches

**Example**:
```python
response = batch_generate(
    model,
    tokenizer,
    prompts=[[1,2,3], [4,5,6]],
    max_tokens=50,
    verbose=True,
    completion_batch_size=32,
    prefill_batch_size=8
)
print(response.texts)  # List of generated strings
print(response.stats)  # BatchStats object
```

---

## Sampling & Logits Processing

### make_sampler()

**Location**: `mlx_lm/sample_utils.py` (lines 10-69)

```python
def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]
```

**Purpose**: Create a sampler function that takes logprobs and returns token IDs.

**Returns**: Sampler function with signature `(logprobs: mx.array) -> mx.array`

**Special Values**:
- `temp = 0.0`: Deterministic (argmax)
- `temp > 0.0`: Stochastic sampling

**Example**:
```python
sampler = make_sampler(temp=0.8, top_p=0.9, top_k=50)
token = sampler(logprobs)  # logprobs shape: [batch, vocab_size]
```

---

### make_logits_processors()

**Location**: `mlx_lm/sample_utils.py` (lines 72-108)

```python
def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
) -> List[Callable[[mx.array, mx.array], mx.array]]
```

**Purpose**: Create logits processors for biasing and repetition penalty.

**Returns**: List of processors with signature `(tokens, logits) -> logits`

**Parameters**:
- `logit_bias`: Dictionary mapping token_id -> bias value
- `repetition_penalty`: Penalty for repeated tokens (>1.0 discourages, <1.0 encourages)
- `repetition_context_size`: How many previous tokens to consider

**Example**:
```python
processors = make_logits_processors(
    logit_bias={50256: -100.0},  # Suppress EOS token
    repetition_penalty=1.2,
    repetition_context_size=50
)
```

---

## Code Examples

### Example 1: Basic BatchGenerator Usage

```python
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Create generator
sampler = make_sampler(temp=0.8, top_p=0.95)
generator = BatchGenerator(
    model,
    max_tokens=100,
    stop_tokens=tokenizer.eos_token_ids,
    sampler=sampler,
    completion_batch_size=32,
    prefill_batch_size=8,
)

# Tokenize prompts
prompts = [
    tokenizer.encode("What is AI?"),
    tokenizer.encode("Explain Python."),
    tokenizer.encode("Write a poem."),
]

# Insert prompts
uids = generator.insert(prompts, max_tokens=[50, 75, 100])

# Collect results
results = {uid: [] for uid in uids}
while responses := generator.next():
    for r in responses:
        if r.finish_reason != "stop":
            results[r.uid].append(r.token)
        if r.finish_reason:
            print(f"UID {r.uid} finished: {r.finish_reason}")

# Decode results
for uid in uids:
    text = tokenizer.decode(results[uid])
    print(f"UID {uid}: {text}")

# Get stats
stats = generator.stats()
print(f"Generation: {stats.generation_tps:.2f} tok/s")

# Clean up
generator.close()
```

---

### Example 2: Dynamic Request Management

```python
generator = BatchGenerator(model, max_tokens=100, stop_tokens=set([2]))

# Initial batch
uids1 = generator.insert([[1, 2, 3], [4, 5, 6]])

# Process 5 steps
for _ in range(5):
    responses = generator.next()
    for r in responses:
        print(f"UID {r.uid}: token {r.token}")

# Add more requests while processing
uids2 = generator.insert([[7, 8, 9, 10]])

# Continue processing (now includes new request)
while responses := generator.next():
    for r in responses:
        if r.finish_reason:
            print(f"UID {r.uid} completed")

# Remove specific requests
generator.remove([uids1[0]])  # Cancel first request
```

---

### Example 3: Per-Request Custom Samplers

```python
from mlx_lm.sample_utils import make_sampler

# Different samplers for different requests
sampler1 = make_sampler(temp=0.3)  # Conservative
sampler2 = make_sampler(temp=1.0, top_p=0.9)  # Creative
sampler3 = make_sampler(temp=0.0)  # Deterministic

prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

generator = BatchGenerator(model, max_tokens=50, stop_tokens=set([2]))
uids = generator.insert(
    prompts,
    samplers=[sampler1, sampler2, sampler3]
)

# Each request uses its own sampler
while responses := generator.next():
    for r in responses:
        print(f"UID {r.uid}: {r.token}")
```

---

### Example 4: Using Prompt Caches

```python
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache

# Create and populate cache
cache = make_prompt_cache(model)
# ... run prefill to populate cache ...
save_prompt_cache("my_cache.safetensors", cache, metadata={"model": "llama-3.2"})

# Later: reuse cache
from mlx_lm.models.cache import load_prompt_cache
cache, metadata = load_prompt_cache("my_cache.safetensors", return_metadata=True)

# Use with generator
prompts = [[1, 2, 3]]  # Suffix prompt
uids = generator.insert(prompts, caches=[cache])
```

---

### Example 5: Logits Processors (Repetition Penalty)

```python
from mlx_lm.sample_utils import make_logits_processors

processors = make_logits_processors(
    repetition_penalty=1.2,
    repetition_context_size=50,
    logit_bias={50256: -100.0}  # Suppress specific token
)

generator = BatchGenerator(
    model,
    max_tokens=100,
    logits_processors=processors,  # Applied to all requests
    stop_tokens=set([2])
)

# Or per-request processors
prompts = [[1, 2, 3], [4, 5, 6]]
uids = generator.insert(
    prompts,
    logits_processors=[processors, []]  # First uses processors, second doesn't
)
```

---

## Gotchas & Notes

### 1. **Padding Convention**

BatchKVCache expects **left-padded** inputs during prefill:

```python
# CORRECT
prompts = [
    [0, 1, 3, 5],    # left_padding = 1
    [0, 0, 0, 7],    # left_padding = 3
    [2, 6, 8, 9]     # left_padding = 0
]

# WRONG - will fail
prompts = [
    [1, 3, 5, 0],    # right-padded
    [7, 0, 0, 0],
    [2, 6, 8, 9]
]
```

### 2. **Prompt Sorting**

`insert()` sorts prompts by length (ascending). UIDs are assigned **before** sorting, so UID order doesn't match sorted order.

### 3. **Cache Ownership**

- `insert(caches=...)`: Caches are **modified in-place**
- `batch_generate(prompt_caches=...)`: Caches are **not modified** (copies made internally)
- `BatchGenerator.Response.prompt_cache`: Returns a **callable**, not the cache itself

```python
# CORRECT
response = generator.next()[0]
cache = response.prompt_cache()  # Call it to get cache

# WRONG
cache = response.prompt_cache  # This is a function, not the cache
```

### 4. **Completion vs Prefill Batch Size**

- `prefill_batch_size`: How many new prompts to process together
- `completion_batch_size`: Maximum total active requests (must be >= prefill_batch_size)

Example: `prefill_batch_size=8, completion_batch_size=32`
- Processes 8 prompts at a time
- Can have up to 32 requests generating simultaneously

### 5. **Stop Tokens**

Stop tokens are checked **after** generation, not before:

```python
generator = BatchGenerator(model, stop_tokens={2})
# If next token is 2, it's yielded with finish_reason="stop"
# The token 2 IS included in the output
```

### 6. **Logits Processors Signature**

Logits processors take `(tokens, logits)` and return processed logits:

```python
def custom_processor(tokens: mx.array, logits: mx.array) -> mx.array:
    # tokens: [seq_len] - all tokens generated so far
    # logits: [batch, vocab_size] - current logits
    return logits  # Modified logits
```

### 7. **Sampler Signature**

Samplers take logprobs and return token IDs:

```python
def custom_sampler(logprobs: mx.array) -> mx.array:
    # logprobs: [batch, vocab_size] - log probabilities
    # Returns: [batch] - sampled token IDs
    return mx.argmax(logprobs, axis=-1)
```

### 8. **Generator Cleanup**

Always call `generator.close()` or use context manager pattern:

```python
# Manual cleanup
generator = BatchGenerator(model)
try:
    # ... use generator ...
finally:
    generator.close()

# Or rely on __del__ (called automatically)
```

### 9. **Stream Management**

All generation happens on `generation_stream`. Don't manually eval arrays from generator:

```python
# WRONG
responses = generator.next()
mx.eval(responses[0].token)  # Don't do this

# RIGHT
responses = generator.next()
# Arrays are auto-evaluated asynchronously
```

### 10. **Empty Responses**

`next()` returns empty list when done:

```python
while responses := generator.next():
    # Process responses
    pass
# Loop exits when responses == []
```

### 11. **Per-Request vs Global Settings**

Some settings can be per-request or global:

**Per-Request** (via `insert()`):
- `max_tokens`
- `samplers`
- `logits_processors`
- `caches`

**Global** (via `__init__()`):
- `stop_tokens`
- `completion_batch_size`
- `prefill_batch_size`
- `prefill_step_size`

### 12. **Quantized KV Cache**

To use quantized KV cache, convert after prefill:

```python
cache = make_prompt_cache(model)
# ... prefill ...
if hasattr(cache[0], 'to_quantized'):
    cache[0] = cache[0].to_quantized(group_size=64, bits=4)
```

Or use quantization parameters in generation functions:

```python
generate(
    model,
    tokenizer,
    prompt,
    kv_bits=4,
    kv_group_size=64,
    quantized_kv_start=1000  # Start quantizing after 1000 tokens
)
```

### 13. **Prefill Step Size**

Larger `prefill_step_size` = faster prefill but more memory. Common values:
- 512: Conservative (low memory)
- 2048: Default (balanced)
- 4096: Aggressive (high memory, faster)

### 14. **Metal Device Memory**

On macOS with Metal, BatchGenerator automatically sets wired limit to max recommended. If you get memory warnings:
- Reduce `completion_batch_size`
- Reduce `prefill_batch_size`
- Use smaller model
- Use quantized KV cache

---

## Performance Tips

### 1. **Batch Size Tuning**

```python
# For throughput: larger batches
generator = BatchGenerator(
    model,
    completion_batch_size=64,
    prefill_batch_size=16
)

# For latency: smaller batches
generator = BatchGenerator(
    model,
    completion_batch_size=16,
    prefill_batch_size=4
)
```

### 2. **Prompt Length Grouping**

Sort prompts by length before inserting to minimize padding:

```python
prompts = [[1,2,3], [4,5,6,7,8], [9,10]]
prompts.sort(key=len)  # [9,10], [1,2,3], [4,5,6,7,8]
uids = generator.insert(prompts)
```

### 3. **Cache Reuse**

For repeated prefixes, save and reuse caches:

```python
# System prompt cache
system_cache = make_prompt_cache(model)
# ... prefill with system prompt ...
save_prompt_cache("system_cache.safetensors", system_cache)

# Reuse for all requests
for user_prompt in user_prompts:
    cache = load_prompt_cache("system_cache.safetensors")
    uids = generator.insert([user_prompt], caches=[cache])
```

---

## Version Compatibility

**mlx_lm 0.30.4** (January 2026):
- Python 3.8+
- mlx >= 0.20.0
- transformers >= 4.0.0

**Breaking Changes from 0.19.x**:
- BatchKVCache now requires explicit `left_padding` parameter
- `prepare()` and `finalize()` methods added to cache API
- `extract()` method added for single-request cache extraction

---

## Related Files

- **generate.py**: Main generation logic (1457 lines)
- **cache.py**: KV cache implementations (1305 lines)
- **sample_utils.py**: Sampling and logits processing (310 lines)
- **tokenizer_utils.py**: Tokenizer wrappers
- **utils.py**: Model loading and utilities

---

## Further Reading

- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Batched Generation Tutorial](https://github.com/ml-explore/mlx-lm/tree/main/examples)

---

**Document Status**: Complete
**Last Updated**: 2026-01-24
**Reviewed**: Source code v0.30.4
