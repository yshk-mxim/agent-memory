# Style Guide & Semantic Paradigm Insights from Papers Analysis

**Extracted from 7 2025-2026 KV Cache Research Papers**

---

## Part 1: Writing Style Patterns for Technical Papers

### 1. Problem-First Opening Pattern

**Winning approach** (used by TRIM-KV, EvicPress, KVSwap):
1. State concrete problem with metrics or O() complexity
2. Highlight why existing solutions fail (specific scenario)
3. Introduce solution concept in 1-2 sentences
4. Elaborate design in structured sections

**Example structure** (TRIM-KV):
> "Memory and computation remain core bottlenecks in long-horizon LLM inference due to the quadratic cost of self-attention and the ever-growing key-value (KV) cache. Existing strategies [compression/eviction/heuristics] either incur high orchestration costs or rely on unreliable attention-based proxies of importance. We propose TRIM-KV, a novel approach that learns each token's intrinsic importance at creation time via a lightweight retention gate."

**Key elements:**
- Opens with O(quadratic) complexity or user-visible metric (10-20% latency increase)
- Names competing approaches explicitly
- "Unreliable" or "suboptimal" for prior work (EvicPress shows *why* with Figure 2)
- Solution stated concisely before deep dive

### 2. Results Presentation Patterns

**Most effective:**
- **Headline wins first**: "10-20% latency reduction" in abstract
- **Multi-dimensional tables**: Model (3-5 sizes) × Dataset (3-5 benchmarks) × Metric
- **Trade-off plots**: Plot quality (y-axis) vs latency (x-axis) with method labels
- **Before/after narratives**: "Compression-only: 75% quality, 0.3s TTFT. Eviction-only: 100% quality, 2.4s TTFT. Joint: 100% quality, 0.5s TTFT" (EvicPress Figure 2)
- **Ablation on key components**: Remove each innovation, measure impact

**Anti-patterns:**
- Generic "faster" without baseline or percentage
- Single-metric focus (TTFT only, ignoring quality)
- Unqualified "overhead" without percentage (e.g., "negligible" = measured <1%)

### 3. Terminology Consistency

**Use across papers (adopt these):**

| Term | How Papers Use It | Guidance |
|------|------------------|----------|
| TTFT | Prefill phase latency only ("reduced TTFT by 15%") | Don't conflate with total latency |
| Compression ratio | "10× compression" (original/compressed), not percentage | Prefer X× notation |
| Memory savings | "12.5× memory compression" OR "8GB → 0.64GB" | Both acceptable; plural "savings" |
| Perplexity | "0.01 degradation" acceptable, ">0.1 noticeable" | Quantify degradation as delta |
| Overhead | "Negligible 1% CPU overhead" never just "negligible" | Always measure in %, µs, or GB |
| Quality drop | "3% quality drop tolerance" with context | Specify benchmark (GSM8K, etc.) |
| Speedup | "28× speedup on repeated queries" (baseline implied by context) | Include what's being compared |

### 4. Notation Standards

**Mathematical presentation patterns:**

- **Greek letters for learned quantities**: β (retention), α (mask), σ (threshold)
- **Subscript/superscript conventions**: β_i^(t-i) for time-dependent importance
- **Bold for vectors**: **x**, **k**, **v** (lowercase); matrices use capital **W**
- **O() notation**: Only for high-level complexity, not throughout paper
- **Equations**: Full derivation for novel concepts, reference existing for standard

**Example** (TRIM-KV style):
```
Retention-gated attention: ȳ_t = Σ_i [(exp(β_i^(t-i) q_t^T k_i)) / (Σ_j exp(β_j^(t-j) q_t^T k_j))] v_i
where β_i ∈ [0,1] is the retention score for token i
```

### 5. Structural Elements

**Sections that appear in all strong papers:**

1. **Introduction** (1-2 pages)
   - Problem statement with metrics
   - Why existing solutions fail
   - Contribution summary

2. **Preliminaries/Background** (0.5-1 page)
   - Math notation (KV cache equations)
   - Problem formulation (constrained optimization, if applicable)
   - Related work (brief; expanded later)

3. **Methodology** (2-3 pages)
   - System architecture diagram
   - Algorithm description (pseudocode if complex)
   - Design justification ("We use exponential decay because...")

4. **Experimental Setup**
   - Models tested (size range, families)
   - Datasets/benchmarks
   - Baseline methods
   - Hardware specifications

5. **Results**
   - Main results (table + narrative)
   - Ablation studies
   - Qualitative analysis (learned gate visualizations, etc.)

6. **Related Work** (1 page)
   - Organize by subarea (KV compression, eviction, offloading)
   - Clear positioning vs each category

7. **Limitations/Future Work**
   - Specific, not generic ("Future work: extend to speculative decoding" not "improve efficiency")

### 6. Figure/Table Best Practices

**From papers:**

- **Figure 1-2**: Problem illustration (architecture or motivating example)
- **Tables**: Model × Benchmark × Metric grid (comprehensive but compact)
- **Figure 3+**: Results visualizations
  - Trade-off plots (quality vs latency) with method labels
  - Scaling curves (throughput vs model size/concurrency)
  - Heatmaps (per-model, per-dataset results)
- **Ablation table**: Column = component, rows = datasets, values = metric change

**Captions:**
- Descriptive, not just labels ("Comparison of compression-only (0.3s TTFT, 75% quality) vs. joint approach (0.5s TTFT, 100% quality)" vs. "Results table")

### 7. Avoiding AI Slop in Research Writing

**Red flags found in analysis:**

1. **Vague efficiency claims** ("our method is more efficient")
   - **Fix**: "Reduces TTFT by 12-65% with <3% quality drop" (specific metrics)

2. **Unqualified "overhead"** ("adds negligible overhead")
   - **Fix**: "Adds <1% CPU overhead during inference" (measured quantity)

3. **Generic future work** ("optimize further in the future")
   - **Fix**: "Extend to speculative decoding to reduce latency variance" (concrete direction)

4. **Over-commenting obvious code** (if papers included code)
   - **Fix**: Let variable names be self-documenting; comment *why*, not *what*

5. **Redundant notation** (same concept with different symbols)
   - **Fix**: Define once (β for retention score), use consistently

6. **Weak comparisons** ("compared to baselines")
   - **Fix**: "Compared to H2O (attention-guided), SnapKV (recent attention), and NACL (learned prefix)" (name specific methods)

---

## Part 2: Semantic Paradigm Alignment

### Papers Most Relevant to Semantic Architecture

| Paper | Relevance | Key Concept | Maps To |
|-------|-----------|-------------|---------|
| **TRIM-KV** | **Very High** | Learned token importance (retention gates) | Block importance scoring in cache |
| **EvicPress** | **High** | Joint optimization of compression + eviction | Multi-factor trade-offs (quality/latency) |
| **RAG-DCache** | **High** | Persistent disk-based KV cache + locality | External working memory patterns |
| **KVSwap** | **High** | Grouped I/O for hierarchical storage | Block-level memory hierarchy |
| **vLLM-MLX** | **Moderate** | Prefix caching + content-based hashing | Cache key identity mechanism |
| **XQuant** | **Moderate** | Cross-layer structure exploitation | Potential layer-wise optimizations |
| **FastKVzip** | **Moderate** | (Content unavailable) | Likely gating for eviction |

### Conceptual Mappings

#### TRIM-KV → Semantic Block Importance

**TRIM-KV approach:**
```
Retention gate: g(x_t) → β_t ∈ [0,1]
Decay: α̅_ti = β_i^(t-i)  (exponential decay with age)
Inference: Evict tokens with lowest current score
```

**Semantic mapping:**
```
Block importance gate: g(block_content) → importance_score ∈ [0,1]
Decay: importance_t = importance_0 × decay^(t-t_created)
Eviction: Remove blocks with lowest importance when budget exceeded
```

**Advantages to adopt:**
- Exponential decay naturally models "diminishing utility"
- Learned globally (all layers jointly) beats greedy per-layer
- Emergent behaviors: naturally learns sink tokens, sliding windows
- No attention recomputation needed at inference

#### EvicPress → Semantic Quality/Latency Trade-offs

**EvicPress approach:**
```
Utility function: U(context_id, tier, compression_method, compression_ratio)
           → single score balancing quality + delay impact
Optimization: Greedy placement maximizing Σ U per storage tier
Adaptation: Periodic re-profiling to track query drift
```

**Semantic mapping:**
```
Utility for block placement: U(block_id, tier, block_compression_method, ratio)
                 → score balancing semantic_quality + throughput_impact
Optimization: Greedy or DP to maximize total utility across batches
Adaptation: Re-profile after N batches to capture workload drift
```

**Advantages to adopt:**
- Single unified metric enables comparison across diverse decisions
- Per-context customization beats global policy
- Periodic adaptation handles distribution shift
- Clear ROI: EvicPress achieves 2.19× faster TTFT

#### KVSwap → Semantic Hierarchical Memory

**KVSwap approach:**
```
Grouped KV prediction: Pack multiple KV entries per I/O
Compression: Low-rank K for prediction (full-precision V)
Hierarchy: GPU memory → disk with I/O grouping
```

**Semantic mapping:**
```
Grouped block transfer: Pack related blocks per I/O operation
Compression: Low-rank representation for prediction (full precision for final)
Hierarchy: GPU → CPU → Disk with block-level I/O optimization
```

**Advantages to adopt:**
- Grouping amortizes I/O overhead (key insight)
- Separate compression for prediction vs actual computation
- First method designed for decoding stage (not just prefill)

#### RAG-DCache → Semantic Persistent Working Memory

**RAG-DCache approach:**
```
Disk-based KV cache: Store precomputed KV for frequently-used documents
Proactive prefetch: Generate cache during query wait times
Locality: 50% of queries use 3.1-31.39% of documents
```

**Semantic mapping:**
```
Persistent block cache: Store important blocks on disk/secondary storage
Proactive batching: Pre-compute block KV during system idle
Locality: Track block reuse patterns across batch sequences
```

**Advantages to adopt:**
- Asynchronous generation exploits waiting periods
- Explicit multi-instance support (shared cache management)
- Quantified locality window enables prefetching strategy

---

## Part 3: Recommended Writing Patterns for Novelty.md Rewrite

### Structure Template

```markdown
# [System Name]: [One-Line Value Prop]

## Problem Statement
- Concrete metric (O(complexity) or user-visible latency)
- Why existing solutions fail (specific scenarios)
- Example: "Attention-guided eviction fails for long-horizon reasoning
           where tokens matter later without recent attention queries"

## Core Innovation
- [Concept Name]: [Brief description]
- Key insight: [Why this specific approach works]

## Technical Approach
### Component 1: [Name]
- Input/output
- Design choices & justification
- Key equations (1-2 per component)

### Component 2: [Name]
...

## System Integration
- How components interact
- Architecture diagram
- Workflow (3-5 steps)

## Key Results
| Model | Metric 1 | Metric 2 | Metric 3 |
|-------|----------|----------|----------|
| ... | ... | ... | ... |

- Headline comparison ("10-20% latency reduction")
- Trade-off analysis ("quality vs throughput")
- Ablation findings

## Positioning vs Related Work
| Aspect | Method A | Method B | [System] |
|--------|----------|----------|----------|
| Learned? | No (heuristic) | Greedy | Yes (global) |
| Per-context? | No | Per-layer | Yes |
| Multi-tier? | No | GPU/CPU | GPU/CPU/disk |

## Implications for Semantic
- Direct mapping to block importance
- Lessons on multi-factor optimization
- Potential adaptations or extensions
```

### Concrete Opening Examples (Adapt These)

**Problem-first opening (TRIM-KV-inspired):**
> "Memory remains a core bottleneck in long-context LLM serving, where the KV cache grows linearly with sequence length. Existing cache eviction methods rely on attention patterns, but this assumption breaks for long-horizon reasoning where tokens matter much later, without recent attention signals. We observe that [semantic insight about your system]."

**Motivation with counterexample (EvicPress-inspired):**
> "Consider two blocks: one compresses to 5% without quality loss, another degrades to 50% with minimal compression. A compression-only approach compresses both equally (suboptimal). An eviction-only approach favors the compression-sensitive block, leaving capacity unused. Joint optimization places the sensitive block in fast memory (compressed conservatively) and aggressively compresses the robust block elsewhere. Semantic implements this insight..."

### Terminology to Use Consistently

- **Block** instead of token/KV pair (for your system's abstraction level)
- **Importance score** instead of "relevance" or "criticality"
- **Decay rate** instead of "forgetting" (mathematical clarity)
- **Utility function** if optimizing multiple factors
- **Retention** for what you keep; **eviction** for what you remove
- **Throughput** in tokens/second (match vLLM-MLX, EvicPress)
- **Quality** with specific metric name (perplexity, accuracy, F1, etc.)

### Figure Types to Include

1. **Problem illustration**: Architecture under memory pressure (show why naive approach fails)
2. **System overview**: Block diagram of components (manager → scheduler → storage)
3. **Trade-off plot**: Quality (y-axis) vs throughput/latency (x-axis) with method labels
4. **Scaling curve**: Throughput vs concurrency or model size
5. **Ablation table**: Component present/absent × benchmark × metric
6. **Qualitative analysis**: Learned importance visualizations (if applicable)

---

## Part 4: Metrics & Benchmarking Guidance

### Standard Metrics Across Papers

**Throughput:**
- Tokens per second (preferred by vLLM-MLX, EvicPress)
- Requests per second (if batched)
- Scale: report both single-request and batched (N concurrent)

**Latency:**
- TTFT (prefill only): seconds or milliseconds
- End-to-end: for full generation sequence
- Percentiles: p50, p99 useful for variance

**Quality:**
- Task-specific (GSM8K pass@1, LongBench score, MMLU accuracy)
- Perplexity if applicable (report change, not absolute)
- Exact match or ROUGE if appropriate

**Memory:**
- Peak GPU memory: GB
- Total footprint: GPU + CPU + disk (if multi-tier)
- Compression ratio: 10× not "10%"

**Overhead (always measured):**
- CPU utilization: percentage
- I/O latency: milliseconds
- Framework overhead: as % of compute time

### Baseline Choices (Match Recent Papers)

- **Eviction baselines**: H2O, SnapKV, LRU
- **Compression baselines**: KVQuant, KIVI, KVZip
- **System baselines**: vLLM (standard), llama.cpp (if Apple Silicon)
- **Methods to compare**: Your method vs. compression-only, vs. eviction-only, vs. joint prior work

### Datasets to Report On (2025 standard)

- **Reasoning**: GSM8K, MATH, AIME (few-shot prompts)
- **Long-context**: LongBench, LongBenchV2, SCBench
- **Long-generation**: LongProc, procedural tasks
- **Chat**: LongMemEval, multi-turn conversations
- **General**: MMLU, HellaSwag (for compatibility check)

---

## Summary: What to Adopt from Papers

### Do

✓ Open with concrete problem (metric or O() complexity)
✓ Show why existing solutions fail with specific examples
✓ Present results with multiple dimensions (model × dataset × metric)
✓ Include ablation studies for each major component
✓ Use trade-off plots (quality vs latency, not just tables)
✓ Explain design choices ("exponential decay because diminishing utility...")
✓ Always quantify "overhead" (1%, 2.1 ms, etc.)
✓ Name baseline methods explicitly (H2O, SnapKV, not "prior work")
✓ Include qualitative analysis (learned patterns, emergent behaviors)
✓ Map to concrete user-visible improvements (TTFT reduction %)

### Don't

✗ Start with literature review; lead with problem
✗ Use "efficient" or "optimized" without metrics
✗ Report results without baselines or models specified
✗ Include more than 3 related work categories
✗ Hand-wave design choices ("we chose X because it's better")
✗ Use single metric; always trade-offs
✗ Mix notations (β vs b for same concept)
✗ Claim "negligible overhead" without measurement
✗ Generic future work; be specific (e.g., "speculative decoding")
✗ Forget ablation studies; they validate contributions

---

**End of Style Guide**

