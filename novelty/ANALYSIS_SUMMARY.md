# Papers Analysis: Executive Summary

**Date:** 2026-02-04
**Papers Analyzed:** 7 research papers (2025-2026)
**Output Documents:** PAPERS_ANALYSIS.md (532 lines) + STYLE_GUIDE_INSIGHTS.md (411 lines)

---

## What Was Analyzed

### Papers Reviewed

1. **RAG-DCache (Lee et al., 2025)**
   - Disk-based KV cache for RAG systems
   - Multi-instance serving with shared cache
   - Relevance: Persistent working memory patterns

2. **XQuant (2025)**
   - Quantize input activations (X) not KV pairs
   - 10-12.5× compression with minimal accuracy loss
   - Relevance: Alternative representations for compression

3. **KVSwap (2025)**
   - Disk-aware KV cache offloading for on-device inference
   - Quality-aware grouping for I/O efficiency
   - Relevance: Multi-tier hierarchical memory management

4. **TRIM-KV (2025)**
   - **Learned retention gates** for token importance
   - Exponential decay models diminishing utility
   - Relevance: Direct mapping to semantic block importance scoring

5. **EvicPress (2025)**
   - **Joint compression + eviction** optimization
   - Per-context customization beats global policy
   - Relevance: Multi-factor optimization (quality/latency trade-offs)

6. **FastKVzip (2026)**
   - Gated KV eviction approach
   - (Content not fully accessible in provided file)

7. **vLLM-MLX (Barrios, 2026)**
   - Native Apple Silicon inference with continuous batching
   - Content-based prefix caching for multimodal models
   - Relevance: Hardware-specific optimization + cache reuse strategy

---

## Key Technical Findings

### Trend 1: Shift from Static to Adaptive Policies

**Before (2024 and earlier):** Fixed heuristics (H2O attention tracking, LRU eviction)

**Now (2025-2026):** Per-context/per-token/per-layer learned customization
- TRIM-KV: Per-token learned gates
- EvicPress: Per-context compression/eviction tuples
- KVSwap: Quality-aware dynamic grouping

**Implication for Semantic:** One-size-fits-all block policies will underperform. Need adaptive importance scoring.

### Trend 2: Joint Optimization Over Single-Factor

**EvicPress insight:** Separate compression and eviction is suboptimal.
- Compression-only: High speed, low quality
- Eviction-only: High quality, high latency
- Joint optimization: Both high quality AND low latency

**Generalization:** Problems rarely have single optimal dimension. Utility function unifies multi-factor trade-offs.

**Implication for Semantic:** Design for quality/throughput/memory trade-offs, not single metric.

### Trend 3: Learned Importance Beats Attention-Guided

**TRIM-KV key finding:** Tokens learned to be important at creation time, not based on recent attention.

**Why heuristics fail:** Long-horizon reasoning where tokens matter much later (not reflected in recent attention patterns).

**Learned gates naturally recover:**
- Sink tokens (first tokens encoding topic)
- Sliding windows (recent context)
- Gist tokens (summary information)

Without explicit design, emergent behavior matches human intuition.

**Implication for Semantic:** Invest in learned importance scoring. It's more robust than heuristics and enables interpretability.

### Trend 4: Hierarchical Storage is Standard

**Multi-tier setups:** GPU → CPU → Disk
- RAG-DCache: GPU/CPU memory → disk
- EvicPress: GPU → CPU → SSD (3-tier)
- KVSwap: Device → disk
- Semantic target: GPU → CPU → persistent block store

**Optimization focus:** I/O efficiency (grouping, prefetching, batching)

### Trend 5: Empirical Validation is Comprehensive

**Standard evaluation:**
- 3-5 models of varying sizes (0.6B to 30B parameters)
- 3-5 datasets/benchmarks (GSM8K, LongBench, etc.)
- 2-3 baseline methods
- Ablation studies on each major component
- Qualitative analysis (visualizations, emergent behaviors)

---

## Writing Style Insights for Novelty.md Rewrite

### Strongest Patterns

1. **Problem-first opening** (TRIM-KV, EvicPress)
   - Concrete metric (10-20% latency increase, O(L·N²·D) complexity)
   - Why existing solutions fail
   - Solution in 1-2 sentences

2. **Motivating counterexample** (EvicPress Figure 2)
   - Shows why single-factor approaches are suboptimal
   - Concrete scenario with different outcomes
   - Leads naturally to joint optimization

3. **Trade-off visualization**
   - Quality (y-axis) vs latency/throughput (x-axis)
   - Methods labeled on plot
   - Clear dominance relationships visible

4. **Headline results first**
   - "10-20% TTFT reduction" before explaining how
   - Multi-dimensional comparisons (model × benchmark × metric)
   - Specific baselines named (H2O, SnapKV, LRU)

5. **Design justification**
   - "Exponential decay models diminishing utility of old context"
   - "Grouping KV entries amortizes I/O overhead"
   - Never just "we chose X because it's better"

### Terminology to Standardize

| Term | Meaning | Example |
|------|---------|---------|
| TTFT | Prefill phase only | "Reduced TTFT by 15%" |
| Compression ratio | 10× (not 90%) | "12.5× compression" |
| Memory savings | GB or % | "8GB → 0.64GB" |
| Overhead | Always quantified | "1% CPU overhead" not "negligible" |
| Quality drop | Delta with context | "3% drop on GSM8K" |
| Speedup | With baseline implied | "28× on repeated images" |
| Eviction | Move to lower tier | Not deletion |
| Retention | What we keep | Not "cache" |

### Figures to Include

1. **Problem illustration** - Architecture or scenario showing why naive approach fails
2. **System overview** - Block diagram of components
3. **Trade-off plot** - Quality vs latency with method labels
4. **Scaling curve** - Throughput vs concurrency/model size
5. **Ablation table** - Each component impact
6. **Qualitative analysis** - Learned patterns (if applicable)

---

## Semantic Architecture Alignment

### Most Relevant Papers (Priority Order)

1. **TRIM-KV** (Very High)
   - Learned token importance → semantic block importance
   - Exponential decay for diminishing utility
   - Globally optimized (not greedy per-layer)
   - Emergent patterns align with intuition

2. **EvicPress** (High)
   - Joint compression + eviction → multi-factor optimization
   - Per-context customization → per-block decisions
   - Utility function → unified scoring metric
   - Periodic re-profiling → workload adaptation

3. **RAG-DCache** (High)
   - Persistent disk-based cache → external working memory
   - Proactive prefetching → batch-optimized cache generation
   - Multi-instance shared cache → distributed block management
   - Quantified locality → predictable access patterns

4. **KVSwap** (High)
   - Grouped I/O for efficiency → block-level memory hierarchy
   - Quality-aware grouping → per-block customization
   - Compression for prediction → separate concerns
   - Designed for decoding stage (not just prefill)

5. **vLLM-MLX** (Moderate)
   - Content-based prefix caching → cache key identity mechanism
   - Continuous batching → multi-request optimization
   - Hardware-specific optimization → unified memory exploitation

### Concrete Mappings to Semantic

**TRIM-KV → Block Importance Scoring**
```
TRIM-KV: g(token) → retention_score ∈ [0,1]
         decay: α_ti = β_i^(t-i)
         eviction: remove lowest-scoring tokens

Semantic: g(block) → importance_score ∈ [0,1]
          decay: importance_t = score_0 × decay_rate^(t-t_created)
          eviction: remove lowest-scoring blocks when budget exceeded
```

**EvicPress → Utility Function for Trade-offs**
```
EvicPress: U(context, tier, compression_method, ratio) → single score

Semantic: U(block, tier, compression_method, ratio) → single score
          Placement: maximize Σ U per storage tier
          Adaptation: re-profile after N batches
```

**KVSwap → Hierarchical Memory Management**
```
KVSwap: Grouped KV prediction + compression + hierarchy

Semantic: Grouped block transfer + compression + GPU/CPU/disk hierarchy
          I/O optimization through batching and prefetching
```

---

## Recommendations for Novelty.md and Style Guide

### Structure Changes

**Current approach:** May emphasize system components equally

**Recommended approach:**
1. **Problem** - Concrete metrics (latency, memory, throughput bottlenecks)
2. **Insight** - Why existing approaches fail + your core idea
3. **Design** - System components with clear responsibilities
4. **Results** - Headline wins, trade-off analysis, ablation
5. **Positioning** - Table vs H2O, SnapKV, EvicPress, TRIM-KV

### Content Additions

- [ ] Problem opening with O() complexity or user-visible metric
- [ ] Motivating example showing why single-factor optimization fails
- [ ] Design justification for each major component
- [ ] Trade-off plots (quality vs latency, throughput vs quality)
- [ ] Ablation studies validating each innovation
- [ ] Comparison table vs competing approaches
- [ ] Qualitative analysis (learned patterns, emergent behaviors)

### Terminology Updates

- Standardize on TRIM-KV / EvicPress vocabulary
- Define retention/eviction clearly
- Use "block" consistently for your abstraction level
- Quantify all overhead claims
- Name specific baselines (not "prior work")

### Benchmarking Setup

Adopt 2025 standard:
- 3-5 models (0.6B to 30B)
- 3-5 datasets (GSM8K, LongBench, etc.)
- Metrics: TTFT, end-to-end latency, throughput, quality
- Baselines: H2O, SnapKV, LRU (eviction); KVQuant, KIVI (compression)

---

## Document Organization

Two comprehensive analysis documents were created:

### 1. **PAPERS_ANALYSIS.md** (532 lines, 25KB)
Detailed technical analysis of all 7 papers covering:
- Core innovation & problem solved
- Technical approach with system components
- Key quantitative results
- System positioning vs related work
- Relevance to semantic paradigm
- Writing style observations
- Technical terminology used

**Use for:** Understanding technical details, architectural patterns, benchmarking approaches

### 2. **STYLE_GUIDE_INSIGHTS.md** (411 lines, 16KB)
Actionable guidance for writing and system design:
- Problem-first writing patterns with examples
- Result presentation best practices
- Terminology consistency guidelines
- Standard notation conventions
- Structural elements of strong papers
- Semantic paradigm alignment with concrete mappings
- Writing patterns template for novelty.md rewrite
- Metrics and benchmarking guidance

**Use for:** Rewriting novelty.md, creating style guide, benchmarking planning

---

## Next Steps

### For novelty.md Rewrite

1. Use "Problem-first opening" structure from STYLE_GUIDE_INSIGHTS
2. Add trade-off plot (quality vs latency) with method labels
3. Include ablation table for each major component
4. Adopt TRIM-KV / EvicPress terminology
5. Add qualitative analysis section
6. Create positioning table vs named baselines

### For Style Guide

1. Adopt terminology table from STYLE_GUIDE_INSIGHTS (TTFT, compression ratio, etc.)
2. Add "Writing Patterns" section with good/bad examples
3. Include "Figures to Create" checklist
4. Add "Results Presentation" section
5. Specify "Benchmarking Standards" (models, datasets, baselines)

### For Future Work

- Implement TRIM-KV-style learned importance gates for blocks
- Design per-context optimization similar to EvicPress
- Explore grouped I/O batching like KVSwap
- Consider persistent cache layer like RAG-DCache
- Add content-based hashing for block identity (like vLLM-MLX)

---

**Analysis Complete**

All insights extracted and organized for semantic style guide and novelty.md rewrite.

Files created:
- `/Users/dev_user/semantic/novelty/PAPERS_ANALYSIS.md` (comprehensive technical analysis)
- `/Users/dev_user/semantic/novelty/STYLE_GUIDE_INSIGHTS.md` (actionable writing and design guidance)
- `/Users/dev_user/semantic/novelty/ANALYSIS_SUMMARY.md` (this file)

