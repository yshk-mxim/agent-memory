# Research Papers Analysis - Index & Quick Reference

**Analysis Date:** 2026-02-04
**Source:** 7 research papers (2025-2026) on KV cache optimization and LLM inference
**Analyst:** Claude (via semantic/novelty investigation)

---

## Three Output Documents

### 1. PAPERS_ANALYSIS.md (532 lines, 25KB)
**What:** Comprehensive technical deep-dive on all 7 papers

**Contents per paper:**
- Core technical innovation (problem + solution)
- Technical approach (algorithms, systems design)
- Key quantitative results
- System positioning vs related work
- Relevance to semantic paradigm
- Writing style observations
- Terminology used

**Best for:**
- Understanding technical details of each paper
- Learning architectural patterns (RAG-DCache, EvicPress, KVSwap)
- Reference on benchmarking approaches
- Studying writing style by examining real examples

**Key sections:**
- Executive Summary (overall findings)
- Cross-Paper Technical Patterns (5 key trends)
- Semantic Paradigm Alignment
- Terminology Glossary

---

### 2. STYLE_GUIDE_INSIGHTS.md (411 lines, 16KB)
**What:** Actionable guidance for writing technical papers and system design

**Part 1: Writing Style Patterns**
- Problem-first opening pattern (7 sections)
- Results presentation patterns
- Terminology consistency (table)
- Notation standards
- Structural elements (7 sections)
- Figure/table best practices
- Avoiding AI slop in research writing

**Part 2: Semantic Paradigm Alignment**
- Papers ranked by relevance (table)
- Conceptual mappings (4 key papers)
- Technical mappings to semantic architecture

**Part 3: Recommended Writing Patterns**
- Structure template for novelty.md rewrite
- Concrete opening examples (2 styles)
- Terminology to use consistently
- Figure types to include

**Part 4: Metrics & Benchmarking Guidance**
- Standard metrics across papers (throughput, latency, quality, memory)
- Baseline choices (eviction, compression, system baselines)
- Datasets to report on (2025 standard)

**Part 5: Summary**
- Do/Don't checklist for writing

**Best for:**
- Rewriting novelty.md with modern style
- Creating project style guide
- Planning benchmarking strategy
- Learning current best practices in ML systems papers

---

### 3. ANALYSIS_SUMMARY.md (336 lines, 12KB)
**What:** Executive summary and quick reference

**Sections:**
1. What was analyzed (brief description of each paper)
2. Key technical findings (5 major trends)
3. Writing style insights
4. Semantic architecture alignment (priority-ranked)
5. Concrete mappings to semantic
6. Recommendations for novelty.md and style guide
7. Document organization guide
8. Next steps

**Best for:**
- Quick overview of analysis
- Deciding which document to read for specific needs
- Next steps and action items
- Understanding how papers relate to semantic

---

## Key Findings at a Glance

### Technology Trends
1. **Shift from static to adaptive policies** - Per-context, per-token, per-layer customization
2. **Joint optimization over single-factor** - Compression AND eviction, not either/or
3. **Learned importance beats attention-guided** - Intrinsic importance at creation time
4. **Hierarchical storage is standard** - GPU → CPU → Disk multi-tier systems
5. **Comprehensive empirical validation** - 3-5 models, 3-5 datasets, 2-3 baselines minimum

### Papers Most Relevant to Semantic (Priority)
| Rank | Paper | Why | Maps To |
|------|-------|-----|---------|
| 1 | TRIM-KV | Learned token importance with exponential decay | Block importance scoring |
| 2 | EvicPress | Joint compression + eviction optimization | Multi-factor trade-offs |
| 3 | RAG-DCache | Persistent disk-based KV cache + locality | External working memory |
| 4 | KVSwap | Grouped I/O for hierarchical storage | Block-level memory hierarchy |
| 5 | vLLM-MLX | Content-based prefix caching | Cache key identity |

### Writing Style Essentials
**Do:**
- Open with concrete problem (metric or O() complexity)
- Show why existing solutions fail
- Present multi-dimensional results
- Include ablation studies
- Use trade-off plots
- Always quantify overhead
- Name specific baselines

**Don't:**
- Start with literature review
- Use "efficient" without metrics
- Report results without context
- Hand-wave design choices
- Claim "negligible overhead" unmeasured
- Write generic future work

---

## Quick Navigation by Use Case

### "I need to rewrite novelty.md"
1. Read: ANALYSIS_SUMMARY.md (5 min)
2. Read: STYLE_GUIDE_INSIGHTS.md Part 3 (20 min)
3. Reference: PAPERS_ANALYSIS.md for specific paper details (as needed)
4. Follow: Structure template in Part 3 of STYLE_GUIDE_INSIGHTS

### "I need to understand semantic's paradigm alignment"
1. Read: ANALYSIS_SUMMARY.md Section 4-5 (15 min)
2. Deep dive: STYLE_GUIDE_INSIGHTS.md Part 2 (30 min)
3. Reference: PAPERS_ANALYSIS.md for technical details on each paper (30 min)

### "I need to plan benchmarking strategy"
1. Read: STYLE_GUIDE_INSIGHTS.md Part 4 (15 min)
2. Reference: PAPERS_ANALYSIS.md for each paper's evaluation setup (30 min)
3. Compile: Models, datasets, baselines based on 2025 standards

### "I need to create a style guide"
1. Read: STYLE_GUIDE_INSIGHTS.md Part 1 + Part 5 (30 min)
2. Extract: Terminology table (Part 1)
3. Compile: Writing patterns with examples (Part 1)
4. Add: Figures to create checklist (Part 1)

### "I want deep technical understanding"
1. Read: PAPERS_ANALYSIS.md completely (1-2 hours)
2. Note: Technical approaches, key results, positioning
3. Reference: Cross-paper patterns section for synthesis

---

## Key Papers Summary Table

| Paper | Problem | Solution | Key Metric | Relevance |
|-------|---------|----------|-----------|-----------|
| RAG-DCache | Long RAG prefills kill latency | Disk KV cache + proactive prefetch | 10-20% TTFT ↓ | High |
| XQuant | KV quantization saturates early | Quantize inputs (X) not KV | 12.5× compression | Moderate |
| KVSwap | Mobile inference memory OOM | Grouped I/O + compression | Decoding stage | High |
| **TRIM-KV** | **Attention heuristics fail on reasoning** | **Learned token importance gates** | **58.4% pass@1 gain** | **Very High** |
| **EvicPress** | **Single policies suboptimal** | **Joint compression+eviction** | **2.19× TTFT faster** | **High** |
| FastKVzip | (Content unavailable) | Gated eviction | N/A | Moderate |
| vLLM-MLX | No unified multimodal inference | Content-based prefix cache | 28× speedup (images) | Moderate |

---

## Terminology Reference

**Core concepts (standardize these in writing):**

- **TTFT** = Time-to-first-token (prefill latency only)
- **Compression ratio** = Original size / compressed size (express as 10×, not 90%)
- **Importance score** = Learned or heuristic token/block importance metric
- **Retention** = What we keep in cache
- **Eviction** = Moving to lower-tier storage (not deletion)
- **Decay rate** = How quickly importance decreases over time
- **Utility function** = Unified metric for multi-factor optimization
- **Overhead** = Always measured (% CPU, ms latency, GB memory)
- **Quality** = Metric-specific (perplexity, accuracy, pass@1)
- **Throughput** = Tokens per second (standard unit)

---

## Concrete Mappings: Semantic Alignment

### TRIM-KV → Block Importance Learning
```
TRIM-KV architecture:
  g(token) → retention_score β ∈ [0,1]
  decay over time: α_ti = β_i^(t-i)
  eviction: drop lowest-scoring tokens

Semantic mapping:
  g(block) → importance_score ∈ [0,1]
  decay over time: importance_t = score_0 × decay_rate^(age)
  eviction: drop lowest-scoring blocks when budget full
```

**Why it matters:** Learned importance is more robust than attention-based heuristics.

### EvicPress → Utility Function
```
EvicPress approach:
  U(context_id, tier, method, ratio) → single score
  optimization: greedy placement maximizing total U per tier
  adaptation: re-profile when distribution drifts

Semantic mapping:
  U(block_id, tier, method, ratio) → single score
  optimization: greedy or DP to maximize utility per batch
  adaptation: re-profile after N batches
```

**Why it matters:** Single metric enables comparison across different decisions.

### KVSwap → Hierarchical Memory
```
KVSwap design:
  - Grouped KV entries for I/O efficiency
  - Separate compression (prediction) from full precision (computation)
  - Hierarchy: memory → disk

Semantic mapping:
  - Grouped block transfers
  - Separate compression (for indexing) from full (for inference)
  - Hierarchy: GPU → CPU → persistent storage
```

**Why it matters:** Grouping amortizes I/O overhead massively.

---

## For Literature Review

**Papers to cite for key concepts:**

- **Learned importance:** TRIM-KV (2025) - "Token Retention for Memory-Bounded KV Cache"
- **Joint optimization:** EvicPress (2025) - "Joint KV-Cache Compression and Eviction"
- **Disk-based caching:** RAG-DCache (2025) - "Shared Disk KV Cache Management"
- **Mobile optimization:** KVSwap (2025) - "Disk-aware KV Cache Offloading"
- **Hardware-aware:** vLLM-MLX (2026) - "Native LLM Inference on Apple Silicon"
- **Input quantization:** XQuant (2025) - "KV Cache Rematerialization"

---

## Example Figures to Include

1. **Problem illustration** - Show latency impact or memory pressure (like EvicPress Figure 2)
2. **Architecture diagram** - Component boxes with data flows
3. **Trade-off plot** - Quality (y-axis) vs latency/throughput (x-axis)
4. **Scaling curve** - Performance vs model size or concurrency
5. **Ablation bar chart** - Impact of removing each component
6. **Results heatmap** - Model × Dataset grid

---

## Next Steps

### For novelty.md
- [ ] Restructure with problem-first opening (STYLE_GUIDE_INSIGHTS Part 3)
- [ ] Add trade-off plot section
- [ ] Include ablation studies
- [ ] Add qualitative analysis
- [ ] Create positioning table vs baselines

### For style guide
- [ ] Adopt terminology table from analysis
- [ ] Add writing patterns section with good/bad examples
- [ ] Specify benchmarking standards (models, datasets, baselines)
- [ ] Create figures checklist
- [ ] Define "overhead" measurement standards

### For future research
- [ ] Implement TRIM-KV-style learned importance
- [ ] Design per-context optimization (EvicPress-style)
- [ ] Explore grouped I/O batching (KVSwap-style)
- [ ] Consider persistent cache layer (RAG-DCache-style)
- [ ] Add content-based hashing for identity (vLLM-MLX-style)

---

## Statistics

**Papers analyzed:** 7 (1 content-limited)
**Total pages extracted:** ~150 (HTML source)
**Analysis documents created:** 3
**Total analysis output:** 1,279 lines, 53KB
**Key findings:** 5 technology trends, 5 semantic alignments
**Terminology defined:** 10 core concepts

---

## Document Locations

```
/Users/dev_user/semantic/novelty/
├── PAPERS_ANALYSIS.md              (532 lines) - Comprehensive technical analysis
├── STYLE_GUIDE_INSIGHTS.md         (411 lines) - Actionable writing & design guidance
├── ANALYSIS_SUMMARY.md             (336 lines) - Executive overview
└── PAPERS_ANALYSIS_INDEX.md        (this file) - Quick reference & navigation
```

---

**End of Index**

For questions, deep dives, or specific paper details, refer to the three main analysis documents above.

