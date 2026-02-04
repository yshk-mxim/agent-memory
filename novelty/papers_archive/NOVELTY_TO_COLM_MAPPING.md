# Mapping novelty.md → COLM 2026 PDF

## Current novelty.md Structure vs. Target COLM Structure

### novelty.md Content Audit

| Section | Current Pages | Words | Status | Target COLM |
|---------|---------------|-------|--------|-------------|
| Abstract | 0.5 | 150 | ✓ Keep | Abstract (one paragraph) |
| 1. Introduction | 2.5 | 1000 | ⚠️ Condense | Section 1 (1.0 page) |
| 1.4 Contributions | 0.5 | 200 | ✓ Keep | Part of intro |
| 2. Background | 1 | 600 | ✓ Keep | Section 2 (0.75 page) |
| 2.3 Comparison | 0.5 | 300 | ⚠️ Move | Table in Related Work |
| 3. System Design | 3 | 2000 | ⚠️ Condense | Section 3 (2.5 pages) |
| 4. Implementation | 1 | 700 | ⚠️ Condense | Section 4 (0.75 page) |
| 5. Evaluation | 3 | 2500 | ❌ Heavy cut | Section 5 (1.5 pages) |
| 6. Discussion | 2.5 | 2200 | ⚠️ Tighten | Section 6 (1.5 pages) |
| 7. Related Work | 2 | 2000 | ❌ Rewrite | Section 7 (1.0 page) |
| 8. Conclusion | 0.5 | 300 | ⚠️ Expand | Section 8 (0.5 page) |
| Appendices | 2 | 1500 | ✓ Keep | Appendices (unlimited) |
| **TOTAL** | **19** | **13550** | — | **9.0** |

---

## Section-by-Section Mapping

### COLM Abstract (0.5 page, one paragraph, 150-200 words)

**Source**: novelty.md Abstract

**Current length**: ~250 words (too long)

**Target**: Keep core claims, condense framing

```
CURRENT: "Multi-agent LLM workflows on edge devices suffer from O(n)
cold-start latency: every agent turn re-prefills the entire
conversation history from scratch. On Apple Silicon, where GPU compute
is 10–50x slower than datacenter accelerators, this means 8–40 seconds
of prefill per turn — rendering multi-agent workflows unusably slow.
We present Semantic, a persistent KV cache management system..."

REWRITE: "Multi-agent LLM workflows on edge devices re-prefill conversation
history from scratch on every turn. On Apple Silicon (M4 Pro, 500 tok/s
prefill), a 4K-token conversation costs 8 seconds. We introduce Semantic:
a system that persists Q4 KV caches to disk with per-agent isolation.
Results: 2.0x–4.3x E2E speedup, 72% memory savings, and a novel
interleaved prefill+decode scheduler enabling true multi-user streaming.
First system combining persistent Q4 caches + batched quantized inference
+ cross-phase working memory on edge UMA."

Changes:
- Remove "10-50x slower than datacenter" (too broad)
- Add specific numbers (4K tokens, 8 seconds)
- Focus on three concrete contributions
- Cut from ~250 to ~150 words
```

---

### COLM Section 1: Introduction (1.0 page)

**Source**: novelty.md Sections 1.1-1.4

**Target subsections**:
1. The Multi-Agent Cold-Start Problem (motivation)
2. The Key Insight (solution overview)
3. Contributions (what's new)

**Condensing strategy**:

```
novelty.md 1.1 (500 words) → COLM Intro opening (200 words)
- Current: Detailed explanation of AutoGen/CrewAI/LangGraph
- Target: One concrete example (5-agent code review)
- Current: "On datacenter GPUs, re-prefill takes ~400ms"
- Target: "On M4 Pro, it takes 8 seconds. For 5 agents, 40 seconds."

novelty.md 1.2 (300 words) → COLM Intro middle (150 words)
- Current: Conceptual discussion of KV cache persistence
- Target: "The KV cache is (K,V) tensors. We save them as Q4 safetensors.
           Reload takes <100ms instead of 8 seconds of prefill."

novelty.md 1.3 (400 words) → Removed from intro
- Reason: "RAG vs working memory" framing is in Discussion (Section 6)
- Incorporate one sentence: "Unlike RAG, which re-retrieves text on each
                            turn, we persist pre-computed attention state."

novelty.md 1.4 (200 words) → COLM Intro closing (200 words)
- Keep all six contributions
- Present as three main: (1) Persistent Block Pool, (2) BatchQuantizedKVCache,
                        (3) Cross-phase Working Memory
```

**Page layout**:
```
Title + Abstract        0.5 page
1. Introduction         1.0 page
  1.1 The Cold-Start Problem (2 paragraphs, 200 words)
  1.2 The Key Insight (1.5 paragraphs, 150 words)
  1.3 Contributions (3 paragraphs, 200 words)
```

---

### COLM Section 2: Background & Motivation (0.75 page)

**Source**: novelty.md Sections 2.1-2.4

**Condensing strategy**:

```
novelty.md 2.1 (300 words) → COLM 2 opening (150 words)
- Keep the re-prefill diagram or describe it in text
- Current: Sequence diagram showing 4 turns
- Target: "Agent Turn 2 re-prefills all 2K tokens (~4 seconds) instead
           of processing only 50 new tokens (~200ms)."

novelty.md 2.2 (400 words) → COLM 2.1 "Apple Silicon UMA" (200 words)
- Keep the UMA diagram (very clear)
- Cut: "Fragmentation" and "MLX stream-level parallelism" details
- Keep: "Bandwidth asymmetry (400 GB/s) vs compute (500 tok/s)"

novelty.md 2.3 (300 words) → COLM Related Work table (not in intro)
- Comparison table is too detailed for background
- Move the 8-system capability table to Section 7

novelty.md 2.4 (400 words) → COLM 2.2 "KV Cache as Memory" (150 words)
- Remove MemArt/Memory³/EM-LLM citations (belong in Related Work)
- Keep: Core principle that KV cache is more efficient than text
- Add: One sentence on working memory in multi-phase scenarios
```

**Page layout**:
```
2. Background & Motivation                    0.75 page
  2.1 Re-Prefill Problem in Multi-Agent Workflows (150 words)
  2.2 Apple Silicon UMA: Opportunities & Constraints (200 words)
  2.3 KV Cache as Agent Working Memory (150 words)
  [Drop "Why Existing Solutions Fall Short" — move to Discussion]
```

---

### COLM Section 3: System Design (2.5 pages)

**Source**: novelty.md Sections 3.1-3.6

**Condensing strategy** (most aggressive cuts needed here):

```
novelty.md 3.1 (800 words) → COLM 3.1 "Block Pool Architecture" (400 words)
- Keep: ModelCacheSpec + BlockPool class diagram
- Keep: Per-agent isolation + model-agnostic abstraction
- Remove: Detailed YAML coordination syntax (move to Implementation)
- Remove: Detailed concurrency model explanation

novelty.md 3.2 (600 words) → COLM 3.2 "Q4 Persistence Pipeline" (350 words)
- Keep: Disk → Unified Memory → Attention flow diagram (excellent)
- Keep: MLX routing mechanism (hasattr check)
- Remove: Detailed quantization format (move to Appendix A)
- Remove: Memory formula details (move to Appendix)
- Keep: "72% memory savings" measurement

novelty.md 3.3 (600 words) → COLM 3.3 "Character-Level Prefix Matching" (250 words)
- Keep: BPE boundary problem explanation (clear)
- Remove: Three match outcomes table (too detailed)
- Keep: Why this solves the problem ("EXACT", "EXTEND", "DIVERGE" as sentences)
- Result: ~250 words

novelty.md 3.4 (600 words) → COLM 3.4 "UMA Memory Management" (300 words)
- Keep: mx.eval() discipline diagram
- Remove: Detailed chunk sizing table
- Keep: "Chunked prefill prevents 15GB peak memory"
- Keep: Three-step reclamation (eval, clear_cache, gc.collect)

novelty.md 3.5 (600 words) → Remove or merge with Implementation
- Reason: YAML coordination is implementation detail, not design
- Where to go: Section 4.4 case study example

novelty.md 3.6 (800 words) → COLM 3.5 "Batched Q4 Inference" (500 words)
- Keep: Conceptual explanation of merge/update_fetch/extract
- Remove: Detailed algorithm descriptions
- Keep: Monkey-patching overview (3 patches, ~130 lines)
- Keep: Per-token decode granularity motivation
- Keep: Cache lifecycle diagram
```

**Result**: 3 → 5 subsections, ~2500 words (was ~3200)

**Page layout**:
```
3. System Design                              2.5 pages
  3.1 Block Pool Architecture (400 words)
  3.2 Q4 Persistence Pipeline (350 words)
  3.3 Character-Level Prefix Matching (250 words)
  3.4 UMA Memory Management (300 words)
  3.5 Continuous Batching with Q4 (500 words)
```

---

### COLM Section 4: Implementation (0.75 page)

**Source**: novelty.md Sections 4.1-4.4

**Condensing strategy**:

```
novelty.md 4.1 (500 words) → COLM 4.1 "Architecture" (200 words)
- Keep: High-level hexagonal diagram (very clear)
- Remove: Layer responsibilities table (too detailed)
- Keep: Inbound/application/domain/adapter distinction

novelty.md 4.2 (150 words) → COLM 4.1 continuation (100 words)
- Supported Models table: Keep exactly as-is

novelty.md 4.3 (400 words) → COLM 4.2 "Data Flow: Agent Resume" (200 words)
- Keep: Sequence diagram showing load/match/process/save
- Remove: Detailed narrative explanation (diagram says it all)

novelty.md 4.4 (600 words) → COLM 4.3 "Working Memory: Case Studies" (250 words)
- Keep: Prisoner's Dilemma setup (2 paragraphs)
- Keep: Gossip Network setup (1 paragraph)
- Remove: Detailed message injection syntax
- Add: Key insight: "agents maintain attention state across phases"
```

**Page layout**:
```
4. Implementation                             0.75 page
  4.1 Architecture Overview (200 words)
  4.2 Data Flow: Agent Resume (200 words)
  4.3 Working Memory Case Studies (250 words)
```

---

### COLM Section 5: Evaluation (1.5 pages)

**Source**: novelty.md Sections 5.1-5.10

**Heavy condensing required** (current: 3 pages → target: 1.5 pages)

**What to KEEP**:
- ✅ 5.5: TTFT Speedup vs Context Length (Figure 5.5 with cold/warm/hot)
- ✅ 5.7: Batch=2 Concurrent Throughput (results table, top metrics)
- ✅ 5.7: Staggered Arrivals (Figure 5.8, User B 2.6x improvement)

**What to REMOVE**:
- ❌ 5.1: Experimental Setup (move to Appendix C)
- ❌ 5.2: Cache I/O Latency (too granular)
- ❌ 5.3: Cold Start Scaling (captured in 5.5)
- ❌ 5.4: End-to-End Multi-Turn Speedup (less interesting than 5.5)
- ❌ 5.6: Streaming vs Non-Streaming (streaming is assumed)
- ❌ 5.8: Memory Efficiency (covered in design section)
- ❌ 5.9: LM Studio Comparison (move to Related Work footnote)
- ❌ 5.10: Character-Level vs Token-Level Matching (move to Appendix)

**Reorganized structure**:
```
5. Evaluation                                 1.5 pages
  5.1 Experimental Setup (100 words, brief)
  5.2 TTFT Speedup vs Context Length (400 words + Figure 5.5)
  5.3 Concurrent Throughput (300 words + Table)
  5.4 Staggered Arrivals (300 words + Figure 5.8)
```

**Page layout**:
- Page 6 (bottom) → 5.1-5.2 with Figure 5.5
- Page 7 (top) → 5.3-5.4 with Table + Figure 5.8

---

### COLM Section 6: Discussion (1.5 pages)

**Source**: novelty.md Sections 6.1-6.8

**Mapping**:
```
novelty.md 6.1 "Novelty Classification" (400 words) → COLM 6.1 (250 words)
- Keep: Core claims (novel on MLX, first combination)
- Remove: Detailed classification table
- Keep: Explicit citations to mlx-lm issue #548

novelty.md 6.2 "Working Memory Paradigm" (600 words) → COLM 6.2 (250 words)
- Keep: RAG vs Working Memory comparison
- Remove: Detailed scenario narratives
- Keep: Key insight: "attention patterns > retrieved text"

novelty.md 6.3-6.4 "Comparisons to vllm-mlx & RAG-DCache" (500 words)
- Combine into COLM 6.3 "Related Systems" (300 words)
- Focus on: vllm-mlx (no persistence), RAG-DCache (no isolation)

novelty.md 6.5 "Why Not Compose Existing Tools?" (400 words)
- Keep exactly: This is a strong section
- Move to COLM 6.4 "Composition Gap" (300 words)

novelty.md 6.6 "Attention-Layer vs Message-Layer" (600 words)
- Remove entirely or reduce to 1 paragraph
- Reason: Too philosophical for COLM (belongs in longer venue)
- Or: Move to "Broader Implications" subsection in Discussion

novelty.md 6.7 "Limitations" (400 words) → COLM 6.5 "Limitations" (200 words)
- Keep: Single-device (now fixed by TB5 RDMA), Q4 accuracy, architectures
- Remove: Detailed hardware-dependent capacity claims
- Keep: One sentence on M5 Neural Engine impact

novelty.md 6.8 "Future Directions" (500 words) → COLM 6.6 (250 words)
- Keep: Top 3 concrete directions
  1. Cross-device disaggregated inference
  2. Integration with vllm-mlx
  3. Quantitative working memory benchmarks
- Remove: Speculative decoding, formal benchmarking details
```

**Page layout**:
```
6. Discussion                                 1.5 pages
  6.1 Novelty Classification (250 words)
  6.2 Working Memory Paradigm (250 words)
  6.3 Related Systems Positioning (300 words)
  6.4 Composition Gap (300 words)
  6.5 Limitations & Opportunities (250 words)
```

---

### COLM Section 7: Related Work (1.0 page)

**Source**: novelty.md Sections 7.1-7.7

**Condensing strategy** (current: 2000 words → target: 1500 words)

```
novelty.md 7.1 "KV Cache Management Systems" (600 words)
- CONDENSE each system to 2-3 sentences max
- vLLM: "PagedAttention for CUDA, no disk persistence"
- vllm-mlx: "Continuous batching on MLX, no isolation"
- LMCache: "Disk persistence on datacenter GPUs"
- NEW: KVSwap, EvicPress (1 sentence each)
- Result: ~250 words (was 600)

novelty.md 7.2 "KV Cache Compression" (400 words)
- CONDENSE: KIVI (2 sentences), KVQuant (1 sentence)
- ADD: CommVQ, QuantSpec (Apple research), XQuant, KVSplit
- Remove: CacheGen details
- Result: ~200 words

novelty.md 7.3 "KV Cache as Memory" (600 words)
- CONDENSE: MemArt (2 sentences), EM-LLM (1 sentence), Memory³ (1 sentence)
- ADD: "Memory in the Age of AI Agents" survey (1 sentence)
- MOVE: RAG-DCache from novelty.md to COLM 7.3
- Result: ~250 words

novelty.md 7.4 "RAG and Alternatives" (200 words)
- KEEP: PageIndex.ai (1 sentence), RAPTOR (1 sentence)
- Rationale: Positioning Semantic against RAG alternatives
- Result: ~50 words

novelty.md 7.5 "Multi-Agent KV Cache Research" (700 words)
- CONDENSE: 7 systems (KVCOMM, KVFlow, Continuum, LRAgent, DroidSpeak,
            Q-KVComm, SR-KI) to 1 sentence each
- ADD: TRIM-KV, Fast KVzip (2 sentences each)
- Result: ~200 words

novelty.md 7.6 "Edge LLM Tools" (500 words)
- CONDENSE: LM Studio (3 sentences), Ollama (2 sentences), llama.cpp (2 sentences)
- New focus: Quantized KV cache support + batching
- Result: ~200 words

novelty.md 7.7 "Agent Frameworks & Communication Protocols" (500 words)
- KEEP: AutoGen/CrewAI/LangGraph (1 sentence)
- CONDENSE: A2A (2 sentences), MCP (1 sentence), Cisco IoC (2 sentences)
- Result: ~150 words

Total 7.1-7.7: 1500 words (was 3500) ✓
```

**Table for Section 7**: Move novelty.md 2.3 comparison table here (optional)

**Page layout**:
```
7. Related Work                               1.0-1.25 pages
  7.1 KV Cache Management Systems (250 words)
  7.2 KV Cache Compression (200 words)
  7.3 KV Cache as Memory & RAG (300 words)
  7.4 Multi-Agent Research (200 words)
  7.5 Edge Tools & Frameworks (300 words)
```

---

### COLM Section 8: Conclusion (0.5 page)

**Source**: novelty.md Section 8

**Changes**:
- Keep: Summary of three contributions
- Remove: "First system combining..." language (already in abstract)
- Add: Concrete next steps (DGX Spark integration, vllm-mlx backend, Thunderbolt 5 RDMA)
- Tone: Confident, specific, action-oriented

**Result**: ~300 words

---

### COLM Appendices (unlimited pages)

**Appendix A: safetensors Q4 Cache Format** (novelty.md Appendix A)
- Tensor naming convention
- Metadata header structure
- Keep exactly as-is (~150 words)

**Appendix B: MLX Lazy Evaluation Pitfalls** (novelty.md Appendix B)
- Table of 6 pitfalls
- Keep exactly as-is (~200 words)

**Appendix C: Benchmark Configuration** (novelty.md Appendix C)
- Hardware, software versions, hyperparameters
- Keep exactly as-is (~150 words)

**Appendix D: Quantization Format Details** (from novelty.md 3.2, cut for space)
- Memory formula for Q4 caches
- Group size, scale/bias details
- ~200 words

**Appendix E: Character-Level Matching Algorithm** (from novelty.md 3.3, cut for space)
- Match outcomes (EXACT, EXTEND, DIVERGE)
- Algorithm pseudo-code
- ~250 words

**Appendix F: Monkey-Patching Details** (from novelty.md 3.6, cut for space)
- Three patches to mlx-lm
- Code snippets
- ~300 words

---

## Drafting Workflow

### Phase 1: Extract Sections (1 session)
1. Copy novelty.md sections 1-8 into skeleton COLM template
2. Mark sections for condensing (using % CONDENSE comments)
3. Identify figures/tables to keep or remove

### Phase 2: Initial Cuts (2 sessions)
1. Condense Evaluation (5 subsections → 4)
2. Condense Related Work (7 subsections → 5, reduce per-system length)
3. Move detailed material to appendices

### Phase 3: Rewrite Pass (2-3 sessions)
1. Tighten introduction (1000 words → 400 words)
2. Apply humanizer rules throughout
3. Add concrete numbers where possible
4. Ensure flows between sections

### Phase 4: Figures & Tables (1 session)
1. Extract/create 4-5 key figures
2. Create 2-3 results tables
3. Add captions and cross-references

### Phase 5: LaTeX Setup (1 session)
1. Copy COLM template structure
2. Convert markdown tables to booktabs
3. Build bibliography from novelty.bib

### Phase 6: Final Polish (1 session)
1. Check page count (must be ≤9)
2. Verify margins, fonts, spacing
3. Grammar and consistency pass

---

## Page Count Verification

Current estimate by section:
```
Cover + Abstract     0.5  ✓
Intro                1.0  ✓
Background          0.75  ✓
Design              2.5   ✓
Implementation      0.75  ✓
Evaluation          1.5   ✓
Discussion          1.5   ✓
Related Work        1.0   ✓
Conclusion          0.5   ✓
───────────────────────
TOTAL              9.75  ⚠️ Over by 0.75 page
```

**Action**: Reduce Discussion from 1.5 to 1.25 pages (trim Limitations/Future Work slightly)

---

## Quality Checklist

- [ ] All 7 novelty.md sections mapped to COLM structure
- [ ] Figures identified (count: 5-6 expected)
- [ ] Tables identified (count: 3-4 expected)
- [ ] Word count per section estimated
- [ ] Appendices identified (6 expected)
- [ ] Page budget verified (≤9 pages)
- [ ] Humanizer rules documented (see STYLE_GUIDE.md)
- [ ] COLM LaTeX setup checklist created (see COLM_QUICK_REF.md)

---

**Document Version**: 1.0
**Date Created**: February 4, 2026
**Target**: COLM 2026 Anonymous Submission
