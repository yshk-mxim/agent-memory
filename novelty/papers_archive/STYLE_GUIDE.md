# Comprehensive Style Guide for Semantic Novelty Write-Up
**PDF Format Based on COLM 2026 Template**

---

## I. ANTI-AI WRITING PATTERNS (Humanizer/SKILL.md Framework)

### A. Content Patterns to Eliminate

**1. Inflated Significance Language**
- ❌ "pivotal," "testament," "crucial," "groundbreaking," "paradigm shift"
- ✅ Use neutral descriptors: "significant," "important," "first," "enables"
- ✅ Show importance through results, not adjectives

**Example Correction:**
- ❌ "This groundbreaking approach represents a paradigm shift in edge inference"
- ✅ "We persist Q4 KV caches to disk, enabling sub-second context restoration on edge devices"

**2. Excessive Notability Claims**
- ❌ Lists of "Media coverage" or "Industry recognition"
- ✅ Focus on technical contribution, not press
- ❌ "Featured in [Journal], [Conference], [Blog]"
- ✅ Cite specific technical papers that build on the work

**3. Superficial Analyses Ending in "-ing"**
- ❌ "This approach enables more efficient cache management, allowing faster inference, supporting multi-agent workflows..."
- ✅ Break into separate claims with evidence: "The block pool achieves 2.0x E2E speedup on multi-turn conversations (Figure 5.4)."

**4. Promotional Language**
- ❌ "vibrant," "nestled," "breathtaking," "elegant," "clever," "innovative"
- ✅ Use neutral technical language: "enables," "reduces," "demonstrates," "achieves"

**5. Vague Attributions**
- ❌ "Industry reports show..." / "Research suggests..." / "It is known that..."
- ✅ "Liu et al. (ICML 2024) demonstrated..." / "We measured [specific value]..."

**6. Formulaic Limitations + Future Work**
- ❌ "Limitations of this work include X, Y, and Z. Future work could explore..."
- ✅ Be specific: "Single-device constraint: macOS Tahoe 26.2 RDMA now enables multi-Mac clusters, making distributed cache sharing feasible (Section 6.7)."

---

### B. Language Patterns to Eliminate

**7. Overused AI Vocabulary**
- ❌ "Additionally," "Furthermore," "Notably," "Interestingly," "It should be noted that"
- ✅ Use simple connectors: "And," "Also," "Next," "Here," or no connector
- ❌ "landscape," "paradigm," "showcase," "leverage," "cutting-edge," "state-of-the-art"
- ✅ "system," "approach," "demonstrate," "use," "recent," "best-performing"

**8. Copula Avoidance (Over-formalization)**
- ❌ "This approach serves as a memory abstraction for agents"
- ✅ "This approach is agent memory"
- ❌ "The block pool functions as a management layer"
- ✅ "The block pool manages per-agent caches"

**9. Negative Parallelisms**
- ❌ "It's not just faster; it's also more memory-efficient"
- ✅ "It is 2.0x faster and uses 72% less memory"

**10. Rule of Three Forcing**
- ❌ "Persistent cache management has three key benefits: (1) speed, (2) memory efficiency, (3) cross-session reuse"
- ✅ List what's actually measured: "Persistent caches enable sub-second restoration (Figure 5.5), 72% memory savings (Table 5.8), and cross-session reuse via character-level prefix matching (Section 3.3)."

**11. Excessive Synonym Substitution (Elegant Variation)**
- ❌ "The KV cache is the memory. This stored state is the agent's context. This persistent artifact is critical..."
- ✅ "The KV cache is the agent's memory. We persist it as Q4 safetensors on disk (Section 3.2)."

**12. False Ranges**
- ❌ "TTFT speedup ranges from 2.0x to 81.6x depending on context length"
- ✅ "TTFT speedup scales with context: 2.0x at 1K tokens, 5.5x at 2K, 81.6x at 16K (Figure 5.5)"

---

### C. Style Patterns to Eliminate

**13. Em Dash Overuse**
- ❌ "We observe multiple benefits — faster inference — reduced memory — better multi-agent support — all on edge devices"
- ✅ "We observe three benefits: (1) 2.0x E2E speedup, (2) 72% memory savings, (3) true per-agent isolation"
- ⚠️ **Rule**: One em dash per 5 sentences maximum

**14. Excessive Boldface**
- ❌ Use boldface for section names only (enforced by \section{})
- ❌ Don't bold terms mid-paragraph unless unavoidable
- ✅ "The **BatchQuantizedKVCache** enables..." (only in definitions)

**15. Inline-Header Vertical Lists**
- ❌ Use bolded inline headers: "**Key Contribution:** This enables... **Secondary Contribution:** This demonstrates..."
- ✅ Use numbered lists or prose: "Our first contribution is persistent block pool isolation. Our second is batched Q4 inference..."

**16. Title Case in All Headings**
- ❌ \section{The Persistent Block Pool Architecture and Its Design}
- ✅ \section{Block Pool Architecture}

**17. Decorative Emojis**
- ❌ Do not use emojis (❌, ✅, 🎯, 🚀)
- ✅ Use text markers: [Not recommended], [Recommended], or omit entirely

**18. Curly Quotation Marks in Code**
- ✅ Use straight quotes in code: `"key"` not `"key"`
- ✅ Use curly quotes in prose: "This is correct"

---

### D. Communication Patterns to Eliminate

**19. Chatbot Artifacts**
- ❌ "I hope this helps," "Thank you for your attention," "Feel free to contact us"
- ✅ Omit entirely in academic writing

**20. Knowledge-Cutoff Disclaimers**
- ❌ "As of my knowledge cutoff in February 2025..."
- ✅ Just state facts with publication dates

**21. Overly Servile Tone**
- ❌ "We are honored to present this work," "We humbly suggest..."
- ✅ "We present," "We propose," "We demonstrate"

---

### E. Filler and Hedging Patterns

**22. Unnecessary Phrases**
- ❌ "In order to achieve," "At this point in time," "Due to the fact that," "On the other hand"
- ✅ "To achieve," "Now," "Because," "However"

**23. Over-Qualification**
- ❌ "It could be argued that this approach might potentially suggest that..."
- ✅ "This approach demonstrates..."
- ❌ "We arguably show" / "This somewhat validates"
- ✅ "We show" / "This validates"

**24. Generic Positive Conclusions**
- ❌ "This represents promising directions for future research"
- ✅ "Concrete next steps include: (1) Cross-device RDMA sharing via Thunderbolt 5 (Section 6.8), (2) Integration with vllm-mlx serving engine"

---

## II. TECHNICAL-TO-EXPLANATORY BALANCE

### A. Section-Level Structure

**Each major section should follow:**
1. **Motivation** (Why does this matter?) — 2-4 sentences, no equations
2. **Technical approach** (How does it work?) — Diagrams, pseudocode, brief math if necessary
3. **Results** (Does it work?) — Numbers, comparisons, measured speedup
4. **Discussion** (What does it mean?) — Context, limitations, implications

### B. Language Density Across Sections

| Section | Tech Density | Explanation Density | Example |
|---------|-------------|-------------------|---------|
| **Introduction** | 20% | 80% | "Agents accumulate conversation history. Re-prefilling 4K tokens on Apple Silicon takes ~8 seconds (Section 2.1). We show how to avoid this via persistent KV caches." |
| **Background** | 40% | 60% | "Unified Memory Architecture (UMA) means CPU and GPU share DRAM. Apple M4 Pro has 24GB at 400 GB/s bandwidth. This changes the optimization landscape..." |
| **Method** | 70% | 30% | "The block pool allocates 256-token blocks per agent. merge() left-pads and stacks Q4 caches. extract() materializes per-agent cache from batch (Algorithm 1)." |
| **Evaluation** | 50% | 50% | "We benchmark Gemma 3 12B with warm cache (in-memory) vs. cold (re-prefill). Warm TTFT is nearly constant (~650ms). Cold TTFT scales linearly with context (Figure 5.5)." |
| **Related Work** | 30% | 70% | "vLLM (Kwon et al., 2023) introduced PagedAttention for NVIDIA GPUs. It evicts inactive cache blocks to CPU memory, reducing VRAM. Unlike Semantic, vLLM does not support disk persistence or per-agent isolation." |
| **Discussion** | 40% | 60% | "The BatchQuantizedKVCache is novel on MLX because... (technical detail). This matters because... (why it's important). However, M5's Neural Accelerators may shift this trade-off (implication)." |

---

### C. Explanation Techniques

**Technique 1: Motivation → Approach → Proof**
```
Motivation: "BPE tokenization is non-compositional (Section 3.3)."
Approach: "We match on character-level text instead of token IDs."
Proof: "This solves the BPE boundary problem: token-level matching fails
         at 20x worse (Table 5.10), while character-level matching extends
         cache correctly 100% of the time."
```

**Technique 2: Problem → Solution → Validation**
```
Problem: "LM Studio's cache degrades at Turn 3+ due to token-level matching."
Solution: "We use character-level matching because stored text is invariant
          to tokenizer changes."
Validation: "Semantic maintains consistent ~1.2s latency at all conversation
           depths (Figure 5.4). LM Studio drops to 2.8s at Turn 3."
```

**Technique 3: Assumption → Evidence → Implication**
```
Assumption: "We assume KV cache is a valid memory substrate."
Evidence: "MemArt achieves 11% accuracy improvement over plaintext memory
         (ICLR 2026). EM-LLM outperforms RAG by 30.5% (ICLR 2025)."
Implication: "We extend this to persistent cross-session memory with
           per-agent isolation."
```

---

## III. FIGURE AND TABLE GUIDELINES

### A. Figure Density and Placement

- **Target**: 1 figure per 2-3 sections (not per subsection)
- **Location**: Immediate after mention, or top of next section
- **Size**:
  - Full-width: 5.5in (text width in COLM)
  - Two-column: 2.5in each
  - Minimum readable font: 10pt (same as body text)

### B. Figure Types and Formatting

**Flowcharts / Architecture Diagrams:**
- Use mermaid or TikZ (in LaTeX)
- **Colors**: Monochrome acceptable, but use consistent palette:
  - Data/Stored: Light blue (#4a9eff)
  - Compute/Processing: Light yellow (#ffd93d)
  - Output/Result: Light green (#2ecc71)
  - Error/Warning: Light red (#ff6b6b)
- **Fonts**: sans-serif (Arial, Helvetica) for labels, monospace for code
- **No shadows or 3D effects** — keep clean and professional

**Example Good Figure:**
```
┌─────────────────────┐
│  Disk (safetensors) │
│  K_weights (uint32) │
│  K_scales (float16) │
│  V_weights (uint32) │
└──────────┬──────────┘
           │ load
           ▼
┌─────────────────────┐
│  Unified Memory     │
│  mx.array tuples    │
│  (w, s, b) format   │
└──────────┬──────────┘
           │ inject
           ▼
┌─────────────────────┐
│  QuantizedKVCache   │
│  .bits = 4 (flag)   │
└──────────┬──────────┘
           │ forward pass
           ▼
┌─────────────────────┐
│  quantized_sdpa()   │
│  Q4 matmul, no FP16 │
└─────────────────────┘
```

**Line Plots / Benchmark Results:**
- X-axis: context length, context, or configuration
- Y-axis: metric (TTFT ms, speedup, tokens/sec)
- Lines/markers: Solid line for cold, dashed for warm, dotted for hot
- Legend: Placed inside plot area if space allows
- Grid: Light gray, major ticks only

**Tables:**
- Use \begin{table} with booktabs (included in COLM template)
- No vertical lines (booktabs style)
- Horizontal lines: \toprule, \midrule, \bottomrule
- Column alignment: l (left) for text, r (right) for numbers
- Font: same 10pt as body
- Width: full text width (5.5in) unless narrower makes sense

### C. Figure Captions and References

**Caption Style (COLM standard):**
- **Bold label**: Figure 1. (with period)
- **Italic descriptive title** (optional): *Staggered Arrival TTFT Comparison.*
- **Prose description**: "Left plot shows sequential mode (User B waits for User A to complete). Right plot shows batched mode (B's prefill interleaved with A's decode). Error bars represent standard deviation across 3 runs."
- **Key findings**: "At 1K context, batching improves B's TTFT by 2.6x (Figure 5.7)."

**Reference in Text:**
- ✅ "As shown in Figure 5.7, staggered arrivals benefit from batching"
- ✅ "Figure 5.7 demonstrates a 2.6x TTFT improvement"
- ❌ "The figure below shows..." (avoid — figures may float)
- ❌ "See the visualization" (be specific about which figure)

---

## IV. TABLE GUIDELINES

### A. Table Structure (COLM/booktabs)

```latex
\begin{table}[t]
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{System} & \textbf{Persistent} & \textbf{Q4} & \textbf{Batched} & \textbf{Per-Agent} \\
\midrule
vLLM & -- & -- & ✓ & -- \\
vllm-mlx & -- & -- & ✓ & -- \\
LMCache & ✓ & -- & -- & -- \\
Semantic & ✓ & ✓ & ✓ & ✓ \\
\bottomrule
\end{tabular}
\caption{Capability comparison table. Bold rows indicate full support.}
\label{tab:comparison}
\end{table}
```

**Rules:**
- ✅ Use ✓/-- for boolean features
- ✅ Right-align numbers for comparison
- ❌ Never use vertical lines
- ❌ Never use ALL CAPS columns (use \textbf{} for emphasis)
- **Footnotes**: Use \textsuperscript{*} in cell, then note after table

### B. Results Tables Format

**Template for results:**
```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Context} & \textbf{Metric} & \textbf{Value} & \textbf{Speedup} \\
\midrule
1,024 & TTFT (hot) & 654 ms & 5.5x \\
      & TTFT (warm) & 949 ms & 2.0x \\
4,096 & TTFT (hot) & 869 ms & 17.8x \\
      & TTFT (warm) & 3,980 ms & 1.7x \\
\bottomrule
\end{tabular}
\caption{TTFT speedup across cache states. Cold baseline for warm
speedup computed from dedicated server-restart test.}
\label{tab:ttft_speedup}
\end{table}
```

---

## V. CITATION STYLE (COLM/natbib)

### A. In-Text Citations

**COLM uses author-year citations with natbib:**

```latex
\citet{Liu2024}          % Liu (2024)
\citep{Liu2024}          % (Liu, 2024)
\citep{Liu2024,Kim2025}  % (Liu, 2024; Kim, 2025)
\citep[p.\ 42]{Liu2024}  % (Liu, 2024, p. 42)
```

**Style in narrative:**
- ✅ "Liu et al. (ICML 2024) demonstrated that keys benefit from per-channel quantization"
- ✅ "Quantization sensitivity varies by layer (KIVI; Liu et al., 2024)"
- ❌ "Liu et al. [1] showed..." (numbered style is for different venues)

### B. Bibliography Format (COLM uses natbib author-year)

**Sample entries for novelty.bib:**

```bibtex
@article{Liu2024,
  title={KIVI: Key-Value Cache Quantization for Large Language Models},
  author={Liu, Zichang and others},
  journal={ICML},
  year={2024}
}

@article{Upadhyay2026,
  title={When KV Cache Reuse Fails: An Empirical Analysis},
  author={Upadhyay, Rohit and others},
  journal={arXiv preprint arXiv:2601.09999},
  year={2026}
}

@inproceedings{Barrios2026,
  title={vllm-mlx: Continuous Batching on Apple Silicon},
  author={Barrios, Jose},
  journal={arXiv preprint arXiv:2601.19139},
  year={2026}
}
```

**Bibliography should be:**
- Sorted alphabetically by author last name
- Include arXiv preprint links where paper not yet published
- Conference papers: include venue (ICML, NeurIPS, etc.)
- Author count: For >3 authors, use "First Author et al."

---

## VI. MATHEMATICAL NOTATION

### A. Conventions for Semantic Paper

**Tensor notation:**
```latex
K, V                    % Key, Value matrices
Q                       % Query matrix
K^{(Q4)}, V^{(Q4)}     % Quantized K/V
w_s, s, b              % Weights, scales, biases (Q4 format)
n_{\text{tokens}}      % Number of tokens
n_{\text{heads}}       % Number of heads
n_{\text{kv\_heads}}   % Number of KV heads
```

**Algorithm notation:**
```latex
\text{merge}()         % Function names in \text{}
\text{extract}()
\text{update\_and\_fetch}()
```

**Performance metrics:**
```latex
\text{TTFT}            % Time to first token
\text{E2E}             % End-to-end
\text{TPS}             % Tokens per second
\text{TPS}_{\text{sys}} % System TPS
```

### B. Equation Integration

- **Inline equations**: Use $...$ (e.g., $n = 2^{10}$)
- **Display equations**: Use \begin{equation}...\end{equation} with \label{}
- **Multiple equations**: Use \begin{align}...\end{align} for alignment
- **Cross-reference**: "As shown in Eq.~\eqref{eq:memory}, the memory footprint..."

---

## VII. STRUCTURE TEMPLATE FOR NOVELTY.PDF

### I. Front Matter
```latex
\documentclass{article}
\usepackage[submission]{colm2026_conference}  % Change to [final] if accepted
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}

\title{Semantic: Persistent Multi-Agent KV Cache Management \\
       for Apple Silicon}

\author{...}  % To be anonymized for submission

\begin{document}
\maketitle
```

### II. Abstract (required, one paragraph max)
- Problem statement: "Multi-agent LLM workflows suffer O(n) cold-start latency"
- Solution: "We persist Q4 KV caches to disk with per-agent isolation"
- Results: "2.0x-4.3x speedup, 72% memory savings, 4 architectures"
- Novelty: "First system combining persistent Q4 KV cache + batched inference + cross-phase working memory on edge UMA"

### III. Body Sections (strict 9-page limit for text)

```
1. Introduction (1-2 pages)
   1.1 The Multi-Agent Cold-Start Problem (motivation)
   1.2 The Key Insight (high-level solution)
   1.3 From RAG to Working Memory (framing)
   1.4 Contributions (what's new)

2. Background & Motivation (1 page)
   2.1 Re-Prefill Problem (why it matters)
   2.2 Apple Silicon UMA (hardware context)
   2.3 Why Existing Solutions Fall Short (comparison)
   2.4 KV Cache as Agent Working Memory (framing)

3. System Design (2-3 pages)
   3.1 Block Pool Architecture (core)
   3.2 Q4 Persistence Pipeline (how)
   3.3 Character-Level Prefix Matching (key technique)
   3.4 UMA Memory Management (practical)
   3.5 Cross-Phase Context Injection (working memory)
   3.6 Continuous Batching (streaming)

4. Implementation (1 page)
   4.1 Architecture (high-level)
   4.2 Supported Models (scope)
   4.3 Data Flow (example)
   4.4 Working Memory Case Studies (examples)

5. Evaluation (1-2 pages)
   5.1 Setup (methodology)
   5.2-5.10 Results (key benchmarks only)
   Focus on: Cold/warm/hot TTFT, batched throughput, staggered arrivals

6. Discussion (1-2 pages)
   6.1 Novelty Classification (what's new)
   6.2 Working Memory Paradigm (framing)
   6.3 Comparison to vllm-mlx (closest work)
   6.4 Comparison to RAG-DCache (disk persistence)
   6.5 Why Not Compose Existing Tools? (composition gap)
   6.6 Attention-Layer vs Message-Layer (positioning)
   6.7 Limitations (honest assessment)
   6.8 Future Directions (concrete next steps)

7. Related Work (1 page, unlimited citations)
   7.1 KV Cache Management Systems
   7.2 KV Cache Compression
   7.3 KV Cache as Memory
   7.4 RAG and Alternatives
   7.5 Multi-Agent KV Cache Research
   7.6 Edge LLM Tools
   7.7 Agent Frameworks

8. Conclusion (0.5 page)
   Recap the three contributions, emphasize novelty position,
   mention concrete next steps.
```

### IV. Appendices (unlimited pages)
- Appendix A: safetensors Q4 Cache Format
- Appendix B: MLX Lazy Evaluation Pitfalls
- Appendix C: Benchmark Configuration

### V. References (unlimited, alphabetically sorted)

---

## VIII. SPECIFIC GUIDELINES FOR NOVELTY.PDF SECTIONS

### A. Introduction Rewrite Rules

**Current novelty.md Intro Issue**: Too much framing, not enough problem specificity.

**COLM-style fix:**
```
OLD: "Agent frameworks orchestrate multiple LLM-powered agents..."
NEW: "A 5-agent workflow on Apple M4 Pro, where each agent
      accumulates 4K tokens of context, pays 40 seconds of
      prefill cost after server restart."

OLD: "The KV cache produced during prefill is the agent's memory..."
NEW: "The KV cache is the set of (K, V) tensors computed during
      the prefill phase. We persist these tensors to disk in Q4
      quantized format, enabling sub-100ms restoration instead of
      8-second re-computation."

OLD: "KV cache as working memory, agents maintain their computed
      attention state as a persistent, reusable artifact."
NEW: "In multi-phase coordination (Section 4.4), agents carry
      cached attention state across phases: the Warden in the
      Prisoner's Dilemma extends the Phase 1 cache during Phase 2
      interrogation, avoiding re-computation of interrogation context."
```

### B. Evaluation Section Condensing

**Current novelty.md**: 10 subsections (5.1-5.10), ~3000 tokens
**COLM-compatible**: ~1500 tokens, focus on 3-4 key results

**Keep these results; remove others:**
- ✅ 5.5: TTFT Speedup vs Context Length (Figure 5.5 only)
- ✅ 5.7: Batch=2 Concurrent Throughput (Figure 5.7, bar chart)
- ✅ 5.7: Staggered Arrivals (Figure 5.8, 2.6x improvement)
- ❌ Remove 5.2, 5.3, 5.4, 5.6, 5.8, 5.9, 5.10 (too granular for conference)

**New Results Presentation:**
```
Subsection: "End-to-End Performance on Gemma 3 12B"
- Figure: Two-panel plot (cold/warm/hot TTFT vs context length)
- Table 1: System throughput comparison (1K and 4K context)
- Paragraph: Interpretation of results, implications

Subsection: "Multi-User Staggered Arrivals"
- Figure: Bar chart (Sequential B TTFT vs Batched B TTFT)
- Table 2: Raw numbers + speedup
- Paragraph: Real-world relevance ("most requests don't arrive simultaneously")
```

### C. Related Work Condensing

**Current novelty.md**: 7.1-7.7 subsections, ~4000 tokens
**COLM-compatible**: ~1500 tokens, 3-4 sentences per system

**Condensing technique:**
```
OLD (3 paragraphs, 150 words):
"vLLM (Kwon et al., 2023) introduced PagedAttention...
 The closest to our persistence approach is LMCache...
 LMCache demonstrates that disk persistence is viable..."

NEW (1 sentence, 20 words):
"vLLM + LMCache (2023-2024) bring disk persistence to datacenter
 GPUs; we extend this to edge UMA with Q4 quantization."
```

**Keep specific comparisons; remove historical narratives**

---

## IX. WRITING CADENCE AND VOICE

### A. Sentence Structure Rules

**Avoid monotone rhythm:**
- ❌ "This system provides Q4 quantization. It also supports batching. It enables persistence."
- ✅ "The system supports Q4 quantization, batching, and persistence. These three capabilities are absent from upstream mlx-lm."

**Vary sentence length:**
- Short (5-8 words): "The cache survives restarts."
- Medium (15-20): "We persist the cache as safetensors files on disk, enabling sub-100ms restoration."
- Long (30+): "While vLLM + LMCache bring datacenter-scale disk persistence to NVIDIA GPUs, and llama.cpp offers slot-based cache with Q4 support, no system provides per-agent isolation, cross-session prefix matching, and batched Q4 inference together."

**Use active voice almost always:**
- ❌ "The cache was persisted by the block pool" (passive)
- ✅ "The block pool persists the cache to disk" (active)

### B. First-Person Usage

**COLM 2026 allows limited first-person for clarity:**
- ✅ "We measure TTFT as the time from request submission to first token"
- ✅ "Our implementation uses 3 monkey-patches to upstream mlx-lm"
- ❌ Avoid "I," "we believe," "in our opinion" — use "we show," "we measure," "we implemented"

### C. Emphasis Techniques

**Replace bold with:** strategic placement + concrete numbers

```
OLD: "The BatchQuantizedKVCache is **novel**"
NEW: "The BatchQuantizedKVCache enables batched inference over Q4 caches.
      mlx-lm issue #548 explicitly requests this capability, unimplemented
      as of v0.30.5 (January 2026)."
```

---

## X. COLM 2026 TECHNICAL REQUIREMENTS

### A. Margins and Layout
- **Text width**: 5.5 inches (33 picas)
- **Text height**: 9 inches (54 picas)
- **Left margin**: 1.5 inches (9 picas)
- **Top margin**: 1 inch from top of page
- **Font**: Palatino (mandatory via \usepackage{mathpazo})
- **Font size**: 10pt body, 17pt title, 14pt section, 12pt subsection
- **Line spacing**: 11pt vertical spacing

### B. Page Limits
- **Main text**: 9 pages strict maximum
- **References**: Unlimited (on separate pages)
- **Appendices**: Unlimited (optional)

### C. Anonymous Submission
- Use `\usepackage[submission]{colm2026_conference}` (default)
- Author names will be redacted automatically
- Include \thanks{} for funding acknowledgments (not author names)

### D. Compilation

```bash
pdflatex colm2026_conference.tex
bibtex colm2026_conference
pdflatex colm2026_conference.tex
pdflatex colm2026_conference.tex
```

---

## XI. QUICK REFERENCE CHECKLIST

**Before final submission:**

- [ ] No AI language patterns (check SKILL.md list)
- [ ] All figures have captions with interpretation
- [ ] All tables have \caption{} and \label{}
- [ ] All citations use \citet{} or \citep{} (natbib format)
- [ ] No page numbers or headers (COLM removes automatically)
- [ ] Main text ≤ 9 pages
- [ ] Title is 17pt, sections are 14pt, subsections are 12pt
- [ ] All margins conform to COLM (5.5in width, 9in height, 1.5in left)
- [ ] Palatino font (via \usepackage{tgpagella} and \usepackage{mathpazo})
- [ ] No vertical lines in tables (booktabs style)
- [ ] Figures placed immediately after first mention (use [t] or [b] option)
- [ ] References are 10pt, same as body text
- [ ] Hyperlinks are dark blue (#000080 RGB)
- [ ] No single-author "et al." (e.g., "Barrios, 2026" not "Barrios et al.")
- [ ] All equations have \label{} and are referenced as Eq.~\eqref{}
- [ ] Conclusion is specific about next steps, not generic
- [ ] Related Work cites all 12-15 key papers, with 2-3 sentences per system

---

## XII. EXAMPLE REWRITES FROM NOVELTY.MD

### Example 1: Introduction Rewrite

**Current (novelty.md):**
> Modern LLM agent workflows — code assistants, multi-persona debates, collaborative analysis — involve multiple agents maintaining independent conversation histories. Each agent accumulates a system prompt, conversation turns, and context that may span thousands of tokens. When an agent resumes after any interruption (server restart, model swap, or session timeout), the entire conversation must be re-processed from scratch through the model's prefill phase.

**Rewritten for COLM (specific, concrete, compelling):**

> Consider a 5-agent code review workflow on Apple Silicon: a lead engineer (code assistant), reviewer, architect, security expert, and quality lead, each maintaining 4K tokens of conversation history. After a server restart, the entire conversation must be re-prefilled from scratch. On Apple M4 Pro (~500 tokens/second prefill speed), this costs **40 seconds** before the first agent can respond — an unacceptable delay for interactive workflows.

**Changes made:**
- Specific use case (5-agent code review)
- Concrete numbers (4K tokens, 500 tok/s, 40 seconds)
- Emotional impact ("unacceptable delay") via specificity
- One paragraph instead of general description

---

### Example 2: Technical Description Rewrite

**Current (novelty.md):**
> The `BatchQuantizedKVCache` class extends MLX's cache hierarchy to support batched inference over Q4-quantized KV pairs. Three operations make this possible: merge(), update_and_fetch(), and extract().

**Rewritten for COLM:**

> When Agents A and B submit concurrent requests, each with a pre-existing Q4 cache (A: 2048 tokens, B: 512 tokens), we must align their sequences for batched processing. The merge() operation left-pads B's cache and stacks both into a single (batch=2, heads, max_len, dim) tensor. This operation occurs entirely on packed uint32 weights and float16 scales — no dequantization to FP16.

**Changes made:**
- Concrete example with numbers
- Explain the "why" (alignment for batching)
- Show what happens technically (left-padding, stacking)
- Note the constraint (no dequantization)

---

### Example 3: Related Work Rewrite

**Current (novelty.md, 150 words):**
> vllm-mlx (Barrios, arXiv:2601.19139, January 2026) brings vLLM-style continuous batching and content-based prefix caching to Apple Silicon. No disk persistence, no per-agent isolation, FP16-only KV cache. A separate vllm-metal (v0.1.0, January 2026) — an official vLLM community plugin under the vllm-project GitHub organization — aims to run vLLM natively on Metal with paged attention and GQA support, bypassing MLX. Early-stage but adds credibility to the Apple Silicon serving ecosystem.

**Rewritten for COLM, 3 sentences, 50 words:**

> vllm-mlx and vllm-metal bring continuous batching to Apple Silicon, but neither supports disk cache persistence or per-agent isolation. Unlike Semantic, they retain only FP16 caches in-memory, limiting context capacity on 24GB systems to ~8K tokens.

**Changes made:**
- Combined two paragraphs into one
- Focused on the comparison point (persistence + isolation + quantization)
- Removed "official plugin" narrative detail
- Added specific capacity comparison (8K vs 35K tokens)

---

## XIII. TIMELINE AND WORKFLOW

### Phase 1: Structure Refinement (1 session)
1. Condense Sections 5 and 7 to COLM page limits
2. Remove verbose framing, add concrete numbers
3. Create 4-5 key figures
4. Extract 2-3 key result tables

### Phase 2: Writing Pass (2-3 sessions)
1. Apply humanizer rules to prose (eliminate AI patterns)
2. Rewrite introduction for specificity
3. Compress related work
4. Clarify discussion section

### Phase 3: LaTeX Preparation (1 session)
1. Set up COLM 2026 template structure
2. Convert markdown tables to LaTeX booktabs
3. Add figure captions and cross-references
4. Build bibliography from novelty.md citations

### Phase 4: Final Polish (1 session)
1. Check margins, fonts, spacing
2. Verify all equations are labeled and referenced
3. Review figures for quality and readability
4. Final grammar and consistency pass

---

**Document Version**: 1.0
**Date Created**: February 4, 2026
**LaTeX Template**: COLM 2026
**Humanizer Rules**: Based on Wikipedia:Signs of AI writing
**Target Length**: 9 pages main text + unlimited references
