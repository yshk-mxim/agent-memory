# Pre-Flight Checklist: COLM 2026 Submission
## "Agent Memory Below the Prompt"

**Date**: 2026-02-04
**Purpose**: Final validation before compilation and submission

---

## ✅ Compilation Prerequisites

### Required Files Present
- [ ] `semantic_colm2026.tex` (main document)
- [ ] `semantic_colm2026.bib` (bibliography, 25+ entries)
- [ ] `colm2026_conference.sty` (style file)
- [ ] `colm2026_conference.bst` (bibliography style)
- [ ] `math_commands.tex` (math macros)
- [ ] `fancyhdr.sty` (headers package)
- [ ] `natbib.sty` (citations package)

### Figures Present (TikZ Source)
- [ ] `figures/fig_architecture.tex` (System architecture)
- [ ] `figures/fig_ttft_scaling.tex` (TTFT scaling chart)
- [ ] `figures/fig_staggered.tex` (Staggered arrivals)
- [ ] `figures/fig_uma_comparison.tex` (UMA comparison)

### Software Installed
- [ ] pdflatex (version 3.14+)
- [ ] bibtex (version 0.99d+)
- [ ] TikZ and pgfplots packages

**Verification command**:
```bash
cd /Users/dev_user/semantic/novelty/paper/
ls -la *.tex *.bib *.sty *.bst figures/*.tex
pdflatex --version
bibtex --version
```

---

## ✅ Content Validation

### Abstract (178 words, target: 150-180)
- [ ] States problem (multi-agent cold-start on edge)
- [ ] Three contributions listed
- [ ] Headline numbers present (2.0--4.3×, 81.6×, 1.95--10.5×, 72% KV cache)
- [ ] Open-source mentioned (anonymized)
- [ ] No em dashes
- [ ] Word count: 150-180

**Verification command**:
```bash
grep -A 20 "\\begin{abstract}" semantic_colm2026.tex | wc -w
# Should output: 170-185 words (LaTeX commands add ~5-7 words)
```

### Main Sections Complete
- [ ] 1. Introduction (1.0 page)
- [ ] 2. Background (0.7 pages)
- [ ] 3. System Design (2.5 pages)
- [ ] 4. Evaluation (1.5 pages)
- [ ] 5. Discussion (1.25 pages)
- [ ] 6. Related Work (1.0 page)
- [ ] 7. Conclusion (0.5 pages)
- [ ] **Total**: ~8.75 pages (within ≤9 page limit)

### Appendices Complete
- [ ] A: safetensors Q4 Format (tensor schema + metadata)
- [ ] B: MLX Lazy Evaluation Pitfalls (6-row table)
- [ ] C: Benchmark Configuration (hardware + software + hyperparameters)

### Figures Integrated
- [ ] Figure 1 referenced in Section 3.0
- [ ] Figure 2 referenced in Section 4.2
- [ ] Figure 3 referenced in Section 4.4
- [ ] Figure 4 referenced in Section 2.2
- [ ] All `\input{figures/...}` commands present
- [ ] All figures have `\label{fig:...}` for cross-reference

### Tables Present
- [ ] Table 1: TTFT scaling (5 columns: 1K, 2K, 4K, 8K, 16K)
- [ ] Table 2: Batched throughput (Sequential vs Batched)
- [ ] Tables in Discussion/Related Work as needed
- [ ] All tables use `booktabs` style

---

## ✅ Citation Quality

### All Citations Resolved (0 TODO remaining)
```bash
grep -n "TODO" semantic_colm2026.tex
# Should output: (empty)
```

### Bibliography Complete (25+ entries)
- [ ] vllm, SGLang, vllm-mlx (KV cache systems)
- [ ] KIVI, KVQuant, CacheGen, CommVQ, QuantSpec (quantization)
- [ ] EM-LLM, Memory3, MemArt, RAPTOR (agent memory)
- [ ] KVCOMM, KVFlow, KVLink (multi-agent)
- [ ] Apple M4 Pro, NVIDIA DGX Spark, A100 specs (hardware)
- [ ] All with complete author lists, year, venue
- [ ] No placeholder [TODO] entries

**Verification command**:
```bash
grep -c "^@" semantic_colm2026.bib
# Should output: 25 or higher
```

### All `\cite{}` Commands Match BibTeX Keys
```bash
# Extract all \cite{...} from .tex
grep -o '\\cite{[^}]*}' semantic_colm2026.tex | sort -u > cited.txt

# Extract all @type{key,...} from .bib
grep "^@" semantic_colm2026.bib | grep -o '{[^,]*' | tr -d '{' | sort > available.txt

# Check for mismatches
comm -23 cited.txt available.txt
# Should output: (empty) or only \cite{} commands, not BibTeX keys
```

---

## ✅ Writing Quality (Zero AI Slop)

### ABSOLUTE PROHIBITION: Em Dashes
```bash
# Check for any form of em dash
grep -E "---|—|–|\\\\textemdash|\\\\emdash" semantic_colm2026.tex
# Should output: (empty)
```

### No Banned Vocabulary
```bash
# Check for AI slop words
grep -iE "landscape|paradigm shift|groundbreaking|showcase|leverage|cutting-edge" semantic_colm2026.tex
# Should output: (empty)

grep -iE "Additionally|Furthermore|Notably|Interestingly|It should be noted" semantic_colm2026.tex
# Should output: (empty)
```

### No Decorative Formatting
- [ ] No bolded inline headers mid-paragraph
- [ ] No italics for emphasis (only for terms/variables)
- [ ] No excessive capitalization

### Active Voice Predominant
- [ ] Introduction uses active voice ("We present", "This work makes")
- [ ] Passive only for established conventions ("KV cache is stored")
- [ ] Methods section can use passive where appropriate

---

## ✅ Numerical Claims Verification

### All Claims Traced to Source
From `review/audit.md` (27 claims audited, all verified or corrected):

- [ ] 2.0--4.3× E2E speedup → benchmark data
- [ ] 81.6× TTFT (hot, 16K) → 68,898ms / 844ms
- [ ] 1.95--10.5× TTFT (warm) → Table 1 range
- [ ] 72% KV cache savings → calculation in Section 3.2
- [ ] 48% system throughput increase → (49.4 - 33.4) / 33.4
- [ ] 2.6× User B speedup → 24.5s / 9.6s
- [ ] 4% User A penalty → (7.3 - 7.0) / 7.0
- [ ] 1.86× combined TTFT → 31.5s / 16.9s

### Critical Fixes Applied (from audit Phase 4)
- [ ] "1.1--2.0×" changed to "1.95--10.5×" (warm speedup)
- [ ] "35%" changed to "48%" (throughput increase)
- [ ] "50.4s vs 90.8s" changed to "16.9s vs 31.5s" (staggered TTFT)
- [ ] "4 architectures" clarified as "4 supported (2 benchmarked)"
- [ ] "72% memory savings" now says "72% KV cache memory savings"

### Hardware Specifications Cited
- [ ] M4 Pro 273 GB/s → \cite{apple2024m4pro}
- [ ] DGX Spark 273 GB/s → \cite{nvidia2025dgxspark}
- [ ] A100 ~10,000 tok/s → \cite{nvidia2024a100bench}

---

## ✅ Positioning and Framing (from feedback.md responses)

### Explicit Systems Paper Positioning
```bash
grep -A 5 "systems paper" semantic_colm2026.tex | head -10
# Should show positioning statement in Discussion section
```

- [ ] Introduction or Discussion states: "This is a systems paper focused on practical edge inference, not claiming algorithmic novelty"
- [ ] Contributions section lists "system-level" and "engineering" contributions separately
- [ ] Working memory framed as conceptual positioning, not main contribution

### Caveats Added
- [ ] Hot cache speedup noted as "expected for in-memory cache access"
- [ ] DGX Spark comparison includes memory capacity caveat (128 GB vs 24 GB)
- [ ] Total memory context added for 72% KV cache savings
- [ ] Model architecture claim shows "4 supported (2 benchmarked)"

---

## ✅ Anonymity (Submission Mode)

### No Identifying Information
```bash
# Check for author names
grep -i "author" semantic_colm2026.tex | grep -v "Anonymous"
# Should output: (only author field with "Anonymous Authors")

# Check for institution names
grep -iE "university|stanford|berkeley|mit|google|openai|apple|nvidia" semantic_colm2026.tex
# Should output: (only in citations, not as affiliation)

# Check for identifying URLs
grep "github.com/[^/]*/" semantic_colm2026.tex
# Should output: [anonymized] or generic repo references
```

### Submission Mode Active
```latex
% In semantic_colm2026.tex, line ~6:
\usepackage[submission]{colm2026_conference}
```
- [ ] `[submission]` option present (not commented out)
- [ ] Produces anonymous PDF with "Anonymous Authors"

---

## ✅ Figure and Table Quality

### TikZ Figures Compile
- [ ] No TikZ compilation errors in .log file
- [ ] All coordinate systems defined
- [ ] Colors consistent (blue for agents, green for system, red for cold, orange for warm, etc.)
- [ ] Font sizes readable (minimum 8pt)

### pgfplots Data Correct
- [ ] Figure 2 (TTFT): Data matches Table 1 exactly
- [ ] Figure 3 (Staggered): Bars match Section 4.4 numbers
- [ ] Axes labeled with units (ms, seconds, tok/s)
- [ ] Legend readable and positioned well

### Table Formatting
- [ ] Booktabs rules only (`\toprule`, `\midrule`, `\bottomrule`)
- [ ] No vertical lines
- [ ] Numbers right-aligned
- [ ] Text left-aligned
- [ ] Column headers clear

---

## ✅ Cross-Reference Integrity

### All `\ref{}` Resolve
```bash
# Compile and check for undefined references
pdflatex semantic_colm2026.tex 2>&1 | grep "Reference.*undefined"
# Should output: (empty after 3 pdflatex passes)
```

- [ ] No "??" in compiled PDF
- [ ] Section references work (Section X)
- [ ] Figure references work (Figure X)
- [ ] Table references work (Table X)
- [ ] Equation references work (if any)

### Citation Cross-References
- [ ] All `\cite{key}` resolve to bibliography entries
- [ ] No "[?]" in compiled PDF
- [ ] Author-year format displays correctly (Author, Year)

---

## ✅ Appendix Completeness

### Appendix A: safetensors Q4 Format
- [ ] Tensor naming schema documented
- [ ] Metadata JSON example provided
- [ ] Group size and block size noted

### Appendix B: MLX Lazy Evaluation Pitfalls
- [ ] 6-row table present
- [ ] Symptom, Root Cause, Fix columns
- [ ] Rule of thumb stated

### Appendix C: Benchmark Configuration
- [ ] Hardware specs (Mac Mini M4 Pro MX2E3LL/A, 24 GB, 273 GB/s)
- [ ] Software versions (macOS, Python, MLX, mlx-lm, Transformers)
- [ ] Model specifications (Gemma 3, DeepSeek-Coder-V2-Lite)
- [ ] Hyperparameters (temperature, top-p, output length, etc.)
- [ ] Benchmark script paths

---

## ✅ Review Documents Complete (Phase 5)

### All 5 Review Documents Generated
- [ ] `review/audit.md` - Claim verification (27 claims, all verified/fixed)
- [ ] `review/evidence.md` - Calculation reproduction (15 calculations checked)
- [ ] `review/feedback.md` - Hostile critique (REJECT stance, issues identified)
- [ ] `review/investigation.md` - Forensic analysis (cross-references validated)
- [ ] `review/debate.md` - 6-expert panel (consensus: CONDITIONAL ACCEPT)

### Critical Issues from Reviews Addressed
- [ ] Memory savings clarified (Tier 1 from debate)
- [ ] Positioning statement added (Tier 1)
- [ ] DGX Spark caveats added (response to hostile critique)
- [ ] Hot cache disclaimer added (response to hostile critique)

---

## ✅ Final Compilation Test

### Successful Compilation
```bash
cd /Users/dev_user/semantic/novelty/paper/
pdflatex semantic_colm2026.tex
bibtex semantic_colm2026
pdflatex semantic_colm2026.tex
pdflatex semantic_colm2026.tex
```

- [ ] `semantic_colm2026.pdf` generated
- [ ] No fatal errors in `.log` file
- [ ] BibTeX processed all citations (check `.blg` file)
- [ ] PDF opens without errors

### Output Validation
```bash
pdfinfo semantic_colm2026.pdf | grep Pages
# Should show: Pages: 11-14 (9 main + 2-5 appendix pages)

ls -lh semantic_colm2026.pdf
# Should show: ~200-500 KB file size
```

---

## ✅ Visual Inspection (Phase 7)

### Page Layout
- [ ] Title page formatted correctly
- [ ] Abstract on page 1, single paragraph
- [ ] No orphan/widow lines (single lines at page top/bottom)
- [ ] Section headers not at bottom of page alone
- [ ] Page breaks logical (not mid-paragraph when avoidable)

### Figure Placement
- [ ] Figures appear on same page or after first reference
- [ ] Never before first reference
- [ ] Captions below figures
- [ ] Format: **Figure N.** Caption text...

### Table Placement
- [ ] Tables appear near first reference
- [ ] Captions above tables
- [ ] Format: **Table N.** Caption text...

### Typography
- [ ] Font: Palatino 10pt (body text)
- [ ] Consistent spacing
- [ ] No overfull hbox extending past margins (check `.log`)
- [ ] Equations render correctly
- [ ] Math symbols display properly

### References Section
- [ ] Starts on new page
- [ ] Author-year format (not numbered)
- [ ] Alphabetically sorted by first author
- [ ] All entries complete (no missing fields)
- [ ] URLs formatted correctly (not breaking layout)

---

## ✅ Submission Package

### Required Files
- [ ] `semantic_colm2026.pdf` (main submission)
- [ ] `semantic_colm2026.tex` (source, if required)
- [ ] `semantic_colm2026.bib` (source, if required)
- [ ] `figures/*.tex` (source, if required)

### Optional Supplementary Materials
- [ ] `supplementary.pdf` - Extended appendices (if needed)
- [ ] `code.zip` - Source code (anonymized GitHub repo)
- [ ] `data.zip` - Full benchmark data (if requested)

### Metadata for Submission System
- [ ] Paper title: "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices"
- [ ] Track: Systems / Edge Computing (check COLM 2026 tracks)
- [ ] Keywords: KV cache, edge inference, multi-agent, quantization, Apple Silicon
- [ ] Abstract text (copy from PDF)

---

## ✅ Final Checklist Summary

**GREEN LIGHT if ALL checked**:

### Critical (Must Pass)
- [ ] ✅ PDF compiles without fatal errors
- [ ] ✅ Page count ≤ 9 pages (main text)
- [ ] ✅ All citations resolve (no "?" in output)
- [ ] ✅ All figures render correctly
- [ ] ✅ Zero em dashes verified
- [ ] ✅ Anonymous (no identifying information)
- [ ] ✅ All numerical claims verified or corrected

### Important (Should Pass)
- [ ] ✅ All 4 figures integrated
- [ ] ✅ All 3 appendices complete
- [ ] ✅ Bibliography has 25+ entries
- [ ] ✅ Positioning as systems paper clear
- [ ] ✅ 72% savings clarified as KV-only
- [ ] ✅ Review documents complete (Phase 5)

### Nice to Have (Optional)
- [ ] 🟡 Error bars or std dev in tables (Tier 1 from debate)
- [ ] 🟡 Perplexity evaluation (Tier 2 from debate)
- [ ] 🟡 Additional model benchmarks (Tier 2)
- [ ] 🟡 Baseline comparisons (llama.cpp, etc.)

---

## 🚀 Ready for Submission

**If all critical and important items are checked**: ✅ **READY TO SUBMIT**

**If any critical items unchecked**: ⚠️ **FIX BEFORE SUBMISSION**

**If optional items desired**: 🟡 **Address in revision after initial review**

---

**Next Step**: Run final compilation, then proceed to `COMPILATION_GUIDE.md` for detailed instructions.
