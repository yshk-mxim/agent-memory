# COLM 2025 Style Calibration Notes

## Papers Analyzed

### 1. PyramidKV: Dynamic KV Cache Compression (COLM 2025)
- **Source**: https://openreview.net/forum?id=ayi7qezU87
- **Abstract**: ~240 words
- **Writing patterns**:
  - First-person collective: "We developed", "Our experimental evaluations"
  - Semicolons used to connect related clauses
  - Technical terminology density: high
  - Parallel structure in presenting benchmark results
  - Definitive claims with quantified performance numbers

### 2. E²-RAG: Editable Efficient RAG (COLM 2025)
- **Source**: https://openreview.net/forum?id=ZZ4tcxJvux
- **Abstract**: ~240 words
- **Writing patterns**:
  - Minimal hedging language
  - Direct, confident claims: "achieves nearly 40x faster editing"
  - First-person plural: "we propose"
  - Formal academic tone
  - Emphasis on measurable performance gains

### 3. AIOS: LLM Agent Operating System (COLM 2025)
- **Source**: https://arxiv.org/abs/2403.16971
- **Abstract**: ~155 words (shorter, more focused)
- **Writing patterns**:
  - Problem-solution structure in abstract
  - Passive voice predominant
  - Technical precision without oversimplification
  - Clear hierarchical organization

## Key Observations for Our Paper

### Abstract Length
- **Range**: 155-240 words
- **Target for our paper**: 150-180 words (conservative, focused)
- **Structure**: Problem → Insight → Contributions → Key results → Availability

### Voice and Tone
- **First person is acceptable**: "We propose", "We developed", "Our system"
- **Balance**: Active voice preferred, passive when conventional
- **Confidence**: Direct claims with quantified evidence, minimal hedging

### Punctuation
- **Semicolons**: Used moderately to connect related clauses
- **Em dashes**: Status UNKNOWN from abstracts only (need to verify in full papers)
- **Parentheticals**: Used sparingly for clarifications

### Technical Content
- **Specificity**: Concrete numbers, named benchmarks, explicit comparisons
- **Terminology**: Domain-specific language expected and appropriate
- **Parallel structure**: When presenting multiple results/benchmarks

### Citation Patterns
- Cannot determine from abstracts alone
- Likely author-year or numbered format (COLM uses natbib)

## Hard Rules for Our Paper (from CLAUDE.md)

### ABSOLUTE PROHIBITION
- **ZERO em dashes** in any form (---, —, –, \textemdash)
  - Replace with: period + new sentence, parentheses, comma, colon, restructuring

### Banned Vocabulary
- landscape, paradigm shift, groundbreaking, showcase, leverage, cutting-edge
- Additionally, Furthermore, Notably, Interestingly, It should be noted that
- No bolded inline headers mid-paragraph
- No decorative formatting

### Required Patterns
- Active voice by default
- Numbers always with units and baselines
- Every claim traced to source or benchmark
- Vary sentence length (no monotone rhythm)

## Section Architecture (from analysis)

### Main Text Allocation (9 pages total)
| Section | Target Pages |
|---------|-------------|
| Abstract | 0.3 |
| 1. Introduction | 1.0 |
| 2. Background | 0.7 |
| 3. System Design | 2.5 |
| 4. Evaluation | 1.5 |
| 5. Discussion | 1.25 |
| 6. Related Work | 1.0 |
| 7. Conclusion | 0.5 |
| **Total** | 8.75 (0.25 margin) |

### Appendices (unlimited)
- A-H: Technical details, additional evaluation, case studies

## Next Steps

1. Verify citation style in actual COLM template
2. Create TikZ figures following academic paper conventions
3. Begin citation verification (Phase 1)
4. Draft sections following abstract → conclusion → figures → body order
