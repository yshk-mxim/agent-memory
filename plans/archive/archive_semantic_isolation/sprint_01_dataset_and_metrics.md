# Sprint 01: Dataset Generation + Automated Metrics Suite (Week 1)

**Duration**: 5 days
**Goal**: Generate full dataset (n=50) and implement comprehensive automated metrics
**Status**: Pending Sprint 00 completion

---

## Objectives

- [ ] Generate 50 diverse multi-agent examples across 5 domains
- [ ] Ensure balanced domain coverage (10 per domain)
- [ ] Implement comprehensive automated metrics suite (19 metrics (16 mechanical + 3 Claude AI judge))
- [ ] Validate metrics on pilot data
- [ ] Create train/val/test splits

---

## Daily Breakdown

### Monday: Domain Setup + Coding Examples + Embedding Clustering

**Morning (3h)**:
- [ ] Finalize embedding-based clustering implementation:
  - Automatic cluster discovery (analyze task description, identify roles)
  - Prototype embeddings for each semantic role
  - Turn-by-turn routing via cosine similarity
  - Validate on pilot data (clustering should match ground truth ~80%+)

- [ ] Generate coding examples (10)
  - Multi-file debugging scenarios
  - Documentation writing tasks
  - Code review synthesis
  - Mix of Python, JavaScript, Go
  - Ensure clear agent boundaries (debug vs docs vs review)

**Afternoon (2h)**:
- [ ] Begin automated metrics implementation
  - Set up metrics framework
  - Implement contamination detection (TF-IDF similarity)
  - Test on pilot data

**Deliverable**: 10 coding examples, contamination metric implemented, `src/embedding_clustering.py` finalized

---

### Tuesday: Research + Business Examples

**Morning (4h)**:
- [ ] Generate research examples (10)
  - Literature review + experiment design + paper writing
  - Clear separation: analysis vs methods vs writing

**Afternoon (3h)**:
- [ ] Generate business examples (10)
  - Technical analysis + strategy planning + synthesis
  - Clear separation: technical vs strategic thinking

**Deliverable**: 20 more examples (total: 30)

---

### Wednesday: Support + Creative Examples

**Morning (3h)**:
- [ ] Generate support examples (10)
  - Technical troubleshooting + billing/account + coordination
  - Mix of software, hardware, account issues

**Afternoon (3h)**:
- [ ] Generate creative examples (10)
  - Story writing + editing + analysis
  - Poetry + review + meta-commentary
  - Script writing + character analysis + synthesis

**Deliverable**: 20 more examples (total: 50 complete)

---

### Thursday: Metrics Implementation

**Morning (4h)**:
- [ ] Implement specialization metrics:
  - Keyword density analyzer
  - Technical density scorer (code/jargon detection)
  - Style consistency calculator
  - Domain classifier (train simple classifier)

**Afternoon (2h)**:
- [ ] Implement synthesis quality metrics:
  - Information coverage (% specialist content in synthesis)
  - Semantic similarity to both specialists
  - Coherence scores (BERTScore variant)
  - Novel content detection

**Deliverable**: 8 more metrics implemented (12 total)

---

### Friday: Standard Metrics + Validation

**Morning (2h)**:
- [ ] Implement standard NLP metrics:
  - ROUGE-L
  - BERTScore
  - Perplexity
  - Sentence-transformer embeddings

**Afternoon (4h)**:
- [ ] Implement Claude AI Judge metrics:
  - Contamination detection (0-5 scale, via Claude Sonnet 4.5)
  - Specialization quality (0-5 scale)
  - Synthesis quality (0-5 scale)
  - Use prompts from `evaluation/claude_judge_prompts.md`
  - Test on pilot data (5 examples × 4 conditions = 20 evaluations)
  - Temperature=0.0 for reproducibility

- [ ] Validate all 19 metrics on pilot data (16 mechanical + 3 Claude)
  - Check calculations are correct
  - Verify discriminative power
  - Test on all 4 conditions
  - Document target values and interpretation

**Evening (1h)**:
- [ ] Create data splits:
  - Train: 30 examples (for any learning-based metrics)
  - Val: 10 examples (for metric tuning)
  - Test: 10 examples (for final evaluation)

**Deliverable**: Complete metrics suite (19 metrics (16 mechanical + 3 Claude AI judge)), dataset splits

---

## Metrics Suite Overview

### Contamination Detection (4 metrics)

1. **TF-IDF Cross-Cluster Similarity**: Measure lexical overlap between specialist outputs
   - Target: <0.3 (low contamination)
   - Formula: cosine(tfidf(output_1), tfidf(output_2))

2. **Domain Vocabulary Leakage**: Count domain-specific terms in wrong cluster
   - Target: <5% (minimal leakage)
   - Method: Maintain domain keyword dictionaries, count occurrences

3. **Lexical Overlap Percentage**: Direct word overlap between outputs
   - Target: <20% (allowing common words)
   - Formula: |words(A) ∩ words(B)| / |words(A) ∪ words(B)|

4. **Cross-Cluster Keyword Bleeding**: Specialist-specific keywords appearing in other specialist's output
   - Target: <3 keywords
   - Method: Extract top-10 keywords per cluster, count cross-appearances

### Specialization Measurement (4 metrics)

5. **Cluster-Specific Keyword Density**: Frequency of relevant keywords for each cluster's domain
   - Target: >0.15 (high specialization)
   - Formula: relevant_keywords / total_words

6. **Technical Density Score**: Code tokens, technical jargon, domain terms
   - Target: Tech cluster >0.25, non-tech <0.10
   - Method: Regex patterns + domain dictionaries

7. **Style Consistency Score**: Vocabulary diversity within vs between clusters
   - Target: Within-cluster > between-cluster
   - Formula: Type-token ratio consistency

8. **Domain Classifier Confidence**: Train simple classifier, measure prediction confidence
   - Target: >0.80 confidence
   - Method: Train LogReg on train set, test on val

### Synthesis Quality (4 metrics)

9. **Information Coverage**: Percentage of specialist content captured in synthesis
   - Target: >70%
   - Method: ROUGE-L between specialists and synthesis

10. **Dual Semantic Similarity**: Synthesis should be similar to BOTH specialists
    - Target: Both >0.70
    - Formula: cosine(embed(synthesis), embed(specialist_i))

11. **Coherence Score**: BERTScore variant for fluency
    - Target: >0.85
    - Method: BERTScore between synthesis and reference

12. **Novel Content Ratio**: Synthesis adds value beyond concatenation
    - Target: 10-30% novel content
    - Method: Measure unique trigrams in synthesis not in either specialist output

### Standard NLP Metrics (4 metrics)

13. **ROUGE-L**: Longest common subsequence
14. **BERTScore**: Semantic similarity via embeddings
15. **Perplexity**: Fluency via language model
16. **Embedding Similarity**: Sentence-transformer cosine similarity

### Claude AI Judge Metrics (3 metrics)

**Purpose**: Use Claude Sonnet 4.5 as automated AI judge for qualitative evaluation

17. **Claude Contamination Score**: AI assessment of cross-domain leakage
   - Target: <1.0 (minimal contamination)
   - Scale: 0-5 (0=clean, 5=severe mixing)
   - Method: Claude evaluates if Output A contains concepts/terminology from Output B's domain
   - Provides: Score + evidence + reasoning + examples
   - Prompt: See `evaluation/claude_judge_prompts.md` (Prompt 1)

18. **Claude Specialization Score**: AI assessment of domain-specific focus
   - Target: >4.0 (high specialization)
   - Scale: 0-5 (0=generic, 5=exceptional expertise)
   - Method: Claude evaluates domain terminology, depth, and insights
   - Provides: Score + domain keywords + depth indicators + examples
   - Prompt: See `evaluation/claude_judge_prompts.md` (Prompt 2)

19. **Claude Synthesis Score**: AI assessment of integration quality
   - Target: >4.0 (excellent synthesis)
   - Scale: 0-5 (0=failed, 5=exceptional integration)
   - Method: Claude evaluates coverage, integration, coherence, added value
   - Provides: Score + coverage % + integration quality + novel insights
   - Prompt: See `evaluation/claude_judge_prompts.md` (Prompt 3)

**Implementation**:
- Use Claude Code CLI Task tool (recommended, no sandbox bypass needed)
- Model: `claude-sonnet-4-5-20250929`
- Temperature: 0.0 (reproducibility)
- Cost: ~$0.05 per evaluation (~$30 for 200 outputs × 3 evaluations = $10 total with batching)

**Benefits**:
- Bridges mechanical metrics and human evaluation
- Captures qualitative nuances mechanical metrics miss
- Reproducible (temperature=0)
- Explainable (provides reasoning)
- Scalable (200 evaluations in ~1 hour)
- Accepted in recent research (LLM-as-judge correlates r=0.85-0.90 with humans)

---

## Dataset Statistics Target

**Total**: 50 examples
**Per domain**: 10 examples each
**Quality threshold**: >80% manually validated as high-quality
**Balance check**: No domain has >20% low-quality examples

**Validation criteria per example**:
- [ ] Clear agent boundaries (identifiable specialist roles)
- [ ] Realistic scenario (not contrived)
- [ ] Sufficient complexity (requires multi-turn interaction)
- [ ] Synthesis opportunity (coordinator adds value)

---

## Success Criteria

- [x] 50 examples generated with balanced domain coverage
- [x] All examples pass quality validation (>80% high-quality)
- [x] Train/val/test splits created (30/10/10)
- [x] All 16 automated metrics implemented and tested
- [x] Metrics show discriminative power on pilot data
- [x] Documentation complete for all metrics

---

## Risk Mitigation

**Risk**: Dataset generation too slow (>3 days)
- **Mitigation**: Use template-based generation, batch API calls
- **Escalation**: Reduce to n=40 if quality maintained

**Risk**: Metrics don't discriminate between conditions
- **Mitigation**: Review metric implementations, increase sensitivity
- **Escalation**: Add more metrics or adjust formulas

**Risk**: Quality imbalance across domains
- **Mitigation**: Regenerate low-quality domain examples
- **Escalation**: Focus on 3 best domains (n=15 each, total=45)

---

## Dependencies

**Requires from Sprint 00**:
- Pilot data and lessons learned
- Working pipeline for all 4 conditions
- Stable MLX setup

**Blocks**:
- Sprint 02 (Experiment Runs)
- Sprint 03 (Analysis)

---

## Tools & Libraries

**For dataset generation**:
- Claude API (for generating examples)
- Python scripts for template-based generation

**For metrics**:
- `sklearn` (TF-IDF, classifiers)
- `transformers` (BERTScore, embeddings)
- `sentence-transformers` (semantic similarity)
- `nltk` or `rouge-score` (ROUGE metrics)
- Custom Python scripts for domain-specific metrics

---

## Deliverables

- [ ] `data/full_dataset_v1.json` (50 examples)
- [ ] `data/domain_statistics.md` (balance analysis)
- [ ] `data/splits/` (train/val/test JSONs)
- [ ] `evaluation/automated_metrics.py` (all 19 metrics (16 mechanical + 3 Claude AI judge))
- [ ] `evaluation/metric_validation.md` (metric performance on pilot)
- [ ] `docs/metrics_documentation.md` (formulas, targets, interpretation)

---

## Next Sprint

**Sprint 02**: Experiment Runs (Weeks 2-3)

---

**Created**: 2026-01-23
**Status**: Pending Sprint 00
**Blockers**: Sprint 00 must complete successfully
