# Sprint 00: Pilot Testing (Week 0)

**Duration**: 5 days
**Goal**: Validate full pipeline with n=5 examples before scaling to n=50
**Status**: Ready to start

---

## Objectives

- [ ] Validate full pipeline works end-to-end
- [ ] Test all 4 baseline conditions on small sample
- [ ] Identify and fix technical issues early
- [ ] Refine automated metrics implementation
- [ ] Confirm instrumentation captures all data

---

## Daily Breakdown

### Monday: Setup & Initial Runs

**Morning (3h)**:
- [ ] Set up embedding model for semantic clustering:
  - Install sentence-transformers (`pip install sentence-transformers`)
  - Download `all-MiniLM-L6-v2` model (~80MB)
  - Test embedding generation (should be <50ms per turn)
  - Verify memory footprint (<200MB)

- [ ] Generate 5 pilot examples (1 per domain)
  - Coding (multi-file debugging + docs + review)
  - Research (literature review + experiment + writing)
  - Business (technical + strategy + synthesis)
  - Support (technical support + billing + account)
  - Creative (storytelling + editing + analysis)

**Afternoon (3h)**:
- [ ] Implement embedding-based semantic clustering:
  - Cluster discovery: Analyze example, identify semantic roles
  - Cluster embeddings: Create prototype embeddings for each role
  - Turn routing: Cosine similarity to assign turns to clusters
  - Test on 1 pilot example (verify routing makes sense)

- [ ] Implement all 4 conditions if not already done:
  - Sequential (baseline, all in one context)
  - Prompted (soft isolation via instructions)
  - Turn-based (naive temporal boundaries)
  - Semantic (RDIC - our method with embedding-based routing)

**Deliverable**: `data/pilot_examples.json` (5 examples), `src/embedding_clustering.py`

---

### Tuesday: Run Experiments & Metrics

**Morning (3h)**:
- [ ] Run pilot experiment: 5 examples × 4 conditions = 20 runs
  - Sequential: 5 runs
  - Prompted: 5 runs
  - Turn-based: 5 runs
  - Semantic: 5 runs
- [ ] Capture all outputs and telemetry

**Afternoon (2h)**:
- [ ] Apply automated metrics to all 20 outputs
  - Test contamination detection metrics
  - Test specialization measurement metrics
  - Test synthesis quality metrics
  - Test standard NLP metrics

**Deliverable**: `results/pilot/` with 20 outputs + metrics

---

### Wednesday: Refinement

**Morning (2h)**:
- [ ] Review metric results
- [ ] Identify issues:
  - Metrics not discriminating between conditions?
  - Unexpected patterns?
  - Calculation errors?
- [ ] Refine metrics based on findings

**Afternoon (2h)**:
- [ ] Test instrumentation system
  - Verify timing data captured
  - Verify memory profiling works
  - Verify cache size tracking accurate
  - Test telemetry export

**Deliverable**: `evaluation/metrics_v1.py` (refined)

---

### Thursday: Bug Fixes & Documentation

**Morning (3h)**:
- [ ] Fix any technical bugs discovered
  - MLX issues
  - Cache management problems
  - Metric calculation errors
  - Instrumentation overhead issues

**Afternoon (1h)**:
- [ ] Document lessons learned
  - What worked well?
  - What needs changing?
  - Any surprises?
  - Recommendations for Phase 1

**Deliverable**: `docs/pilot_lessons.md`

---

### Friday: Finalize & Lock Down

**Morning (2h)**:
- [ ] Finalize experiment protocol
  - Lock down procedure for Phase 1
  - Document any parameter changes
  - Confirm all metrics ready
  - Verify dataset generation process

**Afternoon (1h)**:
- [ ] Review checklist
- [ ] Confirm ready to proceed to Phase 1
- [ ] Brief planning for Week 1

**Deliverable**: Sprint 00 complete, ready for Sprint 01

---

## Success Criteria (All Must Pass)

- [x] All 4 conditions run without crashes
- [x] Automated metrics implemented and tested
- [x] Metrics show clear separation between conditions (semantic should win)
- [x] Instrumentation captures all required data (<5% overhead)
- [x] No major blockers identified
- [x] Documentation complete

---

## Risk Mitigation

**Risk**: Pilot shows no separation between conditions
- **Mitigation**: Review metric implementations, check if examples are too simple
- **Escalation**: Regenerate examples with clearer agent boundaries

**Risk**: Technical crashes during pilot
- **Mitigation**: Debug immediately, don't proceed until stable
- **Escalation**: Simplify conditions or reduce model size if memory issues

**Risk**: Instrumentation overhead >5%
- **Mitigation**: Optimize or run separate instrumented vs clean runs
- **Escalation**: Report metrics post-hoc instead of real-time

---

## Tools & Resources

**Required**:
- MLX framework (already installed)
- Gemma 3 12B 4-bit model (already downloaded)
- Python evaluation scripts (`src/semantic_isolation_mlx.py`)
- Automated metrics suite (to be implemented)

**Data**:
- 5 pilot examples (to be generated)

**Compute**:
- Mac with 24GB RAM
- ~3 hours of compute time (20 runs × 9 min/run)

---

## Next Sprint

**Sprint 01**: Dataset Generation + Automated Metrics Implementation (Week 1)

---

**Created**: 2026-01-23
**Status**: Ready to start
**Blockers**: None
