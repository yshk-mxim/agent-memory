# Day 9 (Tuesday): R1 Clustering - Runs 2 and 3 (Stability)

**Week 2 - Day 2**

---

**Objectives:**
- Verify R1 clustering stability across runs
- Measure cluster agreement between runs
- Identify stable vs variable clusters
- Create consensus clusters

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Run R1 clustering (run 2) | 1.5h | Same prompt, fresh context |
| Run R1 clustering (run 3) | 1.5h | Same prompt, fresh context |
| Implement cluster alignment | 1h | Match clusters across runs |
| Compute stability metrics | 1h | Adjusted Rand Index, NMI |
| Analyze stable patterns | 1h | Which clusters always appear |
| Create consensus clusters | 1h | Merge 3 runs |
| Document stability analysis | 30m | Write results section |

**Stability Metrics:**
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# For each pair of runs, compute:
# - ARI (Adjusted Rand Index): measures cluster agreement
# - NMI (Normalized Mutual Information): measures information preservation

# Target: ARI > 0.6, NMI > 0.7 for "stable" clustering
```

**Files to Create:**
- `/Users/dev_user/semantic/results/r1_run2_clusters.json`
- `/Users/dev_user/semantic/results/r1_run3_clusters.json`
- `/Users/dev_user/semantic/results/r1_stability_analysis.json`
- `/Users/dev_user/semantic/results/r1_consensus_clusters.json`

**Success Criteria:**
- [ ] All 3 runs complete
- [ ] ARI between runs > 0.5 (moderate stability)
- [ ] Core clusters (3-4) appear in all runs
- [ ] Consensus clusters documented

**PIVOT TRIGGER:** If ARI < 0.3, R1 clustering unstable - use simpler heuristics instead.

---

---

## Quick Reference

**Previous Day:** [Day 8](day_08.md) (if exists)
**Next Day:** [Day 10](day_10.md) (if exists)
**Complete Plan:** [Complete 3-Week Plan](../complete_plan.md)

---

## Checklist for Today

- [ ] Review objectives and tasks
- [ ] Set up required files and dependencies
- [ ] Execute all tasks according to timeline
- [ ] Verify success criteria
- [ ] Document any issues or deviations
- [ ] Prepare for next day

---

*Generated from complete_plan.md*
