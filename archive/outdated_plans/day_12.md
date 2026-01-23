# Day 12 (Friday): Experiment 3 - Semantic vs Turn-Based Isolation

**Week 2 - Day 5**

---

**Objectives:**
- Run comparative experiment
- Measure instruction-following under both isolation methods
- Generate Figure 3 and Table 3

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Run turn-based baseline (20 test examples) | 2h | Full pipeline |
| Run semantic isolation (20 test examples) | 2h | Full pipeline |
| Evaluate all outputs | 1h | Both evaluators |
| Statistical comparison | 1h | Paired t-test |
| Analyze by conflict type | 1h | Where does semantic help most? |
| Generate Figure 3 | 30m | Comparison bar chart |
| Generate Table 3 | 30m | Performance comparison |

**Files to Create:**
- `/Users/dev_user/semantic/experiments/exp3_isolation_comparison.py`
- `/Users/dev_user/semantic/results/exp3_results.json`
- `/Users/dev_user/semantic/results/figures/fig3_comparison.png`
- `/Users/dev_user/semantic/results/tables/table3_performance.csv`

**Success Criteria:**
- [ ] Semantic isolation outperforms turn-based by >5%
- [ ] Improvement statistically significant (p<0.05)
- [ ] Clear pattern of which conflict types benefit most
- [ ] All figures/tables generated

**Expected Results:**
- Turn-based: ~55% instruction following
- Semantic: ~65% instruction following (target)
- Improvement: 10 percentage points

**PIVOT TRIGGER:** If semantic <= turn-based, analyze WHY - pivot to "When Does Semantic Isolation Matter?" paper.

---

---

## Quick Reference

**Previous Day:** [Day 11](day_11.md) (if exists)
**Next Day:** [Day 13](day_13.md) (if exists)
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
