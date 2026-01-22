# Day 5 (Friday): Experiment 1 - Compression Degrades Instruction Following

**Week 1 - Day 5**

---

**Objectives:**
- Demonstrate that context compression hurts instruction-following
- Compare full context vs compressed context performance
- Generate Figure 1 and Table 1

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Implement context compression | 1h | Random truncation, simulating KV eviction |
| Run full context baseline (50 examples) | 2h | ~2.5 min/example with Llama |
| Run 50% compressed (50 examples) | 2h | Same examples, compressed context |
| Evaluate all outputs | 1h | Both evaluators |
| Analyze results by conflict type | 1h | Breakdown of degradation |
| Generate Figure 1 | 30m | Degradation bar chart |
| Generate Table 1 | 30m | Mean scores with std dev |

**Compression Method:**
```python
def compress_context_random(full_context: str, compression_ratio: float = 0.5) -> str:
    """Simulate random token eviction (like H2O)"""
    sentences = full_context.split('. ')
    keep_count = max(1, int(len(sentences) * compression_ratio))

    # Keep first and last sentences (like StreamingLLM)
    if keep_count >= 2:
        kept = [sentences[0]]
        middle = sentences[1:-1]
        random.shuffle(middle)
        kept.extend(middle[:keep_count-2])
        kept.append(sentences[-1])
    else:
        kept = sentences[:keep_count]

    return '. '.join(kept)
```

**Files to Create:**
- `/Users/dev_user/semantic/experiments/exp1_compression.py`
- `/Users/dev_user/semantic/src/compression.py`
- `/Users/dev_user/semantic/results/exp1_results.json`
- `/Users/dev_user/semantic/results/figures/fig1_degradation.png`
- `/Users/dev_user/semantic/results/tables/table1_compression.csv`

**Success Criteria:**
- [ ] Full context mean score >0.6
- [ ] Compressed context mean score <0.5
- [ ] Degradation statistically significant (p<0.05)
- [ ] Figure and table generated

**PIVOT TRIGGER:** If degradation <5%, pivot to "Instruction Conflict Analysis" paper focusing on characterization rather than solution.

---

---

## Quick Reference

**Previous Day:** [Day 4](day_04.md) (if exists)
**Next Day:** [Day 6](day_06.md) (if exists)
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
