# Day 4 (Thursday): Implement Evaluation Framework

**Week 1 - Day 4**

---

**Objectives:**
- Build rule-based instruction-following checker
- Implement LLM-as-judge with Claude Haiku
- Validate evaluators on known examples
- Measure inter-evaluator agreement

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Design evaluation rubric | 1h | Criteria for each instruction type |
| Implement rule-based checker | 2h | Pattern matching for tone, format, etc. |
| Implement Claude Haiku judge | 1.5h | Prompt template for scoring |
| Create golden test cases | 1h | 10 examples with known scores |
| Validate rule-based checker | 1h | Test against golden set |
| Validate LLM judge | 1h | Test against golden set, check agreement |
| Measure evaluator agreement | 30m | Correlation between methods |

**Rule-Based Patterns:**
```python
FORMAL_INDICATORS = ['respectfully', 'pursuant', 'accordingly', 'kindly']
CASUAL_INDICATORS = ["don't", "can't", "hey", "yeah", "gonna", "btw"]
BRIEF_THRESHOLD = 100  # words
DETAILED_THRESHOLD = 200  # words
```

**LLM Judge Prompt:**
```python
prompt = f"""Evaluate how well this text follows each instruction.

Instructions:
1. {instruction_1}
2. {instruction_2}

Text to evaluate:
{text}

For each instruction, rate 0-10 how well it was followed.
Output JSON: {{"instruction_1": score, "instruction_2": score, "overall": avg}}"""
```

**Files to Create:**
- `/Users/dev_user/semantic/src/evaluator.py` - Main evaluation module
- `/Users/dev_user/semantic/src/rule_checker.py` - Rule-based checks
- `/Users/dev_user/semantic/src/llm_judge.py` - Claude Haiku judge
- `/Users/dev_user/semantic/data/golden_eval_set.json`

**Success Criteria:**
- [ ] Rule-based checker handles all 5 conflict types
- [ ] LLM judge returns scores 0-1 with reasoning
- [ ] >80% agreement between rule-based and LLM judge on golden set
- [ ] Both evaluators run in <2s per example

---

---

## Quick Reference

**Previous Day:** [Day 3](day_03.md) (if exists)
**Next Day:** [Day 5](day_05.md) (if exists)
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
