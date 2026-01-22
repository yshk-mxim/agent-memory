# Day 2 (Tuesday): Instruction Conflict Dataset Design

**Week 1 - Day 2**

---

**Objectives:**
- Design dataset schema for instruction conflicts
- Create generation prompts for Claude
- Generate first batch of 30 examples
- Validate quality through manual review

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Design conflict taxonomy | 1.5h | 5-7 conflict types |
| Create dataset schema | 1h | JSON structure for conversations |
| Write Claude generation prompt | 1.5h | Prompt engineering for diverse conflicts |
| Generate first batch (30 examples) | 1h | Test prompt, validate output quality |
| Manual review of batch | 1h | Check 10 examples for genuine conflicts |
| Iterate on prompt | 1h | Fix issues found in review |

**Conflict Taxonomy:**
1. **Tone conflicts:** formal vs casual, professional vs friendly
2. **Detail conflicts:** brief vs detailed, concise vs comprehensive
3. **Style conflicts:** technical vs layperson, academic vs conversational
4. **Content conflicts:** cite sources vs no citations, examples vs no examples
5. **Format conflicts:** structured vs freeform, bullet points vs paragraphs

**Dataset Schema:**
```json
{
  "id": "conflict_001",
  "conflict_type": "tone_formal_vs_casual",
  "domain": "business_email",
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "instruction": "Always respond in formal, professional language",
      "content": "Help me draft emails to clients"
    },
    {
      "turn_id": 2,
      "role": "user",
      "instruction": "Be casual and friendly, like texting a friend",
      "content": "Now write me an email about the project delay"
    },
    {
      "turn_id": 3,
      "role": "user",
      "query": "Write the email combining both styles appropriately",
      "expected_conflict": "Cannot be simultaneously formal and casual"
    }
  ],
  "ground_truth_clusters": ["formal_constrained", "casual_creative"]
}
```

**Files to Create:**
- `/Users/dev_user/semantic/src/dataset_generator.py`
- `/Users/dev_user/semantic/data/conflict_schema.json`
- `/Users/dev_user/semantic/data/batch_001.json` (30 examples)

**Success Criteria:**
- [ ] Dataset schema defined with all required fields
- [ ] First batch of 30 examples generated
- [ ] >70% of manually reviewed examples are genuine conflicts
- [ ] Coverage across all 5 conflict types

---

---

## Quick Reference

**Previous Day:** [Day 1](day_01.md) (if exists)
**Next Day:** [Day 3](day_03.md) (if exists)
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
