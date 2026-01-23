# Plan Updates: v2 → v2.1

**Date**: 2026-01-23
**Source**: Round 2 debate consensus (PLAN_DEBATE_ROUND_2.md)
**Outcome**: All 7 minor revisions incorporated
**Time Required**: ~2 hours of changes

---

## Summary

Plan v2.1 incorporates 7 minor clarifications from Round 2 debate consensus. All critical issues from Round 1 remain resolved. Plan is now ready for execution pending final approval.

---

## The 7 Revisions

### 1. Memory Efficiency Claim Reframed ✅

**Issue**: "3X efficiency" was vs parallel (infeasible), not sequential (feasible)

**Changes**:
- **Executive Summary**: Updated goal statement to specify "3X vs parallel (infeasible), 2-3X latency vs sequential"
- **Week 7 Memory Calculation**: Expanded conclusion to clarify:
  - 3X memory efficiency vs parallel true multi-agent (standard architecture but infeasible on 24GB RAM)
  - Comparable memory vs sequential (~1X) but 2-3X latency advantage due to cache reuse

**Impact**: More honest framing, emphasizes latency as secondary benefit

---

### 2. Inter-Rater Reliability Target Standardized ✅

**Issue**: Target was inconsistent (κ>0.5, κ>0.6, κ>0.7 in different places)

**Changes**:
- **Week 2 Success Criteria**: Updated from "κ>0.5" to "κ>0.7 minimum (>0.8 ideal)"
- **Week 5 Success Criteria**: Updated from "κ>0.6" to "κ>0.7 minimum (>0.8 ideal)"
- **Success Criteria Section**: Updated from "κ>0.6" to "κ>0.7 minimum (>0.8 ideal)"

**Impact**: Consistent with Round 1 consensus, higher bar for inter-rater reliability

---

### 3. Baseline Condition Scope Clarified ✅

**Issue**: Confusion about which conditions use n=50 vs n=10

**Changes**:
- **Phase 1 Header**: Added note clarifying:
  - **Primary conditions** (n=50): Sequential, prompted, turn-based, semantic, random clustering
  - **Secondary baselines** (n=10): No-coordinator, true multi-agent
  - Secondary baselines reported descriptively (no formal hypothesis testing)

**Impact**: Clear distinction between full statistical analysis (n=50) and exploratory comparison (n=10)

---

### 4. Rubric Validation Added to Week 5 ✅

**Issue**: No construct validity or convergent validity analysis planned

**Changes**:
- **Week 5 Thursday**: Added 1-hour task:
  - Compute correlation between human and automated metrics (target r>0.6)
  - Compute inter-dimension correlation (target <0.7 for discriminant validity)
- Reduced statistical testing from 4h to 3h to accommodate

**Impact**: Validates that rubric measures what it claims to measure

---

### 5. Communication Protocol Specified for Sequential Agents ✅

**Issue**: How sequential agents pass information wasn't specified

**Changes**:
- **Week 7 Monday**: Expanded "Design sequential 3-agent protocol" task details:
  - "**Communication**: Each agent receives all previous outputs via concatenation in context (Agent 2 gets Agent 1 output, Agent 3 gets Agent 1+2 outputs), mirroring semantic coordinator structure for fair comparison."

**Impact**: Ensures fair comparison between semantic and sequential true multi-agent

---

### 6. Related Work Expanded (Multi-Persona, MoE) ✅

**Issue**: Missing multi-persona LLM systems and MoE comparison

**Changes**:
- **Week 11 Outline Task**: Updated from "3 subsections" to "5 subsections"
- **Related Work Structure**: Added two new sections:
  - **Section 2.2: Multi-Persona LLM Systems** (NEW)
    - Persona-augmented LLMs (instructional methods)
    - Distinction: Soft isolation (prompts) vs hard isolation (KV cache)
    - Our prompted condition tests their approach
  - **Section 2.5: Mixture-of-Experts (MoE)** (NEW)
    - MoE models (GPT-4, Mixtral): Training-time specialization
    - Distinction: MoE requires training, our approach is inference-time
    - Complementary: Could combine both techniques

**Impact**: Addresses potential reviewer questions, strengthens positioning

---

### 7. Evaluation Window Flexibility Noted ✅

**Issue**: Rater workload heavy (13 hours), timeline might be tight

**Changes**:
- **Week 4 Rater Workload Section**: Expanded to note:
  - Total workload updated: ~230 outputs (including secondary baselines)
  - 11-13 hours total per rater
  - **Evaluation window**: 1-2 weeks (flexible)
    - **Option 1**: 1 week intensive (~2.5 hours/day)
    - **Option 2**: 2 weeks relaxed (~1.5 hours/day)
  - Fallback: Reduce to n=30 per rater if needed (7.5 hours)

**Impact**: Provides flexibility if rater recruitment challenging, reduces risk

---

## Overall Impact

**Before (v2)**:
- All Round 1 critical issues resolved
- Some ambiguities remained (memory claim, κ target, baseline scope)
- Missing Related Work coverage (multi-persona, MoE)

**After (v2.1)**:
- All Round 1 critical issues resolved ✅
- All Round 2 minor issues addressed ✅
- Clear and unambiguous plan
- Comprehensive Related Work
- Flexible timelines (evaluation window, fallback options)

---

## Verdict from Round 2 Debate

**Consensus**: **CONDITIONAL CONSENSUS ACHIEVED** → Plan v2.1 ready for execution

**Skeptic A (Systems Expert)**: ⚠️ → ✅ (Memory claim reframed, communication protocol specified)
**Skeptic B (Prior Art Hunter)**: ⚠️ → ✅ (Related Work expanded)
**Skeptic C (Methodology Expert)**: ⚠️ → ✅ (Rubric validation added, κ standardized, baseline scope clarified)

**Proponent A (Novelty Defender)**: ✅ Ready
**Proponent B (Practical Applications)**: ✅ Ready
**Proponent C (Architecture Expert)**: ✅ Ready

---

## Changes by Section

### Executive Summary
- Reframed memory efficiency claim (Revision 1)

### Phase 1 Header
- Added baseline scope clarification (Revision 3)

### Week 2 Success Criteria
- Updated κ target to >0.7 (Revision 2)

### Week 4 Rater Workload
- Added flexible evaluation window (Revision 7)

### Week 5 Thursday
- Added rubric validation task (Revision 4)
- Updated κ target to >0.7 (Revision 2)

### Week 7 Monday
- Specified communication protocol (Revision 5)

### Week 7 Memory Calculation
- Expanded memory efficiency framing (Revision 1)

### Week 11 Related Work Structure
- Added multi-persona LLM subsection (Revision 6)
- Added MoE subsection (Revision 6)
- Updated to 5 subsections from 3 (Revision 6)

### Success Criteria Section
- Updated κ target to >0.7 (Revision 2)

---

## Remaining Concerns (Non-Blocking)

All remaining concerns from Round 2 are **acknowledged risks**, not blocking issues:

1. **Statistical Power** (β=0.70 with n=50, FDR)
   - Mitigation: Report effect sizes prominently
   - Impact: May weaken results if effects are small

2. **Deployment Study Feasibility** (Week 10 optional)
   - Mitigation: Skip if behind schedule
   - Impact: Nice-to-have, not critical

3. **NeurIPS Positioning** (may prefer more theory)
   - Mitigation: Emphasize algorithmic novelty, prepare EMNLP backup
   - Impact: Venue fit uncertain

4. **Instrumentation Overhead** (may be 5-10%, not <5%)
   - Mitigation: Test in Week 0 pilot, optimize if needed
   - Impact: Minor, can measure without instrumentation

---

## Next Steps

1. ✅ **Plan v2.1 Complete** (this document)
2. ➡️ **Final Approval** (if needed, conduct Round 3 review)
3. ➡️ **Begin Week 0** (pilot testing with n=5)
4. Iterate to v2.2 if pilot reveals issues
5. Proceed with Phase 1 upon successful pilot

---

## File Locations

- **v2 (original)**: `/Users/dev_user/semantic/updated_plan.v2.md` (preserved)
- **v2.1 (current)**: `/Users/dev_user/semantic/updated_plan.v2.1.md` (ready for execution)
- **Round 2 debate**: `/Users/dev_user/semantic/PLAN_DEBATE_ROUND_2.md`
- **This summary**: `/Users/dev_user/semantic/PLAN_V2_TO_V2.1_CHANGES.md`

---

**Confidence Level**: **HIGH** - Plan is ready for execution
**Timeline**: 15 weeks starting Jan 23 → ~April 30 (2 weeks before NeurIPS deadline)
**Target**: NeurIPS 2026 (May 15 deadline) - **FEASIBLE**

---

**Date**: 2026-01-23
**Status**: Plan v2.1 complete and ready for execution
