# Project File Structure

**Last Updated**: 2026-01-23
**Status**: Reorganized and cleaned

---

## Directory Structure

```
/Users/dev_user/semantic/
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── CLAUDE.md                   # Claude Code integration notes
├── FILE_STRUCTURE.md           # This file
├── CURRENT_STATUS.md           # Current status and next steps
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── env.json / env.json.example # Environment configuration
│
├── plans/                      # All development plans and sprints
│   ├── updated_plan.v3.md      # Current plan (automated-first)
│   ├── sprint_00_pilot.md      # Week 0: Pilot testing
│   ├── sprint_01_dataset_and_metrics.md  # Week 1: Dataset + metrics
│   ├── sprint_02_experiments.md          # Weeks 2-3: Experiment runs
│   └── sprint_03_analysis_decision.md    # Weeks 4-6: Analysis + decision
│
├── novelty/                    # Novelty documentation and debates
│   ├── NOVELTY.md              # Core novelty claims and positioning
│   ├── DEBATE_FINAL_CONSENSUS.md        # Final 6-round debate consensus
│   ├── DEBATE_CLARIFICATION_ROUND.md    # Critical clarification of goals
│   ├── PLAN_DEBATE_ROUND_1.md          # Plan v1 debate
│   ├── PLAN_DEBATE_ROUND_2.md          # Plan v2 debate
│   └── PLAN_V2_TO_V2.1_CHANGES.md      # v2 → v2.1 changes summary
│
├── src/                        # Source code
│   ├── semantic_isolation_mlx.py        # MLX implementation (Gemma 3 12B)
│   ├── semantic_isolation_hf.py         # HuggingFace impl (legacy)
│   └── ...                              # Other source files
│
├── scripts/                    # Utility and testing scripts
│   ├── README.md               # Scripts documentation
│   ├── test_mlx_basic.py       # Basic MLX test
│   ├── test_gemma3_simple.py   # Gemma 3 test
│   ├── debug_mlx_context.py    # Context debugging
│   └── archive/                # Old utility scripts (8 files)
│
├── data/                       # Datasets
│   ├── validation_001.json     # Current validation example
│   └── ...                     # Generated datasets will go here
│
├── results/                    # Experiment results
│   ├── validation_001_isolation_test_mlx.json  # MLX test results
│   └── ...                     # Experiment outputs will go here
│
├── evaluation/                 # Evaluation scripts and rubrics
│   └── ...                     # To be created
│
├── instrumentation/            # Profiling and telemetry
│   └── ...                     # To be created
│
├── analysis/                   # Statistical analysis scripts
│   └── ...                     # To be created
│
├── docs/                       # Documentation
│   └── ...                     # Pilot lessons, guides, etc.
│
└── archive/                    # Archived/outdated files
    ├── outdated_plans/         # Old plan versions
    │   ├── complete_plan.md    # Original 3-week plan
    │   ├── updated_plan.v1.md  # First revision (12 weeks)
    │   ├── updated_plan.v2.md  # Second revision (15 weeks)
    │   ├── updated_plan.v2.1.md # Third revision (15 weeks, minor fixes)
    │   ├── DAY_1_STATUS.md through DAY_5_POC_STATUS.md
    │   ├── MLX_MIGRATION_PLAN.md
    │   ├── MODEL_MIGRATION.md
    │   ├── REDESIGN_SUMMARY.md
    │   ├── run_days_5_13.md
    │   ├── DAY_2_MANUAL_REVIEW_CHECKLIST.md
    │   ├── GEMMA_TEST_RESULTS.md
    │   ├── LLAMA_TO_GEMMA_VALIDATION.md
    │   ├── TEST_SUCCESS_SUMMARY.md
    │   ├── KV_CACHE_REVIEW.md
    │   ├── CONTEXT_LENGTH_ANALYSIS.md
    │   ├── HOW_TO_REVIEW.md
    │   └── REVIEW_10_EXAMPLES.txt
    │
    ├── debate_original/        # Original 6-round novelty debate
    │   ├── DEBATE_SETUP.md
    │   ├── debate_plan.md
    │   ├── DEBATE_ROUND_1_SKEPTICS.md
    │   ├── DEBATE_ROUND_2_PROPONENTS.md
    │   ├── DEBATE_ROUND_3_SKEPTICS_WEBSEARCH.md
    │   ├── DEBATE_ROUND_4_PROPONENTS_FINAL.md
    │   ├── DEBATE_ROUND_5_SKEPTICS_CORRECTED.md
    │   └── DEBATE_ROUND_6_PROPONENTS_FINAL_CASE.md
    │
    └── outdated_results/       # Old result files
        └── ...
```

---

## Key Files

### Active Development

**Current Plan**:
- `plans/updated_plan.v3.md` - Automated-first evaluation strategy (13-15 weeks)

**Sprints** (actionable breakdowns):
- `plans/sprint_00_pilot.md` - Week 0: Pilot with n=5
- `plans/sprint_01_dataset_and_metrics.md` - Week 1: Generate n=50, implement 16 metrics
- `plans/sprint_02_experiments.md` - Weeks 2-3: Run all experiments (4×50=200 runs)
- `plans/sprint_03_analysis_decision.md` - Weeks 4-6: Analysis + decision on human eval

**Core Documentation**:
- `novelty/NOVELTY.md` - Novelty claims and positioning vs prior art
- `novelty/DEBATE_FINAL_CONSENSUS.md` - 6-round debate final consensus (ACCEPT)
- `README.md` - Project overview

**Implementation**:
- `src/semantic_isolation_mlx.py` - Main implementation (MLX, Gemma 3 12B)

---

## Evolution Summary

### Plan Versions

**v1** (12 weeks):
- Issues: OOM problem, unrealistic timelines, router agent not novel

**v2** (15 weeks):
- Fixed: Sequential execution, extended timelines, dropped router agent
- Issues: Dependent on independent human raters

**v2.1** (15 weeks):
- Fixed: 7 minor clarifications (memory claim, κ targets, etc.)
- Issues: Still requires independent raters

**v3** (13-15 weeks, CURRENT):
- Fixed: Automated-first strategy, no rater dependency
- Flexible: Decision point at Week 6 (human eval optional)
- Budget: $60-90 + $0-120 (vs $60-390 in v2.1)

### Debate Evolution

**Original Debate** (Rounds 1-4):
- Misunderstood goal as turn-by-turn compression
- Found FlowKV/EpiCache → declared redundant → **REJECT**

**Clarification** (After user correction):
- Actual goal: Virtual multi-agent via KV partitioning
- NOT compression, IS agent isolation
- Re-evaluated novelty

**Final Consensus** (Rounds 5-6):
- Novel contribution confirmed
- 3X memory efficiency vs parallel multi-agent
- Verdict: **ACCEPT with Major Revisions**

### File Organization (2026-01-23)

**Before**:
- 35+ markdown files in root directory
- Plans, debates, status updates all mixed
- Hard to find current vs outdated info

**After**:
- 3 markdown files in root (README, CONTRIBUTING, CLAUDE)
- `plans/` directory: Current plan + sprint files
- `novelty/` directory: Novelty docs + debate consensus
- `archive/` directory: All outdated materials organized

---

## Sprint-Based Workflow

### Current Sprint: Sprint 00 (Week 0)
**Status**: Ready to start
**Goal**: Pilot testing with n=5 examples
**Deliverables**: Working pipeline, validated metrics, pilot lessons

### Next Sprints

1. **Sprint 01** (Week 1): Generate n=50 dataset, implement 16 automated metrics
2. **Sprint 02** (Weeks 2-3): Run 200 experiments, collect comprehensive data
3. **Sprint 03** (Weeks 4-6): Statistical analysis, make decision on human eval
4. **Sprint 04** (Weeks 7-8, optional): MTurk validation with anti-AI safeguards
5. **Sprint 05** (Weeks 7-13): Paper writing (workshop or conference track)

### Decision Tree (Week 6)

```
Week 6 Results
├─ Strong (d>0.8, p<0.01) → MTurk validation → NeurIPS main conference
├─ Moderate (d=0.5-0.8) → Automated-only → Workshops (NeurIPS, EMNLP)
└─ Weak (d<0.5) → Pivot → Revise approach
```

---

## Key Changes in v3

### Removed (No longer dependent on)
- ❌ Independent human raters recruitment (Weeks 1-2)
- ❌ Rater training (4-hour session)
- ❌ Web interface for rating
- ❌ Inter-rater reliability calculations

### Added (New capabilities)
- ✅ Comprehensive automated metrics suite (16 metrics)
- ✅ Week 6 decision point (proceed with human eval?)
- ✅ Optional MTurk validation (subset n=20-30, not full n=50)
- ✅ Anti-AI safeguards for MTurk (6 mechanisms)
- ✅ Flexible publication strategy (conference vs workshop)

### Benefits
- ✅ No rater dependency (can start immediately)
- ✅ Lower initial cost ($60-90 vs $60-390)
- ✅ Shorter timeline if results strong (13 weeks vs 15)
- ✅ More control (decide at Week 6 based on results)
- ✅ Workshop venues as respectable fallback

---

## Working with This Structure

### To start Sprint 00:
```bash
cd /Users/dev_user/semantic
cat plans/sprint_00_pilot.md  # Read sprint plan
# Follow daily breakdown
# Update sprint status as you go
```

### To check current plan:
```bash
cat plans/updated_plan.v3.md  # Full 13-15 week plan
```

### To review novelty claims:
```bash
cat novelty/NOVELTY.md               # Core claims
cat novelty/DEBATE_FINAL_CONSENSUS.md  # Debate outcome
```

### To find archived info:
```bash
ls archive/outdated_plans/     # Old plan versions
ls archive/debate_original/    # Original debate rounds
```

---

## Clean State

**Root directory**: Only essential files (9 files total):
- 5 markdown docs (README, CONTRIBUTING, CLAUDE, FILE_STRUCTURE, CURRENT_STATUS)
- requirements.txt (Python dependencies)
- .gitignore (Git config)
- env.json + env.json.example (Environment config)

**Plans**: Organized in plans/ directory with current plan + sprints
**Novelty**: Organized in novelty/ directory with debate consensus
**Scripts**: Organized in scripts/ directory (3 test scripts + 8 archived utilities)
**Archive**: All outdated materials preserved but out of the way

**Status**: ✅ Clean, organized, ready for Sprint 00

---

**Date**: 2026-01-23
**Next Action**: Review plans/sprint_00_pilot.md and begin Week 0 pilot testing
