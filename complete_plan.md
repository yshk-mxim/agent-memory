# RDIC Research Sprint: Complete 3-Week Execution Plan

## Executive Summary

This plan provides a detailed 21-day execution strategy for the "Reasoning-Discovered Instruction Contexts (RDIC)" research project. The goal is to publish an Arxiv paper demonstrating that semantic instruction clustering (discovered by reasoning models like DeepSeek R1) improves instruction-following in multi-turn LLM conversations compared to turn-based isolation approaches like FlowKV.

**Hardware:** Mac with 24GB RAM (primary development)
**Model:** `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (Q4_K_M, ~4.9GB)
**APIs:** Claude (Anthropic), DeepSeek R1 (api.deepseek.com)
**Budget:** $15-25 total API costs

---

## WEEK 1: Data Generation and Problem Validation

### Day 1 (Monday): Environment Setup and API Validation

**Objectives:**
- Set up complete development environment
- Verify all API connections work correctly
- Create project directory structure
- Download and test Llama model

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Install llama-cpp-python with Metal support | 1.5h | `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python` |
| Download Llama-3.1-8B Q4 GGUF model | 1h | Download from HF (~4.9GB) |
| Test Claude API connection | 30m | Simple test call with Haiku model |
| Test DeepSeek R1 API connection | 45m | Use OpenAI-compatible client, verify reasoning traces |
| Test local Llama inference | 1h | Generate simple response, verify Metal acceleration |
| Create utility module | 1.5h | `src/utils.py` with API client wrappers |
| Create requirements.txt | 30m | All dependencies documented |

**Files to Create:**
- `/Users/dev_user/semantic/src/__init__.py`
- `/Users/dev_user/semantic/src/config.py` - Load env.json
- `/Users/dev_user/semantic/src/utils.py` - API client wrappers (Claude, DeepSeek, Llama)
- `/Users/dev_user/semantic/requirements.txt`

**Success Criteria:**
- [ ] Can load Llama-3.1-8B Q4 GGUF and generate response in <30 seconds
- [ ] Claude API returns valid response
- [ ] DeepSeek R1 API returns response with reasoning_content field
- [ ] All APIs callable from unified interface in utils.py

**DeepSeek R1 API Pattern:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="<from env.json>",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": prompt}]
)
# Access: response.choices[0].message.content
```

**Llama GGUF Pattern:**
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1  # Use Metal on Mac
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)
```

---

### Day 2 (Tuesday): Instruction Conflict Dataset Design

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

### Day 3 (Wednesday): Complete Dataset Generation

**Objectives:**
- Generate remaining 70 examples to reach 100 total
- Implement validation pipeline
- Create train/test split
- Document dataset statistics

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Generate batch 2 (35 examples) | 1.5h | Continue with refined prompt |
| Generate batch 3 (35 examples) | 1.5h | Ensure diversity across types |
| Implement automated validation | 1.5h | Check structure, required fields |
| Manual validation of 20 random samples | 1h | Detailed conflict verification |
| Create train/test split (80/20) | 30m | 80 train, 20 test |
| Compute dataset statistics | 30m | Type distribution, domain coverage |
| Document dataset | 30m | README with statistics |

**Files to Create:**
- `/Users/dev_user/semantic/src/validator.py`
- `/Users/dev_user/semantic/data/conflict_dataset.json` (100 examples)
- `/Users/dev_user/semantic/data/train.json` (80 examples)
- `/Users/dev_user/semantic/data/test.json` (20 examples)
- `/Users/dev_user/semantic/data/README.md`

**Success Criteria:**
- [ ] 100 valid conversation examples generated
- [ ] All examples pass structural validation
- [ ] >75% manual validation accuracy on random sample
- [ ] Balanced distribution across conflict types (15-25 each)

**Decision Point:** If validation <70%, regenerate problematic batches with stricter prompt.

---

### Day 4 (Thursday): Implement Evaluation Framework

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

### Day 5 (Friday): Experiment 1 - Compression Degrades Instruction Following

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

### Day 6 (Saturday): Buffer Day / Catch-up

**Objectives:**
- Address issues from Days 1-5
- Re-run failed experiments
- Document early findings

**Tasks:**
- Review all results so far (1h)
- Fix any bugs discovered (2-3h variable)
- Re-run problematic experiments if needed (2h)
- Document methodology (1h)
- Prepare for Week 2 (1h)

**Success Criteria:**
- [ ] All Day 1-5 deliverables complete
- [ ] No blocking bugs remaining
- [ ] Dataset and Experiment 1 results validated

---

### Day 7 (Sunday): Rest / Light Documentation

**Objectives:**
- Rest day with optional light work
- Organize notes and results
- Prepare for R1 experiments

**Tasks:**
- Organize results folder (30m)
- Write experiment log (30m)
- Read FlowKV paper details (1h optional)
- Prepare R1 prompts (1h)

---

## WEEK 2: R1 Clustering and Proof-of-Concept

### Day 8 (Monday): DeepSeek R1 Clustering - Run 1

**Objectives:**
- Design R1 clustering prompt
- Run first clustering analysis
- Analyze discovered clusters
- Document reasoning traces

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Design R1 clustering prompt | 2h | Based on conversation dataset |
| Test prompt on 10 examples | 1h | Verify output format |
| Run R1 on full dataset (100 examples) | 1.5h | Single API call or batched |
| Parse R1 output | 1h | Extract clusters, conflict matrix |
| Analyze cluster quality | 1h | Manual review of taxonomy |
| Document R1 reasoning | 30m | Save reasoning traces |

**R1 Clustering Prompt:**
```
Analyze these 100 multi-turn conversations where instructions may conflict.

For each conversation, examine the instructions across turns and:
1. Identify which instructions semantically conflict
2. Determine what "cluster" each instruction belongs to
3. Output a conflict matrix showing which clusters conflict

Your task is to discover the minimal set of instruction clusters needed
such that instructions within the same cluster are compatible, but
instructions across conflicting clusters interfere with each other.

Output format:
{
  "instruction_clusters": [
    {
      "id": "cluster_name",
      "characteristics": "description",
      "conflicts_with": ["other_cluster_names"],
      "example_instructions": ["list", "of", "examples"]
    }
  ],
  "conversation_assignments": [
    {"conv_id": "001", "turn1_cluster": "X", "turn2_cluster": "Y"}
  ],
  "conflict_matrix": [[0, 1, ...], ...]
}

Conversations:
[INSERT DATASET HERE]
```

**Files to Create:**
- `/Users/dev_user/semantic/experiments/exp2_r1_clustering.py`
- `/Users/dev_user/semantic/results/r1_run1_clusters.json`
- `/Users/dev_user/semantic/results/r1_run1_reasoning.txt`

**Success Criteria:**
- [ ] R1 produces valid JSON output
- [ ] 4-8 clusters discovered (reasonable range)
- [ ] Clusters align with conflict taxonomy (>60% overlap)
- [ ] Reasoning trace captured for paper

---

### Day 9 (Tuesday): R1 Clustering - Runs 2 and 3 (Stability)

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

### Day 10 (Wednesday): Cluster Validation and Ground Truth Comparison

**Objectives:**
- Compare R1 clusters to ground truth conflict types
- Measure cluster-conflict alignment
- Generate Figure 2 and Table 2

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Create ground truth mapping | 1h | Map dataset conflict_type to clusters |
| Compute cluster-GT alignment | 1h | Precision, recall per cluster |
| Implement confusion matrix | 1h | R1 clusters vs GT types |
| Analyze misalignments | 1h | Why do some disagree? |
| Generate Figure 2 | 1h | Cluster-conflict heatmap |
| Generate Table 2 | 30m | Cluster taxonomy table |
| Write clustering results | 1.5h | Section 3 draft |

**Files to Create:**
- `/Users/dev_user/semantic/experiments/exp2_validation.py`
- `/Users/dev_user/semantic/results/cluster_gt_alignment.json`
- `/Users/dev_user/semantic/results/figures/fig2_cluster_heatmap.png`
- `/Users/dev_user/semantic/results/tables/table2_taxonomy.csv`

**Success Criteria:**
- [ ] >70% alignment between R1 clusters and GT conflict types
- [ ] Confusion matrix shows clear diagonal pattern
- [ ] Figure 2 is publication quality
- [ ] R1 discovers at least 1 cluster not in GT (novel insight)

---

### Day 11 (Thursday): Implement Semantic Isolation Framework

**Objectives:**
- Build proof-of-concept semantic isolation system
- Implement embedding-based router
- Create turn-based baseline for comparison

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Design isolation architecture | 1h | Based on RDIC class design |
| Implement embedding router | 2h | sentence-transformers + cosine similarity |
| Implement turn-based baseline | 1h | FlowKV-style isolation |
| Implement semantic isolation | 2h | Route by R1 clusters |
| Unit tests for router | 1h | Verify routing accuracy |
| Integration test | 1h | End-to-end with Llama |

**Architecture:**
```python
class SemanticIsolation:
    def __init__(self, clusters: List[Dict]):
        self.clusters = clusters
        self.router = EmbeddingRouter(clusters)
        self.contexts = {c['id']: [] for c in clusters}

    def add_turn(self, instruction: str, content: str):
        cluster_id = self.router.route(instruction)
        self.contexts[cluster_id].append({
            'instruction': instruction,
            'content': content
        })

    def get_context(self, query: str) -> str:
        relevant_clusters = self.router.route(query, top_k=2)
        # Merge contexts from relevant clusters
        return self._merge_contexts(relevant_clusters)
```

**Files to Create:**
- `/Users/dev_user/semantic/src/isolation.py` - Main isolation classes
- `/Users/dev_user/semantic/src/router.py` - Embedding router
- `/Users/dev_user/semantic/src/baselines.py` - Turn-based baseline
- `/Users/dev_user/semantic/tests/test_isolation.py`

**Success Criteria:**
- [ ] Router correctly routes >80% of instructions to expected cluster
- [ ] Turn-based baseline works end-to-end
- [ ] Semantic isolation works end-to-end
- [ ] Both produce coherent responses

---

### Day 12 (Friday): Experiment 3 - Semantic vs Turn-Based Isolation

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

### Day 13 (Saturday): Buffer Day / Extended Analysis

**Objectives:**
- Address Week 2 issues
- Run additional analysis if experiments succeed
- Begin results synthesis

**Tasks:**
- Review all Week 2 results (1h)
- Fix bugs, re-run if needed (2h variable)
- Additional analysis: ablations (2h if time permits)
- Compile all results (1h)
- Begin paper outline (1h)

**Ablation Studies (if time):**
1. Number of clusters (3, 5, 7)
2. Compression ratios (25%, 50%, 75%)
3. Router threshold sensitivity

**Success Criteria:**
- [ ] All Week 2 deliverables complete
- [ ] Results consistent and reproducible
- [ ] Paper outline drafted

---

### Day 14 (Sunday): Rest / Light Paper Work

**Objectives:**
- Rest day
- Optional: organize paper materials

**Tasks:**
- Rest (recovery)
- Optional: Related work notes (1h)
- Optional: Figure polishing (1h)

---

## WEEK 3: Paper Writing and Finalization

### Day 15 (Monday): Paper Introduction and Related Work

**Objectives:**
- Write complete Introduction section
- Write Related Work section
- Establish paper narrative

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Write Introduction (1.5 pages) | 3h | Problem, gap, contribution |
| Write Related Work (1 page) | 2h | FlowKV, Multi-IF, LLM caching |
| Create contribution list | 30m | 3-4 clear contributions |
| Review and revise | 1.5h | Iterate on framing |

**Paper Structure:**
```
Title: "Reasoning-Discovered Instruction Contexts:
        Semantic Isolation for Multi-Turn LLM Conversations"

Abstract (150 words)
1. Introduction (1.5 pages)
2. Related Work (1 page)
3. Instruction Conflict Dataset (1 page)
4. Semantic Clustering Discovery (1.5 pages)
5. Proof-of-Concept Isolation (1.5 pages)
6. Discussion and Limitations (0.5 pages)
7. Conclusion (0.5 pages)
```

**Files to Create:**
- `/Users/dev_user/semantic/paper/sections/01_introduction.md`
- `/Users/dev_user/semantic/paper/sections/02_related_work.md`
- `/Users/dev_user/semantic/paper/references.bib`

**Success Criteria:**
- [ ] Introduction clearly states problem and contribution
- [ ] Related work covers key prior work
- [ ] Narrative is compelling and novel

---

### Day 16 (Tuesday): Methods Section

**Objectives:**
- Write Dataset section
- Write Clustering methodology
- Write Evaluation methodology

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Write Section 3: Dataset (1 page) | 2h | Generation, validation, statistics |
| Write Section 4: R1 Clustering (1 page) | 2h | Method, stability, results |
| Write evaluation methodology | 1h | Rule-based + LLM judge |
| Add figures/tables to sections | 1h | Integrate visualizations |
| Review technical accuracy | 1h | Check all numbers |

**Files to Create:**
- `/Users/dev_user/semantic/paper/sections/03_dataset.md`
- `/Users/dev_user/semantic/paper/sections/04_clustering.md`

**Success Criteria:**
- [ ] Methods are reproducible from description
- [ ] All hyperparameters documented
- [ ] Figures properly referenced

---

### Day 17 (Wednesday): Results and Analysis

**Objectives:**
- Write Experiments section
- Write Results analysis
- Write Discussion and Limitations

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Write Section 5: Experiments (1.5 pages) | 3h | All three experiments |
| Write Discussion (0.5 pages) | 1.5h | Implications, future work |
| Write Limitations (0.5 pages) | 1h | Honest assessment |
| Write Conclusion (0.5 pages) | 1h | Summary of contributions |
| Review all results claims | 1h | Verify accuracy |

**Limitations to Address:**
1. Synthetic dataset (not real user conversations)
2. Single model (Llama-3.1-8B only)
3. Proof-of-concept (not production system)
4. Limited scale (100 examples)
5. Simulated compression (not actual KV cache manipulation)

**Files to Create:**
- `/Users/dev_user/semantic/paper/sections/05_experiments.md`
- `/Users/dev_user/semantic/paper/sections/06_discussion.md`
- `/Users/dev_user/semantic/paper/sections/07_conclusion.md`

**Success Criteria:**
- [ ] All experimental results clearly presented
- [ ] Discussion addresses "so what?"
- [ ] Limitations are honest and complete

---

### Day 18 (Thursday): Abstract, Full Draft Assembly, and Code Cleanup

**Objectives:**
- Write Abstract
- Assemble complete paper draft
- Clean up code for release
- Prepare GitHub repository

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Write Abstract (150-200 words) | 1h | Summary of whole paper |
| Assemble full draft | 1h | Combine all sections |
| First full read-through | 1.5h | Note issues |
| Code cleanup | 2h | Remove debug code, add comments |
| Create GitHub repo structure | 1h | README, license, structure |
| Test reproducibility | 1h | Run from fresh checkout |

**GitHub Structure:**
```
semantic-cache-isolation/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── REPRODUCE.md
├── data/
│   ├── conflict_dataset.json
│   ├── train.json
│   └── test.json
├── src/
│   ├── utils.py
│   ├── evaluator.py
│   ├── isolation.py
│   └── router.py
├── experiments/
│   ├── exp1_compression.py
│   ├── exp2_clustering.py
│   └── exp3_comparison.py
├── results/
│   ├── figures/
│   └── tables/
└── paper/
    └── rdic_paper.pdf
```

**Files to Create:**
- `/Users/dev_user/semantic/README.md` (GitHub)
- `/Users/dev_user/semantic/LICENSE`
- `/Users/dev_user/semantic/REPRODUCE.md`
- `/Users/dev_user/semantic/paper/rdic_paper.md` (complete)

**Success Criteria:**
- [ ] Complete draft exists
- [ ] Abstract is compelling
- [ ] Code runs from fresh checkout
- [ ] README explains how to reproduce

---

### Day 19 (Friday): Paper Revision and Figures

**Objectives:**
- Revise full paper
- Polish all figures
- Fix formatting issues

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Full paper revision | 3h | Content, clarity, flow |
| Figure polishing | 2h | Publication-quality visuals |
| Table formatting | 1h | Consistent style |
| Reference check | 1h | All citations correct |
| Grammar/spell check | 1h | Final proofread |

**Figure Requirements:**
- Fig 1: Instruction-following degradation under compression
- Fig 2: R1 cluster discovery (taxonomy visualization)
- Fig 3: Semantic vs turn-based comparison
- All figures at 300 DPI minimum

**Success Criteria:**
- [ ] Paper reads smoothly end-to-end
- [ ] All figures at 300 DPI minimum
- [ ] No orphan references or broken links
- [ ] Paper is <10 pages

---

### Day 20 (Saturday): Final Buffer and Arxiv Preparation

**Objectives:**
- Final paper polish
- Prepare Arxiv submission
- Final code verification

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Final read-through | 2h | Fresh eyes review |
| Address any remaining issues | 2h | Variable |
| Create Arxiv source package | 1h | .md or .tex, figures |
| Test Arxiv compilation | 1h | Verify builds correctly |
| Final GitHub push | 30m | All code committed |
| Write social media posts | 30m | Draft Twitter thread |

**Arxiv Checklist:**
- [ ] Paper compiles without errors
- [ ] All figures embedded correctly
- [ ] Author information correct
- [ ] License selected (CC BY 4.0 recommended)
- [ ] Categories: cs.CL (primary), cs.AI (secondary)

**Success Criteria:**
- [ ] Paper ready for submission
- [ ] Arxiv package tested
- [ ] Code repository complete

---

### Day 21 (Sunday): Submission and Dissemination

**Objectives:**
- Submit to Arxiv
- Push code to GitHub
- Begin dissemination

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Arxiv submission | 1h | Upload, verify, submit |
| GitHub public release | 30m | Make repo public |
| Write Twitter/X thread | 1h | Key findings, figures |
| Post to Reddit (r/MachineLearning) | 30m | Discussion thread |
| LinkedIn post | 30m | Professional announcement |
| Email colleagues | 30m | Share with network |
| Celebrate | - | You did it! |

**Success Criteria:**
- [ ] Paper submitted to Arxiv
- [ ] Code publicly available
- [ ] Social media posts live
- [ ] All deliverables complete

---

## Risk Mitigation and Pivot Strategies

### Scenario 1: Compression Shows No Degradation

**Detection:** Day 5 - Full vs compressed shows <5% difference

**Pivot:**
- Focus paper on "characterizing instruction conflicts"
- Contribution: taxonomy + dataset, not system
- Title: "Instruction Conflicts in Multi-Turn Conversations: An Analysis"
- Still publishable as empirical study

### Scenario 2: R1 Clustering is Unstable

**Detection:** Day 9 - ARI < 0.3 across runs

**Pivot:**
- Use simple keyword-based clustering instead
- Claim: "Even simple semantic clustering helps"
- Focus on the insight, not the method

### Scenario 3: Semantic Isolation Shows No Improvement

**Detection:** Day 12 - Semantic <= turn-based

**Pivot:**
- Analyze WHY turn-based suffices
- Focus: "When does semantic isolation matter?"
- Negative result paper: still valuable
- Title: "Turn-Based Isolation is Sufficient for Instruction Following"

### Scenario 4: Hardware/API Failures

**Mitigation:**
- Save results incrementally (every experiment)
- Keep backup of all data locally
- DeepSeek has rate limits - batch carefully
- If Mac fails: most work is API calls, use any machine

### Scenario 5: Running Behind Schedule

**Mitigation:**
- Day 6, 13, 20 are buffer days
- Cut ablations first (nice-to-have)
- Reduce dataset to 50 examples if needed
- Simplify paper scope (focus on 2 experiments not 3)

---

## Budget Estimate

| Item | Cost | Notes |
|------|------|-------|
| Claude API (dataset generation) | $5-10 | 5 batches @ ~$1-2 each |
| Claude Haiku (evaluation) | $5-10 | ~500 eval calls |
| DeepSeek R1 (clustering) | $3-5 | 3 runs @ ~$1 each |
| **Total** | **$15-25** | Conservative estimate |

---

## Complete Deliverables Checklist

### Week 1
- [ ] `src/` - All utility modules (utils.py, evaluator.py, compression.py)
- [ ] `data/conflict_dataset.json` - 100 examples
- [ ] `results/exp1_results.json`
- [ ] Figure 1, Table 1

### Week 2
- [ ] `results/r1_*_clusters.json` - 3 runs
- [ ] `results/r1_consensus_clusters.json`
- [ ] `src/isolation.py`, `src/router.py`
- [ ] `results/exp3_results.json`
- [ ] Figures 2-3, Tables 2-3

### Week 3
- [ ] `paper/rdic_paper.pdf`
- [ ] `README.md`, `LICENSE`, `REPRODUCE.md`
- [ ] GitHub repository (public)
- [ ] Arxiv submission

---

## Verification Checklist

1. **Day 1 Check:** Can generate Llama response in <30s?
2. **Day 3 Check:** 100 examples with >75% validation accuracy?
3. **Day 5 Check:** Compression shows >10% degradation?
4. **Day 9 Check:** R1 clusters stable (ARI > 0.5)?
5. **Day 12 Check:** Semantic > turn-based by >5%?
6. **Day 18 Check:** Paper compiles, code reproducible?
7. **Day 21 Check:** Arxiv submitted?

---

## Critical Success Factors

1. **Environment works Day 1** - All APIs + Llama functional
2. **Dataset quality** - >75% genuine conflicts
3. **R1 stability** - ARI > 0.5 ensures clusters are meaningful
4. **Statistical significance** - All experiments report p-values
5. **Honest limitations** - Acknowledge synthetic data, single model

---

## Resources

### Models
- Llama: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- Installation: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`

### APIs
- DeepSeek: https://api.deepseek.com (OpenAI-compatible)
- Claude: https://console.anthropic.com

### References
- FlowKV: https://arxiv.org/abs/2505.xxxxx (search latest)
- Multi-IF benchmark
- StreamingLLM, H2O, SnapKV papers

---

**End of Complete Plan**

This plan provides detailed day-by-day execution guidance for the full 3-week research sprint. Each day includes specific objectives, time allocations, code examples, and success criteria. Buffer days and pivot strategies ensure flexibility and resilience.
