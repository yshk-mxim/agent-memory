# Embedding-Based Clustering in POC Evaluation

**Date**: 2026-01-23
**Change**: Use embedding model for semantic clustering in POC (not ground truth labels)
**Reason**: Ground truth doesn't prove the approach works in practice

---

## The Problem with Ground Truth Labels

**Original plan**: Use pre-labeled ground truth clusters during POC evaluation
- Examples come with `semantic_clusters` field
- Just read labels from JSON
- No clustering model needed

**Issue**: This doesn't prove the approach is practical!
- Ground truth labels aren't available in production
- Doesn't demonstrate that automatic clustering works
- Unrealistic evaluation scenario
- Can't claim the system is deployable without this proof

---

## Updated Approach: Embedding-Based Clustering in POC

### What Changed

**Now**: Use embedding model (`all-MiniLM-L6-v2`) for actual semantic clustering in all experiments

**Benefits**:
1. ✅ **Realistic**: Proves approach works without ground truth
2. ✅ **Practical**: Demonstrates actual deployment scenario
3. ✅ **Validated**: If embedding routing succeeds, approach is proven
4. ✅ **Same architecture**: POC uses identical setup to production
5. ✅ **Still cheap**: $0 cost, ~200MB memory, ~10-50ms per decision

---

## Implementation

### Sprint 00 (Week 0) - Setup

**Monday Morning**:
- Install sentence-transformers
- Download `all-MiniLM-L6-v2` model (~80MB)
- Test embedding generation (<50ms per turn)
- Verify memory footprint (<200MB)

**Monday Afternoon**:
- Implement cluster discovery (analyze task, identify semantic roles)
- Create prototype embeddings for each role
- Implement turn routing via cosine similarity
- Test on 1 pilot example

**Deliverable**: `src/embedding_clustering.py`

### Sprint 01 (Week 1) - Finalization

**Monday Morning**:
- Finalize automatic cluster discovery
- Implement prototype embeddings for semantic roles
- Validate on pilot data (routing should match intuition ~80%+)

**Deliverable**: `src/embedding_clustering.py` (production-ready)

### Sprint 02 (Weeks 2-3) - Experiments

**All semantic condition runs**:
- Use embedding-based routing for every turn
- No ground truth labels used
- Log routing decisions and confidence scores
- Capture routing telemetry for analysis

**Deliverable**: Routing logs + confidence scores for all 50 examples

### Sprint 03 (Weeks 4-6) - Validation

**Wednesday Morning**:
- Analyze routing confidence scores (should be >0.7)
- Manual inspection of 20 random routing decisions (~80%+ accuracy expected)
- Identify systematic mis-routings
- Compare embedding routing to manual annotation (10 examples)

**Deliverable**: `results/phase1_analysis/embedding_routing_quality.md`

---

## Technical Details

### Cluster Discovery (Per Example)

```python
from sentence_transformers import SentenceTransformer

# One-time setup
model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB

# Per example: Discover semantic roles
def discover_clusters(task_description):
    """
    Analyze task, identify semantic roles.

    Returns: ["Technical Debugging", "Documentation", "Code Review"]

    Options:
    - Simple heuristic (keyword matching)
    - Claude Haiku analysis (~$0.001 per task)
    - Clustering algorithm (HDBSCAN on turn embeddings)
    """
    roles = analyze_task_simple(task_description)  # Heuristic

    # Create prototype embeddings
    cluster_embeddings = {
        role: model.encode(get_role_description(role))
        for role in roles
    }
    return cluster_embeddings
```

### Turn Routing (Per Turn)

```python
def route_turn(turn_text, cluster_embeddings):
    """
    Route turn to most similar semantic cluster.

    Returns: (cluster_name, confidence_score)
    """
    turn_embedding = model.encode(turn_text)  # ~10-50ms

    similarities = {
        cluster: cosine_similarity(turn_embedding, cluster_emb)
        for cluster, cluster_emb in cluster_embeddings.items()
    }

    best_cluster = max(similarities, key=similarities.get)
    confidence = similarities[best_cluster]

    return best_cluster, confidence
```

### Example Output

```json
{
  "example_id": "coding_001",
  "task": "Debug API endpoint, write docs, review code",
  "clusters": {
    "technical_debugging": {
      "turns": [0, 2, 4, 6],
      "avg_confidence": 0.87
    },
    "documentation": {
      "turns": [1, 3, 5],
      "avg_confidence": 0.82
    },
    "code_review": {
      "turns": [7, 8, 9],
      "avg_confidence": 0.91
    }
  },
  "routing_quality": "high (avg_confidence=0.87)"
}
```

---

## Memory & Compute Impact

### Memory

**Before** (ground truth only):
- Gemma 3 12B: 7-10GB
- KV cache: 1-2GB
- Total: ~10GB

**After** (embedding model + Gemma 3):
- Embedding model: ~200MB
- Gemma 3 12B: 7-10GB
- KV cache: 1-2GB
- Total: ~10.2GB

**Increase**: ~200MB (~2% overhead)

### Speed

**Routing time**:
- 200 turns × 50ms = 10 seconds
- Negligible compared to generation (200 outputs × 8s = 27 minutes)

**Total time**: ~27 minutes (routing overhead <1%)

---

## Validation Strategy

### Quantitative Metrics

1. **Routing confidence scores**: Should average >0.70
2. **Agreement with manual annotation**: >80% on 10-example subset
3. **Cluster size balance**: No cluster should have <10% or >60% of turns

### Qualitative Analysis

1. **Manual inspection**: Review 20 random routing decisions
2. **Failure mode identification**: Find patterns in mis-routings
3. **Edge case analysis**: Examine low-confidence decisions (<0.5)

### Success Criteria

- ✅ Average routing confidence >0.70
- ✅ Manual annotation agreement >80%
- ✅ No systematic mis-routing patterns
- ✅ Routing quality doesn't degrade semantic condition performance

---

## Why This Is Better

### Comparison

| Aspect | Ground Truth Labels | Embedding-Based Routing |
|--------|---------------------|------------------------|
| **Realism** | ❌ Not available in production | ✅ Matches deployment scenario |
| **Proof** | ❌ Doesn't prove approach works | ✅ Validates practical viability |
| **Memory** | 0MB | ~200MB (~2% overhead) |
| **Speed** | Instant | ~10-50ms per turn (negligible) |
| **Cost** | $0 | $0 (local inference) |
| **Publishability** | ⚠️ Weak (unrealistic assumption) | ✅ Strong (realistic evaluation) |

### Publication Impact

**Before** (ground truth):
- Reviewers would question: "How does this work without labels?"
- Weak claim: "Approach requires pre-labeled clusters"
- Not deployable without manual annotation

**After** (embedding routing):
- Strong claim: "Fully automatic semantic clustering via embeddings"
- Proven: "POC demonstrates 80%+ routing accuracy"
- Deployable: "Same architecture in research and production"

---

## Updated Sprint Files

All sprint files updated to reflect embedding-based clustering:

1. **`sprint_00_pilot.md`**:
   - Monday morning: Set up embedding model
   - Monday afternoon: Implement embedding clustering
   - Test on pilot data

2. **`sprint_01_dataset_and_metrics.md`**:
   - Monday morning: Finalize embedding clustering
   - Validate routing accuracy on pilot

3. **`sprint_02_experiments.md`**:
   - All semantic runs use embedding routing
   - Log routing decisions and confidence scores

4. **`sprint_03_analysis_decision.md`**:
   - Wednesday morning: Validate embedding routing quality
   - Analyze confidence scores and manual agreement

5. **`SEMANTIC_CLUSTERING_APPROACH.md`**:
   - Updated all sections to reflect embedding use in POC
   - Removed "ground truth" approach from research plan
   - Emphasized POC = Production architecture

---

## Cost Analysis

**No change to budget**:
- Embedding model: $0 (local inference, sentence-transformers)
- Memory overhead: ~200MB
- Speed overhead: ~10 seconds for 200 routing decisions (negligible)

**Total project budget**: Still $90-240
- Dataset generation: $60-90
- Claude AI judge: $30
- Optional MTurk: $0-120

---

## Risks & Mitigation

### Risk: Embedding routing accuracy <80%

**Mitigation**:
- Try different embedding models (larger = better accuracy)
- Improve cluster discovery (use Claude Haiku for complex tasks)
- Add confidence threshold (manual review for low-confidence decisions)

**Escalation**:
- If routing quality poor (<70%), this is critical finding
- May need to revise approach or use hybrid (embeddings + occasional LLM verification)

### Risk: Memory constraints with both models loaded

**Mitigation**:
- Embedding model is tiny (~200MB), should be no issue
- Can unload embedding model after routing decisions made
- Measure actual memory usage in Sprint 00

**Escalation**:
- Very unlikely given 24GB RAM and only 200MB overhead

---

## Summary

**What changed**: POC now uses embedding-based semantic clustering (not ground truth labels)

**Why**: Proves the approach works in practice, not just in theory

**Impact**:
- ✅ More realistic evaluation
- ✅ Stronger publication case
- ✅ Proves deployment viability
- ✅ Same architecture in POC and production
- ✅ Minimal overhead (~200MB memory, <1% speed impact)
- ✅ Still $0 cost

**Timeline**: No change (still 13-15 weeks)

**Budget**: No change (still $90-240)

---

**Created**: 2026-01-23
**Status**: Integrated into all sprint plans
**Ready for**: Sprint 00 pilot testing with embedding-based clustering

