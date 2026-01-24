# Semantic Clustering Approach: Cost-Effective Implementation

**Date**: 2026-01-23
**Issue**: Using Gemma 3 12B for clustering would be expensive
**Solution**: Two-stage approach with cheap clustering + expensive generation

---

## The Problem

**User's concern**: If we use Gemma 3 12B for semantic clustering decisions, costs will explode.

**Example**:
- 50 examples × 10 turns each = 500 clustering decisions
- Using Gemma 3 12B: 500 × ~150 tokens × $cost = **expensive**
- Plus generation cost on top of that

**You're right!** This would be inefficient.

---

## The Solution: Two-Stage Architecture

### CRITICAL CLARIFICATION: Model Usage

**Gemma 3 12B is ONLY used for generation, NOT for clustering!**

**Why?**
- Gemma 3 12B: 7-10GB memory, slow inference (~5-10s per decision)
- Clustering is a simple task, doesn't need 12B parameters
- **Wasteful** to use huge model for routing decisions

**Instead:**
- **Clustering/Routing**: Tiny embedding model (80-500MB, ~10-50ms)
- **Generation**: Gemma 3 12B (7-10GB, ~5-10s per output)

This is the **compute-efficient** approach!

---

### Stage 1: Cluster Discovery (One-Time, Preprocessing)

**Purpose**: Identify semantic agent roles for a given task

**Options**:

**Option A: Pre-labeled Dataset (Current Plan - Research)**
- Dataset already has ground truth clusters from generation
- Examples come with `semantic_clusters` labels
- **No model needed**: Just read labels from JSON
- **Memory**: 0MB (just data loading)
- **Cost**: $0 (already labeled during dataset generation)
- **Use for**: Research evaluation (what we're doing)

**Option B: Automatic Cluster Discovery (Future System)**
- Use **Claude Sonnet 4.5** or **DeepSeek R1** to analyze task ONCE
- Prompt: "Analyze this multi-agent task and identify distinct semantic roles"
- Returns: Cluster definitions (e.g., "Technical Analysis", "Business Strategy", "Synthesis")
- **Cost**: ~$0.10 per task (one-time)
- **Use for**: Production system

**Option C: Embedding-Based Clustering (Cheapest)**
- Use **sentence-transformers** (e.g., `all-MiniLM-L6-v2`)
- Compute embeddings for all turns
- Apply clustering algorithm (K-means, HDBSCAN)
- **Cost**: $0 (local, CPU-based)
- **Use for**: Simple tasks with clear boundaries

---

### Stage 2: Turn Routing (Per-Turn, Runtime)

**Purpose**: Assign each new turn to the correct cluster

**Cheap Options**:

**Option 1: Embedding Similarity (Recommended)**
```python
from sentence_transformers import SentenceTransformer

# One-time setup
model = SentenceTransformer('all-MiniLM-L6-v2')  # Tiny, fast, free
cluster_embeddings = {
    'technical': model.encode(technical_examples),
    'business': model.encode(business_examples)
}

# Per turn (cheap)
turn_embedding = model.encode(new_turn)
similarities = {
    cluster: cosine_similarity(turn_embedding, cluster_emb)
    for cluster, cluster_emb in cluster_embeddings.items()
}
assigned_cluster = max(similarities, key=similarities.get)
```

**Cost**: $0 (local inference, ~10ms per turn)
**Quality**: Good enough for most cases

**Option 2: Keyword Matching (Fastest)**
```python
# Define cluster keywords once
cluster_keywords = {
    'technical': ['api', 'code', 'function', 'bug', 'performance'],
    'business': ['revenue', 'strategy', 'market', 'customer', 'roi']
}

# Per turn (very cheap)
def route_turn(turn_text):
    scores = {}
    for cluster, keywords in cluster_keywords.items():
        scores[cluster] = sum(1 for kw in keywords if kw in turn_text.lower())
    return max(scores, key=scores.get)
```

**Cost**: $0 (regex matching, <1ms per turn)
**Quality**: Decent for clear domain separation

**Option 3: Tiny Classifier (Good Balance)**
```python
from transformers import pipeline

# One-time setup
classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")  # Or DistilBERT

# Per turn (cheap, ~50ms)
result = classifier(
    new_turn,
    candidate_labels=["technical analysis", "business strategy", "synthesis"]
)
assigned_cluster = result['labels'][0]
```

**Cost**: $0 (local inference, small models)
**Quality**: Better than embeddings, still fast

---

### Stage 3: Generation (Per-Cluster, Large Model)

**Purpose**: Generate agent outputs within each cluster

**Model**: **Gemma 3 12B** (via MLX, local)

**Memory**: 7-10GB (4-bit quantization)
**Speed**: ~5-10 seconds per generation
**Cost**: $0 (local inference, no API)
**Quality**: Excellent (12B parameter model)

**IMPORTANT**: This is the ONLY time we use Gemma 3 12B!
- NOT for clustering
- NOT for routing
- ONLY for generating specialist outputs

---

## Memory & Compute Analysis

### Current Plan (Research Evaluation - Realistic POC)

**Clustering**: Sentence-transformers embedding model (~80-200MB)
**Routing**: Embedding-based similarity (~10-50ms per decision)
**Generation**: Gemma 3 12B via MLX (7-10GB)

**Peak Memory Usage**: ~10-13GB
- Embedding model: ~200MB
- Gemma 3 12B: 7-10GB
- KV cache (3 clusters): 1-2GB
- Python overhead: 1GB

**Compute Cost**: Moderate
- Embedding model: ~200MB, fast (~10-50ms per routing decision)
- Gemma 3 12B: 7-10GB, ~5-10 seconds per generation
- Total time for 200 outputs:
  - Routing: 200 decisions × 50ms = 10 seconds
  - Generation: 200 outputs × 8s = 27 minutes
  - **Total**: ~27 minutes (routing overhead negligible)

**API Cost**: $0 (local inference)

---

## Cost Analysis

### Current Plan (Research Evaluation)

**Clustering**: $0 (pre-labeled dataset, no model)
**Routing**: $0 (not needed - ground truth clusters provided)
**Generation**: $0 (Gemma 3 12B via MLX, local)

**Total inference cost**: **$0**

**Only costs**:
- Dataset generation (Claude API): $60-90
- Claude AI judge evaluation: $30
- Optional MTurk: $0-120

**Total**: $90-240 (all for data generation/evaluation, not inference)

---

### Future Production System

**Scenario**: User provides new task, system creates virtual agents

**System Architecture**:
```
┌─────────────────────────────────────┐
│  Tiny Embedding Model (200MB)      │ ← Routing only
│  sentence-transformers/all-MiniLM  │
└─────────────────────────────────────┘
              ↓ (decides cluster)
┌─────────────────────────────────────┐
│  Gemma 3 12B (7-10GB)              │ ← Generation only
│  mlx-community/gemma-3-12b-it-4bit │
└─────────────────────────────────────┘
```

**Step 1: Cluster Discovery** (one-time per task):
- Option A: Embedding-based clustering: $0 (local, ~100ms)
- Option B: Claude Haiku analysis: ~$0.01 (for complex tasks)
- **Recommended**: Embeddings for simple tasks, Claude Haiku for complex
- **Memory**: 200MB (embedding model)

**Step 2: Turn Routing** (per turn, ~50ms each):
- Embedding model: Encode new turn (200MB model)
- Compute similarity to cluster embeddings
- Assign to highest-similarity cluster
- **Memory**: 200MB (same embedding model)
- **Cost**: $0 (local inference)

**Step 3: Generation** (per cluster, ~5-10s each):
- Gemma 3 12B: Generate specialist output
- Only runs AFTER routing decision
- Uses cluster-specific context
- **Memory**: 7-10GB (Gemma 3 12B)
- **Cost**: $0 (local inference)

**Total Memory**: ~10.2GB (both models loaded)
**Total Cost**: **~$0 per task** (local inference)
**Speed**:
- Routing: 500 turns × 50ms = 25 seconds
- Generation: 500 outputs × 8s = 67 minutes
- **Total**: ~68 minutes for complex task (can run overnight)

---

## Why Gemma 3 12B Is Cost-Effective

**Local Inference via MLX**:
- Runs on Mac with 24GB RAM
- 4-bit quantization: ~7-10GB memory
- No API calls needed
- Unlimited inference for free

**Comparison**:
- Claude Sonnet API: ~$0.003 per 1K tokens
- GPT-4: ~$0.01 per 1K tokens
- **Gemma 3 12B (local)**: $0

**For 500 generations @ 200 tokens each**:
- Claude cost: ~$300
- **Gemma cost**: $0

This is why we chose MLX + Gemma 3 12B!

---

## Implementation for Our Research POC

### What We're Actually Doing (Realistic Evaluation)

**Week 0-1**: Implement and test embedding-based clustering
- **NO ground truth labels used during evaluation**
- Embedding model (`all-MiniLM-L6-v2`) performs actual clustering
- This proves the approach works in practice, not just in theory

```python
from sentence_transformers import SentenceTransformer

# One-time setup (Sprint 00, Monday)
model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB

# Per example: Automatic cluster discovery
def discover_clusters(task_description):
    # Analyze task, identify semantic roles
    # Returns: ["Technical Debugging", "Documentation", "Code Review"]
    roles = analyze_task(task_description)  # Simple heuristic or Claude Haiku

    # Create prototype embeddings for each role
    cluster_embeddings = {
        role: model.encode(get_role_description(role))
        for role in roles
    }
    return cluster_embeddings

# Per turn: Route to cluster (Sprint 01+)
def route_turn(turn_text, cluster_embeddings):
    turn_embedding = model.encode(turn_text)  # ~10-50ms
    similarities = {
        cluster: cosine_similarity(turn_embedding, cluster_emb)
        for cluster, cluster_emb in cluster_embeddings.items()
    }
    return max(similarities, key=similarities.get)  # Assign to best match
```

**Cost**: $0 (local inference)
**Memory**: ~200MB (embedding model)
**Speed**: ~10-50ms per routing decision

**Why this is better**:
- ✅ **Realistic**: Proves approach works without ground truth
- ✅ **Practical**: Demonstrates actual deployment scenario
- ✅ **Validated**: If embedding routing works well, approach is proven
- ✅ **Still cheap**: $0 cost, minimal memory, fast

---

### If We Built a Real System (Future Work)

**Step 1**: User provides task description
- "Help me debug API endpoints and write documentation"

**Step 2**: Cheap cluster discovery (embedding-based)
```python
# Analyze task, identify semantic roles
roles = discover_semantic_roles(task_description)  # Uses embeddings, $0
# Returns: ["API Debugging", "Documentation Writing", "Integration"]
```

**Step 3**: Per-turn routing (embedding similarity)
```python
# Each new user turn
assigned_cluster = route_turn_to_cluster(user_turn, roles)  # Embeddings, $0
```

**Step 4**: Generation (Gemma 3 12B)
```python
# Generate response in assigned cluster
response = generate_in_cluster(assigned_cluster, context)  # MLX, $0
```

**Total cost**: $0

---

## Updated Sprint Plans

### Sprint 00: Added Embedding Model Setup

**Changes**:
- Monday morning: Set up sentence-transformers and test embedding model
- Monday afternoon: Implement embedding-based semantic clustering
- Test on pilot data to verify routing works

**Deliverable**: `src/embedding_clustering.py`

### Sprint 01: Added Embedding Clustering Finalization

**Changes**:
- Monday morning: Finalize automatic cluster discovery and turn routing
- Validate on pilot data (clustering should match intuition ~80%+)

**Reason**: Prove the approach works in practice, not just with ground truth

### Sprint 02: Use Embedding-Based Routing

**Changes**: All semantic condition experiments use embedding model for routing
- No ground truth labels used
- Every turn is routed via embedding similarity (~10-50ms per decision)

**Cost**: $0 (local inference)
**Memory overhead**: ~200MB (embedding model loaded alongside Gemma 3 12B)

### Sprint 03: Validate Embedding Clustering Quality

**Changes**: Add analysis of embedding routing accuracy
- Compare routing decisions to manual inspection
- Measure routing confidence scores
- Identify any systematic mis-routings

---

## Comparison: Our Approach vs Alternatives

### Memory Usage Comparison (Key Difference!)

| Approach | Clustering Memory | Generation Memory | Peak Memory | Clustering Speed |
|----------|------------------|-------------------|-------------|------------------|
| **Our approach (POC & Production)** | **200MB** (embedding model) | 7-10GB | **10.2GB** | ~10-50ms |
| **BAD: Gemma 3 for clustering** + Gemma 3 for generation | **7-10GB** | 7-10GB | **10GB** | ~5-10s per decision |
| Ground truth only (unrealistic) | **0MB** (labels provided) | 7-10GB | **10GB** | Instant |
| Claude API for both | 0MB (cloud) | 0MB (cloud) | **1GB** (minimal) | ~1-2s per decision |

**Why not use Gemma 3 12B for clustering?**
- ❌ **Overkill**: 12B parameters for a simple routing decision
- ❌ **Slow**: 5-10 seconds per clustering decision vs 50ms for embeddings
- ❌ **Wasteful**: Loading 7-10GB model just to decide "technical vs business"
- ❌ **Same memory**: Doesn't save memory (already loading for generation)
- ❌ **Inefficient**: Would take 200 clustering decisions × 8s = 27 minutes just for routing!

**Why use tiny embedding model for clustering?**
- ✅ **Efficient**: 80-200MB model, perfect for routing
- ✅ **Fast**: 10-50ms per decision (100x faster than Gemma)
- ✅ **Good enough**: Embeddings work well for semantic similarity
- ✅ **Memory friendly**: 200MB vs 7-10GB
- ✅ **Can run alongside**: Both models loaded = 10.2GB total (still fits in 24GB RAM)

---

### API Cost Comparison

| Approach | Clustering Cost | Generation Cost | Total |
|----------|----------------|-----------------|-------|
| **Our approach (POC & Production)** | $0 (local embeddings) | $0 (local Gemma 3) | **$0** |
| Gemma 3 for clustering + Gemma 3 for generation | $0 (local) | $0 (local) | $0 (but very slow!) |
| Ground truth only (unrealistic) | $0 (labels provided) | $0 (local Gemma 3) | $0 (not practical) |
| Claude for clustering + Claude for generation | ~$50 | ~$300 | **$350** |
| GPT-4 for clustering + GPT-4 for generation | ~$100 | ~$1000 | **$1100** |

**Our approach is optimal**: Tiny embedding model for routing ($0, fast), Gemma 3 only for generation ($0, high quality)

---

## Key Takeaway: Addressing Your Concern

**Your concern is 100% valid and important!**

### What You're Worried About (Correctly!)

❌ **Bad approach**: Using Gemma 3 12B (7-10GB, slow) for clustering decisions
- Would waste compute on simple routing
- 5-10 seconds per decision × 200 decisions = 27 minutes just for routing
- Inefficient use of large model

### What We're Actually Doing (Good!)

✅ **Research POC evaluation** (now - REALISTIC):
- **Clustering**: Tiny embedding model (200MB, 10-50ms per decision)
- **Generation**: Gemma 3 12B (7-10GB, only for output generation)
- **Peak memory**: ~10.2GB
- **Efficient**: Right-sized models for each task
- **Realistic**: Proves the approach works without ground truth

✅ **Production system** (future - SAME as POC!):
- **Clustering**: Tiny embedding model (200MB, 10-50ms per decision)
- **Generation**: Gemma 3 12B (7-10GB, only for output generation)
- **Peak memory**: ~10.2GB (both loaded)
- **Efficient**: Identical architecture to POC
- **Validated**: POC proves production viability

### Memory Breakdown

```
Research Evaluation (Realistic POC):
┌──────────────────────┐
│ Embedding model      │  200 MB
├──────────────────────┤
│ Gemma 3 12B (gen)   │  7-10 GB
├──────────────────────┤
│ KV cache (3 clusters)│  1-2 GB
└──────────────────────┘
Total: ~10-13 GB ✅ Fits in 24GB RAM

Production System (Same as POC!):
┌──────────────────────┐
│ Embedding model      │  200 MB
├──────────────────────┤
│ Gemma 3 12B (gen)   │  7-10 GB
├──────────────────────┤
│ KV cache (3 clusters)│  1-2 GB
└──────────────────────┘
Total: ~10-13 GB ✅ Same memory footprint
```

**Key insight**: Research POC now uses the SAME architecture as production! This proves the approach is practical.

### Speed Comparison

| Task | Using Gemma 3 12B (BAD) | Using Embeddings (GOOD) |
|------|------------------------|-------------------------|
| Route 1 turn | ~8 seconds ❌ | ~50ms ✅ |
| Route 200 turns | ~27 minutes ❌ | ~10 seconds ✅ |
| Generate 200 outputs | ~27 minutes | ~27 minutes |
| **Total time** | **54 minutes** ❌ | **28 minutes** ✅ |

**Using embeddings for routing saves ~26 minutes (nearly 2× faster)!**

### Summary

✅ **We're already optimized!**
1. Research: No clustering model needed (ground truth)
2. Production: Tiny embedding model for routing (200MB, 50ms)
3. Generation: Gemma 3 12B only (right-sized for output quality)
4. Peak memory: ~10-13GB (fits comfortably in 24GB RAM)

**You caught an important optimization principle!** We're following it correctly:
- **Small model for simple tasks** (routing/clustering)
- **Large model for complex tasks** (generation/reasoning)

**Total runtime cost**: $0 (all local inference)
**Only costs**: Dataset generation ($60-90) + Claude AI judge ($30) = $90-120

---

## Recommended Clustering Stack

### For Research (Now)
- **Clustering**: Ground truth labels from dataset
- **Generation**: Gemma 3 12B (MLX, local)
- **Cost**: $0

### For Production (Future)
- **Clustering**: sentence-transformers embeddings + cosine similarity
- **Fallback**: Claude Haiku for ambiguous cases (~$0.001 per decision)
- **Generation**: Gemma 3 12B (MLX, local) or Llama 3.1 8B (even faster)
- **Cost**: ~$0 per task

---

## Action Items

1. ✅ **Clarify in documentation**: Clustering uses embedding model in both POC and production
2. ✅ **Add to Sprint 00**: Implement and test embedding-based routing on pilot data
3. ✅ **Add to Sprint 01**: Finalize embedding clustering implementation
4. ✅ **Update all sprints**: Use embedding-based routing in all semantic condition experiments
5. ✅ **Update NOVELTY.md**: Mention cheap clustering is feasible (not a cost barrier)
6. ✅ **POC proves production viability**: Same architecture used in research and deployment

---

**Summary**:
- ✅ Clustering: Free, realistic (embedding model in POC and production)
- ✅ Generation: Free (Gemma 3 12B via MLX, local)
- ✅ Total inference cost: $0
- ✅ Only costs: Dataset generation + Claude AI judge (~$90-120)
- ✅ POC uses same architecture as production (proves practicality)

**Your concern is valid and we're addressing it properly!** We're using tiny embedding model for clustering (not expensive Gemma 3 12B), and we're proving it works in the POC evaluation (not just assuming ground truth labels).

---

**Created**: 2026-01-23
**Status**: Cost-optimized approach confirmed
