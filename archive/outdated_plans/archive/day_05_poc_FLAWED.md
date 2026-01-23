# Day 5 POC: Turn-Based vs. Semantic Isolation (Core Hypothesis Test)

**Week 1 - Day 5**

---

## Strategic Pivot

**SKIP:** KV cache compression (too complex, examples too short, tangential to hypothesis)

**FOCUS:** Core RDIC hypothesis:
> Semantic instruction isolation outperforms turn-based isolation for conflicting multi-turn instructions

**Approach:** Test with prompt organization (simpler than KV cache manipulation for POC)

---

## Core Hypothesis

When instructions conflict across multiple turns:
- **Baseline (Sequential):** Chronological presentation struggles with conflicts
- **Turn-Based Isolation:** Grouping by turn boundaries helps somewhat
- **Semantic Isolation:** Grouping by instruction similarity helps MORE

**Research Question:** Does organizing conflicting instructions by semantic similarity (rather than turn boundaries) improve instruction-following?

---

## Objectives

1. Generate longer test examples (5-7 turns, ~400-500 tokens) using Claude Haiku CLI
2. Test 3 prompt organization strategies with Gemma 3 12B
3. Evaluate instruction-following scores
4. Determine if semantic organization helps vs. turn-based

**Success:** Semantic > Turn-Based > Baseline (with statistical significance)

---

## Tasks

| Task | Time | Details |
|------|------|---------|
| Generate 20 longer examples via Claude Haiku | 1.5h | Use Task tool with Haiku model (cost savings) |
| Implement 3 prompt strategies | 1h | Sequential, turn-grouped, semantic-grouped |
| Run Gemma 3 on all 60 tests (20×3) | 2h | ~2min per generation |
| Evaluate with hybrid evaluator | 1h | Rule-based + Claude Haiku LLM judge |
| Statistical analysis | 30m | Paired t-tests, effect sizes |
| Generate results visualization | 30m | Bar chart comparing strategies |

**Total:** ~6-7 hours (single day)

---

## Example Generation (Claude Haiku via CLI)

Generate longer examples that stress Gemma 3's instruction-following:

```python
def generate_long_example_via_cli(conflict_type, domain):
    """
    Use Claude CLI (Haiku) to generate longer test examples.
    5-7 turns, ~400-500 tokens, clear conflicting instructions.
    """

    prompt = f"""Generate a multi-turn conversation example for testing instruction-following under conflicting instructions.

**Conflict Type:** {conflict_type}
**Domain:** {domain}
**Length:** 5-7 turns, approximately 400-500 tokens total

**Structure:**
Turn 1: Introduce Instruction A (e.g., "Use formal professional tone")
Turn 2: Provide context/content (2-3 sentences)
Turn 3: Introduce Instruction B that conflicts with A (e.g., "Use casual friendly tone")
Turn 4: Provide more context/content (2-3 sentences)
Turn 5: Additional context (optional)
Turn 6: Final query that requires following BOTH instructions

**Requirements:**
1. Instructions must GENUINELY conflict (cannot both be satisfied)
2. Each turn should be substantial (50-100 tokens)
3. Context should be realistic and relevant
4. Final query should make the conflict unavoidable
5. Include ground truth: which instructions semantically cluster together

Output JSON:
{{
  "id": "conflict_XXX",
  "conflict_type": "{conflict_type}",
  "domain": "{domain}",
  "turns": [
    {{"turn_id": 1, "role": "user", "instruction": "...", "content": "..."}},
    ...
  ],
  "semantic_clusters": {{
    "cluster_A": ["instruction from turn 1", ...],
    "cluster_B": ["instruction from turn 3", ...]
  }}
}}
"""

    # Use Task tool with Haiku model for cost savings
    from Task import run_task

    result = run_task(
        subagent_type="general-purpose",
        model="haiku",  # Cheaper than Sonnet
        prompt=prompt,
        description="Generate long RDIC test example"
    )

    return json.loads(result)


# Generate 20 examples
for i in range(20):
    example = generate_long_example_via_cli(
        conflict_type=random.choice(['tone', 'detail', 'style', 'format', 'content']),
        domain=random.choice(DOMAINS)
    )
    save_json(example, f"data/long_examples/conflict_{i:03d}.json")
```

**Cost Estimate:**
- 20 examples × ~500 tokens input × ~600 tokens output = ~22K tokens
- Claude Haiku: $0.25/MTok input, $1.25/MTok output
- Cost: ~$0.01 input + ~$0.03 output = **~$0.04 total** (very cheap!)

---

## Three Prompt Strategies

### Strategy 1: Sequential (Baseline)

Present all information chronologically (current approach):

```python
def format_sequential(example):
    """
    Baseline: All turns in chronological order.
    Model must track conflicting instructions across turns.
    """
    parts = []
    for turn in example['turns']:
        if 'instruction' in turn:
            parts.append(f"Instruction: {turn['instruction']}")
        if 'content' in turn:
            parts.append(f"Context: {turn['content']}")
        if 'query' in turn:
            parts.append(f"Query: {turn['query']}")

    return "\n\n".join(parts) + "\n\nPlease provide a response following ALL instructions."
```

**Expected:** Model struggles with conflicting instructions (they're mixed together)

---

### Strategy 2: Turn-Based Grouping (FlowKV-style)

Group information by turn boundaries:

```python
def format_turn_based(example):
    """
    Turn-based isolation: Group content by turn.
    Helps model see turn structure more clearly.
    """
    sections = []

    for i, turn in enumerate(example['turns'], 1):
        turn_content = [f"=== Turn {i} ==="]

        if 'instruction' in turn:
            turn_content.append(f"Instruction: {turn['instruction']}")
        if 'content' in turn:
            turn_content.append(f"Context: {turn['content']}")
        if 'query' in turn:
            turn_content.append(f"Query: {turn['query']}")

        sections.append("\n".join(turn_content))

    return "\n\n".join(sections) + "\n\nPlease provide a response following ALL instructions from all turns."
```

**Expected:** Better than sequential (turn structure is clearer), but still mixes conflicting instructions

---

### Strategy 3: Semantic Grouping (RDIC-style)

Group instructions by semantic similarity:

```python
def format_semantic(example):
    """
    Semantic isolation: Group by instruction similarity.
    Conflicting instructions in separate sections.
    """
    # Extract semantic clusters from example
    clusters = example['semantic_clusters']

    sections = []

    # Section for each semantic cluster
    for cluster_name, instructions in clusters.items():
        cluster_content = [f"=== Instruction Group: {cluster_name} ==="]

        # Find turns that match these instructions
        for instruction in instructions:
            for turn in example['turns']:
                if turn.get('instruction') == instruction:
                    cluster_content.append(f"Instruction: {instruction}")
                    if 'content' in turn:
                        cluster_content.append(f"Context: {turn['content']}")

        sections.append("\n".join(cluster_content))

    # Add final query
    final_query = next(
        turn['query'] for turn in example['turns']
        if 'query' in turn
    )
    sections.append(f"=== Final Query ===\nQuery: {final_query}")

    return "\n\n".join(sections) + "\n\nPlease provide a response. Consider both instruction groups, but prioritize semantic consistency."
```

**Expected:** Best performance (conflicting instructions are separated, model can handle each cluster coherently)

---

## Example Comparison

**Original Short Example (150 tokens):**
```
Turn 1: Use formal tone
Turn 2: [brief context]
Turn 3: Use casual tone, write response
```

**New Long Example (400+ tokens):**
```
Turn 1: Use formal professional tone. Address recipient as "Distinguished Colleague"
Turn 2: [Context: 3-4 sentences about project status, challenges, next steps]
Turn 3: Use casual friendly tone. Include personal anecdotes and informal language
Turn 4: [Context: 3-4 sentences about team dynamics, celebrations, personal updates]
Turn 5: [Context: 2-3 sentences about budget considerations]
Turn 6: Write a comprehensive email to your project partner discussing all points above

Instructions:
- From Turn 1: Formal, professional, "Distinguished Colleague"
- From Turn 3: Casual, friendly, personal anecdotes
- CONFLICT: Cannot satisfy both simultaneously
```

**Why Longer Examples Matter:**
1. More content to track (stresses model more)
2. Conflicting instructions further apart (harder to reconcile)
3. More realistic multi-turn scenario
4. Can test if semantic grouping helps model handle complexity

---

## Experiment Execution

```python
def run_poc_experiment(examples):
    """
    Test 3 strategies on all examples.
    Incremental execution with progress tracking.
    """
    results = []

    strategies = {
        'sequential': format_sequential,
        'turn_based': format_turn_based,
        'semantic': format_semantic
    }

    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Processing {example['id']}")

        example_results = {
            'id': example['id'],
            'conflict_type': example['conflict_type'],
            'strategies': {}
        }

        for strategy_name, format_func in strategies.items():
            print(f"  Testing {strategy_name}...")

            # Format prompt
            prompt = format_func(example)

            # Generate with Gemma 3 (no KV cache manipulation!)
            response = generate_gemma3(prompt, max_tokens=300)

            # Evaluate against both conflicting instructions
            instructions = [
                turn['instruction']
                for turn in example['turns']
                if 'instruction' in turn
            ]

            scores = []
            for instruction in instructions:
                score = evaluator.evaluate(instruction, response)
                scores.append(score.combined_score)

            mean_score = np.mean(scores)

            example_results['strategies'][strategy_name] = {
                'response': response,
                'scores': scores,
                'mean_score': mean_score
            }

            print(f"    Score: {mean_score:.3f}")

        results.append(example_results)

        # Save incrementally
        save_json(results, "results/poc_partial.json")

        # Show running statistics
        current_means = {
            strategy: np.mean([
                r['strategies'][strategy]['mean_score']
                for r in results
            ])
            for strategy in strategies.keys()
        }
        print(f"  Running means: {current_means}")

    return results
```

---

## Statistical Analysis

```python
from scipy import stats

def analyze_poc_results(results):
    """Compare 3 strategies with statistical tests."""

    # Extract scores by strategy
    sequential_scores = [r['strategies']['sequential']['mean_score'] for r in results]
    turn_based_scores = [r['strategies']['turn_based']['mean_score'] for r in results]
    semantic_scores = [r['strategies']['semantic']['mean_score'] for r in results]

    # Paired t-tests
    t1, p1 = stats.ttest_rel(sequential_scores, turn_based_scores)
    t2, p2 = stats.ttest_rel(sequential_scores, semantic_scores)
    t3, p3 = stats.ttest_rel(turn_based_scores, semantic_scores)

    print("Statistical Results:")
    print(f"Sequential vs. Turn-Based: t={t1:.3f}, p={p1:.4f} {'***' if p1<0.05 else 'ns'}")
    print(f"Sequential vs. Semantic: t={t2:.3f}, p={p2:.4f} {'***' if p2<0.05 else 'ns'}")
    print(f"Turn-Based vs. Semantic: t={t3:.3f}, p={p3:.4f} {'***' if p3<0.05 else 'ns'}")

    # Effect sizes (Cohen's d)
    d_turn = (np.mean(turn_based_scores) - np.mean(sequential_scores)) / np.std(sequential_scores)
    d_semantic = (np.mean(semantic_scores) - np.mean(sequential_scores)) / np.std(sequential_scores)

    print(f"\nEffect Sizes:")
    print(f"Turn-Based improvement: d={d_turn:.3f}")
    print(f"Semantic improvement: d={d_semantic:.3f}")

    # Success criteria
    print(f"\n{'='*60}")
    print("Success Criteria:")
    print(f"✓ Semantic > Sequential: {np.mean(semantic_scores) > np.mean(sequential_scores)}")
    print(f"✓ Semantic > Turn-Based: {np.mean(semantic_scores) > np.mean(turn_based_scores)}")
    print(f"✓ Statistical significance (p<0.05): {p3 < 0.05}")

    if np.mean(semantic_scores) > np.mean(turn_based_scores) > np.mean(sequential_scores):
        print("\n✅ HYPOTHESIS CONFIRMED: Semantic > Turn-Based > Sequential")
    else:
        print("\n⚠️ HYPOTHESIS NOT CONFIRMED - Need deeper analysis")

    return {
        'sequential_mean': np.mean(sequential_scores),
        'turn_based_mean': np.mean(turn_based_scores),
        'semantic_mean': np.mean(semantic_scores),
        'p_values': {'seq_vs_turn': p1, 'seq_vs_sem': p2, 'turn_vs_sem': p3},
        'effect_sizes': {'turn_based': d_turn, 'semantic': d_semantic}
    }
```

---

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def generate_poc_figure(analysis):
    """Bar chart comparing 3 strategies."""

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ['Sequential\n(Baseline)', 'Turn-Based\n(FlowKV-style)', 'Semantic\n(RDIC)']
    means = [
        analysis['sequential_mean'],
        analysis['turn_based_mean'],
        analysis['semantic_mean']
    ]

    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    bars = ax.bar(strategies, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add significance stars
    if analysis['p_values']['turn_vs_sem'] < 0.05:
        ax.plot([1, 2], [max(means) + 0.05, max(means) + 0.05], 'k-', linewidth=1.5)
        ax.text(1.5, max(means) + 0.06, '***', ha='center', fontsize=14)

    ax.set_ylabel('Instruction-Following Score', fontsize=14, fontweight='bold')
    ax.set_title('Prompt Organization Strategies for Conflicting Instructions',
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/figures/poc_strategies.png', dpi=300, bbox_inches='tight')
    print("✓ Saved figure to results/figures/poc_strategies.png")
```

---

## Files to Create

- `/Users/dev_user/semantic/data/long_examples/` - 20 longer test examples (via Claude Haiku)
- `/Users/dev_user/semantic/src/prompt_strategies.py` - 3 formatting functions
- `/Users/dev_user/semantic/experiments/poc_isolation.py` - Main POC experiment
- `/Users/dev_user/semantic/results/poc_results.json` - Full results
- `/Users/dev_user/semantic/results/figures/poc_strategies.png` - Bar chart
- `/Users/dev_user/semantic/DAY_5_POC_STATUS.md` - Completion report

---

## Success Criteria

- [ ] 20 long examples generated (~400-500 tokens each)
- [ ] All 60 tests complete (20 examples × 3 strategies)
- [ ] Semantic strategy significantly outperforms turn-based (p<0.05)
- [ ] Turn-based outperforms sequential baseline
- [ ] Effect size for semantic > 0.3 (meaningful improvement)
- [ ] Results visualization generated

---

## Expected Findings

**Hypothesis:** Semantic > Turn-Based > Sequential

**Why Semantic Should Win:**
- Conflicting instructions are separated
- Model can process each semantic cluster coherently
- Reduces confusion from interleaved conflicts

**Why Turn-Based Should Beat Sequential:**
- Turn boundaries provide structure
- Easier to track instruction changes
- But doesn't separate semantic conflicts within complex turns

**If Hypothesis Fails:**
- Sequential = Turn-Based = Semantic → Prompt organization doesn't matter (Gemma 3 too weak?)
- Turn-Based > Semantic → Turn boundaries more important than semantics
- All perform poorly → Examples still too easy or instructions not genuinely conflicting

---

## Next Steps After POC

**If Semantic > Turn-Based (SUCCESS):**
- Days 6-7: Extend to 50-100 examples
- Week 2: Implement true KV cache-based semantic isolation (not just prompting)
- Week 3: Test with DeepSeek R1 discovered clusters

**If Results Inconclusive:**
- Analyze failure modes
- Generate harder examples
- Try different prompt formats
- Consider if Gemma 3 is capable enough for this task

---

## Key Advantages of This Approach

1. **Tests core hypothesis directly** - Semantic vs. turn-based isolation
2. **No KV cache complexity** - Pure prompt organization test (simpler POC)
3. **Uses Claude CLI** - Cost-effective example generation (~$0.04)
4. **Longer examples** - More realistic, stresses model more
5. **Incremental execution** - See progress in real-time
6. **Fast turnaround** - Can complete in 1 day
7. **Clear go/no-go decision** - Either semantic wins or we pivot

---

## Cost Estimate

- Example generation: ~$0.04 (Claude Haiku via CLI)
- Evaluation: ~$0.50 (Claude Haiku LLM judge, 60 evaluations × ~$0.008 each)
- **Total: ~$0.54** (very affordable!)

---

**Quick Reference:**
- **Previous Day:** [Day 4](day_04.md) - Evaluation framework
- **Next Day:** [Day 6](day_06.md) - Extended testing or KV cache implementation
- **Complete Plan:** [Complete 3-Week Plan](../complete_plan.md)

---

*Last Updated: 2026-01-22 (Strategic pivot to core hypothesis)*
