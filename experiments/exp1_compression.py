"""
Experiment 1: Compression Degrades Instruction Following

This experiment demonstrates that KV cache compression hurts instruction-following
performance by comparing full context vs. compressed context generation.

Hypothesis:
- Full context: High instruction-following score (>0.6)
- Compressed context (50%): Lower score (<0.5)
- Degradation is statistically significant (p<0.05)

Usage:
    python -m experiments.exp1_compression
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import APIClients, save_json, load_json, print_section
from src.compression import compress_turns
from src.evaluator import HybridEvaluator


def format_conversation_context(turns: List[Dict]) -> str:
    """
    Format multi-turn conversation into a single context string.

    Args:
        turns: List of conversation turns

    Returns:
        Formatted context string
    """
    parts = []
    for turn in turns:
        if 'instruction' in turn:
            parts.append(f"Instruction: {turn['instruction']}")
        if 'content' in turn:
            parts.append(f"Context: {turn['content']}")
        if 'query' in turn:
            parts.append(f"Query: {turn['query']}")

    return "\n\n".join(parts)


def generate_response(clients: APIClients, context: str, max_tokens: int = 500) -> str:
    """
    Generate response using Gemma 3 12B.

    Args:
        clients: APIClients instance
        context: Full conversation context
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    prompt = f"{context}\n\nPlease provide a response that addresses the query while following ALL instructions given in the conversation above."

    try:
        response = clients.call_gemma(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def run_experiment(
    test_data_path: str = "data/test.json",
    results_path: str = "results/exp1_results.json",
    compression_ratio: float = 0.5,
    max_examples: int = None
):
    """
    Run compression degradation experiment.

    Args:
        test_data_path: Path to test dataset
        results_path: Path to save results
        compression_ratio: Compression ratio (0.5 = 50%)
        max_examples: Maximum number of examples to test (None = all)
    """
    print_section("Experiment 1: Compression Degrades Instruction Following")

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = load_json(test_data_path)

    if max_examples:
        test_data = test_data[:max_examples]

    print(f"✓ Loaded {len(test_data)} examples")

    # Initialize clients
    print("\nInitializing API clients...")
    clients = APIClients()

    # Initialize evaluator
    print("Initializing hybrid evaluator (40% rule + 60% LLM)...")
    evaluator = HybridEvaluator(use_cache=True)

    # Run experiment
    results = []

    print(f"\nRunning experiment on {len(test_data)} examples...")
    print("="*60)

    for i, example in enumerate(test_data, 1):
        print(f"\n[{i}/{len(test_data)}] Processing {example['id']} ({example['conflict_type']})")

        # Extract instructions from turns
        instructions = []
        for turn in example['turns']:
            if 'instruction' in turn:
                instructions.append(turn['instruction'])

        # Get final query
        final_query = ""
        for turn in example['turns']:
            if 'query' in turn:
                final_query = turn['query']
                break

        # Format full context
        full_context = format_conversation_context(example['turns'])

        # Compress context (50%)
        compressed_turns = compress_turns(
            example['turns'],
            compression_ratio=compression_ratio,
            method='random'
        )
        compressed_context = format_conversation_context(compressed_turns)

        print(f"  Full context: {len(full_context)} chars")
        print(f"  Compressed: {len(compressed_context)} chars ({len(compressed_context)/len(full_context)*100:.1f}%)")

        # Generate with FULL context
        print("  Generating with FULL context...")
        start_time = time.time()
        full_response = generate_response(clients, full_context)
        full_gen_time = time.time() - start_time
        print(f"    ✓ Generated in {full_gen_time:.2f}s ({len(full_response)} chars)")

        # Generate with COMPRESSED context
        print("  Generating with COMPRESSED context...")
        start_time = time.time()
        compressed_response = generate_response(clients, compressed_context)
        compressed_gen_time = time.time() - start_time
        print(f"    ✓ Generated in {compressed_gen_time:.2f}s ({len(compressed_response)} chars)")

        # Evaluate both responses against all instructions
        instruction_evaluations = []

        for instruction in instructions:
            # Evaluate full response
            full_eval = evaluator.evaluate(instruction, full_response)

            # Evaluate compressed response
            compressed_eval = evaluator.evaluate(instruction, compressed_response)

            instruction_evaluations.append({
                'instruction': instruction,
                'full_score': full_eval.combined_score,
                'compressed_score': compressed_eval.combined_score,
                'full_rule_score': full_eval.rule_score,
                'compressed_rule_score': compressed_eval.rule_score,
                'full_llm_score': full_eval.llm_score,
                'compressed_llm_score': compressed_eval.llm_score
            })

        # Calculate average scores for this example
        avg_full_score = statistics.mean([e['full_score'] for e in instruction_evaluations])
        avg_compressed_score = statistics.mean([e['compressed_score'] for e in instruction_evaluations])

        print(f"  Evaluation:")
        print(f"    Full context score: {avg_full_score:.3f}")
        print(f"    Compressed score: {avg_compressed_score:.3f}")
        print(f"    Degradation: {(avg_full_score - avg_compressed_score):.3f} ({((avg_full_score - avg_compressed_score)/avg_full_score*100):.1f}%)")

        # Store results
        results.append({
            'id': example['id'],
            'conflict_type': example['conflict_type'],
            'domain': example['domain'],
            'full_context': full_context,
            'compressed_context': compressed_context,
            'full_response': full_response,
            'compressed_response': compressed_response,
            'full_gen_time': full_gen_time,
            'compressed_gen_time': compressed_gen_time,
            'instruction_evaluations': instruction_evaluations,
            'avg_full_score': avg_full_score,
            'avg_compressed_score': avg_compressed_score,
            'degradation': avg_full_score - avg_compressed_score,
            'degradation_pct': ((avg_full_score - avg_compressed_score) / avg_full_score * 100) if avg_full_score > 0 else 0
        })

    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to {results_path}...")
    save_json(results, results_path)

    return results


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze experiment results and compute statistics.

    Args:
        results: List of experiment results

    Returns:
        Dictionary with analysis results
    """
    print_section("Statistical Analysis")

    # Overall statistics
    full_scores = [r['avg_full_score'] for r in results]
    compressed_scores = [r['avg_compressed_score'] for r in results]
    degradations = [r['degradation'] for r in results]
    degradation_pcts = [r['degradation_pct'] for r in results]

    print("Overall Statistics:")
    print(f"  Full context mean: {statistics.mean(full_scores):.3f} ± {statistics.stdev(full_scores):.3f}")
    print(f"  Compressed mean: {statistics.mean(compressed_scores):.3f} ± {statistics.stdev(compressed_scores):.3f}")
    print(f"  Mean degradation: {statistics.mean(degradations):.3f} ({statistics.mean(degradation_pcts):.1f}%)")

    # Paired t-test
    from scipy import stats
    t_statistic, p_value = stats.ttest_rel(full_scores, compressed_scores)

    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✓ Statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Not statistically significant (p >= 0.05)")

    # By conflict type
    conflict_types = {}
    for r in results:
        ct = r['conflict_type']
        if ct not in conflict_types:
            conflict_types[ct] = {
                'full_scores': [],
                'compressed_scores': [],
                'degradations': []
            }
        conflict_types[ct]['full_scores'].append(r['avg_full_score'])
        conflict_types[ct]['compressed_scores'].append(r['avg_compressed_score'])
        conflict_types[ct]['degradations'].append(r['degradation'])

    print(f"\nBy Conflict Type:")
    conflict_stats = []
    for ct, data in sorted(conflict_types.items()):
        mean_full = statistics.mean(data['full_scores'])
        mean_compressed = statistics.mean(data['compressed_scores'])
        mean_degradation = statistics.mean(data['degradations'])

        std_full = statistics.stdev(data['full_scores']) if len(data['full_scores']) > 1 else 0
        std_compressed = statistics.stdev(data['compressed_scores']) if len(data['compressed_scores']) > 1 else 0

        print(f"\n  {ct}:")
        print(f"    Full: {mean_full:.3f} ± {std_full:.3f}")
        print(f"    Compressed: {mean_compressed:.3f} ± {std_compressed:.3f}")
        print(f"    Degradation: {mean_degradation:.3f}")

        conflict_stats.append({
            'conflict_type': ct,
            'count': len(data['full_scores']),
            'mean_full': mean_full,
            'std_full': std_full,
            'mean_compressed': mean_compressed,
            'std_compressed': std_compressed,
            'mean_degradation': mean_degradation
        })

    # Check success criteria
    print(f"\n{'='*60}")
    print("Success Criteria:")
    full_mean = statistics.mean(full_scores)
    compressed_mean = statistics.mean(compressed_scores)

    print(f"  Full context mean > 0.6: {full_mean:.3f} {'✓' if full_mean > 0.6 else '✗'}")
    print(f"  Compressed mean < 0.5: {compressed_mean:.3f} {'✓' if compressed_mean < 0.5 else '✗'}")
    print(f"  p-value < 0.05: {p_value:.6f} {'✓' if p_value < 0.05 else '✗'}")

    all_criteria_met = full_mean > 0.6 and compressed_mean < 0.5 and p_value < 0.05
    print(f"\n  {'✓ ALL CRITERIA MET' if all_criteria_met else '✗ SOME CRITERIA NOT MET'}")

    return {
        'overall': {
            'full_mean': statistics.mean(full_scores),
            'full_std': statistics.stdev(full_scores),
            'compressed_mean': statistics.mean(compressed_scores),
            'compressed_std': statistics.stdev(compressed_scores),
            'mean_degradation': statistics.mean(degradations),
            'mean_degradation_pct': statistics.mean(degradation_pcts)
        },
        'statistical_test': {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'by_conflict_type': conflict_stats,
        'success_criteria': {
            'full_mean_gt_0.6': full_mean > 0.6,
            'compressed_mean_lt_0.5': compressed_mean < 0.5,
            'p_value_lt_0.05': p_value < 0.05,
            'all_met': all_criteria_met
        }
    }


def main():
    """Main experiment runner"""
    # Run experiment
    results = run_experiment(
        test_data_path="data/test.json",
        results_path="results/exp1_results.json",
        compression_ratio=0.5,
        max_examples=20  # Use all 20 examples
    )

    # Analyze results
    analysis = analyze_results(results)

    # Save analysis
    save_json(analysis, "results/exp1_analysis.json")

    print(f"\n{'='*60}")
    print("Experiment 1 Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    print(f"  - results/exp1_results.json")
    print(f"  - results/exp1_analysis.json")
    print(f"\nNext steps:")
    print(f"  - Generate visualizations (Figure 1)")
    print(f"  - Generate tables (Table 1)")
    print(f"  - Create DAY_5_STATUS.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
