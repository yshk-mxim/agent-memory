import json

# Load the batch
with open('data/batch_001.json', 'r') as f:
    data = json.load(f)

# Extract first 10 examples for review
review_examples = data[:10]

# Format for manual review
print("=" * 80)
print("DAY 2 MANUAL REVIEW - 10 Examples to Check")
print("=" * 80)
print()

for idx, example in enumerate(review_examples, 1):
    print(f"\n{'='*80}")
    print(f"EXAMPLE {idx}: {example['id']}")
    print(f"{'='*80}")
    print(f"Conflict Type: {example['conflict_type']}")
    print(f"Domain: {example['domain']}")
    print()

    for turn in example['turns']:
        print(f"Turn {turn['turn_id']} ({turn['role']}):")
        if 'instruction' in turn:
            print(f"  INSTRUCTION: {turn['instruction']}")
        if 'content' in turn:
            print(f"  CONTENT: {turn['content']}")
        if 'query' in turn:
            print(f"  QUERY: {turn['query']}")
        if 'expected_conflict' in turn:
            print(f"  EXPECTED CONFLICT: {turn['expected_conflict']}")
        if 'expected_behavior' in turn:
            print(f"  WHAT THIS TESTS: {turn['expected_behavior']}")
        if 'rdic_value' in turn:
            print(f"  RDIC VALUE: {turn['rdic_value']}")
        print()

    print(f"Ground Truth Clusters: {example['ground_truth_clusters']}")
    print(f"Purpose: {example.get('purpose', 'N/A')}")
    print()
    print("Review Checklist (Context Isolation Testing):")
    print("  [ ] 1. Genuine Conflict - Do the two instructions create incompatible semantic contexts?")
    print("  [ ] 2. Realistic Scenario - Could this context-switching happen in real conversations?")
    print("  [ ] 3. Tests Context Isolation - Does Turn 3 require maintaining BOTH contexts separately?")
    print("  [ ] 4. Clear Ground Truth - Are the semantic clusters well-separated?")
    print("  [ ] 5. RDIC Value - Would isolating KV contexts prevent instruction degradation here?")
    print()
    print("OVERALL: PASS / FAIL")
    print("Notes: _________________________________________________________________")
    print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Count how many examples PASS all 4 criteria:")
print("  - Need ≥7/10 to meet 70% target")
print("  - Each example must pass ALL 4 criteria to count as PASS")
print("="*80)
