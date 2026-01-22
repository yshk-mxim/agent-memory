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
        print()

    print(f"Ground Truth Clusters: {example['ground_truth_clusters']}")
    print()
    print("Review Checklist:")
    print("  [ ] 1. Genuine Conflict - Are instructions truly incompatible?")
    print("  [ ] 2. Realistic Scenario - Could this happen in real life?")
    print("  [ ] 3. Unavoidable Conflict - Does final query require BOTH constraints?")
    print("  [ ] 4. Clear Ground Truth - Are semantic clusters well-defined?")
    print()
    print("PASS/FAIL: _______  Notes: _________________________________")
    print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Count how many examples PASS all 4 criteria:")
print("  - Need ≥7/10 to meet 70% target")
print("  - Each example must pass ALL 4 criteria to count as PASS")
print("="*80)
