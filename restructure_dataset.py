"""
Restructure dataset from unresolvable conflicts to context isolation testing.

OLD: Turn 3 asks for impossible merge of incompatible constraints
NEW: Turn 3 tests whether system maintains separate contexts across turns
"""

import json
from datetime import datetime


def get_context_test_query(conflict_type, domain, turn1_instruction, turn2_instruction):
    """
    Generate a Turn 3 query that tests context isolation.

    Instead of asking for both constraints in one response,
    ask for something that requires the system to maintain
    both contexts separately without degradation.
    """

    # Extract the conflict category
    conflict_category = conflict_type.split('_')[0]

    templates = {
        'tone': {
            'query': "Now provide two versions: first using the professional approach from earlier, then using the casual approach. Show me both.",
            'expected': f"System should maintain both {turn1_instruction.split(',')[0]} context and {turn2_instruction.split(',')[0]} context separately. Under KV cache compression, the earlier formal instruction often degrades when casual instruction appears, causing the system to forget the formal constraint. RDIC should isolate these semantic contexts to prevent degradation."
        },
        'detail': {
            'query': "Give me both versions we discussed: the brief version first, then the comprehensive version.",
            'expected': f"System should maintain both contexts: the {turn1_instruction.split()[0].lower()} constraint and the {turn2_instruction.split()[0].lower()} constraint. KV compression typically causes the earlier brevity instruction to degrade when detailed instruction appears. RDIC isolates these to prevent instruction loss."
        },
        'style': {
            'query': "Show me both approaches: first the technical version, then the accessible version.",
            'expected': f"System should maintain separate contexts for technical and accessible writing styles. KV cache compression often causes style instructions to interfere with each other. RDIC prevents this by isolating semantic style clusters."
        },
        'content': {
            'query': "Provide both versions: one following the first approach, one following the second approach.",
            'expected': f"System should maintain both content instruction contexts separately. Compression causes later content instructions to override earlier ones. RDIC isolates content constraint clusters to preserve both."
        },
        'format': {
            'query': "Show both formats: the structured version first, then the narrative version.",
            'expected': f"System should maintain both format contexts: structured and narrative. KV compression causes format instructions to degrade when conflicting formats appear. RDIC isolates format clusters to prevent degradation."
        }
    }

    template = templates.get(conflict_category, templates['tone'])
    return template['query'], template['expected']


def restructure_example(example):
    """Restructure a single example to test context isolation"""

    conflict_type = example['conflict_type']
    domain = example['domain']

    # Keep turns 1 and 2 as-is
    turns = example['turns'][:2]

    # Extract instruction info from turns 1 and 2
    turn1_instruction = turns[0].get('instruction', '')
    turn2_instruction = turns[1].get('instruction', '')

    # Generate new Turn 3 that tests context isolation
    new_query, new_expected_conflict = get_context_test_query(
        conflict_type,
        domain,
        turn1_instruction,
        turn2_instruction
    )

    # Create new Turn 3
    turn3 = {
        "turn_id": 3,
        "role": "user",
        "query": new_query,
        "expected_behavior": new_expected_conflict,
        "rdic_value": "Maintains isolated KV contexts for each instruction cluster, preventing compression-based degradation when conflicting instructions appear across turns"
    }

    turns.append(turn3)

    # Update the example
    example['turns'] = turns
    example['purpose'] = "context_isolation_test"
    example['metadata']['restructured_at'] = datetime.now().isoformat()
    example['metadata']['original_structure'] = "unresolvable_merge"
    example['metadata']['new_structure'] = "context_isolation"

    return example


def main():
    # Load original batch
    with open('data/batch_001.json', 'r') as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")
    print("Restructuring from unresolvable merges to context isolation tests...")

    # Restructure each example
    restructured = []
    for example in examples:
        try:
            new_example = restructure_example(example)
            restructured.append(new_example)
            print(f"  ✓ {example['id']}: {example['conflict_type']}")
        except Exception as e:
            print(f"  ✗ {example['id']}: Error - {e}")
            # Keep original on error
            restructured.append(example)

    # Save restructured batch
    output_path = 'data/batch_001_restructured.json'
    with open(output_path, 'w') as f:
        json.dump(restructured, f, indent=2)

    print(f"\n✓ Restructured {len(restructured)} examples")
    print(f"✓ Saved to: {output_path}")

    # Show example transformation
    print("\n" + "="*80)
    print("EXAMPLE TRANSFORMATION")
    print("="*80)

    original = examples[0]
    new = restructured[0]

    print(f"\nOriginal Turn 3 (unresolvable):")
    print(f"  Query: {original['turns'][2].get('query', 'N/A')[:100]}...")

    print(f"\nNew Turn 3 (context isolation test):")
    print(f"  Query: {new['turns'][2].get('query', 'N/A')}")
    print(f"  Expected: {new['turns'][2].get('expected_behavior', 'N/A')[:150]}...")

    print("\n" + "="*80)
    print("\nNext steps:")
    print("1. Review data/batch_001_restructured.json")
    print("2. If approved, replace batch_001.json with restructured version")
    print("3. Update batch_001_review.md to reflect new structure")


if __name__ == "__main__":
    main()
