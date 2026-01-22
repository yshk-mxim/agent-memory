#!/usr/bin/env python3
"""
Generate batch 002 (35 examples) for RDIC dataset.
Continues from batch 001 with IDs conflict_031 to conflict_065.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_generator import ConflictDatasetGenerator

def main():
    generator = ConflictDatasetGenerator()

    # Load schema for conflict types
    schema_path = Path(__file__).parent / 'data' / 'conflict_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Target: 35 examples, 7 per conflict type (5 types)
    conflict_types = list(schema['conflict_types'].keys())
    domains = schema['domains']

    examples = []
    example_id = 31  # Starting from conflict_031

    print(f"Generating batch 002 (35 examples)...")
    print(f"Starting ID: conflict_{example_id:03d}")
    print(f"Target: 7 examples per conflict type")
    print()

    # Generate 7 examples per type
    for conflict_type in conflict_types:
        subtypes = schema['conflict_types'][conflict_type]['subtypes']

        print(f"Generating {conflict_type} conflicts...")

        for i in range(7):
            # Cycle through subtypes and domains
            subtype = subtypes[i % len(subtypes)]
            domain = domains[i % len(domains)]

            conflict_name = f"{conflict_type}_{subtype}"

            print(f"  [{example_id:03d}] {conflict_name} ({domain})...", end='', flush=True)

            try:
                example = generator.generate_example(
                    conflict_type=conflict_type,
                    conflict_subtype=subtype,
                    domain=domain
                )

                # Set ID
                example['id'] = f"conflict_{example_id:03d}"

                examples.append(example)
                print(" ✓")

            except Exception as e:
                print(f" ✗ Error: {e}")
                continue

            example_id += 1

    # Save batch
    output_path = Path(__file__).parent / 'data' / 'batch_002.json'
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print()
    print(f"✓ Generated {len(examples)} examples")
    print(f"✓ Saved to {output_path}")
    print()

    # Distribution summary
    type_counts = {}
    for ex in examples:
        conflict_type = ex['conflict_type'].split('_')[0]
        type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1

    print("Distribution:")
    for ctype, count in sorted(type_counts.items()):
        print(f"  {ctype}: {count}")

    print()
    print("Next: Run generate_batch_003.py to generate batch 3")

if __name__ == '__main__':
    main()
