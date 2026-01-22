#!/usr/bin/env python3
"""
Combine all batches into complete dataset and create train/test split.
"""

import json
import random
from pathlib import Path


def main():
    data_dir = Path(__file__).parent / 'data'

    # Load all batches
    all_examples = []

    for batch_num in [1, 2, 3]:
        batch_file = data_dir / f'batch_{batch_num:03d}.json'
        print(f"Loading {batch_file.name}...")

        with open(batch_file, 'r') as f:
            batch = json.load(f)
            all_examples.extend(batch)
            print(f"  ✓ Loaded {len(batch)} examples")

    print(f"\n✓ Total examples: {len(all_examples)}")

    # Save complete dataset
    output_file = data_dir / 'conflict_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2)

    print(f"✓ Saved complete dataset to {output_file.name}")

    # Create train/test split (80/20)
    random.seed(42)  # For reproducibility
    shuffled = all_examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]

    # Save train set
    train_file = data_dir / 'train.json'
    with open(train_file, 'w') as f:
        json.dump(train_set, f, indent=2)
    print(f"✓ Saved train set to {train_file.name} ({len(train_set)} examples)")

    # Save test set
    test_file = data_dir / 'test.json'
    with open(test_file, 'w') as f:
        json.dump(test_set, f, indent=2)
    print(f"✓ Saved test set to {test_file.name} ({len(test_set)} examples)")

    # Compute statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    # Conflict type distribution
    conflict_dist = {}
    for ex in all_examples:
        ctype = ex['conflict_type'].split('_')[0]
        conflict_dist[ctype] = conflict_dist.get(ctype, 0) + 1

    print("\nConflict Type Distribution (Total):")
    for ctype, count in sorted(conflict_dist.items()):
        print(f"  {ctype}: {count}")

    # Domain distribution
    domain_dist = {}
    for ex in all_examples:
        domain = ex['domain']
        domain_dist[domain] = domain_dist.get(domain, 0) + 1

    print("\nDomain Distribution (Total):")
    for domain, count in sorted(domain_dist.items()):
        print(f"  {domain}: {count}")

    # Train/test distribution
    train_conflict_dist = {}
    for ex in train_set:
        ctype = ex['conflict_type'].split('_')[0]
        train_conflict_dist[ctype] = train_conflict_dist.get(ctype, 0) + 1

    test_conflict_dist = {}
    for ex in test_set:
        ctype = ex['conflict_type'].split('_')[0]
        test_conflict_dist[ctype] = test_conflict_dist.get(ctype, 0) + 1

    print("\nConflict Distribution (Train/Test):")
    print("  Type          Train  Test")
    print("  " + "-"*30)
    for ctype in sorted(conflict_dist.keys()):
        train_count = train_conflict_dist.get(ctype, 0)
        test_count = test_conflict_dist.get(ctype, 0)
        print(f"  {ctype:<12} {train_count:>5}  {test_count:>4}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples: {len(all_examples)}")
    print(f"Train examples: {len(train_set)} (80%)")
    print(f"Test examples: {len(test_set)} (20%)")
    print(f"Conflict types: {len(conflict_dist)}")
    print(f"Domains: {len(domain_dist)}")
    print("\nFiles created:")
    print(f"  - {output_file.name}")
    print(f"  - {train_file.name}")
    print(f"  - {test_file.name}")


if __name__ == '__main__':
    main()
