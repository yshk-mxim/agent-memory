#!/usr/bin/env python3
"""
Add missing 'purpose' field to batch 002 and 003.
"""

import json
from pathlib import Path

def add_purpose_field(batch_path: Path):
    """Add purpose field to all examples in batch"""
    with open(batch_path, 'r') as f:
        examples = json.load(f)

    # Add purpose field to each example
    for example in examples:
        if 'purpose' not in example:
            example['purpose'] = 'context_isolation_test'

    # Save updated batch
    with open(batch_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"✓ Added 'purpose' field to {len(examples)} examples in {batch_path.name}")

def main():
    data_dir = Path(__file__).parent / 'data'

    # Fix batch 002 and 003
    for batch_file in ['batch_002.json', 'batch_003.json']:
        batch_path = data_dir / batch_file
        if batch_path.exists():
            add_purpose_field(batch_path)
        else:
            print(f"⚠ {batch_file} not found")

if __name__ == '__main__':
    main()
