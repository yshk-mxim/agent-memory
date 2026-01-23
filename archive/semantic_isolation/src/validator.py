"""
Validation pipeline for RDIC conflict dataset.

Validates structural correctness, field requirements, and context isolation design.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ConflictDatasetValidator:
    """Validates instruction conflict examples for RDIC"""

    def __init__(self, schema_path: str = None):
        """
        Initialize validator with schema.

        Args:
            schema_path: Path to conflict_schema.json
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / 'data' / 'conflict_schema.json'

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        self.valid_conflict_types = list(self.schema['conflict_types'].keys())
        self.valid_domains = self.schema['domains']

    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single example.

        Args:
            example: Example dict to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required top-level fields
        required_fields = ['id', 'conflict_type', 'domain', 'turns', 'ground_truth_clusters', 'purpose']
        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate ID format
        if not example['id'].startswith('conflict_'):
            errors.append(f"Invalid ID format: {example['id']} (should start with 'conflict_')")

        # Validate conflict type
        conflict_base = example['conflict_type'].split('_')[0]
        if conflict_base not in self.valid_conflict_types:
            errors.append(f"Invalid conflict type: {conflict_base}")

        # Validate domain
        if example['domain'] not in self.valid_domains:
            errors.append(f"Invalid domain: {example['domain']}")

        # Validate purpose
        if example['purpose'] != 'context_isolation_test':
            errors.append(f"Invalid purpose: {example['purpose']} (should be 'context_isolation_test')")

        # Validate turns structure
        if not isinstance(example['turns'], list) or len(example['turns']) != 3:
            errors.append(f"Invalid turns: expected list of 3 turns, got {len(example.get('turns', []))}")
        else:
            # Validate each turn
            for i, turn in enumerate(example['turns'], 1):
                turn_errors = self._validate_turn(turn, i)
                errors.extend(turn_errors)

        # Validate ground truth clusters
        if not isinstance(example['ground_truth_clusters'], list) or len(example['ground_truth_clusters']) != 2:
            errors.append(f"Invalid ground_truth_clusters: expected list of 2 clusters")

        # Validate metadata (if present)
        if 'metadata' in example:
            if not isinstance(example['metadata'], dict):
                errors.append("Invalid metadata: should be dict")

        return len(errors) == 0, errors

    def _validate_turn(self, turn: Dict[str, Any], turn_number: int) -> List[str]:
        """Validate a single turn"""
        errors = []

        # Check required fields
        required = ['turn_id', 'role']
        for field in required:
            if field not in turn:
                errors.append(f"Turn {turn_number}: Missing required field '{field}'")

        if errors:
            return errors

        # Validate turn_id matches expected
        if turn['turn_id'] != turn_number:
            errors.append(f"Turn {turn_number}: turn_id mismatch (expected {turn_number}, got {turn['turn_id']})")

        # Validate role
        if turn['role'] not in ['user', 'assistant']:
            errors.append(f"Turn {turn_number}: Invalid role '{turn['role']}'")

        # Turn-specific validation
        if turn_number in [1, 2]:
            # Turn 1 and 2 should have instruction and content
            if 'instruction' not in turn:
                errors.append(f"Turn {turn_number}: Missing 'instruction' field")
            if 'content' not in turn:
                errors.append(f"Turn {turn_number}: Missing 'content' field")

        elif turn_number == 3:
            # Turn 3 should have query and context isolation fields
            if 'query' not in turn:
                errors.append(f"Turn 3: Missing 'query' field")
            else:
                # Check that query requests "both versions" (context isolation test)
                query_lower = turn['query'].lower()
                if not any(phrase in query_lower for phrase in ['both versions', 'both:', 'first', 'then']):
                    errors.append(f"Turn 3: Query doesn't appear to request both versions (context isolation)")

            if 'expected_behavior' not in turn:
                errors.append(f"Turn 3: Missing 'expected_behavior' field")

            if 'rdic_value' not in turn:
                errors.append(f"Turn 3: Missing 'rdic_value' field")

        return errors

    def validate_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of examples.

        Args:
            examples: List of examples to validate

        Returns:
            Validation report dict
        """
        total = len(examples)
        valid_count = 0
        errors_by_example = {}

        for example in examples:
            is_valid, errors = self.validate_example(example)
            if is_valid:
                valid_count += 1
            else:
                errors_by_example[example.get('id', 'unknown')] = errors

        # Compute statistics
        conflict_distribution = {}
        domain_distribution = {}

        for example in examples:
            # Count conflict types
            conflict_base = example.get('conflict_type', '').split('_')[0]
            conflict_distribution[conflict_base] = conflict_distribution.get(conflict_base, 0) + 1

            # Count domains
            domain = example.get('domain', 'unknown')
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        report = {
            'total_examples': total,
            'valid_examples': valid_count,
            'invalid_examples': total - valid_count,
            'validation_rate': valid_count / total if total > 0 else 0,
            'errors_by_example': errors_by_example,
            'conflict_distribution': conflict_distribution,
            'domain_distribution': domain_distribution
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print validation report"""
        print("=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print()
        print(f"Total examples: {report['total_examples']}")
        print(f"Valid examples: {report['valid_examples']} ({report['validation_rate']*100:.1f}%)")
        print(f"Invalid examples: {report['invalid_examples']}")
        print()

        if report['errors_by_example']:
            print("ERRORS:")
            for example_id, errors in report['errors_by_example'].items():
                print(f"\n{example_id}:")
                for error in errors:
                    print(f"  - {error}")
            print()

        print("CONFLICT TYPE DISTRIBUTION:")
        for ctype, count in sorted(report['conflict_distribution'].items()):
            print(f"  {ctype}: {count}")
        print()

        print("DOMAIN DISTRIBUTION:")
        for domain, count in sorted(report['domain_distribution'].items()):
            print(f"  {domain}: {count}")
        print()


def validate_dataset(dataset_path: str, schema_path: str = None) -> Dict[str, Any]:
    """
    Validate a dataset file.

    Args:
        dataset_path: Path to dataset JSON file
        schema_path: Optional path to schema

    Returns:
        Validation report
    """
    with open(dataset_path, 'r') as f:
        examples = json.load(f)

    validator = ConflictDatasetValidator(schema_path)
    report = validator.validate_batch(examples)

    return report


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validator.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    report = validate_dataset(dataset_path)

    validator = ConflictDatasetValidator()
    validator.print_report(report)

    # Exit with error code if validation failed
    if report['invalid_examples'] > 0:
        sys.exit(1)
