"""
Dataset generator for instruction conflict examples.

Generates multi-turn conversations with conflicting instructions using Claude API.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

from .config import get_config


class ConflictDatasetGenerator:
    """Generate instruction conflict examples for RDIC training"""

    def __init__(self, schema_path: str = None):
        """
        Initialize the dataset generator.

        Args:
            schema_path: Path to conflict_schema.json
        """
        self.config = get_config()
        self.client = Anthropic(api_key=self.config.claude_api_key)

        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "data" / "conflict_schema.json"

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        self.conflict_types = self.schema['conflict_types']
        self.domains = self.schema['domains']

    def _get_generation_prompt(self, conflict_type: str, conflict_subtype: str, domain: str) -> str:
        """
        Create a prompt for generating a specific conflict example.

        Args:
            conflict_type: Type of conflict (tone, detail, style, content, format)
            conflict_subtype: Specific subtype (e.g., formal_vs_casual)
            domain: Domain for the example (e.g., business_email)

        Returns:
            Prompt string for Claude API
        """
        conflict_info = self.conflict_types[conflict_type]

        prompt = f"""Generate a realistic multi-turn conversation that demonstrates an instruction conflict.

**Conflict Type:** {conflict_info['name']}
**Specific Conflict:** {conflict_subtype}
**Domain:** {domain}
**Description:** {conflict_info['description']}

Create a JSON object with the following structure:

{{
  "id": "conflict_XXX",
  "conflict_type": "{conflict_type}_{conflict_subtype}",
  "domain": "{domain}",
  "turns": [
    {{
      "turn_id": 1,
      "role": "user",
      "instruction": "<first instruction that sets a constraint>",
      "content": "<user message establishing context>"
    }},
    {{
      "turn_id": 2,
      "role": "user",
      "instruction": "<second instruction that conflicts with the first>",
      "content": "<user message with new context or request>"
    }},
    {{
      "turn_id": 3,
      "role": "user",
      "query": "<final query that requires resolving both conflicting instructions>",
      "expected_conflict": "<clear description of why these instructions conflict>"
    }}
  ],
  "ground_truth_clusters": ["<semantic cluster 1>", "<semantic cluster 2>"]
}}

**Requirements:**
1. The conflict must be GENUINE - the two instructions should be truly incompatible
2. The conflict should be REALISTIC - something that could happen in real conversations
3. The final query should make the conflict UNAVOIDABLE - it must require both constraints
4. Ground truth clusters should represent the incompatible semantic spaces
5. Make the scenario specific to the {domain} domain
6. The conflict should specifically demonstrate {conflict_subtype.replace('_', ' ')}

**Example for tone conflict (formal_vs_casual) in business_email domain:**
{{
  "id": "conflict_001",
  "conflict_type": "tone_formal_vs_casual",
  "domain": "business_email",
  "turns": [
    {{
      "turn_id": 1,
      "role": "user",
      "instruction": "Always use formal, professional language with proper business etiquette",
      "content": "I need help drafting emails to our enterprise clients about the Q4 contract renewal"
    }},
    {{
      "turn_id": 2,
      "role": "user",
      "instruction": "Actually, write in a casual, friendly tone like you're texting a close friend",
      "content": "Draft an email to Anderson Corp about their delayed payment"
    }},
    {{
      "turn_id": 3,
      "role": "user",
      "query": "Write the email now, following both the formal professional style and the casual friendly tone",
      "expected_conflict": "Cannot simultaneously maintain formal business etiquette and casual texting style - they are mutually exclusive communication registers"
    }}
  ],
  "ground_truth_clusters": ["formal_professional_constrained", "casual_friendly_creative"]
}}

Return ONLY the JSON object, no additional text or explanation."""

        return prompt

    def generate_example(
        self,
        conflict_type: str,
        conflict_subtype: str,
        domain: str,
        model: str = "claude-sonnet-4-5-20250929"
    ) -> Dict[str, Any]:
        """
        Generate a single conflict example.

        Args:
            conflict_type: Type of conflict
            conflict_subtype: Specific subtype
            domain: Domain for the example
            model: Claude model to use

        Returns:
            Generated example as dictionary
        """
        prompt = self._get_generation_prompt(conflict_type, conflict_subtype, domain)

        response = self.client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.8,  # Higher temp for more diverse examples
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract JSON from response
        content = response.content[0].text

        # Try to parse JSON - handle cases where there might be markdown code blocks
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            example = json.loads(content)

            # Add metadata
            example['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'model': model,
                'reviewed': False,
                'quality_score': None
            }

            return example

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw content: {content}")
            raise

    def generate_batch(
        self,
        num_examples: int = 30,
        output_path: str = None,
        distribute_evenly: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of conflict examples.

        Args:
            num_examples: Number of examples to generate
            output_path: Path to save the batch (optional)
            distribute_evenly: Whether to distribute across conflict types evenly

        Returns:
            List of generated examples
        """
        examples = []

        if distribute_evenly:
            # Calculate how many examples per conflict type
            conflict_type_keys = list(self.conflict_types.keys())
            examples_per_type = num_examples // len(conflict_type_keys)
            remainder = num_examples % len(conflict_type_keys)

            for idx, conflict_type in enumerate(conflict_type_keys):
                subtypes = self.conflict_types[conflict_type]['subtypes']

                # Add extra examples to first types if there's a remainder
                num_for_this_type = examples_per_type + (1 if idx < remainder else 0)

                for i in range(num_for_this_type):
                    subtype = random.choice(subtypes)
                    domain = random.choice(self.domains)

                    print(f"Generating example {len(examples)+1}/{num_examples}: "
                          f"{conflict_type}/{subtype} in {domain}")

                    try:
                        example = self.generate_example(conflict_type, subtype, domain)
                        example['id'] = f"conflict_{len(examples)+1:03d}"
                        examples.append(example)
                    except Exception as e:
                        print(f"Error generating example: {e}")
                        continue
        else:
            # Random distribution
            for i in range(num_examples):
                conflict_type = random.choice(list(self.conflict_types.keys()))
                subtype = random.choice(self.conflict_types[conflict_type]['subtypes'])
                domain = random.choice(self.domains)

                print(f"Generating example {i+1}/{num_examples}: "
                      f"{conflict_type}/{subtype} in {domain}")

                try:
                    example = self.generate_example(conflict_type, subtype, domain)
                    example['id'] = f"conflict_{i+1:03d}"
                    examples.append(example)
                except Exception as e:
                    print(f"Error generating example: {e}")
                    continue

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"\nBatch saved to: {output_path}")

        return examples

    def validate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a generated example for quality.

        Args:
            example: Example to validate

        Returns:
            Validation results with scores and issues
        """
        issues = []

        # Check required fields
        required_fields = ['id', 'conflict_type', 'domain', 'turns', 'ground_truth_clusters']
        for field in required_fields:
            if field not in example:
                issues.append(f"Missing required field: {field}")

        # Check turns structure
        if 'turns' in example:
            if len(example['turns']) < 2:
                issues.append("Need at least 2 turns to have a conflict")

            for turn in example['turns']:
                if 'turn_id' not in turn or 'role' not in turn:
                    issues.append("Turn missing turn_id or role")

                # Check that at least first two turns have instructions
                if turn.get('turn_id', 0) <= 2 and 'instruction' not in turn:
                    issues.append(f"Turn {turn.get('turn_id')} missing instruction")

        # Check ground truth clusters
        if 'ground_truth_clusters' in example:
            if len(example['ground_truth_clusters']) < 2:
                issues.append("Need at least 2 ground truth clusters for a conflict")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_issues': len(issues)
        }


def main():
    """Main function for testing dataset generation"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate instruction conflict dataset')
    parser.add_argument('--num', type=int, default=30, help='Number of examples to generate')
    parser.add_argument('--output', type=str, default='data/batch_001.json',
                       help='Output file path')
    parser.add_argument('--validate', action='store_true',
                       help='Validate examples after generation')

    args = parser.parse_args()

    # Create generator
    generator = ConflictDatasetGenerator()

    # Generate batch
    print(f"Generating {args.num} examples...")
    examples = generator.generate_batch(
        num_examples=args.num,
        output_path=args.output
    )

    print(f"\nGenerated {len(examples)} examples")

    # Validate if requested
    if args.validate:
        print("\nValidating examples...")
        valid_count = 0
        for example in examples:
            result = generator.validate_example(example)
            if result['valid']:
                valid_count += 1
            else:
                print(f"Example {example['id']} has issues: {result['issues']}")

        print(f"\n{valid_count}/{len(examples)} examples are valid "
              f"({100*valid_count/len(examples):.1f}%)")


if __name__ == "__main__":
    main()
