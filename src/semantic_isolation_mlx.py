#!/usr/bin/env python3
"""
Semantic KV Cache Isolation using MLX Framework.

TRUE KV cache isolation for semantic instruction clustering on Apple Silicon.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class IsolationResult:
    """Results from one isolation condition."""
    outputs: Dict[str, str]
    cache_sizes: Dict[str, int]
    total_tokens_processed: int
    generation_time: float


class SemanticIsolationTesterMLX:
    """
    Test semantic KV cache isolation using MLX.

    Implements TRUE cache isolation by maintaining separate cache objects
    for each semantic cluster.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-12b-it-4bit",
        random_seed: int = 42,
        use_neutral_prompts: bool = True
    ):
        """
        Initialize with MLX model.

        Args:
            model_name: MLX model identifier
            random_seed: Random seed for reproducibility
            use_neutral_prompts: If True, use neutral prompts
        """
        print(f"Loading {model_name}...")

        self.random_seed = random_seed
        self.use_neutral_prompts = use_neutral_prompts

        # Set random seed
        mx.random.seed(random_seed)

        # Load model and tokenizer
        self.model, self.tokenizer = load(model_name)

        print("✓ Model loaded successfully")
        print(f"  Random seed: {random_seed}")
        print(f"  Neutral prompts: {use_neutral_prompts}")

    def get_cache_size(self, cache: Optional[List]) -> int:
        """
        Get size of KV cache in tokens.

        Args:
            cache: MLX cache (list of (keys, values) per layer)

        Returns:
            Number of tokens in cache
        """
        if cache is None or len(cache) == 0:
            return 0
        # cache[0] = (keys, values) for first layer
        # keys shape: [batch, seq_len, num_heads, head_dim]
        return cache[0][0].shape[1]

    def build_context_for_turns(
        self,
        turns: List[Dict[str, Any]]
    ) -> Tuple[str, int]:
        """
        Build context text from turns.

        Args:
            turns: List of turn dictionaries with 'instruction' or 'content' fields

        Returns:
            Tuple of (context_text, total_tokens)
        """
        # Extract text from each turn
        turn_texts = []
        for turn in turns:
            # Get text from either instruction or content field
            text = turn.get("instruction") or turn.get("content") or ""
            if text:
                turn_texts.append(text)

        # Build full context string
        full_context = "\n\n".join(turn_texts)
        tokens = self.tokenizer.encode(full_context)
        total_tokens = len(tokens)

        return full_context, total_tokens

    def generate_from_context(
        self,
        context_text: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text given context text and a prompt.

        In MLX, we concatenate context + prompt and generate.

        Args:
            context_text: Context text string
            prompt: Generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Combine context + prompt as text
        full_text = context_text + prompt

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)

        # Generate using MLX
        response = generate(
            self.model,
            self.tokenizer,
            prompt=full_text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )

        # Extract only the generated part (after full_text)
        if response.startswith(full_text):
            return response[len(full_text):]
        return response

    def get_generation_prompts(self, neutral: Optional[bool] = None) -> Dict[str, str]:
        """Get generation prompts (neutral or semantic)."""
        if neutral is None:
            neutral = self.use_neutral_prompts

        if neutral:
            return {
                'technical': "\n\nGenerate output A based on the provided context:",
                'business': "\n\nGenerate output B based on the provided context:",
                'synthesis': "\n\nGenerate output C that integrates outputs A and B:"
            }
        else:
            return {
                'technical': "\n\nBased on the technical performance analysis above, provide recommendations:",
                'business': "\n\nBased on the business strategy discussion above, provide recommendations:",
                'synthesis': "\n\nBased on the technical and business outputs, provide an integrated executive summary:"
            }

    def condition_1_sequential(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 1: Sequential baseline.
        All 15 turns processed with ONE accumulating context.
        """
        start_time = time.time()

        # Build context from all turns (turns array is already in order)
        all_turns = example.get("turns", [])

        context_text, total_tokens = self.build_context_for_turns(all_turns)
        prompts = self.get_generation_prompts()

        # Generate outputs
        output_technical = self.generate_from_context(
            context_text,
            prompts['technical'],
            max_tokens=300
        )

        output_business = self.generate_from_context(
            context_text,
            prompts['business'],
            max_tokens=300
        )

        output_synthesis = self.generate_from_context(
            context_text,
            prompts['synthesis'],
            max_tokens=300
        )

        gen_time = time.time() - start_time

        return IsolationResult(
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'unified': total_tokens},
            total_tokens_processed=total_tokens,
            generation_time=gen_time
        )

    def condition_2_prompted(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 2: Prompted soft isolation.
        Same as sequential but with explicit "keep separate" instruction.
        """
        start_time = time.time()

        # Build context with isolation instruction
        isolation_prompt = "IMPORTANT: Keep the following topics separate: technical performance analysis and business strategy.\n\n"

        all_turns = example.get("turns", []).copy()

        # Add instruction to first turn
        if all_turns:
            first_turn = all_turns[0].copy()
            instruction_text = first_turn.get('instruction') or first_turn.get('content') or ""
            first_turn['instruction'] = isolation_prompt + instruction_text
            first_turn['content'] = None
            all_turns[0] = first_turn

        context_text, total_tokens = self.build_context_for_turns(all_turns)
        prompts = self.get_generation_prompts()

        # Generate outputs
        output_technical = self.generate_from_context(
            context_text,
            prompts['technical'],
            max_tokens=300
        )

        output_business = self.generate_from_context(
            context_text,
            prompts['business'],
            max_tokens=300
        )

        output_synthesis = self.generate_from_context(
            context_text,
            prompts['synthesis'],
            max_tokens=300
        )

        gen_time = time.time() - start_time

        return IsolationResult(
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'unified_with_prompt': total_tokens},
            total_tokens_processed=total_tokens,
            generation_time=gen_time
        )

    def condition_3_turn_based(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 3: Turn-based naive isolation.
        Adds turn markers but still shares one cache.
        """
        start_time = time.time()

        # Build context with turn markers
        all_turns = []
        for turn in example.get("turns", []):
            turn_copy = turn.copy()
            turn_id = turn.get('turn_id', 0)
            text = turn.get('instruction') or turn.get('content') or ""
            turn_copy['instruction'] = f"[Turn {turn_id}]\n{text}"
            turn_copy['content'] = None
            all_turns.append(turn_copy)

        context_text, total_tokens = self.build_context_for_turns(all_turns)
        prompts = self.get_generation_prompts()

        # Generate outputs
        output_technical = self.generate_from_context(
            context_text,
            prompts['technical'],
            max_tokens=300
        )

        output_business = self.generate_from_context(
            context_text,
            prompts['business'],
            max_tokens=300
        )

        output_synthesis = self.generate_from_context(
            context_text,
            prompts['synthesis'],
            max_tokens=300
        )

        gen_time = time.time() - start_time

        return IsolationResult(
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'turn_marked': total_tokens},
            total_tokens_processed=total_tokens,
            generation_time=gen_time
        )

    def condition_4_semantic(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 4: Semantic isolation (RDIC - our method).
        Each semantic cluster gets its OWN isolated context.
        """
        start_time = time.time()

        # Get cluster assignments
        clusters = example.get('semantic_clusters', {})
        cluster_1_turn_ids = clusters.get('cluster_1', {}).get('turns', [])
        cluster_2_turn_ids = clusters.get('cluster_2', {}).get('turns', [])
        cluster_3_turn_ids = clusters.get('cluster_3', {}).get('turns', [])

        # Filter turns by cluster
        all_turns = example.get("turns", [])
        c1_turns = [t for t in all_turns if t.get('turn_id') in cluster_1_turn_ids]
        c2_turns = [t for t in all_turns if t.get('turn_id') in cluster_2_turn_ids]
        c3_turns = [t for t in all_turns if t.get('turn_id') in cluster_3_turn_ids]

        context_text_c1, tokens_c1 = self.build_context_for_turns(c1_turns)
        context_text_c2, tokens_c2 = self.build_context_for_turns(c2_turns)

        prompts = self.get_generation_prompts()

        # Generate from cluster 1 (technical)
        output_technical = self.generate_from_context(
            context_text_c1,
            prompts['technical'],
            max_tokens=300
        )

        # Generate from cluster 2 (business)
        output_business = self.generate_from_context(
            context_text_c2,
            prompts['business'],
            max_tokens=300
        )

        # Cluster 3: synthesis with message passing
        # Build context from cluster 3 turns + outputs from c1 and c2
        message_passing = f"\nOutput A: {output_technical}\n\nOutput B: {output_business}\n\n"
        c3_turns_with_messages = c3_turns.copy()
        if c3_turns_with_messages:
            first_turn = c3_turns_with_messages[0].copy()
            text = first_turn.get('instruction') or first_turn.get('content') or ""
            first_turn['instruction'] = message_passing + text
            first_turn['content'] = None
            c3_turns_with_messages[0] = first_turn

        context_text_c3, tokens_c3 = self.build_context_for_turns(c3_turns_with_messages)

        output_synthesis = self.generate_from_context(
            context_text_c3,
            prompts['synthesis'],
            max_tokens=300
        )

        gen_time = time.time() - start_time
        total_tokens = tokens_c1 + tokens_c2 + tokens_c3

        return IsolationResult(
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={
                'cluster_1_technical': tokens_c1,
                'cluster_2_business': tokens_c2,
                'cluster_3_synthesis': tokens_c3
            },
            total_tokens_processed=total_tokens,
            generation_time=gen_time
        )

    def test_all_conditions(self, example: Dict[str, Any]) -> Dict[str, IsolationResult]:
        """
        Run all four isolation conditions on an example.

        Args:
            example: Example dictionary with turns and cluster assignments

        Returns:
            Dictionary mapping condition name to results
        """
        results = {}

        print(f"\nTesting example: {example['id']}")
        print("=" * 60)

        print("\n[1/4] Running Sequential (baseline)...")
        results['sequential'] = self.condition_1_sequential(example)
        print(f"  Cache size: {results['sequential'].cache_sizes}")
        print(f"  Time: {results['sequential'].generation_time:.2f}s")

        print("\n[2/4] Running Prompted (soft isolation)...")
        results['prompted'] = self.condition_2_prompted(example)
        print(f"  Cache size: {results['prompted'].cache_sizes}")
        print(f"  Time: {results['prompted'].generation_time:.2f}s")

        print("\n[3/4] Running Turn-Based (naive isolation)...")
        results['turn_based'] = self.condition_3_turn_based(example)
        print(f"  Cache size: {results['turn_based'].cache_sizes}")
        print(f"  Time: {results['turn_based'].generation_time:.2f}s")

        print("\n[4/4] Running Semantic (RDIC - our method)...")
        results['semantic'] = self.condition_4_semantic(example)
        print(f"  Cache sizes: {results['semantic'].cache_sizes}")
        print(f"  Time: {results['semantic'].generation_time:.2f}s")

        print("\n" + "=" * 60)
        print("✓ All conditions complete")

        return results


def main():
    """Test semantic isolation on validation example."""

    # Load validation example
    with open('data/3cluster_examples/validation_001_software_eng.json', 'r') as f:
        example = json.load(f)

    print("Initializing Semantic Isolation Tester (MLX)...")
    tester = SemanticIsolationTesterMLX(
        model_name="mlx-community/gemma-3-12b-it-4bit"
    )

    # Run all conditions
    results = tester.test_all_conditions(example)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    output = {
        'example_id': example['id'],
        'framework': 'MLX',
        'model': 'gemma-2-9b-it-4bit',
        'results': {
            cond: {
                'outputs': res.outputs,
                'cache_sizes': res.cache_sizes,
                'total_tokens': res.total_tokens_processed,
                'time': res.generation_time
            }
            for cond, res in results.items()
        }
    }

    output_path = 'results/validation_001_isolation_test_mlx.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
