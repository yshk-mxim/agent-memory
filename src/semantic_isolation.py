"""
Semantic KV Cache Isolation Implementation

Tests 4 conditions for multi-task conversation handling:
1. Sequential: All turns in one KV cache (baseline)
2. Prompted: Same as sequential but with isolation instruction
3. Turn-Based: Reset cache at turn boundaries (naive isolation)
4. Semantic: Partition cache by semantic cluster (RDIC - our method)

Uses TRUE KV cache manipulation via HuggingFace Transformers.
"""

import json
import torch
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class IsolationResult:
    """Results from one isolation condition."""
    condition: str
    outputs: Dict[str, str]  # {'technical': '...', 'business': '...', 'synthesis': '...'}
    cache_sizes: Dict[str, int]  # Token counts in each cache
    total_tokens_processed: int
    generation_time: float


class SemanticIsolationTester:
    """
    Test semantic KV cache isolation with TRUE cache manipulation.

    Loads Gemma 2 12B (4-bit quantized) and provides access to past_key_values
    for direct KV cache partitioning.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",  # Gemma 2 2B Instruct
        load_in_4bit: bool = False,  # MPS doesn't support 4-bit quantization
        device: str = "auto",  # Will detect MPS/CUDA/CPU
        random_seed: int = 42,
        use_neutral_prompts: bool = True
    ):
        """
        Initialize with language model for KV cache isolation testing.

        Default: Gemma 2 2B Instruct (gated but accessible, works well on Mac MPS)

        For CUDA GPUs with more memory:
        - "google/gemma-2-9b-it" or "google/gemma-2-27b-it" with load_in_4bit=True

        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Use 4-bit quantization (CUDA only, not MPS)
            device: Device to use ("auto", "mps", "cuda", "cpu")
            random_seed: Random seed for reproducibility
            use_neutral_prompts: If True, use neutral prompts that don't leak semantic info
        """
        print(f"Loading {model_name}...")

        # Set random seed for reproducibility
        self.random_seed = random_seed
        self.use_neutral_prompts = use_neutral_prompts
        torch.manual_seed(random_seed)

        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print("  Detected: CUDA GPU")
            elif torch.backends.mps.is_available():
                device = "mps"
                print("  Detected: Apple Silicon (MPS)")
            else:
                device = "cpu"
                print("  Detected: CPU only")

        self.device = device
        print(f"  Using device: {device}")

        # MPS doesn't support 4-bit quantization
        if device == "mps" and load_in_4bit:
            print("  Warning: MPS doesn't support 4-bit quantization, using fp16 instead")
            load_in_4bit = False

        # Configure quantization (CUDA only)
        quantization_config = None
        if load_in_4bit and device == "cuda":
            print("  Using 4-bit quantization (CUDA)")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load model
        load_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            # For MPS/CPU, load to specific device
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Move to device if not using device_map
        if load_kwargs["device_map"] is None:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.model.device if hasattr(self.model, 'device') else device}")

        # Estimate memory
        if load_in_4bit:
            mem_est = "~7GB (4-bit)"
        elif "2b" in model_name.lower():
            mem_est = "~4GB (fp16)"
        elif "7b" in model_name.lower():
            mem_est = "~14GB (fp16)"
        elif "12b" in model_name.lower():
            mem_est = "~24GB (fp16)"
        else:
            mem_est = "unknown"

        print(f"  Memory estimate: {mem_est} + KV cache overhead")
        print(f"  Random seed: {random_seed}")
        print(f"  Neutral prompts: {use_neutral_prompts}")

    def _cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _report_memory(self):
        """Report current memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"CUDA Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        elif self.device == "mps":
            # MPS doesn't have detailed memory reporting like CUDA
            print(f"MPS Memory: (detailed reporting not available)")
        else:
            print(f"CPU Memory: (using system RAM)")

    def validate_example(self, example: Dict[str, Any]):
        """
        Validate example has required structure for semantic isolation.

        Raises:
            AssertionError if example is invalid
        """
        assert 'turns' in example, "Example missing 'turns' field"
        assert len(example['turns']) > 0, "Example has no turns"

        # Check all turns have cluster labels
        for i, turn in enumerate(example['turns']):
            assert 'cluster' in turn, f"Turn {turn.get('turn_id', i)} missing 'cluster' field"

        # Check cluster distribution
        clusters = {t['cluster'] for t in example['turns']}
        assert clusters == {1, 2, 3}, f"Expected clusters {{1,2,3}}, got {clusters}"

        c1_count = sum(1 for t in example['turns'] if t['cluster'] == 1)
        c2_count = sum(1 for t in example['turns'] if t['cluster'] == 2)
        c3_count = sum(1 for t in example['turns'] if t['cluster'] == 3)

        assert c1_count >= 3, f"Cluster 1 has only {c1_count} turns (need ≥3)"
        assert c2_count >= 3, f"Cluster 2 has only {c2_count} turns (need ≥3)"
        assert c3_count >= 3, f"Cluster 3 has only {c3_count} turns (need ≥3)"

    def get_generation_prompts(self, neutral: bool = None) -> Dict[str, str]:
        """
        Get generation prompts for outputs.

        Args:
            neutral: Override use_neutral_prompts setting

        Returns:
            Dict with prompts for technical, business, synthesis
        """
        if neutral is None:
            neutral = self.use_neutral_prompts

        if neutral:
            # Neutral prompts that don't leak semantic information
            return {
                'technical': "Generate output A based on the provided context:",
                'business': "Generate output B based on the provided context:",
                'synthesis': "Generate output C that integrates outputs A and B:"
            }
        else:
            # Targeted prompts (may help model but confound isolation test)
            return {
                'technical': "Based on the technical performance analysis, provide your recommendations:",
                'business': "Based on the business product strategy, provide your recommendations:",
                'synthesis': "Provide an executive strategic roadmap combining technical and business priorities:"
            }

    def get_cache_size(self, past_key_values: Optional[Tuple]) -> int:
        """Get size of KV cache in tokens with validation."""
        if past_key_values is None:
            return 0

        try:
            # past_key_values structure: tuple of (num_layers) tuples of (key, value) tensors
            # key/value shape: [batch_size, num_heads, seq_len, head_dim]
            key_tensor = past_key_values[0][0]

            assert key_tensor.dim() == 4, f"Expected 4D tensor, got {key_tensor.dim()}D"
            batch_size, num_heads, seq_len, head_dim = key_tensor.shape

            assert batch_size == 1, f"Batch size {batch_size} != 1 (not supported)"

            return seq_len

        except Exception as e:
            print(f"Warning: Could not get cache size: {e}")
            return 0

    def generate_from_cache(
        self,
        past_key_values: Optional[Tuple],
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using existing KV cache (TRUE cache isolation).

        Implements manual token-by-token generation to properly handle past_key_values.

        Args:
            past_key_values: Existing KV cache or None for fresh generation
            prompt: Text prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_input_ids = inputs.input_ids
        prompt_length = prompt_input_ids.shape[1]

        # If we have no cache, process the prompt first to build the initial cache
        if past_key_values is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=prompt_input_ids,
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = outputs.past_key_values
            # Start generation from the last token's logits
            next_token_logits = outputs.logits[:, -1, :]
        else:
            # We have a cache, so we need to process the prompt tokens incrementally
            with torch.no_grad():
                outputs = self.model(
                    input_ids=prompt_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        # Generate tokens one at a time
        generated_ids = []

        for _ in range(max_new_tokens):
            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-p filtering (simplified - just use top-k for now)
            # For production, implement proper nucleus sampling
            probs = torch.softmax(next_token_logits, dim=-1)

            # Simple sampling
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.item())

            # Get next token's logits
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def condition_1_sequential(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 1: Sequential baseline.

        All 15 turns processed in chronological order with ONE accumulating KV cache.
        Expected: Context pollution - technical and business contexts mixed together.

        Args:
            example: 3-cluster example with 15 turns

        Returns:
            IsolationResult with outputs and cache statistics
        """
        import time
        start_time = time.time()

        # Validate example structure
        self.validate_example(example)

        # Get generation prompts (neutral or targeted)
        prompts = self.get_generation_prompts()

        past_kv = None
        total_tokens = 0
        skipped_turns = []

        # Process all 15 turns in order, accumulating cache
        for turn in example['turns']:
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                skipped_turns.append(turn.get('turn_id', '?'))
                continue

            inputs = self.tokenizer(turn_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    past_key_values=past_kv,
                    use_cache=True
                )
                past_kv = outputs.past_key_values

            total_tokens += inputs.input_ids.shape[1]

        if skipped_turns:
            print(f"  Warning: Skipped {len(skipped_turns)} empty turns: {skipped_turns}")

        cache_size = self.get_cache_size(past_kv)

        # Generate all three outputs from the SAME mixed cache
        output_technical = self.generate_from_cache(
            past_kv,
            prompts['technical'],
            max_new_tokens=300
        )

        output_business = self.generate_from_cache(
            past_kv,
            prompts['business'],
            max_new_tokens=300
        )

        output_synthesis = self.generate_from_cache(
            past_kv,
            prompts['synthesis'],
            max_new_tokens=400
        )

        elapsed = time.time() - start_time

        return IsolationResult(
            condition="sequential",
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'unified': cache_size},
            total_tokens_processed=total_tokens,
            generation_time=elapsed
        )

    def condition_2_prompted(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 2: Prompted isolation (soft isolation).

        Same as sequential but with explicit instruction to keep tasks separate.
        Tests: Can prompts achieve isolation without architectural support?

        Expected: Some improvement but still interference due to shared KV cache.
        """
        import time
        start_time = time.time()

        # Add isolation instruction at the beginning
        isolation_instruction = """
IMPORTANT: This conversation covers multiple distinct tasks:
1. Technical performance optimization
2. Business product strategy
3. Executive synthesis

Keep these contexts SEPARATE. Do not mix technical terminology into business analysis or vice versa.
Generate three distinct, focused outputs.
"""

        past_kv = None
        total_tokens = 0

        # Process instruction first
        inputs = self.tokenizer(isolation_instruction, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
        total_tokens += inputs.input_ids.shape[1]

        # Process all 15 turns with accumulating cache (same as sequential)
        for turn in example['turns']:
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                continue

            inputs = self.tokenizer(turn_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values

            total_tokens += inputs.input_ids.shape[1]

        cache_size = self.get_cache_size(past_kv)

        # Generate outputs (same as sequential)
        output_technical = self.generate_from_cache(
            past_kv,
            "Based on the technical performance analysis, provide your recommendations:",
            max_new_tokens=300
        )

        output_business = self.generate_from_cache(
            past_kv,
            "Based on the business product strategy, provide your recommendations:",
            max_new_tokens=300
        )

        output_synthesis = self.generate_from_cache(
            past_kv,
            "Provide an executive strategic roadmap combining technical and business priorities:",
            max_new_tokens=400
        )

        elapsed = time.time() - start_time

        return IsolationResult(
            condition="prompted",
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'unified_with_prompt': cache_size},
            total_tokens_processed=total_tokens,
            generation_time=elapsed
        )

    def condition_3_turn_based(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 3: Turn-based isolation (naive isolation).

        Add turn markers but keep full conversation history.
        Reset would lose too much context, so we use formatting instead.

        Expected: Better than sequential (structure visible) but not as good as semantic
        (still mixes contexts, just with turn boundaries marked).
        """
        import time
        start_time = time.time()

        past_kv = None
        total_tokens = 0

        # Process each turn with explicit markers
        for i, turn in enumerate(example['turns'], 1):
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                continue

            # Add turn marker
            marked_turn = f"=== Turn {i} ===\n{turn_text}"

            inputs = self.tokenizer(marked_turn, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values

            total_tokens += inputs.input_ids.shape[1]

        cache_size = self.get_cache_size(past_kv)

        # Generate outputs
        output_technical = self.generate_from_cache(
            past_kv,
            "Based on turns 1-5 (technical analysis), provide your recommendations:",
            max_new_tokens=300
        )

        output_business = self.generate_from_cache(
            past_kv,
            "Based on turns 6-10 (business strategy), provide your recommendations:",
            max_new_tokens=300
        )

        output_synthesis = self.generate_from_cache(
            past_kv,
            "Based on turns 11-15 (synthesis), provide an executive strategic roadmap:",
            max_new_tokens=400
        )

        elapsed = time.time() - start_time

        return IsolationResult(
            condition="turn_based",
            outputs={
                'technical': output_technical,
                'business': output_business,
                'synthesis': output_synthesis
            },
            cache_sizes={'turn_marked': cache_size},
            total_tokens_processed=total_tokens,
            generation_time=elapsed
        )

    def condition_4_semantic(self, example: Dict[str, Any]) -> IsolationResult:
        """
        Condition 4: Semantic isolation (RDIC - our method).

        CRITICAL: Each semantic cluster gets its OWN isolated KV cache.
        - Cluster 1 (technical): past_kv_c1 (ONLY sees technical turns)
        - Cluster 2 (business): past_kv_c2 (ONLY sees business turns)
        - Cluster 3 (synthesis): past_kv_c3 (sees synthesis turns + outputs from c1+c2)

        This is TRUE architectural isolation - different caches prevent cross-attention.

        NOTE: Cluster 3 uses "message passing" - sees OUTPUTS (not caches) from c1+c2.
        This is intentional multi-agent pattern, but creates indirect leakage.

        Expected: Best performance - no interference, high quality on each task.
        """
        import time
        start_time = time.time()

        # Validate example structure
        self.validate_example(example)

        # Get generation prompts
        prompts = self.get_generation_prompts()

        cache_sizes = {}
        outputs = {}
        total_tokens = 0

        # Extract clusters from example
        cluster_1_turns = [t for t in example['turns'] if t['cluster'] == 1]
        cluster_2_turns = [t for t in example['turns'] if t['cluster'] == 2]
        cluster_3_turns = [t for t in example['turns'] if t['cluster'] == 3]

        # ===== CLUSTER 1: Technical Analysis (ISOLATED CACHE) =====
        past_kv_c1 = None
        for turn in cluster_1_turns:
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                continue

            inputs = self.tokenizer(turn_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs_obj = self.model(**inputs, past_key_values=past_kv_c1, use_cache=True)
                past_kv_c1 = outputs_obj.past_key_values  # Accumulate ONLY cluster 1 context

            total_tokens += inputs.input_ids.shape[1]

        cache_sizes['cluster_1_technical'] = self.get_cache_size(past_kv_c1)

        # Generate technical output from isolated cache
        outputs['technical'] = self.generate_from_cache(
            past_kv_c1,
            prompts['technical'],
            max_new_tokens=300
        )

        # ===== CLUSTER 2: Business Strategy (FRESH ISOLATED CACHE) =====
        past_kv_c2 = None  # RESET - does NOT see cluster 1!
        for turn in cluster_2_turns:
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                continue

            inputs = self.tokenizer(turn_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs_obj = self.model(**inputs, past_key_values=past_kv_c2, use_cache=True)
                past_kv_c2 = outputs_obj.past_key_values  # Accumulate ONLY cluster 2 context

            total_tokens += inputs.input_ids.shape[1]

        cache_sizes['cluster_2_business'] = self.get_cache_size(past_kv_c2)

        # Generate business output from isolated cache
        outputs['business'] = self.generate_from_cache(
            past_kv_c2,
            prompts['business'],
            max_new_tokens=300
        )

        # ===== CLUSTER 3: Synthesis (FRESH CACHE + MESSAGE PASSING) =====
        # Key: Cluster 3 sees OUTPUTS (not caches) from clusters 1 & 2
        # This creates "message passing" - intentional for multi-agent simulation
        synthesis_context = f"""
Output A:
{outputs['technical']}

Output B:
{outputs['business']}

Now consider the integration instructions:
"""

        past_kv_c3 = None  # FRESH cache

        # Add synthesis context
        inputs = self.tokenizer(synthesis_context, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs_obj = self.model(**inputs, past_key_values=past_kv_c3, use_cache=True)
            past_kv_c3 = outputs_obj.past_key_values
        total_tokens += inputs.input_ids.shape[1]

        # Add synthesis turns
        for turn in cluster_3_turns:
            turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
            if not turn_text:
                continue

            inputs = self.tokenizer(turn_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs_obj = self.model(**inputs, past_key_values=past_kv_c3, use_cache=True)
                past_kv_c3 = outputs_obj.past_key_values

            total_tokens += inputs.input_ids.shape[1]

        cache_sizes['cluster_3_synthesis'] = self.get_cache_size(past_kv_c3)

        # Generate synthesis from cluster 3 cache
        outputs['synthesis'] = self.generate_from_cache(
            past_kv_c3,
            prompts['synthesis'],
            max_new_tokens=400
        )

        elapsed = time.time() - start_time

        return IsolationResult(
            condition="semantic",
            outputs=outputs,
            cache_sizes=cache_sizes,
            total_tokens_processed=total_tokens,
            generation_time=elapsed
        )

    def test_all_conditions(self, example: Dict[str, Any]) -> Dict[str, IsolationResult]:
        """
        Run all 4 conditions on one example with memory cleanup.

        Args:
            example: 3-cluster conversation example

        Returns:
            Dict mapping condition name to IsolationResult
        """
        import gc

        print(f"\nTesting example: {example['id']}")
        print("=" * 60)

        results = {}

        # Condition 1: Sequential
        print("\n[1/4] Running Sequential (baseline)...")
        results['sequential'] = self.condition_1_sequential(example)
        print(f"  Cache size: {results['sequential'].cache_sizes}")
        print(f"  Time: {results['sequential'].generation_time:.2f}s")

        # Free GPU memory after each condition
        self._cleanup_memory()
        gc.collect()

        # Condition 2: Prompted
        print("\n[2/4] Running Prompted (soft isolation)...")
        results['prompted'] = self.condition_2_prompted(example)
        print(f"  Cache size: {results['prompted'].cache_sizes}")
        print(f"  Time: {results['prompted'].generation_time:.2f}s")

        self._cleanup_memory()
        gc.collect()

        # Condition 3: Turn-Based
        print("\n[3/4] Running Turn-Based (naive isolation)...")
        results['turn_based'] = self.condition_3_turn_based(example)
        print(f"  Cache size: {results['turn_based'].cache_sizes}")
        print(f"  Time: {results['turn_based'].generation_time:.2f}s")

        self._cleanup_memory()
        gc.collect()

        # Condition 4: Semantic
        print("\n[4/4] Running Semantic (RDIC - our method)...")
        results['semantic'] = self.condition_4_semantic(example)
        print(f"  Cache sizes: {results['semantic'].cache_sizes}")
        print(f"  Time: {results['semantic'].generation_time:.2f}s")

        self._cleanup_memory()
        gc.collect()

        print("\n" + "=" * 60)
        print("✓ All conditions complete")

        # Report memory usage
        self._report_memory()

        return results


def main():
    """Test semantic isolation on validation example."""

    # Load validation example
    with open('data/3cluster_examples/validation_001_software_eng.json', 'r') as f:
        example = json.load(f)

    print("Initializing Semantic Isolation Tester...")

    # Detect device and choose appropriate model
    if torch.cuda.is_available():
        print("CUDA GPU detected - using Gemma 2 9B with 4-bit quantization")
        tester = SemanticIsolationTester(
            model_name="google/gemma-2-9b-it",
            load_in_4bit=True,
            device="cuda"
        )
    elif torch.backends.mps.is_available():
        print("Apple Silicon (MPS) detected - using Gemma 2 2B with fp16")
        print("Note: MPS doesn't support 4-bit quantization")
        tester = SemanticIsolationTester(
            model_name="google/gemma-2-2b-it",
            load_in_4bit=False,
            device="mps"
        )
    else:
        print("No GPU detected - using Gemma 2 2B on CPU (will be slow)")
        tester = SemanticIsolationTester(
            model_name="google/gemma-2-2b-it",
            load_in_4bit=False,
            device="cpu"
        )

    # Run all conditions
    results = tester.test_all_conditions(example)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    output = {
        'example_id': example['id'],
        'device': tester.device,
        'model': tester.model.config.name_or_path if hasattr(tester.model.config, 'name_or_path') else 'unknown',
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

    with open('results/validation_001_isolation_test.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✓ Results saved to results/validation_001_isolation_test.json")


if __name__ == "__main__":
    main()
