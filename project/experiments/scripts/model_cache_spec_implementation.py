#!/usr/bin/env python3
"""
ModelCacheSpec.from_model() Implementation
Based on EXP-001 findings

This is a reference implementation ready to be integrated into the codebase.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelCacheSpec:
    """Cache specification extracted from model."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    sliding_window: Optional[int] = None
    layer_pattern: Optional[Dict[str, Any]] = None

    @classmethod
    def from_model(cls, model) -> "ModelCacheSpec":
        """
        Extract cache specification from loaded MLX model.

        Supports:
        - Standard models (Qwen, Llama)
        - Gemma 3 with nested text_config
        - MoE models (Qwen1.5-MoE)
        - Hybrid attention patterns

        Args:
            model: Loaded MLX model from mlx_lm.load()

        Returns:
            ModelCacheSpec with extracted parameters

        Raises:
            ValueError: If required attributes cannot be extracted
        """
        args = model.args

        # Step 1: Extract basic attributes (handle Gemma 3 nested config)
        if hasattr(args, 'text_config'):
            # Gemma 3: nested config
            config = args.text_config
            num_layers = config.get('num_hidden_layers')
            num_kv_heads = config.get('num_key_value_heads')
            num_heads = config.get('num_attention_heads')
            hidden_size = config.get('hidden_size')
            sliding_window = config.get('sliding_window', None)
            model_type = getattr(args, 'model_type', 'unknown')
        else:
            # Standard models (Qwen, Llama, etc.)
            num_layers = getattr(args, 'num_hidden_layers', None)
            num_kv_heads = getattr(args, 'num_key_value_heads', None)
            num_heads = getattr(args, 'num_attention_heads', None)
            hidden_size = getattr(args, 'hidden_size', None)
            sliding_window = getattr(args, 'sliding_window', None)
            model_type = getattr(args, 'model_type', 'unknown')

        # Step 2: Validate required attributes
        if num_layers is None:
            raise ValueError("Cannot extract num_layers from model.args")
        if num_kv_heads is None:
            raise ValueError("Cannot extract num_kv_heads from model.args")
        if hidden_size is None or num_heads is None:
            raise ValueError("Cannot compute head_dim: missing hidden_size or num_heads")

        # Step 3: Compute head dimension (ALWAYS compute, never rely on attribute)
        head_dim = hidden_size // num_heads

        # Step 4: Detect layer pattern (hybrid vs uniform attention)
        layer_pattern = cls._detect_layer_pattern(model, args, model_type)

        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
            layer_pattern=layer_pattern
        )

    @staticmethod
    def _detect_layer_pattern(model, args, model_type: str) -> Dict[str, Any]:
        """
        Detect layer attention pattern (uniform vs hybrid).

        Three-tier detection approach:
        1. Check layer_types attribute (most reliable)
        2. Inspect layer objects for use_sliding
        3. Fallback to model-type heuristics

        Args:
            model: Loaded MLX model
            args: Model args/config
            model_type: Model type string (e.g., 'gemma3', 'llama')

        Returns:
            dict with 'type' and pattern details
        """
        # Tier 1: Check layer_types attribute (e.g., Llama 3.1)
        if hasattr(args, 'layer_types'):
            layer_types = args.layer_types
            unique_types = set(layer_types)

            if len(unique_types) > 1:
                # Multiple layer types = hybrid attention
                return {
                    'type': 'hybrid',
                    'detection_method': 'layer_types_attribute',
                    'layer_types': layer_types,
                    'unique_types': list(unique_types)
                }
            else:
                # Single layer type = uniform attention
                return {
                    'type': 'uniform',
                    'detection_method': 'layer_types_attribute',
                    'attention_type': list(unique_types)[0]
                }

        # Tier 2: Inspect layer objects for use_sliding attribute
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            use_sliding_values = []

            for layer in layers:
                if hasattr(layer, 'use_sliding'):
                    use_sliding_values.append(layer.use_sliding)

            if use_sliding_values:
                unique_sliding = set(use_sliding_values)

                if len(unique_sliding) > 1:
                    # Mix of True/False = hybrid attention
                    return {
                        'type': 'hybrid',
                        'detection_method': 'layer_use_sliding_attribute',
                        'use_sliding_per_layer': use_sliding_values
                    }
                elif True in unique_sliding:
                    # All use sliding window
                    return {
                        'type': 'uniform',
                        'detection_method': 'layer_use_sliding_attribute',
                        'attention_type': 'sliding_window'
                    }
                else:
                    # All use full attention
                    return {
                        'type': 'uniform',
                        'detection_method': 'layer_use_sliding_attribute',
                        'attention_type': 'full_attention'
                    }

        # Tier 3: Model-type heuristics (for known hybrid models)
        HYBRID_PATTERNS = {
            'gemma3': {
                'type': 'hybrid',
                'detection_method': 'model_type_heuristic',
                'global_layers': 8,
                'sliding_layers': 40,
                'description': 'First 8 layers use global attention, remaining 40 use sliding window (1024 tokens)',
                'pattern': 'global_first'
            }
        }

        if model_type in HYBRID_PATTERNS:
            return HYBRID_PATTERNS[model_type]

        # Default: assume uniform full attention
        return {
            'type': 'uniform',
            'detection_method': 'default_assumption',
            'attention_type': 'full_attention'
        }

    def get_layer_cache_size(
        self,
        seq_length: int,
        batch_size: int = 1,
        dtype_bytes: int = 2  # float16 = 2 bytes
    ) -> int:
        """
        Calculate cache size for a single layer.

        Args:
            seq_length: Sequence length (number of tokens)
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for float16, 4 for float32)

        Returns:
            Total cache size in bytes for one layer
        """
        # Cache shape: [batch_size, num_kv_heads, seq_length, head_dim]
        # We have K and V, so multiply by 2
        size_per_kv = batch_size * self.num_kv_heads * seq_length * self.head_dim * dtype_bytes
        total_size = size_per_kv * 2  # K + V

        return total_size

    def get_total_cache_size(
        self,
        seq_length: int,
        batch_size: int = 1,
        dtype_bytes: int = 2
    ) -> int:
        """
        Calculate total cache size for all layers.

        Args:
            seq_length: Sequence length (number of tokens)
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for float16, 4 for float32)

        Returns:
            Total cache size in bytes for all layers
        """
        return self.get_layer_cache_size(seq_length, batch_size, dtype_bytes) * self.num_layers

    def __repr__(self) -> str:
        """Human-readable representation."""
        sliding_info = f", sliding_window={self.sliding_window}" if self.sliding_window else ""
        pattern_type = self.layer_pattern.get('type', 'unknown') if self.layer_pattern else 'unknown'

        return (
            f"ModelCacheSpec("
            f"num_layers={self.num_layers}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}"
            f"{sliding_info}, "
            f"pattern={pattern_type})"
        )


# Example Usage
if __name__ == "__main__":
    from mlx_lm import load

    # Test with different models
    test_models = [
        ("Gemma 3 12B", "mlx-community/gemma-3-12b-it-4bit"),
        ("Qwen 2.5-14B", "mlx-community/Qwen2.5-14B-Instruct-4bit"),
        ("Llama 3.1-8B", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ("Qwen1.5-MoE-A2.7B", "mlx-community/Qwen1.5-MoE-A2.7B-4bit"),
    ]

    for name, model_id in test_models:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        try:
            model, tokenizer = load(model_id)
            spec = ModelCacheSpec.from_model(model)

            print(f"\nExtracted Spec: {spec}")
            print(f"\nLayer Pattern:")
            if spec.layer_pattern:
                for key, value in spec.layer_pattern.items():
                    print(f"  {key}: {value}")

            print(f"\nCache Sizes:")
            seq_len = 1024
            layer_size = spec.get_layer_cache_size(seq_len)
            total_size = spec.get_total_cache_size(seq_len)
            print(f"  Per layer (1024 tokens): {layer_size / 1024 / 1024:.2f} MB")
            print(f"  Total ({spec.num_layers} layers): {total_size / 1024 / 1024:.2f} MB")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Testing complete!")
