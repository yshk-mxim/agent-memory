#!/usr/bin/env python3
"""
EXP-001: Detailed Model Inspection
Extract head_dim and sliding_window_pattern information.
"""

import json
from mlx_lm import load

# Models to inspect
MODELS = {
    "Gemma 3 12B": "mlx-community/gemma-3-12b-it-4bit",
    "Qwen 2.5-14B": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "Llama 3.1-8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}

def compute_head_dim(hidden_size: int, num_heads: int) -> int:
    """Compute head dimension from hidden size and number of heads."""
    return hidden_size // num_heads

def inspect_model_detailed(model_name: str, model_id: str):
    """Detailed inspection focusing on cache spec attributes."""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")

    try:
        model, tokenizer = load(model_id)

        # Get model.args
        args = model.args
        print(f"\n[Model Args]")

        # Extract key attributes
        if hasattr(args, 'text_config'):
            # Gemma 3 has nested text_config
            text_config = args.text_config
            num_layers = text_config.get('num_hidden_layers', 'NOT FOUND')
            num_kv_heads = text_config.get('num_key_value_heads', 'NOT FOUND')
            num_heads = text_config.get('num_attention_heads', 'NOT FOUND')
            hidden_size = text_config.get('hidden_size', 'NOT FOUND')
            sliding_window = text_config.get('sliding_window', 'NOT FOUND')

            print(f"  num_hidden_layers: {num_layers}")
            print(f"  num_key_value_heads: {num_kv_heads}")
            print(f"  num_attention_heads: {num_heads}")
            print(f"  hidden_size: {hidden_size}")
            print(f"  sliding_window: {sliding_window}")

            if isinstance(hidden_size, int) and isinstance(num_heads, int):
                head_dim = compute_head_dim(hidden_size, num_heads)
                print(f"  head_dim (computed): {head_dim}")
        else:
            # Standard attributes
            num_layers = getattr(args, 'num_hidden_layers', 'NOT FOUND')
            num_kv_heads = getattr(args, 'num_key_value_heads', 'NOT FOUND')
            num_heads = getattr(args, 'num_attention_heads', 'NOT FOUND')
            hidden_size = getattr(args, 'hidden_size', 'NOT FOUND')
            sliding_window = getattr(args, 'sliding_window', 'NOT FOUND')
            head_dim = getattr(args, 'head_dim', None)

            print(f"  num_hidden_layers: {num_layers}")
            print(f"  num_key_value_heads: {num_kv_heads}")
            print(f"  num_attention_heads: {num_heads}")
            print(f"  hidden_size: {hidden_size}")
            print(f"  sliding_window: {sliding_window}")
            print(f"  head_dim: {head_dim}")

            if head_dim is None and isinstance(hidden_size, int) and isinstance(num_heads, int):
                head_dim = compute_head_dim(hidden_size, num_heads)
                print(f"  head_dim (computed): {head_dim}")

        # Check for sliding_window_pattern
        if hasattr(args, 'sliding_window_pattern'):
            print(f"  sliding_window_pattern: {args.sliding_window_pattern}")
        elif hasattr(args, 'text_config') and 'sliding_window_pattern' in args.text_config:
            print(f"  sliding_window_pattern: {args.text_config['sliding_window_pattern']}")
        else:
            print(f"  sliding_window_pattern: NOT FOUND")

        # Check for layer_types (indicator of hybrid attention)
        if hasattr(args, 'layer_types'):
            layer_types = args.layer_types
            print(f"\n[Layer Types]")
            print(f"  Found layer_types attribute!")
            print(f"  Total layers: {len(layer_types)}")
            print(f"  Unique types: {set(layer_types)}")
            print(f"  First 10: {layer_types[:10]}")

        # Inspect actual layer objects
        print(f"\n[Layer Inspection]")
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            print(f"  Total layers: {len(layers)}")

            # Check first layer
            if len(layers) > 0:
                first_layer = layers[0]
                print(f"  First layer type: {type(first_layer).__name__}")

                # Check for use_sliding attribute
                if hasattr(first_layer, 'use_sliding'):
                    print(f"  First layer has 'use_sliding': {first_layer.use_sliding}")

                # Check attention
                if hasattr(first_layer, 'self_attn'):
                    attn = first_layer.self_attn
                    print(f"  Attention type: {type(attn).__name__}")
                    if hasattr(attn, 'head_dim'):
                        print(f"  Attention.head_dim: {attn.head_dim}")
                    if hasattr(attn, 'n_heads'):
                        print(f"  Attention.n_heads: {attn.n_heads}")
                    if hasattr(attn, 'n_kv_heads'):
                        print(f"  Attention.n_kv_heads: {attn.n_kv_heads}")
                elif hasattr(first_layer, 'attention'):
                    attn = first_layer.attention
                    print(f"  Attention type: {type(attn).__name__}")
                    if hasattr(attn, 'head_dim'):
                        print(f"  Attention.head_dim: {attn.head_dim}")
                    if hasattr(attn, 'n_heads'):
                        print(f"  Attention.n_heads: {attn.n_heads}")
                    if hasattr(attn, 'n_kv_heads'):
                        print(f"  Attention.n_kv_heads: {attn.n_kv_heads}")

            # Check if layers have different types (hybrid attention)
            layer_type_names = [type(layer).__name__ for layer in layers]
            unique_layer_types = set(layer_type_names)
            print(f"  Unique layer class types: {unique_layer_types}")

            if len(unique_layer_types) > 1:
                print(f"  ⚠ HYBRID ATTENTION DETECTED - Multiple layer types!")

            # Check for use_sliding variations
            use_sliding_list = []
            for i, layer in enumerate(layers):
                if hasattr(layer, 'use_sliding'):
                    use_sliding_list.append((i, layer.use_sliding))

            if use_sliding_list:
                print(f"  Layers with use_sliding attribute:")
                for layer_idx, use_sliding in use_sliding_list[:5]:
                    print(f"    Layer {layer_idx}: use_sliding={use_sliding}")
                if len(use_sliding_list) > 5:
                    print(f"    ... and {len(use_sliding_list) - 5} more")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main inspection routine."""
    print("EXP-001: Detailed Model Inspection")
    print("Focus: Cache spec attributes, head_dim, sliding_window_pattern")

    for model_name, model_id in MODELS.items():
        inspect_model_detailed(model_name, model_id)

    print(f"\n{'='*80}")
    print("Inspection complete!")

if __name__ == "__main__":
    main()
