#!/usr/bin/env python3
"""
EXP-001: MoE Model Inspection
Test Mixtral MoE model for expert-specific attributes.
"""

from mlx_lm import load

def inspect_moe_model():
    """Inspect Mixtral MoE model for expert configuration."""
    model_id = "mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx"

    print(f"Loading: {model_id}")
    print("="*80)

    try:
        model, tokenizer = load(model_id)
        args = model.args

        print("\n[Model Args - All Attributes]")
        all_attrs = [a for a in dir(args) if not a.startswith('_')]
        for attr in sorted(all_attrs):
            value = getattr(args, attr, None)
            if not callable(value):
                print(f"  {attr}: {value}")

        # MoE-specific attributes
        print("\n[MoE-Specific Attributes]")
        moe_attrs = [
            'num_experts', 'num_experts_per_tok', 'num_local_experts',
            'num_experts_per_token', 'expert_top_k',
            'router_aux_loss_coef', 'output_router_logits'
        ]

        for attr in moe_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                print(f"  ✓ {attr}: {value}")

        # Standard attributes
        print("\n[Standard Cache Attributes]")
        standard_attrs = [
            'num_hidden_layers', 'n_layers',
            'num_key_value_heads', 'n_kv_heads',
            'num_attention_heads', 'n_heads',
            'hidden_size', 'dim',
            'head_dim',
            'sliding_window', 'sliding_window_size'
        ]

        for attr in standard_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                print(f"  ✓ {attr}: {value}")

        # Inspect layers
        print("\n[Layer Inspection]")
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            print(f"  Total layers: {len(layers)}")

            if len(layers) > 0:
                first_layer = layers[0]
                print(f"  First layer type: {type(first_layer).__name__}")
                print(f"  First layer attributes: {[a for a in dir(first_layer) if not a.startswith('_') and not callable(getattr(first_layer, a))]}")

                # Check for MoE-specific components
                if hasattr(first_layer, 'block_sparse_moe'):
                    print(f"  ✓ Has MoE component: block_sparse_moe")
                    moe = first_layer.block_sparse_moe
                    print(f"    MoE type: {type(moe).__name__}")
                    print(f"    MoE attributes: {[a for a in dir(moe) if not a.startswith('_') and not callable(getattr(moe, a))]}")

        print("\n✓ Successfully inspected MoE model")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_moe_model()
