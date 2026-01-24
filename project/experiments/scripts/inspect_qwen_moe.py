#!/usr/bin/env python3
"""
EXP-001: Qwen MoE Model Inspection
Test Qwen1.5-MoE model for expert-specific attributes.
"""

from mlx_lm import load

def inspect_moe_model():
    """Inspect Qwen1.5-MoE model for expert configuration."""
    model_id = "mlx-community/Qwen1.5-MoE-A2.7B-4bit"

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
            'router_aux_loss_coef', 'output_router_logits',
            'decoder_sparse_step', 'moe_intermediate_size'
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

        # Compute head_dim if needed
        if hasattr(args, 'hidden_size') and hasattr(args, 'num_attention_heads'):
            head_dim_computed = args.hidden_size // args.num_attention_heads
            print(f"  head_dim (computed): {head_dim_computed}")

        # Inspect layers
        print("\n[Layer Inspection]")
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            print(f"  Total layers: {len(layers)}")

            if len(layers) > 0:
                first_layer = layers[0]
                print(f"  First layer type: {type(first_layer).__name__}")

                # Get attributes that aren't methods
                layer_attrs = [a for a in dir(first_layer) if not a.startswith('_')]
                non_method_attrs = []
                for attr in layer_attrs:
                    try:
                        val = getattr(first_layer, attr)
                        if not callable(val):
                            non_method_attrs.append(attr)
                    except:
                        pass

                print(f"  First layer attributes: {non_method_attrs}")

                # Check for MoE-specific components
                moe_component_names = ['mlp', 'feed_forward', 'moe', 'block_sparse_moe', 'experts']
                for component in moe_component_names:
                    if hasattr(first_layer, component):
                        comp = getattr(first_layer, component)
                        print(f"  ✓ Has component: {component}")
                        print(f"    Type: {type(comp).__name__}")

                        # Get component attributes
                        comp_attrs = [a for a in dir(comp) if not a.startswith('_')]
                        comp_non_methods = []
                        for attr in comp_attrs:
                            try:
                                val = getattr(comp, attr)
                                if not callable(val):
                                    comp_non_methods.append(f"{attr}={val}")
                            except:
                                pass
                        if comp_non_methods:
                            print(f"    Attributes: {comp_non_methods}")

        print("\n✓ Successfully inspected Qwen MoE model")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_moe_model()
