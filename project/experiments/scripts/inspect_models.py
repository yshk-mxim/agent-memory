#!/usr/bin/env python3
"""
EXP-001: Model Args Inspection
Inspect model.args and config.json for all 4 target models.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import mlx.core as mx
from mlx_lm import load

# Target models
MODELS = {
    "Gemma 3 12B": "mlx-community/gemma-3-12b-it-4bit",
    "GPT-OSS-20B": "mlx-community/gpt-oss-20b-4bit",
    "Qwen 2.5-14B": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "Llama 3.1-8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
}

def inspect_model(model_name: str, model_id: str) -> Dict[str, Any]:
    """
    Load model and inspect both config.json and model.args.

    Returns:
        Dictionary with findings for this model
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*80}")

    findings = {
        "model_name": model_name,
        "model_id": model_id,
        "config_json": {},
        "model_args": {},
        "model_type": None,
        "layer_info": {}
    }

    try:
        # Load model
        print("\n[1] Loading model with mlx_lm.load()...")
        model, tokenizer = load(model_id)

        # Get model type
        findings["model_type"] = type(model).__name__
        print(f"    Model class: {findings['model_type']}")

        # Inspect model.args
        print("\n[2] Inspecting model.args attributes...")
        if hasattr(model, 'args'):
            args = model.args
            findings["model_args"]["has_args"] = True

            # Common attributes to check
            attrs_to_check = [
                'num_hidden_layers', 'n_layers', 'num_layers',
                'num_key_value_heads', 'n_kv_heads', 'num_kv_heads',
                'num_attention_heads', 'n_heads', 'num_heads',
                'hidden_size', 'dim', 'model_dim',
                'head_dim',
                'sliding_window', 'sliding_window_size',
                'sliding_window_pattern',
                'vocab_size',
                'model_type',
                'architectures',
                'attention_bias',
                'rope_theta',
                'rope_traditional',
                'rope_scaling',
            ]

            for attr in attrs_to_check:
                if hasattr(args, attr):
                    value = getattr(args, attr)
                    findings["model_args"][attr] = value
                    print(f"    ✓ {attr}: {value}")

            # List all available attributes
            print("\n    All model.args attributes:")
            all_attrs = [a for a in dir(args) if not a.startswith('_')]
            for attr in sorted(all_attrs):
                if attr not in attrs_to_check:
                    value = getattr(args, attr, None)
                    if not callable(value):
                        findings["model_args"][f"other_{attr}"] = str(value)
                        print(f"      - {attr}: {value}")
        else:
            findings["model_args"]["has_args"] = False
            print("    ⚠ Model has no 'args' attribute")

        # Find and read config.json
        print("\n[3] Reading config.json from HuggingFace cache...")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        # Search for the model directory
        model_dirs = list(cache_dir.glob(f"models--{model_id.replace('/', '--')}*"))
        if model_dirs:
            model_dir = model_dirs[0]
            # Look for config.json in snapshots
            config_files = list(model_dir.glob("**/config.json"))
            if config_files:
                config_path = config_files[0]
                print(f"    Found: {config_path}")

                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    findings["config_json"] = config_data

                print("\n    Key config.json attributes:")
                for key in sorted(config_data.keys()):
                    value = config_data[key]
                    # Print relevant attributes
                    if any(term in key.lower() for term in ['layer', 'head', 'dim', 'window', 'vocab', 'hidden', 'attention']):
                        print(f"      {key}: {value}")

        # Inspect model layers
        print("\n[4] Inspecting model layers...")
        if hasattr(model, 'model'):
            inner_model = model.model
            if hasattr(inner_model, 'layers'):
                layers = inner_model.layers
                findings["layer_info"]["num_layers"] = len(layers)
                print(f"    Total layers: {len(layers)}")

                # Inspect first layer
                if len(layers) > 0:
                    first_layer = layers[0]
                    print(f"    First layer type: {type(first_layer).__name__}")
                    print(f"    First layer attributes: {[a for a in dir(first_layer) if not a.startswith('_')]}")

                    # Check for attention attributes
                    if hasattr(first_layer, 'self_attn') or hasattr(first_layer, 'attention'):
                        attn = getattr(first_layer, 'self_attn', None) or getattr(first_layer, 'attention', None)
                        print(f"    Attention type: {type(attn).__name__}")
                        attn_attrs = [a for a in dir(attn) if not a.startswith('_') and not callable(getattr(attn, a))]
                        print(f"    Attention attributes: {attn_attrs}")

                        findings["layer_info"]["attention_type"] = type(attn).__name__

        print(f"\n✓ Successfully inspected {model_name}")

    except Exception as e:
        print(f"\n✗ Error inspecting {model_name}: {e}")
        findings["error"] = str(e)
        import traceback
        traceback.print_exc()

    return findings

def main():
    """Main inspection routine."""
    print("EXP-001: Model Args Inspection")
    print("=" * 80)

    all_findings = {}

    for model_name, model_id in MODELS.items():
        findings = inspect_model(model_name, model_id)
        all_findings[model_name] = findings

    # Save findings to JSON
    output_file = "/Users/dev_user/semantic/model_inspection_results.json"
    print(f"\n{'='*80}")
    print(f"Saving findings to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_findings, f, indent=2)

    print("\n✓ Inspection complete!")
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for model_name, findings in all_findings.items():
        print(f"\n{model_name}:")
        if "error" in findings:
            print(f"  ✗ Error: {findings['error']}")
        else:
            print(f"  Model type: {findings.get('model_type', 'Unknown')}")
            print(f"  Has model.args: {findings['model_args'].get('has_args', False)}")

            # Show key attributes
            args = findings['model_args']
            layers = args.get('num_hidden_layers') or args.get('n_layers') or args.get('num_layers')
            kv_heads = args.get('num_key_value_heads') or args.get('n_kv_heads')
            heads = args.get('num_attention_heads') or args.get('n_heads')
            head_dim = args.get('head_dim')
            sliding_window = args.get('sliding_window')
            sliding_pattern = args.get('sliding_window_pattern')

            print(f"  Layers: {layers}")
            print(f"  KV heads: {kv_heads}")
            print(f"  Attention heads: {heads}")
            print(f"  Head dim: {head_dim}")
            print(f"  Sliding window: {sliding_window}")
            print(f"  Sliding pattern: {sliding_pattern}")

if __name__ == "__main__":
    main()
