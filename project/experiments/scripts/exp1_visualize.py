"""
Generate visualizations and tables for Experiment 1.

Creates:
- Figure 1: Bar chart showing degradation by conflict type
- Table 1: Mean scores with standard deviation (CSV format)

Usage:
    python -m experiments.exp1_visualize
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, print_section


def generate_bar_chart(
    analysis: dict,
    output_path: str = "results/figures/fig1_degradation.png"
):
    """
    Generate bar chart showing degradation by conflict type.

    Args:
        analysis: Analysis results from experiment
        output_path: Path to save figure
    """
    print_section("Generating Figure 1: Degradation by Conflict Type")

    # Extract data by conflict type
    conflict_stats = analysis['by_conflict_type']

    # Sort by conflict type name for consistency
    conflict_stats = sorted(conflict_stats, key=lambda x: x['conflict_type'])

    # Prepare data
    conflict_types = [s['conflict_type'].replace('_', ' ').title() for s in conflict_stats]
    full_means = [s['mean_full'] for s in conflict_stats]
    compressed_means = [s['mean_compressed'] for s in conflict_stats]
    full_stds = [s['std_full'] for s in conflict_stats]
    compressed_stds = [s['std_compressed'] for s in conflict_stats]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set positions
    x = np.arange(len(conflict_types))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, full_means, width, yerr=full_stds,
                   label='Full Context', capsize=5, color='#2E7D32', alpha=0.8)
    bars2 = ax.bar(x + width/2, compressed_means, width, yerr=compressed_stds,
                   label='Compressed Context (50%)', capsize=5, color='#C62828', alpha=0.8)

    # Customize
    ax.set_ylabel('Mean Instruction-Following Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Conflict Type', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1: KV Cache Compression Degrades Instruction Following',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conflict_types, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels on bars
    def autolabel(bars):
        """Attach a text label above each bar displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")

    # Also save as PDF for paper
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved PDF to {pdf_path}")

    plt.close()


def generate_overall_comparison(
    analysis: dict,
    output_path: str = "results/figures/fig1_overall.png"
):
    """
    Generate overall comparison bar chart.

    Args:
        analysis: Analysis results from experiment
        output_path: Path to save figure
    """
    print("\nGenerating overall comparison chart...")

    overall = analysis['overall']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data
    conditions = ['Full Context', 'Compressed\n(50%)']
    means = [overall['full_mean'], overall['compressed_mean']]
    stds = [overall['full_std'], overall['compressed_std']]

    # Create bars
    colors = ['#2E7D32', '#C62828']
    bars = ax.bar(conditions, means, yerr=stds, capsize=10, color=colors, alpha=0.8)

    # Customize
    ax.set_ylabel('Mean Instruction-Following Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance: Full vs. Compressed Context',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add degradation annotation
    degradation = overall['mean_degradation']
    degradation_pct = overall['mean_degradation_pct']
    ax.annotate(f'Degradation:\n{degradation:.3f}\n({degradation_pct:.1f}%)',
                xy=(0.5, min(means)/2), xycoords='data',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")

    plt.close()


def generate_table(
    analysis: dict,
    output_path: str = "results/tables/table1_compression.csv"
):
    """
    Generate CSV table with mean scores and standard deviations.

    Args:
        analysis: Analysis results from experiment
        output_path: Path to save table
    """
    print_section("Generating Table 1: Performance Summary")

    # Prepare data for table
    rows = []

    # Overall row
    overall = analysis['overall']
    rows.append({
        'Conflict Type': 'OVERALL',
        'Count': '',
        'Full Context Mean': f"{overall['full_mean']:.3f}",
        'Full Context Std': f"{overall['full_std']:.3f}",
        'Compressed Mean': f"{overall['compressed_mean']:.3f}",
        'Compressed Std': f"{overall['compressed_std']:.3f}",
        'Degradation': f"{overall['mean_degradation']:.3f}",
        'Degradation %': f"{overall['mean_degradation_pct']:.1f}%"
    })

    # Add empty row
    rows.append({k: '' for k in rows[0].keys()})

    # By conflict type
    for stat in sorted(analysis['by_conflict_type'], key=lambda x: x['conflict_type']):
        rows.append({
            'Conflict Type': stat['conflict_type'].replace('_', ' ').title(),
            'Count': stat['count'],
            'Full Context Mean': f"{stat['mean_full']:.3f}",
            'Full Context Std': f"{stat['std_full']:.3f}",
            'Compressed Mean': f"{stat['mean_compressed']:.3f}",
            'Compressed Std': f"{stat['std_compressed']:.3f}",
            'Degradation': f"{stat['mean_degradation']:.3f}",
            'Degradation %': f"{(stat['mean_degradation']/stat['mean_full']*100):.1f}%" if stat['mean_full'] > 0 else 'N/A'
        })

    # Add statistical test results
    rows.append({k: '' for k in rows[0].keys()})
    rows.append({
        'Conflict Type': 'STATISTICAL TEST',
        'Count': '',
        'Full Context Mean': '',
        'Full Context Std': '',
        'Compressed Mean': '',
        'Compressed Std': '',
        'Degradation': '',
        'Degradation %': ''
    })

    stat_test = analysis['statistical_test']
    rows.append({
        'Conflict Type': 'Paired t-test',
        'Count': '',
        'Full Context Mean': '',
        'Full Context Std': '',
        'Compressed Mean': f"t={stat_test['t_statistic']:.4f}",
        'Compressed Std': f"p={stat_test['p_value']:.6f}",
        'Degradation': 'Significant' if stat_test['significant'] else 'Not Significant',
        'Degradation %': ''
    })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved table to {output_path}")

    # Also print to console
    print("\nTable 1: Compression Performance Summary")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def main():
    """Main visualization generator"""
    print_section("Experiment 1 Visualization Generator")

    # Load analysis results
    analysis_path = "results/exp1_analysis.json"
    print(f"Loading analysis from {analysis_path}...")
    analysis = load_json(analysis_path)

    # Generate visualizations
    generate_bar_chart(analysis)
    generate_overall_comparison(analysis)

    # Generate table
    generate_table(analysis)

    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - results/figures/fig1_degradation.png")
    print("  - results/figures/fig1_degradation.pdf")
    print("  - results/figures/fig1_overall.png")
    print("  - results/tables/table1_compression.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
