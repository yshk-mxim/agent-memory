"""
Experiment 2 Validation: Compare R1 Clusters to Ground Truth

Validates discovered clusters against ground truth conflict types.
Measures alignment, generates confusion matrix, and creates visualizations.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_dataset_ground_truth(dataset_path: str = "data/conflict_dataset.json") -> Dict[str, str]:
    """
    Load ground truth conflict types from dataset.

    Returns:
        Dict mapping conversation ID to conflict type
    """
    dataset = load_json(dataset_path)

    ground_truth = {}
    for example in dataset:
        conv_id = example['id']
        conflict_type = example['conflict_type']
        ground_truth[conv_id] = conflict_type

    return ground_truth


def load_r1_clusters(run_path: str) -> Dict[str, Any]:
    """
    Load R1 clustering results from a run.

    Returns:
        Dict with clusters and assignments
    """
    return load_json(run_path)


def create_cluster_mapping(r1_clusters: Dict[str, Any], ground_truth: Dict[str, str]) -> Tuple[Dict, Dict]:
    """
    Create mapping between R1 clusters and GT conflict types.

    Uses majority voting: each R1 cluster is assigned to the GT type
    that appears most frequently in its members.

    Returns:
        - cluster_to_gt: Dict mapping cluster name to GT type
        - gt_to_clusters: Dict mapping GT type to list of clusters
    """
    # Get conversation assignments from R1
    assignments = r1_clusters['clusters']['conversation_assignments']

    # Count GT types in each cluster
    cluster_gt_counts = {}

    for assignment in assignments:
        conv_id = assignment['conv_id']

        # Get all clusters this conversation is assigned to
        turn_clusters = assignment.get('turn_clusters', [])

        # Get GT type for this conversation
        gt_type = ground_truth.get(conv_id, 'unknown')

        # Count for each cluster
        for cluster_name in set(turn_clusters):
            if cluster_name not in cluster_gt_counts:
                cluster_gt_counts[cluster_name] = {}

            if gt_type not in cluster_gt_counts[cluster_name]:
                cluster_gt_counts[cluster_name][gt_type] = 0

            cluster_gt_counts[cluster_name][gt_type] += 1

    # Assign each cluster to majority GT type
    cluster_to_gt = {}
    for cluster_name, gt_counts in cluster_gt_counts.items():
        majority_gt = max(gt_counts.items(), key=lambda x: x[1])[0]
        cluster_to_gt[cluster_name] = majority_gt

    # Reverse mapping
    gt_to_clusters = {}
    for cluster_name, gt_type in cluster_to_gt.items():
        if gt_type not in gt_to_clusters:
            gt_to_clusters[gt_type] = []
        gt_to_clusters[gt_type].append(cluster_name)

    return cluster_to_gt, gt_to_clusters


def compute_alignment_metrics(
    r1_clusters: Dict[str, Any],
    ground_truth: Dict[str, str],
    cluster_to_gt: Dict[str, str]
) -> Dict[str, Any]:
    """
    Compute alignment metrics between R1 clusters and GT.

    Metrics:
    - Overall accuracy
    - Precision, recall, F1 per GT type
    - Confusion matrix
    """
    assignments = r1_clusters['clusters']['conversation_assignments']

    # Predict GT type for each conversation based on R1 clusters
    y_true = []
    y_pred = []

    for assignment in assignments:
        conv_id = assignment['conv_id']
        gt_type = ground_truth.get(conv_id, 'unknown')

        # Get primary cluster (most frequent in turn_clusters)
        turn_clusters = assignment.get('turn_clusters', [])
        if turn_clusters:
            # Use most common cluster
            from collections import Counter
            cluster_counts = Counter(turn_clusters)
            primary_cluster = cluster_counts.most_common(1)[0][0]

            # Map to GT type
            predicted_gt = cluster_to_gt.get(primary_cluster, 'unknown')
        else:
            predicted_gt = 'unknown'

        y_true.append(gt_type)
        y_pred.append(predicted_gt)

    # Get unique labels
    labels = sorted(set(y_true + y_pred))

    # Compute metrics
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = []
    for i, label in enumerate(labels):
        per_class.append({
            'gt_type': label,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(sum(1 for t in y_true if t == label))
        })

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'per_class': per_class,
        'y_true': y_true,
        'y_pred': y_pred
    }


def generate_confusion_matrix_heatmap(
    confusion_matrix: List[List[int]],
    labels: List[str],
    output_path: str = "results/figures/fig2_cluster_heatmap.png"
):
    """Generate confusion matrix heatmap (Figure 2)."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )

    plt.title('R1 Cluster-Ground Truth Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted (R1 Cluster → GT Type)', fontsize=12)
    plt.ylabel('Actual (Ground Truth Type)', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved confusion matrix heatmap to {output_path}")


def generate_cluster_taxonomy_table(
    r1_clusters: Dict[str, Any],
    cluster_to_gt: Dict[str, str],
    output_path: str = "results/tables/table2_taxonomy.csv"
):
    """Generate cluster taxonomy table (Table 2)."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    instruction_clusters = r1_clusters['clusters']['instruction_clusters']

    rows = []
    for cluster in instruction_clusters:
        cluster_id = cluster['id']
        description = cluster['description']
        conflicts_with = ', '.join(cluster.get('conflicts_with', []))
        example_count = len(cluster.get('example_instructions', []))
        mapped_gt = cluster_to_gt.get(cluster_id, 'N/A')

        rows.append({
            'R1 Cluster ID': cluster_id,
            'Description': description,
            'Conflicts With': conflicts_with,
            'Example Count': example_count,
            'Mapped GT Type': mapped_gt
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"✓ Saved cluster taxonomy table to {output_path}")


def analyze_misalignments(
    y_true: List[str],
    y_pred: List[str],
    assignments: List[Dict]
) -> List[Dict]:
    """
    Analyze cases where R1 clusters disagree with GT.

    Returns list of misalignment examples with explanations.
    """
    misalignments = []

    for i, (true_type, pred_type, assignment) in enumerate(zip(y_true, y_pred, assignments)):
        if true_type != pred_type:
            misalignments.append({
                'conv_id': assignment['conv_id'],
                'gt_type': true_type,
                'predicted_type': pred_type,
                'r1_clusters': assignment.get('turn_clusters', [])
            })

    return misalignments


def main():
    """Main validation pipeline."""
    print("="*60)
    print("Experiment 2 Validation: R1 Clusters vs Ground Truth")
    print("="*60)

    # Load ground truth
    print("\nLoading ground truth from dataset...")
    ground_truth = load_dataset_ground_truth("data/conflict_dataset.json")
    print(f"✓ Loaded {len(ground_truth)} examples")

    # Load R1 consensus clusters (from Day 9)
    print("\nLoading R1 consensus clusters...")
    r1_clusters = load_r1_clusters("results/r1_consensus_clusters.json")
    print(f"✓ Loaded {len(r1_clusters['clusters']['instruction_clusters'])} clusters")

    # Create cluster mapping
    print("\nMapping R1 clusters to GT conflict types...")
    cluster_to_gt, gt_to_clusters = create_cluster_mapping(r1_clusters, ground_truth)

    print(f"✓ Mapped {len(cluster_to_gt)} R1 clusters")
    print("\nMapping:")
    for cluster, gt in sorted(cluster_to_gt.items()):
        print(f"  {cluster} → {gt}")

    # Compute alignment metrics
    print("\nComputing alignment metrics...")
    metrics = compute_alignment_metrics(r1_clusters, ground_truth, cluster_to_gt)

    print(f"\n{'='*60}")
    print("Alignment Results:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"\nPer-Class Metrics:")
    for cls in metrics['per_class']:
        print(f"\n  {cls['gt_type']}:")
        print(f"    Precision: {cls['precision']:.3f}")
        print(f"    Recall: {cls['recall']:.3f}")
        print(f"    F1: {cls['f1']:.3f}")
        print(f"    Support: {cls['support']}")

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")

    generate_confusion_matrix_heatmap(
        metrics['confusion_matrix'],
        metrics['labels']
    )

    generate_cluster_taxonomy_table(
        r1_clusters,
        cluster_to_gt
    )

    # Analyze misalignments
    print("\nAnalyzing misalignments...")
    assignments = r1_clusters['clusters']['conversation_assignments']
    misalignments = analyze_misalignments(
        metrics['y_true'],
        metrics['y_pred'],
        assignments
    )

    print(f"✓ Found {len(misalignments)} misalignments ({len(misalignments)/len(ground_truth)*100:.1f}%)")

    # Save full results
    print("\nSaving results...")
    results = {
        'cluster_to_gt_mapping': cluster_to_gt,
        'gt_to_clusters_mapping': gt_to_clusters,
        'alignment_metrics': metrics,
        'misalignments': misalignments
    }

    save_json(results, "results/cluster_gt_alignment.json")
    print("✓ Saved to results/cluster_gt_alignment.json")

    # Check success criteria
    print(f"\n{'='*60}")
    print("Success Criteria:")
    print(f"  Alignment >70%: {metrics['accuracy']:.2%} {'✓' if metrics['accuracy'] > 0.7 else '✗'}")

    # Check for diagonal pattern in confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    diagonal_sum = np.trace(cm)
    total_sum = np.sum(cm)
    diagonal_ratio = diagonal_sum / total_sum if total_sum > 0 else 0
    print(f"  Clear diagonal pattern: {diagonal_ratio:.2%} {'✓' if diagonal_ratio > 0.6 else '✗'}")

    print(f"\n{'='*60}")
    print("Validation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
