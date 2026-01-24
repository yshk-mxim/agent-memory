"""
Experiment 2: DeepSeek R1 Instruction Clustering

Discovers semantic instruction clusters using R1's reasoning capabilities.
Runs multiple times to test stability.
"""

import json
import os
from typing import Dict, List, Any
from openai import OpenAI


def load_dataset(path: str = "data/conflict_dataset.json") -> List[Dict[str, Any]]:
    """Load the conflict dataset."""
    with open(path, 'r') as f:
        return json.load(f)


def build_r1_prompt(examples: List[Dict[str, Any]]) -> str:
    """
    Build the R1 clustering prompt.

    Asks R1 to:
    1. Identify semantic instruction clusters
    2. Determine which clusters conflict
    3. Assign each conversation to clusters
    """

    # Format conversations for R1
    conversations = []
    for i, ex in enumerate(examples):
        conv = {
            "id": ex.get("id", f"conv_{i:03d}"),
            "turns": []
        }

        for turn in ex["conversation"]:
            if turn["role"] == "user":
                conv["turns"].append({
                    "role": "user",
                    "content": turn["content"]
                })

        conversations.append(conv)

    prompt = f"""Analyze these {len(conversations)} multi-turn conversations where instructions may conflict.

Your task is to discover the minimal set of **instruction clusters** needed to represent the semantic space of instructions, where:
- Instructions within the same cluster are semantically compatible
- Instructions across different clusters may conflict or interfere with each other

For each conversation, examine all user instructions and:
1. Identify the semantic characteristics of each instruction
2. Group instructions into clusters based on their semantic properties
3. Determine which clusters semantically conflict with each other

**Example instruction clusters might include:**
- Formal/professional tone vs casual/friendly tone
- Brief/concise vs detailed/comprehensive
- Technical/jargon vs simple/layperson language
- Example-based vs abstract explanation
- Structured/bullets vs narrative/paragraphs

**Output format (JSON only, no additional text):**
```json
{{
  "instruction_clusters": [
    {{
      "id": "cluster_name",
      "description": "semantic characteristics of this cluster",
      "conflicts_with": ["list", "of", "cluster", "names"],
      "example_instructions": ["example 1", "example 2", "example 3"]
    }}
  ],
  "conversation_assignments": [
    {{
      "conv_id": "conv_001",
      "turn_clusters": ["cluster_A", "cluster_B", "cluster_A"]
    }}
  ],
  "conflict_matrix": [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
  ],
  "reasoning_summary": "brief explanation of your clustering strategy"
}}
```

**Conversations:**

{json.dumps(conversations, indent=2)}

Now discover the instruction clusters and output the JSON."""

    return prompt


def call_deepseek_r1(
    prompt: str,
    model: str = "deepseek-reasoner",
    max_tokens: int = 8000
) -> Dict[str, Any]:
    """
    Call DeepSeek R1 API.

    Returns:
        Dict with 'reasoning' and 'content' keys
    """
    # Load API key from config
    from src.config import get_config
    config = get_config()

    client = OpenAI(
        api_key=config.env.get("deepseek_api_key"),
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0  # Deterministic for first run
    )

    # Extract reasoning and content
    message = response.choices[0].message

    result = {
        "reasoning": getattr(message, 'reasoning_content', ''),
        "content": message.content,
        "model": model,
        "usage": response.usage._asdict() if response.usage else {}
    }

    return result


def parse_r1_output(content: str) -> Dict[str, Any]:
    """Parse R1's JSON output."""

    # R1 might wrap in markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def run_clustering_iteration(
    dataset: List[Dict[str, Any]],
    run_number: int,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run one iteration of R1 clustering.

    Args:
        dataset: Full dataset
        run_number: Run number (1, 2, 3)
        output_dir: Where to save results

    Returns:
        Clustering results with metadata
    """
    print(f"\n=== R1 Clustering Run {run_number} ===\n")

    # Build prompt
    print("Building R1 prompt...")
    prompt = build_r1_prompt(dataset)
    print(f"Prompt length: {len(prompt)} characters")

    # Call R1
    print(f"Calling DeepSeek R1 (model: deepseek-reasoner)...")
    response = call_deepseek_r1(prompt)

    print(f"Reasoning trace length: {len(response['reasoning'])} chars")
    print(f"Response length: {len(response['content'])} chars")

    # Parse output
    print("Parsing R1 output...")
    clusters = parse_r1_output(response['content'])

    # Add metadata
    result = {
        "run_number": run_number,
        "clusters": clusters,
        "reasoning_trace": response['reasoning'],
        "usage": response['usage'],
        "num_clusters": len(clusters.get("instruction_clusters", [])),
        "num_conversations": len(dataset)
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save main results
    output_file = f"{output_dir}/r1_run{run_number}_clusters.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved to {output_file}")

    # Save reasoning trace separately (can be large)
    reasoning_file = f"{output_dir}/r1_run{run_number}_reasoning.txt"
    with open(reasoning_file, 'w') as f:
        f.write(response['reasoning'])
    print(f"✓ Saved reasoning to {reasoning_file}")

    # Print summary
    print(f"\nRun {run_number} Summary:")
    print(f"  Clusters discovered: {result['num_clusters']}")
    print(f"  Conversations analyzed: {result['num_conversations']}")
    if 'usage' in result and result['usage']:
        usage = result['usage']
        print(f"  Tokens used: {usage.get('total_tokens', 'N/A')}")

    return result


def analyze_clusters(clusters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cluster quality metrics."""

    analysis = {
        "num_clusters": len(clusters.get("instruction_clusters", [])),
        "cluster_names": [c["id"] for c in clusters.get("instruction_clusters", [])],
        "avg_examples_per_cluster": 0,
        "conflict_density": 0.0
    }

    # Average examples per cluster
    if clusters.get("instruction_clusters"):
        total_examples = sum(
            len(c.get("example_instructions", []))
            for c in clusters["instruction_clusters"]
        )
        analysis["avg_examples_per_cluster"] = total_examples / len(clusters["instruction_clusters"])

    # Conflict density (proportion of cluster pairs that conflict)
    if "conflict_matrix" in clusters and clusters["conflict_matrix"]:
        matrix = clusters["conflict_matrix"]
        n = len(matrix)
        if n > 1:
            total_pairs = n * (n - 1) / 2
            conflicting_pairs = sum(
                matrix[i][j]
                for i in range(n)
                for j in range(i+1, n)
            )
            analysis["conflict_density"] = conflicting_pairs / total_pairs

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DeepSeek R1 clustering")
    parser.add_argument("--run", type=int, default=1, help="Run number (1, 2, or 3)")
    parser.add_argument("--dataset", type=str, default="data/conflict_dataset.json")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

    # Run clustering
    result = run_clustering_iteration(dataset, args.run, args.output_dir)

    # Analyze results
    print("\nAnalyzing clusters...")
    analysis = analyze_clusters(result["clusters"])

    print("\nCluster Analysis:")
    print(f"  Number of clusters: {analysis['num_clusters']}")
    print(f"  Cluster names: {', '.join(analysis['cluster_names'])}")
    print(f"  Avg examples/cluster: {analysis['avg_examples_per_cluster']:.1f}")
    print(f"  Conflict density: {analysis['conflict_density']:.2%}")

    print(f"\n✓ Run {args.run} complete!")
