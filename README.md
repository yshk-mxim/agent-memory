# RDIC: Reflective Debate with Instruction Conflicts

A novel semantic debate framework for training language models to handle conflicting instructions through multi-agent deliberation and semantic clustering.

## Overview

RDIC (Reflective Debate with Instruction Conflicts) is a research project that explores how language models can better handle contradictory or incompatible user instructions by engaging in structured debates between specialized agents with different semantic perspectives.

### Key Features

- **Instruction Conflict Dataset**: Curated multi-turn conversations with genuine semantic conflicts across 5 categories
- **Multi-Agent Debate Framework**: Specialized agents representing different constraint spaces
- **Semantic Clustering**: Ground-truth clustering of incompatible instruction sets
- **Automated Generation Pipeline**: Claude-powered dataset generation with quality validation
- **Comprehensive Evaluation**: Metrics for conflict detection and resolution quality

## Project Status

- ✅ **Day 1**: Environment setup and API integration
- ✅ **Day 2**: Dataset design and first batch generation (30 examples, 100% quality)
- 🔄 **Day 3-21**: Scaling, training, and evaluation (in progress)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RDIC Framework                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │  Dataset   │→ │   Debate   │→ │   Evaluation    │  │
│  │ Generation │  │  Framework │  │   & Metrics     │  │
│  └────────────┘  └────────────┘  └─────────────────┘  │
│        ↓               ↓                    ↓           │
│  ┌────────────────────────────────────────────────┐   │
│  │          Semantic Clustering Engine            │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Conflict Types

The framework handles five categories of instruction conflicts:

1. **Tone Conflicts**: formal vs casual, professional vs friendly, serious vs humorous
2. **Detail Conflicts**: brief vs detailed, concise vs comprehensive, summary vs exhaustive
3. **Style Conflicts**: technical vs layperson, academic vs conversational, jargon vs simple
4. **Content Conflicts**: citations vs no citations, examples vs no examples, opinions vs facts
5. **Format Conflicts**: structured vs freeform, bullets vs paragraphs, lists vs prose

## Installation

### Prerequisites

- Python 3.8+
- Claude API key (Anthropic)
- DeepSeek API key (optional, for DeepSeek R1)

### Setup

```bash
# Clone the repository
git clone https://github.com/yshk-mxim/rdic.git
cd rdic

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env.json.example env.json
# Edit env.json with your API keys
```

### Environment Configuration

Create `env.json` in the project root:

```json
{
  "claude_api_key": "sk-ant-api03-...",
  "deepseek_api_key": "sk-..."
}
```

## Usage

### Generate Conflict Examples

```bash
# Generate 30 examples with validation
python -m src.dataset_generator --num 30 --output data/batch_001.json --validate

# Generate 100 examples distributed evenly across conflict types
python -m src.dataset_generator --num 100 --output data/batch_002.json
```

### Test API Connections

```bash
# Verify all API integrations
python -m tests.test_apis
```

### Configuration

API configuration is managed through `src/config.py`:

```python
from src.config import get_config

config = get_config()
claude_key = config.claude_api_key
deepseek_key = config.deepseek_api_key
```

## API Models Used

The project uses the latest Claude and DeepSeek models:

- **Claude Haiku 4.5**: `claude-haiku-4-5-20251001` (fast, cost-effective)
- **Claude Sonnet 4.5**: `claude-sonnet-4-5-20250929` (primary model for generation)
- **DeepSeek R1**: `deepseek-reasoner` (reasoning-focused alternative)

See [CLAUDE.md](CLAUDE.md) for detailed API usage and model specifications.

## Project Structure

```
rdic/
├── src/
│   ├── __init__.py
│   ├── config.py              # API configuration management
│   ├── dataset_generator.py   # Conflict example generation
│   └── utils.py               # Shared utilities
├── data/
│   ├── conflict_schema.json   # Dataset schema definition
│   ├── batch_001_review.md    # Quality review reports
│   └── batch_*.json           # Generated datasets (gitignored)
├── tests/
│   └── test_apis.py           # API integration tests
├── plans/
│   └── day_*.md               # Daily implementation plans
├── requirements.txt           # Python dependencies
├── env.json                   # API keys (gitignored)
└── README.md                  # This file
```

## Dataset Schema

Each conflict example follows this structure:

```json
{
  "id": "conflict_001",
  "conflict_type": "tone_formal_vs_casual",
  "domain": "business_email",
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "instruction": "Use formal, professional language...",
      "content": "I need help drafting emails..."
    },
    {
      "turn_id": 2,
      "role": "user",
      "instruction": "Write in a casual, friendly tone...",
      "content": "Draft an email about..."
    },
    {
      "turn_id": 3,
      "role": "user",
      "query": "Write the email using both styles",
      "expected_conflict": "Cannot simultaneously be formal and casual..."
    }
  ],
  "ground_truth_clusters": [
    "formal_professional_constrained",
    "casual_friendly_creative"
  ],
  "metadata": {
    "generated_at": "2026-01-22T09:20:50.954220",
    "model": "claude-sonnet-4-5-20250929",
    "reviewed": false,
    "quality_score": null
  }
}
```

## Results

### Day 2 Metrics

- **Examples Generated**: 30/30 (100% success rate)
- **Quality Score**: 100% genuine conflicts (target: 70%)
- **Conflict Type Distribution**: Perfect 20% per type
- **Domain Coverage**: 10/10 domains represented
- **Generation Time**: ~8 seconds per example

### Quality Criteria

Each example is validated for:
- ✅ Genuine semantic conflict (mutually exclusive constraints)
- ✅ Realistic scenario (plausible real-world context)
- ✅ Unavoidable conflict (final query requires both constraints)
- ✅ Clear ground truth clusters (well-defined semantic spaces)

## Development Timeline

### Week 1: Dataset & Foundation
- Day 1: ✅ Environment setup, API integration
- Day 2: ✅ Dataset design, schema, first batch (30 examples)
- Day 3: Dataset scaling (300+ examples)
- Day 4: Quality filtering and validation
- Day 5: Baseline model selection
- Day 6: Semantic router implementation
- Day 7: Week 1 review and checkpoint

### Week 2: Debate Framework
- Days 8-14: Multi-agent debate system implementation

### Week 3: Training & Evaluation
- Days 15-21: Model training, evaluation, and documentation

See [complete_plan.md](complete_plan.md) for full timeline.

## Contributing

This is a research project. For questions or collaboration:
- Open an issue on GitHub
- Review the daily status reports in `DAY_*_STATUS.md`

## License

[Add your license here]

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{rdic2026,
  title={RDIC: Reflective Debate with Instruction Conflicts},
  author={[Your Name]},
  year={2026},
  url={https://github.com/yshk-mxim/rdic}
}
```

## Acknowledgments

- Built with [Anthropic Claude API](https://www.anthropic.com/)
- DeepSeek R1 integration for reasoning experiments
- llama.cpp for local Gemma 3 inference

---

**Status**: Active Development | **Last Updated**: 2026-01-22 | **Version**: 0.2.0
