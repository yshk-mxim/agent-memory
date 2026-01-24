# Claude AI Judge Evaluation Prompts

**Purpose**: Use Claude Sonnet 4.5 as an automated AI judge for qualitative evaluation
**Model**: `claude-sonnet-4-5-20250929`
**Temperature**: 0.0 (reproducibility)
**Integration**: Via Claude Code CLI Task tool or direct API

---

## Evaluation Framework

Claude evaluates three dimensions on a 0-5 scale:
1. **Contamination** (Lower is better: 0 = clean, 5 = severe contamination)
2. **Specialization** (Higher is better: 0 = generic, 5 = highly specialized)
3. **Synthesis Quality** (Higher is better: 0 = poor, 5 = excellent)

---

## Prompt 1: Contamination Detection

**Purpose**: Detect if specialist outputs leak concepts/terminology across domains

```
You are evaluating the quality of a multi-agent system's output isolation.

**Task**: Assess CONTAMINATION between two specialist outputs.

**Definition**: Contamination occurs when Output A contains concepts, terminology, or style that should only appear in Output B (or vice versa).

**Inputs**:
- **Cluster 1 Purpose**: {cluster_1_description}
- **Cluster 2 Purpose**: {cluster_2_description}
- **Output A (Cluster 1)**: {output_1}
- **Output B (Cluster 2)**: {output_2}

**Evaluation Criteria**:
- **0 (Clean)**: Outputs are completely isolated, no cross-domain leakage
- **1 (Minimal)**: 1-2 minor terms appear in wrong output (e.g., "user" in technical doc)
- **2 (Low)**: 3-5 terms or concepts leak, but outputs remain distinct
- **3 (Moderate)**: Significant overlap in vocabulary or concepts (10-15 terms)
- **4 (High)**: Outputs blend together, hard to distinguish purposes (>20 terms)
- **5 (Severe)**: Complete mixing, outputs are indistinguishable

**Output Format (JSON)**:
```json
{
  "contamination_score": <0-5>,
  "evidence": {
    "cluster_1_leakage": ["example term 1", "example term 2"],
    "cluster_2_leakage": ["example term 3", "example term 4"]
  },
  "reasoning": "<2-3 sentence explanation>",
  "examples": "<1-2 specific quotes showing contamination or cleanliness>"
}
```

**Instructions**:
1. Read both outputs carefully
2. Identify the intended domain/purpose of each
3. Look for terminology, concepts, or style that crosses boundaries
4. Assign contamination score (0-5)
5. Provide evidence and reasoning

Return ONLY the JSON object, no additional text.
```

---

## Prompt 2: Specialization Quality

**Purpose**: Assess how well each specialist focuses on its specific role

```
You are evaluating the specialization quality of agent outputs in a multi-agent system.

**Task**: Assess how well each specialist output demonstrates SPECIALIZATION in its intended domain.

**Definition**: Specialization means the output shows deep focus, appropriate terminology, and domain-specific insights aligned with its purpose.

**Inputs**:
- **Cluster Purpose**: {cluster_description}
- **Example Cluster**: {cluster_type} (e.g., "technical", "business", "creative")
- **Output**: {output_text}

**Evaluation Criteria**:
- **0 (Generic)**: Output is vague, could apply to any domain, no specialization
- **1 (Weak)**: Minimal domain-specific content, mostly generic observations
- **2 (Basic)**: Some domain-appropriate terms, but lacks depth (surface-level)
- **3 (Moderate)**: Good domain focus, appropriate terminology, decent depth
- **4 (Strong)**: Clear specialization, rich domain vocabulary, insightful analysis
- **5 (Excellent)**: Exceptional domain expertise, precise terminology, deep insights

**Output Format (JSON)**:
```json
{
  "specialization_score": <0-5>,
  "evidence": {
    "domain_keywords": ["keyword1", "keyword2", "keyword3"],
    "depth_indicators": ["demonstrates understanding of X", "provides Y-level detail"],
    "weaknesses": ["lacks Z", "could improve W"]
  },
  "reasoning": "<2-3 sentence explanation>",
  "examples": "<1-2 specific quotes showing specialization quality>"
}
```

**Instructions**:
1. Understand the intended cluster purpose
2. Evaluate domain-appropriate terminology usage
3. Assess depth of domain-specific insights
4. Assign specialization score (0-5)
5. Provide evidence and reasoning

Return ONLY the JSON object, no additional text.
```

---

## Prompt 3: Synthesis Quality

**Purpose**: Evaluate how well the coordinator integrates specialist perspectives

```
You are evaluating the synthesis quality of a coordinator agent that integrates multiple specialist outputs.

**Task**: Assess how well the SYNTHESIS output integrates perspectives from multiple specialists.

**Definition**: Good synthesis preserves key information from all specialists, creates coherent narrative, adds integration insights, and avoids redundancy.

**Inputs**:
- **Specialist Output A**: {specialist_1_output}
- **Specialist Output B**: {specialist_2_output}
- **Synthesis Output**: {synthesis_output}

**Evaluation Criteria**:
- **0 (Failed)**: Synthesis ignores one or both specialists, or just concatenates
- **1 (Poor)**: Mentions both but superficially, no real integration
- **2 (Basic)**: Covers both specialists but separately, minimal integration
- **3 (Moderate)**: Good coverage, some integration, occasional insights
- **4 (Strong)**: Excellent integration, clear connections, added perspective
- **5 (Exceptional)**: Masterful synthesis, deep integration, novel insights, coherent narrative

**Quality Dimensions**:
1. **Coverage**: Does synthesis include key points from both specialists? (0-100%)
2. **Integration**: Are perspectives connected/contrasted meaningfully?
3. **Coherence**: Does synthesis flow naturally, not just list points?
4. **Added Value**: Does synthesis provide insights beyond concatenation?

**Output Format (JSON)**:
```json
{
  "synthesis_score": <0-5>,
  "dimensions": {
    "coverage_percent": <0-100>,
    "integration_quality": "<poor/fair/good/excellent>",
    "coherence_quality": "<poor/fair/good/excellent>",
    "added_value": "<none/minimal/moderate/substantial>"
  },
  "evidence": {
    "specialist_a_coverage": ["point 1 included", "point 2 included"],
    "specialist_b_coverage": ["point 3 included", "point 4 included"],
    "integration_examples": ["connects X from A with Y from B"],
    "novel_insights": ["insight 1", "insight 2"]
  },
  "reasoning": "<2-3 sentence explanation>",
  "examples": "<1-2 specific quotes showing synthesis quality>"
}
```

**Instructions**:
1. Read both specialist outputs to understand their key points
2. Read synthesis output
3. Assess coverage of both specialists
4. Evaluate quality of integration
5. Check for coherence and added value
6. Assign synthesis score (0-5)
7. Provide evidence and reasoning

Return ONLY the JSON object, no additional text.
```

---

## Implementation Notes

### Via Claude Code CLI (Recommended)

Use the Task tool with Claude:

```python
from antml_function_calls import Task

result = Task(
    subagent_type="general-purpose",
    model="sonnet",  # Claude Sonnet 4.5
    description="Evaluate contamination",
    prompt=contamination_prompt.format(
        cluster_1_description=desc1,
        cluster_2_description=desc2,
        output_1=output1,
        output_2=output2
    )
)
```

**Benefits**:
- No sandbox bypass required
- Automatic credential management
- Built-in error handling

### Via Direct API (Alternative)

```python
from anthropic import Anthropic
import json

client = Anthropic(api_key=config.claude_api_key)

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1000,
    temperature=0.0,  # Reproducibility
    messages=[{
        "role": "user",
        "content": prompt
    }]
)

# Parse JSON response
result = json.loads(response.content[0].text)
```

### Parsing Response

```python
def parse_claude_judgment(response_text: str) -> dict:
    """Extract JSON from Claude's response"""
    # Handle markdown code blocks
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()

    return json.loads(json_str)
```

---

## Aggregation Strategy

For each condition (Sequential, Prompted, Turn-based, Semantic):

1. **Per-Example Scores**: Claude evaluates each of 50 examples
2. **Aggregate Metrics**: Compute mean, std dev, median per condition
3. **Comparison**: Compare aggregated scores across conditions

**Expected Pattern**:
- **Sequential**: High contamination (3-4), low specialization (1-2), poor synthesis (1-2)
- **Prompted**: Moderate contamination (2-3), moderate specialization (2-3), fair synthesis (2-3)
- **Turn-based**: Low contamination (1-2), moderate specialization (2-3), fair synthesis (2-3)
- **Semantic (RDIC)**: Minimal contamination (0-1), high specialization (4-5), excellent synthesis (4-5)

---

## Cost Estimation

**Per evaluation**:
- Input: ~500 tokens (prompts + outputs)
- Output: ~200 tokens (JSON response)
- Cost: ~$0.05 per evaluation

**Total for 200 outputs × 3 evaluations**:
- 600 evaluations total
- Cost: ~$30

**Budget-friendly alternative**:
- Use Claude Haiku 4.5 for initial screening (10x cheaper)
- Use Sonnet 4.5 for final evaluation or subset validation

---

## Validation

### Inter-Rater Reliability with Mechanical Metrics

Compute correlation between Claude judgments and mechanical metrics:

```python
# Contamination
correlation(claude_contamination, tfidf_similarity)  # Expect r > 0.7

# Specialization
correlation(claude_specialization, keyword_density)  # Expect r > 0.6

# Synthesis
correlation(claude_synthesis, information_coverage)  # Expect r > 0.6
```

**Interpretation**:
- r > 0.7: Strong agreement (Claude validates mechanical metrics)
- r = 0.4-0.7: Moderate agreement (Claude captures additional nuance)
- r < 0.4: Weak agreement (investigate discrepancies)

---

## Advantages of Claude as AI Judge

1. **Qualitative Assessment**: Captures nuances mechanical metrics miss
2. **Reproducible**: Temperature=0 gives consistent scores
3. **Scalable**: Can evaluate 200 outputs in ~1 hour
4. **Cost-Effective**: $30 vs $300+ for human raters
5. **Explainable**: Provides reasoning and evidence for scores
6. **Accepted**: Recent research shows LLM-as-judge correlates well with humans
7. **No Rater Dependency**: No recruitment, training, or scheduling needed

---

## Literature Support

Recent papers validating LLM-as-judge:
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- "Can Large Language Models Be Reliable Evaluators?" (2024)
- "LLM-Eval: Unified Multi-Dimensional Automatic Evaluation" (2024)

**Key findings**:
- Claude/GPT-4 correlation with human judgments: r = 0.85-0.90
- Higher agreement than inter-human rater agreement in some tasks
- Cost reduction: 10-100x vs human evaluation

---

**Created**: 2026-01-23
**Purpose**: Add AI judge capability to automated evaluation strategy
**Integration**: Sprint 01 (implementation), Sprint 02 (execution), Sprint 03 (analysis)
