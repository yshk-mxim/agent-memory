# Claude Code Guidelines

**IMPORTANT**: Current year is 2026. When searching for documentation, use 2026 or 2025-2026 in search queries to get the most recent information.

This document contains:
1. **Code Quality Standards** - Mandatory guidelines for all code
2. **AI Slop Prevention** - Patterns to avoid in AI-generated code
3. **Architecture Rules** - Hexagonal architecture compliance
4. **Claude API Usage** - API integration specifics

---

## Code Quality Standards (MANDATORY)

### Architecture: Hexagonal (Ports & Adapters)

**Dependency Rule**: All dependencies point INWARD. Domain has ZERO external imports.

```
INBOUND ADAPTERS → APPLICATION SERVICES → DOMAIN CORE ← OUTBOUND ADAPTERS
     (FastAPI)        (BatchEngine)        (BlockPool)      (MLX, safetensors)
```

**Layer Responsibilities**:

| Layer | Location | Can Import | Cannot Import |
|-------|----------|------------|---------------|
| **Domain** | `src/semantic/domain/` | stdlib only | mlx, fastapi, safetensors, numpy |
| **Application** | `src/semantic/application/` | domain, ports | mlx, fastapi, safetensors, numpy |
| **Ports** | `src/semantic/ports/` | domain, typing | Any infrastructure |
| **Adapters** | `src/semantic/adapters/` | Everything | - |

**CRITICAL RULE**: If you see `import mlx` or `import numpy` or `from safetensors` in domain/ or application/ - THIS IS A BUG. Fix it by moving to adapters.

### Code Complexity Limits

| Metric | Limit | Enforcement |
|--------|-------|-------------|
| Function length | ≤50 lines | Block merge |
| Cyclomatic complexity | ≤10 | Block merge |
| Nesting depth | ≤3 levels | Block merge |
| Class length | ≤300 lines | Warning |

### Type Safety

- Use `mypy --strict` compliance
- Prefer `str | None` over `Optional[str]`
- Avoid `Any` except at adapter boundaries
- All public functions must have type hints

---

## AI Slop Prevention (CRITICAL)

**Research shows AI-generated code has 1.7x more issues than human code (CodeRabbit 2025)**. Watch for these patterns:

### Red Flags - NEVER Do These

#### 1. Over-Commenting
```python
# BAD: Comments restating obvious code
# 1. Validate inputs
if model is None:
    raise ValueError("Model required")
# 2. Store the model
self._model = model

# GOOD: Self-documenting code, comment only non-obvious behavior
self._model = model  # Retained for hot-swap during model transitions
```

#### 2. Excessive Docstrings
```python
# BAD: 15-line docstring for trivial method
def is_empty(self) -> bool:
    """Check if empty.

    Returns:
        bool: True if empty, False otherwise.

    Example:
        >>> obj.is_empty()
        True
    """
    return self.tokens == 0

# GOOD: No docstring needed - name + type hint says it all
def is_empty(self) -> bool:
    return self.tokens == 0
```

**When Docstrings ARE Required**:
- Public API methods
- Non-obvious algorithms
- Methods with side effects
- Complex parameters

#### 3. Sprint/Ticket References in Code
```python
# BAD: These become meaningless
# Sprint 3.5 fix: CRITICAL-1
# NEW-5: Domain validation
# TODO (Day 7): Implement this

# GOOD: Use ADR references only
# See ADR-002: Universal 256-token block size
```

#### 4. Generic Variable Names
```python
# BAD
data = fetch_cache()
result = process(data)
temp = result.blocks

# GOOD
agent_cache = fetch_cache()
processed_blocks = process(agent_cache)
layer_blocks = processed_blocks.blocks
```

#### 5. Magic Numbers
```python
# BAD
if layers > 8:
    global_layers = layers - 8

# GOOD
GEMMA3_GLOBAL_LAYERS = 8  # First 8 layers use global attention
if layers > GEMMA3_GLOBAL_LAYERS:
    global_layers = layers - GEMMA3_GLOBAL_LAYERS
```

#### 6. Placeholder Code
```python
# BAD
def process(self) -> None:
    pass  # TODO: implement

# GOOD - Either implement or explicitly raise
def process(self) -> None:
    raise NotImplementedError("Requires CachePersistencePort adapter")
```

#### 7. Defensive Programming Gone Wrong
```python
# BAD: Unnecessary checks
if hasattr(block, 'layer_data'):  # KVBlock ALWAYS has layer_data
    block.layer_data = None

# GOOD: Trust the type system
block.layer_data = None  # Type hint guarantees attribute exists
```

#### 8. Silent Exception Swallowing
```python
# BAD
try:
    save_cache()
except Exception:
    pass  # Hope it works next time

# GOOD
try:
    save_cache()
except OSError as e:
    logger.error(f"Cache save failed: {e}")
    raise CachePersistenceError(f"Disk write failed: {e}") from e
```

#### 9. Runtime Imports in Functions
```python
# BAD: Import overhead on every call
def process(self):
    import numpy as np  # Imported every call!
    return np.array(self.data)

# GOOD: Module-level imports
import numpy as np

def process(self):
    return np.array(self.data)
```

#### 10. Test Code in Production
```python
# BAD: Test detection in production code
if tensor.__class__.__name__ == 'FakeTensor':
    return self._handle_test_mode()

# GOOD: Use dependency injection
def __init__(self, extractor: CacheExtractorPort):
    self._extractor = extractor  # Inject fake in tests
```

### YAGNI: You Aren't Gonna Need It

**Rule of Three**: Don't create abstraction until you have 3+ real use cases.

```python
# BAD: Strategy pattern with 2 implementations
class DetectionStrategy(Protocol): ...
class Gemma3Strategy: ...
class UniformStrategy: ...

# GOOD: Simple if/else until you need more
def detect_layers(model) -> list[str]:
    if is_gemma3(model):
        return ["global"] * 8 + ["sliding"] * (n - 8)
    return ["full_attention"] * n
```

---

## Automated Enforcement

### Required Pre-commit Hooks

All PRs must pass these checks:

```yaml
# .pre-commit-config.yaml (excerpt)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        args: [--strict]

  - repo: local
    hooks:
      - id: sloppylint
        name: AI slop detector
        entry: sloppylint
        args: [--ci, --severity, high, src/semantic/]

      - id: vulture
        name: Dead code check
        entry: vulture
        args: [src/semantic/, --min-confidence, "80"]

      - id: radon-cc
        name: Complexity check
        entry: radon
        args: [cc, --min, C, src/semantic/]
```

### CI Quality Gates

| Check | Command | Threshold |
|-------|---------|-----------|
| Lint | `ruff check .` | 0 errors |
| Types | `mypy --strict src/` | 0 errors |
| Security | `bandit -r src/` | 0 high/critical |
| Complexity | `radon cc --min C` | All ≤ C grade |
| Coverage | `pytest --cov --cov-fail-under=80` | ≥80% |

---

## Code Review Checklist

Before approving any PR, verify:

### Architecture
- [ ] No MLX/safetensors/numpy imports in domain/ or application/
- [ ] All infrastructure code in adapters/
- [ ] Dependencies injected via constructor

### AI Slop
- [ ] No numbered comments (# 1., # 2.)
- [ ] No excessive docstrings on trivial methods
- [ ] No sprint/ticket references in code
- [ ] No generic variable names (data, result, temp)
- [ ] No magic numbers without constants
- [ ] No bare `except:` or `except Exception:`

### Quality
- [ ] No functions over 50 lines
- [ ] No methods over 100 lines
- [ ] Cyclomatic complexity ≤ 10
- [ ] All imports at module level

### Testing
- [ ] Uses dependency injection (not internal mocking)
- [ ] Meaningful assertions (not just "assert True")
- [ ] Tests behavior, not implementation

---

## File Organization

### Correct Structure (Sprint 4+)

```
src/semantic/
├── domain/           # Pure Python, NO external deps
│   ├── entities.py   # KVBlock, AgentBlocks
│   ├── value_objects.py  # ModelCacheSpec (data only, no from_model!)
│   ├── services.py   # BlockPool
│   └── errors.py     # Domain exceptions
├── ports/            # Protocol definitions
│   ├── inbound.py    # InferencePort, GenerationEnginePort
│   └── outbound.py   # ModelBackendPort, CachePersistencePort
├── application/      # Orchestration (uses ports only)
│   ├── batch_engine.py
│   └── agent_cache_store.py
├── adapters/         # Infrastructure implementations
│   ├── inbound/      # API adapters (FastAPI)
│   ├── outbound/     # MLX, safetensors adapters
│   └── config/       # Pydantic settings
└── entrypoints/      # CLI, server startup
```

### Dead Code (DO NOT USE)

These files are deprecated and will be removed:

```
src/*.py              # OLD POC - use src/semantic/ instead
tests/test_*.py       # OLD POC tests - use tests/unit/ and tests/integration/
```

---

## Related Documentation

- **Code Quality Patterns**: `/plans/code_quality_patterns.md` - Full red flag catalog
- **Remediation Plan**: `/plans/remediation_plan.md` - Specific fixes for existing issues
- **Production Plan**: `/plans/production_plan.md` - Sprint roadmap
- **Backend Plan**: `/plans/backend_plan.md` - Architecture details

---

# Claude API Integration Guide

This section details the Claude API usage and best practices.

## Models Used

### Claude Sonnet 4.5 (Primary)

**Model ID**: `claude-sonnet-4-5-20250929`

**Use Cases**:
- Dataset generation (conflict examples)
- High-quality text generation
- Complex reasoning tasks
- Primary model for production use

**Specifications**:
- Context window: 200K tokens
- Output: Up to 8K tokens
- Temperature: 0.8 (for dataset diversity)
- Cost: Moderate (see pricing)

**Example Usage**:
```python
from anthropic import Anthropic

client = Anthropic(api_key=config.claude_api_key)

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2000,
    temperature=0.8,
    messages=[{
        "role": "user",
        "content": "Generate a conflict example..."
    }]
)

content = response.content[0].text
```

### Claude Haiku 4.5 (Fast)

**Model ID**: `claude-haiku-4-5-20251001`

**Use Cases**:
- Quick validation tasks
- API connectivity testing
- Simple text processing
- Cost-sensitive operations

**Specifications**:
- Context window: 200K tokens
- Output: Up to 8K tokens
- Speed: Fastest in Claude family
- Cost: Most economical

**Example Usage**:
```python
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{
        "role": "user",
        "content": "Validate this example..."
    }]
)
```

## Configuration

### API Key Setup

1. Obtain API key from [Anthropic Console](https://console.anthropic.com/)
2. Add to `env.json`:

```json
{
  "claude_api_key": "sk-ant-api03-..."
}
```

3. Load via config manager:

```python
from src.config import get_config

config = get_config()
api_key = config.claude_api_key
```

### Configuration Manager

The `src/config.py` module provides centralized configuration:

```python
class Config:
    """Configuration manager for RDIC project"""

    @property
    def claude_api_key(self) -> str:
        """Get Claude/Anthropic API key"""
        # Validates and returns API key

    def get(self, key: str, default=None) -> Any:
        """Get any configuration value"""
```

**Features**:
- Automatic config file discovery
- Validation of required keys
- Singleton pattern for global access
- Clear error messages

## Dataset Generation

### Generation Parameters

```python
# In src/dataset_generator.py

def generate_example(
    conflict_type: str,
    conflict_subtype: str,
    domain: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> Dict[str, Any]:
    """
    Generate a single conflict example.

    Uses Claude Sonnet 4.5 with:
    - Temperature: 0.8 (for diversity)
    - Max tokens: 2000 (sufficient for multi-turn examples)
    - Detailed prompting for structured output
    """
```

### Prompt Engineering

The dataset generator uses structured prompts with:

1. **Clear task definition**
2. **Conflict type specification**
3. **Domain context**
4. **Example structure**
5. **Quality requirements**
6. **JSON formatting instructions**

Example prompt structure:
```
Generate a realistic multi-turn conversation that demonstrates an instruction conflict.

**Conflict Type:** {type}
**Specific Conflict:** {subtype}
**Domain:** {domain}

Create a JSON object with the following structure:
{schema}

**Requirements:**
1. The conflict must be GENUINE - truly incompatible
2. The conflict should be REALISTIC - real-world scenario
3. The final query should make the conflict UNAVOIDABLE
4. Ground truth clusters should represent incompatible semantic spaces

Return ONLY the JSON object, no additional text.
```

### JSON Parsing

The generator handles Claude's output robustly:

```python
# Extract JSON from response
content = response.content[0].text

# Handle markdown code blocks
if "```json" in content:
    content = content.split("```json")[1].split("```")[0].strip()
elif "```" in content:
    content = content.split("```")[1].split("```")[0].strip()

example = json.loads(content)
```

## Best Practices

### 1. Temperature Settings

- **Dataset generation (diversity)**: 0.8
- **Validation tasks**: 0.0 - 0.3
- **Creative tasks**: 0.7 - 1.0
- **Structured output**: 0.3 - 0.5

### 2. Token Limits

- **Short responses**: 50-100 tokens
- **Examples**: 1000-2000 tokens
- **Long documents**: 4000-8000 tokens
- **Max safety margin**: Leave 10% buffer

### 3. Error Handling

```python
try:
    response = client.messages.create(...)
except anthropic.APIError as e:
    print(f"API error: {e}")
    # Handle retry logic
except anthropic.APIConnectionError as e:
    print(f"Connection error: {e}")
    # Check network
except json.JSONDecodeError as e:
    print(f"JSON parsing failed: {e}")
    # Log raw output for debugging
```

### 4. Rate Limiting

- Implement exponential backoff
- Batch requests when possible
- Monitor API usage
- Use Haiku for high-volume tasks

### 5. Cost Optimization

- Use Haiku for simple tasks
- Use Sonnet for complex reasoning
- Cache common prompts
- Batch similar requests
- Set appropriate max_tokens

## API Testing

### Verification Script

Test all models with `tests/test_apis.py`:

```bash
python -m tests.test_apis
```

Expected output:
```
✓ Claude Haiku 4.5 API works!
✓ Claude Sonnet 4.5 API works!
✓ DeepSeek R1 API works!

🎉 All Day 1 requirements met!
```

### Individual Model Tests

```python
from src.utils import APIClients

clients = APIClients()

# Test Haiku
response = clients.call_claude(
    "Hello",
    model="claude-haiku-4-5-20251001",
    max_tokens=50
)

# Test Sonnet
response = clients.call_claude(
    "Generate a complex example...",
    model="claude-sonnet-4-5-20250929",
    max_tokens=2000
)
```

## Performance Metrics

### Generation Speed (Day 2 Results)

- **Average time per example**: ~7-8 seconds
- **30 examples total time**: ~4 minutes
- **Success rate**: 100% (30/30)
- **JSON parsing success**: 100%

### Quality Metrics

- **Genuine conflicts**: 100% (exceeds 70% target)
- **Realistic scenarios**: 100%
- **Unavoidable conflicts**: 100%
- **Clear ground truth**: 100%

## Model Selection Guide

Choose the right model for your task:

| Task | Recommended Model | Reasoning |
|------|------------------|-----------|
| Dataset generation | Sonnet 4.5 | High quality, structured output |
| Validation | Haiku 4.5 | Fast, cost-effective |
| Complex reasoning | Sonnet 4.5 | Superior analytical capabilities |
| Bulk operations | Haiku 4.5 | Speed and cost efficiency |
| Final production | Sonnet 4.5 | Best quality for end users |

## Sandbox Environment

### Network Connectivity Resolution

When running in Claude Code CLI sandbox environment, external API calls are blocked by default.

**Symptom**: Connection errors when calling Claude API, even with valid credentials:
```
anthropic.APIConnectionError: Connection error
```

**Root Cause**: Sandbox security restrictions block external network access.

**Solution**: Use `dangerouslyDisableSandbox: true` parameter in Bash commands:

```python
# When calling API from scripts via Bash tool
Bash(
    command="python generate_batch_002.py",
    description="Generate dataset batch 2",
    dangerouslyDisableSandbox=true  # Required for API calls
)
```

**When to Use**:
- ✅ Calling external APIs (Claude, DeepSeek, etc.)
- ✅ Dataset generation scripts
- ✅ Any script that makes network requests
- ❌ Local file operations (use normal sandbox mode)
- ❌ Git operations (use normal sandbox mode)

**Security Note**: Only use sandbox bypass when necessary for external API calls. All other operations should use default sandbox mode.

### Alternative: Claude CLI Integration

For Claude API calls, consider using Claude Code's built-in Task tool with appropriate model:

```python
# Instead of calling API directly, use Task tool
Task(
    subagent_type="general-purpose",
    prompt="Generate dataset examples...",
    model="haiku",  # or "sonnet" for complex tasks
    description="Generate examples via CLI"
)
```

**Benefits**:
- No sandbox bypass required
- Automatic credential management
- Better context preservation
- Built-in error handling

## Troubleshooting

### Common Issues

**Issue**: `Connection error` in sandbox
- **Solution**: Use `dangerouslyDisableSandbox: true` for API calls (see Sandbox Environment section)
- **Alternative**: Use Task tool with Claude CLI integration

**Issue**: `Connection error` outside sandbox
- **Solution**: Check network, verify API key, check Anthropic status

**Issue**: `Model not found error`
- **Solution**: Verify model ID matches exactly (case-sensitive)

**Issue**: `JSON parsing failed`
- **Solution**: Check response format, review prompt instructions

**Issue**: `Rate limit exceeded`
- **Solution**: Implement backoff, reduce request frequency

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now API calls will show detailed information
```

## Migration Notes

### From Older Claude Models

If migrating from previous Claude versions:

- ✅ Replace `claude-3-5-sonnet-20241022` → `claude-sonnet-4-5-20250929`
- ✅ Replace `claude-3-haiku-20240307` → `claude-haiku-4-5-20251001`
- ✅ Update all references in code and configuration
- ✅ Test thoroughly with new models

### Backwards Compatibility

The current implementation uses:
- **Anthropic SDK**: `anthropic>=0.25.0`
- **Python**: 3.8+
- **API Version**: Latest (2023-06-01)

## Security

### API Key Protection

- ✅ Store keys in `env.json` (gitignored)
- ✅ Never commit keys to version control
- ✅ Use environment variables in production
- ✅ Rotate keys periodically
- ✅ Monitor API usage for anomalies

### Safe Practices

```python
# ✅ GOOD: Load from config
from src.config import get_config
api_key = get_config().claude_api_key

# ❌ BAD: Hardcode in code
api_key = "sk-ant-api03-..."  # Never do this!
```

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Claude Model Specifications](https://docs.anthropic.com/claude/docs/models-overview)
- [Python SDK Reference](https://github.com/anthropics/anthropic-sdk-python)
- [API Console](https://console.anthropic.com/)
- [Pricing Information](https://www.anthropic.com/pricing)

## Support

For Claude API issues:
- Check [Anthropic Status](https://status.anthropic.com/)
- Review [API Documentation](https://docs.anthropic.com/)
- Contact Anthropic support

For project-specific issues:
- Check project GitHub issues
- Review daily status reports
- Test with `tests/test_apis.py`

---

## Quick Reference: What Changed in Sprint 4

1. **Old POC code removed**: `src/*.py` files deleted (use `src/semantic/` only)
2. **Architecture enforced**: No MLX in domain/application layers
3. **AI slop prevention**: New pre-commit hooks for quality
4. **Complexity limits**: Max 50 lines/function, 10 cyclomatic complexity

---

**Last Updated**: 2026-01-25 | **Claude SDK Version**: 0.25.0+
**Code Quality Version**: 1.0.0 (Sprint 4 standards)
