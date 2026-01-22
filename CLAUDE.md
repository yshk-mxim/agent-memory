# Claude API Integration Guide

This document details the Claude API usage, model specifications, and best practices for the RDIC project.

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

## Troubleshooting

### Common Issues

**Issue**: `Connection error`
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

**Last Updated**: 2026-01-22 | **Claude SDK Version**: 0.25.0+
