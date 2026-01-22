# Contributing to RDIC

Thank you for your interest in contributing to the RDIC (Reflective Debate with Instruction Conflicts) project!

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Claude API key from [Anthropic](https://console.anthropic.com/)
- (Optional) DeepSeek API key

### Initial Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yshk-mxim/rdic.git
   cd rdic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp env.json.example env.json
   # Edit env.json with your API keys
   ```

4. **Verify setup**
   ```bash
   python -m tests.test_apis
   ```

## Project Structure

```
rdic/
├── src/               # Source code
├── data/              # Datasets and schemas
├── tests/             # Test files
├── plans/             # Implementation plans
└── docs/              # Documentation (if added)
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow PEP 8 style guide
- Add docstrings to functions
- Keep functions focused and small
- Write clear commit messages

### 3. Test Your Changes

```bash
# Run API tests
python -m tests.test_apis

# Test dataset generation
python -m src.dataset_generator --num 5 --validate
```

### 4. Commit

```bash
git add .
git commit -m "Add: brief description of changes"
```

**Commit message format**:
- `Add: ...` for new features
- `Fix: ...` for bug fixes
- `Update: ...` for modifications
- `Refactor: ...` for code improvements
- `Docs: ...` for documentation

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable names

**Example**:
```python
def generate_example(
    conflict_type: str,
    domain: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> Dict[str, Any]:
    """
    Generate a single conflict example.

    Args:
        conflict_type: Type of conflict to generate
        domain: Domain for the example
        model: Claude model to use

    Returns:
        Generated example as dictionary
    """
    # Implementation
    pass
```

### Documentation

- Add docstrings to all public functions
- Update README.md for major changes
- Include examples in documentation
- Keep CLAUDE.md updated for API changes

## Testing

### Running Tests

```bash
# All tests
python -m pytest

# Specific test
python -m tests.test_apis

# With coverage
pytest --cov=src tests/
```

### Writing Tests

Create test files in `tests/` directory:

```python
def test_example_generation():
    """Test that example generation works"""
    generator = ConflictDatasetGenerator()
    example = generator.generate_example("tone", "formal_vs_casual", "business")
    assert example is not None
    assert "id" in example
    assert "conflict_type" in example
```

## API Usage

### Model Selection

Always use the correct model IDs:

- **Claude Sonnet 4.5**: `claude-sonnet-4-5-20250929`
- **Claude Haiku 4.5**: `claude-haiku-4-5-20251001`
- **DeepSeek R1**: `deepseek-reasoner`

See [CLAUDE.md](CLAUDE.md) for detailed API documentation.

### Error Handling

Always handle API errors gracefully:

```python
try:
    response = client.messages.create(...)
except anthropic.APIError as e:
    logger.error(f"API error: {e}")
    return None
```

## Dataset Guidelines

### Quality Standards

All generated examples must meet:

1. **Genuine Conflict**: Instructions are truly incompatible
2. **Realistic Scenario**: Plausible real-world context
3. **Unavoidable Conflict**: Final query requires both constraints
4. **Clear Ground Truth**: Well-defined semantic clusters

### Schema Compliance

Follow the schema in `data/conflict_schema.json`:

```json
{
  "id": "conflict_XXX",
  "conflict_type": "type_subtype",
  "domain": "domain_name",
  "turns": [...],
  "ground_truth_clusters": [...],
  "metadata": {...}
}
```

## Documentation

### What to Document

- New features
- API changes
- Breaking changes
- Configuration options
- Examples and usage

### Where to Document

- **README.md**: Overview, quick start
- **CLAUDE.md**: Claude API specifics
- **Code comments**: Complex logic
- **Docstrings**: All functions
- **Status reports**: Daily progress

## Release Process

1. Update version in README.md
2. Update CHANGELOG.md (if exists)
3. Tag the release: `git tag v0.3.0`
4. Push tags: `git push --tags`

## Questions?

- Review existing code and documentation
- Check [CLAUDE.md](CLAUDE.md) for API questions
- Open an issue for bugs or feature requests
- Review daily status reports for context

## Code of Conduct

- Be respectful and professional
- Focus on the issue, not the person
- Provide constructive feedback
- Help others learn and grow

---

Thank you for contributing to RDIC!
