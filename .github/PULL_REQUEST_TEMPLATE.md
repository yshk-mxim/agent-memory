## Summary

Brief description of the changes in this PR.

## Type

- [ ] Feature (new functionality)
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] Refactor (no functional changes)
- [ ] Documentation
- [ ] Tests
- [ ] CI/Build

## Changes

- Change 1
- Change 2

## Testing

Describe the tests you ran to verify your changes:

- [ ] Unit tests pass (`pytest tests/unit/ -x -q`)
- [ ] Integration tests pass (`pytest tests/integration/ -x -q`)
- [ ] New tests added for new functionality
- [ ] Manual testing performed (describe below)

## Architecture Compliance

- [ ] No MLX/safetensors/numpy imports in `domain/` or `application/`
- [ ] All infrastructure code in `adapters/`
- [ ] Dependencies injected via constructor
- [ ] No functions over 50 lines
- [ ] No cyclomatic complexity over 10

## Checklist

- [ ] Code follows the [Code Quality Guide](docs/code-quality-guide.md)
- [ ] Self-review of my own code completed
- [ ] Comments added only where behavior is non-obvious
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes to public API (or documented in changelog)
