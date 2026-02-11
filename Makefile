.PHONY: help install dev-install lint format typecheck security complexity licenses \
        test test-unit test-integration test-smoke test-e2e coverage bench docs docs-build \
        clean validate quality-full ci all

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies only
	pip install --no-deps mlx-lm==0.30.4
	pip install -e .

dev-install:  ## Install all dependencies including dev and docs
	pip install --no-deps mlx-lm==0.30.4
	pip install -e ".[dev,docs]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check src tests

format:  ## Format code with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck:  ## Run mypy type checker
	mypy --explicit-package-bases src/agent_memory tests/unit tests/conftest.py

security:  ## Run security scans (ruff S rules)
	@echo "==> Running ruff security rules (bandit equivalent)..."
	ruff check --select S src

complexity:  ## Check cyclomatic complexity (ruff C90 rules)
	@echo "==> Ruff complexity check (CC < 15 enforced via pylint rules)..."
	ruff check --select C90,PLR src

licenses:  ## Check OSS license compliance
	liccheck --sfile pyproject.toml || echo "⚠️  License check found unknown packages (non-blocking)"

quality-full:  ## Run full quality pipeline
	@echo "=========================================="
	@echo "COMPREHENSIVE QUALITY VALIDATION"
	@echo "=========================================="
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) security
	$(MAKE) complexity
	$(MAKE) licenses
	$(MAKE) test-unit
	@echo "\n=========================================="
	@echo "QUALITY VALIDATION COMPLETE"
	@echo "=========================================="

test:  ## Run all tests (unit + integration + smoke)
	pytest -v -m "not e2e"

test-unit:  ## Run unit tests only (no MLX dependencies)
	pytest -v -m unit tests/unit tests/conftest.py

test-integration:  ## Run integration tests (requires Apple Silicon + MLX)
	pytest -v -m integration

test-smoke:  ## Run smoke tests (server lifecycle)
	pytest -v -m smoke

test-e2e:  ## Run end-to-end tests (slow, requires Apple Silicon)
	pytest -v -m e2e

coverage:  ## Generate coverage report
	pytest --cov --cov-report=term-missing --cov-report=html

bench:  ## Run benchmarks (see benchmarks/ for full suite)
	python benchmarks/full_benchmark.py

docs:  ## Build and serve documentation locally
	mkdocs serve

docs-build:  ## Build documentation (static HTML)
	mkdocs build --strict

clean:  ## Clean build artifacts and caches
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all:  ## Run full quality pipeline (CI equivalent)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) security
	$(MAKE) complexity
	$(MAKE) licenses
	$(MAKE) test

ci:  ## CI pipeline (lint + typecheck + unit tests)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test-unit

validate:  ## Validate configuration files
	python -c "import tomllib; f = open('pyproject.toml', 'rb'); tomllib.load(f); f.close(); print('✓ pyproject.toml valid')"
	python -c "import tomllib; f = open('config/default.toml', 'rb'); tomllib.load(f); f.close(); print('✓ config/default.toml valid')"
	@echo "✓ Configuration validation complete"
