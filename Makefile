.PHONY: help install dev-install lint format typecheck security complexity licenses test test-unit test-integration test-smoke test-e2e bench docs clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies only
	pip install -e .

dev-install:  ## Install all dependencies including dev and docs
	pip install -e ".[dev,docs]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check src tests

format:  ## Format code with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck:  ## Run mypy type checker
	mypy src

security:  ## Run security scans (ruff S rules + semgrep)
	ruff check --select S src
	@echo "Note: semgrep requires separate installation and configuration"
	@echo "Run: semgrep --config=auto src/"

complexity:  ## Check cyclomatic complexity
	ruff check --select C90 src
	@echo "Target: CC < 10 for new code, CC < 7 for domain logic"

licenses:  ## Check OSS license compliance
	liccheck --sfile pyproject.toml

test:  ## Run all tests (unit + integration + smoke)
	pytest -v -m "not e2e"

test-unit:  ## Run unit tests only (no MLX dependencies)
	pytest -v -m unit

test-integration:  ## Run integration tests (requires Apple Silicon + MLX)
	pytest -v -m integration

test-smoke:  ## Run smoke tests (server lifecycle)
	pytest -v -m smoke

test-e2e:  ## Run end-to-end tests (slow, requires Apple Silicon)
	pytest -v -m e2e

coverage:  ## Generate coverage report
	pytest --cov --cov-report=term-missing --cov-report=html

bench:  ## Run benchmarks
	python benchmarks/bench_block_pool.py
	python benchmarks/bench_batched_decode.py
	python benchmarks/bench_cache_load.py

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
	python -c "import toml; toml.load('pyproject.toml'); print('✓ pyproject.toml valid')"
	python -c "import toml; toml.load('config/default.toml'); print('✓ config/default.toml valid')" || echo "⚠ config/default.toml not yet created"
	@echo "✓ Configuration validation complete"
