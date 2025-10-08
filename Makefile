.PHONY: test test-verbose test-coverage test-fast test-integration test-unit clean help format lint type-check quality-check

# Default test command
test:
	rm -rf cache/api/*
	pytest tests/ -v

# Code quality commands
format:
	black src/ config/ integrations/ scripts/ tests/

lint:
	flake8 src/ config/ integrations/ scripts/ tests/

type-check:
	mypy src/ config/ integrations/ scripts/

quality-check: format lint type-check

# Verbose testing
test-verbose:
	pytest -v

# Test with coverage
test-coverage:
	pytest --cov=src --cov=config --cov=integrations --cov-report=html --cov-report=term-missing

# Fast tests only (skip slow tests)
test-fast:
	pytest -m "not slow"

# Run integration tests only
test-integration:
	pytest -m integration

# Run unit tests only
test-unit:
	pytest -m unit

# Run edge case tests
test-edge-cases:
	pytest tests/test_edge_cases.py

# Run performance tests
test-performance:
	pytest tests/test_performance.py

# Clean test artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Show help
help:
	@echo "Available commands:"
	@echo "  test              - Run all tests"
	@echo "  test-verbose      - Run all tests with verbose output"
	@echo "  test-coverage     - Run tests with coverage report"
	@echo "  test-fast         - Run fast tests only (skip slow tests)"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-edge-cases   - Run edge case tests"
	@echo "  test-performance  - Run performance tests"
	@echo "  format            - Format code with black"
	@echo "  lint              - Lint code with flake8"
	@echo "  type-check        - Type check with mypy"
	@echo "  quality-check     - Run all code quality checks (format, lint, type-check)"
	@echo "  clean             - Clean test artifacts"
	@echo "  help              - Show this help"
	@echo ""
	@echo "Direct pytest usage:"
	@echo "  pytest                           - Run all tests"
	@echo "  pytest -v                        - Verbose output"
	@echo "  pytest -k 'test_name'            - Run specific test"
	@echo "  pytest tests/test_file.py        - Run specific file"
	@echo "  pytest --cov=src                 - With coverage"