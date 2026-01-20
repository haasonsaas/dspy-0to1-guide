.PHONY: help install setup clean test lint format type-check run-examples evaluate docker-build docker-run

# Default target
help: ## Show this help message
	@echo "DSPy 0-to-1 Guide - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install all dependencies
	pip install -e ".[dev,ollama,docker]"

setup: ## Initial setup - create virtual environment and install dependencies
	python -m venv .venv
	@echo "Activate virtual environment with: source .venv/bin/activate"
	@echo "Then run: make install"

clean: ## Clean up cache and temporary files
	rm -rf .pytest_cache/
	rm -rf .dspy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Code quality
lint: ## Run linting checks
	flake8 src/ examples/ tests/
	black --check src/ examples/ tests/
	isort --check-only src/ examples/ tests/

format: ## Format code with black and isort
	black src/ examples/ tests/
	isort src/ examples/ tests/

type-check: ## Run type checking with mypy
	mypy src/

# Testing
test: ## Run all tests
	pytest tests/ -v

test-fast: ## Run tests without slow integration tests
	pytest tests/ -v -m "not slow"

test-integration: ## Run only integration tests
	pytest tests/ -v -m "integration"

# Examples
run-basic: ## Run basic DSPy examples
	python examples/basic/hello_world.py
	python examples/basic/math_qa.py
	python examples/basic/summarizer.py

run-personas: ## Run persona-driven examples
	python examples/personas/support_sam.py
	python examples/personas/legal_lucy.py
	python examples/personas/data_dana.py

run-advanced: ## Run advanced examples
	python examples/advanced/gepa_optimization.py
	python examples/advanced/pydantic_validation.py
	python examples/advanced/parallel_execution.py

run-all-examples: run-basic run-personas run-advanced ## Run all examples

# Evaluation
evaluate: ## Run evaluation pipeline
	python scripts/evaluate_all.py

evaluate-personas: ## Evaluate persona examples
	python scripts/evaluate_personas.py

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

# Infrastructure
start-monitoring: ## Start Prometheus monitoring
	python examples/infrastructure/prometheus_metrics.py &
	@echo "Prometheus metrics server started on port 8000"

stop-monitoring: ## Stop monitoring processes
	pkill -f prometheus_metrics.py || true

# Docker
docker-build: ## Build Docker image
	docker build -t dspy-0to1-guide .

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 dspy-0to1-guide

docker-dev: ## Run Docker container in development mode
	docker run -it --rm -v $(PWD):/app -p 8000:8000 dspy-0to1-guide bash

# Data preparation
prepare-datasets: ## Prepare sample datasets
	python scripts/prepare_datasets.py

download-models: ## Download required models for local execution
	ollama pull llama3
	ollama pull codellama
	ollama pull mistral

# Documentation
docs: ## Generate documentation
	@echo "Documentation is available in the docs/ directory"
	@echo "Main guide: docs/README.md"

# Development workflow
dev-setup: setup install prepare-datasets ## Complete development setup
	@echo "Development environment ready!"
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"
	@echo "Then try 'make run-basic' to test the installation"

ci: lint type-check test ## Run CI pipeline (lint, type-check, test)

# Quick start
quick-start: ## Quick start - run hello world example
	@echo "Running DSPy Hello World example..."
	python examples/basic/hello_world.py
