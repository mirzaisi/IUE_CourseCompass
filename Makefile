# ═══════════════════════════════════════════════════════════════════════════════
# IUE CourseCompass - Makefile
# ═══════════════════════════════════════════════════════════════════════════════
# Common commands for development, testing, and running the application.
# Usage: make <target>
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help install install-dev lint format typecheck test test-cov \
        scrape index app evaluate clean clean-cache clean-index \
        setup-dirs check all

# Default target
.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "IUE CourseCompass - Available Commands"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make setup-dirs    Create required data directories"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with black and ruff"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make check         Run all quality checks (lint + typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run tests with pytest"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make scrape        Run ingestion pipeline (scrape + parse + clean)"
	@echo "  make index         Build vector index from processed data"
	@echo ""
	@echo "Application:"
	@echo "  make app           Run Streamlit GUI"
	@echo "  make evaluate      Run evaluation harness"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make clean-cache   Remove scraped HTML cache"
	@echo "  make clean-index   Remove vector index (requires rebuild)"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Setup & Installation
# ─────────────────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

setup-dirs:
	@echo "Creating data directories..."
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p data/index
	mkdir -p data/manifests
	mkdir -p evaluation/question_sets
	mkdir -p evaluation/results
	@echo "Done."

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

check: lint typecheck
	@echo "All quality checks passed!"

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/iue_coursecompass --cov-report=html --cov-report=term-missing

# ─────────────────────────────────────────────────────────────────────────────
# Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────
scrape:
	python -m iue_coursecompass.cli.main ingest --all

scrape-dept:
	@echo "Usage: make scrape-dept DEPT=se"
	@echo "Available departments: se, ce, eee, ie"
	python -m iue_coursecompass.cli.main ingest --department $(DEPT)

index:
	python -m iue_coursecompass.cli.main index --rebuild

index-update:
	python -m iue_coursecompass.cli.main index

# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────
app:
	streamlit run src/iue_coursecompass/app/streamlit_app.py

# CLI query example
query:
	@echo "Usage: make query Q='your question here'"
	python -m iue_coursecompass.cli.main query "$(Q)"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
evaluate:
	python -m iue_coursecompass.cli.main evaluate --all

evaluate-retrieval:
	python -m iue_coursecompass.cli.main evaluate --retrieval-only

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning build artifacts and caches..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."

clean-cache:
	@echo "Removing scraped HTML cache..."
	rm -rf data/raw/*
	@echo "Done."

clean-index:
	@echo "Removing vector index (will require rebuild)..."
	rm -rf data/index/*
	@echo "Done."

clean-all: clean clean-cache clean-index
	@echo "All data and caches removed."

# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
all: setup-dirs scrape index
	@echo "Full pipeline complete!"

# ─────────────────────────────────────────────────────────────────────────────
# Development Shortcuts
# ─────────────────────────────────────────────────────────────────────────────
dev: install-dev setup-dirs
	@echo "Development environment ready!"

# Show current configuration
config:
	python -m iue_coursecompass.cli.main config --show

# Show index status
status:
	python -m iue_coursecompass.cli.main status
