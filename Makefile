# Makefile for Zuup development

.PHONY: help setup clean check format lint type-check build test-dev test-type test-tooling install dev

# Default target
help:
	@echo "Zuup Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          Set up development environment"
	@echo "  install        Install package in development mode"
	@echo "  clean          Clean build artifacts and caches"
	@echo ""
	@echo "Code Quality:"
	@echo "  format         Format code with ruff"
	@echo "  lint           Run linting checks"
	@echo "  type-check     Run type checking with mypy"
	@echo "  check          Run all code quality checks"
	@echo ""
	@echo "Build and Package:"
	@echo "  build          Build package"
	@echo "  build-all      Clean, check, and build"
	@echo ""
	@echo "Testing:"
	@echo "  test-dev       Test development environment"
	@echo "  test-type      Test type safety configuration"
	@echo "  test-tooling   Test tooling integration"
	@echo "  test-all       Run all manual tests"
	@echo ""
	@echo "Development:"
	@echo "  dev            Start development server"
	@echo "  dev-gui        Start GUI application"

# Setup and Installation
setup:
	@echo "🚀 Setting up development environment..."
	python scripts/setup_dev.py

install:
	@echo "📦 Installing in development mode..."
	uv pip install -e .

clean:
	@echo "🧹 Cleaning build artifacts..."
	python scripts/build.py --clean

# Code Quality
format:
	@echo "🎨 Formatting code..."
	uv run ruff format src/ examples/ scripts/

lint:
	@echo "🔍 Running linting..."
	uv run ruff check src/ examples/ scripts/

type-check:
	@echo "🔍 Running type checking..."
	uv run mypy src/zuup/

check: format lint type-check
	@echo "✅ All code quality checks completed"

# Build and Package
build:
	@echo "📦 Building package..."
	python scripts/build.py --build

build-all:
	@echo "🏗️  Running full build process..."
	python scripts/build.py --all

# Testing
test-dev:
	@echo "🧪 Testing development environment..."
	python examples/manual_tests/test_development_environment.py

test-type:
	@echo "🧪 Testing type safety..."
	python examples/manual_tests/test_type_safety.py

test-tooling:
	@echo "🧪 Testing tooling integration..."
	python examples/manual_tests/test_tooling_integration.py

test-all: test-dev test-type test-tooling
	@echo "✅ All manual tests completed"

# Development
dev:
	@echo "🚀 Starting development server..."
	uv run python -m zuup.main --server

dev-gui:
	@echo "🖥️  Starting GUI application..."
	uv run python -m zuup.main --gui

# Advanced development tools
dev-tools:
	@echo "🛠️  Running development tools..."
	python scripts/dev_tools.py --all

fix:
	@echo "🔧 Fixing code issues automatically..."
	python scripts/dev_tools.py --fix

validate:
	@echo "✅ Validating project structure..."
	python scripts/dev_tools.py --validate

# Quick development workflow
quick-check: format lint
	@echo "⚡ Quick code quality check completed"

# Full development workflow
full-check: clean check build test-all
	@echo "🎉 Full development workflow completed"