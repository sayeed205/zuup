# Development Guide

This document provides information for developers working on Zuup.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd zuup
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

4. **Development tools are ready**:
   ```bash
   uv run ruff check src/
   uv run mypy src/
   ```

## Development Workflow

### Code Quality

- **Type Checking**: `uv run mypy src/`
- **Linting**: `uv run ruff check src/`
- **Formatting**: `uv run ruff format src/`
- **Type checking**: `uv run mypy src/`

### Testing

- **Unit Tests**: `uv run pytest tests/unit/`
- **Integration Tests**: `uv run pytest tests/integration/`
- **All Tests**: `uv run pytest`
- **Coverage**: `uv run pytest --cov`

### Running the Application

- **Development Mode**: `uv run zuup start --gui`
- **Server Only**: `uv run zuup start --server-only`

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Core**: Application logic and task management
- **Engines**: Protocol-specific download implementations
- **Storage**: Data persistence and caching
- **API**: REST API server
- **GUI**: PySide6 user interface
- **CLI**: Command-line interface

## Contributing

1. Create a feature branch
2. Make changes with proper type annotations
3. Add tests for new functionality
4. Ensure all checks pass
5. Submit a pull request

## Type Safety

This project uses strict type checking with mypy. All code must:

- Have explicit type annotations
- Pass mypy strict mode checks
- Use Pydantic models for data validation
- Follow typing best practices
