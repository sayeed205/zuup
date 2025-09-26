# Development Guide

This guide covers the development environment setup, tooling, and workflows for the Zuup download manager project.

## Quick Start

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up development environment**:
   ```bash
   python scripts/setup_dev.py
   # or
   make setup
   ```

3. **Run development checks**:
   ```bash
   make check
   ```

## Development Environment

### Prerequisites

- Python 3.10 or higher
- uv package manager
- Git

### Project Structure

```
zuup/
├── src/zuup/              # Main source code
├── examples/              # Example scripts and configurations
├── scripts/               # Development scripts
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
├── mypy.ini              # Type checking configuration
├── .editorconfig         # Editor configuration
└── Makefile              # Development commands
```

### Development Tools

The project uses several tools for code quality and development:

- **uv**: Package manager and virtual environment
- **mypy**: Static type checking
- **ruff**: Linting and code formatting
- **Pydantic**: Runtime type validation

## Development Workflow

### 1. Code Quality Checks

Run these commands before committing code:

```bash
# Format code
make format
# or
uv run ruff format src/

# Check linting
make lint
# or
uv run ruff check src/

# Check types
make type-check
# or
uv run mypy src/zuup/

# Run all checks
make check
```

### 2. Manual Testing

The project includes comprehensive manual testing scripts:

```bash
# Test development environment
make test-dev

# Test type safety configuration
make test-type

# Test tooling integration
make test-tooling

# Run all tests
make test-all
```

### 3. Building

```bash
# Build package
make build

# Full build (clean + check + build)
make build-all
```

## Configuration

### Type Checking (mypy)

Type checking is configured in `mypy.ini` with strict settings:

- Strict mode enabled
- No implicit Any types
- Comprehensive error reporting
- Third-party library stubs

Key configuration options:
```ini
[mypy]
strict = true
disallow_any_generics = true
disallow_untyped_defs = true
warn_return_any = true
```

### Linting and Formatting (ruff)

Ruff configuration is in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF"]
```

### Editor Configuration

The `.editorconfig` file provides consistent formatting across editors:

- UTF-8 encoding
- LF line endings
- 4-space indentation for Python
- 2-space indentation for YAML/JSON

## Development Scripts

### setup_dev.py

Sets up the development environment:
- Checks system dependencies
- Installs Python dependencies
- Creates necessary directories
- Runs initial code quality checks

```bash
python scripts/setup_dev.py
```

### build.py

Handles building and packaging:
- Cleans build artifacts
- Runs code quality checks
- Builds packages
- Generates requirements.txt

```bash
# Show help
python scripts/build.py --help

# Clean build artifacts
python scripts/build.py --clean

# Run checks only
python scripts/build.py --check

# Build package
python scripts/build.py --build

# Full workflow
python scripts/build.py --all
```

### dev_tools.py

Advanced development utilities:
- Dependency checking
- Detailed type checking
- Code metrics
- Project structure validation

```bash
# Show help
python scripts/dev_tools.py --help

# Run all tools
python scripts/dev_tools.py --all

# Fix code issues
python scripts/dev_tools.py --fix

# Validate project structure
python scripts/dev_tools.py --validate
```

## Manual Testing

### Test Scripts

The project includes comprehensive manual testing:

1. **Development Environment Test** (`examples/manual_tests/test_development_environment.py`)
   - Tests uv installation
   - Verifies Python version
   - Checks dependencies
   - Validates project structure
   - Tests tooling functionality

2. **Type Safety Test** (`examples/manual_tests/test_type_safety.py`)
   - Tests basic type annotations
   - Verifies Pydantic integration
   - Tests generic types
   - Tests Protocol types
   - Tests async types
   - Tests strict mode features

3. **Tooling Integration Test** (`examples/manual_tests/test_tooling_integration.py`)
   - Tests script execution
   - Tests mypy configuration
   - Tests ruff configuration
   - Tests complete workflow
   - Tests configuration files

### Running Tests

```bash
# Individual tests
python examples/manual_tests/test_development_environment.py
python examples/manual_tests/test_type_safety.py
python examples/manual_tests/test_tooling_integration.py

# All tests via Makefile
make test-all
```

## Configuration Examples

The `examples/config_examples/` directory contains:

- `example_global_config.json`: Standard configuration
- `development_config.json`: Development-specific settings
- `production_config.json`: Production-optimized settings
- `example_task_config.json`: Task-specific configuration

## Common Development Tasks

### Adding New Dependencies

1. Add to `pyproject.toml`:
   ```toml
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```

2. Update environment:
   ```bash
   uv sync
   ```

### Adding Type Stubs

For packages without type information:

1. Add to `mypy.ini`:
   ```ini
   [mypy-package_name.*]
   ignore_missing_imports = true
   ```

2. Or install type stubs:
   ```bash
   uv add types-package-name --dev
   ```

### Code Quality Issues

Common fixes:

```bash
# Fix formatting automatically
uv run ruff format src/

# Fix linting issues automatically
uv run ruff check src/ --fix

# Check what mypy found
uv run mypy src/zuup/ --show-error-codes
```

## Troubleshooting

### Common Issues

1. **uv not found**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart terminal
   ```

2. **Type checking errors**
   - Check `mypy.ini` configuration
   - Add missing type annotations
   - Add type stubs for third-party packages

3. **Import errors**
   - Ensure virtual environment is activated
   - Run `uv sync` to update dependencies

4. **Permission errors**
   - Check file permissions
   - Ensure proper directory ownership

### Getting Help

1. Check this documentation
2. Run manual tests to identify issues
3. Check tool-specific help:
   ```bash
   uv --help
   uv run mypy --help
   uv run ruff --help
   ```

## Best Practices

### Code Style

- Use type annotations for all functions and methods
- Follow PEP 8 style guidelines (enforced by ruff)
- Use Pydantic models for data validation
- Write docstrings for public APIs
- Keep functions focused and small

### Type Safety

- Use strict mypy configuration
- Avoid `Any` types when possible
- Use Protocol classes for interfaces
- Use generic types for collections
- Add type guards for runtime checks

### Development Workflow

1. Make changes
2. Run `make format` to format code
3. Run `make check` to verify quality
4. Run manual tests if needed
5. Commit changes

### Testing

- Use manual testing for functionality verification
- Test edge cases and error conditions
- Verify type safety with mypy
- Test configuration changes
- Document test procedures

## Performance Considerations

### Development Speed

- Use `make quick-check` for fast feedback
- Run full checks before committing
- Use incremental mypy checking
- Cache dependencies with uv

### Build Optimization

- Clean build artifacts regularly
- Use parallel processing where possible
- Optimize dependency resolution
- Monitor build times

## Security

### Development Security

- Keep dependencies updated
- Use secure configuration defaults
- Validate all inputs
- Follow security best practices
- Regular security audits

### Credential Management

- Never commit secrets
- Use environment variables
- Use secure storage for sensitive data
- Rotate credentials regularly