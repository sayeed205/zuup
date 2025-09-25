# Zuup

A unified download manager supporting HTTP/HTTPS, FTP/SFTP, BitTorrent, and media downloads with a modern PySide6 GUI and browser extension support.

## Features

- **Multi-Protocol Support**: HTTP/HTTPS, FTP/SFTP, BitTorrent, and media downloads (yt-dlp)
- **Modern GUI**: PySide6-based user interface with dark/light themes
- **REST API**: Compatible with qBittorrent API for browser extension support
- **Type Safety**: Strict type checking with mypy and Pydantic models
- **Flexible Deployment**: GUI mode, server-only mode, or combined mode
- **Configuration Management**: Hierarchical configuration with global and per-task settings

## Quick Start

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd zuup
   uv sync --dev
   ```

3. **Setup development environment**:
   ```bash
   uv run python scripts/setup_dev.py
   ```

### Usage

#### GUI Mode
```bash
uv run zuup start --gui
```

#### Server Mode
```bash
uv run zuup start --server-only --port 8080
```

#### Combined Mode (GUI + Server)
```bash
uv run zuup start --port 8080
```

#### Command Line Download
```bash
uv run zuup download "https://example.com/file.zip" -o ~/Downloads
```

## Development

### Project Structure

```
src/zuup/
├── core/           # Core application logic
├── engines/        # Download protocol implementations
├── storage/        # Data persistence and models
├── api/           # REST API server
├── gui/           # PySide6 GUI application
├── cli/           # Command line interface
├── config/        # Configuration management
└── utils/         # Utility functions
```

### Development Workflow

1. **Code Quality Checks**:
   ```bash
   uv run mypy src/
   uv run ruff check src/
   uv run ruff format src/
   ```

2. **Testing**:
   ```bash
   uv run pytest
   uv run python examples/manual_tests/test_basic_functionality.py
   ```

3. **Build**:
   ```bash
   uv run python scripts/build.py --all
   ```

### Type Safety

This project uses strict type checking:
- All functions must have type annotations
- Pydantic models for data validation
- mypy with strict configuration
- Protocol classes for interfaces

## Configuration

Configuration files are stored in:
- **Linux/macOS**: `~/.config/zuup/`
- **Windows**: `%APPDATA%\zuup\`

See `docs/configuration.md` for detailed configuration options.

## API Documentation

The REST API is compatible with qBittorrent API. See `docs/api.md` for endpoint documentation.

## Contributing

1. Create a feature branch
2. Make changes with proper type annotations
3. Add tests for new functionality
4. Ensure all checks pass: `uv run python scripts/build.py --check`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Status

🚧 **Under Development** - This project is currently being implemented following a spec-driven development approach.

Current implementation status:
- ✅ Task 1: Project structure and dependency management
- ⏳ Task 2: Core type-safe data models (next)
- ⏳ Task 3: Configuration management system
- ⏳ Task 4: Download engine interfaces
- ⏳ Task 5: Task management system
- ⏳ Task 6: Logging and monitoring
- ⏳ Task 7: Application controller
- ⏳ Task 8: Development tooling
