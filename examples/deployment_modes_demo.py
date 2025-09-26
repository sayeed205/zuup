#!/usr/bin/env python3
"""
Demonstration of different deployment modes for Zuup Download Manager.

This script shows how to programmatically start the application in different modes
and interact with the API.
"""

import asyncio
import logging
from pathlib import Path
import sys

import httpx
from rich.console import Console
from rich.table import Table

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zuup.config.manager import ConfigManager
from zuup.core.app import Application

console = Console()
logger = logging.getLogger(__name__)


async def test_server_mode():
    """Test headless server mode."""
    console.print("[bold blue]Testing Headless Server Mode[/bold blue]")

    # Initialize configuration
    config_manager = ConfigManager()

    # Create application
    app = Application(config_manager=config_manager)

    try:
        # Start server in background
        server_task = asyncio.create_task(
            asyncio.to_thread(app.start_server, "127.0.0.1", 8081)
        )

        # Wait a moment for server to start
        await asyncio.sleep(2)

        # Test API endpoints
        async with httpx.AsyncClient() as client:
            # Health check
            response = await client.get("http://127.0.0.1:8081/api/v1/health")
            console.print(f"Health check: {response.json()}")

            # Status check
            response = await client.get("http://127.0.0.1:8081/api/v1/status")
            console.print(f"Status: {response.json()}")

            # List tasks
            response = await client.get("http://127.0.0.1:8081/api/v1/tasks")
            console.print(f"Tasks: {response.json()}")

        console.print("[green]✓ Server mode test completed[/green]")

        # Stop server
        server_task.cancel()

    except Exception as e:
        console.print(f"[red]✗ Server mode test failed: {e}[/red]")


def test_gui_mode():
    """Test GUI mode (requires display)."""
    console.print("[bold blue]Testing GUI Mode[/bold blue]")

    try:
        # Check if display is available
        import os

        if not os.environ.get("DISPLAY") and not sys.platform.startswith("win"):
            console.print("[yellow]⚠ No display available, skipping GUI test[/yellow]")
            return

        # Initialize configuration
        config_manager = ConfigManager()

        # Create application
        app = Application(config_manager=config_manager)

        console.print(
            "[dim]GUI mode would start here (requires user interaction)[/dim]"
        )
        console.print("[green]✓ GUI mode initialization successful[/green]")

    except ImportError as e:
        console.print(f"[yellow]⚠ GUI dependencies not available: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ GUI mode test failed: {e}[/red]")


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples."""
    console.print("[bold blue]CLI Usage Examples[/bold blue]")

    table = Table(title="Deployment Modes")
    table.add_column("Mode", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Description", style="white")

    table.add_row("Combined (Default)", "zuup start", "GUI with embedded server")
    table.add_row("GUI Only", "zuup start --gui", "GUI with local server")
    table.add_row(
        "Server Only", "zuup start --server-only", "Headless server for remote access"
    )
    table.add_row("Custom Port", "zuup start --port 9000", "Custom server port")
    table.add_row(
        "Remote Server",
        "zuup start --server-only --host 0.0.0.0",
        "Server accessible from network",
    )
    table.add_row("Status Check", "zuup status", "Check if server is running")

    console.print(table)


async def main():
    """Main demonstration function."""
    console.print(
        "[bold green]Zuup Download Manager - Deployment Modes Demo[/bold green]"
    )
    console.print()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Demonstrate CLI usage
    demonstrate_cli_usage()
    console.print()

    # Test server mode
    await test_server_mode()
    console.print()

    # Test GUI mode
    test_gui_mode()
    console.print()

    console.print("[bold green]Demo completed![/bold green]")
    console.print()
    console.print("[dim]To try the actual application:[/dim]")
    console.print("  [cyan]zuup start[/cyan]                    # Combined mode")
    console.print("  [cyan]zuup start --server-only[/cyan]     # Server mode")
    console.print("  [cyan]zuup start --gui[/cyan]             # GUI mode")
    console.print("  [cyan]zuup status[/cyan]                  # Check status")


if __name__ == "__main__":
    asyncio.run(main())
