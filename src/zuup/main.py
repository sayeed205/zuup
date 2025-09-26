"""
Main entry point for the Zuup download manager application.

Provides CLI interface and application startup logic.
"""

import logging
from pathlib import Path
import sys

import click
from rich.console import Console

from .config.manager import ConfigManager
from .core.app import Application
from .utils.logging import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
@click.option(
    "--config-dir",
    type=click.Path(exists=False, path_type=Path),
    help="Configuration directory path",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.pass_context
def cli(ctx: click.Context, config_dir: Path | None, log_level: str) -> None:
    """Zuup Download Manager CLI."""
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level)

    # Initialize configuration
    config_manager = ConfigManager(config_dir=config_dir)
    ctx.obj["config_manager"] = config_manager


@cli.command()
@click.option("--gui", is_flag=True, help="Start with GUI interface only")
@click.option("--server-only", is_flag=True, help="Start in headless server mode")
@click.option("--port", type=int, help="Server port (overrides config)")
@click.option("--host", help="Server host (overrides config)")
@click.pass_context
def start(
    ctx: click.Context,
    gui: bool,
    server_only: bool,
    port: int | None,
    host: str | None,
) -> None:
    """
    Start the Zuup download manager application.
    
    Deployment modes:
    - Default: GUI with embedded server (combined mode)
    - --gui: GUI only with embedded server on localhost
    - --server-only: Headless server mode for remote access
    """
    config_manager = ctx.obj["config_manager"]

    try:
        app = Application(config_manager=config_manager)

        # Get configuration for defaults
        global_config = config_manager.get_global_config()
        server_host = host or global_config.server_host
        server_port = port or global_config.server_port

        if server_only:
            console.print(f"[green]Starting headless server on {server_host}:{server_port}[/green]")
            console.print("[dim]Use Ctrl+C to stop the server[/dim]")
            app.start_server(host=server_host, port=server_port)
        elif gui:
            console.print("[green]Starting GUI with embedded server[/green]")
            app.start_gui()
        else:
            console.print(
                f"[green]Starting combined mode (GUI + server) on {server_host}:{server_port}[/green]"
            )
            console.print("[dim]Access API at http://{server_host}:{server_port}[/dim]")
            app.start_combined(host=server_host, port=server_port)

    except KeyboardInterrupt:
        console.print("\n[yellow]Application stopped by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error starting application: {e}[/red]")
        logger.exception("Application startup failed")
        sys.exit(1)


@cli.command()
@click.argument("url")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output directory"
)
@click.option(
    "--engine",
    type=click.Choice(["auto", "http", "ftp", "torrent", "media"]),
    default="auto",
)
def download(
    url: str,
    output: Path | None,
    engine: str,
) -> None:
    """Download a file from URL."""
    try:
        # This will be implemented in later tasks
        console.print(
            "[yellow]Download functionality will be implemented in task 5[/yellow]"
        )
        console.print(f"URL: {url}")
        console.print(f"Output: {output}")
        console.print(f"Engine: {engine}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--port", type=int, help="Server port to check")
@click.option("--host", default="127.0.0.1", help="Server host to check")
@click.pass_context
def status(
    ctx: click.Context,
    port: int | None,
    host: str,
) -> None:
    """Check application status via API."""
    config_manager = ctx.obj["config_manager"]

    try:
        # Get configuration for defaults
        global_config = config_manager.get_global_config()
        server_port = port or global_config.server_port

        import httpx

        # Check if server is running
        url = f"http://{host}:{server_port}/api/v1/health"

        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✓[/green] Server running on {host}:{server_port}")
                console.print(f"Status: {data.get('status', 'unknown')}")

                components = data.get('components', {})
                for component, status in components.items():
                    status_icon = "[green]✓[/green]" if status else "[red]✗[/red]"
                    console.print(f"  {status_icon} {component}")
            else:
                console.print(f"[red]✗[/red] Server error: HTTP {response.status_code}")

    except httpx.ConnectError:
        console.print(f"[red]✗[/red] Cannot connect to server on {host}:{server_port}")
        console.print("[dim]Make sure the application is running with --server-only or combined mode[/dim]")
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
