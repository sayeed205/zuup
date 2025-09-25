"""
Main entry point for the Zuup download manager application.

Provides CLI interface and application startup logic.
"""

from pathlib import Path
import sys

import click
from rich.console import Console

from .config.manager import ConfigManager
from .core.app import Application
from .utils.logging import setup_logging

console = Console()


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
@click.option("--gui", is_flag=True, help="Start with GUI interface")
@click.option("--server-only", is_flag=True, help="Start in server-only mode")
@click.option("--port", type=int, default=8080, help="Server port")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.pass_context
def start(
    ctx: click.Context,
    gui: bool,
    server_only: bool,
    port: int,
    host: str,
) -> None:
    """Start the Zuup download manager application."""
    config_manager = ctx.obj["config_manager"]

    try:
        app = Application(config_manager=config_manager)

        if server_only:
            console.print(f"[green]Starting server on {host}:{port}[/green]")
            app.start_server(host=host, port=port)
        elif gui:
            console.print("[green]Starting GUI application[/green]")
            app.start_gui()
        else:
            console.print(
                f"[green]Starting application with GUI and server on {host}:{port}[/green]"
            )
            app.start_combined(host=host, port=port)

    except KeyboardInterrupt:
        console.print("\n[yellow]Application stopped by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error starting application: {e}[/red]")
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


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
