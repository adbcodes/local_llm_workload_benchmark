from __future__ import annotations

from pathlib import Path

import typer

from llm_workload_benchmark import __version__
from llm_workload_benchmark.config import ConfigError, load_config
from llm_workload_benchmark.manifest import create_run

app = typer.Typer(
    name="llm-benchmark",
    help="Benchmark local language models on a reproducible workload.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """Local LLM workload benchmark commands."""


@app.command("create-run")
def create_run_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML benchmark configuration to record.",
    ),
) -> None:
    """Create a run directory and write its manifest."""
    try:
        config = load_config(config_path)
        run_directory = create_run(config, config_path)
    except (ConfigError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(run_directory)
