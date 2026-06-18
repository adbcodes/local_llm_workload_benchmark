from __future__ import annotations

from pathlib import Path

import typer

from llm_workload_benchmark import __version__
from llm_workload_benchmark.config import ConfigError, load_config
from llm_workload_benchmark.dataset import DatasetError
from llm_workload_benchmark.runner import (
    EvaluationError,
    RunProgress,
    run_benchmark,
    run_matrix,
)

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


@app.command("run")
def run_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML benchmark configuration to execute.",
    ),
) -> None:
    """Run the schema-pilot workload on one enabled local model."""
    try:
        config = load_config(config_path)
        run_directory = run_benchmark(config, config_path)
    except (ConfigError, DatasetError, EvaluationError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(run_directory)


@app.command("benchmark")
def benchmark_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML model-matrix configuration to execute.",
    ),
) -> None:
    """Run every enabled model sequentially in one experiment bundle."""

    def show_progress(progress: RunProgress) -> None:
        typer.echo(
            f"[{progress.model_number}/{progress.model_count}] "
            f"{progress.model_id} | {progress.benchmark} | "
            f"item {progress.completed_items}/{progress.total_items} | "
            f"{progress.elapsed_seconds:.1f}s"
        )

    try:
        config = load_config(config_path)
        experiment_directory = run_matrix(
            config,
            config_path,
            progress_callback=show_progress,
        )
    except (ConfigError, DatasetError, EvaluationError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(experiment_directory)
