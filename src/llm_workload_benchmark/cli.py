from __future__ import annotations

from pathlib import Path

import typer

from llm_workload_benchmark import __version__
from llm_workload_benchmark.authoring import build_authoring_suite
from llm_workload_benchmark.catalog import CatalogError, validate_catalog
from llm_workload_benchmark.config import ConfigError, load_config
from llm_workload_benchmark.dataset import DatasetError
from llm_workload_benchmark.preference import (
    PreferenceError,
    completed_model_ids,
)
from llm_workload_benchmark.preference_terminal import run_terminal_preferences
from llm_workload_benchmark.report import ReportError, generate_comparison_report
from llm_workload_benchmark.runtime_matrix import (
    RuntimeMatrixError,
    combination_count,
    load_runtime_matrix,
    run_runtime_matrix,
)
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
dataset_app = typer.Typer(help="Build and validate benchmark datasets.")
app.add_typer(dataset_app, name="dataset")


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


@dataset_app.command("build")
def dataset_build_command(
    suite_path: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Suite manifest whose YAML authoring files should be compiled.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Fail if generated JSONL differs instead of updating it.",
    ),
) -> None:
    """Compile readable YAML authoring files into reusable runtime JSONL."""
    try:
        result = build_authoring_suite(suite_path, check=check)
    except (DatasetError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if result.written:
        typer.echo(f"Built {len(result.written)} JSONL file(s).")
    else:
        typer.echo("Dataset JSONL is already up to date.")


@dataset_app.command("validate")
def dataset_validate_command(
    catalog_path: Path = typer.Option(
        Path("data/catalog.yaml"),
        "--catalog",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Final benchmark catalog to validate.",
    ),
) -> None:
    """Validate the six-suite catalog, templates, resources, and all suite."""
    try:
        result = validate_catalog(catalog_path)
    except (CatalogError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Valid: {result.benchmark_count} catalog entries, "
        f"{result.question_set_count} question sets, "
        f"{result.current_question_count} current questions, "
        f"{result.planned_question_set_count} empty templates."
    )


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
        _build_configured_dataset(config.benchmark.workload_path)
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
    human_eval: bool = typer.Option(
        True,
        "--human-eval/--skip-human-eval",
        help="Run blind multi-model terminal voting after model generation.",
    ),
    color: bool = typer.Option(
        True,
        "--color/--no-color",
        help="Enable Python syntax colours in the human ballot.",
    ),
) -> None:
    """Run the model matrix, then collect blind terminal preferences."""

    def show_progress(progress: RunProgress) -> None:
        typer.echo(
            f"[{progress.model_number}/{progress.model_count}] "
            f"{progress.model_id} | {progress.benchmark} | "
            f"item {progress.completed_items}/{progress.total_items} | "
            f"{progress.elapsed_seconds:.1f}s"
        )

    try:
        config = load_config(config_path)
        _build_configured_dataset(config.benchmark.workload_path)
        experiment_directory = run_matrix(
            config,
            config_path,
            progress_callback=show_progress,
        )
    except (ConfigError, DatasetError, EvaluationError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if human_eval:
        try:
            model_ids = completed_model_ids(experiment_directory)
            if len(model_ids) >= 2:
                typer.echo(
                    f"\nHUMAN PREFERENCE  {len(model_ids)} anonymous answers"
                )
                result = run_terminal_preferences(
                    experiment_directory,
                    model_ids=model_ids,
                    seed=config.benchmark.seed,
                    input_fn=typer.prompt,
                    output_fn=lambda value: typer.echo(value, color=color),
                    color=color,
                )
                if not result.is_complete:
                    typer.echo("Human evaluation paused. Run `prefer` to resume.")
                else:
                    typer.echo("Ballot complete. Vote saved.")
        except (PreferenceError, OSError) as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1) from error

    typer.echo(f"\nExperiment saved to {experiment_directory}")


@app.command("runtime-matrix")
def runtime_matrix_command(
    config_path: Path = typer.Option(
        Path("configs/runtime_matrix.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Quantization and runtime-axis matrix configuration.",
    ),
) -> None:
    """Run every configured quantization and setting combination."""

    def show_progress(progress: RunProgress) -> None:
        typer.echo(
            f"[{progress.model_number}/{progress.model_count}] "
            f"{progress.model_id} | {progress.benchmark} | "
            f"{progress.completed_items}/{progress.total_items} | "
            f"{progress.elapsed_seconds:.1f}s"
        )

    try:
        runtime_config = load_runtime_matrix(config_path)
        _build_configured_dataset(runtime_config.benchmark.workload_path)
        typer.echo(
            f"Running {combination_count(runtime_config)} runtime combinations."
        )
        experiment = run_runtime_matrix(
            runtime_config,
            config_path,
            progress_callback=show_progress,
        )
        report_path = generate_comparison_report(experiment)
    except (
        RuntimeMatrixError,
        ConfigError,
        DatasetError,
        EvaluationError,
        ReportError,
        OSError,
    ) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Runtime results: {experiment / 'runtime_results.json'}")
    typer.echo(f"Graph-ready CSV: {experiment / 'runtime_runs.csv'}")
    typer.echo(f"Comparison report: {report_path}")


def _build_configured_dataset(workload_path: Path) -> None:
    suite_path = (
        workload_path if workload_path.is_absolute() else Path.cwd() / workload_path
    )
    build_authoring_suite(suite_path)


@app.command("report")
def report_command(
    experiment_directory: Path = typer.Option(
        ...,
        "--experiment",
        "-e",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Saved matrix experiment directory to summarize.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Markdown destination; defaults to <experiment>/comparison.md.",
    ),
) -> None:
    """Generate a Markdown comparison from a saved matrix experiment."""
    try:
        report_path = generate_comparison_report(
            experiment_directory,
            output_path=output_path,
        )
    except (ReportError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(report_path)


@app.command("prefer")
def prefer_command(
    experiment_directory: Path = typer.Option(
        ...,
        "--experiment",
        "-e",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Saved matrix experiment containing completed model answers.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Seed for stable anonymous answer ordering.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Vote destination; defaults to the experiment arena artifact.",
    ),
    color: bool = typer.Option(
        True,
        "--color/--no-color",
        help="Enable Python syntax colours in the terminal ballot.",
    ),
) -> None:
    """Run one blind terminal ballot for all completed model answers."""

    try:
        model_ids = completed_model_ids(experiment_directory)
        typer.echo(f"HUMAN PREFERENCE  {len(model_ids)} anonymous answers")
        result = run_terminal_preferences(
            experiment_directory,
            model_ids=model_ids,
            seed=seed,
            output_path=output_path,
            input_fn=typer.prompt,
            output_fn=lambda value: typer.echo(value, color=color),
            color=color,
        )
        if not result.output_path.exists():
            typer.echo("\nVoting stopped before a vote was cast.")
            return
        if not result.is_complete:
            typer.echo("\nVoting progress saved.")
            return
        typer.echo("Ballot complete. Vote saved.")
    except (PreferenceError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
