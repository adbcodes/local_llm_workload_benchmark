from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TextIO

import typer

from llm_workload_benchmark import __version__
from llm_workload_benchmark.artifacts import (
    ArtifactError,
    export_experiment_artifacts,
)
from llm_workload_benchmark.adjudication import (
    AdjudicationError,
    adjudicate_experiment,
)
from llm_workload_benchmark.authoring import build_authoring_suite
from llm_workload_benchmark.catalog import CatalogError, validate_catalog
from llm_workload_benchmark.config import ConfigError, load_config
from llm_workload_benchmark.dataset import DatasetError
from llm_workload_benchmark.final_figures import generate_final_figure_bundle
from llm_workload_benchmark.rejudge import RejudgeError, rejudge_experiment
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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    rounded = int(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _progress_line(progress: RunProgress, *, width: int = 24) -> str:
    fraction = (
        progress.completed_items / progress.total_items
        if progress.total_items
        else 0.0
    )
    fraction = max(0.0, min(1.0, fraction))
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    eta = None
    if progress.completed_items:
        eta = (
            progress.elapsed_seconds
            / progress.completed_items
            * (progress.total_items - progress.completed_items)
        )
    return (
        f"[{bar}] {fraction:6.1%} | "
        f"model {progress.model_number}/{progress.model_count} "
        f"{progress.model_id} | {progress.benchmark} | "
        f"item {progress.completed_items}/{progress.total_items} | "
        f"elapsed {_format_duration(progress.elapsed_seconds)} | "
        f"ETA {_format_duration(eta)}"
    )


class _ProgressDisplay:
    """Render one updating terminal line, with readable output when redirected."""

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self._stream = stream
        self._live = bool(getattr(stream, "isatty", lambda: False)())
        self._line_open = False

    def update(self, progress: RunProgress) -> None:
        line = _progress_line(progress)
        completed = progress.completed_items >= progress.total_items
        if not self._live:
            self._stream.write(line + "\n")
            self._stream.flush()
            return

        self._stream.write("\r\033[2K" + line)
        if completed:
            self._stream.write("\n")
            self._line_open = False
        else:
            self._line_open = True
        self._stream.flush()

    def finish(self) -> None:
        if self._line_open:
            self._stream.write("\n")
            self._stream.flush()
            self._line_open = False


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
    skip_human_eval: bool = typer.Option(
        False,
        "--skip-human-eval",
        hidden=True,
        help="Compatibility flag; human preference collection has been removed.",
    ),
    resume_experiment: Path | None = typer.Option(
        None,
        "--resume-experiment",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Resume an interrupted matrix, preserving completed model runs.",
    ),
) -> None:
    """Run a resumable model matrix and export its artifacts."""

    del skip_human_eval
    progress_display = _ProgressDisplay()

    def show_progress(progress: RunProgress) -> None:
        progress_display.update(progress)

    try:
        config = load_config(config_path)
        _build_configured_dataset(config.benchmark.workload_path)
        experiment_directory = run_matrix(
            config,
            config_path,
            progress_callback=show_progress,
            resume_experiment=resume_experiment,
        )
    except (ConfigError, DatasetError, EvaluationError, OSError) as error:
        progress_display.finish()
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    progress_display.finish()

    try:
        artifact_paths = export_experiment_artifacts(
            experiment_directory,
            experiment_metadata={"kind": "model_matrix"},
        )
    except (ArtifactError, OSError) as error:
        typer.echo(
            f"Error: inference completed but artifact export failed: {error}; "
            f"raw experiment: {experiment_directory}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"\nExperiment saved to {experiment_directory}")
    _show_artifact_paths(artifact_paths)


@app.command("rejudge")
def rejudge_command(
    experiment: Path = typer.Option(
        ...,
        "--experiment",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Stopped experiment whose completed saved answers should be copied.",
    ),
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Matching config containing the replacement judge settings.",
    ),
) -> None:
    """Copy completed runs and re-evaluate saved answers with another judge."""

    def show_progress(model_id: str, completed: int, total: int) -> None:
        typer.echo(f"Rejudging {completed}/{total}: {model_id}")

    try:
        config = load_config(config_path)
        target = rejudge_experiment(
            experiment,
            config,
            config_path,
            progress_callback=show_progress,
        )
    except (ConfigError, DatasetError, RejudgeError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    index = json.loads((target / "experiment.json").read_text(encoding="utf-8"))
    if index.get("status") == "paused_rate_limit":
        typer.echo(
            "Rejudging paused after a rate-limit response. Completed judgments "
            f"are saved in {target}; rerun the same command to resume.",
            err=True,
        )
        raise typer.Exit(code=75)
    typer.echo(f"Rejudged experiment saved to {target}")


@app.command("adjudicate")
def adjudicate_command(
    experiment: Path = typer.Option(
        ...,
        "--experiment",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Completed inference experiment whose semantic requirements should be judged.",
    ),
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Matching workload config containing one judge.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        max=16,
        help="Bounded concurrent judge calls after inference is complete.",
    ),
) -> None:
    """Judge prose meaning into immutable sidecars without rerunning inference."""

    def show_progress(completed: int, total: int, model_id: str) -> None:
        typer.echo(f"Adjudicating {completed}/{total}: {model_id}")

    try:
        config = load_config(config_path)
        output = adjudicate_experiment(
            experiment,
            config,
            workers=workers,
            progress_callback=show_progress,
        )
    except (ConfigError, DatasetError, AdjudicationError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_status") == "paused_rate_limit":
        typer.echo(
            "Adjudication paused after a rate-limit response. Completed sidecars "
            f"are saved in {output}; rerun the same command to resume.",
            err=True,
        )
        raise typer.Exit(code=75)
    typer.echo(f"Adjudication sidecars saved to {output}")


def _build_configured_dataset(workload_path: Path) -> None:
    suite_path = (
        workload_path if workload_path.is_absolute() else Path.cwd() / workload_path
    )
    build_authoring_suite(suite_path)


def _show_artifact_paths(paths: dict[str, Path]) -> None:
    typer.echo(f"Artifact manifest: {paths['manifest']}")
    typer.echo(f"Configuration CSV: {paths['configurations']}")


@app.command("artifacts")
def artifacts_command(
    experiment_directory: Path = typer.Option(
        ...,
        "--experiment",
        "-e",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Saved matrix experiment to export without rerunning inference.",
    ),
) -> None:
    """Regenerate artifacts from a saved matrix experiment."""
    try:
        paths = export_experiment_artifacts(experiment_directory)
    except (ArtifactError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    _show_artifact_paths(paths)


@app.command("figures")
def figures_command(
    five_experiment: Path = typer.Option(..., "--five-experiment", exists=True,
                                         file_okay=False, resolve_path=True),
    retrieval_experiment: Path = typer.Option(..., "--retrieval-experiment", exists=True,
                                              file_okay=False, resolve_path=True),
    grounded_experiment: Path = typer.Option(..., "--grounded-experiment", exists=True,
                                             file_okay=False, resolve_path=True),
    output_directory: Path = typer.Option(
        Path("final_figures"),
        "--output",
        file_okay=False,
        resolve_path=True,
        help="Repository-level destination for the combined final figure bundle.",
    ),
) -> None:
    """Generate final adjudication-aware figures from all three result runs."""
    try:
        root = generate_final_figure_bundle(
            five_experiment,
            retrieval_experiment,
            grounded_experiment,
            output_directory,
        )
    except (ArtifactError, OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Final figure manifest: {root / 'manifest.json'}")
