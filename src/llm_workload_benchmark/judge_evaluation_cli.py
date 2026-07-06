from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

import typer

from llm_workload_benchmark.judge_evaluation import (
    JudgeEvaluationError,
    load_judge_evaluation_config,
    run_judge_evaluation,
)
from llm_workload_benchmark.judge import JudgeError


app = typer.Typer(
    name="llm-judge-evaluation",
    help="Evaluate the production LLM judge against a human-labelled fixture.",
    no_args_is_help=True,
)


def _progress_line(completed: int, total: int, case_id: str, *, width: int = 28) -> str:
    fraction = completed / total if total else 0.0
    fraction = max(0.0, min(1.0, fraction))
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction:6.1%} | {completed}/{total} | {case_id}"


class _ProgressDisplay:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self._stream = stream
        self._line_open = False

    def update(self, completed: int, total: int, case_id: str) -> None:
        self._stream.write("\r\033[2K" + _progress_line(completed, total, case_id))
        self._line_open = completed < total
        if not self._line_open:
            self._stream.write("\n")
        self._stream.flush()

    def finish(self) -> None:
        if self._line_open:
            self._stream.write("\n")
            self._stream.flush()
            self._line_open = False


@app.command()
def run(
    config_path: Path = typer.Option(
        Path("configs/judge_evaluation.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_directory: Path | None = typer.Option(
        None,
        "--output",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Use a fixed directory to resume or inspect a calibration run.",
    ),
) -> None:
    """Send fixture cases to the judge one at a time and report reliability."""

    progress_display = _ProgressDisplay()

    def progress(completed: int, total: int, case_id: str) -> None:
        progress_display.update(completed, total, case_id)

    try:
        config = load_judge_evaluation_config(config_path)
        output = run_judge_evaluation(
            config,
            output_directory=output_directory,
            progress_callback=progress,
        )
    except (JudgeEvaluationError, JudgeError, OSError) as error:
        progress_display.finish()
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Judge evaluation saved to {output}")
