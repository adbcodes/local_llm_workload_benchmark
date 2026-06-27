from io import StringIO

from llm_workload_benchmark.cli import (
    _format_duration,
    _ProgressDisplay,
    _progress_line,
)
from llm_workload_benchmark.runner import RunProgress


def test_progress_line_shows_bar_percentage_timing_and_eta() -> None:
    line = _progress_line(
        RunProgress(
            model_id="model-q4",
            model_number=2,
            model_count=4,
            benchmark="structured-output",
            completed_items=25,
            total_items=100,
            elapsed_seconds=50.0,
        ),
        width=8,
    )

    assert "[##------]" in line
    assert "25.0%" in line
    assert "model 2/4 model-q4" in line
    assert "item 25/100" in line
    assert "elapsed 00:50" in line
    assert "ETA 02:30" in line


def test_format_duration_supports_hours_and_unknown_eta() -> None:
    assert _format_duration(3723.9) == "1:02:03"
    assert _format_duration(None) == "--:--"


class _TerminalBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def _progress(completed: int, total: int = 4) -> RunProgress:
    return RunProgress(
        model_id="tiny-model",
        model_number=1,
        model_count=2,
        benchmark="tiny-suite",
        completed_items=completed,
        total_items=total,
        elapsed_seconds=float(completed),
    )


def test_live_progress_updates_one_terminal_line() -> None:
    output = _TerminalBuffer()
    display = _ProgressDisplay(output)

    display.update(_progress(1))
    display.update(_progress(2))
    display.update(_progress(4))

    rendered = output.getvalue()
    assert rendered.count("\r\033[2K") == 3
    assert rendered.count("\n") == 1
    assert "25.0%" in rendered
    assert "50.0%" in rendered
    assert "100.0%" in rendered


def test_live_progress_finishes_partial_line_before_error_output() -> None:
    output = _TerminalBuffer()
    display = _ProgressDisplay(output)
    display.update(_progress(1))

    display.finish()

    assert output.getvalue().endswith("\n")
