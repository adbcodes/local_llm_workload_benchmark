from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportError(ValueError):
    """Raised when an experiment cannot produce a comparison report."""


def generate_comparison_report(
    experiment_directory: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Create a small Markdown model comparison from saved matrix summaries."""

    experiment_root = experiment_directory.resolve()
    index = _read_json_object(experiment_root / "experiment.json")
    model_entries = index.get("models")
    if not isinstance(model_entries, list) or not model_entries:
        raise ReportError("experiment index must contain at least one model entry")

    rows = [_build_row(experiment_root, entry) for entry in model_entries]
    destination = (output_path or experiment_root / "comparison.md").resolve()
    report = _render_markdown(index, rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(report, encoding="utf-8")
    temporary_path.replace(destination)
    return destination


def _build_row(experiment_root: Path, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ReportError("every experiment model entry must be an object")
    model_id = entry.get("model_id")
    status = entry.get("status")
    if not isinstance(model_id, str) or not isinstance(status, str):
        raise ReportError("model entries require string model_id and status fields")

    row: dict[str, Any] = {"model_id": model_id, "status": status}
    summary_reference = entry.get("summary")
    if not isinstance(summary_reference, str):
        return row

    summary_path = (experiment_root / summary_reference).resolve()
    if not summary_path.is_relative_to(experiment_root):
        raise ReportError(f"summary path escapes experiment directory: {summary_reference}")
    summary = _read_json_object(summary_path)
    totals = summary.get("totals")
    if not isinstance(totals, dict):
        return row
    judge = summary.get("judge")
    by_origin = summary.get("by_origin")
    licensed = (
        by_origin.get("licensed_anchor") if isinstance(by_origin, dict) else None
    )
    generated = (
        by_origin.get("fresh_generated") if isinstance(by_origin, dict) else None
    )
    row.update(
        {
            "attempted": totals.get("attempted"),
            "passed": totals.get("passed"),
            "pass_rate": totals.get("pass_rate"),
            "mean_score": totals.get("mean_score"),
            "licensed_pass_rate": (
                licensed.get("pass_rate") if isinstance(licensed, dict) else None
            ),
            "generated_pass_rate": (
                generated.get("pass_rate") if isinstance(generated, dict) else None
            ),
            "mean_latency_seconds": totals.get("mean_latency_seconds"),
            "mean_ttft_seconds": totals.get("mean_time_to_first_token_seconds"),
            "mean_output_rate": totals.get(
                "mean_output_tokens_per_second_end_to_end"
            ),
            "peak_memory_bytes": totals.get("peak_process_memory_bytes"),
            "judge_cost_usd": (
                judge.get("estimated_cost_usd") if isinstance(judge, dict) else None
            ),
        }
    )
    return row


def _render_markdown(index: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    experiment_id = index.get("experiment_id", "unknown")
    status = index.get("status", "unknown")
    dataset = index.get("dataset", "unknown")
    lines = [
        "# Model Comparison",
        "",
        f"- Experiment: `{experiment_id}`",
        f"- Status: `{status}`",
        f"- Dataset: `{dataset}`",
        "",
        "| Model | Status | Passes | Pass rate | Mean score | Licensed anchors | Fresh generated | Mean latency | Mean TTFT | Output tok/s | Peak RSS | Judge cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(row["model_id"]),
                    _escape_cell(row["status"]),
                    _format_passes(row.get("passed"), row.get("attempted")),
                    _format_percent(row.get("pass_rate")),
                    _format_percent(row.get("mean_score")),
                    _format_percent(row.get("licensed_pass_rate")),
                    _format_percent(row.get("generated_pass_rate")),
                    _format_seconds(row.get("mean_latency_seconds")),
                    _format_seconds(row.get("mean_ttft_seconds")),
                    _format_number(row.get("mean_output_rate")),
                    _format_memory(row.get("peak_memory_bytes")),
                    _format_cost(row.get("judge_cost_usd")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "> Pilot results verify the pipeline; they are not model rankings.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportError(f"required report artifact does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"could not read report artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"report artifact must contain a JSON object: {path}")
    return value


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _format_passes(passed: Any, attempted: Any) -> str:
    if isinstance(passed, int) and isinstance(attempted, int):
        return f"{passed}/{attempted}"
    return "—"


def _format_percent(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, int | float) else "—"


def _format_seconds(value: Any) -> str:
    return f"{value:.3f}s" if isinstance(value, int | float) else "—"


def _format_number(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, int | float) else "—"


def _format_memory(value: Any) -> str:
    return f"{value / (1024**3):.2f} GiB" if isinstance(value, int | float) else "—"


def _format_cost(value: Any) -> str:
    return f"${value:.6f}" if isinstance(value, int | float) else "—"
