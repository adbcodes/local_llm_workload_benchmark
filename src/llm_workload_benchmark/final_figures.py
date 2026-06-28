from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil
from statistics import median
from typing import Any


FAMILY_ORDER = ["qwen2.5", "gemma3", "qwen3"]
FAMILY_LABELS = {
    "qwen2.5": "Qwen2.5 3B",
    "gemma3": "Gemma 3 4B",
    "qwen3": "Qwen3 8B",
}
FAMILY_COLORS = {
    "qwen2.5": "#1565C0",
    "gemma3": "#D32F2F",
    "qwen3": "#2E7D32",
}
QUANT_ORDER = ["Q8", "Q6", "Q4", "Q3"]
CONTEXT_ORDER = ["2K", "4K", "8K"]
QUANTIZATION_BENCHMARKS = {
    "quant_applied_reasoning": (
        "applied_reasoning",
        "Applied Reasoning vs Quantization",
    ),
    "quant_code_debug_repair": (
        "code_debug_repair",
        "Code Debug & Repair vs Quantization",
    ),
    "quant_messy_text_to_schema": (
        "messy_text_to_schema",
        "Schema Extraction vs Quantization",
    ),
    "quant_confidence_correctness": (
        "confidence_correctness",
        "Confidence & Correctness vs Quantization",
    ),
    "quant_raw_output_discipline": (
        "raw_output_discipline",
        "Raw Output Discipline vs Quantization",
    ),
}


def quantization_benchmark_score(
    root: Path,
    item_rows: list[dict[str, Any]],
    *,
    benchmark: str,
    stem: str,
    title: str,
) -> dict[str, Any]:
    """Plot one benchmark's mean evaluator score across quantizations."""

    grouped: dict[tuple[str, str], list[float]] = {}
    for source in item_rows:
        if source.get("benchmark") != benchmark:
            continue
        family = str(source.get("family"))
        quantization = _quant(source.get("quantization"))
        score = _number_or_none(source.get("score"))
        if family not in FAMILY_COLORS or quantization not in QUANT_ORDER or score is None:
            continue
        grouped.setdefault((family, quantization), []).append(score)

    rows = [
        {
            "family": family,
            "quantization": quantization,
            "mean_score_percent": sum(values) / len(values) * 100,
            "n": len(values),
        }
        for (family, quantization), values in grouped.items()
    ]
    rows.sort(key=lambda row: (
        FAMILY_ORDER.index(str(row["family"])),
        QUANT_ORDER.index(str(row["quantization"])),
    ))
    if not rows:
        return _skipped(f"requires item scores for {benchmark}")

    _write_csv(root / "data" / f"{stem}.csv", rows)
    paths = _render_line_chart(
        root,
        stem,
        rows,
        x_key="quantization",
        x_order=QUANT_ORDER,
        y_key="mean_score_percent",
        y_label="Mean evaluator score (%)",
        title=title,
        y_limit=(0, 105),
    )
    return _generated(paths, len(rows), benchmark=benchmark)


def quantization_tool_trace_parseability(
    root: Path, item_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Plot JSON tool-trace parseability without claiming tool correctness."""

    grouped: dict[tuple[str, str], list[bool]] = {}
    for source in item_rows:
        if source.get("benchmark") != "tool_use":
            continue
        family = str(source.get("family"))
        quantization = _quant(source.get("quantization"))
        if family not in FAMILY_COLORS or quantization not in QUANT_ORDER:
            continue
        grouped.setdefault((family, quantization), []).append(
            source.get("integration_outcome")
            in {"scored", "scored_cleanly", "scored_after_recovery"}
        )

    rows = [
        {
            "family": family,
            "quantization": quantization,
            "parseable": sum(values),
            "attempted": len(values),
            "parseability_percent": sum(values) / len(values) * 100,
            "n": len(values),
        }
        for (family, quantization), values in grouped.items()
        if values
    ]
    rows.sort(key=lambda row: (
        FAMILY_ORDER.index(str(row["family"])),
        QUANT_ORDER.index(str(row["quantization"])),
    ))
    if not rows:
        return _skipped("requires tool-use integration outcomes")

    stem = "quant_tool_trace_parseability"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    paths = _render_line_chart(
        root,
        stem,
        rows,
        x_key="quantization",
        x_order=QUANT_ORDER,
        y_key="parseability_percent",
        y_label="Parseable JSON tool traces (%)",
        title="Tool Trace Parseability vs Quantization",
        y_limit=(0, 105),
    )
    return _generated(paths, len(rows), benchmark="tool_use")


def constraint_load_score(
    root: Path, item_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Plot Q4 task score as the number of cumulative constraints increases."""

    rows = _constraint_rows(item_rows, metric="score")
    if not rows:
        return _skipped("requires Q4 constraint-load item scores")
    stem = "constraint_load_score"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    paths = _render_line_chart(
        root,
        stem,
        rows,
        x_key="constraint_count",
        x_order=[1, 2, 3, 4],
        y_key="mean_score_percent",
        y_label="Mean task score (%)",
        title="Performance as Constraints Increase — Q4",
        y_limit=(0, 105),
    )
    return _generated(paths, len(rows), quantization="Q4")


def constraint_content_retention(
    root: Path, item_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Plot Q4 content retention as cumulative constraints increase."""

    rows = _constraint_rows(item_rows, metric="content_score")
    if not rows:
        return _skipped("requires Q4 constraint-load content scores")
    stem = "constraint_content_retention"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    paths = _render_line_chart(
        root,
        stem,
        rows,
        x_key="constraint_count",
        x_order=[1, 2, 3, 4],
        y_key="mean_score_percent",
        y_label="Content retained (%)",
        title="Content Retention as Constraints Increase — Q4",
        y_limit=(0, 105),
    )
    return _generated(paths, len(rows), quantization="Q4")


def context_ttft(root: Path, context_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Plot median time to first token at Q4 for 2K, 4K, and 8K contexts."""

    grouped: dict[tuple[str, str], list[float]] = {}
    for source in context_items:
        family = str(source.get("family"))
        if family not in FAMILY_COLORS or _quant(source.get("quantization")) != "Q4":
            continue
        context_length = _context_length(_tags(source.get("tags")))
        ttft = _number_or_none(source.get("ttft_seconds"))
        if context_length is None or ttft is None:
            continue
        grouped.setdefault((family, context_length), []).append(ttft)

    rows = [
        {
            "family": family,
            "context_length": context_length,
            "median_ttft_seconds": median(values),
            "n": len(values),
        }
        for (family, context_length), values in grouped.items()
    ]
    rows.sort(key=lambda row: (
        FAMILY_ORDER.index(str(row["family"])),
        CONTEXT_ORDER.index(str(row["context_length"])),
    ))
    if not rows:
        return _skipped("requires Q4 context-profile TTFT measurements")

    stem = "context_ttft"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    maximum = max(float(row["median_ttft_seconds"]) for row in rows)
    paths = _render_line_chart(
        root,
        stem,
        rows,
        x_key="context_length",
        x_order=CONTEXT_ORDER,
        y_key="median_ttft_seconds",
        y_label="Median time to first token (seconds)",
        title="Prompt Delay as Context Grows — Q4",
        y_limit=(0, maximum * 1.15),
    )
    return _generated(paths, len(rows), quantization="Q4")


def generate_final_figure_bundle(
    default_experiment: Path,
    context_experiment: Path,
) -> Path:
    """Generate the compact evidence-backed figure bundle from saved runs."""

    from llm_workload_benchmark.artifacts import export_experiment_artifacts

    for experiment in dict.fromkeys((default_experiment, context_experiment)):
        export_experiment_artifacts(experiment)

    default_root = default_experiment / "artifacts"
    context_root = context_experiment / "artifacts"
    output = default_root / "final_figures"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    items = _read_csv(default_root / "data" / "items.csv")
    context_items = _read_csv(context_root / "data" / "items.csv")
    plots = {
        stem: quantization_benchmark_score(
            output,
            items,
            benchmark=benchmark,
            stem=stem,
            title=title,
        )
        for stem, (benchmark, title) in QUANTIZATION_BENCHMARKS.items()
    }
    plots.update(
        {
            "quant_tool_trace_parseability": quantization_tool_trace_parseability(
                output, items
            ),
            "constraint_load_score": constraint_load_score(output, items),
            "constraint_content_retention": constraint_content_retention(output, items),
            "context_ttft": context_ttft(output, context_items),
        }
    )
    manifest = {
        "schema_version": 2,
        "sources": {
            "default": str(default_experiment),
            "context": str(context_experiment),
        },
        "family_colors": FAMILY_COLORS,
        "design": {
            "one_metric_per_figure": True,
            "marker_shapes_encode_data": False,
            "quantization": QUANT_ORDER,
        },
        "plots": plots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _constraint_rows(
    item_rows: list[dict[str, Any]], *, metric: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for source in item_rows:
        family = str(source.get("family"))
        if (
            source.get("benchmark") != "constraint_load_curve"
            or family not in FAMILY_COLORS
            or _quant(source.get("quantization")) != "Q4"
        ):
            continue
        count = _constraint_count(_tags(source.get("tags")))
        if count is None:
            continue
        value = (
            _number_or_none(source.get("score"))
            if metric == "score"
            else _number_or_none(_json_object(source.get("evaluation_details")).get(metric))
        )
        if value is not None:
            grouped.setdefault((family, count), []).append(value)
    rows = [
        {
            "family": family,
            "constraint_count": count,
            "mean_score_percent": sum(values) / len(values) * 100,
            "n": len(values),
        }
        for (family, count), values in grouped.items()
    ]
    rows.sort(key=lambda row: (
        FAMILY_ORDER.index(str(row["family"])),
        int(row["constraint_count"]),
    ))
    return rows


def _render_line_chart(
    root: Path,
    stem: str,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_order: list[Any],
    y_key: str,
    y_label: str,
    title: str,
    y_limit: tuple[float, float],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for family in FAMILY_ORDER:
        by_x = {row[x_key]: row for row in rows if row.get("family") == family}
        series = [by_x[value] for value in x_order if value in by_x]
        if not series:
            continue
        axis.plot(
            [row[x_key] for row in series],
            [row[y_key] for row in series],
            color=FAMILY_COLORS[family],
            linewidth=3,
            marker="o",
            markersize=7,
            label=FAMILY_LABELS[family],
        )
    counts = sorted({int(row["n"]) for row in rows if row.get("n") is not None})
    sample_note = (
        f"n={counts[0]} per point" if len(counts) == 1 else
        f"n per point: {counts[0]}–{counts[-1]}"
    ) if counts else ""
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=18)
    if sample_note:
        axis.text(0, 1.01, sample_note, transform=axis.transAxes,
                  color="#4B5563", fontsize=10, va="bottom")
    axis.set_xlabel(_x_label(x_key))
    axis.set_ylabel(y_label)
    axis.set_xticks(x_order)
    axis.set_ylim(*y_limit)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.16))
    return _save_figure(root, stem, figure, plt)


def _save_figure(root: Path, stem: str, figure: Any, plt: Any) -> dict[str, str]:
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"{stem}.png"
    svg = plot_dir / f"{stem}.svg"
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    return {
        "png": str(png.relative_to(root)),
        "svg": str(svg.relative_to(root)),
        "data": f"data/{stem}.csv",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _quant(value: Any) -> str:
    normalized = str(value).upper()
    return next((quant for quant in QUANT_ORDER if normalized.startswith(quant)), normalized)


def _constraint_count(tags: list[str]) -> int | None:
    for tag in tags:
        match = re.fullmatch(r"constraint_load_([1-4])", tag)
        if match:
            return int(match.group(1))
    return None


def _context_length(tags: list[str]) -> str | None:
    for label, tag in (("2K", "2k_context"), ("4K", "4k_context"),
                       ("8K", "8k_context")):
        if tag in tags:
            return label
    return None


def _x_label(key: str) -> str:
    return {
        "quantization": "Quantization",
        "constraint_count": "Number of cumulative constraints",
        "context_length": "Context length",
    }[key]


def _generated(paths: dict[str, str], row_count: int, **extra: Any) -> dict[str, Any]:
    return {"status": "generated", **paths, "row_count": row_count, **extra}


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, **extra}
