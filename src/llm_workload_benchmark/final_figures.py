from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shutil
from statistics import median
from typing import Any


FAMILY_ORDER = ["llama3.1", "qwen3", "mistral", "granite", "qwen2.5-coder"]
FAMILY_LABELS = {
    "llama3.1": "Llama 3.1 8B",
    "qwen3": "Qwen3 8B",
    "mistral": "Mistral 7B v0.3",
    "granite": "Granite 3.3 8B",
    "qwen2.5-coder": "Qwen2.5-Coder 7B",
}
FAMILY_COLORS = {
    "llama3.1": "#E69F00",
    "qwen3": "#009E73",
    "mistral": "#CC79A7",
    "granite": "#D55E00",
    "qwen2.5-coder": "#0072B2",
}
QUANT_ORDER = ["Q8", "Q6", "Q4", "Q3"]
DISPLAY_QUANT_ORDER = ["Q3", "Q4", "Q6", "Q8"]
STRICT_COLOR = "#243447"
SEMANTIC_COLOR = "#3A8D77"
GRID_COLOR = "#D9DEE3"
FIGURE_FACE = "#FAFAF8"


def generate_final_figure_bundle(
    workload_experiment: Path,
    retrieval_experiment: Path,
    grounded_experiment: Path,
    output_directory: Path,
) -> Path:
    """Generate the final eleven-figure bundle from completed experiments."""

    from llm_workload_benchmark.artifacts import export_experiment_artifacts

    for experiment in dict.fromkeys(
        (workload_experiment, retrieval_experiment, grounded_experiment)
    ):
        export_experiment_artifacts(experiment)

    workload_root = workload_experiment / "artifacts"
    output = output_directory
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    workload_items = _labeled_items(workload_experiment, source_name="workloads")
    retrieval_items = _labeled_items(retrieval_experiment, source_name="retrieval")
    grounded_items = _labeled_items(
        grounded_experiment, source_name="grounded_compression"
    )
    quality_items = workload_items + retrieval_items
    configurations = _read_csv(workload_root / "data" / "configurations.csv")
    data_exports = _export_final_item_data(output, quality_items, grounded_items)

    plots = {
        "quality_memory_pareto": quality_memory_pareto(
            output, quality_items, configurations
        ),
        "quantization_quality": quantization_quality(
            output, quality_items, configurations
        ),
        "quantization_memory": quantization_memory(
            output, quality_items, configurations
        ),
        "quantization_decode_speed": quantization_decode_speed(
            output, quality_items, configurations
        ),
        "workload_fit_heatmap": workload_fit_heatmap(output, quality_items),
        "format_tax": format_tax_figure(output, quality_items),
        "constraint_load_overall": constraint_load_overall(
            output, workload_items
        ),
        "constraint_load_by_family": constraint_load_by_family(
            output, workload_items
        ),
        "retrieval_ttft_scaling": retrieval_ttft_scaling(
            output, retrieval_items
        ),
        "grounded_compression_success": grounded_compression_success(
            output, grounded_items
        ),
        "grounded_compression_critical_errors": grounded_compression_critical_errors(
            output, grounded_items
        ),
    }
    manifest = {
        "schema_version": 4,
        "sources": {
            "workloads": _source_reference(workload_experiment, output),
            "retrieval": _source_reference(retrieval_experiment, output),
            "grounded_compression": _source_reference(
                grounded_experiment, output
            ),
        },
        "family_colors": FAMILY_COLORS,
        "design": {
            "adjudication_aware": True,
            "confidence_interval": "Wilson 95% for proportions",
            "family_color_palette": "Okabe-Ito derived",
            "quantization": DISPLAY_QUANT_ORDER,
            "visible_labels": ["Strict pass", "Semantic pass", "Format tax"],
        },
        "data_exports": data_exports,
        "plots": plots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _source_reference(experiment: Path, output: Path) -> str:
    try:
        return str(experiment.resolve().relative_to(output.resolve().parent))
    except ValueError:
        return experiment.name


def merge_adjudication_labels(
    item_rows: list[dict[str, Any]],
    adjudication_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach final strict, semantic, and format-tax labels to artifact rows."""

    sidecars = {
        (
            str(row.get("model_id")),
            str(row.get("item_id")),
            _integer(row.get("repetition"), default=1),
        ): row
        for row in adjudication_rows
    }
    merged: list[dict[str, Any]] = []
    for source in item_rows:
        row = dict(source)
        original_pass = _bool(row.get("passed"))
        semantic_outcome = str(row.get("semantic_outcome", "")).lower()
        semantic_pass = semantic_outcome == "correct" if semantic_outcome else original_pass
        key = (
            str(row.get("variant_id") or row.get("model_id")),
            str(row.get("item_id")),
            _integer(row.get("repetition"), default=1),
        )
        sidecar = sidecars.get(key)
        derived = _json_object((sidecar or {}).get("derived"))
        if derived:
            strict_pass = _bool(derived.get("strict_pass"))
            semantic_pass = _bool(
                derived.get("semantic_correct", derived.get("loose_pass"))
            )
            format_tax = _bool(derived.get("format_tax"))
            label_source = "adjudication"
            adjudication_status = str((sidecar or {}).get("status", ""))
            judge_route = str((sidecar or {}).get("judge_route", ""))
            route_reason = str((sidecar or {}).get("route_reason", ""))
            adjudication_error = str((sidecar or {}).get("error") or "")
        else:
            strict_pass = original_pass
            format_tax = semantic_pass and not strict_pass
            label_source = "deterministic_or_primary_judge"
            adjudication_status = "not_routed"
            judge_route = ""
            route_reason = ""
            adjudication_error = ""
        row.update(
            {
                "strict_pass": strict_pass,
                "semantic_pass": semantic_pass,
                "format_tax": format_tax,
                "label_source": label_source,
                "adjudication_status": adjudication_status,
                "judge_route": judge_route,
                "route_reason": route_reason,
                "adjudication_error": adjudication_error,
                "evaluation_details": _json_object(row.get("evaluation_details")),
            }
        )
        merged.append(row)
    return merged


def _labeled_items(
    experiment: Path, *, source_name: str
) -> list[dict[str, Any]]:
    items = _read_csv(experiment / "artifacts" / "data" / "items.csv")
    merged = merge_adjudication_labels(items, _adjudication_rows(experiment))
    for row in merged:
        row["source_matrix"] = source_name
    return merged


def _adjudication_rows(experiment: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, Path]] = []
    for manifest_path in (experiment / "adjudications").glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("run_status") == "completed":
            results_path = manifest_path.with_name("results.jsonl")
            if results_path.is_file():
                candidates.append((manifest_path, results_path))
    if not candidates:
        return []
    _, results_path = max(candidates, key=lambda pair: pair[0].stat().st_mtime_ns)
    with results_path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _export_final_item_data(
    output: Path,
    quality_items: list[dict[str, Any]],
    grounded_items: list[dict[str, Any]],
) -> dict[str, Any]:
    adjudicated_rows = []
    local_rows = []
    for row in quality_items:
        identity = {
            "source_matrix": row.get("source_matrix"),
            "variant_id": row.get("variant_id"),
            "family": row.get("family"),
            "quantization": row.get("quantization"),
            "benchmark": row.get("benchmark"),
            "item_id": row.get("item_id"),
            "repetition": row.get("repetition"),
            "difficulty": row.get("difficulty"),
            "subcategory": row.get("subcategory"),
            "status": row.get("status"),
        }
        local = {
            **identity,
            "local_pass": _bool(row.get("passed")),
            "local_score": row.get("score"),
            "local_semantic_outcome": row.get("semantic_outcome"),
            "local_semantic_score": row.get("semantic_score"),
            "local_protocol_outcome": row.get("protocol_outcome"),
            "local_protocol_score": row.get("protocol_score"),
            "local_integration_outcome": row.get("integration_outcome"),
        }
        local_rows.append(local)
        adjudicated_rows.append(
            {
                **local,
                "adjudication_status": row.get("adjudication_status"),
                "judge_route": row.get("judge_route"),
                "route_reason": row.get("route_reason"),
                "adjudication_error": row.get("adjudication_error"),
                "label_source": row.get("label_source"),
                "strict_pass": _bool(row.get("strict_pass")),
                "semantic_pass": _bool(row.get("semantic_pass")),
                "format_tax": _bool(row.get("format_tax")),
            }
        )

    grounded_rows = []
    for row in grounded_items:
        details = _json_object(row.get("evaluation_details"))
        checks = _json_object(details.get("deterministic_checks"))
        rubric = _json_object(details.get("rubric"))
        criteria = _json_object(rubric.get("criterion_scores"))
        grounded_rows.append(
            {
                "source_matrix": row.get("source_matrix"),
                "variant_id": row.get("variant_id"),
                "family": row.get("family"),
                "quantization": row.get("quantization"),
                "benchmark": row.get("benchmark"),
                "item_id": row.get("item_id"),
                "repetition": row.get("repetition"),
                "difficulty": row.get("difficulty"),
                "subcategory": row.get("subcategory"),
                "status": row.get("status"),
                "judge_pass": _bool(row.get("strict_pass")),
                "judge_score": row.get("score"),
                "semantic_outcome": row.get("semantic_outcome"),
                "within_word_limit": _bool(checks.get("within_word_limit")),
                "critical_error": _bool(details.get("critical_error")),
                "coverage_score": criteria.get("coverage"),
                "faithfulness_score": criteria.get("faithfulness"),
                "concision_score": criteria.get("concision"),
                "clarity_score": criteria.get("clarity"),
                "relevance_score": criteria.get("relevance"),
                "missing_required_fact_count": len(
                    details.get("missing_required_facts") or []
                ),
                "unsupported_claim_count": len(details.get("unsupported_claims") or []),
            }
        )

    exports = (
        (
            "adjudicated_items.csv",
            adjudicated_rows,
            "Canonical final main and retrieval labels after adjudication",
        ),
        (
            "local_evaluation_items.csv",
            local_rows,
            "Original local evaluator outcomes before external adjudication",
        ),
        (
            "grounded_judged_items.csv",
            grounded_rows,
            "Grounded compression direct pointwise-judge outcomes",
        ),
    )
    manifest: dict[str, Any] = {}
    for filename, rows, description in exports:
        _write_csv(output / "data" / filename, rows)
        manifest[filename.removesuffix(".csv")] = {
            "path": f"data/{filename}",
            "rows": len(rows),
            "description": description,
        }
    return manifest


def quality_memory_pareto(
    root: Path,
    quality_items: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_items:
        grouped[str(row.get("variant_id"))].append(row)
    runtime = {str(row.get("variant_id")): row for row in configurations}
    rows: list[dict[str, Any]] = []
    for variant_id, values in grouped.items():
        config = runtime.get(variant_id)
        if config is None:
            continue
        family = str(config.get("family"))
        if family not in FAMILY_ORDER:
            continue
        memory = _number_or_none(config.get("peak_process_rss_bytes"))
        decode = _number_or_none(config.get("decode_tokens_per_second"))
        elapsed = _number_or_none(config.get("run_elapsed_seconds"))
        if memory is None or decode is None:
            continue
        successes = sum(_bool(value.get("strict_pass")) for value in values)
        semantic_successes = sum(_bool(value.get("semantic_pass")) for value in values)
        low, high = _wilson_interval(successes, len(values))
        rows.append(
            {
                "variant_id": variant_id,
                "family": family,
                "quantization": _quant(config.get("quantization")),
                "strict_pass_percent": successes / len(values) * 100,
                "semantic_pass_percent": semantic_successes / len(values) * 100,
                "format_tax_points": (semantic_successes - successes) / len(values) * 100,
                "ci_95_low": low,
                "ci_95_high": high,
                "peak_memory_gib": memory / 2**30,
                "decode_tokens_per_second": decode,
                "main_run_minutes": elapsed / 60 if elapsed is not None else None,
                "n": len(values),
            }
        )
    rows = [row for row in rows if row["quantization"] in {"Q4", "Q8"}]
    if not rows:
        return _skipped("requires labeled quality rows and configuration telemetry")
    rows.sort(key=lambda row: (FAMILY_ORDER.index(str(row["family"])), str(row["quantization"])))
    stem = "quality_memory_pareto"
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
    _style_axis(axis)
    lookup = {(str(row["family"]), str(row["quantization"])): row for row in rows}
    for family in FAMILY_ORDER:
        q4 = lookup.get((family, "Q4"))
        q8 = lookup.get((family, "Q8"))
        if q4 is None or q8 is None:
            continue
        color = FAMILY_COLORS[family]
        axis.plot(
            [float(q4["peak_memory_gib"]), float(q8["peak_memory_gib"])],
            [float(q4["strict_pass_percent"]), float(q8["strict_pass_percent"])],
            color=color,
            linewidth=2,
            alpha=0.65,
            zorder=1,
        )
        axis.scatter(
            float(q4["peak_memory_gib"]),
            float(q4["strict_pass_percent"]),
            color=color,
            marker="o",
            s=100,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.scatter(
            float(q8["peak_memory_gib"]),
            float(q8["strict_pass_percent"]),
            facecolor=FIGURE_FACE,
            edgecolor=color,
            marker="D",
            s=92,
            linewidth=2.2,
            zorder=3,
        )
    _title(
        axis,
        "Quality versus memory: Q4 and Q8",
        "Adjudicated Strict pass • 360 questions per configuration",
    )
    axis.set_xlabel("Peak process memory (GiB)")
    axis.set_ylabel("Strict pass rate (%)")
    axis.set_xlim(4.1, 11.0)
    axis.set_ylim(8, 59)
    from matplotlib.lines import Line2D
    family_handles = [Line2D([0], [0], color=FAMILY_COLORS[family], linewidth=3, label=FAMILY_LABELS[family]) for family in FAMILY_ORDER]
    family_legend = axis.legend(handles=family_handles, frameon=False, loc="lower right", title="Model family", fontsize=8.5)
    axis.add_artist(family_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E7781", markersize=8, label="Q4"),
            Line2D([0], [0], marker="D", color="#6E7781", markerfacecolor=FIGURE_FACE, markersize=7, label="Q8"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.43, -0.10),
        title="Quantization",
        ncol=2,
    )
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows), n_per_configuration=360)


def _quantization_tradeoff_rows(
    quality_items: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in quality_items:
        family = str(item.get("family"))
        if family in FAMILY_ORDER:
            grouped[(family, _quant(item.get("quantization")))].append(item)
    runtime = {
        (str(row.get("family")), _quant(row.get("quantization"))): row
        for row in configurations
    }
    rows: list[dict[str, Any]] = []
    for key, values in grouped.items():
        config = runtime.get(key)
        if config is None:
            continue
        successes = sum(_bool(value.get("strict_pass")) for value in values)
        semantic_successes = sum(_bool(value.get("semantic_pass")) for value in values)
        low, high = _wilson_interval(successes, len(values))
        rows.append(
            {
                "family": key[0],
                "quantization": key[1],
                "strict_pass_percent": successes / len(values) * 100,
                "semantic_pass_percent": semantic_successes / len(values) * 100,
                "format_tax_points": (semantic_successes - successes) / len(values) * 100,
                "ci_95_low": low,
                "ci_95_high": high,
                "peak_memory_gib": float(config["peak_process_rss_bytes"]) / 2**30,
                "decode_tokens_per_second": float(config["decode_tokens_per_second"]),
                "n": len(values),
            }
        )
    rows.sort(key=lambda row: (FAMILY_ORDER.index(str(row["family"])), DISPLAY_QUANT_ORDER.index(str(row["quantization"]))))
    return rows


def quantization_quality(
    root: Path,
    quality_items: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    return _quantization_metric_figure(
        root,
        _quantization_tradeoff_rows(quality_items, configurations),
        stem="quantization_quality",
        metric="strict_pass_percent",
        y_label="Strict pass rate (%)",
        title="Quality by quantization",
        subtitle="Adjudicated Strict pass • n=360 per configuration • 95% CI",
        y_limit=(0, 62),
        confidence_intervals=True,
    )


def quantization_memory(
    root: Path,
    quality_items: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    return _quantization_metric_figure(
        root,
        _quantization_tradeoff_rows(quality_items, configurations),
        stem="quantization_memory",
        metric="peak_memory_gib",
        y_label="Peak process memory (GiB)",
        title="Memory by quantization",
        subtitle="Main-matrix telemetry • Q4 highlighted",
        y_limit=(4, 11.3),
    )


def quantization_decode_speed(
    root: Path,
    quality_items: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    return _quantization_metric_figure(
        root,
        _quantization_tradeoff_rows(quality_items, configurations),
        stem="quantization_decode_speed",
        metric="decode_tokens_per_second",
        y_label="Decode speed (tokens/s)",
        title="Decode speed by quantization",
        subtitle="Main-matrix telemetry • Q4 highlighted",
        y_limit=(15, 44),
    )


def _quantization_metric_figure(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    stem: str,
    metric: str,
    y_label: str,
    title: str,
    subtitle: str,
    y_limit: tuple[float, float],
    confidence_intervals: bool = False,
) -> dict[str, Any]:
    if not rows:
        return _skipped("requires quality and runtime rows by quantization")
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10.3, 6.2), layout="constrained")
    _style_axis(axis)
    axis.axvspan(0.72, 1.28, color="#F3E7B3", alpha=0.45, zorder=0)
    for family in FAMILY_ORDER:
        series = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: DISPLAY_QUANT_ORDER.index(str(row["quantization"])),
        )
        axis.plot(
            range(len(series)),
            [float(row[metric]) for row in series],
            color=FAMILY_COLORS[family],
            marker="o",
            markersize=6,
            linewidth=2.5,
            label=FAMILY_LABELS[family],
        )
        if confidence_intervals:
            y = [float(row[metric]) for row in series]
            axis.errorbar(
                range(len(series)),
                y,
                yerr=[
                    [value - float(row["ci_95_low"]) for value, row in zip(y, series, strict=True)],
                    [float(row["ci_95_high"]) - value for value, row in zip(y, series, strict=True)],
                ],
                fmt="none",
                ecolor=FAMILY_COLORS[family],
                alpha=0.32,
                capsize=2,
            )
    axis.set_xticks(range(4), DISPLAY_QUANT_ORDER)
    axis.set_xlabel("Quantization")
    axis.set_ylabel(y_label)
    axis.set_ylim(*y_limit)
    _title(axis, title, subtitle)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def workload_fit_heatmap(
    root: Path, quality_items: list[dict[str, Any]]
) -> dict[str, Any]:
    benchmark_order = [
        "applied_reasoning",
        "code_debug_repair",
        "messy_text_to_schema",
        "constraint_load_curve",
        "tool_use",
        "long_text_retrieval",
        "email_to_action",
    ]
    labels = {
        "applied_reasoning": "Reasoning",
        "code_debug_repair": "Coding",
        "messy_text_to_schema": "Extraction",
        "constraint_load_curve": "Constraints",
        "tool_use": "Tool use",
        "long_text_retrieval": "Retrieval",
        "email_to_action": "Email action",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in quality_items:
        key = (str(row.get("family")), str(row.get("benchmark")))
        if key[0] in FAMILY_ORDER and key[1] in benchmark_order:
            grouped[key].append(row)
    rows = []
    for family in FAMILY_ORDER:
        for benchmark in benchmark_order:
            values = grouped.get((family, benchmark), [])
            if values:
                rows.append(
                    {
                        "family": family,
                        "benchmark": benchmark,
                        "strict_pass_percent": sum(_bool(row.get("strict_pass")) for row in values) / len(values) * 100,
                        "semantic_pass_percent": sum(_bool(row.get("semantic_pass")) for row in values) / len(values) * 100,
                        "format_tax_points": sum(_bool(row.get("format_tax")) for row in values) / len(values) * 100,
                        "n": len(values),
                    }
                )
    if not rows:
        return _skipped("requires labeled workload rows")
    stem = "workload_fit_heatmap"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    lookup = {(row["family"], row["benchmark"]): float(row["strict_pass_percent"]) for row in rows}
    matrix = [[lookup.get((family, benchmark), math.nan) for benchmark in benchmark_order] for family in FAMILY_ORDER]

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(12.7, 5.8), layout="constrained")
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for y, family in enumerate(FAMILY_ORDER):
        for x, benchmark in enumerate(benchmark_order):
            value = matrix[y][x]
            if not math.isnan(value):
                axis.text(x, y, f"{value:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if value >= 52 else "#1F2933")
    axis.set_xticks(range(len(benchmark_order)), [labels[value] for value in benchmark_order], rotation=28, ha="right")
    axis.set_yticks(range(len(FAMILY_ORDER)), [FAMILY_LABELS[value] for value in FAMILY_ORDER])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.025)
    colorbar.set_label("Strict pass rate (%)")
    _title(axis, "Workload fit by model family", "Adjudicated Strict pass • all quantizations pooled")
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def format_tax_figure(
    root: Path, quality_items: list[dict[str, Any]]
) -> dict[str, Any]:
    panels: dict[str, dict[str, list[dict[str, Any]]]] = {
        "workload": defaultdict(list),
        "family": defaultdict(list),
    }
    for row in quality_items:
        panels["workload"][str(row.get("benchmark"))].append(row)
        panels["family"][str(row.get("family"))].append(row)
    rows = []
    for panel, groups in panels.items():
        for group, values in groups.items():
            if not values:
                continue
            strict = sum(_bool(value.get("strict_pass")) for value in values) / len(values) * 100
            semantic = sum(_bool(value.get("semantic_pass")) for value in values) / len(values) * 100
            rows.append(
                {
                    "panel": panel,
                    "group": group,
                    "strict_pass_percent": strict,
                    "semantic_pass_percent": semantic,
                    "format_tax_points": semantic - strict,
                    "n": len(values),
                }
            )
    if not rows:
        return _skipped("requires strict and semantic labels")
    stem = "format_tax"
    _write_csv(root / "data" / f"{stem}.csv", rows)
    friendly = {
        "applied_reasoning": "Reasoning",
        "code_debug_repair": "Coding",
        "constraint_load_curve": "Constraints",
        "email_to_action": "Email action",
        "messy_text_to_schema": "Extraction",
        "tool_use": "Tool use",
        "long_text_retrieval": "Retrieval",
        **FAMILY_LABELS,
    }

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(16.2, 6.0), layout="constrained")
    for axis, panel, title in zip(axes, ("workload", "family"), ("By workload", "By model family"), strict=True):
        _style_axis(axis, x_grid=True, y_grid=False)
        series = sorted((row for row in rows if row["panel"] == panel), key=lambda row: float(row["format_tax_points"]), reverse=True)
        for y, row in enumerate(series):
            strict_value = float(row["strict_pass_percent"])
            semantic_value = float(row["semantic_pass_percent"])
            axis.plot([strict_value, semantic_value], [y, y], color="#C5CBD1", linewidth=3, solid_capstyle="round")
            axis.scatter(strict_value, y, color=STRICT_COLOR, s=62, zorder=3)
            axis.scatter(semantic_value, y, color=SEMANTIC_COLOR, s=62, zorder=3)
            if float(row["format_tax_points"]) >= 1:
                axis.text(semantic_value + 1.2, y, f"+{float(row['format_tax_points']):.1f}", va="center", fontsize=9, color="#59636D")
        axis.set_yticks(range(len(series)), [friendly.get(str(row["group"]), str(row["group"])) for row in series])
        axis.invert_yaxis()
        axis.set_xlim(0, 88 if panel == "workload" else 65)
        axis.set_xlabel("Pass rate (%)")
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    _figure_title(figure, "Format tax", "Gap between Semantic pass and Strict pass")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(wspace=0.18)
    from matplotlib.lines import Line2D
    figure.legend(
        [Line2D([0], [0], marker="o", color="none", markerfacecolor=STRICT_COLOR, markersize=8), Line2D([0], [0], marker="o", color="none", markerfacecolor=SEMANTIC_COLOR, markersize=8)],
        ["Strict pass", "Semantic pass"],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def _constraint_load_plot_rows(
    workload_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in workload_items:
        if row.get("benchmark") != "constraint_load_curve":
            continue
        count = _constraint_count(_tags(row.get("tags"))) or {
            "one_constraint": 1,
            "two_constraints": 2,
            "three_constraints": 3,
            "four_constraints": 4,
        }.get(str(row.get("subcategory")))
        if count is None:
            continue
        family = str(row.get("family"))
        grouped[(family, count)].append(row)
        overall[count].append(row)
    rows = []
    for count, values in overall.items():
        strict_successes = sum(_bool(row.get("strict_pass")) for row in values)
        semantic_successes = sum(_bool(row.get("semantic_pass")) for row in values)
        strict_low, strict_high = _wilson_interval(strict_successes, len(values))
        semantic_low, semantic_high = _wilson_interval(semantic_successes, len(values))
        rows.append({"series": "overall", "family": "all", "constraint_count": count, "strict_pass_percent": strict_successes / len(values) * 100, "semantic_pass_percent": semantic_successes / len(values) * 100, "strict_ci_low": strict_low, "strict_ci_high": strict_high, "semantic_ci_low": semantic_low, "semantic_ci_high": semantic_high, "n": len(values)})
    for (family, count), values in grouped.items():
        rows.append({"series": "family", "family": family, "constraint_count": count, "strict_pass_percent": sum(_bool(row.get("strict_pass")) for row in values) / len(values) * 100, "semantic_pass_percent": sum(_bool(row.get("semantic_pass")) for row in values) / len(values) * 100, "n": len(values)})
    rows.sort(key=lambda row: (str(row["series"]), str(row["family"]), int(row["constraint_count"])))
    return rows


def constraint_load_overall(
    root: Path, workload_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = _constraint_load_plot_rows(workload_items)
    aggregate = sorted((row for row in rows if row["series"] == "overall"), key=lambda row: int(row["constraint_count"]))
    if not aggregate:
        return _skipped("requires constraint-load labels")
    stem = "constraint_load_overall"
    _write_csv(root / "data" / f"{stem}.csv", aggregate)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.8, 6.1), layout="constrained")
    _style_axis(axis)
    for metric, label, color, low_key, high_key in (
        ("strict_pass_percent", "Strict pass", STRICT_COLOR, "strict_ci_low", "strict_ci_high"),
        ("semantic_pass_percent", "Semantic pass", SEMANTIC_COLOR, "semantic_ci_low", "semantic_ci_high"),
    ):
        y = [float(row[metric]) for row in aggregate]
        axis.errorbar(
            [int(row["constraint_count"]) for row in aggregate],
            y,
            yerr=[
                [value - float(row[low_key]) for value, row in zip(y, aggregate, strict=True)],
                [float(row[high_key]) - value for value, row in zip(y, aggregate, strict=True)],
            ],
            label=label,
            color=color,
            linewidth=3,
            marker="o",
            markersize=7,
            capsize=3,
        )
    _title(axis, "Pass rate by constraint count", "All models and quantizations pooled • n=240 per step • 95% CI")
    axis.set_xticks([1, 2, 3, 4])
    axis.set_xlabel("Number of simultaneous constraints")
    axis.set_ylabel("Pass rate (%)")
    axis.set_ylim(0, 82)
    axis.legend(frameon=False)
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(aggregate))


def constraint_load_by_family(
    root: Path, workload_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = [row for row in _constraint_load_plot_rows(workload_items) if row["series"] == "family"]
    if not rows:
        return _skipped("requires constraint-load labels by family")
    stem = "constraint_load_by_family"
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.8, 6.1), layout="constrained")
    _style_axis(axis)
    for family in FAMILY_ORDER:
        series = sorted((row for row in rows if row["series"] == "family" and row["family"] == family), key=lambda row: int(row["constraint_count"]))
        axis.plot([int(row["constraint_count"]) for row in series], [float(row["strict_pass_percent"]) for row in series], color=FAMILY_COLORS[family], linewidth=2.5, marker="o", markersize=6, label=FAMILY_LABELS[family])
    _title(axis, "Constraint following by model family", "Adjudicated Strict pass • all quantizations pooled")
    axis.set_xticks([1, 2, 3, 4])
    axis.set_xlabel("Number of simultaneous constraints")
    axis.set_ylabel("Strict pass rate (%)")
    axis.set_ylim(0, 82)
    axis.legend(frameon=False, fontsize=9)
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def retrieval_ttft_scaling(
    root: Path, retrieval_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in retrieval_items:
        if _quant(row.get("quantization")) != "Q4":
            continue
        prompt_tokens = _number_or_none(row.get("prompt_tokens"))
        ttft = _number_or_none(row.get("ttft_seconds"))
        family = str(row.get("family"))
        if prompt_tokens is None or ttft is None or family not in FAMILY_ORDER:
            continue
        band = "2–3K" if prompt_tokens < 3500 else "4–6K" if prompt_tokens < 6500 else "8–10K"
        grouped[(family, band)].append((prompt_tokens, ttft))
        rows.append({"row_type": "item", "family": family, "band": band, "prompt_tokens": prompt_tokens, "ttft_seconds": ttft, "n": 1})
    band_order = ["2–3K", "4–6K", "8–10K"]
    summaries = []
    for (family, band), values in grouped.items():
        summary = {"row_type": "median", "family": family, "band": band, "prompt_tokens": median(value[0] for value in values), "ttft_seconds": median(value[1] for value in values), "n": len(values)}
        summaries.append(summary)
        rows.append(summary)
    if not summaries:
        return _skipped("requires Q4 retrieval prompt-token and TTFT measurements")
    stem = "retrieval_ttft_scaling"
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10.8, 6.3), layout="constrained")
    _style_axis(axis)
    for family in FAMILY_ORDER:
        raw = [row for row in rows if row["row_type"] == "item" and row["family"] == family]
        series = sorted((row for row in summaries if row["family"] == family), key=lambda row: band_order.index(str(row["band"])))
        axis.scatter([float(row["prompt_tokens"]) / 1000 for row in raw], [float(row["ttft_seconds"]) for row in raw], s=14, alpha=0.10, color=FAMILY_COLORS[family], edgecolors="none")
        axis.plot([float(row["prompt_tokens"]) / 1000 for row in series], [float(row["ttft_seconds"]) for row in series], color=FAMILY_COLORS[family], linewidth=2.7, marker="o", markersize=6, label=FAMILY_LABELS[family])
    _title(axis, "Time to first token by prompt length", "Q4 • dots are items; lines are length-band medians")
    axis.set_xlabel("Prompt length (thousand tokens)")
    axis.set_ylabel("Time to first token (seconds)")
    axis.set_xlim(1.7, 10.5)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows), quantization="Q4")


def _grounded_compression_rows(
    grounded_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grounded_items:
        family = str(row.get("family"))
        if family in FAMILY_ORDER:
            grouped[family].append(row)
    rows = []
    for family, values in grouped.items():
        within = []
        critical = []
        for value in values:
            details = _json_object(value.get("evaluation_details"))
            checks = _json_object(details.get("deterministic_checks"))
            if checks.get("within_word_limit") is not None:
                within.append(_bool(checks.get("within_word_limit")))
            if details.get("critical_error") is not None:
                critical.append(_bool(details.get("critical_error")))
        passes = sum(_bool(value.get("strict_pass")) for value in values)
        pass_low, pass_high = _wilson_interval(passes, len(values))
        within_successes = sum(within)
        within_low, within_high = _wilson_interval(within_successes, len(within))
        critical_successes = sum(critical)
        critical_low, critical_high = _wilson_interval(critical_successes, len(critical))
        rows.append({"family": family, "pass_percent": passes / len(values) * 100, "pass_ci_low": pass_low, "pass_ci_high": pass_high, "within_word_limit_percent": within_successes / len(within) * 100, "within_ci_low": within_low, "within_ci_high": within_high, "critical_error_percent": critical_successes / len(critical) * 100, "critical_ci_low": critical_low, "critical_ci_high": critical_high, "n": len(values)})
    rows.sort(key=lambda row: float(row["pass_percent"]), reverse=True)
    return rows


def grounded_compression_success(
    root: Path, grounded_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = _grounded_compression_rows(grounded_items)
    if not rows:
        return _skipped("requires completed grounded-compression judgments")
    stem = "grounded_compression_success"
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10.2, 6.1), layout="constrained")
    _style_axis(axis, x_grid=True, y_grid=False)
    y = list(range(len(rows)))
    labels = [FAMILY_LABELS[str(row["family"])] for row in rows]
    offsets = (-0.12, 0.12)
    for offset, metric, low_key, high_key, label, color in (
        (offsets[0], "pass_percent", "pass_ci_low", "pass_ci_high", "Passed", STRICT_COLOR),
        (offsets[1], "within_word_limit_percent", "within_ci_low", "within_ci_high", "Within word limit", SEMANTIC_COLOR),
    ):
        values = [float(row[metric]) for row in rows]
        axis.errorbar(values, [value + offset for value in y], xerr=[
            [value - float(row[low_key]) for value, row in zip(values, rows, strict=True)],
            [float(row[high_key]) - value for value, row in zip(values, rows, strict=True)],
        ], fmt="o", color=color, markersize=7, capsize=2.5, label=label)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Responses (%)")
    axis.legend(frameon=False, loc="lower right")
    _title(axis, "Grounded-compression success", "Final judge results • all quantizations pooled • n=120 per family • 95% CI")
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def grounded_compression_critical_errors(
    root: Path, grounded_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = _grounded_compression_rows(grounded_items)
    if not rows:
        return _skipped("requires completed grounded-compression judgments")
    stem = "grounded_compression_critical_errors"
    _write_csv(root / "data" / f"{stem}.csv", rows)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.6, 6.1), layout="constrained")
    _style_axis(axis, x_grid=True, y_grid=False)
    y = list(range(len(rows)))
    labels = [FAMILY_LABELS[str(row["family"])] for row in rows]
    critical_values = [float(row["critical_error_percent"]) for row in rows]
    axis.errorbar(critical_values, y, xerr=[
        [value - float(row["critical_ci_low"]) for value, row in zip(critical_values, rows, strict=True)],
        [float(row["critical_ci_high"]) - value for value, row in zip(critical_values, rows, strict=True)],
    ], fmt="o", color="#B24C63", markersize=7, capsize=2.5)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 55)
    axis.set_xlabel("Responses with a critical error (%)")
    _title(axis, "Critical grounding errors by model family", "Final judge results • all quantizations pooled • n=120 per family • 95% CI")
    paths = _save_figure(root, stem, figure, plt)
    return _generated(paths, len(rows))


def _save_figure(root: Path, stem: str, figure: Any, plt: Any) -> dict[str, str]:
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"{stem}.png"
    svg = plot_dir / f"{stem}.svg"
    figure.savefig(png, dpi=200, facecolor=FIGURE_FACE)
    figure.savefig(svg, facecolor=FIGURE_FACE)
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


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlecolor": "#17212B",
            "axes.labelcolor": "#303942",
            "xtick.color": "#4C5661",
            "ytick.color": "#4C5661",
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": FIGURE_FACE,
            "savefig.facecolor": FIGURE_FACE,
        }
    )
    return plt


def _style_axis(axis: Any, *, x_grid: bool = False, y_grid: bool = True) -> None:
    axis.set_axisbelow(True)
    axis.grid(False)
    if x_grid:
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    if y_grid:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B7BEC5")
    axis.spines["bottom"].set_color("#B7BEC5")


def _title(axis: Any, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=24)
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5B6570",
    )


def _figure_title(figure: Any, title: str, subtitle: str) -> None:
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0, 0, 1, 0.86))
    figure.suptitle(
        title,
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#17212B",
    )
    figure.text(
        0.01,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color="#5B6570",
    )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _integer(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - half_width) * 100, min(1.0, center + half_width) * 100


def _generated(paths: dict[str, str], row_count: int, **extra: Any) -> dict[str, Any]:
    return {"status": "generated", **paths, "row_count": row_count, **extra}


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, **extra}
