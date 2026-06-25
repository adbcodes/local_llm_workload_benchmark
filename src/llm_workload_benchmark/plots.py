from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any


QUANTIZATION_PLOT_ID = "quantization_survival"
QUANTIZATION_PLOT_STEM = "quantization-survival"
MEMORY_PLOT_ID = "memory_quality_frontier"
MEMORY_PLOT_STEM = "memory-quality-frontier"
SPEED_PLOT_ID = "speed_quality_frontier"
SPEED_PLOT_STEM = "speed-quality-frontier"
HEATMAP_PLOT_ID = "workload_fit_heatmap"
HEATMAP_PLOT_STEM = "workload-fit-heatmap"
SETTING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "max_output_tokens",
    "constrained_decoding",
    "context_window",
    "threads",
    "batch_size",
    "gpu_layers",
    "flash_attention",
    "kv_cache_type",
)
PLOT_DATA_FIELDS = [
    "architecture",
    "family",
    "variant_id",
    "quantization",
    "quantization_label",
    "quantization_bits",
    "mean_score",
    "score_percent",
    *SETTING_FIELDS,
]
CANONICAL_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "constrained_decoding": "none",
}
FRONTIER_DATA_FIELDS = [
    "variant_id",
    "architecture",
    "family",
    "quantization",
    "quantization_label",
    "mean_score",
    "score_percent",
    "peak_process_memory_bytes",
    "peak_process_memory_gib",
    "mean_output_tokens_per_second",
    "is_pareto",
]
QUANTIZATION_MARKERS = {8: "o", 6: "s", 4: "D", 3: "^"}
HEATMAP_DATA_FIELDS = [
    "variant_id",
    "architecture",
    "family",
    "quantization",
    "quantization_label",
    "suite",
    "mean_score",
    "score_percent",
    "available",
]


def generate_plots(
    artifact_root: Path,
    configuration_rows: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Generate every plot supported by this artifact bundle."""

    return {
        QUANTIZATION_PLOT_ID: generate_quantization_survival(
            artifact_root, configuration_rows
        ),
        MEMORY_PLOT_ID: _generate_frontier(
            artifact_root,
            configuration_rows,
            stem=MEMORY_PLOT_STEM,
            x_field="peak_process_memory_gib",
            source_x_field="peak_process_memory_bytes",
            x_label="Peak process memory (GiB)",
            title="Memory–Quality Frontier",
            minimize_x=True,
        ),
        SPEED_PLOT_ID: _generate_frontier(
            artifact_root,
            configuration_rows,
            stem=SPEED_PLOT_STEM,
            x_field="mean_output_tokens_per_second",
            source_x_field="mean_output_tokens_per_second",
            x_label="End-to-end output tokens per second",
            title="Speed–Quality Frontier",
            minimize_x=False,
        ),
        HEATMAP_PLOT_ID: _generate_workload_heatmap(artifact_root, suite_rows),
    }


def generate_quantization_survival(
    artifact_root: Path,
    configuration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate the quantization survival plot or explain why it is unavailable."""

    plot_rows = _select_plot_rows(configuration_rows)
    architectures = {row["architecture"] for row in plot_rows}
    if not plot_rows:
        return {
            "status": "skipped",
            "reason": (
                "requires at least one explicit model architecture with completed "
                "scores for two or more quantizations"
            ),
        }

    data_path = artifact_root / "data" / "plots" / f"{QUANTIZATION_PLOT_STEM}.csv"
    png_path = artifact_root / "plots" / f"{QUANTIZATION_PLOT_STEM}.png"
    svg_path = artifact_root / "plots" / f"{QUANTIZATION_PLOT_STEM}.svg"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _write_plot_data(data_path, plot_rows)
    _render_plot(plot_rows, png_path, svg_path)
    return {
        "status": "generated",
        "png": str(png_path.relative_to(artifact_root)),
        "svg": str(svg_path.relative_to(artifact_root)),
        "data": str(data_path.relative_to(artifact_root)),
        "row_count": len(plot_rows),
        "series_count": len(architectures),
        "x": "quantization_bits",
        "y": "mean_score",
        "series": "architecture",
    }


def _generate_frontier(
    artifact_root: Path,
    configuration_rows: list[dict[str, Any]],
    *,
    stem: str,
    x_field: str,
    source_x_field: str,
    x_label: str,
    title: str,
    minimize_x: bool,
) -> dict[str, Any]:
    rows = _frontier_rows(configuration_rows)
    rows = [row for row in rows if isinstance(row.get(source_x_field), int | float)]
    if len(rows) < 2:
        return {
            "status": "skipped",
            "reason": (
                f"requires at least two completed configurations with mean_score "
                f"and {source_x_field}"
            ),
        }

    if source_x_field == "peak_process_memory_bytes":
        for row in rows:
            row[x_field] = row[source_x_field] / (1024**3)
    pareto_ids = {
        row["variant_id"]
        for row in _pareto_frontier(
            rows,
            x_field=x_field,
            minimize_x=minimize_x,
        )
    }
    for row in rows:
        row["is_pareto"] = row["variant_id"] in pareto_ids
    rows.sort(key=lambda row: (float(row[x_field]), str(row["variant_id"])))

    data_path = artifact_root / "data" / "plots" / f"{stem}.csv"
    png_path = artifact_root / "plots" / f"{stem}.png"
    svg_path = artifact_root / "plots" / f"{stem}.svg"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(data_path, FRONTIER_DATA_FIELDS, rows)
    _render_frontier(
        rows,
        png_path,
        svg_path,
        x_field=x_field,
        x_label=x_label,
        title=title,
    )
    return {
        "status": "generated",
        "png": str(png_path.relative_to(artifact_root)),
        "svg": str(svg_path.relative_to(artifact_root)),
        "data": str(data_path.relative_to(artifact_root)),
        "row_count": len(rows),
        "pareto_count": len(pareto_ids),
        "x": x_field,
        "y": "mean_score",
        "point": "variant_id",
    }


def _frontier_rows(
    configuration_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in configuration_rows:
        score = row.get("mean_score")
        if row.get("status") != "completed" or not isinstance(score, int | float):
            continue
        quantization = _quantization(row.get("quantization")) or {
            "quantization_label": row.get("quantization") or "unknown",
            "quantization_bits": None,
        }
        rows.append({**row, **quantization, "score_percent": float(score) * 100})
    return rows


def _pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    minimize_x: bool,
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        candidate_x = float(candidate[x_field])
        candidate_y = float(candidate["score_percent"])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_x = float(other[x_field])
            other_y = float(other["score_percent"])
            x_is_better = (
                other_x <= candidate_x if minimize_x else other_x >= candidate_x
            )
            x_is_strict = (
                other_x < candidate_x if minimize_x else other_x > candidate_x
            )
            if (
                x_is_better
                and other_y >= candidate_y
                and (x_is_strict or other_y > candidate_y)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _generate_workload_heatmap(
    artifact_root: Path,
    suite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    configurations: dict[str, dict[str, Any]] = {}
    suites: set[str] = set()
    for row in suite_rows:
        variant_id = row.get("variant_id")
        suite = row.get("suite")
        score = row.get("mean_score")
        if (
            not isinstance(variant_id, str)
            or not isinstance(suite, str)
            or not isinstance(score, int | float)
        ):
            continue
        key = (variant_id, suite)
        if key in observed:
            raise ValueError(
                f"duplicate workload heatmap score for {variant_id!r} and {suite!r}"
            )
        quantization = _quantization(row.get("quantization")) or {
            "quantization_label": row.get("quantization") or "unknown",
            "quantization_bits": None,
        }
        normalized = {**row, **quantization, "score_percent": float(score) * 100}
        observed[key] = normalized
        configurations.setdefault(variant_id, normalized)
        suites.add(suite)

    if len(configurations) < 2 or not suites:
        return {
            "status": "skipped",
            "reason": (
                "requires suite scores for at least two completed configurations"
            ),
        }

    ordered_variants = sorted(
        configurations,
        key=lambda variant_id: (
            _model_label(configurations[variant_id]),
            -(configurations[variant_id].get("quantization_bits") or -1),
            variant_id,
        ),
    )
    ordered_suites = sorted(suites)
    plot_rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    missing_count = 0
    for variant_id in ordered_variants:
        configuration = configurations[variant_id]
        matrix_row: list[float] = []
        for suite in ordered_suites:
            score_row = observed.get((variant_id, suite))
            available = score_row is not None
            score_percent = score_row["score_percent"] if score_row else None
            if not available:
                missing_count += 1
            matrix_row.append(float(score_percent) if available else float("nan"))
            plot_rows.append(
                {
                    **configuration,
                    "suite": suite,
                    "mean_score": score_row.get("mean_score") if score_row else None,
                    "score_percent": score_percent,
                    "available": available,
                }
            )
        matrix.append(matrix_row)

    data_path = artifact_root / "data" / "plots" / f"{HEATMAP_PLOT_STEM}.csv"
    png_path = artifact_root / "plots" / f"{HEATMAP_PLOT_STEM}.png"
    svg_path = artifact_root / "plots" / f"{HEATMAP_PLOT_STEM}.svg"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(data_path, HEATMAP_DATA_FIELDS, plot_rows)
    _render_workload_heatmap(
        matrix,
        [configurations[variant_id] for variant_id in ordered_variants],
        ordered_suites,
        png_path,
        svg_path,
    )
    return {
        "status": "generated",
        "png": str(png_path.relative_to(artifact_root)),
        "svg": str(svg_path.relative_to(artifact_root)),
        "data": str(data_path.relative_to(artifact_root)),
        "row_count": len(plot_rows),
        "configuration_count": len(ordered_variants),
        "suite_count": len(ordered_suites),
        "observed_count": len(plot_rows) - missing_count,
        "missing_count": missing_count,
        "rows": "variant_id",
        "columns": "suite",
        "value": "mean_score",
    }


def _select_plot_rows(
    configuration_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {}
    for row in configuration_rows:
        architecture = row.get("architecture")
        quantization = _quantization(row.get("quantization"))
        score = row.get("mean_score")
        if (
            row.get("status") != "completed"
            or not isinstance(architecture, str)
            or not architecture.strip()
            or quantization is None
            or not isinstance(score, int | float)
        ):
            continue
        setting_key = tuple(row.get(field) for field in SETTING_FIELDS)
        candidates.setdefault(architecture, {}).setdefault(setting_key, []).append(
            {**row, **quantization, "score_percent": float(score) * 100}
        )

    selected: list[dict[str, Any]] = []
    for architecture in sorted(candidates):
        eligible = [
            rows
            for rows in candidates[architecture].values()
            if len({row["quantization_bits"] for row in rows}) >= 2
        ]
        if not eligible:
            continue
        chosen = min(
            eligible,
            key=lambda rows: (
                -_canonical_match_count(rows[0]),
                -len({row["quantization_bits"] for row in rows}),
                tuple(str(rows[0].get(field)) for field in SETTING_FIELDS),
            ),
        )
        selected.extend(
            sorted(chosen, key=lambda row: row["quantization_bits"], reverse=True)
        )
    return selected


def _canonical_match_count(row: dict[str, Any]) -> int:
    return sum(row.get(field) == value for field, value in CANONICAL_SETTINGS.items())


def _quantization(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^Q(\d+)", value.upper())
    if match is None:
        return None
    bits = int(match.group(1))
    return {"quantization_label": f"Q{bits}", "quantization_bits": bits}


def _write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows(path, PLOT_DATA_FIELDS, rows)


def _write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fieldnames}
            for row in rows
        )


def _render_plot(rows: list[dict[str, Any]], png_path: Path, svg_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    architectures = sorted({row["architecture"] for row in rows})
    colours = plt.get_cmap("tab10").colors
    all_bits = sorted({row["quantization_bits"] for row in rows}, reverse=True)
    positions = {bits: index for index, bits in enumerate(all_bits)}

    for index, architecture in enumerate(architectures):
        series = [row for row in rows if row["architecture"] == architecture]
        axis.plot(
            [positions[row["quantization_bits"]] for row in series],
            [row["score_percent"] for row in series],
            color=colours[index % len(colours)],
            marker="o",
            linewidth=2.25,
            markersize=7,
            label=architecture,
        )
    axis.set_title("Quantization Survival", fontsize=16, fontweight="bold", loc="left")
    axis.set_xlabel("Quantization")
    axis.set_ylabel("Mean workload score (%)")
    axis.set_xticks(range(len(all_bits)), [f"Q{bits}" for bits in all_bits])
    axis.set_ylim(0, 105)
    axis.legend(title="Model", frameon=False)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)


def _render_frontier(
    rows: list[dict[str, Any]],
    png_path: Path,
    svg_path: Path,
    *,
    x_field: str,
    x_label: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    model_labels = sorted({_model_label(row) for row in rows})
    colours = plt.get_cmap("tab10").colors
    colour_by_model = {
        label: colours[index % len(colours)]
        for index, label in enumerate(model_labels)
    }
    combinations: dict[tuple[str, str, int | None], Any] = {}
    for row in rows:
        model_label = _model_label(row)
        bits = row.get("quantization_bits")
        marker = QUANTIZATION_MARKERS.get(bits, "X")
        is_pareto = row["is_pareto"]
        axis.scatter(
            row[x_field],
            row["score_percent"],
            color=colour_by_model[model_label],
            marker=marker,
            s=95 if is_pareto else 60,
            edgecolor="#111827" if is_pareto else "white",
            linewidth=1.4 if is_pareto else 0.8,
            zorder=3 if is_pareto else 2,
        )
        combinations[(model_label, row["quantization_label"], bits)] = marker

    frontier = sorted(
        (row for row in rows if row["is_pareto"]),
        key=lambda row: float(row[x_field]),
    )
    axis.plot(
        [row[x_field] for row in frontier],
        [row["score_percent"] for row in frontier],
        color="#111827",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    x_values = [float(item[x_field]) for item in rows]
    for row in frontier:
        near_right_edge = float(row[x_field]) >= max(x_values)
        near_top_edge = float(row["score_percent"]) >= 95
        axis.annotate(
            _short_label(str(row["variant_id"])),
            (row[x_field], row["score_percent"]),
            xytext=(-6 if near_right_edge else 6, -14 if near_top_edge else 7),
            textcoords="offset points",
            fontsize=7,
            horizontalalignment="right" if near_right_edge else "left",
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            color="none",
            markerfacecolor=colour_by_model[model_label],
            markeredgecolor="white",
            markersize=7,
            label=f"{model_label} {quantization}",
        )
        for (model_label, quantization, _), marker in sorted(combinations.items())
    ]
    legend_handles.append(
        Line2D([0], [0], color="#111827", linestyle="--", label="Pareto frontier")
    )
    axis.set_title(title, fontsize=16, fontweight="bold", loc="left")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean workload score (%)")
    axis.set_ylim(0, 105)
    axis.margins(x=0.08)
    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)


def _render_workload_heatmap(
    matrix: list[list[float]],
    configurations: list[dict[str, Any]],
    suites: list[str],
    png_path: Path,
    svg_path: Path,
) -> None:
    import math
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    height = max(4.5, 1.8 + len(configurations) * 0.36)
    width = max(8.0, 4.5 + len(suites) * 1.1)
    figure, axis = plt.subplots(figsize=(width, height), layout="constrained")
    colour_map = plt.get_cmap("viridis").with_extremes(bad="#e5e7eb")
    image = axis.imshow(matrix, cmap=colour_map, vmin=0, vmax=100, aspect="auto")

    labels = _heatmap_labels(configurations)
    axis.set_title("Workload Fit", fontsize=16, fontweight="bold", loc="left")
    axis.set_xlabel("Workload suite")
    axis.set_ylabel("Model configuration")
    axis.set_xticks(range(len(suites)), suites, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks([index - 0.5 for index in range(1, len(suites))], minor=True)
    axis.set_yticks(
        [index - 0.5 for index in range(1, len(configurations))],
        minor=True,
    )
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.grid(which="major", visible=False)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            missing = math.isnan(value)
            axis.text(
                column_index,
                row_index,
                "—" if missing else f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "#6b7280"
                    if missing
                    else "#111827"
                    if value >= 65
                    else "white"
                ),
            )
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.82)
    colour_bar.set_label("Mean workload score (%)")
    axis.spines[:].set_visible(False)
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)


def _heatmap_labels(configurations: list[dict[str, Any]]) -> list[str]:
    base_labels = [
        f"{_model_label(row)} · {row.get('quantization_label', 'unknown')}"
        for row in configurations
    ]
    counts = {label: base_labels.count(label) for label in set(base_labels)}
    return [
        (
            f"{label} · {_short_label(str(row.get('variant_id')), maximum=24)}"
            if counts[label] > 1
            else label
        )
        for label, row in zip(base_labels, configurations)
    ]


def _model_label(row: dict[str, Any]) -> str:
    for field in ("architecture", "family", "variant_id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown-model"


def _short_label(value: str, maximum: int = 32) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"
