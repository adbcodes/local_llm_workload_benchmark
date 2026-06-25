from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any


PLOT_ID = "quantization_survival"
PLOT_STEM = "quantization-survival"
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

    data_path = artifact_root / "data" / "plots" / f"{PLOT_STEM}.csv"
    png_path = artifact_root / "plots" / f"{PLOT_STEM}.png"
    svg_path = artifact_root / "plots" / f"{PLOT_STEM}.svg"
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
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PLOT_DATA_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in PLOT_DATA_FIELDS}
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
    axis.set_ylim(0, 100)
    axis.legend(title="Model", frameon=False)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
