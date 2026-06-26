from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


FAMILY_COLORS = {
    "qwen2.5": "#0072B2",
    "gemma3": "#E69F00",
    "qwen3": "#6A3D9A",
}
QUANT_MARKERS = {"Q8": "o", "Q6": "s", "Q4": "D", "Q3": "^"}
QUANT_ORDER = ["Q8", "Q6", "Q4", "Q3"]
WORKLOAD_GROUPS = {
    "reasoning": {"applied_reasoning"},
    "coding": {"code_debug_repair"},
    "knowledge": {"knowledge_abstention"},
    "extraction": {"messy_text_to_schema"},
    "decisions / routing": {"tables_to_decisions", "inbox_routing"},
    "tool use": {"tool_use"},
    "instruction control": {
        "constraint_load_curve", "negative_instructions", "instruction_hierarchy",
        "raw_output_discipline",
    },
    "communication": {"grounded_compression", "india_focused_tasks"},
    "reliability / trust": {
        "answer_stability", "clean_vs_noisy", "confidence_correctness",
        "conversation_memory", "false_missing_information", "long_text_retrieval",
        "over_refusal", "prompt_format_sensitivity", "shuffled_choices",
    },
}


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in rows
        if not any(
            other is not candidate
            and _number(other.get("memory_gb")) <= _number(candidate.get("memory_gb"))
            and _number(other.get("pass_rate_percent"))
            >= _number(candidate.get("pass_rate_percent"))
            and (
                _number(other.get("memory_gb")) < _number(candidate.get("memory_gb"))
                or _number(other.get("pass_rate_percent"))
                > _number(candidate.get("pass_rate_percent"))
            )
            for other in rows
        )
    ]


def laptop_value_frontier(
    root: Path, configuration_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    for source in configuration_rows:
        attempted, passed = _integer(source.get("attempted")), _integer(source.get("passed"))
        memory = _number_or_none(source.get("peak_process_memory_bytes"))
        if not attempted or passed is None or memory is None:
            continue
        low, high = wilson_interval(passed, attempted) or (0.0, 0.0)
        rows.append(
            {
                **source,
                "memory_gb": memory / 1_000_000_000,
                "pass_rate_percent": passed / attempted * 100,
                "ci_low_percent": low * 100,
                "ci_high_percent": high * 100,
                "speed_tps": _number_or_none(source.get("mean_output_tokens_per_second")),
                "energy_per_correct_j": _number_or_none(
                    source.get("energy_per_correct_answer_joules")
                ),
                "quant_label": _quant(source.get("quantization")),
            }
        )
    if len(rows) < 2:
        return _skipped("requires at least two completed default configurations"), []
    frontier = pareto_frontier(rows)
    frontier_ids = [str(row["variant_id"]) for row in frontier]
    for row in rows:
        row["is_pareto"] = row["variant_id"] in frontier_ids
    rows.sort(key=lambda row: (row["memory_gb"], str(row["variant_id"])))
    _write_csv(root / "data" / "laptop_value_frontier.csv", rows)

    def draw(axis: Any, plt: Any) -> None:
        from matplotlib.colors import Normalize

        energies = [row["energy_per_correct_j"] for row in rows if row["energy_per_correct_j"]]
        norm = Normalize(min(energies), max(energies)) if len(set(energies)) > 1 else None
        cmap = plt.get_cmap("RdYlGn_r")
        for row in rows:
            family = str(row.get("family"))
            energy = row["energy_per_correct_j"]
            face = cmap(norm(energy)) if norm is not None and energy is not None else "#9CA3AF"
            speed = row["speed_tps"] or 1
            axis.errorbar(
                row["memory_gb"], row["pass_rate_percent"],
                yerr=[[row["pass_rate_percent"] - row["ci_low_percent"]],
                      [row["ci_high_percent"] - row["pass_rate_percent"]]],
                color=FAMILY_COLORS.get(family, "#374151"), capsize=3, linewidth=1,
            )
            axis.scatter(
                row["memory_gb"], row["pass_rate_percent"], s=35 + speed * 2,
                c=[face], edgecolor=FAMILY_COLORS.get(family, "#374151"), linewidth=2,
                marker=QUANT_MARKERS.get(row["quant_label"], "o"), zorder=3,
            )
            axis.annotate(str(row["variant_id"]), (row["memory_gb"], row["pass_rate_percent"]),
                          xytext=(4, 5), textcoords="offset points", fontsize=7)
        ordered = sorted(frontier, key=lambda row: row["memory_gb"])
        axis.plot([row["memory_gb"] for row in ordered],
                  [row["pass_rate_percent"] for row in ordered], "--", color="#111827")
        axis.set(xlabel="Peak memory (GB)", ylabel="Overall pass rate (%)",
                 title="Laptop Value Frontier")
        axis.set_ylim(0, 105)

    paths = _render(root, "laptop_value_frontier", draw, figsize=(10, 6))
    return _generated(paths, len(rows), pareto_count=len(frontier)), frontier_ids


def workload_decision_matrix(
    root: Path, item_rows: list[dict[str, Any]], frontier_ids: list[str],
    memory_by_id: dict[str, float],
) -> dict[str, Any]:
    variants = sorted(frontier_ids, key=lambda value: (memory_by_id.get(value, math.inf), value))
    if not variants:
        return _skipped("laptop value frontier has no configurations")
    plot_rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    annotations: list[list[str]] = []
    for group, benchmarks in WORKLOAD_GROUPS.items():
        cells: list[dict[str, Any]] = []
        for variant in variants:
            selected = [row for row in item_rows if row.get("variant_id") == variant
                        and row.get("benchmark") in benchmarks]
            successes = sum(_boolean(row.get("passed")) is True for row in selected)
            interval = wilson_interval(successes, len(selected))
            rate = successes / len(selected) * 100 if selected else float("nan")
            cells.append({"rate": rate, "interval": interval, "n": len(selected)})
        available = [cell for cell in cells if cell["n"]]
        best = max(available, key=lambda cell: cell["rate"]) if available else None
        best_interval = best["interval"] if best else None
        matrix.append([cell["rate"] for cell in cells])
        labels: list[str] = []
        for variant, cell in zip(variants, cells):
            overlap = bool(
                best_interval and cell["interval"]
                and cell["interval"][1] >= best_interval[0]
                and best_interval[1] >= cell["interval"][0]
            )
            low, high = cell["interval"] or (None, None)
            plot_rows.append({
                "workload": group, "variant_id": variant, "memory_gb": memory_by_id.get(variant),
                "passed": round(cell["rate"] * cell["n"] / 100) if cell["n"] else 0,
                "attempted": cell["n"], "pass_rate_percent": cell["rate"],
                "ci_low_percent": low * 100 if low is not None else None,
                "ci_high_percent": high * 100 if high is not None else None,
                "is_tied_best": overlap,
            })
            labels.append(
                "—" if not cell["n"] else
                f"{cell['rate']:.0f}\n[{low * 100:.0f}–{high * 100:.0f}]{' *' if overlap else ''}"
            )
        annotations.append(labels)
    _write_csv(root / "data" / "workload_decision_matrix.csv", plot_rows)

    def draw(axis: Any, plt: Any) -> None:
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for y, labels in enumerate(annotations):
            for x, label in enumerate(labels):
                axis.text(x, y, label, ha="center", va="center", fontsize=7,
                          color="white" if not math.isnan(matrix[y][x]) and matrix[y][x] < 60 else "black")
        axis.set_xticks(range(len(variants)),
                        [f"{value}\n{memory_by_id.get(value, float('nan')):.1f} GB" for value in variants],
                        rotation=25, ha="right")
        axis.set_yticks(range(len(WORKLOAD_GROUPS)), list(WORKLOAD_GROUPS))
        axis.set_title("Workload Decision Matrix")
        axis.figure.colorbar(image, ax=axis, label="Pass rate (%)")

    paths = _render(root, "workload_decision_matrix", draw,
                    figsize=(max(9, len(variants) * 2), 8))
    return _generated(paths, len(plot_rows), configuration_count=len(variants))


def _render(root: Path, stem: str, draw: Callable[[Any, Any], None], *, figsize: tuple[float, float]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    draw(axis, plt)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png, svg = plot_dir / f"{stem}.png", plot_dir / f"{stem}.svg"
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    return {"png": str(png.relative_to(root)), "svg": str(svg.relative_to(root)),
            "data": f"data/{stem}.csv"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float:
    return float(value)


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return None


def _quant(value: Any) -> str:
    text = str(value).upper()
    return next((quant for quant in QUANT_ORDER if text.startswith(quant)), text)


def _generated(paths: dict[str, str], row_count: int, **extra: Any) -> dict[str, Any]:
    return {"status": "generated", **paths, "row_count": row_count, **extra}


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, **extra}
