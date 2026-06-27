from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import shutil
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


def quant_survival(root: Path, item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    plot_rows: list[dict[str, Any]] = []
    highlighted: set[str] = set()
    for workload, benchmarks in WORKLOAD_GROUPS.items():
        for family in FAMILY_COLORS:
            family_rows: list[dict[str, Any]] = []
            for quant in QUANT_ORDER:
                selected = [row for row in item_rows if row.get("family") == family
                            and _quant(row.get("quantization")) == quant
                            and row.get("benchmark") in benchmarks]
                passed = sum(_boolean(row.get("passed")) is True for row in selected)
                interval = wilson_interval(passed, len(selected))
                if not selected or interval is None:
                    continue
                family_rows.append({
                    "workload": workload, "family": family, "quantization": quant,
                    "passed": passed, "attempted": len(selected),
                    "pass_rate_percent": passed / len(selected) * 100,
                    "ci_low_percent": interval[0] * 100,
                    "ci_high_percent": interval[1] * 100,
                })
            ordered = sorted(family_rows, key=lambda row: QUANT_ORDER.index(row["quantization"]))
            if any(left["pass_rate_percent"] - right["pass_rate_percent"] > 15
                   for left, right in zip(ordered, ordered[1:])):
                highlighted.add(workload)
            plot_rows.extend(family_rows)
    if not plot_rows:
        return _skipped("requires default per-item results across quantizations")
    _write_csv(root / "data" / "quant_survival.csv", plot_rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True,
                                layout="constrained")
    positions = range(len(QUANT_ORDER))
    for axis, workload in zip(axes.flat, WORKLOAD_GROUPS):
        for family, color in FAMILY_COLORS.items():
            series = [row for row in plot_rows if row["workload"] == workload
                      and row["family"] == family]
            if not series:
                continue
            series.sort(key=lambda row: QUANT_ORDER.index(row["quantization"]))
            x = [QUANT_ORDER.index(row["quantization"]) for row in series]
            y = [row["pass_rate_percent"] for row in series]
            axis.plot(x, y, marker="o", color=color, label=family)
            axis.fill_between(x, [row["ci_low_percent"] for row in series],
                              [row["ci_high_percent"] for row in series], color=color, alpha=.15)
        axis.set_title(workload, color="#B91C1C" if workload in highlighted else "#111827")
        if workload in highlighted:
            axis.set_facecolor("#FEF2F2")
        axis.set_xticks(list(positions), QUANT_ORDER)
        axis.set_ylim(0, 105)
    axes[1, 0].set_ylabel("Pass rate (%)")
    axes[-1, 1].set_xlabel("Quantization")
    axes[0, 0].legend(frameon=False, fontsize=8)
    paths = _save_figure(root, "quant_survival", figure, plt)
    return _generated(paths, len(plot_rows), highlighted_panels=sorted(highlighted))


def deployment_risk(
    root: Path, configuration_rows: list[dict[str, Any]], item_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for config in configuration_rows:
        variant = config.get("variant_id")
        attempted, passed = _integer(config.get("attempted")), _integer(config.get("passed"))
        structured = [row for row in item_rows if row.get("variant_id") == variant
                      and row.get("benchmark") in {"messy_text_to_schema", "tool_use"}]
        invalid = sum(str(row.get("integration_outcome")) != "scored" for row in structured)
        if not attempted or passed is None or not structured:
            continue
        pass_ci = wilson_interval(passed, attempted)
        invalid_ci = wilson_interval(invalid, len(structured))
        rows.append({
            **config, "pass_rate_percent": passed / attempted * 100,
            "pass_ci_low_percent": pass_ci[0] * 100, "pass_ci_high_percent": pass_ci[1] * 100,
            "invalid": invalid, "structured_attempted": len(structured),
            "invalid_rate_percent": invalid / len(structured) * 100,
            "invalid_ci_low_percent": invalid_ci[0] * 100,
            "invalid_ci_high_percent": invalid_ci[1] * 100,
        })
    if not rows:
        return _skipped("requires structured-category item results")
    _write_csv(root / "data" / "deployment_risk.csv", rows)

    def draw(axis: Any, plt: Any) -> None:
        axis.axhspan(.1, 3, color="#DCFCE7", alpha=.8, label="deployable zone")
        for row in rows:
            y = max(row["invalid_rate_percent"], .1)
            axis.errorbar(
                row["pass_rate_percent"], y,
                xerr=[[row["pass_rate_percent"] - row["pass_ci_low_percent"]],
                      [row["pass_ci_high_percent"] - row["pass_rate_percent"]]],
                yerr=[[max(0, y - max(row["invalid_ci_low_percent"], .1))],
                      [row["invalid_ci_high_percent"] - row["invalid_rate_percent"]]],
                fmt=QUANT_MARKERS.get(_quant(row.get("quantization")), "o"),
                color=FAMILY_COLORS.get(str(row.get("family")), "#374151"), capsize=3,
            )
            axis.annotate(str(row["variant_id"]), (row["pass_rate_percent"], y),
                          xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set_yscale("log")
        axis.set(xlabel="Overall pass rate (%)", ylabel="Invalid-output rate (%)",
                 title="Deployment Risk")
        axis.legend(frameon=False)

    paths = _render(root, "deployment_risk", draw, figsize=(10, 6))
    return _generated(paths, len(rows))


def trust_profile(
    root: Path, item_rows: list[dict[str, Any]], frontier_ids: list[str]
) -> dict[str, Any]:
    definitions = [
        ("sycophancy flip", "answer_stability", {"confident_wrong_suggestion"}),
        ("are-you-sure flip", "answer_stability", {"are_you_sure_challenge"}),
        ("false-premise acceptance", "false_missing_information", {"false_premise"}),
        ("over-refusal", "over_refusal", None),
    ]
    rows: list[dict[str, Any]] = []
    for variant in frontier_ids:
        config_rows = [row for row in item_rows if row.get("variant_id") == variant]
        family = str(config_rows[0].get("family")) if config_rows else ""
        for metric, benchmark, subcategories in definitions:
            selected = [row for row in config_rows if row.get("benchmark") == benchmark
                        and (subcategories is None or row.get("subcategory") in subcategories)]
            if metric in {"sycophancy flip", "are-you-sure flip"}:
                paired = [(row, _json_object(row.get("evaluation_details")).get("transition"))
                          for row in selected]
                selected = [row for row, transition in paired
                            if transition in {"stood_by_correct", "flipped_correct_to_wrong"}]
                events = sum(
                    _json_object(row.get("evaluation_details")).get("transition")
                    == "flipped_correct_to_wrong" for row in selected
                )
            else:
                events = sum(_boolean(row.get("passed")) is False for row in selected)
            interval = wilson_interval(events, len(selected))
            if interval is None:
                continue
            rows.append({
                "variant_id": variant, "family": family, "metric": metric,
                "events": events, "attempted": len(selected), "rate_percent": events / len(selected) * 100,
                "ci_low_percent": interval[0] * 100, "ci_high_percent": interval[1] * 100,
            })
    if not rows:
        return _skipped("requires trust-suite results for frontier configurations")
    _write_csv(root / "data" / "trust_profile.csv", rows)

    def draw(axis: Any, plt: Any) -> None:
        from matplotlib.patches import Patch
        metrics = [value[0] for value in definitions]
        width = .18
        hatches = ["", "//", "xx", ".."]
        for index, metric in enumerate(metrics):
            series = [next((row for row in rows if row["variant_id"] == variant
                            and row["metric"] == metric), None) for variant in frontier_ids]
            x = [position + (index - 1.5) * width for position in range(len(frontier_ids))]
            heights = [row["rate_percent"] if row else 0 for row in series]
            bars = axis.bar(x, heights, width, color=[FAMILY_COLORS.get(
                row["family"] if row else "", "#9CA3AF") for row in series],
                hatch=hatches[index], edgecolor="white")
            for bar, row in zip(bars, series):
                if row:
                    axis.errorbar(bar.get_x() + bar.get_width() / 2, row["rate_percent"],
                                  yerr=[[row["rate_percent"] - row["ci_low_percent"]],
                                        [row["ci_high_percent"] - row["rate_percent"]]],
                                  fmt="none", color="#111827", capsize=2)
        axis.set_xticks(range(len(frontier_ids)), frontier_ids, rotation=25, ha="right")
        axis.set(ylabel="Rate (%) — lower is better", title="Trust Profile", ylim=(0, 105))
        axis.legend(handles=[Patch(facecolor="#6B7280", hatch=hatches[i], label=metric)
                             for i, metric in enumerate(metrics)], frameon=False, fontsize=8)

    paths = _render(root, "trust_profile", draw, figsize=(max(10, len(frontier_ids) * 1.8), 6))
    return _generated(paths, len(rows))


def retrieval_depth(
    root: Path, configuration_rows: list[dict[str, Any]], item_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    best: dict[str, dict[str, Any]] = {}
    for config in configuration_rows:
        attempted, passed = _integer(config.get("attempted")), _integer(config.get("passed"))
        family = str(config.get("family"))
        if not attempted or passed is None or family not in FAMILY_COLORS:
            continue
        candidate = (passed / attempted, -QUANT_ORDER.index(_quant(config.get("quantization"))))
        current = best.get(family)
        if current is None or candidate > current["rank"]:
            best[family] = {"config": config, "rank": candidate}
    rows: list[dict[str, Any]] = []
    positions = ["start", "middle", "end"]
    lengths = [("2K", "2k_context"), ("4K", "4k_context"), ("8K", "8k_context")]
    for family, selected in best.items():
        config = selected["config"]
        for position in positions:
            for label, tag in lengths:
                items = [row for row in item_rows if row.get("variant_id") == config.get("variant_id")
                         and row.get("benchmark") == "long_text_retrieval"
                         and row.get("subcategory") == f"fact_at_{position}"
                         and tag in _tags(row.get("tags"))]
                passed = sum(_boolean(row.get("passed")) is True for row in items)
                interval = wilson_interval(passed, len(items))
                if interval:
                    rows.append({
                        "family": family, "variant_id": config.get("variant_id"),
                        "position": position, "context_length": label,
                        "passed": passed, "attempted": len(items),
                        "pass_rate_percent": passed / len(items) * 100,
                        "ci_low_percent": interval[0] * 100, "ci_high_percent": interval[1] * 100,
                    })
    if not rows:
        return _skipped("requires long-retrieval results with context tags")
    _write_csv(root / "data" / "retrieval_depth.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, len(best), figsize=(5 * len(best), 4.8),
                                sharex=True, sharey=True, layout="constrained")
    axes_list = [axes] if len(best) == 1 else list(axes)
    for axis, family in zip(axes_list, best):
        matrix = []
        for position in positions:
            matrix.append([next((row["pass_rate_percent"] for row in rows
                                 if row["family"] == family and row["position"] == position
                                 and row["context_length"] == label), float("nan"))
                           for label, _ in lengths])
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        for y, position in enumerate(positions):
            for x, (label, _) in enumerate(lengths):
                row = next((row for row in rows if row["family"] == family
                            and row["position"] == position and row["context_length"] == label), None)
                if row:
                    axis.text(x, y, f"{row['pass_rate_percent']:.0f}\n"
                              f"[{row['ci_low_percent']:.0f}–{row['ci_high_percent']:.0f}]",
                              ha="center", va="center", fontsize=8)
        axis.set_xticks(range(3), [label for label, _ in lengths])
        axis.set_yticks(range(3), positions)
        axis.set_title(f"{family} — {best[family]['config']['quantization']}")
    figure.colorbar(image, ax=axes_list, label="Retrieval pass rate (%)")
    paths = _save_figure(root, "retrieval_depth", figure, plt)
    return _generated(paths, len(rows), family_count=len(best))


def context_speed(
    root: Path, frontier_configs: list[dict[str, Any]], context_items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    points = [(0, None), (2048, "2k_context"), (4096, "4k_context"), (8192, "8k_context")]
    for config in frontier_configs:
        matched = [row for row in context_items
                   if row.get("architecture") == config.get("architecture")
                   and _quant(row.get("quantization")) == _quant(config.get("quantization"))]
        for tokens, tag in points:
            selected = [row for row in matched if (
                (_number_or_none(row.get("prompt_tokens")) or math.inf) <= 256
                if tag is None else tag in _tags(row.get("tags"))
            )]
            speeds = [_number_or_none(row.get("output_tokens_per_second")) for row in selected]
            ttfts = [_number_or_none(row.get("ttft_seconds")) for row in selected]
            speeds, ttfts = [value for value in speeds if value is not None], [value for value in ttfts if value is not None]
            if speeds and ttfts:
                rows.append({
                    "variant_id": config.get("variant_id"), "family": config.get("family"),
                    "tokens_in_context": tokens, "generation_speed_tps": sum(speeds) / len(speeds),
                    "ttft_seconds": sum(ttfts) / len(ttfts), "n": len(selected),
                })
    if not rows:
        return _skipped("requires context-profile token, speed, and TTFT measurements")
    _write_csv(root / "data" / "context_speed.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for variant in dict.fromkeys(row["variant_id"] for row in rows):
        series = sorted([row for row in rows if row["variant_id"] == variant], key=lambda row: row["tokens_in_context"])
        color = FAMILY_COLORS.get(str(series[0]["family"]), "#374151")
        axes[0].plot([row["tokens_in_context"] for row in series],
                     [row["generation_speed_tps"] for row in series], marker="o", color=color, label=variant)
        axes[1].plot([row["tokens_in_context"] for row in series],
                     [row["ttft_seconds"] for row in series], marker="o", color=color)
    worst = max(rows, key=lambda row: row["ttft_seconds"])
    axes[1].annotate(f"worst: {worst['variant_id']} {worst['ttft_seconds']:.2f}s",
                     (worst["tokens_in_context"], worst["ttft_seconds"]), xytext=(5, 6), textcoords="offset points")
    axes[0].set(ylabel="Generation speed (tok/s)", title="Context Speed")
    axes[1].set(xlabel="Tokens already in context", ylabel="Time to first token (s)")
    axes[1].set_xticks([value for value, _ in points], ["0", "2K", "4K", "8K"])
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    paths = _save_figure(root, "context_speed", figure, plt)
    return _generated(paths, len(rows))


def config_effects(
    root: Path, default_items: list[dict[str, Any]], setting_items: list[dict[str, Any]]
) -> dict[str, Any]:
    keys = sorted({(str(row.get("architecture")), _quant(row.get("quantization")))
                   for row in setting_items})
    rows: list[dict[str, Any]] = []
    for architecture, quant in keys:
        treatments = [row for row in setting_items if row.get("architecture") == architecture
                      and _quant(row.get("quantization")) == quant]
        for effect, chosen in [
            ("temperature", [row for row in treatments if _number_or_none(row.get("temperature")) == .7]),
            ("repeat_penalty", [row for row in treatments if _number_or_none(row.get("repeat_penalty")) == 1.1]),
        ]:
            chosen_ids = {row.get("item_id") for row in chosen}
            base = [row for row in default_items if row.get("architecture") == architecture
                    and _quant(row.get("quantization")) == quant
                    and row.get("item_id") in chosen_ids]
            for level, selected in [("off", base), ("on", chosen)]:
                _append_rate(rows, architecture, quant, effect, level, "probe pass rate", selected)
        structured_on = [row for row in treatments
                         if str(row.get("constrained_decoding")) != "none"]
        structured_ids = {row.get("item_id") for row in structured_on}
        structured_base = [row for row in default_items
                           if row.get("architecture") == architecture
                           and _quant(row.get("quantization")) == quant
                           and row.get("item_id") in structured_ids]
        for level, selected in [("off", structured_base), ("on", structured_on)]:
            parseable = [row for row in selected if row.get("integration_outcome") == "scored"]
            _append_rate(rows, architecture, quant, "grammar", level, "parse rate", selected,
                         successes=len(parseable))
            _append_rate(rows, architecture, quant, "grammar", level, "value accuracy",
                         parseable)
    has_pair = any(
        {row["level"] for row in rows if row["architecture"] == architecture
         and row["quantization"] == quant and row["effect"] == effect
         and row["metric"] == metric} == {"off", "on"}
        for architecture, quant in keys
        for effect, metric in [("temperature", "probe pass rate"),
                               ("repeat_penalty", "probe pass rate"),
                               ("grammar", "parse rate")]
    )
    if not rows or not has_pair:
        return _skipped("requires matching default and setting-specific probe results")
    _write_csv(root / "data" / "config_effects.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, layout="constrained")
    for axis, effect, title in zip(axes, ["temperature", "repeat_penalty", "grammar"],
                                   ["Temperature 0 → 0.7", "Repeat penalty off → on", "Grammar off → on"]):
        metrics = ["parse rate", "value accuracy"] if effect == "grammar" else ["probe pass rate"]
        for architecture, quant in keys:
            family = next((row.get("family") for row in setting_items if row.get("architecture") == architecture), "")
            for metric in metrics:
                series = [row for row in rows if row["architecture"] == architecture and row["quantization"] == quant
                          and row["effect"] == effect and row["metric"] == metric]
                if len(series) != 2:
                    continue
                series.sort(key=lambda row: row["level"] == "on")
                style = "--" if metric == "value accuracy" else "-"
                axis.plot([0, 1], [row["rate_percent"] for row in series], style,
                          color=FAMILY_COLORS.get(str(family), "#374151"), marker=QUANT_MARKERS.get(quant, "o"),
                          alpha=.8, label=f"{architecture} {quant} {metric}" if effect == "grammar" else f"{architecture} {quant}")
                axis.errorbar([0, 1], [row["rate_percent"] for row in series],
                              yerr=[[row["rate_percent"] - row["ci_low_percent"] for row in series],
                                    [row["ci_high_percent"] - row["rate_percent"] for row in series]],
                              fmt="none", color=FAMILY_COLORS.get(str(family), "#374151"), capsize=2)
        axis.set_xticks([0, 1], ["off", "on"])
        axis.set_title(title)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[-1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    paths = _save_figure(root, "config_effects", figure, plt)
    return _generated(paths, len(rows))


def calibration(root: Path, item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [(index / 5, (index + 1) / 5) for index in range(5)]
    rows: list[dict[str, Any]] = []
    thin: list[dict[str, Any]] = []
    for family in FAMILY_COLORS:
        confidence_rows = [row for row in item_rows if row.get("family") == family
                           and row.get("benchmark") == "confidence_correctness"]
        for index, (low, high) in enumerate(bins):
            selected = []
            for row in confidence_rows:
                details = _json_object(row.get("evaluation_details"))
                probability = _number_or_none(details.get("confidence_probability"))
                if probability is not None and low <= probability <= (high if index == 4 else high - 1e-12):
                    selected.append(details)
            if len(selected) < 30:
                thin.append({"family": family, "bin": f"{low:.1f}-{high:.1f}", "n": len(selected)})
                continue
            correct = sum(bool(details.get("answer_correct")) for details in selected)
            interval = wilson_interval(correct, len(selected))
            rows.append({
                "family": family, "bin_low": low * 100, "bin_high": high * 100,
                "confidence_midpoint_percent": (low + high) * 50,
                "correct": correct, "n": len(selected), "accuracy_percent": correct / len(selected) * 100,
                "ci_low_percent": interval[0] * 100, "ci_high_percent": interval[1] * 100,
            })
    if thin:
        return _skipped("every one of the 15 family confidence bins must have n >= 30",
                        thin_bins=thin)
    _write_csv(root / "data" / "calibration.csv", rows)

    def draw(axis: Any, plt: Any) -> None:
        axis.plot([0, 100], [0, 100], "--", color="#6B7280", label="perfect calibration")
        for family, color in FAMILY_COLORS.items():
            series = [row for row in rows if row["family"] == family]
            axis.errorbar([row["confidence_midpoint_percent"] for row in series],
                          [row["accuracy_percent"] for row in series],
                          yerr=[[row["accuracy_percent"] - row["ci_low_percent"] for row in series],
                                [row["ci_high_percent"] - row["accuracy_percent"] for row in series]],
                          marker="o", capsize=3, color=color, label=family)
        axis.set(xlabel="Stated confidence bin (%)", ylabel="Actual accuracy (%)",
                 title="Calibration", xlim=(0, 100), ylim=(0, 100))
        axis.legend(frameon=False)

    paths = _render(root, "calibration", draw, figsize=(7, 7))
    return _generated(paths, len(rows))


def thermal_drift(root: Path, item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    series_by_variant: dict[str, list[tuple[int, float]]] = {}
    for variant in dict.fromkeys(str(row.get("variant_id")) for row in item_rows):
        raw_series = [
            (_integer(row.get("run_order")), _number_or_none(row.get("output_tokens_per_second")))
            for row in item_rows if str(row.get("variant_id")) == variant
        ]
        clean = sorted((order, speed) for order, speed in raw_series
                       if order is not None and speed is not None)
        if len(clean) < 25:
            continue
        slope, intercept = _linear_fit(clean)
        start, end = intercept + slope * clean[0][0], intercept + slope * clean[-1][0]
        drop = (start - end) / start * 100 if start > 0 else 0.0
        rng = random.Random(f"thermal:{variant}")
        orders = [value[0] for value in clean]
        speeds = [value[1] for value in clean]
        more_negative = 0
        for _ in range(1999):
            shuffled = speeds.copy()
            rng.shuffle(shuffled)
            permuted_slope, _ = _linear_fit(list(zip(orders, shuffled)))
            more_negative += permuted_slope <= slope
        p_value = (more_negative + 1) / 2000
        diagnostics.append({"variant_id": variant, "slope_tps_per_run": slope,
                            "p_value_one_sided": p_value, "predicted_drop_percent": drop,
                            "n": len(clean)})
        series_by_variant[variant] = clean
    eligible = [row for row in diagnostics if row["slope_tps_per_run"] < 0
                and row["p_value_one_sided"] < .05 and row["predicted_drop_percent"] > 10]
    if not eligible:
        return _skipped("no configuration has p < 0.05 and a predicted speed drop above 10%",
                        diagnostics=diagnostics)
    rolling_rows: list[dict[str, Any]] = []
    for variant, series in series_by_variant.items():
        for index in range(24, len(series)):
            window = series[index - 24:index + 1]
            rolling_rows.append({"variant_id": variant, "run_order": series[index][0],
                                 "rolling_mean_tps": sum(value[1] for value in window) / 25})
    _write_csv(root / "data" / "thermal_drift.csv", rolling_rows)

    def draw(axis: Any, plt: Any) -> None:
        for variant in series_by_variant:
            series = [row for row in rolling_rows if row["variant_id"] == variant]
            family = next((str(row.get("family")) for row in item_rows
                           if str(row.get("variant_id")) == variant), "")
            axis.plot([row["run_order"] for row in series],
                      [row["rolling_mean_tps"] for row in series],
                      color=FAMILY_COLORS.get(family, "#374151"), label=variant)
        axis.set(xlabel="Run order within batch", ylabel="Rolling mean speed (tok/s)",
                 title="Thermal Drift — window 25")
        axis.legend(frameon=False, fontsize=7, ncol=2)

    paths = _render(root, "thermal_drift", draw, figsize=(11, 6))
    return _generated(paths, len(rolling_rows), diagnostics=diagnostics)


def generate_final_figure_bundle(
    default_experiment: Path,
    temperature_experiment: Path,
    constrained_experiment: Path,
    repetition_experiment: Path,
    context_experiment: Path,
) -> Path:
    from llm_workload_benchmark.artifacts import export_experiment_artifacts

    experiments = [default_experiment, temperature_experiment, constrained_experiment,
                   repetition_experiment, context_experiment]
    for experiment in dict.fromkeys(experiments):
        export_experiment_artifacts(experiment)
    default_root = default_experiment / "artifacts"
    context_root = context_experiment / "artifacts"
    output = default_root / "final_figures"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    configurations = _read_csv(default_root / "data" / "configurations.csv")
    items = _read_csv(default_root / "data" / "items.csv")
    setting_items = []
    for experiment in (temperature_experiment, constrained_experiment,
                       repetition_experiment):
        setting_items.extend(_read_csv(experiment / "artifacts" / "data" / "items.csv"))
    context_items = _read_csv(context_root / "data" / "items.csv")
    frontier_plot, frontier_ids = laptop_value_frontier(output, configurations)
    memory_by_id = {str(row["variant_id"]): (_number_or_none(row.get("peak_process_memory_bytes")) or 0) / 1e9
                    for row in configurations}
    frontier_configs = [row for row in configurations if row.get("variant_id") in frontier_ids]
    plots = {
        "laptop_value_frontier": frontier_plot,
        "workload_decision_matrix": workload_decision_matrix(output, items, frontier_ids, memory_by_id),
        "quant_survival": quant_survival(output, items),
        "deployment_risk": deployment_risk(output, configurations, items),
        "trust_profile": trust_profile(output, items, frontier_ids),
        "retrieval_depth": retrieval_depth(output, configurations, items),
        "context_speed": context_speed(output, frontier_configs, context_items),
        "config_effects": config_effects(output, items, setting_items),
        "calibration": calibration(output, items),
        "thermal_drift": thermal_drift(output, items),
    }
    manifest = {
        "schema_version": 1,
        "sources": {
            "default": str(default_experiment),
            "temperature": str(temperature_experiment),
            "constrained_decoding": str(constrained_experiment),
            "repetition_penalty": str(repetition_experiment),
            "context": str(context_experiment),
        },
        "family_colors": FAMILY_COLORS,
        "confidence_interval": "wilson_95",
        "plots": plots,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return output


def _append_rate(rows: list[dict[str, Any]], architecture: str, quant: str,
                 effect: str, level: str, metric: str, selected: list[dict[str, Any]],
                 *, successes: int | None = None) -> None:
    successes = (sum(_boolean(row.get("passed")) is True for row in selected)
                 if successes is None else successes)
    interval = wilson_interval(successes, len(selected))
    if interval:
        rows.append({"architecture": architecture, "quantization": quant, "effect": effect,
                     "level": level, "metric": metric, "successes": successes,
                     "attempted": len(selected), "rate_percent": successes / len(selected) * 100,
                     "ci_low_percent": interval[0] * 100, "ci_high_percent": interval[1] * 100})


def _linear_fit(series: list[tuple[int, float]]) -> tuple[float, float]:
    mean_x = sum(value[0] for value in series) / len(series)
    mean_y = sum(value[1] for value in series) / len(series)
    denominator = sum((value[0] - mean_x) ** 2 for value in series)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in series) / denominator
             if denominator else 0.0)
    return slope, mean_y - slope * mean_x


def _render(root: Path, stem: str, draw: Callable[[Any, Any], None], *, figsize: tuple[float, float]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    draw(axis, plt)
    return _save_figure(root, stem, figure, plt)


def _save_figure(root: Path, stem: str, figure: Any, plt: Any) -> dict[str, str]:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _quant(value: Any) -> str:
    text = str(value).upper()
    return next((quant for quant in QUANT_ORDER if text.startswith(quant)), text)


def _generated(paths: dict[str, str], row_count: int, **extra: Any) -> dict[str, Any]:
    return {"status": "generated", **paths, "row_count": row_count, **extra}


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, **extra}
