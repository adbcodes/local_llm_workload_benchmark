from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


class ArtifactError(ValueError):
    """Raised when saved experiment data cannot produce an artifact bundle."""


MODEL_FIELDS = [
    "variant_id", "architecture", "family", "backend", "quantization",
    "temperature", "top_p", "top_k", "repeat_penalty", "max_output_tokens",
    "constrained_decoding", "context_window", "threads", "batch_size",
    "gpu_layers", "flash_attention", "kv_cache_type",
]
CONFIGURATION_FIELDS = MODEL_FIELDS + [
    "status", "summary_status", "error", "model_load_seconds", "model_file_bytes",
    "attempted", "completed", "passed", "pass_rate", "mean_score",
    "mean_latency_seconds", "mean_ttft_seconds", "mean_output_tokens_per_second",
    "mean_process_cpu_seconds", "mean_process_cpu_utilization_percent",
    "peak_process_memory_bytes", "integration_friction_rate",
    "run_to_run_flip_rate", "mean_system_gpu_utilization_percent",
    "peak_system_gpu_utilization_percent", "mean_cpu_power_watts",
    "mean_gpu_power_watts", "mean_system_power_watts", "mean_cpu_temperature_c",
    "telemetry_sample_count", "score_delta_vs_q8", "score_retained_vs_q8",
    "memory_saved_vs_q8_bytes", "speed_ratio_vs_q8",
]
GROUP_FIELDS = [
    "attempted", "completed", "passed", "pass_rate", "mean_score",
    "mean_latency_seconds", "mean_ttft_seconds", "mean_output_tokens_per_second",
    "peak_process_memory_bytes", "integration_friction_rate",
]
SUITE_FIELDS = MODEL_FIELDS + [
    "suite", *GROUP_FIELDS, "score_delta_vs_q8", "score_retained_vs_q8",
]
BENCHMARK_FIELDS = MODEL_FIELDS + [
    "benchmark", "reported_score", "score_formula", *GROUP_FIELDS,
    "score_delta_vs_q8", "score_retained_vs_q8",
]
ITEM_FIELDS = MODEL_FIELDS + [
    "benchmark", "suite", "item_id", "source_item", "subcategory", "difficulty",
    "split", "visibility", "dataset_origin", "repetition", "seed", "status",
    "passed", "score", "integration_outcome", "latency_seconds", "ttft_seconds",
    "prompt_tokens", "output_tokens", "reasoning_tokens", "output_characters",
    "output_tokens_per_second", "process_wall_seconds", "process_cpu_seconds",
    "process_cpu_utilization_percent", "process_memory_bytes", "finish_reason",
    "prompt_sha256", "system_prompt_sha256", "error",
]


def export_experiment_artifacts(
    experiment_directory: Path,
    *,
    experiment_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Build normalized CSV tables and a manifest from a saved matrix experiment."""

    experiment = experiment_directory.resolve()
    index = _read_json_object(experiment / "experiment.json")
    entries = index.get("models")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError("experiment index must contain at least one model entry")

    rows = _collect_rows(experiment, entries)
    _attach_q8_baselines(rows["configurations"])
    _attach_q8_baselines(rows["suites"], group_field="suite")
    _attach_q8_baselines(
        rows["benchmarks"],
        group_field="benchmark",
        score_field="reported_score",
    )

    temporary = Path(tempfile.mkdtemp(prefix=".artifacts-", dir=experiment))
    destination = experiment / "artifacts"
    try:
        data_directory = temporary / "data"
        data_directory.mkdir()
        tables = {
            "configurations": (CONFIGURATION_FIELDS, rows["configurations"]),
            "suites": (SUITE_FIELDS, rows["suites"]),
            "benchmarks": (BENCHMARK_FIELDS, rows["benchmarks"]),
            "items": (ITEM_FIELDS, rows["items"]),
        }
        table_manifest: dict[str, dict[str, Any]] = {}
        for name, (fieldnames, table_rows) in tables.items():
            relative_path = Path("data") / f"{name}.csv"
            _write_csv(temporary / relative_path, fieldnames, table_rows)
            table_manifest[name] = {
                "path": str(relative_path),
                "row_count": len(table_rows),
            }

        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "experiment_id": index.get("experiment_id"),
                "experiment_status": index.get("status"),
                "source": "../experiment.json",
                "experiment_metadata": experiment_metadata or {},
                "machine": rows["machine"],
                "tables": table_manifest,
                "plots": {},
            },
        )
        _replace_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "root": destination,
        "manifest": destination / "manifest.json",
        **{
            name: destination / details["path"]
            for name, details in table_manifest.items()
        },
    }


def _collect_rows(experiment: Path, entries: list[Any]) -> dict[str, Any]:
    configuration_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    machine: dict[str, Any] | None = None

    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError("every experiment model entry must be an object")
        model_id = entry.get("model_id")
        status = entry.get("status")
        run_reference = entry.get("run_directory")
        if not isinstance(model_id, str) or not isinstance(status, str):
            raise ArtifactError("model entries require string model_id and status fields")
        if not isinstance(run_reference, str):
            raise ArtifactError(f"model {model_id!r} has no run_directory reference")
        run_directory = _resolve_reference(experiment, run_reference)

        manifest_path = run_directory / "manifest.json"
        if manifest_path.is_file() and machine is None:
            manifest = _read_json_object(manifest_path)
            machine = {
                "environment": manifest.get("environment"),
                "git": manifest.get("git"),
                "project_version": manifest.get("project_version"),
            }

        summary_reference = entry.get("summary")
        if summary_reference is None:
            if status == "completed":
                raise ArtifactError(f"completed model {model_id!r} has no summary reference")
            summary: dict[str, Any] = {}
        elif isinstance(summary_reference, str):
            summary = _read_json_object(_resolve_reference(experiment, summary_reference))
        else:
            raise ArtifactError(f"model {model_id!r} has an invalid summary reference")

        model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
        base = _model_fields(model, model_id)
        totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
        telemetry = (
            summary.get("telemetry")
            if isinstance(summary.get("telemetry"), dict)
            else {}
        )
        configuration_rows.append(
            {
                **base,
                "status": status,
                "summary_status": summary.get("status"),
                "error": entry.get("error") or summary.get("error"),
                "model_load_seconds": summary.get("model_load_seconds"),
                "model_file_bytes": model.get("file_size_bytes"),
                **_aggregate_fields(totals),
                "mean_process_cpu_seconds": totals.get("mean_process_cpu_seconds"),
                "mean_process_cpu_utilization_percent": totals.get(
                    "mean_process_cpu_utilization_percent"
                ),
                "run_to_run_flip_rate": totals.get("run_to_run_flip_rate"),
                "mean_system_gpu_utilization_percent": telemetry.get(
                    "mean_system_gpu_utilization_percent"
                ),
                "peak_system_gpu_utilization_percent": telemetry.get(
                    "peak_system_gpu_utilization_percent"
                ),
                "mean_cpu_power_watts": telemetry.get("mean_cpu_power_watts"),
                "mean_gpu_power_watts": telemetry.get("mean_gpu_power_watts"),
                "mean_system_power_watts": telemetry.get("mean_system_power_watts"),
                "mean_cpu_temperature_c": telemetry.get("mean_cpu_temperature_c"),
                "telemetry_sample_count": telemetry.get("sample_count"),
            }
        )

        for suite, aggregate in _mapping(summary.get("suites")).items():
            suite_rows.append({**base, "suite": suite, **_aggregate_fields(aggregate)})
        for benchmark, group in _mapping(summary.get("benchmarks")).items():
            overall = (
                group.get("overall")
                if isinstance(group.get("overall"), dict)
                else {}
            )
            benchmark_rows.append(
                {
                    **base,
                    "benchmark": benchmark,
                    "reported_score": group.get("reported_score"),
                    "score_formula": group.get("score_formula"),
                    **_aggregate_fields(overall),
                }
            )

        results_path = run_directory / "results.jsonl"
        if results_path.is_file():
            for record in _read_jsonl(results_path):
                evaluation = (
                    record.get("evaluation")
                    if isinstance(record.get("evaluation"), dict)
                    else {}
                )
                item_rows.append(
                    {
                        **base,
                        "benchmark": record.get("benchmark"),
                        "suite": record.get("suite"),
                        "item_id": record.get("item_id"),
                        "source_item": record.get("source_item"),
                        "subcategory": record.get("subcategory"),
                        "difficulty": record.get("difficulty"),
                        "split": record.get("split"),
                        "visibility": record.get("visibility"),
                        "dataset_origin": record.get("dataset_origin"),
                        "repetition": record.get("repetition"),
                        "seed": record.get("seed"),
                        "status": record.get("status"),
                        "passed": evaluation.get("passed"),
                        "score": evaluation.get("score"),
                        "integration_outcome": record.get("integration_outcome"),
                        "latency_seconds": record.get("latency_seconds"),
                        "ttft_seconds": record.get("time_to_first_token_seconds"),
                        "prompt_tokens": record.get("prompt_tokens"),
                        "output_tokens": record.get("output_tokens"),
                        "reasoning_tokens": record.get("reasoning_tokens"),
                        "output_characters": record.get("output_characters"),
                        "output_tokens_per_second": record.get(
                            "output_tokens_per_second_end_to_end"
                        ),
                        "process_wall_seconds": record.get("process_wall_seconds"),
                        "process_cpu_seconds": record.get("process_cpu_seconds"),
                        "process_cpu_utilization_percent": record.get(
                            "process_cpu_utilization_percent"
                        ),
                        "process_memory_bytes": record.get("peak_process_memory_bytes"),
                        "finish_reason": record.get("finish_reason"),
                        "prompt_sha256": record.get("prompt_sha256"),
                        "system_prompt_sha256": record.get("system_prompt_sha256"),
                        "error": record.get("error"),
                    }
                )

    return {
        "configurations": configuration_rows,
        "suites": suite_rows,
        "benchmarks": benchmark_rows,
        "items": item_rows,
        "machine": machine or {"environment": None, "git": None, "project_version": None},
    }


def _model_fields(model: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    generation = model.get("generation") if isinstance(model.get("generation"), dict) else {}
    return {
        "variant_id": model.get("id", fallback_id),
        "architecture": model.get("architecture"),
        "family": model.get("family"),
        "backend": model.get("backend"),
        "quantization": model.get("quantization"),
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "repeat_penalty": generation.get("repeat_penalty"),
        "max_output_tokens": generation.get("max_output_tokens"),
        "constrained_decoding": generation.get("constrained_decoding"),
        "context_window": model.get("context_window"),
        "threads": model.get("threads"),
        "batch_size": model.get("batch_size"),
        "gpu_layers": model.get("gpu_layers"),
        "flash_attention": model.get("flash_attention"),
        "kv_cache_type": model.get("kv_cache_type"),
    }


def _aggregate_fields(aggregate: Any) -> dict[str, Any]:
    values = aggregate if isinstance(aggregate, dict) else {}
    return {
        "attempted": values.get("attempted"),
        "completed": values.get("completed"),
        "passed": values.get("passed"),
        "pass_rate": values.get("pass_rate"),
        "mean_score": values.get("mean_score"),
        "mean_latency_seconds": values.get("mean_latency_seconds"),
        "mean_ttft_seconds": values.get("mean_time_to_first_token_seconds"),
        "mean_output_tokens_per_second": values.get("mean_output_tokens_per_second_end_to_end"),
        "peak_process_memory_bytes": values.get("peak_process_memory_bytes"),
        "integration_friction_rate": values.get("integration_friction_rate"),
    }


def _attach_q8_baselines(
    rows: list[dict[str, Any]],
    *,
    group_field: str | None = None,
    score_field: str = "mean_score",
) -> None:
    axes = (
        "architecture", "family", "temperature", "top_p", "top_k",
        "repeat_penalty", "max_output_tokens", "constrained_decoding",
        "context_window", "threads", "batch_size", "gpu_layers",
        "flash_attention", "kv_cache_type",
    )

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values = tuple(row.get(name) for name in axes)
        return values + ((row.get(group_field),) if group_field else ())

    baselines = {
        key(row): row
        for row in rows
        if str(row.get("quantization", "")).upper().startswith("Q8")
    }
    for row in rows:
        baseline = baselines.get(key(row))
        baseline_score = baseline.get(score_field) if baseline else None
        row["score_delta_vs_q8"] = _difference(
            row.get(score_field), baseline_score
        )
        row["score_retained_vs_q8"] = _ratio(
            row.get(score_field), baseline_score
        )
        if group_field is None:
            row["memory_saved_vs_q8_bytes"] = _difference(
                baseline.get("peak_process_memory_bytes") if baseline else None,
                row.get("peak_process_memory_bytes"),
            )
            row["speed_ratio_vs_q8"] = _ratio(
                row.get("mean_output_tokens_per_second"),
                baseline.get("mean_output_tokens_per_second") if baseline else None,
            )


def _resolve_reference(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError(f"artifact reference escapes experiment directory: {reference}")
    return path


def _mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, dict)
    }


def _difference(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, int | float) or not isinstance(baseline, int | float):
        return None
    return float(value - baseline)


def _ratio(value: Any, baseline: Any) -> float | None:
    if (
        not isinstance(value, int | float)
        or not isinstance(baseline, int | float)
        or baseline == 0
    ):
        return None
    return float(value / baseline)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactError(f"required experiment artifact does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read experiment artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"experiment artifact must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ArtifactError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read experiment artifact {path}: {error}") from error
    return records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, dict | list)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_directory(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)
