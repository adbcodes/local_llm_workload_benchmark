from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml

from llm_workload_benchmark.config import (
    BenchmarkConfig,
    BenchmarkSettings,
    GenerationConfig,
    JudgeConfig,
    JudgePanelConfig,
    ModelConfig,
)
from llm_workload_benchmark.runner import (
    BackendFactory,
    JudgeBackendFactory,
    LlamaCppBackend,
    RunProgress,
    _default_judge_backend_factory,
    run_matrix,
)


class RuntimeMatrixError(ValueError):
    pass


class QuantizationVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    quantization: str = Field(min_length=1)
    model_path: Path
    enabled: bool = True


class RuntimeModelDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["llama_cpp"] = "llama_cpp"
    architecture: str | None = None
    family: str | None = None
    context_window: int = Field(default=8192, ge=512)
    gpu_layers: int = Field(default=-1, ge=-1)
    threads: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=512, ge=1)
    flash_attention: bool = False
    kv_cache_type: str | None = None
    chat_format: str | None = None
    response_cleanup: Literal["none", "strip_empty_think", "strip_think"] = "none"
    system_prompt: str = Field(min_length=1)
    verbose: bool = False


class RuntimeAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: list[float] = Field(default_factory=lambda: [0.0], min_length=1)
    top_p: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    top_k: list[int] = Field(default_factory=lambda: [40], min_length=1)
    repeat_penalty: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    max_output_tokens: list[int] = Field(default_factory=lambda: [256], min_length=1)
    constrained_decoding: list[Literal["none", "json", "json_when_requested"]] = Field(
        default_factory=lambda: ["none"], min_length=1
    )
    context_window: list[int] | None = None
    threads: list[int | None] | None = None
    batch_size: list[int] | None = None
    gpu_layers: list[int] | None = None
    flash_attention: list[bool] | None = None
    kv_cache_type: list[str | None] | None = None

    @field_validator(
        "temperature", "top_p", "top_k", "repeat_penalty", "max_output_tokens",
        "constrained_decoding", "context_window", "threads", "batch_size",
        "gpu_layers", "flash_attention", "kv_cache_type",
    )
    @classmethod
    def values_must_be_unique(cls, values: list[Any] | None) -> list[Any] | None:
        if values is not None and len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise ValueError("runtime axis values must be unique")
        return values


class RuntimeMatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    benchmark: BenchmarkSettings
    model: RuntimeModelDefaults
    quantizations: list[QuantizationVariant] = Field(min_length=1)
    axes: RuntimeAxes = Field(default_factory=RuntimeAxes)
    judge: JudgeConfig | None = None
    judge_panel: JudgePanelConfig | None = None


def load_runtime_matrix(path: Path) -> RuntimeMatrixConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeMatrixError(f"runtime matrix does not exist: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeMatrixError(f"could not read runtime matrix: {error}") from error
    try:
        return RuntimeMatrixConfig.model_validate(payload)
    except ValidationError as error:
        raise RuntimeMatrixError(f"runtime matrix validation failed:\n{error}") from error


def combination_count(config: RuntimeMatrixConfig) -> int:
    enabled_quantizations = sum(variant.enabled for variant in config.quantizations)
    values = _axis_values(config)
    count = enabled_quantizations
    for axis in values.values():
        count *= len(axis)
    return count


def expand_runtime_matrix(config: RuntimeMatrixConfig) -> BenchmarkConfig:
    axes = _axis_values(config)
    names = list(axes)
    models: list[ModelConfig] = []
    for quantization in config.quantizations:
        if not quantization.enabled:
            continue
        for combination in itertools.product(*(axes[name] for name in names)):
            settings = dict(zip(names, combination))
            generation = GenerationConfig(
                max_output_tokens=settings["max_output_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                repeat_penalty=settings["repeat_penalty"],
                constrained_decoding=settings["constrained_decoding"],
            )
            model_id = _variant_id(quantization.id, settings)
            defaults = config.model
            models.append(
                ModelConfig(
                    id=model_id,
                    backend=defaults.backend,
                    model_path=quantization.model_path,
                    quantization=quantization.quantization,
                    architecture=defaults.architecture,
                    family=defaults.family,
                    context_window=settings["context_window"],
                    gpu_layers=settings["gpu_layers"],
                    threads=settings["threads"],
                    batch_size=settings["batch_size"],
                    flash_attention=settings["flash_attention"],
                    kv_cache_type=settings["kv_cache_type"],
                    chat_format=defaults.chat_format,
                    response_cleanup=defaults.response_cleanup,
                    verbose=defaults.verbose,
                    system_prompt=defaults.system_prompt,
                    generation=generation,
                )
            )
    if not models:
        raise RuntimeMatrixError("runtime matrix has no enabled quantizations")
    return BenchmarkConfig(
        schema_version=1,
        benchmark=config.benchmark,
        models=models,
        judge=config.judge,
        judge_panel=config.judge_panel,
    )


def validate_model_files(config: BenchmarkConfig, root: Path) -> None:
    missing = sorted(
        {
            str(path)
            for model in config.models
            if not (path := _resolve(root, model.model_path)).is_file()
        }
    )
    if missing:
        raise RuntimeMatrixError(
            "missing quantization model files:\n- " + "\n- ".join(missing)
        )


def run_runtime_matrix(
    runtime_config: RuntimeMatrixConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    backend_factory: BackendFactory = LlamaCppBackend,
    judge_backend_factory: JudgeBackendFactory = _default_judge_backend_factory,
    progress_callback: Callable[[RunProgress], None] | None = None,
    peak_memory_reader: Callable[[], int | None] | None = None,
    validate_files: bool = True,
) -> Path:
    root = (project_root or Path.cwd()).resolve()
    benchmark_config = expand_runtime_matrix(runtime_config)
    if validate_files:
        validate_model_files(benchmark_config, root)
    experiment = run_matrix(
        benchmark_config,
        config_path,
        project_root=root,
        backend_factory=backend_factory,
        judge_backend_factory=judge_backend_factory,
        peak_memory_reader=peak_memory_reader,
        progress_callback=progress_callback,
    )
    export_runtime_artifacts(experiment, runtime_config)
    return experiment


def export_runtime_artifacts(
    experiment: Path,
    config: RuntimeMatrixConfig,
) -> dict[str, Path]:
    index = _read_json(experiment / "experiment.json")
    run_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    machine: dict[str, Any] | None = None

    for model_entry in index.get("models", []):
        run_directory = experiment / model_entry["run_directory"]
        manifest_path = run_directory / "manifest.json"
        summary_path = run_directory / "summary.json"
        if manifest_path.is_file() and machine is None:
            manifest = _read_json(manifest_path)
            machine = {
                "environment": manifest.get("environment"),
                "git": manifest.get("git"),
                "project_version": manifest.get("project_version"),
            }
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        base = _model_axes(summary.get("model", {}), model_entry["model_id"])
        totals = summary.get("totals", {})
        telemetry = summary.get("telemetry") or {}
        run_rows.append(
            {
                **base,
                "status": model_entry.get("status"),
                "summary_status": summary.get("status"),
                "model_load_seconds": summary.get("model_load_seconds"),
                "model_file_bytes": summary.get("model", {}).get("file_size_bytes"),
                "attempted": totals.get("attempted"),
                "completed": totals.get("completed"),
                "pass_rate": totals.get("pass_rate"),
                "mean_score": totals.get("mean_score"),
                "mean_latency_seconds": totals.get("mean_latency_seconds"),
                "mean_ttft_seconds": totals.get("mean_time_to_first_token_seconds"),
                "mean_output_tokens_per_second": totals.get("mean_output_tokens_per_second_end_to_end"),
                "mean_process_cpu_seconds": totals.get("mean_process_cpu_seconds"),
                "mean_process_cpu_utilization_percent": totals.get("mean_process_cpu_utilization_percent"),
                "peak_process_memory_bytes": totals.get("peak_process_memory_bytes"),
                "integration_friction_rate": totals.get("integration_friction_rate"),
                "run_to_run_flip_rate": totals.get("run_to_run_flip_rate"),
                "mean_system_gpu_utilization_percent": telemetry.get("mean_system_gpu_utilization_percent"),
                "peak_system_gpu_utilization_percent": telemetry.get("peak_system_gpu_utilization_percent"),
                "mean_cpu_power_watts": telemetry.get("mean_cpu_power_watts"),
                "mean_gpu_power_watts": telemetry.get("mean_gpu_power_watts"),
                "mean_system_power_watts": telemetry.get("mean_system_power_watts"),
                "mean_cpu_temperature_c": telemetry.get("mean_cpu_temperature_c"),
                "telemetry_sample_count": telemetry.get("sample_count"),
                "error": model_entry.get("error"),
            }
        )
        for benchmark, group in summary.get("benchmarks", {}).items():
            overall = group.get("overall", {})
            benchmark_rows.append(
                {
                    **base,
                    "benchmark": benchmark,
                    "reported_score": group.get("reported_score"),
                    "score_formula": group.get("score_formula"),
                    "attempted": overall.get("attempted"),
                    "pass_rate": overall.get("pass_rate"),
                    "mean_score": overall.get("mean_score"),
                    "mean_latency_seconds": overall.get("mean_latency_seconds"),
                    "mean_ttft_seconds": overall.get("mean_time_to_first_token_seconds"),
                    "mean_output_tokens_per_second": overall.get("mean_output_tokens_per_second_end_to_end"),
                    "peak_process_memory_bytes": overall.get("peak_process_memory_bytes"),
                }
            )
        results_path = run_directory / "results.jsonl"
        if results_path.is_file():
            for record in _read_jsonl(results_path):
                evaluation = record.get("evaluation") or {}
                item_rows.append(
                    {
                        **base,
                        "benchmark": record.get("benchmark"),
                        "suite": record.get("suite"),
                        "item_id": record.get("item_id"),
                        "difficulty": record.get("difficulty"),
                        "repetition": record.get("repetition"),
                        "status": record.get("status"),
                        "passed": evaluation.get("passed"),
                        "score": evaluation.get("score"),
                        "latency_seconds": record.get("latency_seconds"),
                        "ttft_seconds": record.get("time_to_first_token_seconds"),
                        "output_tokens_per_second": record.get("output_tokens_per_second_end_to_end"),
                        "process_cpu_seconds": record.get("process_cpu_seconds"),
                        "process_cpu_utilization_percent": record.get("process_cpu_utilization_percent"),
                        "process_memory_bytes": record.get("peak_process_memory_bytes"),
                        "integration_outcome": record.get("integration_outcome"),
                    }
                )

    _attach_q8_baselines(run_rows)
    _attach_q8_baselines(benchmark_rows, include_benchmark=True)
    paths = {
        "json": experiment / "runtime_results.json",
        "runs_csv": experiment / "runtime_runs.csv",
        "benchmarks_csv": experiment / "runtime_benchmarks.csv",
        "items_csv": experiment / "runtime_items.csv",
        "machine_json": experiment / "machine.json",
    }
    machine = machine or {"environment": None, "git": None, "project_version": None}
    _write_json(paths["machine_json"], machine)
    _write_csv(paths["runs_csv"], run_rows)
    _write_csv(paths["benchmarks_csv"], benchmark_rows)
    _write_csv(paths["items_csv"], item_rows)
    _write_json(
        paths["json"],
        {
            "schema_version": 1,
            "experiment_id": index.get("experiment_id"),
            "status": index.get("status"),
            "combination_count": combination_count(config),
            "axes": config.axes.model_dump(mode="json"),
            "quantizations": [
                variant.model_dump(mode="json") for variant in config.quantizations
            ],
            "machine": machine,
            "runs": run_rows,
            "benchmarks": benchmark_rows,
            "artifacts": {name: path.name for name, path in paths.items()},
        },
    )
    return paths


def _axis_values(config: RuntimeMatrixConfig) -> dict[str, list[Any]]:
    defaults = config.model
    axes = config.axes
    return {
        "temperature": axes.temperature,
        "top_p": axes.top_p,
        "top_k": axes.top_k,
        "repeat_penalty": axes.repeat_penalty,
        "max_output_tokens": axes.max_output_tokens,
        "constrained_decoding": axes.constrained_decoding,
        "context_window": axes.context_window or [defaults.context_window],
        "threads": axes.threads or [defaults.threads],
        "batch_size": axes.batch_size or [defaults.batch_size],
        "gpu_layers": axes.gpu_layers or [defaults.gpu_layers],
        "flash_attention": axes.flash_attention or [defaults.flash_attention],
        "kv_cache_type": axes.kv_cache_type or [defaults.kv_cache_type],
    }


def _variant_id(quantization_id: str, settings: dict[str, Any]) -> str:
    parts = [quantization_id]
    labels = {
        "temperature": "t",
        "top_p": "p",
        "top_k": "k",
        "repeat_penalty": "rp",
        "max_output_tokens": "out",
        "constrained_decoding": "decode",
        "context_window": "ctx",
        "threads": "thr",
        "batch_size": "batch",
        "gpu_layers": "gpu",
        "flash_attention": "fa",
        "kv_cache_type": "kv",
    }
    for name, value in settings.items():
        rendered = "auto" if value is None else str(value).lower().replace(".", "p")
        parts.append(f"{labels[name]}{rendered}")
    return re.sub(r"[^a-zA-Z0-9._-]", "-", "-".join(parts))


def _model_axes(model: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    generation = model.get("generation", {})
    return {
        "variant_id": model.get("id", fallback_id),
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


def _attach_q8_baselines(
    rows: list[dict[str, Any]],
    *,
    include_benchmark: bool = False,
) -> None:
    axis_names = (
        "temperature", "top_p", "top_k", "repeat_penalty",
        "max_output_tokens", "constrained_decoding", "context_window",
        "threads", "batch_size", "gpu_layers", "flash_attention",
        "kv_cache_type",
    )

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values: tuple[Any, ...] = tuple(row.get(name) for name in axis_names)
        return values + ((row.get("benchmark"),) if include_benchmark else ())

    baselines = {
        key(row): row
        for row in rows
        if str(row.get("quantization", "")).upper().startswith("Q8")
    }
    for row in rows:
        baseline = baselines.get(key(row))
        score_name = "reported_score" if include_benchmark else "mean_score"
        score = row.get(score_name)
        baseline_score = baseline.get(score_name) if baseline else None
        row["score_delta_vs_q8"] = _difference(score, baseline_score)
        row["score_retained_vs_q8"] = _ratio(score, baseline_score)
        if not include_benchmark:
            row["memory_saved_vs_q8_bytes"] = _difference(
                baseline.get("peak_process_memory_bytes") if baseline else None,
                row.get("peak_process_memory_bytes"),
            )
            row["speed_ratio_vs_q8"] = _ratio(
                row.get("mean_output_tokens_per_second"),
                baseline.get("mean_output_tokens_per_second") if baseline else None,
            )


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


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict | list)
                    else value
                    for key, value in row.items()
                }
            )
