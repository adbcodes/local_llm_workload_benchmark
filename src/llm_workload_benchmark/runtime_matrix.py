from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml

from llm_workload_benchmark.artifacts import ArtifactError, export_experiment_artifacts
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
    try:
        export_experiment_artifacts(
            experiment,
            experiment_metadata={
                "kind": "runtime_matrix",
                "combination_count": combination_count(runtime_config),
                "axes": runtime_config.axes.model_dump(mode="json"),
                "quantizations": [
                    variant.model_dump(mode="json")
                    for variant in runtime_config.quantizations
                ],
            },
        )
    except ArtifactError as error:
        raise RuntimeMatrixError(
            f"runtime inference completed but artifact export failed: {error}; "
            f"raw experiment: {experiment}"
        ) from error
    return experiment


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


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
