from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(ValueError):
    """Raised when a benchmark configuration cannot be loaded or validated."""


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    backend: Literal["llama_cpp"]
    model_path: Path
    quantization: str | None = None
    enabled: bool = True
    context_window: int = Field(default=4096, ge=512)
    gpu_layers: int = Field(default=-1, ge=-1)
    threads: int | None = Field(default=None, ge=1)
    chat_format: str | None = None
    response_cleanup: Literal["none", "strip_empty_think"] = "none"
    verbose: bool = False
    system_prompt: str = Field(
        default=(
            "Follow the user's instructions precisely. Return only the requested "
            "answer without commentary."
        ),
        min_length=1,
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    workload_path: Path
    output_root: Path = Path("runs")
    repetitions: int = Field(default=1, ge=1)
    seed: int = 42


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    benchmark: BenchmarkSettings
    models: list[ModelConfig] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def model_ids_must_be_unique(cls, models: list[ModelConfig]) -> list[ModelConfig]:
        model_ids = [model.id for model in models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model ids must be unique")
        return models


def load_config(path: Path) -> BenchmarkConfig:
    """Load a YAML file and validate it as a benchmark configuration."""
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"config file does not exist: {path}") from error
    except OSError as error:
        raise ConfigError(f"could not read config file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"config file is not valid YAML: {error}") from error

    if not isinstance(raw_config, dict):
        raise ConfigError("config file must contain a YAML object at its root")

    try:
        return BenchmarkConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ConfigError(f"config validation failed:\n{error}") from error
