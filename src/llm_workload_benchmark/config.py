from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """Raised when a benchmark configuration cannot be loaded or validated."""


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.0, gt=0.0)
    constrained_decoding: Literal["none", "json", "json_when_requested"] = "none"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    backend: Literal["llama_cpp"]
    model_path: Path
    quantization: str | None = None
    architecture: str | None = None
    family: str | None = None
    role: Literal["candidate", "difficulty_anchor"] = "candidate"
    enabled: bool = True
    context_window: int = Field(default=4096, ge=512)
    gpu_layers: int = Field(default=-1, ge=-1)
    threads: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=512, ge=1)
    flash_attention: bool = False
    kv_cache_type: str | None = None
    chat_format: str | None = None
    response_cleanup: Literal["none", "strip_empty_think", "strip_think"] = "none"
    verbose: bool = False
    system_prompt: str = Field(
        default=(
            "Follow the user's instructions precisely. Return only the requested "
            "answer without commentary."
        ),
        min_length=1,
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class JudgeConfig(BaseModel):
    """Configuration for an external model that evaluates generated answers."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["groq"] = "groq"
    model: str = Field(default="openai/gpt-oss-120b", min_length=1)
    family: str = Field(default="openai", min_length=1)
    api_key_env: str = Field(
        default="GROQ_API_KEY",
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    max_completion_tokens: int = Field(default=4096, ge=1, le=8192)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_cooldown_retries: int = Field(default=1, ge=0, le=3)
    rate_limit_fallback_wait_seconds: float = Field(default=60.0, gt=0)
    rate_limit_max_wait_seconds: float = Field(default=3600.0, gt=0)
    max_candidate_characters: int = Field(default=12_000, ge=1)
    cache_path: Path | None = None
    input_price_per_million_tokens: float = Field(default=0.15, ge=0)
    cached_input_price_per_million_tokens: float = Field(default=0.075, ge=0)
    output_price_per_million_tokens: float = Field(default=0.60, ge=0)


class JudgePanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judges: list[JudgeConfig] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def families_are_distinct(self) -> "JudgePanelConfig":
        families = [judge.family.casefold() for judge in self.judges]
        if len(families) != len(set(families)):
            raise ValueError("judge panel requires three different model families")
        return self


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    workload_path: Path
    output_root: Path = Path("runs")
    repetitions: int = Field(default=1, ge=1)
    seed: int = 42
    probe_version: str | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    benchmark: BenchmarkSettings
    models: list[ModelConfig] = Field(min_length=1)
    judge: JudgeConfig | None = None
    judge_panel: JudgePanelConfig | None = None

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
