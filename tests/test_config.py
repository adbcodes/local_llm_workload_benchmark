from pathlib import Path

import pytest

from llm_workload_benchmark.config import ConfigError, load_config


def test_load_config_parses_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
schema_version: 1
benchmark:
  name: test-run
  workload_path: data/workload.jsonl
models:
  - id: model-q4
    backend: llama_cpp
    model_path: models/model-q4.gguf
    quantization: q4
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.benchmark.name == "test-run"
    assert config.benchmark.repetitions == 1
    assert config.models[0].context_window == 4096
    assert config.models[0].gpu_layers == -1
    assert config.models[0].generation.temperature == 0.0


def test_load_config_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
schema_version: 1
benchmark:
  name: test-run
  workload_path: data/workload.jsonl
models:
  - id: repeated-model
    backend: llama_cpp
    model_path: models/first.gguf
  - id: repeated-model
    backend: llama_cpp
    model_path: models/second.gguf
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="model ids must be unique"):
        load_config(config_path)


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
schema_version: 1
benchmark:
  name: test-run
  workload_path: data/workload.jsonl
  typo_field: true
models:
  - id: model-q4
    backend: llama_cpp
    model_path: models/model-q4.gguf
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="typo_field"):
        load_config(config_path)
