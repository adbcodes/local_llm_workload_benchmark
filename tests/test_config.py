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
    assert config.models[0].context_window == 16384
    assert config.models[0].generation.max_output_tokens == 4096
    assert config.models[0].gpu_layers == -1
    assert config.models[0].response_cleanup == "none"
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


def test_retained_matrix_configs_load() -> None:
    smoke = load_config(Path("configs/smoke_matrix.yaml"))
    workloads = load_config(Path("configs/final_workloads_matrix.yaml"))
    retrieval = load_config(Path("configs/final_retrieval_matrix.yaml"))
    compression = load_config(
        Path("configs/final_grounded_compression_matrix.yaml")
    )

    assert len(smoke.models) == 5
    assert len(workloads.models) == len(retrieval.models) == len(compression.models) == 20
    assert smoke.benchmark.workload_path == Path("data/suites/smoke.yaml")
    assert {model.quantization for model in smoke.models} == {
        "Q3_K_M",
        "Q4_K_M",
        "Q6_K",
        "Q8_0",
    }
    assert workloads.benchmark.workload_path == Path("data/suites/final_workloads.yaml")
    assert retrieval.benchmark.workload_path == Path(
        "data/suites/final_retrieval.yaml"
    )
    assert compression.benchmark.workload_path == Path(
        "data/suites/grounded_compression.yaml"
    )
    assert compression.judge is not None
