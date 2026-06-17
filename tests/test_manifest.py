import json
from datetime import UTC, datetime
from pathlib import Path

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.manifest import create_run


def test_create_run_writes_reproducibility_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
schema_version: 1
benchmark:
  name: manifest-test
  workload_path: data/workload.jsonl
  output_root: generated-runs
models:
  - id: model-q4
    backend: llama_cpp
    model_path: models/model-q4.gguf
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    created_at = datetime(2026, 7, 15, 10, 30, tzinfo=UTC)

    run_directory = create_run(
        config,
        config_path,
        project_root=tmp_path,
        now=created_at,
    )

    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert run_directory.parent == tmp_path / "generated-runs"
    assert run_directory.name.startswith("2026-07-15_10-30-00-")
    assert manifest["run_id"] == run_directory.name
    assert manifest["created_at_utc"] == "2026-07-15T10:30:00+00:00"
    assert manifest["config"]["benchmark"]["name"] == "manifest-test"
    assert manifest["config_source"]["sha256"]
    assert manifest["environment"]["python_version"]
    assert "physical_memory_bytes" in manifest["environment"]
    assert "hostname" not in manifest["environment"]


def test_create_run_uses_a_unique_directory_for_same_timestamp(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
schema_version: 1
benchmark:
  name: unique-run-test
  workload_path: data/workload.jsonl
models:
  - id: model-q4
    backend: llama_cpp
    model_path: models/model-q4.gguf
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    created_at = datetime(2026, 7, 15, 10, 30, tzinfo=UTC)

    first_run = create_run(config, config_path, project_root=tmp_path, now=created_at)
    second_run = create_run(config, config_path, project_root=tmp_path, now=created_at)

    assert first_run != second_run
    assert first_run.exists()
    assert second_run.exists()
