import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_workload_benchmark.artifacts import ArtifactError, export_experiment_artifacts
from llm_workload_benchmark.cli import app


def _write_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "matrix-test"
    completed = experiment / "models" / "model-q8"
    failed = experiment / "models" / "model-q4"
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    model = {
        "id": "model-q8",
        "architecture": "model-3b",
        "family": "model",
        "backend": "llama_cpp",
        "quantization": "Q8_0",
        "file_size_bytes": 1_000,
        "context_window": 4096,
        "threads": None,
        "batch_size": 512,
        "gpu_layers": -1,
        "flash_attention": True,
        "kv_cache_type": None,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 40,
            "repeat_penalty": 1.0,
            "max_output_tokens": 128,
            "constrained_decoding": "none",
        },
    }
    aggregate = {
        "attempted": 1,
        "completed": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "mean_score": 1.0,
        "mean_latency_seconds": 0.2,
        "mean_time_to_first_token_seconds": 0.05,
        "mean_output_tokens_per_second_end_to_end": 20.0,
        "peak_process_memory_bytes": 2_000,
        "integration_friction_rate": 0.0,
    }
    (completed / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model": model,
                "model_load_seconds": 0.1,
                "totals": aggregate,
                "suites": {"A": aggregate},
                "benchmarks": {
                    "reasoning": {
                        "overall": aggregate,
                        "reported_score": 1.0,
                        "score_formula": "mean_score",
                    }
                },
                "telemetry": {"sample_count": 2},
            }
        ),
        encoding="utf-8",
    )
    (completed / "manifest.json").write_text(
        json.dumps(
            {
                "project_version": "0.1.0",
                "environment": {"machine_model": "test-machine"},
                "git": {"commit": "abc123"},
            }
        ),
        encoding="utf-8",
    )
    (completed / "results.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark": "reasoning",
                "suite": "A",
                "item_id": "reasoning_001",
                "difficulty": "easy",
                "repetition": 1,
                "evaluation": {"passed": True, "score": 1.0},
                "latency_seconds": 0.2,
                "time_to_first_token_seconds": 0.05,
                "output_tokens_per_second_end_to_end": 20.0,
                "peak_process_memory_bytes": 2_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "matrix-test",
                "status": "partial_failure",
                "models": [
                    {
                        "model_id": "model-q8",
                        "status": "completed",
                        "run_directory": "models/model-q8",
                        "summary": "models/model-q8/summary.json",
                        "error": None,
                    },
                    {
                        "model_id": "model-q4",
                        "status": "failed",
                        "run_directory": "models/model-q4",
                        "summary": None,
                        "error": {"type": "RuntimeError", "message": "load failed"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return experiment


def test_export_experiment_artifacts_writes_normalized_partial_results(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)

    paths = export_experiment_artifacts(experiment)

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["experiment_status"] == "partial_failure"
    assert manifest["machine"]["environment"]["machine_model"] == "test-machine"
    assert manifest["tables"] == {
        "benchmarks": {"path": "data/benchmarks.csv", "row_count": 1},
        "configurations": {"path": "data/configurations.csv", "row_count": 2},
        "items": {"path": "data/items.csv", "row_count": 1},
        "suites": {"path": "data/suites.csv", "row_count": 1},
    }
    with paths["configurations"].open(newline="") as source:
        configurations = list(csv.DictReader(source))
    assert [row["status"] for row in configurations] == ["completed", "failed"]
    assert configurations[0]["score_retained_vs_q8"] == "1.0"
    assert json.loads(configurations[1]["error"])["message"] == "load failed"
    with paths["suites"].open(newline="") as source:
        assert list(csv.DictReader(source))[0]["suite"] == "A"


def test_export_experiment_artifacts_replaces_only_managed_bundle(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    first = export_experiment_artifacts(
        experiment,
        experiment_metadata={"kind": "runtime_matrix"},
    )
    (first["root"] / "stale.txt").write_text("stale", encoding="utf-8")

    second = export_experiment_artifacts(experiment)

    assert not (second["root"] / "stale.txt").exists()
    second_manifest = json.loads(second["manifest"].read_text())
    assert second_manifest["experiment_metadata"] == {"kind": "runtime_matrix"}
    assert (experiment / "experiment.json").is_file()
    assert (experiment / "models" / "model-q8" / "results.jsonl").is_file()


def test_export_experiment_artifacts_rejects_escaping_references(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    index_path = experiment / "experiment.json"
    index = json.loads(index_path.read_text())
    index["models"][0]["summary"] = "../outside.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ArtifactError, match="escapes experiment directory"):
        export_experiment_artifacts(experiment)


def test_artifacts_cli_regenerates_saved_experiment(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)

    result = CliRunner().invoke(
        app,
        ["artifacts", "--experiment", str(experiment)],
    )

    assert result.exit_code == 0
    assert "Artifact manifest:" in result.output
    assert "Configuration CSV:" in result.output
    assert (experiment / "artifacts" / "manifest.json").is_file()


def test_artifacts_cli_reports_invalid_saved_experiment(tmp_path: Path) -> None:
    experiment = tmp_path / "broken-experiment"
    experiment.mkdir()
    (experiment / "experiment.json").write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["artifacts", "--experiment", str(experiment)],
    )

    assert result.exit_code == 1
    assert "Error: experiment index must contain at least one model entry" in result.output
