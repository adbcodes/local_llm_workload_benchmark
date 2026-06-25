import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from typer.testing import CliRunner

from llm_workload_benchmark.artifacts import ArtifactError, export_experiment_artifacts
from llm_workload_benchmark.cli import app
from llm_workload_benchmark.plots import _pareto_frontier


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
        "peak_process_memory_bytes": 2 * 1024**3,
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
                "peak_process_memory_bytes": 2 * 1024**3,
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


def _write_quantization_experiment(tmp_path: Path) -> Path:
    experiment = _write_experiment(tmp_path)
    q8_summary_path = experiment / "models" / "model-q8" / "summary.json"
    q8_summary = json.loads(q8_summary_path.read_text())
    q4_summary = json.loads(q8_summary_path.read_text())
    q4_summary["model"] = {
        **q4_summary["model"],
        "id": "model-q4",
        "quantization": "Q4_K_M",
        "file_size_bytes": 600,
    }
    for aggregate in [
        q4_summary["totals"],
        q4_summary["suites"]["A"],
        q4_summary["benchmarks"]["reasoning"]["overall"],
    ]:
        aggregate["passed"] = 0
        aggregate["pass_rate"] = 0.0
        aggregate["mean_score"] = 0.75
        aggregate["peak_process_memory_bytes"] = int(1.2 * 1024**3)
        aggregate["mean_output_tokens_per_second_end_to_end"] = 30.0
    q4_summary["benchmarks"]["reasoning"]["reported_score"] = 0.75
    q4_directory = experiment / "models" / "model-q4"
    (q4_directory / "summary.json").write_text(
        json.dumps(q4_summary),
        encoding="utf-8",
    )
    (q4_directory / "results.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark": "reasoning",
                "suite": "A",
                "item_id": "reasoning_001",
                "difficulty": "easy",
                "repetition": 1,
                "evaluation": {"passed": False, "score": 0.75},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = experiment / "experiment.json"
    index = json.loads(index_path.read_text())
    index["status"] = "completed"
    index["models"][1].update(
        {
            "status": "completed",
            "summary": "models/model-q4/summary.json",
            "error": None,
        }
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return experiment


def test_export_experiment_artifacts_writes_normalized_partial_results(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)

    paths = export_experiment_artifacts(experiment)

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["experiment_status"] == "partial_failure"
    assert manifest["machine"]["environment"]["machine_model"] == "test-machine"
    assert manifest["plots"]["quantization_survival"]["status"] == "skipped"
    assert manifest["plots"]["workload_fit_heatmap"]["status"] == "skipped"
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


def test_export_experiment_artifacts_writes_quantization_plot_and_data(
    tmp_path: Path,
) -> None:
    experiment = _write_quantization_experiment(tmp_path)

    paths = export_experiment_artifacts(experiment)

    manifest = json.loads(paths["manifest"].read_text())
    plot = manifest["plots"]["quantization_survival"]
    assert plot["status"] == "generated"
    assert plot["row_count"] == 2
    assert plot["series_count"] == 1
    png_path = paths["root"] / plot["png"]
    svg_path = paths["root"] / plot["svg"]
    data_path = paths["root"] / plot["data"]
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert ET.parse(svg_path).getroot().tag.endswith("svg")
    with data_path.open(newline="") as source:
        plot_rows = list(csv.DictReader(source))
    assert [row["quantization_label"] for row in plot_rows] == ["Q8", "Q4"]
    assert [row["score_percent"] for row in plot_rows] == ["100.0", "75.0"]


def test_export_experiment_artifacts_writes_quality_frontiers(
    tmp_path: Path,
) -> None:
    experiment = _write_quantization_experiment(tmp_path)

    paths = export_experiment_artifacts(experiment)

    manifest = json.loads(paths["manifest"].read_text())
    for plot_id in ["memory_quality_frontier", "speed_quality_frontier"]:
        plot = manifest["plots"][plot_id]
        assert plot["status"] == "generated"
        assert plot["row_count"] == 2
        assert plot["pareto_count"] == 2
        assert (paths["root"] / plot["png"]).read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        assert ET.parse(paths["root"] / plot["svg"]).getroot().tag.endswith("svg")
        with (paths["root"] / plot["data"]).open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 2
        assert {row["is_pareto"] for row in rows} == {"True"}


def test_pareto_frontier_excludes_dominated_configurations() -> None:
    rows = [
        {"variant_id": "balanced", "cost": 2.0, "score_percent": 80.0},
        {"variant_id": "dominated", "cost": 3.0, "score_percent": 70.0},
        {"variant_id": "quality", "cost": 4.0, "score_percent": 90.0},
    ]

    frontier = _pareto_frontier(rows, x_field="cost", minimize_x=True)

    assert {row["variant_id"] for row in frontier} == {"balanced", "quality"}


def test_export_experiment_artifacts_writes_workload_fit_heatmap(
    tmp_path: Path,
) -> None:
    experiment = _write_quantization_experiment(tmp_path)
    q8_summary_path = experiment / "models" / "model-q8" / "summary.json"
    q8_summary = json.loads(q8_summary_path.read_text())
    missing_skill = dict(q8_summary["suites"]["A"])
    missing_skill.update({"passed": 0, "pass_rate": 0.0, "mean_score": 0.0})
    q8_summary["suites"]["B"] = missing_skill
    q8_summary_path.write_text(json.dumps(q8_summary), encoding="utf-8")

    paths = export_experiment_artifacts(experiment)

    manifest = json.loads(paths["manifest"].read_text())
    assert set(manifest["plots"]) == {
        "quantization_survival",
        "memory_quality_frontier",
        "speed_quality_frontier",
        "workload_fit_heatmap",
    }
    plot = manifest["plots"]["workload_fit_heatmap"]
    assert plot["status"] == "generated"
    assert plot["configuration_count"] == 2
    assert plot["suite_count"] == 2
    assert plot["observed_count"] == 3
    assert plot["missing_count"] == 1
    assert (paths["root"] / plot["png"]).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert ET.parse(paths["root"] / plot["svg"]).getroot().tag.endswith("svg")
    with (paths["root"] / plot["data"]).open(newline="") as source:
        rows = list(csv.DictReader(source))
    zero = next(
        row for row in rows
        if row["variant_id"] == "model-q8" and row["suite"] == "B"
    )
    missing = next(
        row for row in rows
        if row["variant_id"] == "model-q4" and row["suite"] == "B"
    )
    assert zero["available"] == "True"
    assert zero["score_percent"] == "0.0"
    assert missing["available"] == "False"
    assert missing["score_percent"] == ""


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


def test_plot_failure_preserves_previous_artifact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_quantization_experiment(tmp_path)
    first = export_experiment_artifacts(experiment)
    original_manifest = first["manifest"].read_text()

    def fail_plot(*args, **kwargs):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(
        "llm_workload_benchmark.artifacts.generate_plots",
        fail_plot,
    )

    with pytest.raises(ArtifactError, match="renderer failed"):
        export_experiment_artifacts(experiment)

    assert first["manifest"].read_text() == original_manifest
    assert (first["root"] / "plots" / "quantization-survival.png").is_file()


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
