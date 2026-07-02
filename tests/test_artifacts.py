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
        "sha256": "a" * 64,
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
        "semantic_pass_rate": 1.0,
        "mean_semantic_score": 1.0,
        "protocol_compliance_rate": 1.0,
        "mean_protocol_score": 1.0,
        "integration_success_rate": 1.0,
        "mean_integration_score": 1.0,
        "integration_parse_rate": 1.0,
        "recovery_rate": 0.0,
        "recoverable_friction_rate": 0.0,
        "parse_failure_rate": 0.0,
        "latency_seconds": 0.2,
        "mean_latency_seconds": 0.2,
        "mean_time_to_first_token_seconds": 0.05,
        "mean_output_tokens_per_second_end_to_end": 20.0,
        "output_tokens_per_second_end_to_end": 18.0,
        "prompt_tokens_per_second": 200.0,
        "prompt_eval_tokens": 20,
        "prompt_cached_tokens": 3,
        "prompt_eval_seconds": 0.1,
        "decode_eval_tokens": 4,
        "decode_eval_seconds": 0.16,
        "decode_graphs_reused": 2,
        "decode_tokens_per_second": 25.0,
        "generation_process_cpu_seconds": 0.15,
        "peak_process_rss_bytes": 2 * 1024**3,
        "peak_process_memory_bytes": 2 * 1024**3,
        "integration_friction_rate": 0.0,
    }
    (completed / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model": model,
                "model_load_seconds": 0.1,
                "warmup": {
                    "performed": True,
                    "excluded_from_results": True,
                    "latency_seconds": 0.02,
                },
                "resume": {
                    "segment": 1,
                    "resumed_from_items": 0,
                    "checkpoint_granularity": "item",
                },
                "process_rss_before_model_load_bytes": 100,
                "process_rss_after_model_load_bytes": 800,
                "model_load_rss_delta_bytes": 700,
                "peak_process_rss_bytes": 2 * 1024**3,
                "peak_process_rss_delta_from_preload_bytes": 2 * 1024**3 - 100,
                "totals": aggregate,
                "suites": {"A": aggregate},
                "benchmarks": {
                    "reasoning": {
                        "overall": aggregate,
                        "reported_score": 1.0,
                        "score_formula": "mean_score",
                    }
                },
                "telemetry": {
                    "sample_count": 2,
                    "elapsed_seconds": 0.35,
                    "mean_system_power_watts": 5.0,
                    "sensor_status": {"temperature_and_power": "available"},
                },
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
                "run_order": 1,
                "variant_of": None,
                "tags": ["short"],
                "response_contract": {"type": "number", "format": None},
                "scoring_method": "numeric_tolerance",
                "difficulty": "easy",
                "repetition": 1,
                "evaluation": {
                    "passed": True,
                    "score": 1.0,
                    "semantic_outcome": "correct",
                    "semantic_score": 1.0,
                    "protocol_outcome": "compliant",
                    "protocol_score": 1.0,
                    "protocol_violations": [],
                    "integration_outcome": "scored_cleanly",
                    "integration_score": 1.0,
                },
                "latency_seconds": 0.2,
                "generation_latency_seconds": 0.2,
                "time_to_first_token_seconds": 0.05,
                "prompt_eval_tokens": 20,
                "prompt_cached_tokens": 3,
                "prompt_eval_seconds": 0.1,
                "prompt_tokens_per_second": 200.0,
                "decode_eval_tokens": 4,
                "decode_eval_seconds": 0.16,
                "decode_graphs_reused": 2,
                "decode_tokens_per_second": 25.0,
                "output_tokens_per_second_end_to_end": 20.0,
                "item_rss_before_generation_bytes": 800,
                "item_peak_process_rss_bytes": 2 * 1024**3,
                "item_peak_rss_delta_from_post_load_bytes": 2 * 1024**3 - 800,
                "process_rss_before_model_load_bytes": 100,
                "process_rss_after_model_load_bytes": 800,
                "model_load_rss_delta_bytes": 700,
                "resume_segment": 1,
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
                "elapsed_seconds": 1.5,
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
        aggregate["peak_process_rss_bytes"] = int(1.2 * 1024**3)
        aggregate["mean_output_tokens_per_second_end_to_end"] = 30.0
        aggregate["output_tokens_per_second_end_to_end"] = 28.0
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
    assert manifest["schema_version"] == 3
    assert manifest["experiment_status"] == "partial_failure"
    assert manifest["experiment_elapsed_seconds"] == 1.5
    assert manifest["machine"]["environment"]["machine_model"] == "test-machine"
    assert manifest["plots"] == {}
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
    assert configurations[0]["pass_rate_ci_95_low"]
    assert configurations[0]["estimated_generation_energy_joules"] == "1.0"
    assert configurations[0]["energy_per_correct_answer_joules"] == "1.0"
    assert configurations[0]["run_elapsed_seconds"] == "0.35"
    assert configurations[0]["item_latency_seconds_total"] == "0.2"
    assert configurations[0]["model_sha256"] == "a" * 64
    assert configurations[0]["warmup_performed"] == "True"
    assert configurations[0]["checkpoint_granularity"] == "item"
    assert configurations[0]["output_tokens_per_second_end_to_end"] == "18.0"
    assert configurations[0]["prompt_tokens_per_second"] == "200.0"
    assert configurations[0]["prompt_cached_tokens"] == "3"
    assert configurations[0]["decode_graphs_reused"] == "2"
    assert configurations[0]["decode_tokens_per_second"] == "25.0"
    assert configurations[0]["model_load_rss_delta_bytes"] == "700"
    assert json.loads(configurations[0]["sensor_status"])[
        "temperature_and_power"
    ] == "available"
    assert json.loads(configurations[1]["error"])["message"] == "load failed"
    with paths["suites"].open(newline="") as source:
        suite = list(csv.DictReader(source))[0]
    assert suite["suite"] == "A"
    assert suite["latency_seconds_total"] == "0.2"
    with paths["items"].open(newline="") as source:
        item = list(csv.DictReader(source))[0]
    assert item["run_order"] == "1"
    assert json.loads(item["tags"]) == ["short"]
    assert json.loads(item["response_contract"])["type"] == "number"
    assert item["semantic_outcome"] == "correct"
    assert item["protocol_outcome"] == "compliant"
    assert item["integration_outcome"] == "scored_cleanly"
    assert item["generation_latency_seconds"] == "0.2"
    assert item["prompt_tokens_per_second"] == "200.0"
    assert item["prompt_cached_tokens"] == "3"
    assert item["decode_graphs_reused"] == "2"
    assert item["decode_tokens_per_second"] == "25.0"
    assert item["output_tokens_per_second_end_to_end"] == "20.0"
    assert item["item_peak_process_rss_bytes"] == str(2 * 1024**3)
    assert item["resume_segment"] == "1"


def test_artifact_pass_rate_counts_errors_for_older_saved_summaries(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "models" / "model-q8" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for aggregate in (
        summary["totals"],
        summary["suites"]["A"],
        summary["benchmarks"]["reasoning"]["overall"],
    ):
        aggregate.update(attempted=2, completed=1, passed=1, pass_rate=1.0)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    paths = export_experiment_artifacts(experiment)

    with paths["configurations"].open(newline="") as source:
        configuration = list(csv.DictReader(source))[0]
    assert configuration["pass_rate"] == "0.5"
    assert float(configuration["pass_rate_ci_95_high"]) < 1.0


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


def test_figures_cli_writes_compact_manifest(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "figures",
            "--five-experiment", str(experiment),
            "--retrieval-experiment", str(experiment),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Final figure manifest:" in result.output
    manifest_path = experiment / "artifacts" / "final_figures" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["plots"]) == 8
    assert set(manifest["sources"]) == {"five_workloads", "retrieval"}
