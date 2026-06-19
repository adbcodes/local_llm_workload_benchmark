import json
from pathlib import Path

import pytest

from llm_workload_benchmark.report import ReportError, generate_comparison_report


def _write_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "matrix-test"
    summary_directory = experiment / "models" / "model-a"
    summary_directory.mkdir(parents=True)
    (summary_directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "totals": {
                    "attempted": 7,
                    "passed": 5,
                    "pass_rate": 5 / 7,
                    "mean_score": 0.8,
                    "mean_latency_seconds": 0.25,
                    "mean_time_to_first_token_seconds": 0.1,
                    "mean_output_tokens_per_second_end_to_end": 24.5,
                    "peak_process_memory_bytes": 2 * 1024**3,
                },
            }
        ),
        encoding="utf-8",
    )
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "matrix-test",
                "status": "partial_failure",
                "dataset": "data/benchmarks/v1/suite.yaml",
                "models": [
                    {
                        "model_id": "model-a",
                        "status": "completed",
                        "summary": "models/model-a/summary.json",
                    },
                    {
                        "model_id": "model-b",
                        "status": "failed",
                        "summary": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return experiment


def test_generate_comparison_report_from_saved_summaries(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)

    report_path = generate_comparison_report(experiment)

    assert report_path == experiment / "comparison.md"
    report = report_path.read_text(encoding="utf-8")
    assert "# Model Comparison" in report
    assert "| model-a | completed | 5/7 | 71.4% | 80.0% |" in report
    assert "| model-b | failed | — | — | — |" in report
    assert "2.00 GiB" in report
    assert "not model rankings" in report


def test_report_rejects_summary_path_outside_experiment(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    index_path = experiment / "experiment.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["models"][0]["summary"] = "../outside.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ReportError, match="escapes experiment directory"):
        generate_comparison_report(experiment)
