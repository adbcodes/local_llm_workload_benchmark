import json
from pathlib import Path

import pytest

from llm_workload_benchmark.telemetry import RuntimeTelemetry


def test_runtime_telemetry_saves_graph_ready_samples_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_workload_benchmark.telemetry._current_process_rss_bytes",
        lambda: 2_000_000,
    )
    monkeypatch.setattr(
        "llm_workload_benchmark.telemetry.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "llm_workload_benchmark.telemetry._apple_gpu_metrics",
        lambda: {
            "system_gpu_utilization_percent": 75.0,
            "gpu_allocated_system_memory_bytes": 1_000_000,
        },
    )
    monkeypatch.setattr(
        "llm_workload_benchmark.telemetry._can_sample_powermetrics",
        lambda: False,
    )
    output = tmp_path / "telemetry.jsonl"
    telemetry = RuntimeTelemetry(output, interval_seconds=60)

    telemetry.start()
    telemetry._sample()
    summary = telemetry.stop()

    samples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(samples) == 3
    assert all(sample["process_rss_bytes"] == 2_000_000 for sample in samples)
    assert summary["peak_sampled_process_rss_bytes"] == 2_000_000
    assert summary["mean_system_gpu_utilization_percent"] == 75.0
    assert summary["sensor_status"]["process_cpu"] == "available"
