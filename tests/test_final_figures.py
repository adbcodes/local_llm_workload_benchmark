import csv
from pathlib import Path

from llm_workload_benchmark.final_figures import (
    laptop_value_frontier,
    workload_decision_matrix,
)


def _config(variant: str, memory: int, passed: int) -> dict[str, object]:
    return {
        "variant_id": variant, "architecture": variant.split("-")[0],
        "family": "qwen3", "quantization": "Q4_K_M", "attempted": 10,
        "passed": passed, "peak_process_memory_bytes": memory,
        "mean_output_tokens_per_second": 20, "energy_per_correct_answer_joules": 3,
    }


def test_frontier_and_decision_matrix_write_ci_data(tmp_path: Path) -> None:
    configurations = [
        _config("small", 2_000_000_000, 7),
        _config("dominated", 3_000_000_000, 6),
        _config("quality", 4_000_000_000, 9),
    ]
    frontier, ids = laptop_value_frontier(tmp_path, configurations)
    assert frontier["status"] == "generated"
    assert ids == ["small", "quality"]
    items = [
        {"variant_id": variant, "benchmark": benchmark, "passed": index % 2 == 0}
        for variant in ids
        for benchmark in {value for values in __import__(
            "llm_workload_benchmark.final_figures", fromlist=["WORKLOAD_GROUPS"]
        ).WORKLOAD_GROUPS.values() for value in values}
        for index in range(3)
    ]
    matrix = workload_decision_matrix(
        tmp_path, items, ids, {"small": 2.0, "quality": 4.0}
    )
    assert matrix["status"] == "generated"
    with (tmp_path / matrix["data"]).open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 18
    assert all(row["ci_low_percent"] and row["ci_high_percent"] for row in rows)
