import csv
from pathlib import Path

from llm_workload_benchmark.final_figures import (
    constraint_content_retention,
    constraint_load_score,
    context_ttft,
    quantization_benchmark_score,
    quantization_tool_trace_parseability,
)


def _item(
    family: str,
    quantization: str,
    benchmark: str,
    *,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "family": family,
        "quantization": quantization,
        "benchmark": benchmark,
        "score": score,
        "integration_outcome": "scored",
    }


def test_quantization_score_writes_one_metric_rows(tmp_path: Path) -> None:
    rows = [
        _item(family, quantization, "applied_reasoning", score=score)
        for family, score in (("qwen2.5", 0.5), ("gemma3", 0.6), ("qwen3", 0.8))
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M")
    ]

    result = quantization_benchmark_score(
        tmp_path,
        rows,
        benchmark="applied_reasoning",
        stem="quant_applied_reasoning",
        title="Applied Reasoning vs Quantization",
    )

    assert result["status"] == "generated"
    with (tmp_path / result["data"]).open(newline="") as source:
        data = list(csv.DictReader(source))
    assert len(data) == 12
    assert set(data[0]) == {"family", "quantization", "mean_score_percent", "n"}
    assert (tmp_path / result["png"]).is_file()
    assert (tmp_path / result["svg"]).is_file()


def test_tool_trace_figure_reports_parseability_not_correctness(tmp_path: Path) -> None:
    rows = []
    for family in ("qwen2.5", "gemma3", "qwen3"):
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"):
            rows.extend(
                [
                    {
                        **_item(family, quantization, "tool_use"),
                        "integration_outcome": "scored_after_recovery",
                    },
                    {
                        **_item(family, quantization, "tool_use"),
                        "integration_outcome": "unparseable",
                    },
                ]
            )

    result = quantization_tool_trace_parseability(tmp_path, rows)

    assert result["status"] == "generated"
    with (tmp_path / result["data"]).open(newline="") as source:
        data = list(csv.DictReader(source))
    assert {row["parseability_percent"] for row in data} == {"50.0"}
    assert all(row["attempted"] == "2" for row in data)


def test_constraint_figures_use_q4_and_cumulative_load(tmp_path: Path) -> None:
    rows = []
    for family in ("qwen2.5", "gemma3", "qwen3"):
        for count in range(1, 5):
            rows.append(
                {
                    **_item(family, "Q4_K_M", "constraint_load_curve", score=count / 10),
                    "tags": [f"constraint_load_{count}"],
                    "evaluation_details": {"content_score": count / 20},
                }
            )
            rows.append(
                {
                    **_item(family, "Q8_0", "constraint_load_curve", score=1.0),
                    "tags": [f"constraint_load_{count}"],
                    "evaluation_details": {"content_score": 1.0},
                }
            )

    score = constraint_load_score(tmp_path, rows)
    content = constraint_content_retention(tmp_path, rows)

    assert score["status"] == content["status"] == "generated"
    with (tmp_path / score["data"]).open(newline="") as source:
        score_rows = list(csv.DictReader(source))
    with (tmp_path / content["data"]).open(newline="") as source:
        content_rows = list(csv.DictReader(source))
    assert len(score_rows) == len(content_rows) == 12
    assert score_rows[0]["mean_score_percent"] == "10.0"
    assert content_rows[0]["mean_score_percent"] == "5.0"


def test_context_ttft_uses_q4_medians_only(tmp_path: Path) -> None:
    rows = []
    for family in ("qwen2.5", "gemma3", "qwen3"):
        for label, tag in (("2K", "2k_context"), ("4K", "4k_context"),
                           ("8K", "8k_context")):
            rows.extend(
                [
                    {
                        **_item(family, "Q4_K_M", "long_text_retrieval"),
                        "tags": [tag],
                        "ttft_seconds": value,
                    }
                    for value in (1.0, 3.0, 100.0)
                ]
            )
            rows.append(
                {
                    **_item(family, "Q8_0", "long_text_retrieval"),
                    "tags": [tag],
                    "ttft_seconds": 999.0,
                }
            )

    result = context_ttft(tmp_path, rows)

    assert result["status"] == "generated"
    with (tmp_path / result["data"]).open(newline="") as source:
        data = list(csv.DictReader(source))
    assert len(data) == 9
    assert {row["median_ttft_seconds"] for row in data} == {"3.0"}
    assert {row["context_length"] for row in data} == {"2K", "4K", "8K"}
