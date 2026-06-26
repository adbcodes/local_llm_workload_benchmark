import csv
from pathlib import Path

from llm_workload_benchmark.final_figures import (
    calibration,
    deployment_risk,
    config_effects,
    context_speed,
    laptop_value_frontier,
    quant_survival,
    retrieval_depth,
    thermal_drift,
    trust_profile,
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


def test_quant_deployment_and_trust_figures_generate(tmp_path: Path) -> None:
    configs = [_config("small", 2_000_000_000, 7), _config("quality", 4_000_000_000, 9)]
    items = []
    for config in configs:
        for quant in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"):
            for benchmark, subcategory in [
                ("applied_reasoning", "math"), ("messy_text_to_schema", "clean"),
                ("tool_use", "call"), ("answer_stability", "confident_wrong_suggestion"),
                ("answer_stability", "are_you_sure_challenge"),
                ("false_missing_information", "false_premise"),
                ("over_refusal", "benign_security_terminology"),
            ]:
                items.append({
                    "variant_id": config["variant_id"], "family": "qwen3",
                    "quantization": quant, "benchmark": benchmark,
                    "subcategory": subcategory, "passed": benchmark == "applied_reasoning",
                    "integration_outcome": "scored" if benchmark != "tool_use" else "unparseable_output",
                })
    assert quant_survival(tmp_path, items)["status"] == "generated"
    assert deployment_risk(tmp_path, configs, items)["status"] == "generated"
    assert trust_profile(tmp_path, items, ["small", "quality"])["status"] == "generated"


def test_retrieval_context_and_config_figures_generate(tmp_path: Path) -> None:
    config = _config("qwen3-q4", 2_000_000_000, 8)
    config.update({"architecture": "qwen3-8b", "family": "qwen3", "quantization": "Q4_K_M"})
    retrieval = []
    for tag in ("2k_context", "4k_context", "8k_context"):
        for position in ("start", "middle", "end"):
            retrieval.append({**config, "benchmark": "long_text_retrieval",
                              "subcategory": f"fact_at_{position}", "tags": [tag], "passed": True})
    assert retrieval_depth(tmp_path, [config], retrieval)["status"] == "generated"
    context = [
        {**config, "prompt_tokens": tokens, "tags": [] if tag is None else [tag],
         "output_tokens_per_second": 20 - index, "ttft_seconds": .1 + index,
         "variant_id": "context-q4"}
        for index, (tokens, tag) in enumerate([(100, None), (2000, "2k_context"),
                                                (4000, "4k_context"), (8000, "8k_context")])
    ]
    assert context_speed(tmp_path, [config], context)["status"] == "generated"
    default = [{**config, "item_id": "probe", "passed": True,
                "benchmark": "messy_text_to_schema", "integration_outcome": "scored"}]
    tier2 = [
        {**default[0], "temperature": .7, "repeat_penalty": 1.0, "constrained_decoding": "none"},
        {**default[0], "temperature": 0.0, "repeat_penalty": 1.1, "constrained_decoding": "none"},
        {**default[0], "temperature": 0.0, "repeat_penalty": 1.0,
         "constrained_decoding": "json_when_requested"},
    ]
    assert config_effects(tmp_path, default, tier2)["status"] == "generated"


def test_conditional_figures_report_gate_evidence(tmp_path: Path) -> None:
    thin = calibration(tmp_path, [])
    assert thin["status"] == "skipped"
    assert len(thin["thin_bins"]) == 15
    stable = [
        {"variant_id": "stable", "family": "qwen3", "run_order": index + 1,
         "output_tokens_per_second": 20.0}
        for index in range(30)
    ]
    thermal = thermal_drift(tmp_path, stable)
    assert thermal["status"] == "skipped"
    assert thermal["diagnostics"][0]["predicted_drop_percent"] == 0
