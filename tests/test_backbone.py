import json
from pathlib import Path

import pytest
import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import DatasetItem, score_answer
from llm_workload_benchmark.runner import GenerationOutput, run_benchmark


def _item(
    method: str,
    expected,
    *,
    contract_type: str = "text",
    parameters: dict | None = None,
) -> DatasetItem:
    return DatasetItem.model_validate(
        {
            "id": f"backbone_{method}_001",
            "benchmark": "backbone_test",
            "subcategory": "contract",
            "difficulty": "easy",
            "split": "dev",
            "prompt": "Return the requested answer.",
            "response_contract": {"type": contract_type, "format": None},
            "expected": {"value": expected},
            "scoring": {"method": method, "parameters": parameters or {}},
            "provenance": {"kind": "hand_authored", "review_status": "draft"},
        }
    )


def test_reusable_set_behavior_tool_and_confidence_evaluators() -> None:
    set_item = _item("set_match", ["billing", "urgent"])
    assert score_answer(set_item, "urgent,billing").passed
    assert not score_answer(set_item, "billing,billing").passed

    behavior_item = _item(
        "behavior_rules",
        {
            "label": "abstain",
            "required_any": ["cannot verify", "not enough information"],
            "forbidden": ["definitely happened"],
        },
    )
    assert score_answer(behavior_item, "I cannot verify that claim.").passed
    assert not score_answer(behavior_item, "It definitely happened.").passed

    tool_item = _item(
        "tool_trace",
        {
            "calls": [
                {"tool": "get_weather", "arguments": {"city": "Bengaluru"}},
                {"tool": "create_event", "arguments": {"time": "07:00"}},
            ],
            "final_state": {"event_created": True},
        },
        contract_type="json",
    )
    correct_trace = json.dumps(tool_item.expected["value"])
    assert score_answer(tool_item, correct_trace).passed
    wrong_trace = json.loads(correct_trace)
    wrong_trace["calls"][1]["arguments"]["time"] = "08:00"
    result = score_answer(tool_item, json.dumps(wrong_trace))
    assert not result.passed
    assert 0 < result.score < 1

    confidence_item = _item(
        "confidence_value",
        {"answer": 120},
        parameters={"answer_type": "numeric", "absolute_tolerance": 0},
    )
    result = score_answer(confidence_item, "120\nconfidence: 80")
    assert result.passed
    assert result.details["confidence"] == 80
    assert result.details["brier_component"] == pytest.approx(0.04)


def test_deterministic_scorers_apply_generic_semantic_normalization() -> None:
    text_item = _item(
        "exact_match",
        "Window seats",
        parameters={"case_sensitive": False},
    )
    assert score_answer(text_item, "WINDOW-SEATS!").passed

    number_item = _item(
        "numeric_tolerance",
        1234567,
        contract_type="number",
        parameters={"absolute_tolerance": 0},
    )
    numeric = score_answer(number_item, "₹12,34,567")
    assert numeric.passed
    assert "remove_currency_symbol" in numeric.details["normalization_steps"]

    set_item = _item(
        "set_match",
        ["Billing", "Urgent"],
        parameters={"case_sensitive": False},
    )
    assert score_answer(set_item, "urgent!, BILLING").passed

    confidence_item = _item(
        "confidence_value",
        {"answer": False},
        parameters={"answer_type": "exact", "case_sensitive": False},
    )
    confidence = score_answer(confidence_item, "FALSE\n85")
    assert confidence.passed
    assert confidence.details["confidence"] == 85
    assert confidence.details["protocol_violations"] == [
        "missing_confidence_label"
    ]


def _write_single_item_run(
    tmp_path: Path,
    item: dict,
    *,
    response_cleanup: str = "none",
) -> tuple[Path, Path]:
    data = tmp_path / "data"
    benchmark = data / "benchmark"
    suites = data / "suites"
    benchmark.mkdir(parents=True)
    suites.mkdir()
    (benchmark / "items.jsonl").write_text(json.dumps(item) + "\n")
    (benchmark / "benchmark.yaml").write_text(
        yaml.safe_dump(
            {
                "id": item["benchmark"], "title": "Backbone", "description": "Test",
                "suite": "E", "status": "started", "execution_mode": "multi_turn",
                "task_types": ["memory"], "metrics": ["accuracy"],
                "evaluation_policy": {
                    "primary_outcome": "semantic",
                    "primary_metric": "semantic_pass_rate",
                    "protocol_requirement": "diagnostic",
                    "partial_credit_metric": "mean_semantic_score",
                },
                "items_path": "items.jsonl", "current_question_count": 1,
                "target_question_count": 1,
                "current_difficulty_distribution": {"easy": 1, "medium": 0, "hard": 0},
                "difficulty_distribution": {"easy": 1, "medium": 0, "hard": 0},
                "order_rule": "easy_to_hard", "scoring_methods": [item["scoring"]["method"]],
            },
            sort_keys=False,
        )
    )
    suite_path = suites / "suite.yaml"
    suite_path.write_text(
        "schema_version: 1\nname: backbone\nversion: 1\nstatus: pilot\n"
        "benchmark_files: [../benchmark/benchmark.yaml]\n"
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark": {
                    "name": "backbone", "workload_path": str(suite_path),
                    "output_root": str(tmp_path / "runs"),
                },
                "models": [{
                    "id": "fake",
                    "backend": "llama_cpp",
                    "model_path": str(model_path),
                    "response_cleanup": response_cleanup,
                }],
            },
            sort_keys=False,
        )
    )
    return config_path, suite_path


def test_runner_supports_message_history_and_suite_confidence_intervals(tmp_path: Path) -> None:
    item = _item("exact_match", "blue").model_dump(mode="json")
    item["conversation"] = [
        {"role": "user", "content": "Remember that the colour is blue."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What was the colour?"},
    ]
    config_path, _ = _write_single_item_run(tmp_path, item)

    class ConversationBackend:
        def generate_messages(self, messages, generation, *, seed):
            assert messages[-1]["content"] == "What was the colour?"
            return GenerationOutput(text="blue", output_tokens=1)

    run = run_benchmark(
        load_config(config_path), config_path,
        backend_factory=lambda model, path, seed: ConversationBackend(),
    )
    record = json.loads((run / "results.jsonl").read_text())
    assert record["suite"] == "E"
    assert record["integration_outcome"] == "scored"


def test_json_fence_is_integration_friction_not_a_wrong_scorable_answer(tmp_path: Path) -> None:
    item = _item(
        "json_exact", {"answer": 7}, contract_type="json",
        parameters={"allow_diagnostic_normalization": True},
    ).model_dump(mode="json")
    config_path, _ = _write_single_item_run(tmp_path, item)

    class FencedBackend:
        def generate(self, prompt, generation, *, seed):
            return GenerationOutput(text='```json\n{"answer":7}\n```', output_tokens=5)

    run = run_benchmark(
        load_config(config_path), config_path,
        backend_factory=lambda model, path, seed: FencedBackend(),
    )
    record = json.loads((run / "results.jsonl").read_text())
    totals = json.loads((run / "summary.json").read_text())["totals"]
    assert record["integration_outcome"] == "markdown_fence"
    assert record["evaluation"]["score"] == 1.0
    assert record["evaluation"]["details"]["content_exact"] is True
    assert totals["scored"] == 1
    assert totals["integration_failures"] == 1
    assert totals["integration_friction_rate"] == 1.0
    assert totals["pass_rate"] == 0.0
    assert totals["mean_score"] == 1.0


def test_runner_executes_fake_tool_results_between_model_turns(tmp_path: Path) -> None:
    expected = {
        "calls": [
            {"tool": "get_weather", "arguments": {"city": "Bengaluru"}},
            {"tool": "create_event", "arguments": {"time": "07:00"}},
        ],
        "observations": [
            {"rain": False},
            {"event_id": "evt-1"},
        ],
        "final_state": {"event_created": True},
    }
    item = _item("tool_trace", expected, contract_type="json").model_dump(mode="json")
    config_path, _ = _write_single_item_run(tmp_path, item)

    class ToolBackend:
        turn = 0

        def generate_messages(self, messages, generation, *, seed):
            replies = [
                '{"tool":"get_weather","arguments":{"city":"Bengaluru"}}',
                '{"tool":"create_event","arguments":{"time":"07:00"}}',
                '{"final_state":{"event_created":true}}',
            ]
            if self.turn == 1:
                assert '"rain": false' in messages[-1]["content"]
            self.turn += 1
            return GenerationOutput(text=replies[self.turn - 1], output_tokens=5)

    run = run_benchmark(
        load_config(config_path), config_path,
        backend_factory=lambda model, path, seed: ToolBackend(),
    )
    record = json.loads((run / "results.jsonl").read_text())
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["observations_ok"] is True


def test_runner_cleans_think_blocks_before_parsing_tool_actions(tmp_path: Path) -> None:
    expected = {
        "calls": [{"tool": "get_weather", "arguments": {"city": "Pune"}}],
        "observations": [{"temperature_c": 29}],
        "final_state": {"temperature_c": 29},
    }
    item = _item("tool_trace", expected, contract_type="json").model_dump(mode="json")
    config_path, _ = _write_single_item_run(
        tmp_path,
        item,
        response_cleanup="strip_think",
    )

    class ThinkingToolBackend:
        turn = 0

        def generate_messages(self, messages, generation, *, seed):
            replies = [
                '<think>choose the weather tool</think>\n'
                '{"tool":"get_weather","arguments":{"city":"Pune"}}',
                '<think>use the observation</think>\n'
                '{"final_state":{"temperature_c":29}}',
            ]
            response = replies[self.turn]
            self.turn += 1
            return GenerationOutput(text=response, output_tokens=5)

    run = run_benchmark(
        load_config(config_path),
        config_path,
        backend_factory=lambda model, path, seed: ThinkingToolBackend(),
    )
    record = json.loads((run / "results.jsonl").read_text())
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["call_count_actual"] == 1
