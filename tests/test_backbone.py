import json
from pathlib import Path

import pytest
import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import DatasetItem, load_suite, score_answer
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
    exact_set = score_answer(set_item, "urgent,billing")
    assert exact_set.passed
    assert exact_set.details["precision"] == 1.0
    assert exact_set.details["recall"] == 1.0
    assert exact_set.details["f1"] == 1.0
    assert not score_answer(set_item, "billing,billing").passed

    behavior_item = _item(
        "behavior_rules",
        {
            "decision": "unanswerable",
            "reference_answer": "There is not enough information to determine that.",
            "evidence_patterns": [r"not\s+enough\s+information"],
            "forbidden_patterns": [r"definitely\s+happened"],
        },
    )
    assert score_answer(behavior_item, "I cannot determine that from the data.").passed
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
    assert result.details["tool_choice_accuracy"] == 1.0
    assert result.details["argument_accuracy"] == 0.5
    assert result.details["order_ok"] is True

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


def test_routing_reports_partial_label_quality_separately_from_exact_match() -> None:
    item = _item(
        "set_match",
        ["billing", "urgent"],
        parameters={"case_sensitive": False},
    )

    result = score_answer(item, "BILLING")

    assert not result.passed
    assert result.details["exact_match"] is False
    assert result.details["precision"] == 1.0
    assert result.details["recall"] == 0.5
    assert result.details["f1"] == pytest.approx(2 / 3)
    assert result.details["jaccard"] == 0.5


def test_tool_grading_separates_parseability_and_execution_components() -> None:
    item = _item(
        "tool_trace",
        {
            "calls": [
                {"tool": "schedule_timer", "arguments": {"seconds": 300}}
            ],
            "observations": [{"timer_id": "T-5"}],
            "final_state": {"timer_id": "T-5", "seconds": 300},
        },
        contract_type="json",
    )

    recovered_call = score_answer(
        item,
        '```json\n{"tool":"schedule_timer","arguments":{"seconds":"300"}}\n```',
    )
    assert not recovered_call.passed
    assert recovered_call.details["parseable"] is True
    assert recovered_call.details["tool_choice_accuracy"] == 1.0
    assert recovered_call.details["argument_accuracy"] == 1.0
    assert recovered_call.details["observations_ok"] is False
    assert recovered_call.details["final_state_ok"] is False
    assert "markdown_fence" in recovered_call.details["protocol_violations"]

    complete = score_answer(
        item,
        json.dumps(
            {
                **item.expected["value"],
                "final_state": {
                    "timer_id": "T-5",
                    "seconds": 300,
                    "message": "Timer created",
                },
            }
        ),
    )
    assert complete.passed
    assert complete.details["integration_success"] is True

    malformed = score_answer(item, '{"tool":"a"}{"tool":"b"}')
    assert not malformed.passed
    assert malformed.details["parseable"] is False
    assert malformed.details["parse_status"] == "ambiguous"


def test_single_turn_tool_call_scores_tools_arguments_no_tool_and_format() -> None:
    weather = _item(
        "tool_call",
        {
            "tool_call": "get_weather",
            "arguments": {"location": "Pune", "unit": "celsius"},
        },
        contract_type="json",
    )
    correct = score_answer(
        weather,
        '{"tool_call":"get_weather","arguments":{"location":"Pune","unit":"celsius"}}',
    )
    assert correct.passed
    assert correct.details["tool_choice_accuracy"] == 1.0
    assert correct.details["argument_accuracy"] == 1.0

    wrong_arguments = score_answer(
        weather,
        '{"tool_call":"get_weather","arguments":{"location":"Pune","unit":"fahrenheit"}}',
    )
    assert not wrong_arguments.passed
    assert wrong_arguments.details["tool_choice_accuracy"] == 1.0
    assert wrong_arguments.details["argument_accuracy"] == 0.5

    fenced = score_answer(
        weather,
        '```json\n{"tool_call":"get_weather","arguments":{"location":"Pune","unit":"celsius"}}\n```',
    )
    assert not fenced.passed
    assert fenced.details["argument_accuracy"] == 1.0
    assert fenced.details["format_compliant"] is False

    arithmetic = _item(
        "tool_call",
        {"tool_call": None, "arguments": {}, "answer": "154"},
        contract_type="json",
    )
    assert score_answer(
        arithmetic,
        '{"tool_call":null,"arguments":{},"answer":"154"}',
    ).passed
    unnecessary = score_answer(
        arithmetic,
        '{"tool_call":"web_search","arguments":{"query":"89+65"}}',
    )
    assert not unnecessary.passed
    assert unnecessary.details["tool_choice_accuracy"] == 0.0


def test_tool_call_rejects_wrong_types_and_extra_nested_arguments() -> None:
    nested = _item(
        "tool_call",
        {
            "tool_call": "search_users",
            "arguments": {
                "filters": {"status": "active", "team": "Data"},
                "fields": ["email"],
            },
        },
        contract_type="json",
    )

    extra_nested_key = score_answer(
        nested,
        json.dumps(
            {
                "tool_call": "search_users",
                "arguments": {
                    "filters": {
                        "status": "active",
                        "team": "Data",
                        "role": "admin",
                    },
                    "fields": ["email"],
                },
            }
        ),
    )
    assert not extra_nested_key.passed
    assert extra_nested_key.details["argument_accuracy"] == 0.5

    timer = _item(
        "tool_call",
        {"tool_call": "schedule_timer", "arguments": {"seconds": 300, "label": "tea"}},
        contract_type="json",
    )
    wrong_numeric_type = score_answer(
        timer,
        '{"tool_call":"schedule_timer","arguments":{"seconds":"300","label":"tea"}}',
    )
    assert not wrong_numeric_type.passed
    assert wrong_numeric_type.details["argument_accuracy"] == 0.5


def test_tool_call_accepts_item_specific_equivalent_direct_answers() -> None:
    item = _item(
        "tool_call",
        {
            "tool_call": None,
            "arguments": {},
            "answer": "No event created because rain is expected.",
        },
        contract_type="json",
        parameters={
            "direct_answer_patterns": [
                r"(?i)rain",
                r"(?i)(?:no event (?:was )?created|did not create (?:the )?event|skip(?:ped)? (?:creating )?(?:the )?event)",
            ]
        },
    )

    equivalent = score_answer(
        item,
        json.dumps(
            {
                "tool_call": None,
                "arguments": {},
                "answer": "I skipped the event because the forecast reports rain.",
            }
        ),
    )
    assert equivalent.passed
    assert equivalent.details["direct_answer_pattern_matches"] == [True, True]

    contradictory = score_answer(
        item,
        json.dumps(
            {
                "tool_call": None,
                "arguments": {},
                "answer": "The event was created despite the rain.",
            }
        ),
    )
    assert not contradictory.passed
    assert contradictory.details["direct_answer_pattern_matches"] == [True, False]


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
    assert record["integration_outcome"] == "scored_cleanly"


def test_runner_sends_tool_definitions_as_a_single_turn_system_message(
    tmp_path: Path,
) -> None:
    item = _item(
        "tool_call",
        {
            "tool_call": "get_weather",
            "arguments": {"location": "Pune", "unit": "celsius"},
        },
        contract_type="json",
    ).model_dump(mode="json")
    item["conversation"] = [
        {
            "role": "system",
            "content": (
                "Tools: get_weather(location, unit), web_search(query). "
                "Return one raw JSON tool call."
            ),
        },
        {"role": "user", "content": "What is the weather in Pune? Use Celsius."},
    ]
    config_path, _ = _write_single_item_run(tmp_path, item)

    class ToolCallBackend:
        calls = 0

        def generate_messages(self, messages, generation, *, seed):
            self.calls += 1
            assert messages[0]["role"] == "system"
            assert "get_weather" in messages[0]["content"]
            assert messages[-1] == {
                "role": "user",
                "content": "What is the weather in Pune? Use Celsius.",
            }
            return GenerationOutput(
                text=(
                    '{"tool_call":"get_weather","arguments":'
                    '{"location":"Pune","unit":"celsius"}}'
                ),
                output_tokens=12,
            )

    backend = ToolCallBackend()
    run = run_benchmark(
        load_config(config_path),
        config_path,
        backend_factory=lambda model, path, seed: backend,
    )
    record = json.loads((run / "results.jsonl").read_text())
    assert backend.calls == 1
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["argument_accuracy"] == 1.0


def test_runner_scores_the_next_action_after_a_prefilled_tool_response(
    tmp_path: Path,
) -> None:
    source = next(
        item
        for item in load_suite(Path("data/suites/final_deterministic.yaml")).items["tool_use"]
        if item.subcategory == "second_tool_required"
        and item.expected["value"]["tool_call"] == "get_weather_coordinates"
    )
    item = source.model_dump(mode="json")
    item["difficulty"] = "easy"
    config_path, _ = _write_single_item_run(tmp_path, item)

    class SecondToolBackend:
        def generate_messages(self, messages, generation, *, seed):
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
                "user",
            ]
            assert '"tool_call":"geocode"' in messages[2]["content"]
            assert '"latitude":28.6129' in messages[3]["content"]
            return GenerationOutput(
                text=json.dumps(source.expected["value"], separators=(",", ":")),
                output_tokens=20,
            )

    run = run_benchmark(
        load_config(config_path),
        config_path,
        backend_factory=lambda model, path, seed: SecondToolBackend(),
    )
    record = json.loads((run / "results.jsonl").read_text())
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["tool_choice_accuracy"] == 1.0


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
    assert record["integration_outcome"] == "scored_after_recovery"
    assert record["evaluation"]["score"] == 1.0
    assert record["evaluation"]["details"]["content_exact"] is True
    assert totals["scored"] == 1
    assert totals["integration_failures"] == 0
    assert totals["integration_friction_rate"] == 1.0
    assert totals["recovery_rate"] == 1.0
    assert totals["recoverable_friction_rate"] == 1.0
    assert totals["semantic_pass_rate"] == 1.0
    assert totals["protocol_compliance_rate"] == 0.0
    assert totals["pass_rate"] == 1.0
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
