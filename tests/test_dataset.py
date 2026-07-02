import json
import shutil
from itertools import permutations
from pathlib import Path

import pytest

from llm_workload_benchmark.dataset import (
    BenchmarkDefinition,
    DatasetError,
    load_dataset,
    load_suite,
    score_answer,
)

SUITE_PATH = Path("data/suites/core.yaml")
CONSTRAINT_PATH = Path("data/constraint_load_curve/items.jsonl")


def test_zero_count_benchmarks_can_use_an_empty_dataset(tmp_path: Path) -> None:
    definition = BenchmarkDefinition.model_validate(
        {
            "id": "planned_benchmark",
            "title": "Planned benchmark",
            "description": "An empty benchmark template.",
            "evaluation_policy": {
                "primary_outcome": "semantic",
                "primary_metric": "semantic_pass_rate",
                "protocol_requirement": "diagnostic",
                "partial_credit_metric": "mean_semantic_score",
            },
            "items_path": "items.jsonl",
            "authoring_paths": ["questions.yaml"],
            "current_question_count": 0,
            "target_question_count": 2,
            "current_difficulty_distribution": {
                "easy": 0,
                "medium": 0,
                "hard": 0,
            },
            "difficulty_distribution": {"easy": 1, "medium": 1, "hard": 0},
            "order_rule": "easy_to_hard",
            "scoring_methods": ["exact_match"],
        }
    )
    dataset_path = tmp_path / definition.items_path
    dataset_path.write_text("", encoding="utf-8")

    assert load_dataset(dataset_path, allow_empty=True) == []
    with pytest.raises(DatasetError, match="contains no items"):
        load_dataset(dataset_path)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            {
                "primary_outcome": "semantic",
                "primary_metric": "protocol_pass_rate",
                "protocol_requirement": "diagnostic",
                "partial_credit_metric": "mean_semantic_score",
            },
            "requires primary metric 'semantic_pass_rate'",
        ),
        (
            {
                "primary_outcome": "protocol",
                "primary_metric": "protocol_pass_rate",
                "protocol_requirement": "diagnostic",
                "partial_credit_metric": "mean_protocol_score",
            },
            "must require protocol compliance",
        ),
    ],
)
def test_benchmark_definition_rejects_inconsistent_evaluation_policy(
    policy: dict[str, str],
    message: str,
) -> None:
    definition = {
        "id": "invalid_policy",
        "title": "Invalid policy",
        "description": "Exercise evaluation-policy validation.",
        "evaluation_policy": policy,
        "items_path": "items.jsonl",
        "current_question_count": 0,
        "target_question_count": 1,
        "current_difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
        "difficulty_distribution": {"easy": 1, "medium": 0, "hard": 0},
        "order_rule": "easy_to_hard",
        "scoring_methods": ["exact_match"],
    }

    with pytest.raises(ValueError, match=message):
        BenchmarkDefinition.model_validate(definition)


def test_active_pilot_suite_loads_with_difficulty_progression() -> None:
    suite = load_suite(SUITE_PATH)

    assert set(suite.items) == {
        "applied_reasoning",
        "messy_text_to_schema",
    }
    assert sum(len(items) for items in suite.items.values()) == 96
    reasoning = suite.items["applied_reasoning"]
    assert len(reasoning) == 48
    assert [item.difficulty for item in reasoning].count("easy") == 8
    assert [item.difficulty for item in reasoning].count("medium") == 24
    assert [item.difficulty for item in reasoning].count("hard") == 16
    schema_items = suite.items["messy_text_to_schema"]
    assert len(schema_items) == 48
    assert [item.difficulty for item in schema_items].count("easy") == 8
    assert [item.difficulty for item in schema_items].count("medium") == 25
    assert [item.difficulty for item in schema_items].count("hard") == 15
    assert sum(item.split == "dev" for item in reasoning) == 8


def test_numeric_and_exact_answer_verifiers() -> None:
    suite = load_suite(SUITE_PATH)
    items = {item.id: item for item in suite.items["applied_reasoning"]}
    percentage = items["reason_arithmeticperc_001"]
    calendar = items["reason_calendartime_002"]
    ordering = items["reason_orderingconstr_007"]
    rational = items["reason_probabilitycou_007"]

    assert calendar.scoring.method == "date_value"
    percentage_result = score_answer(percentage, "112")
    assert percentage_result.passed
    assert percentage_result.type == "deterministic"
    assert percentage_result.evaluator == "numeric_tolerance"
    assert percentage_result.version == 2
    assert score_answer(percentage, "112.0").passed
    assert score_answer(percentage, '"112"').passed
    assert score_answer(percentage, "The answer is 112.").passed
    assert score_answer(percentage, "FINAL: 112 kg").passed
    assert score_answer(percentage, "FINAL: 112 kilograms").passed
    assert not score_answer(percentage, "FINAL: 112 lb").passed
    derivation = score_answer(
        percentage,
        "17.5 percent of 640 is 112.\nFINAL: 112",
    )
    assert derivation.passed
    assert derivation.details["answer_extraction"] == "final_marker"
    assert derivation.details["final_marker_compliant"] is True
    assert not score_answer(percentage, "113").passed
    assert not score_answer(percentage, "It could be 112 or 113.").passed
    assert not score_answer(percentage, "FINAL: 112 or 113").passed
    duplicate_final = score_answer(percentage, "FINAL: 112\nFINAL: 113")
    assert not duplicate_final.passed
    assert duplicate_final.details["reason"] == "multiple_final_answers"
    assert score_answer(calendar, "The date is 2027-05-19.").passed
    assert score_answer(calendar, "The answer is May 19, 2027.").passed
    assert score_answer(
        calendar,
        "Occurrence 1 is 2027-02-03. After five three-week intervals, "
        "the final date is 2027-05-19.\nFINAL: 2027-05-19",
    ).passed
    assert not score_answer(calendar, "The answer is 2027-05-20.").passed
    expected_order = str(ordering.expected["value"])
    assert score_answer(ordering, expected_order).passed
    assert score_answer(ordering, expected_order.replace(",", " ")).passed
    assert not score_answer(ordering, expected_order[::-1]).passed
    rational_gold = str(rational.expected["value"])
    assert score_answer(rational, rational_gold).passed
    assert score_answer(rational, f"**{rational_gold}**").passed
    assert score_answer(
        rational,
        f"Compute the conditional probability.\nFINAL: {rational_gold}",
    ).passed
    assert not score_answer(rational, "1/999").passed
    assert score_answer(ordering, expected_order.lower()).passed
    assert score_answer(ordering, expected_order.lower().replace(",", " ")).passed


def test_reasoning_and_order_gold_answers_are_independently_derived() -> None:
    suite = load_suite(SUITE_PATH)
    items = {item.id: item for item in suite.items["applied_reasoning"]}
    calendar = items["reason_calendartime_002"]
    ordering = items["reason_orderingconstr_002"]

    assert calendar.expected["value"] == "2027-05-19"

    valid_orders: list[str] = []
    for candidate in permutations("ABCDE"):
        positions = {label: candidate.index(label) for label in candidate}
        if (
            positions["B"] == positions["A"] + 1
            and positions["D"] == positions["B"] + 1
            and positions["B"] < positions["E"]
            and abs(positions["C"] - positions["B"]) != 1
            and positions["E"] == positions["B"] + 2
            and positions["A"] == positions["C"] + 1
        ):
            valid_orders.append(",".join(candidate))
    assert valid_orders == [ordering.expected["value"]]


@pytest.mark.parametrize(
    ("invalid_lineage", "message"),
    [
        ("missing", "references unknown base item"),
        ("cross_benchmark", "same benchmark"),
        ("chained", "not another variant"),
    ],
)
def test_suite_rejects_invalid_variant_lineage(
    tmp_path: Path,
    invalid_lineage: str,
    message: str,
) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(Path("data"), data_root)
    items_path = data_root / "applied_reasoning" / "items.jsonl"
    items = items_path.read_text(encoding="utf-8").splitlines()
    variant = json.loads(items[2])
    variant["variant_of"] = json.loads(items[0])["id"]
    if invalid_lineage == "missing":
        variant["variant_of"] = "missing_base_item"
    elif invalid_lineage == "cross_benchmark":
        variant["variant_of"] = "schema_invoice_001"
    else:
        parent = json.loads(items[1])
        parent["variant_of"] = json.loads(items[0])["id"]
        items[1] = json.dumps(parent)
        variant["variant_of"] = parent["id"]
    items[2] = json.dumps(variant)
    items_path.write_text("\n".join(items) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=message):
        load_suite(data_root / "suites" / "core.yaml")


def test_json_verifier_reports_partial_leaf_accuracy() -> None:
    suite = load_suite(SUITE_PATH)
    invoice = suite.items["messy_text_to_schema"][0]

    correct = json.dumps(invoice.expected["value"])
    assert score_answer(invoice, correct).passed

    partial = json.dumps(
        {
            "invoice_number": "INV-204",
            "vendor": "Wrong Vendor",
            "currency": "INR",
            "total": 4250.0,
        }
    )
    result = score_answer(invoice, partial)
    assert not result.passed
    assert result.details["leaf_accuracy"] == pytest.approx(0.75)
    assert score_answer(invoice, "not json").details["reason"] == "invalid_json"

    fenced = score_answer(invoice, f"```json\n{correct}\n```")
    assert not fenced.passed
    assert fenced.score == 1.0
    assert fenced.details["protocol_compliant"] is False
    assert fenced.details["content_exact"] is True
    assert fenced.details["content_score"] == 1.0
    assert fenced.details["protocol_score"] == 0.0
    assert fenced.details["diagnostic_wrapper"] == "markdown_fence"

    wrong_type = json.dumps({**invoice.expected["value"], "total": True})
    wrong_type_result = score_answer(invoice, wrong_type)
    assert not wrong_type_result.passed
    assert wrong_type_result.details["content_exact"] is False


def test_noisy_order_gold_total_matches_its_line_items_and_tax() -> None:
    suite = load_suite(SUITE_PATH)
    order = next(
        item
        for item in suite.items["messy_text_to_schema"]
        if item.id == "schema_order_001"
    ).expected["value"]

    calculated_total = sum(
        item["quantity"] * item["unit_price"] for item in order["line_items"]
    ) + order["tax"]
    assert calculated_total == order["total"]


def test_constraint_verifier_checks_one_to_four_rule_items() -> None:
    items = {item.id: item for item in load_dataset(CONSTRAINT_PATH)}
    easy = items["constraint_api_rate_limiting_001"]
    two_rules = items["constraint_api_rate_limiting_002"]
    three_rules = items["constraint_api_rate_limiting_003"]
    hard = items["constraint_api_rate_limiting_004"]

    assert score_answer(easy, str(easy.expected["value"])).passed
    assert score_answer(two_rules, str(two_rules.expected["value"])).passed
    assert score_answer(three_rules, str(three_rules.expected["value"])).passed
    assert score_answer(hard, str(hard.expected["value"])).passed

    empty = score_answer(easy, "")
    assert not empty.passed
    assert empty.details["checks"]["exact_sentences"] is False
    assert empty.details["content_preserved"] is False

    missing_retry = score_answer(
        two_rules,
        "Rate limiting protects the service from overload. "
        "Servers reject bursts. Clients should wait. This keeps access fair.",
    )
    assert not missing_retry.passed
    assert missing_retry.details["content_preserved"] is False
    assert missing_retry.details["fact_checks"]["retry_after"] is False
    assert missing_retry.details["checks"]["required_terms"] is False

    json_item = items["constraint_employee_json_004"]
    wrong_seniority = json.loads(str(json_item.expected["value"]))
    wrong_seniority[0]["seniority"] = "junior"
    seniority_failure = score_answer(json_item, json.dumps(wrong_seniority))
    assert not seniority_failure.passed
    assert seniority_failure.details["content_preserved"] is False
    assert seniority_failure.details["checks"]["json_derived_bands"] is False


def test_data_heavy_constraint_tasks_use_larger_inputs_and_changing_answers() -> None:
    items = {item.id: item for item in load_dataset(CONSTRAINT_PATH)}

    employee_answers = [
        str(items[f"constraint_employee_json_{level:03d}"].expected["value"])
        for level in range(1, 5)
    ]
    assert len(json.loads(employee_answers[0])) == 12
    assert len(set(employee_answers)) == 4

    access_answers = [
        str(items[f"constraint_access_provisioning_{level:03d}"].expected["value"])
        for level in range(1, 5)
    ]
    assert len(json.loads(access_answers[0])) == 6
    assert len(json.loads(access_answers[1])) == 4
    assert json.loads(access_answers[2])[1]["expiration_days"] is None
    assert len(set(access_answers)) == 3

    classification_answers = [
        str(items[f"constraint_message_classification_{level:03d}"].expected["value"])
        for level in range(1, 5)
    ]
    assert len(json.loads(classification_answers[0])) == 16
    assert len(set(classification_answers)) == 4

def test_loader_rejects_difficulty_regression(tmp_path: Path) -> None:
    source_items = Path(
        "data/applied_reasoning/items.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    easy = next(line for line in source_items if json.loads(line)["difficulty"] == "easy")
    medium = next(
        line for line in source_items if json.loads(line)["difficulty"] == "medium"
    )
    hard = next(line for line in source_items if json.loads(line)["difficulty"] == "hard")
    bad_order = [medium, easy, hard]
    dataset_path = tmp_path / "bad_order.jsonl"
    dataset_path.write_text("\n".join(bad_order), encoding="utf-8")

    with pytest.raises(DatasetError, match="ordered from easy to hard"):
        load_dataset(dataset_path)


def test_loader_rejects_unknown_and_impossible_constraint_rules(
    tmp_path: Path,
) -> None:
    source = next(
        json.loads(line)
        for line in Path("data/constraint_load_curve/items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["id"] == "constraint_api_rate_limiting_004"
    )
    rules = source["scoring"]["parameters"]["rules"]
    rules["max_word"] = rules.pop("max_words")
    unknown_path = tmp_path / "unknown-rule.jsonl"
    unknown_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(DatasetError, match="unknown constraint rules"):
        load_dataset(unknown_path)

    rules["max_words"] = 10
    rules.pop("max_word")
    rules["exact_words"] = 11
    impossible_path = tmp_path / "impossible-rule.jsonl"
    impossible_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(DatasetError, match="exact_words cannot exceed max_words"):
        load_dataset(impossible_path)


def test_loader_rejects_a_gold_answer_that_cannot_pass(tmp_path: Path) -> None:
    source = json.loads(
        Path("data/constraint_load_curve/items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    source["expected"]["value"] = "Friday."
    dataset_path = tmp_path / "invalid-gold.jsonl"
    dataset_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(DatasetError, match="does not satisfy its scorer"):
        load_dataset(dataset_path)


def test_every_declared_difficulty_distribution_matches_target_count() -> None:
    suite = load_suite(SUITE_PATH)

    for definition in suite.definitions.values():
        assert sum(definition.current_difficulty_distribution.values()) == (
            definition.current_question_count
        )
        assert sum(definition.difficulty_distribution.values()) == (
            definition.target_question_count
        )
