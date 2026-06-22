import json
import shutil
from itertools import permutations
from pathlib import Path

import pytest

from llm_workload_benchmark.dataset import (
    DatasetError,
    load_dataset,
    load_suite,
    score_answer,
)

SUITE_PATH = Path("data/suites/core.yaml")
CONSTRAINT_PATH = Path("data/constraint_load_curve/items.jsonl")


def test_active_pilot_suite_loads_with_difficulty_progression() -> None:
    suite = load_suite(SUITE_PATH)

    assert set(suite.items) == {
        "applied_reasoning",
        "messy_text_to_schema",
    }
    assert sum(len(items) for items in suite.items.values()) == 51
    reasoning = suite.items["applied_reasoning"]
    assert len(reasoning) == 48
    assert [item.difficulty for item in reasoning].count("easy") == 12
    assert [item.difficulty for item in reasoning].count("medium") == 24
    assert [item.difficulty for item in reasoning].count("hard") == 12
    assert [item.difficulty for item in suite.items["messy_text_to_schema"]] == [
        "easy",
        "medium",
        "hard",
    ]
    assert sum(item.split == "dev" for item in reasoning) == 8


def test_numeric_and_exact_answer_verifiers() -> None:
    suite = load_suite(SUITE_PATH)
    items = {item.id: item for item in suite.items["applied_reasoning"]}
    percentage = items["reason_percentage_001"]
    calendar = items["reason_calendar_001"]
    ordering = items["reason_ordering_001"]
    rational = items["anchor_math_odds_001"]

    assert calendar.scoring.method == "date_value"
    percentage_result = score_answer(percentage, "120")
    assert percentage_result.passed
    assert percentage_result.type == "deterministic"
    assert percentage_result.evaluator == "numeric_tolerance"
    assert percentage_result.version == 1
    assert score_answer(percentage, "120.0").passed
    assert score_answer(percentage, '"120"').passed
    assert score_answer(percentage, "The answer is 120.").passed
    assert not score_answer(percentage, "121").passed
    assert not score_answer(percentage, "It could be 120 or 121.").passed
    assert score_answer(calendar, "The date is 2026-04-01.").passed
    assert score_answer(calendar, "The answer is April 1, 2026.").passed
    assert not score_answer(calendar, "The answer is 2026-04-02.").passed
    assert score_answer(ordering, "D,A,B,E,C").passed
    assert score_answer(ordering, "Therefore: D, A, B, E, C.").passed
    assert not score_answer(ordering, "A,B,D,E,C").passed
    assert score_answer(rational, "4/7").passed
    assert score_answer(rational, r"The answer is \frac{4}{7}.").passed
    assert score_answer(rational, "0.5714285714285714").passed
    assert not score_answer(rational, "3/7").passed


def test_reasoning_and_order_gold_answers_are_independently_derived() -> None:
    suite = load_suite(SUITE_PATH)
    items = {item.id: item for item in suite.items["applied_reasoning"]}
    calendar = items["reason_calendar_001"]
    ordering = items["reason_ordering_001"]

    assert calendar.expected["value"] == "2026-04-01"

    valid_orders: list[str] = []
    for candidate in permutations("ABCDE"):
        positions = {label: candidate.index(label) for label in candidate}
        if (
            positions["B"] == positions["A"] + 1
            and positions["C"] == positions["E"] + 1
            and positions["D"] < positions["A"]
            and positions["B"] < positions["E"]
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
    assert fenced.details["diagnostic_wrapper"] == "markdown_fence"

    wrong_type = json.dumps({**invoice.expected["value"], "total": True})
    wrong_type_result = score_answer(invoice, wrong_type)
    assert not wrong_type_result.passed
    assert wrong_type_result.details["content_exact"] is False


def test_noisy_order_gold_total_matches_its_line_items_and_tax() -> None:
    suite = load_suite(SUITE_PATH)
    order = suite.items["messy_text_to_schema"][2].expected["value"]

    calculated_total = sum(
        item["quantity"] * item["unit_price"] for item in order["line_items"]
    ) + order["tax"]
    assert calculated_total == order["total"]


def test_constraint_verifier_checks_one_three_and_six_rule_items() -> None:
    easy, medium, hard = load_dataset(CONSTRAINT_PATH)

    assert score_answer(easy, str(easy.expected["value"])).passed
    assert score_answer(medium, str(medium.expected["value"])).passed
    assert score_answer(hard, str(hard.expected["value"])).passed

    empty = score_answer(easy, "")
    assert not empty.passed
    assert empty.details["checks"]["max_words"] is True
    assert empty.details["content_preserved"] is False

    friday_substring = score_answer(
        easy,
        "The payments deployment passed automated checks and launches "
        "Fridayish with support monitoring transaction failures.",
    )
    assert not friday_substring.passed
    assert friday_substring.details["fact_checks"]["friday_release"] is False

    hard_failure = score_answer(
        hard,
        "Payments deployment passed automated checks and launches Friday evening, "
        "with support monitoring transaction failures for one hour afterward.",
    )
    assert not hard_failure.passed
    assert hard_failure.details["content_preserved"] is True
    assert hard_failure.details["checks"]["forbidden_punctuation"] is False


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
    source = json.loads(
        Path("data/constraint_load_curve/items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1]
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
