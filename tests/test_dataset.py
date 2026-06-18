import json
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

import pytest

from llm_workload_benchmark.dataset import (
    DatasetError,
    load_dataset,
    load_suite,
    score_answer,
)

SUITE_PATH = Path("data/benchmarks/v1/suite.yaml")
CONSTRAINT_PATH = Path("data/benchmarks/v1/constraint_load_curve/items.jsonl")


def test_pilot_suite_loads_three_benchmarks_with_difficulty_progression() -> None:
    suite = load_suite(SUITE_PATH)

    assert set(suite.items) == {
        "applied_reasoning",
        "messy_text_to_schema",
    }
    assert sum(len(items) for items in suite.items.values()) == 6
    for items in suite.items.values():
        assert [item.difficulty for item in items] == ["easy", "medium", "hard"]
        assert all(item.split == "dev" for item in items)


def test_numeric_and_exact_answer_verifiers() -> None:
    suite = load_suite(SUITE_PATH)
    percentage, calendar, ordering = suite.items["applied_reasoning"]

    assert calendar.scoring.method == "date_value"
    assert score_answer(percentage, "120").passed
    assert score_answer(percentage, "120.0").passed
    assert score_answer(percentage, '"120"').passed
    assert score_answer(percentage, "The answer is 120.").passed
    assert not score_answer(percentage, "121").passed
    assert not score_answer(percentage, "It could be 120 or 121.").passed
    assert score_answer(calendar, "The seventh occurrence is 2026-05-26.").passed
    assert score_answer(calendar, "The answer is 26-05-2026.").passed
    assert score_answer(calendar, "The answer is 26/05/2026.").passed
    assert score_answer(calendar, "The answer is May 26, 2026.").passed
    assert score_answer(calendar, "The answer is 26 May 2026.").passed
    assert score_answer(calendar, "The answer is 05/26/2026.").passed
    assert not score_answer(calendar, "The answer is 05/06/2026.").passed
    assert not score_answer(calendar, "Either 2026-05-26 or 2026-06-02.").passed
    assert score_answer(ordering, "D,C,A,B").passed
    assert score_answer(ordering, "Therefore the order is D, C, A, B.").passed
    assert not score_answer(ordering, "A,B,D,C").passed


def test_reasoning_and_order_gold_answers_are_independently_derived() -> None:
    suite = load_suite(SUITE_PATH)
    _, calendar, ordering = suite.items["applied_reasoning"]

    seventh_occurrence = date(2026, 3, 3) + timedelta(weeks=2 * 6)
    assert seventh_occurrence.isoformat() == calendar.expected["value"]

    valid_orders: list[str] = []
    for candidate in permutations("ABCD"):
        positions = {label: candidate.index(label) for label in candidate}
        if (
            positions["B"] > positions["A"]
            and positions["D"] + 1 == positions["C"]
            and positions["A"] != 0
        ):
            valid_orders.append(",".join(candidate))
    assert valid_orders == [ordering.expected["value"]]


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
        "data/benchmarks/v1/applied_reasoning/items.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    bad_order = [source_items[1], source_items[0], source_items[2]]
    dataset_path = tmp_path / "bad_order.jsonl"
    dataset_path.write_text("\n".join(bad_order), encoding="utf-8")

    with pytest.raises(DatasetError, match="ordered from easy to hard"):
        load_dataset(dataset_path)


def test_loader_rejects_unknown_and_impossible_constraint_rules(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path("data/benchmarks/v1/constraint_load_curve/items.jsonl")
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
        Path("data/benchmarks/v1/constraint_load_curve/items.jsonl")
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
