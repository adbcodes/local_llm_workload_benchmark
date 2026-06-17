import json
from pathlib import Path

import pytest

from llm_workload_benchmark.dataset import (
    DatasetError,
    load_dataset,
    load_suite,
    score_answer,
)

SUITE_PATH = Path("data/benchmarks/v1/suite.yaml")


def test_pilot_suite_loads_three_benchmarks_with_difficulty_progression() -> None:
    suite = load_suite(SUITE_PATH)

    assert set(suite.items) == {
        "applied_reasoning",
        "messy_text_to_schema",
        "constraint_load_curve",
    }
    assert sum(len(items) for items in suite.items.values()) == 9
    for items in suite.items.values():
        assert [item.difficulty for item in items] == ["easy", "medium", "hard"]
        assert all(item.split == "dev" for item in items)


def test_numeric_and_exact_answer_verifiers() -> None:
    suite = load_suite(SUITE_PATH)
    percentage, calendar, ordering = suite.items["applied_reasoning"]

    assert score_answer(percentage, "120").passed
    assert not score_answer(percentage, "121").passed
    assert score_answer(calendar, "2026-05-26\n").passed
    assert score_answer(ordering, "D,C,A,B").passed
    assert not score_answer(ordering, "A,B,D,C").passed


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


def test_constraint_verifier_checks_one_three_and_six_rule_items() -> None:
    suite = load_suite(SUITE_PATH)
    easy, medium, hard = suite.items["constraint_load_curve"]

    assert score_answer(easy, "Deployment launches Friday.").passed
    assert score_answer(
        medium,
        "The deployment passed every check and launches Friday.",
    ).passed
    assert score_answer(
        hard,
        "Deployment passed every check and launches this Friday.",
    ).passed

    hard_failure = score_answer(
        hard,
        "The successful deployment launches Friday.",
    )
    assert not hard_failure.passed
    assert hard_failure.details["checks"]["forbidden_terms"] is False
    assert hard_failure.details["checks"]["prefix"] is False


def test_loader_rejects_difficulty_regression(tmp_path: Path) -> None:
    source_items = Path(
        "data/benchmarks/v1/applied_reasoning/items.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    bad_order = [source_items[1], source_items[0], source_items[2]]
    dataset_path = tmp_path / "bad_order.jsonl"
    dataset_path.write_text("\n".join(bad_order), encoding="utf-8")

    with pytest.raises(DatasetError, match="ordered from easy to hard"):
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
