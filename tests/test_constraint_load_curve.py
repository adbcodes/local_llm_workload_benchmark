from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.dataset import load_suite, score_answer


SUITE_PATH = Path("data/suites/instruction.yaml")
QUESTIONS_PATH = Path("data/constraint_load_curve/questions.yaml")
REVIEW_PATH = Path("docs/TEMP_CONSTRAINT_LOAD_CURVE_REVIEW.md")


def test_constraint_load_curve_has_ten_complete_comparison_groups() -> None:
    items = load_suite(SUITE_PATH).items["constraint_load_curve"]

    assert len(items) == 40
    assert Counter(item.subcategory for item in items) == {
        "one_constraint": 10,
        "two_constraints": 10,
        "three_constraints": 10,
        "four_constraints": 10,
    }
    assert Counter(item.difficulty for item in items) == {
        "easy": 10,
        "medium": 20,
        "hard": 10,
    }
    assert Counter(len(item.scoring.parameters["rules"]) for item in items) == {
        1: 10,
        2: 10,
        3: 10,
        4: 10,
    }
    assert Counter(
        tag
        for item in items
        if item.variant_of is None
        for tag in item.tags
        if tag.endswith("_carrier")
    ) == {
        "prose_carrier": 3,
        "extraction_carrier": 1,
        "list_carrier": 1,
        "structured_json_carrier": 1,
        "structured_csv_carrier": 1,
        "classification_carrier": 1,
        "structured_yaml_carrier": 1,
        "ordering_carrier": 1,
    }

    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(item.variant_of or item.id, []).append(item)
        assert score_answer(item, str(item.expected["value"])).passed

    assert len(groups) == 10
    for base_id, variants in groups.items():
        variants.sort(key=lambda item: len(item.scoring.parameters["rules"]))
        assert [len(item.scoring.parameters["rules"]) for item in variants] == [
            1,
            2,
            3,
            4,
        ]
        assert variants[0].id == base_id
        previous_rules: set[str] = set()
        for item in variants:
            active_rules = set(item.scoring.parameters["rules"])
            assert previous_rules < active_rules
            previous_rules = active_rules


def test_final_task_mix_and_interaction_hotspots_are_frozen() -> None:
    items = load_suite(SUITE_PATH).items["constraint_load_curve"]
    bases = [item for item in items if item.variant_of is None]

    assert len(bases) == 10
    rewrite = next(item for item in items if item.id == "constraint_paragraph_rewrite_003")
    email = next(item for item in items if item.id == "constraint_vendor_email_004")
    assert "short_rewrite_with_banned_verbs" in rewrite.tags
    assert "word_range_with_paragraph_structure" in email.tags

    source = rewrite.prompt.split("Source data: ", 1)[1].split(
        " Mandatory constraints:", 1
    )[0]
    assert len(source.split()) == 80


def test_extraction_filters_cancelled_order_and_checks_gold_values() -> None:
    items = load_suite(SUITE_PATH).items["constraint_load_curve"]
    item = next(item for item in items if item.id == "constraint_order_extraction_003")

    result = score_answer(item, "1007,1015,1024,1031,1042")

    assert not result.passed
    assert not result.details["content_preserved"]
    assert result.details["checks"]["excluded_values"] is False


def test_classification_checks_ticket_content_and_summary_separately() -> None:
    items = load_suite(SUITE_PATH).items["constraint_load_curve"]
    item = next(
        item for item in items if item.id == "constraint_message_classification_004"
    )

    wrong_count_value = json.loads(str(item.expected["value"]))
    wrong_count_value[-1]["summary"]["billing"] = 3
    wrong_count = score_answer(item, json.dumps(wrong_count_value))
    assert wrong_count.details["content_preserved"] is False
    assert wrong_count.details["checks"]["json_summary_counts"] is False

    wrong_route_value = json.loads(str(item.expected["value"]))
    wrong_route_value[4]["category"] = "general"
    wrong_route_value[-1]["summary"]["billing"] = 3
    wrong_route_value[-1]["summary"]["general"] = 5
    wrong_route = score_answer(item, json.dumps(wrong_route_value))
    assert wrong_route.details["content_score"] < 1
    assert wrong_route.details["checks"]["json_summary_counts"] is True


def test_structured_and_interacting_constraint_helpers_fail_independently() -> None:
    items = {
        item.id: item
        for item in load_suite(SUITE_PATH).items["constraint_load_curve"]
    }

    csv_item = items["constraint_book_csv_003"]
    old_book = str(csv_item.expected["value"]) + "\nNorthern Byte,Kabir Sen,2014"
    csv_result = score_answer(csv_item, old_book)
    assert csv_result.details["content_preserved"] is False
    assert csv_result.details["checks"]["csv_year_min"] is False

    yaml_item = items["constraint_service_yaml_004"]
    yaml_without_healthcheck = (
        "service: atlas-api\nimage: registry.example/atlas:2.4\nport: 8080\n"
        "replicas: 3\nregion: ap-south-1"
    )
    yaml_result = score_answer(yaml_item, yaml_without_healthcheck)
    assert yaml_result.details["checks"]["yaml_only"] is True
    assert yaml_result.details["checks"]["yaml_healthcheck"] is False

    ordering_item = items["constraint_point_ordering_004"]
    wrong_tie_order = str(items["constraint_point_ordering_003"].expected["value"])
    ordering_result = score_answer(ordering_item, wrong_tie_order)
    assert ordering_result.details["checks"]["sorted_by_points"] is True
    assert ordering_result.details["checks"]["ties_alphabetical"] is False

    email_item = items["constraint_vendor_email_004"]
    collapsed_email = str(email_item.expected["value"]).replace("\n\n", "\n")
    email_result = score_answer(email_item, collapsed_email)
    assert email_result.details["checks"]["word_range"] is True
    assert email_result.details["checks"]["exact_paragraphs"] is False


def test_materialized_questions_match_generator(tmp_path: Path) -> None:
    regenerated_path = tmp_path / "questions.yaml"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_constraint_load_questions.py",
            "--output",
            str(regenerated_path),
        ],
        check=True,
    )

    committed = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    regenerated = yaml.safe_load(regenerated_path.read_text(encoding="utf-8"))
    assert committed == regenerated


def test_temporary_markdown_review_matches_dataset(tmp_path: Path) -> None:
    regenerated_path = tmp_path / "review.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_constraint_load_review.py",
            "--output",
            str(regenerated_path),
        ],
        check=True,
    )

    review = REVIEW_PATH.read_text(encoding="utf-8")
    assert review == regenerated_path.read_text(encoding="utf-8")
    assert "466 questions across 16 benchmarks" in review
    assert "[Applied Reasoning Gauntlet](#applied-reasoning)" in review
    assert "[Following Multiple Rules](#constraint-load-curve)" in review
    assert "[Messy Text to Schema](#messy-text-to-schema)" in review
    assert "[Long-Text Retrieval](#long-text-retrieval)" in review
    assert "[Over-Refusal](#over-refusal)" in review
    assert "Messy Text to Schema — 48 questions" in review
    assert "Confidence vs Correctness" not in review
    assert "**Conversation shown to the model**" in review
    assert review.count("<summary><code>") == 466
    assert review.count("<details>") == 482
    assert not Path("docs/TEMP_CODE_DEBUG_REPAIR_REVIEW.html").exists()
