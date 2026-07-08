from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.dataset import load_suite, score_answer


SUITE = Path("data/suites/final_deterministic.yaml")
QUESTIONS = Path("data/email_to_action/questions.yaml")


def _items():
    return load_suite(SUITE).items["email_to_action"]


def _raw_email(prompt: str) -> str:
    return prompt.split("--- RAW EMAIL ---\n", 1)[1].split(
        "\n--- END EMAIL ---", 1
    )[0]


def test_email_family_preserves_counts_mix_and_deterministic_contract() -> None:
    items = _items()

    assert len(items) == 40
    assert Counter(item.difficulty for item in items) == {
        "easy": 5,
        "medium": 15,
        "hard": 20,
    }
    assert Counter(item.visibility for item in items) == {
        "public": 20,
        "held_out": 20,
    }
    assert Counter(item.subcategory for item in items) == {
        "single_label_clear": 8,
        "multi_label": 8,
        "urgency_polite": 6,
        "subject_body_mismatch": 5,
        "no_response_notification": 5,
        "phishing_lookalike": 4,
        "vague_missing_fields": 4,
    }
    assert all(item.scoring.method == "json_exact" for item in items)
    assert all(
        item.scoring.parameters["unordered_array_paths"] == ["$.labels"]
        for item in items
    )
    assert all(
        score_answer(item, json.dumps(item.expected["value"])).passed
        for item in items
    )


def test_email_sources_cover_micro_through_very_long_realistic_shapes() -> None:
    items = _items()
    shapes = {
        "micro_email",
        "short_email",
        "medium_email",
        "long_email",
        "very_long_email",
    }

    assert Counter(next(tag for tag in item.tags if tag in shapes) for item in items) == {
        "micro_email": 6,
        "short_email": 14,
        "medium_email": 10,
        "long_email": 8,
        "very_long_email": 2,
    }
    assert Counter(len(item.expected["value"]["fields"]) for item in items) == {
        2: 11,
        3: 21,
        4: 8,
    }

    raw_emails = [_raw_email(item.prompt) for item in items]
    for raw_email in raw_emails:
        assert raw_email.startswith("From:")
        assert "\nTo:" in raw_email
        assert "\nSubject:" in raw_email
        assert "\nDate:" in raw_email
    assert min(len(raw.split()) for raw in raw_emails) <= 45
    assert max(len(raw.split()) for raw in raw_emails) >= 300
    assert sum(
        marker in raw
        for raw in raw_emails
        for marker in ("---- Original message ----", "--- forwarded thread ---", "----- Forwarded message -----")
    ) >= 3
    assert any("Page 1 of 1" in raw for raw in raw_emails)
    assert any("[Harbour Bank Secure Message]" in raw for raw in raw_emails)


def test_email_label_arrays_are_order_insensitive_but_not_duplicate_insensitive() -> None:
    item = next(item for item in _items() if len(item.expected["value"]["labels"]) > 1)
    expected = item.expected["value"]
    reversed_labels = {**expected, "labels": list(reversed(expected["labels"]))}
    assert score_answer(item, json.dumps(reversed_labels)).passed

    duplicated = {**expected, "labels": [*expected["labels"], expected["labels"][0]]}
    assert not score_answer(item, json.dumps(duplicated)).passed


def test_materialized_email_questions_match_generator(tmp_path: Path) -> None:
    generated = tmp_path / "questions.yaml"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_email_to_action.py",
            "--output",
            str(generated),
        ],
        check=True,
    )
    assert yaml.safe_load(generated.read_text(encoding="utf-8")) == yaml.safe_load(
        QUESTIONS.read_text(encoding="utf-8")
    )
