from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.dataset import load_suite, score_answer


SCHEMA_SUITE = Path("data/suites/structured.yaml")
SUMMARY_SUITE = Path("data/suites/judged.yaml")


def test_schema_set_has_target_size_and_varied_document_shapes() -> None:
    items = load_suite(SCHEMA_SUITE).items["messy_text_to_schema"]

    assert len(items) == 30
    assert Counter(item.difficulty for item in items) == {
        "easy": 8,
        "medium": 15,
        "hard": 7,
    }
    assert len({item.subcategory for item in items}) >= 25
    assert all("Return the raw JSON directly" in item.prompt for item in items)
    assert all("Do not wrap it in ``` or ```json" in item.prompt for item in items)
    assert sum(
        isinstance(item.expected["value"], list)
        or any(
            isinstance(value, (dict, list))
            for value in item.expected["value"].values()
        )
        for item in items
    ) >= 8
    assert all(
        score_answer(item, json.dumps(item.expected["value"])).passed
        for item in items
    )


def test_summary_set_has_target_size_and_distinct_tasks() -> None:
    items = load_suite(SUMMARY_SUITE).items["grounded_compression"]

    assert len(items) == 20
    assert Counter(item.difficulty for item in items) == {
        "easy": 5,
        "medium": 10,
        "hard": 5,
    }
    assert len({item.subcategory for item in items}) == 20
    assert len({item.scoring.parameters["max_words"] for item in items}) >= 5
    assert all("Source:" in item.prompt for item in items)
    assert all(len(item.prompt.split()) >= 50 for item in items)


def test_materialized_schema_and_summary_questions_match_generator(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.yaml"
    summary_path = tmp_path / "summary.yaml"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_schema_and_summary_questions.py",
            "--schema-output",
            str(schema_path),
            "--summary-output",
            str(summary_path),
        ],
        check=True,
    )

    assert yaml.safe_load(schema_path.read_text(encoding="utf-8")) == yaml.safe_load(
        Path("data/messy_text_to_schema/questions.yaml").read_text(encoding="utf-8")
    )
    assert yaml.safe_load(summary_path.read_text(encoding="utf-8")) == yaml.safe_load(
        Path("data/grounded_compression/questions.yaml").read_text(encoding="utf-8")
    )
