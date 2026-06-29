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

    assert len(items) == 48
    assert Counter(item.difficulty for item in items) == {
        "easy": 8,
        "medium": 25,
        "hard": 15,
    }
    assert len({item.subcategory for item in items}) >= 43
    assert Counter(item.visibility for item in items) == {"public": 24, "held_out": 24}
    assert Counter(item.split for item in items) == {"dev": 24, "test": 24}
    assert all(item.provenance.review_status == "human_checked" for item in items)
    assert all(item.provenance.generator == "messy_text_to_schema_v2" for item in items)
    assert all(item.provenance.seed == 20260731 for item in items)
    assert all("Schema:\n" in item.prompt for item in items)
    assert all("type annotations, not literal output values" in item.prompt for item in items)
    assert all("Return raw JSON only" in item.prompt for item in items)
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


def test_schema_difficulty_tracks_observable_extraction_features() -> None:
    items = load_suite(SCHEMA_SUITE).items["messy_text_to_schema"]

    for item in items:
        feature_tags = {tag for tag in item.tags if tag.startswith("feature_")}
        assert feature_tags
        if item.difficulty == "hard":
            assert len(feature_tags) >= 3


def test_schema_gold_resolves_revisions_and_computed_values() -> None:
    loaded_items = load_suite(SCHEMA_SUITE).items["messy_text_to_schema"]
    items = {item.id: item.expected["value"] for item in loaded_items}
    prompts = {item.id: item.prompt for item in loaded_items}

    purchase_order = items["schema_purchase_order_001"]
    assert purchase_order["pre_tax_total"] == (
        sum(row["quantity"] * row["unit_price"] for row in purchase_order["items"])
        + purchase_order["freight"]
    )
    manifest = items["schema_manifest_001"]
    assert manifest["total_weight_kg"] == sum(
        package["weight_kg"] for package in manifest["packages"]
    )
    incident = items["schema_incident_001"]
    assert incident["mitigated_at"] == "09:41"
    assert '"email": "string or null"' in prompts["schema_batch_001"]
    assert '"humidity_percent": "number or null"' in prompts["schema_sensor_001"]
    usage_bill = items["schema_usage_bill_001"]
    assert usage_bill["subtotal"] == sum(charge["amount"] for charge in usage_bill["charges"])
    energy_log = items["schema_energy_001"]
    assert energy_log["total_kwh"] == sum(
        reading["energy_kwh"] for reading in energy_log["readings"]
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

    schema_document = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert schema_document["generated_by"] == "messy_text_to_schema_v2"
    assert schema_document["seed"] == 20260731
