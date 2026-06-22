from pathlib import Path

import yaml


CATALOG_PATH = Path("data/catalog.yaml")
EXPECTED_BENCHMARKS = {
    "applied_reasoning",
    "messy_text_to_schema",
    "constraint_load_curve",
    "code_debug_repair",
    "grounded_compression",
    "tables_to_decisions",
    "inbox_routing",
    "tool_use",
    "practical_writing",
    "personal_model_preference",
    "long_text_retrieval",
    "conversation_memory",
    "noisy_text_robustness",
    "false_or_missing_information",
    "answer_stability",
    "confidence_calibration",
    "shuffled_choices",
    "india_focused_tasks",
    "quantization_survival_curve",
    "laptop_value_frontier",
}


def test_catalog_covers_every_planned_benchmark() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = catalog["benchmarks"]
    ids = [entry["id"] for entry in entries]

    assert catalog["schema_version"] == 1
    assert len(ids) == len(set(ids)) == 20
    assert set(ids) == EXPECTED_BENCHMARKS


def test_every_catalog_definition_exists_and_matches_its_entry() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    for entry in catalog["benchmarks"]:
        definition_path = CATALOG_PATH.parent / entry["definition_path"]
        definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))

        assert definition["id"] == entry["id"]
        if entry["status"] == "planned" and entry["kind"] == "question_set":
            assert definition["status"] == "planned"
            assert definition["starter_items"]
            assert all(
                item["status"] == "draft" for item in definition["starter_items"]
            )
        if entry["kind"] == "derived":
            source_suite = (definition_path.parent / definition["source_suite"]).resolve()
            assert source_suite.exists()
