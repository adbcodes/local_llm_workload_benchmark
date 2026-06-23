from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.dataset import load_suite
from llm_workload_benchmark.catalog import validate_catalog


CATALOG_PATH = Path("data/catalog.yaml")
QUESTION_SET_IDS = {
    "applied_reasoning", "code_debug_repair", "knowledge_abstention",
    "messy_text_to_schema", "tables_to_decisions", "inbox_routing", "tool_use",
    "constraint_load_curve", "negative_instructions", "instruction_hierarchy",
    "raw_output_discipline", "grounded_compression", "india_focused_tasks",
    "long_text_retrieval", "conversation_memory", "clean_vs_noisy",
    "false_missing_information", "answer_stability", "confidence_correctness",
    "shuffled_choices", "prompt_format_sensitivity", "over_refusal",
}


def test_catalog_matches_the_six_suite_plan() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    assert catalog["schema_version"] == 2
    assert [suite["id"] for suite in catalog["suites"]] == list("ABCDEF")
    entries = catalog["benchmarks"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)) == 24
    assert {entry["id"] for entry in entries if entry["kind"] == "question_set"} == QUESTION_SET_IDS


def test_every_catalog_definition_exists_and_declares_its_task_types() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    for entry in catalog["benchmarks"]:
        path = CATALOG_PATH.parent / entry["definition_path"]
        definition = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert definition["id"] == entry["id"]
        if entry["kind"] == "question_set":
            assert definition["suite"] == entry["suite"]
            assert definition["task_types"]
            assert definition["metrics"]
            assert definition["scoring_methods"]


def test_planned_question_sets_are_empty_but_runnable_templates() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in catalog["benchmarks"]:
        if entry["kind"] != "question_set" or entry["status"] != "planned":
            continue
        directory = (CATALOG_PATH.parent / entry["definition_path"]).parent
        definition = yaml.safe_load((directory / "benchmark.yaml").read_text())
        questions = yaml.safe_load((directory / "questions.yaml").read_text())
        assert definition["current_question_count"] == 0
        assert questions["schema_version"] == 1
        assert questions["benchmark"] == entry["id"]
        assert questions["item_template"]
        assert questions["items"] == []
        assert (directory / "items.jsonl").read_text(encoding="utf-8") == ""

    suite = load_suite(Path("data/suites/all.yaml"))
    assert sum(len(items) for items in suite.items.values()) == 186


def test_backbone_generator_reproduces_planned_templates(tmp_path: Path) -> None:
    generated_root = tmp_path / "data"
    subprocess.run(
        [sys.executable, "scripts/generate_benchmark_backbone.py", "--root", str(generated_root)],
        check=True,
    )
    for benchmark_id in QUESTION_SET_IDS - {
        "applied_reasoning", "code_debug_repair", "messy_text_to_schema",
        "constraint_load_curve", "grounded_compression",
    }:
        for filename in ("benchmark.yaml", "questions.yaml", "items.jsonl"):
            assert (generated_root / benchmark_id / filename).read_text() == (
                Path("data") / benchmark_id / filename
            ).read_text()


def test_evaluation_and_probe_definitions_exist() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    for relative_path in catalog["evaluation_files"].values():
        assert (CATALOG_PATH.parent / relative_path).is_file()
    for relative_path in catalog["probe_sets"]:
        assert (CATALOG_PATH.parent / relative_path).is_file()
    assert not list(Path("data/planned").glob("*.yaml"))

    result = validate_catalog(CATALOG_PATH)
    assert result.benchmark_count == 24
    assert result.question_set_count == 22
    assert result.current_question_count == 186
    assert result.planned_question_set_count == 17
