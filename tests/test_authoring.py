import shutil
from pathlib import Path

import pytest

from llm_workload_benchmark.authoring import build_authoring_suite
from llm_workload_benchmark.dataset import DatasetError, load_suite


SOURCE_ROOT = Path("data/benchmarks/v1")


def test_authoring_build_is_reusable_and_only_rewrites_changed_output(
    tmp_path: Path,
) -> None:
    suite_root = tmp_path / "v1"
    shutil.copytree(SOURCE_ROOT, suite_root)
    suite_path = suite_root / "all_suite.yaml"

    first = build_authoring_suite(suite_path)
    assert not first.written
    assert len(first.unchanged) == 5

    source_path = (
        suite_root / "applied_reasoning" / "authoring" / "arithmetic.yaml"
    )
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace("15% of 800", "20% of 600"),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="generated JSONL is out of date"):
        build_authoring_suite(suite_path, check=True)

    rebuilt = build_authoring_suite(suite_path)
    assert rebuilt.written == (suite_root / "applied_reasoning" / "items.jsonl",)
    assert len(rebuilt.unchanged) == 4
    assert "20% of 600" in rebuilt.written[0].read_text(encoding="utf-8")


def test_suite_filters_select_smoke_items_across_benchmarks() -> None:
    suite = load_suite(SOURCE_ROOT / "smoke_suite.yaml")

    selected_ids = {
        item.id for benchmark_items in suite.items.values() for item in benchmark_items
    }
    assert selected_ids == {
        "reason_percentage_001",
        "schema_invoice_001",
        "constraint_deployment_001",
        "code_deduplicate_001",
        "summary_incident_001",
    }


def test_suite_filters_reject_unknown_item_ids(tmp_path: Path) -> None:
    suite_root = tmp_path / "v1"
    shutil.copytree(SOURCE_ROOT, suite_root)
    suite_path = suite_root / "smoke_suite.yaml"
    suite_path.write_text(
        suite_path.read_text(encoding="utf-8").replace(
            "reason_percentage_001", "missing_question"
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="unknown item ids: missing_question"):
        load_suite(suite_path)
