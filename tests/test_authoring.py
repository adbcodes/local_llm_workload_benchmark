import shutil
from pathlib import Path

import pytest

from llm_workload_benchmark.authoring import build_authoring_suite
from llm_workload_benchmark.dataset import DatasetError, load_suite


SOURCE_ROOT = Path("data")


def test_authoring_build_is_reusable_and_only_rewrites_changed_output(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(SOURCE_ROOT, data_root)
    suite_path = data_root / "suites" / "final_six.yaml"

    first = build_authoring_suite(suite_path)
    assert not first.written
    assert len(first.unchanged) == 6

    source_path = (
        data_root / "applied_reasoning" / "generated.yaml"
    )
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "we get 640 kg of rice for the community kitchen each month",
            "we get 640 kg of rice for the relief kitchen each month",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="generated JSONL is out of date"):
        build_authoring_suite(suite_path, check=True)

    rebuilt = build_authoring_suite(suite_path)
    assert rebuilt.written == (data_root / "applied_reasoning" / "items.jsonl",)
    assert len(rebuilt.unchanged) == 5
    assert "relief kitchen" in rebuilt.written[0].read_text(encoding="utf-8")


def test_suite_filters_select_smoke_items_across_benchmarks() -> None:
    suite = load_suite(SOURCE_ROOT / "suites" / "smoke.yaml")

    selected_ids = {
        item.id for benchmark_items in suite.items.values() for item in benchmark_items
    }
    assert selected_ids == {
        "reason_arithmeticperc_001",
        "schema_invoice_001",
        "constraint_api_rate_limiting_001",
        "code_normalize_event_codes_001",
        "long_text_retrieval_001",
        "tool_use_001",
    }


def test_deterministic_suite_excludes_external_judge_items() -> None:
    suite = load_suite(SOURCE_ROOT / "suites" / "final_six.yaml")

    assert sum(len(items) for items in suite.items.values()) == 320
    assert all(
        item.scoring.method != "llm_judge"
        for items in suite.items.values()
        for item in items
    )


def test_suite_filters_reject_unknown_item_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(SOURCE_ROOT, data_root)
    suite_path = data_root / "suites" / "smoke.yaml"
    suite_path.write_text(
        suite_path.read_text(encoding="utf-8").replace(
            "reason_arithmeticperc_001", "missing_question"
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="unknown item ids: missing_question"):
        load_suite(suite_path)
