import json
from collections import Counter
from pathlib import Path

import pytest

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import DatasetError, load_suite, score_answer
from llm_workload_benchmark.executable import evaluate_python
from llm_workload_benchmark.authoring import _validate_gold
from llm_workload_benchmark.runner import GenerationOutput, run_benchmark

CODING_SUITE_PATH = Path("data/suites/coding.yaml").resolve()


def _coding_item():
    return load_suite(CODING_SUITE_PATH).items["code_debug_repair"][0]


def test_coding_dataset_has_agreed_task_mix_and_fresh_implementation_contract() -> None:
    items = load_suite(CODING_SUITE_PATH).items["code_debug_repair"]

    assert len(items) == 48
    assert Counter(item.subcategory for item in items) == {
        "function_implementation": 30,
        "bug_diagnosis": 10,
        "code_repair": 8,
    }
    assert Counter(item.difficulty for item in items) == {
        "easy": 16,
        "medium": 24,
        "hard": 8,
    }
    implementations = [
        item for item in items if item.subcategory == "function_implementation"
    ]
    assert all("practical_python" in item.tags for item in implementations)
    assert all("fresh_composed" in item.tags for item in implementations)
    assert all(item.expected["value"].get("reference_solution") for item in implementations)
    assert all(
        any(test.get("preserve_args") for test in item.expected["value"]["tests"])
        for item in implementations
    )
    assert Counter(item.visibility for item in items) == {
        "public": 24,
        "held_out": 24,
    }
    assert Counter(item.split for item in items) == {"dev": 12, "test": 36}


def test_bug_diagnosis_items_use_deterministic_labels() -> None:
    items = load_suite(CODING_SUITE_PATH).items["code_debug_repair"]
    diagnoses = [item for item in items if item.subcategory == "bug_diagnosis"]

    assert len(diagnoses) == 10
    assert len({item.expected["value"] for item in diagnoses}) == 10
    assert {
        item.id for item in diagnoses if item.difficulty == "hard"
    } == {"diagnose_dependency_edges_001", "diagnose_route_choice_001"}
    for item in diagnoses:
        assert item.scoring.method == "exact_match"
        assert score_answer(item, item.expected["value"].upper()).passed


def test_repair_items_use_verified_references_and_three_killed_mutants() -> None:
    items = load_suite(CODING_SUITE_PATH).items["code_debug_repair"]
    repairs = [item for item in items if item.subcategory == "code_repair"]

    assert len(repairs) == 8
    assert {item.id for item in repairs} == {
        "repair_quota_adjustments_001",
        "repair_latest_webhooks_001",
        "repair_refund_total_001",
        "repair_availability_windows_001",
        "repair_compact_ranges_001",
        "repair_rolling_totals_001",
        "repair_lookup_path_001",
        "repair_lru_cache_001",
    }
    for item in repairs:
        specification = item.expected["value"]
        assert "generated_mutation" in item.tags
        assert evaluate_python(item, specification["reference_solution"]).passed
        assert len(specification["mutants"]) >= 3
        assert all(
            not evaluate_python(item, mutant["source"]).passed
            for mutant in specification["mutants"]
        )


def test_hard_executable_items_have_five_behavioral_checks() -> None:
    items = load_suite(CODING_SUITE_PATH).items["code_debug_repair"]
    hard_executable = [
        item
        for item in items
        if item.difficulty == "hard" and item.scoring.method == "executable_python"
    ]

    assert len(hard_executable) == 6
    assert all(len(item.expected["value"]["tests"]) >= 5 for item in hard_executable)


def test_coding_fixture_uses_restricted_executable_contract() -> None:
    item = _coding_item()

    assert item.response_contract.type == "code"
    assert item.scoring.method == "executable_python"
    with pytest.raises(DatasetError, match="restricted Python execution"):
        score_answer(item, "def deduplicate_preserving_order(values): return values")


def test_python_evaluator_scores_unit_test_pass_rate_and_accepts_fences() -> None:
    result = evaluate_python(
        _coding_item(),
        """```python
def normalize_event_codes(codes):
    result = []
    for code in codes:
        value = code.strip().upper()
        if value and value not in result:
            result.append(value)
    return result
```""",
    )

    assert result.type == "executable"
    assert result.evaluator == "restricted_python_tests"
    assert result.passed
    assert result.score == 1
    assert result.details["tests_passed"] == 4
    assert result.details["diagnostic_wrapper"] == "markdown_fence"

    partial = evaluate_python(
        _coding_item(),
        "def normalize_event_codes(codes):\n    return codes",
    )
    assert not partial.passed
    assert partial.score == pytest.approx(0.25)
    assert partial.details["tests_passed"] == 1


def test_python_evaluator_enforces_declared_input_preservation() -> None:
    item = _coding_item().model_copy(deep=True)
    result = evaluate_python(
        item,
        "def normalize_event_codes(codes):\n"
        "    codes[:] = list(dict.fromkeys(code.strip().upper() for code in codes if code.strip()))\n"
        "    return codes",
    )

    assert not result.passed
    assert result.score == pytest.approx(0.25)
    assert result.details["failures"] == [
        {"test_index": 1, "mutated_arguments": [0]},
        {"test_index": 3, "mutated_arguments": [0]},
        {"test_index": 4, "mutated_arguments": [0]},
    ]


def test_authoring_validation_executes_reference_and_mutant_sources() -> None:
    item = _coding_item().model_copy(deep=True)
    item.expected["value"]["reference_solution"] = (
        "def normalize_event_codes(codes):\n"
        "    return list(dict.fromkeys(code.strip().upper() for code in codes if code.strip()))"
    )
    item.expected["value"]["mutants"] = [
        {
            "id": "returns_input",
            "source": "def normalize_event_codes(codes):\n    return codes",
        }
    ]

    _validate_gold(item)

    item.expected["value"]["mutants"][0]["source"] = item.expected["value"][
        "reference_solution"
    ]
    with pytest.raises(DatasetError, match="do not kill mutants: returns_input"):
        _validate_gold(item)


def test_python_evaluator_allows_safe_lambda_expressions() -> None:
    result = evaluate_python(
        _coding_item(),
        "def normalize_event_codes(codes):\n"
        "    ordered = sorted(enumerate(codes), key=lambda pair: pair[0])\n"
        "    return list(dict.fromkeys(code.strip().upper() for _, code in ordered if code.strip()))",
    )

    assert result.passed
    assert result.details["reason"] == "all_tests_passed"


def test_python_evaluator_stops_infinite_loop_at_timeout() -> None:
    result = evaluate_python(
        _coding_item(),
        "def normalize_event_codes(codes):\n    while True:\n        pass",
    )

    assert not result.passed
    assert result.score == 0
    assert result.details["reason"] in {"timeout", "resource_limit"}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\ndef normalize_event_codes(codes): return codes", "one function"),
        ("def wrong_name(values): return values", "expected function"),
        ("def normalize_event_codes(codes):\n    return open('/tmp/x')", "open"),
        ("def normalize_event_codes(codes):\n    return codes.__class__", "private"),
    ],
)
def test_python_evaluator_rejects_unsafe_or_invalid_shapes(
    source: str, message: str
) -> None:
    result = evaluate_python(_coding_item(), source)

    assert not result.passed
    assert result.details["reason"] == "rejected_source"
    assert message in result.details["validation_error"]


def test_runner_dispatches_python_evaluator_and_saves_evidence(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake model")
    config_path = tmp_path / "coding.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: coding-runner-test
  workload_path: {CODING_SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  repetitions: 1
  seed: 42
models:
  - id: fake-local-model
    backend: llama_cpp
    model_path: {model_path}
""".strip(),
        encoding="utf-8",
    )

    class CodingBackend:
        def generate(self, prompt, generation, *, seed):
            return GenerationOutput(
                    text=(
                        "def normalize_event_codes(codes):\n"
                        "    result = []\n"
                        "    for code in codes:\n"
                        "        value = code.strip().upper()\n"
                        "        if value and value not in result:\n"
                        "            result.append(value)\n"
                        "    return result"
                ),
                prompt_tokens=20,
                output_tokens=30,
            )

    run_directory = run_benchmark(
        load_config(config_path),
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, path, seed: CodingBackend(),
    )
    records = [
        json.loads(line)
        for line in (run_directory / "results.jsonl").read_text().splitlines()
    ]
    record = next(
        candidate
        for candidate in records
        if candidate["item_id"] == "code_normalize_event_codes_001"
    )

    assert record["evaluation"]["type"] == "executable"
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["test_count"] == 4
