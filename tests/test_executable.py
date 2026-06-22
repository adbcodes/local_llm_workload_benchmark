import json
from pathlib import Path

import pytest

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import DatasetError, load_suite, score_answer
from llm_workload_benchmark.executable import evaluate_python
from llm_workload_benchmark.runner import GenerationOutput, run_benchmark

CODING_SUITE_PATH = Path("data/suites/coding.yaml").resolve()


def _coding_item():
    return load_suite(CODING_SUITE_PATH).items["code_debug_repair"][0]


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
def deduplicate_preserving_order(values):
    result = []
    for value in values:
        if value not in result:
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
        "def deduplicate_preserving_order(values):\n    return values",
    )
    assert not partial.passed
    assert partial.score == pytest.approx(0.25)
    assert partial.details["tests_passed"] == 1


def test_python_evaluator_stops_infinite_loop_at_timeout() -> None:
    result = evaluate_python(
        _coding_item(),
        "def deduplicate_preserving_order(values):\n    while True:\n        pass",
    )

    assert not result.passed
    assert result.score == 0
    assert result.details["reason"] in {"timeout", "resource_limit"}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\ndef deduplicate_preserving_order(values): return values", "one function"),
        ("def wrong_name(values): return values", "expected function"),
        ("def deduplicate_preserving_order(values):\n    return open('/tmp/x')", "open"),
        ("def deduplicate_preserving_order(values):\n    return values.__class__", "private"),
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
                    "def deduplicate_preserving_order(values):\n"
                    "    result = []\n"
                    "    for value in values:\n"
                    "        if value not in result:\n"
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
    record = json.loads((run_directory / "results.jsonl").read_text())

    assert record["evaluation"]["type"] == "executable"
    assert record["evaluation"]["passed"] is True
    assert record["evaluation"]["details"]["test_count"] == 4
