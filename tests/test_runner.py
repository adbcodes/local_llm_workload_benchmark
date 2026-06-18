import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite
from llm_workload_benchmark.runner import (
    EvaluationError,
    GenerationOutput,
    LlamaCppBackend,
    run_benchmark,
)

SUITE_PATH = Path("data/benchmarks/v1/suite.yaml").resolve()


class AnsweringBackend:
    def __init__(self, answers: dict[str, str], failing_prompt: str | None = None):
        self.answers = answers
        self.failing_prompt = failing_prompt

    def generate(self, prompt, generation, *, seed):
        if prompt == self.failing_prompt:
            raise RuntimeError("simulated generation failure")
        answer = self.answers[prompt]
        return GenerationOutput(
            text=answer,
            prompt_tokens=20,
            output_tokens=max(1, len(answer.split())),
            finish_reason="stop",
        )


def _correct_answers() -> dict[str, str]:
    suite = load_suite(SUITE_PATH)
    answers: dict[str, str] = {}
    for items in suite.items.values():
        for item in items:
            value = item.expected["value"]
            answers[item.prompt] = (
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
    return answers


def _write_config(tmp_path: Path, *, enabled: bool = True) -> Path:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake model used by the injected test backend")
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: runner-test
  workload_path: {SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  repetitions: 1
  seed: 42
models:
  - id: fake-model
    backend: llama_cpp
    model_path: {model_path}
    quantization: test
    enabled: {str(enabled).lower()}
    context_window: 2048
    gpu_layers: 0
    generation:
      max_output_tokens: 64
      temperature: 0.0
      top_p: 1.0
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_runner_evaluates_all_pilot_items_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()

    run_directory = run_benchmark(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: AnsweringBackend(answers),
    )

    assert (run_directory / "manifest.json").is_file()
    result_lines = (run_directory / "results.jsonl").read_text().splitlines()
    assert len(result_lines) == 9
    records = [json.loads(line) for line in result_lines]
    assert all(record["status"] == "completed" for record in records)
    assert all(record["passed"] for record in records)
    assert all(record["prompt_tokens"] == 20 for record in records)
    assert {record["difficulty"] for record in records} == {
        "easy",
        "medium",
        "hard",
    }

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["totals"]["attempted"] == 9
    assert summary["totals"]["passed"] == 9
    assert summary["totals"]["pass_rate"] == 1.0
    assert summary["total_prompt_tokens"] == 180
    assert len(summary["dataset"]["sha256"]) == 64


def test_runner_records_item_error_and_continues(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()
    failing_prompt = next(prompt for prompt in answers if "seventh occurrence" in prompt)

    run_directory = run_benchmark(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: AnsweringBackend(
            answers,
            failing_prompt=failing_prompt,
        ),
    )

    records = [
        json.loads(line)
        for line in (run_directory / "results.jsonl").read_text().splitlines()
    ]
    errors = [record for record in records if record["status"] == "error"]
    assert len(records) == 9
    assert len(errors) == 1
    assert errors[0]["error"] == {
        "type": "RuntimeError",
        "message": "simulated generation failure",
    }

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["totals"]["completed"] == 8
    assert summary["totals"]["errors"] == 1
    assert summary["totals"]["passed"] == 8


def test_runner_requires_exactly_one_enabled_model(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, enabled=False)
    config = load_config(config_path)

    with pytest.raises(EvaluationError, match="exactly one enabled model"):
        run_benchmark(config, config_path, project_root=tmp_path)

    assert not (tmp_path / "runs").exists()


def test_runner_preserves_model_load_failure_summary(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)

    def fail_to_load(model, model_path, seed):
        raise RuntimeError("simulated load failure")

    with pytest.raises(EvaluationError, match="failure summary"):
        run_benchmark(
            config,
            config_path,
            project_root=tmp_path,
            backend_factory=fail_to_load,
        )

    run_directories = list((tmp_path / "runs").iterdir())
    assert len(run_directories) == 1
    summary = json.loads((run_directories[0] / "summary.json").read_text())
    assert summary["status"] == "model_load_error"
    assert summary["error"] == {
        "type": "RuntimeError",
        "message": "simulated load failure",
    }
    assert not (run_directories[0] / "results.jsonl").exists()


def test_llama_cpp_adapter_uses_config_and_extracts_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, **arguments):
            captured["load"] = arguments

        def create_chat_completion(self, **arguments):
            captured["generation"] = arguments
            return {
                "choices": [
                    {
                        "message": {"content": "120"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 17, "completion_tokens": 1},
            }

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    config_path = _write_config(tmp_path)
    model = load_config(config_path).models[0]
    backend = LlamaCppBackend(model, model.model_path, seed=42)

    output = backend.generate("What is 15% of 800?", model.generation, seed=43)

    assert output == GenerationOutput(
        text="120",
        prompt_tokens=17,
        output_tokens=1,
        finish_reason="stop",
    )
    assert captured["load"] == {
        "model_path": str(model.model_path),
        "n_ctx": 2048,
        "n_gpu_layers": 0,
        "seed": 42,
        "verbose": False,
    }
    generation = captured["generation"]
    assert generation["seed"] == 43
    assert generation["messages"][1] == {
        "role": "user",
        "content": "What is 15% of 800?",
    }
