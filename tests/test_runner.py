import json
import shutil
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
    _prepare_response_for_scoring,
    _suite_hash,
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
            time_to_first_token_seconds=0.05,
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


def _write_config(
    tmp_path: Path,
    *,
    enabled: bool = True,
    response_cleanup: str = "none",
) -> Path:
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
    response_cleanup: {response_cleanup}
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
        peak_memory_reader=lambda: 4_000_000_000,
    )

    assert (run_directory / "manifest.json").is_file()
    result_lines = (run_directory / "results.jsonl").read_text().splitlines()
    assert len(result_lines) == 6
    records = [json.loads(line) for line in result_lines]
    assert all(record["status"] == "completed" for record in records)
    assert all(record["passed"] for record in records)
    assert all(record["prompt_tokens"] == 20 for record in records)
    assert all(record["time_to_first_token_seconds"] == 0.05 for record in records)
    assert all(record["output_characters"] > 0 for record in records)
    assert all(record["peak_process_memory_bytes"] == 4_000_000_000 for record in records)
    assert {record["difficulty"] for record in records} == {
        "easy",
        "medium",
        "hard",
    }

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["totals"]["attempted"] == 6
    assert summary["totals"]["passed"] == 6
    assert summary["totals"]["pass_rate"] == 1.0
    assert summary["totals"]["mean_time_to_first_token_seconds"] == pytest.approx(0.05)
    assert summary["totals"]["peak_process_memory_bytes"] == 4_000_000_000
    assert summary["peak_process_memory_after_model_load_bytes"] == 4_000_000_000
    assert summary["total_prompt_tokens"] == 120
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
    assert len(records) == 6
    assert len(errors) == 1
    assert errors[0]["error"] == {
        "type": "RuntimeError",
        "message": "simulated generation failure",
    }

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["totals"]["completed"] == 5
    assert summary["totals"]["errors"] == 1
    assert summary["totals"]["passed"] == 5


def test_runner_applies_only_configured_empty_think_cleanup(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        response_cleanup="strip_empty_think",
    )
    config = load_config(config_path)
    answers = _correct_answers()

    class EmptyThinkBackend(AnsweringBackend):
        def generate(self, prompt, generation, *, seed):
            output = super().generate(prompt, generation, seed=seed)
            return GenerationOutput(
                text=f"<think>\n\n</think>\n\n{output.text}",
                prompt_tokens=output.prompt_tokens,
                output_tokens=output.output_tokens,
                time_to_first_token_seconds=output.time_to_first_token_seconds,
                finish_reason=output.finish_reason,
            )

    run_directory = run_benchmark(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: EmptyThinkBackend(answers),
        peak_memory_reader=lambda: 4_000_000_000,
    )
    records = [
        json.loads(line)
        for line in (run_directory / "results.jsonl").read_text().splitlines()
    ]

    assert all(record["passed"] for record in records)
    assert all(record["raw_response"].startswith("<think>") for record in records)
    assert all(not record["evaluated_response"].startswith("<think>") for record in records)
    assert all(record["response_cleanup"] == "strip_empty_think" for record in records)

    nonempty = "<think>real reasoning</think>\n120"
    assert _prepare_response_for_scoring(nonempty, "strip_empty_think") == (
        nonempty,
        None,
    )


def test_suite_hash_includes_only_active_benchmark_files(tmp_path: Path) -> None:
    suite_root = tmp_path / "v1"
    shutil.copytree(Path("data/benchmarks/v1"), suite_root)
    suite_path = suite_root / "suite.yaml"
    original_hash = _suite_hash(suite_path)

    inactive_items = suite_root / "constraint_load_curve" / "items.jsonl"
    inactive_items.write_text(
        inactive_items.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _suite_hash(suite_path) == original_hash

    active_items = suite_root / "applied_reasoning" / "items.jsonl"
    active_items.write_text(
        active_items.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _suite_hash(suite_path) != original_hash


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


def test_llama_cpp_adapter_streams_and_extracts_performance_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, **arguments):
            captured["load"] = arguments
            self.n_tokens = 17

        def create_chat_completion(self, **arguments):
            captured["generation"] = arguments
            return iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {"delta": {"content": "1"}, "finish_reason": None}
                        ]
                    },
                    {
                        "choices": [
                            {"delta": {"content": "20"}, "finish_reason": None}
                        ]
                    },
                    {
                        "choices": [
                            {"delta": {}, "finish_reason": "stop"}
                        ]
                    },
                ]
            )

        def tokenize(self, text, *, add_bos, special):
            captured["tokenize"] = {
                "text": text,
                "add_bos": add_bos,
                "special": special,
            }
            return [120]

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    clock_values = iter([10.0, 10.25])
    monkeypatch.setattr(
        "llm_workload_benchmark.runner.time.perf_counter",
        lambda: next(clock_values),
    )
    config_path = _write_config(tmp_path)
    model = load_config(config_path).models[0]
    backend = LlamaCppBackend(model, model.model_path, seed=42)

    output = backend.generate("What is 15% of 800?", model.generation, seed=43)

    assert output == GenerationOutput(
        text="120",
        prompt_tokens=17,
        output_tokens=1,
        time_to_first_token_seconds=0.25,
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
    assert generation["stream"] is True
    assert generation["messages"][1] == {
        "role": "user",
        "content": "What is 15% of 800?",
    }
    assert captured["tokenize"] == {
        "text": b"120",
        "add_bos": False,
        "special": True,
    }
