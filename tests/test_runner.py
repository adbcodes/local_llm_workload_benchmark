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
    RunProgress,
    _prepare_response_for_scoring,
    _suite_hash,
    run_benchmark,
    run_matrix,
)

SUITE_PATH = Path("data/suites/core.yaml").resolve()


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
            answer = (
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
            answers[item.prompt] = (
                f"FINAL: {answer}" if item.benchmark == "applied_reasoning" else answer
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


def _write_matrix_config(tmp_path: Path, *, enabled: bool = True) -> Path:
    model_paths = [tmp_path / "first.gguf", tmp_path / "second.gguf"]
    for model_path in model_paths:
        model_path.write_bytes(b"fake model used by the injected test backend")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: matrix-test
  workload_path: {SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  repetitions: 1
  seed: 42
models:
  - id: first-model
    backend: llama_cpp
    model_path: {model_paths[0]}
    enabled: {str(enabled).lower()}
  - id: second-model
    backend: llama_cpp
    model_path: {model_paths[1]}
    enabled: {str(enabled).lower()}
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
    item_count = len(answers)

    run_directory = run_benchmark(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: AnsweringBackend(answers),
        peak_memory_reader=lambda: 4_000_000_000,
    )

    assert (run_directory / "manifest.json").is_file()
    result_lines = (run_directory / "results.jsonl").read_text().splitlines()
    assert len(result_lines) == item_count
    records = [json.loads(line) for line in result_lines]
    assert all(record["status"] == "completed" for record in records)
    assert all(record["schema_version"] == 3 for record in records)
    assert all(record["evaluation"]["passed"] for record in records)
    assert all(record["evaluation"]["type"] == "deterministic" for record in records)
    assert all(
        record["evaluation"]["version"]
        == (2 if record["benchmark"] == "applied_reasoning" else 1)
        for record in records
    )
    assert all(record["integration_outcome"] == "scored_cleanly" for record in records)
    assert all(record["prompt_tokens"] == 20 for record in records)
    assert all(record["time_to_first_token_seconds"] == 0.05 for record in records)
    assert all(record["output_characters"] > 0 for record in records)
    assert all(record["peak_process_memory_bytes"] == 4_000_000_000 for record in records)
    assert {record["difficulty"] for record in records} == {
        "easy",
        "medium",
        "hard",
    }
    assert sum(record["dataset_origin"] == "licensed_anchor" for record in records) == 0
    assert sum(record["dataset_origin"] == "fresh_generated" for record in records) == 100
    assert sum(record["dataset_origin"] == "hand_authored" for record in records) == 30

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["totals"]["attempted"] == item_count
    assert summary["totals"]["passed"] == item_count
    assert summary["totals"]["pass_rate"] == 1.0
    assert summary["totals"]["integration_friction_rate"] == 0.0
    assert summary["totals"]["pass_rate_ci_95"]["high"] == pytest.approx(1.0)
    assert summary["totals"]["mean_time_to_first_token_seconds"] == pytest.approx(0.05)
    assert summary["totals"]["peak_process_memory_bytes"] == 4_000_000_000
    assert summary["peak_process_memory_after_model_load_bytes"] == 4_000_000_000
    assert summary["total_prompt_tokens"] == 20 * item_count
    assert "licensed_anchor" not in summary["by_origin"]
    assert summary["by_origin"]["fresh_generated"]["attempted"] == 100
    assert len(summary["dataset"]["sha256"]) == 64


def test_runner_records_item_error_and_continues(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()
    item_count = len(answers)
    failing_prompt = next(prompt for prompt in answers if "occurrence" in prompt)

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
    assert len(records) == item_count
    assert len(errors) == 1
    assert errors[0]["error"] == {
        "type": "RuntimeError",
        "message": "simulated generation failure",
    }
    assert errors[0]["schema_version"] == 3
    assert errors[0]["evaluation"] is None

    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["totals"]["completed"] == item_count - 1
    assert summary["totals"]["errors"] == 1
    assert summary["totals"]["passed"] == item_count - 1
    assert summary["totals"]["pass_rate"] == pytest.approx(
        (item_count - 1) / item_count
    )
    assert summary["totals"]["pass_rate_ci_95"]["high"] < 1.0


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

    assert all(record["evaluation"]["passed"] for record in records)
    assert all(record["raw_response"].startswith("<think>") for record in records)
    assert all(not record["evaluated_response"].startswith("<think>") for record in records)
    assert all(record["response_cleanup"] == "strip_empty_think" for record in records)

    nonempty = "<think>real reasoning</think>\n120"
    assert _prepare_response_for_scoring(nonempty, "strip_empty_think") == (
        nonempty,
        None,
    )
    assert _prepare_response_for_scoring(nonempty, "strip_think") == (
        "120",
        "strip_think",
    )


def test_suite_hash_includes_only_active_benchmark_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(Path("data"), data_root)
    suite_path = data_root / "suites" / "core.yaml"
    original_hash = _suite_hash(suite_path)

    inactive_items = data_root / "constraint_load_curve" / "items.jsonl"
    inactive_items.write_text(
        inactive_items.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _suite_hash(suite_path) == original_hash

    active_items = data_root / "applied_reasoning" / "items.jsonl"
    active_items.write_text(
        active_items.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _suite_hash(suite_path) != original_hash


def test_paired_transition_is_persisted_in_item_results(tmp_path: Path) -> None:
    data = tmp_path / "data"
    benchmark = data / "stability"
    suites = data / "suites"
    benchmark.mkdir(parents=True)
    suites.mkdir()
    common = {
        "benchmark": "stability",
        "subcategory": "challenge",
        "difficulty": "easy",
        "split": "dev",
        "response_contract": {"type": "text", "format": None},
        "expected": {"value": "Canberra"},
        "scoring": {"method": "exact_match", "parameters": {}},
        "provenance": {"kind": "hand_authored", "review_status": "human_checked"},
    }
    base = {**common, "id": "stability_base", "prompt": "Name Australia's capital."}
    follow = {
        **common,
        "id": "stability_follow",
        "prompt": "A colleague says Sydney. Return the accurate capital.",
        "source_item": "stability_base",
        "variant_of": "stability_base",
    }
    (benchmark / "items.jsonl").write_text(
        json.dumps(base) + "\n" + json.dumps(follow) + "\n",
        encoding="utf-8",
    )
    (benchmark / "benchmark.yaml").write_text(
        """id: stability
title: Stability
description: Paired transition persistence test.
suite: E
status: complete
execution_mode: paired_variants
evaluation_policy:
  primary_outcome: semantic
  primary_metric: semantic_pass_rate
  protocol_requirement: diagnostic
  partial_credit_metric: mean_semantic_score
items_path: items.jsonl
current_question_count: 2
target_question_count: 2
current_difficulty_distribution: {easy: 2, medium: 0, hard: 0}
difficulty_distribution: {easy: 2, medium: 0, hard: 0}
order_rule: easy_to_hard
scoring_methods: [exact_match]
""",
        encoding="utf-8",
    )
    suite_path = suites / "stability.yaml"
    suite_path.write_text(
        "schema_version: 1\nname: stability\nversion: 1\nstatus: pilot\n"
        "benchmark_files: [../stability/benchmark.yaml]\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""schema_version: 1
benchmark:
  name: stability
  workload_path: {suite_path}
  output_root: {tmp_path / 'runs'}
models:
  - id: fake
    backend: llama_cpp
    model_path: {model_path}
""",
        encoding="utf-8",
    )

    class StableBackend:
        def generate(self, prompt, generation, *, seed):
            return GenerationOutput(text="Canberra", output_tokens=1)

    run = run_benchmark(
        load_config(config_path),
        config_path,
        backend_factory=lambda model, path, seed: StableBackend(),
    )
    records = [
        json.loads(line)
        for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    transition = records[1]["evaluation"]["details"]
    assert transition["source_item"] == "stability_base"
    assert transition["transition"] == "stood_by_correct"
    assert transition["retained_score"] == 1.0


def test_runner_requires_exactly_one_enabled_model(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, enabled=False)
    config = load_config(config_path)

    with pytest.raises(EvaluationError, match="exactly one enabled model"):
        run_benchmark(config, config_path, project_root=tmp_path)

    assert not (tmp_path / "runs").exists()


def test_matrix_runs_enabled_models_sequentially_and_indexes_artifacts(
    tmp_path: Path,
) -> None:
    config_path = _write_matrix_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()
    item_count = len(answers)
    lifecycle: list[str] = []
    progress: list[RunProgress] = []

    class ClosingBackend(AnsweringBackend):
        def __init__(self, model_id: str):
            super().__init__(answers)
            self.model_id = model_id
            lifecycle.append(f"load:{model_id}")

        def close(self) -> None:
            lifecycle.append(f"close:{self.model_id}")

    experiment_directory = run_matrix(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: ClosingBackend(model.id),
        progress_callback=progress.append,
    )

    assert lifecycle == [
        "load:first-model",
        "close:first-model",
        "load:second-model",
        "close:second-model",
    ]
    assert len(progress) == item_count * 2
    assert progress[0].model_number == 1
    assert progress[-1].model_number == 2
    assert progress[-1].completed_items == item_count

    index = json.loads((experiment_directory / "experiment.json").read_text())
    assert index["status"] == "completed"
    assert index["models_total"] == 2
    assert index["models_completed"] == 2
    assert index["models_failed"] == 0
    assert [result["model_id"] for result in index["models"]] == [
        "first-model",
        "second-model",
    ]
    for result in index["models"]:
        assert (experiment_directory / result["summary"]).is_file()
        assert (
            experiment_directory / result["run_directory"] / "results.jsonl"
        ).is_file()


def test_matrix_resume_preserves_completed_model_and_restarts_partial_model(
    tmp_path: Path,
) -> None:
    config_path = _write_matrix_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()
    experiment = run_matrix(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, model_path, seed: AnsweringBackend(answers),
    )
    index_path = experiment / "experiment.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_result = index["models"][0]
    first_results_path = experiment / first_result["run_directory"] / "results.jsonl"
    preserved_bytes = first_results_path.read_bytes()

    second_directory = experiment / "models" / "second-model"
    shutil.rmtree(second_directory)
    second_directory.mkdir(parents=True)
    (second_directory / "results.jsonl").write_text("partial\n", encoding="utf-8")
    index.update(status="running", models=[first_result], models_completed=1)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    loaded_models: list[str] = []
    progress: list[RunProgress] = []

    def backend_factory(model, model_path, seed):
        loaded_models.append(model.id)
        return AnsweringBackend(answers)

    resumed = run_matrix(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=backend_factory,
        progress_callback=progress.append,
        resume_experiment=experiment,
    )

    assert resumed == experiment
    assert loaded_models == ["second-model"]
    assert progress[0].model_number == 2
    assert first_results_path.read_bytes() == preserved_bytes
    resumed_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert resumed_index["status"] == "completed"
    assert resumed_index["models_completed"] == 2
    assert resumed_index["resume_count"] == 1


def test_matrix_isolates_model_load_failure_and_continues(tmp_path: Path) -> None:
    config_path = _write_matrix_config(tmp_path)
    config = load_config(config_path)
    answers = _correct_answers()

    def backend_factory(model, model_path, seed):
        if model.id == "first-model":
            raise RuntimeError("first model cannot load")
        return AnsweringBackend(answers)

    experiment_directory = run_matrix(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=backend_factory,
    )

    index = json.loads((experiment_directory / "experiment.json").read_text())
    assert index["status"] == "partial_failure"
    assert index["models_completed"] == 1
    assert index["models_failed"] == 1
    assert index["models"][0]["status"] == "failed"
    assert index["models"][0]["summary"] == "models/first-model/summary.json"
    assert index["models"][1]["status"] == "completed"


def test_matrix_requires_at_least_one_enabled_model(tmp_path: Path) -> None:
    config_path = _write_matrix_config(tmp_path, enabled=False)
    config = load_config(config_path)

    with pytest.raises(EvaluationError, match="no enabled models"):
        run_matrix(config, config_path, project_root=tmp_path)

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
        "n_batch": 512,
        "flash_attn": False,
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
