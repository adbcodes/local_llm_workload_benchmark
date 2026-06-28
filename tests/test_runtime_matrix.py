import csv
import json
from pathlib import Path

import pytest

from llm_workload_benchmark.config import GenerationConfig
from llm_workload_benchmark.dataset import DatasetItem
from llm_workload_benchmark.runner import GenerationOutput, _generation_for_item
from llm_workload_benchmark.runtime_matrix import (
    RuntimeMatrixError,
    combination_count,
    expand_runtime_matrix,
    load_runtime_matrix,
    run_runtime_matrix,
    validate_model_files,
)


def test_default_runtime_matrix_expands_four_quantizations_and_sampler_axes() -> None:
    runtime = load_runtime_matrix(Path("configs/runtime_matrix.yaml"))
    expanded = expand_runtime_matrix(runtime)

    assert combination_count(runtime) == 32
    assert len(expanded.models) == 32
    assert {model.quantization for model in expanded.models} == {
        "Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"
    }
    assert {model.generation.temperature for model in expanded.models} == {0.0, 0.7}
    assert {model.generation.repeat_penalty for model in expanded.models} == {1.0, 1.1}
    assert {model.generation.constrained_decoding for model in expanded.models} == {
        "none", "json_when_requested"
    }
    assert len({model.id for model in expanded.models}) == 32


def test_runtime_matrix_preflight_lists_missing_model_files(tmp_path: Path) -> None:
    runtime = load_runtime_matrix(Path("configs/runtime_matrix.yaml"))
    expanded = expand_runtime_matrix(runtime)

    with pytest.raises(RuntimeMatrixError, match="missing quantization model files"):
        validate_model_files(expanded, tmp_path)


def test_item_aware_json_decoding_only_changes_json_tasks() -> None:
    generation = GenerationConfig(constrained_decoding="json_when_requested")
    base = {
        "id": "runtime_item_001", "benchmark": "runtime", "subcategory": "format",
        "difficulty": "easy", "split": "dev", "prompt": "Return the answer.",
        "expected": {"value": "blue"},
        "scoring": {"method": "exact_match", "parameters": {}},
        "provenance": {"kind": "hand_authored", "review_status": "draft"},
    }
    text_item = DatasetItem.model_validate(
        {**base, "response_contract": {"type": "text", "format": None}}
    )
    json_item = DatasetItem.model_validate(
        {
            **base,
            "id": "runtime_item_002",
            "response_contract": {"type": "json", "format": None},
            "expected": {"value": {"answer": "blue"}},
            "scoring": {"method": "json_exact", "parameters": {}},
        }
    )

    assert _generation_for_item(generation, text_item).constrained_decoding == "none"
    assert _generation_for_item(generation, json_item).constrained_decoding == "json"


def _write_runtime_fixture(tmp_path: Path) -> Path:
    benchmark = tmp_path / "data" / "tiny"
    suites = tmp_path / "data" / "suites"
    benchmark.mkdir(parents=True)
    suites.mkdir()
    item = {
        "id": "tiny_001", "benchmark": "tiny", "subcategory": "fact",
        "difficulty": "easy", "split": "dev", "visibility": "public",
        "prompt": "Return blue.",
        "response_contract": {"type": "text", "format": None},
        "expected": {"value": "blue"},
        "scoring": {"method": "exact_match", "parameters": {}},
        "provenance": {"kind": "hand_authored", "review_status": "draft"},
        "tags": [],
    }
    (benchmark / "items.jsonl").write_text(json.dumps(item) + "\n")
    (benchmark / "benchmark.yaml").write_text(
        "id: tiny\ntitle: Tiny\ndescription: Tiny runtime fixture.\nsuite: A\n"
        "status: started\nexecution_mode: single_turn\ntask_types: [fact]\n"
        "metrics: [accuracy]\nscore_formula: mean_score\n"
        "evaluation_policy:\n  primary_outcome: semantic\n"
        "  primary_metric: semantic_pass_rate\n"
        "  protocol_requirement: diagnostic\n"
        "  partial_credit_metric: mean_semantic_score\nitems_path: items.jsonl\n"
        "authoring_paths: []\ncurrent_question_count: 1\ntarget_question_count: 1\n"
        "current_difficulty_distribution: {easy: 1, medium: 0, hard: 0}\n"
        "difficulty_distribution: {easy: 1, medium: 0, hard: 0}\n"
        "target_visibility_distribution: {public: 1, held_out: 0}\n"
        "order_rule: easy_to_hard\nscoring_methods: [exact_match]\n"
    )
    (suites / "all.yaml").write_text(
        "schema_version: 1\nname: tiny\nversion: 1\nstatus: pilot\n"
        "benchmark_files: [../tiny/benchmark.yaml]\n"
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "q8.gguf").write_bytes(b"q8")
    (models / "q4.gguf").write_bytes(b"q4")
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "schema_version: 1\n"
        "benchmark:\n  name: tiny-runtime\n  workload_path: data/suites/all.yaml\n"
        "  output_root: runs\n  repetitions: 1\n  seed: 42\n"
        "model:\n  backend: llama_cpp\n  family: fake\n  context_window: 1024\n"
        "  gpu_layers: 0\n  batch_size: 32\n  flash_attention: false\n"
        "  response_cleanup: none\n  verbose: false\n  system_prompt: Return answers.\n"
        "quantizations:\n"
        "  - {id: fake-q8, quantization: Q8_0, model_path: models/q8.gguf}\n"
        "  - {id: fake-q4, quantization: Q4_K_M, model_path: models/q4.gguf}\n"
        "axes:\n  temperature: [0.0, 0.7]\n  top_p: [1.0]\n  top_k: [40]\n"
        "  repeat_penalty: [1.0]\n  max_output_tokens: [16]\n"
        "  constrained_decoding: [none]\n"
    )
    return config


def test_runtime_matrix_runs_all_combinations_and_exports_graph_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_runtime_fixture(tmp_path)
    runtime = load_runtime_matrix(config_path)
    monkeypatch.setattr(
        "llm_workload_benchmark.manifest._graphics_details", lambda: []
    )

    class FakeBackend:
        def generate(self, prompt, generation, *, seed):
            return GenerationOutput(
                text="blue", output_tokens=1, time_to_first_token_seconds=0.01
            )

    experiment = run_runtime_matrix(
        runtime,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, path, seed: FakeBackend(),
        peak_memory_reader=lambda: 123_456,
    )

    artifact_root = experiment / "artifacts"
    artifact_manifest = json.loads((artifact_root / "manifest.json").read_text())
    assert artifact_manifest["experiment_metadata"]["combination_count"] == 4
    assert artifact_manifest["tables"]["configurations"]["row_count"] == 4
    assert artifact_manifest["tables"]["benchmarks"]["row_count"] == 4
    assert artifact_manifest["tables"]["items"]["row_count"] == 4

    with (artifact_root / "data" / "configurations.csv").open(newline="") as source:
        run_rows = list(csv.DictReader(source))
    with (artifact_root / "data" / "items.csv").open(newline="") as source:
        item_rows = list(csv.DictReader(source))
    assert len(run_rows) == 4
    assert len(item_rows) == 4
    assert all(row["status"] == "completed" for row in run_rows)
    assert all(row["score_retained_vs_q8"] == "1.0" for row in run_rows)
    assert {row["quantization"] for row in run_rows} == {"Q8_0", "Q4_K_M"}
    assert {row["temperature"] for row in run_rows} == {"0.0", "0.7"}
    assert all(row["ttft_seconds"] == "0.01" for row in item_rows)
