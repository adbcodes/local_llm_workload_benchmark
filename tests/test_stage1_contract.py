from __future__ import annotations

import hashlib
from pathlib import Path
import re

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "data" / "stage1_contract.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage1_dataset_and_evaluator_contract_is_frozen() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    suite_contract = contract["suite"]
    suite_path = ROOT / suite_contract["path"]
    suite = load_suite(suite_path)

    assert _sha256(suite_path) == suite_contract["sha256"]
    assert sum(map(len, suite.items.values())) == suite_contract["question_count"]
    assert {
        dataset["id"]: dataset["questions"]
        for dataset in suite_contract["datasets"]
    } == {benchmark: len(items) for benchmark, items in suite.items.items()}

    for entry in suite_contract["datasets"]:
        assert _sha256(ROOT / entry["path"]) == entry["sha256"]
    for entry in contract["evaluators"]["implementation_files"]:
        assert _sha256(ROOT / entry["path"]) == entry["sha256"]

    assert contract["evaluators"]["versions"] == {
        "numeric_tolerance": 2,
        "restricted_python_tests": 2,
        "json_exact": 1,
        "exact_match": 1,
        "constraint_rules": 1,
        "tool_call": 1,
    }


def test_stage1_model_inventory_matches_the_main_config() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    run_contract = contract["main_run"]
    config_path = ROOT / run_contract["config_path"]
    config = load_config(config_path)
    frozen_models = {entry["id"]: entry for entry in contract["models"]}

    assert _sha256(config_path) == run_contract["config_sha256"]
    assert len(config.models) == len(frozen_models) == 20
    assert set(frozen_models) == {model.id for model in config.models}
    assert config.benchmark.repetitions == run_contract["repetitions"] == 1

    for model in config.models:
        frozen = frozen_models[model.id]
        model_path = ROOT / model.model_path
        assert frozen["path"] == str(model.model_path)
        assert frozen["quantization"] == model.quantization
        assert model_path.stat().st_size == frozen["bytes"]
        assert re.fullmatch(r"[0-9a-f]{64}", frozen["sha256"])
        assert model.generation.model_dump() == run_contract["generation"]

    assert run_contract["command"].endswith(
        "llm-benchmark benchmark --config configs/final_default_matrix.yaml "
        "--skip-human-eval"
    )


def test_stage1_smoke_contract_spans_families_and_quantizations() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    smoke_contract = contract["preflight_smoke"]
    config_path = ROOT / smoke_contract["config_path"]
    config = load_config(config_path)

    assert _sha256(config_path) == smoke_contract["config_sha256"]
    assert len(config.models) == smoke_contract["models_completed"] == 5
    assert smoke_contract["models_failed"] == 0
    assert smoke_contract["item_records"] == 30
    assert {model.quantization for model in config.models} == set(
        smoke_contract["quantizations"]
    )
    assert {model.id for model in config.models} == {
        "qwen2.5-3b-q3",
        "gemma-3-4b-q6",
        "qwen3-8b-q8",
        "phi-4-mini-q4",
        "llama-3.1-8b-q4",
    }
