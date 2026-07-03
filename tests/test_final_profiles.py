from pathlib import Path

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


ROOT = Path(__file__).resolve().parents[1]


def test_final_quantization_profiles_are_independent_and_complete() -> None:
    five = load_config(ROOT / "configs" / "final_default_matrix.yaml")
    retrieval = load_config(ROOT / "configs" / "final_retrieval_matrix.yaml")

    assert len(five.models) == len(retrieval.models) == 20
    assert five.benchmark.workload_path == Path("data/suites/final_five.yaml")
    assert retrieval.benchmark.workload_path == Path(
        "data/suites/final_retrieval.yaml"
    )
    assert five.benchmark.repetitions == retrieval.benchmark.repetitions == 1
    assert five.judge is retrieval.judge is None

    expected_models = {
        (architecture, quantization)
        for architecture in (
            "llama-3.1-8b",
            "qwen3-8b",
            "mistral-7b-v0.3",
            "gemma-3-12b",
            "qwen2.5-coder-7b",
        )
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M")
    }
    assert {(model.architecture, model.quantization) for model in five.models} == (
        expected_models
    )
    assert [model.model_dump() for model in retrieval.models] == [
        model.model_dump() for model in five.models
    ]


def test_final_profiles_match_pinned_model_sources() -> None:
    config = load_config(ROOT / "configs" / "final_default_matrix.yaml")
    sources = yaml.safe_load(
        (ROOT / "data" / "model_sources.yaml").read_text(encoding="utf-8")
    )
    pinned = {
        (family["architecture"], file["quantization"], file["local_path"])
        for family in sources["families"].values()
        for file in family["files"]
    }

    assert len(sources["families"]) == 5
    assert len(pinned) == 20
    assert {
        (model.architecture, model.quantization, str(model.model_path))
        for model in config.models
    } == pinned


def test_split_suites_partition_the_frozen_six() -> None:
    frozen = load_suite(ROOT / "data" / "suites" / "final_six.yaml")
    five = load_suite(ROOT / "data" / "suites" / "final_five.yaml")
    retrieval = load_suite(ROOT / "data" / "suites" / "final_retrieval.yaml")

    assert set(five.items) == {
        "applied_reasoning",
        "code_debug_repair",
        "messy_text_to_schema",
        "constraint_load_curve",
        "tool_use",
    }
    assert set(retrieval.items) == {"long_text_retrieval"}
    assert set(five.items).isdisjoint(retrieval.items)
    assert set(frozen.items) == set(five.items) | set(retrieval.items)
    assert sum(map(len, five.items.values())) == 272
    assert sum(map(len, retrieval.items.values())) == 48
    assert sum(map(len, frozen.items.values())) == 320


def test_split_profiles_keep_the_6400_generation_budget() -> None:
    attempts = 0
    for config_name, suite_name in (
        ("final_default_matrix.yaml", "final_five.yaml"),
        ("final_retrieval_matrix.yaml", "final_retrieval.yaml"),
    ):
        config = load_config(ROOT / "configs" / config_name)
        suite = load_suite(ROOT / "data" / "suites" / suite_name)
        attempts += len(config.models) * sum(map(len, suite.items.values()))

    assert attempts == 6_400
