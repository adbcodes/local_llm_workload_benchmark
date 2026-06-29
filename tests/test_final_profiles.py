from pathlib import Path

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


ROOT = Path(__file__).resolve().parents[1]


def test_final_profiles_have_expected_configuration_counts() -> None:
    default = load_config(ROOT / "configs" / "final_default_matrix.yaml")
    temperature = load_config(ROOT / "configs" / "final_temperature_matrix.yaml")
    constrained = load_config(ROOT / "configs" / "final_constrained_matrix.yaml")
    repetition = load_config(ROOT / "configs" / "final_repetition_matrix.yaml")
    context = load_config(ROOT / "configs" / "final_context_matrix.yaml")

    assert len(default.models) == 12
    assert len(temperature.models) == 12
    assert len(constrained.models) == 12
    assert len(repetition.models) == 12
    assert len(context.models) == 12
    assert temperature.benchmark.repetitions == 3
    assert constrained.benchmark.repetitions == 1
    assert repetition.benchmark.repetitions == 1
    assert {model.generation.temperature for model in temperature.models} == {0.7}
    assert {model.generation.constrained_decoding for model in constrained.models} == {
        "json_when_requested"
    }
    assert {model.generation.repeat_penalty for model in repetition.models} == {1.1}
    assert {model.generation.max_output_tokens for model in default.models} == {4096}
    assert {(model.architecture, model.quantization) for model in default.models} == {
        (architecture, quantization)
        for architecture in ("qwen2.5-3b", "gemma-3-4b", "qwen3-8b")
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M")
    }


def test_final_setting_suites_are_frozen_sized_and_targeted() -> None:
    temperature = load_suite(ROOT / "data" / "suites" / "temperature_stability.yaml")
    constrained = load_suite(ROOT / "data" / "suites" / "constrained_decoding.yaml")
    repetition = load_suite(ROOT / "data" / "suites" / "repetition_penalty.yaml")
    context = load_suite(ROOT / "data" / "suites" / "context_speed.yaml")

    assert temperature.manifest.status == "frozen"
    assert constrained.manifest.status == "frozen"
    assert repetition.manifest.status == "frozen"
    assert sum(map(len, temperature.items.values())) == 30
    assert sum(map(len, constrained.items.values())) == 15
    assert sum(map(len, repetition.items.values())) == 15
    assert sum(map(len, context.items.values())) == 28

    temperature_items = [item for items in temperature.items.values() for item in items]
    assert {item.difficulty for item in temperature_items} == {"easy", "medium", "hard"}
    assert {
        item.subcategory
        for item in temperature_items
        if item.benchmark == "code_debug_repair"
    } == {"function_implementation", "bug_diagnosis", "code_repair"}
    assert {
        item.subcategory
        for item in temperature_items
        if item.benchmark == "constraint_load_curve"
    } == {"one_constraint", "four_constraints"}

    constrained_items = [item for items in constrained.items.values() for item in items]
    assert all(item.response_contract.type == "json" for item in constrained_items)
    assert {item.difficulty for item in constrained_items} == {"easy", "medium", "hard"}

    repetition_items = [item for items in repetition.items.values() for item in items]
    assert {item.scoring.method for item in repetition_items} >= {
        "llm_judge", "executable_python", "tool_trace"
    }


def test_probe_metadata_matches_executable_suite_ids() -> None:
    pairs = [
        ("temperature_stability_v1.yaml", "temperature_stability.yaml"),
        ("constrained_decoding_v1.yaml", "constrained_decoding.yaml"),
        ("repetition_penalty_v1.yaml", "repetition_penalty.yaml"),
    ]
    for probe_name, suite_name in pairs:
        probe = yaml.safe_load((ROOT / "data" / "probes" / probe_name).read_text())
        suite = load_suite(ROOT / "data" / "suites" / suite_name)
        suite_ids = [item.id for items in suite.items.values() for item in items]
        assert probe["status"] == "frozen"
        assert probe["target_question_count"] == len(suite_ids)
        assert set(probe["item_ids"]) == set(suite_ids)


def test_final_profiles_keep_the_approved_generation_budget() -> None:
    profiles = [
        ("final_default_matrix.yaml", "all.yaml"),
        ("final_temperature_matrix.yaml", "temperature_stability.yaml"),
        ("final_constrained_matrix.yaml", "constrained_decoding.yaml"),
        ("final_repetition_matrix.yaml", "repetition_penalty.yaml"),
        ("final_context_matrix.yaml", "context_speed.yaml"),
    ]
    attempts = 0
    for config_name, suite_name in profiles:
        config = load_config(ROOT / "configs" / config_name)
        suite = load_suite(ROOT / "data" / "suites" / suite_name)
        attempts += (
            len(config.models)
            * config.benchmark.repetitions
            * sum(map(len, suite.items.values()))
        )
    assert attempts == 7_152
