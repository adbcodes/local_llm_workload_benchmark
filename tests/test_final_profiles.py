from pathlib import Path

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


ROOT = Path(__file__).resolve().parents[1]


def test_final_profiles_have_expected_configuration_counts() -> None:
    default = load_config(ROOT / "configs" / "final_default_matrix.yaml")
    tier2 = load_config(ROOT / "configs" / "final_tier2_matrix.yaml")
    context = load_config(ROOT / "configs" / "final_context_matrix.yaml")

    assert len(default.models) == 12
    assert len(tier2.models) == 36
    assert len(context.models) == 12
    assert {(model.architecture, model.quantization) for model in default.models} == {
        (architecture, quantization)
        for architecture in ("qwen2.5-3b", "gemma-3-4b", "qwen3-8b")
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M")
    }


def test_final_probe_suites_are_frozen_and_sized() -> None:
    tier2 = load_suite(ROOT / "data" / "suites" / "tier2_probe.yaml")
    context = load_suite(ROOT / "data" / "suites" / "context_speed.yaml")

    assert tier2.manifest.status == "frozen"
    assert sum(map(len, tier2.items.values())) == 40
    assert sum(map(len, context.items.values())) == 28
