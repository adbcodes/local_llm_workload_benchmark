from collections import Counter
from fractions import Fraction
from pathlib import Path

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


SUITE_PATH = Path("data/suites/reasoning.yaml")
GENERATED_PATH = Path(
    "data/applied_reasoning/generated.yaml"
)
QUANT_CONFIG_PATH = Path("configs/final_default_matrix.yaml")


def test_applied_reasoning_has_fresh_headline_dataset() -> None:
    items = load_suite(SUITE_PATH).items["applied_reasoning"]

    assert len(items) == 48
    assert all(
        item.prompt.endswith(
            "End with exactly one final line in this format: FINAL: <answer>"
        )
        for item in items
    )
    assert Counter(item.subcategory for item in items) == {
        "arithmetic_percentages": 6,
        "ratios_rates_work": 6,
        "algebra_word_problems": 6,
        "number_properties_sequences": 6,
        "calendar_time": 6,
        "probability_counting": 6,
        "deductive_logic": 6,
        "ordering_constraint_puzzles": 6,
    }
    assert Counter(item.difficulty for item in items) == {
        "easy": 8,
        "medium": 24,
        "hard": 16,
    }

    assert all(item.visibility == "held_out" for item in items)
    assert all(item.provenance.kind == "synthetic" for item in items)


def test_licensed_anchors_record_source_and_license() -> None:
    document = yaml.safe_load(Path("data/applied_reasoning/external.yaml").read_text())
    sources = [item["provenance"]["source"] for item in document["items"]]

    assert Counter(source["dataset"] for source in sources) == {
        "MATH": 15,
        "BIG-Bench Hard": 9,
    }
    assert all(source["license"] == "MIT" for source in sources)
    assert all(len(source["content_sha256"]) == 64 for source in sources)
    assert len({(source["dataset"], source["record_id"]) for source in sources}) == 24
    assert Path(
        "data/applied_reasoning/THIRD_PARTY_NOTICES.md"
    ).is_file()


def test_applied_reasoning_yaml_is_the_curated_authoring_source() -> None:
    document = yaml.safe_load(GENERATED_PATH.read_text(encoding="utf-8"))

    assert document["generated_by"] == "applied_reasoning_v3"
    assert document["seed"] == 20260731
    assert len(document["items"]) == 48
    assert len({item["prompt"] for item in document["items"]}) == 48
    assert all(
        item["provenance"]["review_status"] == "human_checked"
        for item in document["items"]
    )


def test_every_applied_reasoning_gold_matches_independent_ledger() -> None:
    """Lock all golds to a separately calculated review ledger."""

    items = {
        item.id: str(item.expected["value"])
        for item in load_suite(SUITE_PATH).items["applied_reasoning"]
    }
    expected = {
        "reason_arithmeticperc_001": "112",
        "reason_arithmeticperc_002": "1803.9",
        "reason_arithmeticperc_003": "10530",
        "reason_arithmeticperc_004": "6",
        "reason_arithmeticperc_005": str(Fraction(4779, 10)),
        "reason_arithmeticperc_006": "5",
        "reason_ratiosrateswor_001": "55",
        "reason_ratiosrateswor_002": str(Fraction(36, 5)),
        "reason_ratiosrateswor_003": "1170",
        "reason_ratiosrateswor_004": "295",
        "reason_ratiosrateswor_005": str(Fraction(1720, 9)),
        "reason_ratiosrateswor_006": "112",
        "reason_algebrawordpro_001": "13",
        "reason_algebrawordpro_002": "8",
        "reason_algebrawordpro_003": "7",
        "reason_algebrawordpro_004": "1200",
        "reason_algebrawordpro_005": "500",
        "reason_algebrawordpro_006": "350",
        "reason_numberproperti_001": "72",
        "reason_numberproperti_002": "42",
        "reason_numberproperti_003": "34",
        "reason_numberproperti_004": str(Fraction(27675, 4)),
        "reason_numberproperti_005": "66",
        "reason_numberproperti_006": "54",
        "reason_calendartime_001": "16:20",
        "reason_calendartime_002": "2027-05-19",
        "reason_calendartime_003": "2027-05-17",
        "reason_calendartime_004": "2028-10-14 19:45",
        "reason_calendartime_005": "2029-01-12",
        "reason_calendartime_007": "2027-03-28 00:00",
        "reason_probabilitycou_001": str(Fraction(3, 10)),
        "reason_probabilitycou_002": str(Fraction(8, 13)),
        "reason_probabilitycou_003": str(Fraction(137, 228)),
        "reason_probabilitycou_004": str(Fraction(29, 4000)),
        "reason_probabilitycou_005": str(Fraction(15, 23)),
        "reason_probabilitycou_007": str(Fraction(5, 8)),
        "reason_deductivelogic_001": "no",
        "reason_deductivelogic_002": "cannot_be_determined",
        "reason_deductivelogic_003": "repo_read",
        "reason_deductivelogic_004": "yes",
        "reason_deductivelogic_005": "no",
        "reason_deductivelogic_007": "B,E",
        "reason_orderingconstr_001": "M,N,O",
        "reason_orderingconstr_002": "C,A,B,D,E",
        "reason_orderingconstr_003": "A,B,C,D,E",
        "reason_orderingconstr_004": "P,Q,R",
        "reason_orderingconstr_005": "B,D,F,A,C,E",
        "reason_orderingconstr_007": "C,F,A,D,B,E",
    }

    assert items == expected


def test_applied_reasoning_covers_requested_operational_mechanisms() -> None:
    items = load_suite(SUITE_PATH).items["applied_reasoning"]
    tags = {tag for item in items for tag in item.tags}

    assert {
        "billing_reconciliation",
        "capacity",
        "maintenance_windows",
        "retry_backoff",
        "business_days",
        "timezone",
        "access_policy",
        "dependency_ordering",
        "scheduling",
        "resource_constraints",
    } <= tags


def test_detector_posterior_uses_valid_priors() -> None:
    items = {
        item.id: item for item in load_suite(SUITE_PATH).items["applied_reasoning"]
    }
    detector = items["reason_probabilitycou_007"]

    assert "equally likely" in detector.prompt
    assert detector.expected["value"] == "5/8"


def test_final_quantization_config_includes_applied_reasoning() -> None:
    config = load_config(QUANT_CONFIG_PATH)

    assert config.benchmark.workload_path == Path("data/suites/final_five.yaml")
    assert len(config.models) == 20
    assert {model.quantization for model in config.models} == {
        "Q8_0",
        "Q6_K",
        "Q4_K_M",
        "Q3_K_M",
    }
    assert all(model.generation.max_output_tokens == 4096 for model in config.models)
