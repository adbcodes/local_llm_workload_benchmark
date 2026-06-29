from collections import Counter
from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


SUITE_PATH = Path("data/suites/reasoning.yaml")
GENERATED_PATH = Path(
    "data/applied_reasoning/generated.yaml"
)
QUANT_CONFIG_PATH = Path("configs/applied_reasoning_quant_matrix.yaml")


def test_applied_reasoning_has_fresh_headline_dataset() -> None:
    items = load_suite(SUITE_PATH).items["applied_reasoning"]

    assert len(items) == 100
    assert all(
        item.prompt.endswith(
            "End with exactly one final line in this format: FINAL: <answer>"
        )
        for item in items
    )
    assert Counter(item.subcategory for item in items) == {
        "arithmetic_percentages": 13,
        "ratios_rates_work": 13,
        "algebra_word_problems": 13,
        "number_properties_sequences": 13,
        "calendar_time": 12,
        "probability_counting": 12,
        "deductive_logic": 12,
        "ordering_constraint_puzzles": 12,
    }
    assert Counter(item.difficulty for item in items) == {
        "easy": 8,
        "medium": 44,
        "hard": 48,
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


def test_materialized_generated_yaml_matches_generator(tmp_path: Path) -> None:
    document = yaml.safe_load(GENERATED_PATH.read_text(encoding="utf-8"))
    regenerated_path = tmp_path / "generated.yaml"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_applied_reasoning.py",
            "--output",
            str(regenerated_path),
            "--seed",
            str(document["seed"]),
        ],
        check=True,
    )
    regenerated = yaml.safe_load(regenerated_path.read_text(encoding="utf-8"))

    assert document["generated_by"] == "applied_reasoning_v2"
    assert document["seed"] == 20260731
    assert document == regenerated


def test_quantization_config_runs_only_the_headline_suite() -> None:
    config = load_config(QUANT_CONFIG_PATH)

    assert config.benchmark.workload_path == Path("data/suites/reasoning.yaml")
    assert len(config.models) == 12
    assert {model.quantization for model in config.models} == {
        "Q8_0",
        "Q6_K",
        "Q4_K_M",
        "Q3_K_M",
    }
    assert all(model.generation.max_output_tokens == 4096 for model in config.models)
