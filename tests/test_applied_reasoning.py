from collections import Counter
from pathlib import Path
import subprocess
import sys

import yaml

from llm_workload_benchmark.dataset import load_suite


SUITE_PATH = Path("data/suites/reasoning.yaml")
GENERATED_PATH = Path(
    "data/applied_reasoning/generated.yaml"
)


def test_applied_reasoning_has_balanced_hybrid_dataset() -> None:
    items = load_suite(SUITE_PATH).items["applied_reasoning"]

    assert len(items) == 48
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
        "easy": 12,
        "medium": 24,
        "hard": 12,
    }

    licensed = [item for item in items if item.provenance.source is not None]
    generated = [item for item in items if item.provenance.kind == "synthetic"]
    assert len(licensed) == len(generated) == 24
    for group in (licensed, generated):
        assert Counter(item.difficulty for item in group) == {
            "easy": 6,
            "medium": 12,
            "hard": 6,
        }


def test_licensed_anchors_record_source_and_license() -> None:
    items = load_suite(SUITE_PATH).items["applied_reasoning"]
    sources = [item.provenance.source for item in items if item.provenance.source]

    assert Counter(source.dataset for source in sources) == {
        "MATH": 15,
        "BIG-Bench Hard": 9,
    }
    assert all(source.license == "MIT" for source in sources)
    assert all(len(source.content_sha256) == 64 for source in sources)
    assert len({(source.dataset, source.record_id) for source in sources}) == 24
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

    assert document["generated_by"] == "applied_reasoning_v1"
    assert document["seed"] == 20260721
    assert document == regenerated
