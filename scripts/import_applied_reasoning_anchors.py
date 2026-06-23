from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MATH_URL = "https://huggingface.co/datasets/qwedsacf/competition_math"
BBH_URL = "https://github.com/suzgunmirac/BIG-Bench-Hard"
FINAL_ANSWER_INSTRUCTION = (
    "You may show concise working. End with exactly one final line in this format: "
    "FINAL: <answer>"
)


@dataclass(frozen=True)
class MathSelection:
    row: int
    item_id: str
    subcategory: str
    difficulty: str
    expected: int | float | str
    scoring: str = "numeric_tolerance"


@dataclass(frozen=True)
class BbhSelection:
    task: str
    row: int
    item_id: str
    subcategory: str
    difficulty: str


MATH_SELECTIONS = (
    MathSelection(5943, "anchor_math_average_001", "arithmetic_percentages", "easy", 91),
    MathSelection(5819, "anchor_math_discount_001", "arithmetic_percentages", "medium", 35),
    MathSelection(5942, "anchor_math_combined_mean_001", "arithmetic_percentages", "medium", 19),
    MathSelection(5570, "anchor_math_ratio_001", "ratios_rates_work", "easy", 6),
    MathSelection(5598, "anchor_math_recipe_ratio_001", "ratios_rates_work", "medium", 22),
    MathSelection(6198, "anchor_math_rate_001", "ratios_rates_work", "medium", 315),
    MathSelection(7, "anchor_math_substitution_001", "algebra_word_problems", "easy", 11),
    MathSelection(4, "anchor_math_work_days_001", "algebra_word_problems", "medium", 6),
    MathSelection(1, "anchor_math_band_001", "algebra_word_problems", "hard", 98),
    MathSelection(4699, "anchor_math_primes_001", "number_properties_sequences", "easy", 17),
    MathSelection(4701, "anchor_math_factor_rows_001", "number_properties_sequences", "medium", 5),
    MathSelection(4711, "anchor_math_remainders_001", "number_properties_sequences", "hard", 301),
    MathSelection(1751, "anchor_math_odds_001", "probability_counting", "medium", "4/7", "rational_value"),
    MathSelection(1772, "anchor_math_coin_001", "probability_counting", "medium", "7/8", "rational_value"),
    MathSelection(1760, "anchor_math_dice_001", "probability_counting", "hard", "27/128", "rational_value"),
)

BBH_SELECTIONS = (
    BbhSelection("date_understanding", 5, "anchor_bbh_date_visit_001", "calendar_time", "medium"),
    BbhSelection("date_understanding", 9, "anchor_bbh_date_monday_001", "calendar_time", "medium"),
    BbhSelection("date_understanding", 12, "anchor_bbh_date_elapsed_001", "calendar_time", "hard"),
    BbhSelection("boolean_expressions", 0, "anchor_bbh_boolean_001", "deductive_logic", "easy"),
    BbhSelection("formal_fallacies", 0, "anchor_bbh_fallacy_001", "deductive_logic", "medium"),
    BbhSelection("web_of_lies", 0, "anchor_bbh_lies_001", "deductive_logic", "hard"),
    BbhSelection("logical_deduction_three_objects", 0, "anchor_bbh_order_three_001", "ordering_constraint_puzzles", "easy"),
    BbhSelection("logical_deduction_five_objects", 1, "anchor_bbh_order_five_001", "ordering_constraint_puzzles", "medium"),
    BbhSelection("logical_deduction_seven_objects", 1, "anchor_bbh_order_seven_001", "ordering_constraint_puzzles", "hard"),
)


def import_anchors(math_parquet: Path, bbh_root: Path) -> list[dict[str, Any]]:
    math_rows = _load_math_rows(math_parquet)
    items = [_math_item(selection, math_rows[selection.row]) for selection in MATH_SELECTIONS]
    items.extend(_bbh_item(selection, bbh_root) for selection in BBH_SELECTIONS)
    return items


def write_anchors(output: Path, math_parquet: Path, bbh_root: Path) -> None:
    document = {
        "schema_version": 1,
        "benchmark": "applied_reasoning",
        "prompt_suffix": FINAL_ANSWER_INSTRUCTION,
        "items": import_anchors(math_parquet, bbh_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=88),
        encoding="utf-8",
    )


def _load_math_rows(path: Path) -> dict[int, dict[str, Any]]:
    selected_rows = ",".join(str(selection.row) for selection in MATH_SELECTIONS)
    query = (
        "WITH source AS (SELECT row_number() OVER () - 1 AS source_row, * "
        f"FROM read_parquet('{path}')) SELECT * FROM source "
        f"WHERE source_row IN ({selected_rows}) ORDER BY source_row"
    )
    try:
        result = subprocess.run(
            ["duckdb", "-json", "-c", query],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not read MATH parquet with duckdb: {error}") from error
    rows = json.loads(result.stdout)
    indexed = {int(row["source_row"]): row for row in rows}
    missing = {selection.row for selection in MATH_SELECTIONS} - set(indexed)
    if missing:
        raise RuntimeError(f"MATH source rows are missing: {sorted(missing)}")
    return indexed


def _math_item(selection: MathSelection, row: dict[str, Any]) -> dict[str, Any]:
    expected_text = str(selection.expected)
    source_expected = (
        "{" + expected_text.replace("/", "}{") + "}"
        if selection.scoring == "rational_value"
        else expected_text
    )
    if source_expected not in row["solution"]:
        raise RuntimeError(
            f"expected value {expected_text!r} was not found in MATH row {selection.row}"
        )
    parameters = (
        {"absolute_tolerance": 1e-9, "allow_surrounding_text": True}
        if selection.scoring == "rational_value"
        else {"absolute_tolerance": 0, "allow_surrounding_text": True}
    )
    return {
        "id": selection.item_id,
        "subcategory": selection.subcategory,
        "difficulty": selection.difficulty,
        "split": "test",
        "prompt": row["problem"],
        "response_contract": {"type": "number", "format": None},
        "expected": {"value": selection.expected},
        "scoring": {"method": selection.scoring, "parameters": parameters},
        "provenance": {
            "kind": "adapted",
            "review_status": "human_checked",
            "source": {
                "dataset": "MATH",
                "record_id": f"train:{selection.row}",
                "url": MATH_URL,
                "license": "MIT",
                "content_sha256": _content_hash(row["problem"]),
            },
        },
        "tags": ["licensed_anchor", "math", row["type"].lower().replace(" ", "_")],
    }


def _bbh_item(selection: BbhSelection, root: Path) -> dict[str, Any]:
    task_path = root / "bbh" / f"{selection.task}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    example = task["examples"][selection.row]
    return {
        "id": selection.item_id,
        "subcategory": selection.subcategory,
        "difficulty": selection.difficulty,
        "split": "test",
        "prompt": example["input"],
        "response_contract": {"type": "text", "format": "source_label"},
        "expected": {"value": example["target"]},
        "scoring": {
            "method": "exact_match",
            "parameters": {
                "strip": True,
                "case_sensitive": False,
                "allow_surrounding_text": False,
            },
        },
        "provenance": {
            "kind": "adapted",
            "review_status": "human_checked",
            "source": {
                "dataset": "BIG-Bench Hard",
                "record_id": f"{selection.task}:{selection.row}",
                "url": f"{BBH_URL}/blob/main/bbh/{selection.task}.json",
                "license": "MIT",
                "content_sha256": _content_hash(example["input"]),
            },
        },
        "tags": ["licensed_anchor", "big_bench_hard", selection.task],
    }


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-parquet", type=Path, required=True)
    parser.add_argument("--bbh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_anchors(args.output, args.math_parquet, args.bbh_root)


if __name__ == "__main__":
    main()
