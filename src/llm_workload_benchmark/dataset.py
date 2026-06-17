from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["dev", "test"]
ScoringMethod = Literal[
    "numeric_tolerance",
    "exact_match",
    "json_exact",
    "constraint_rules",
]

DIFFICULTY_ORDER: dict[Difficulty, int] = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}


class DatasetError(ValueError):
    """Raised when benchmark dataset files are missing or invalid."""


class ResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "date", "text", "json"]
    format: str | None = None


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: ScoringMethod
    parameters: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["hand_authored", "synthetic", "adapted"]
    review_status: Literal["draft", "human_checked"]
    generator: str | None = None
    seed: int | None = None


class DatasetItem(BaseModel):
    """A common envelope shared by every benchmark item."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    benchmark: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    subcategory: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    difficulty: Difficulty
    split: Split
    prompt: str = Field(min_length=1)
    response_contract: ResponseContract
    expected: dict[str, Any]
    scoring: ScoringSpec
    provenance: Provenance
    tags: list[str] = Field(default_factory=list)
    variant_of: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    @model_validator(mode="after")
    def scoring_contract_is_consistent(self) -> Self:
        if "value" not in self.expected:
            raise ValueError("expected must contain a value field")

        value = self.expected["value"]
        method = self.scoring.method
        contract_type = self.response_contract.type

        if method == "numeric_tolerance":
            if contract_type != "number":
                raise ValueError("numeric_tolerance requires a number contract")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("numeric_tolerance requires a numeric expected value")
            tolerance = self.scoring.parameters.get("absolute_tolerance", 0)
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                raise ValueError("absolute_tolerance must be numeric")
            if tolerance < 0:
                raise ValueError("absolute_tolerance cannot be negative")
        elif method == "json_exact":
            if contract_type != "json":
                raise ValueError("json_exact requires a json contract")
            if not isinstance(value, (dict, list)):
                raise ValueError("json_exact requires an object or array expected value")
        elif method == "constraint_rules":
            if contract_type != "text":
                raise ValueError("constraint_rules requires a text contract")
            rules = self.scoring.parameters.get("rules")
            if not isinstance(rules, dict) or not rules:
                raise ValueError("constraint_rules requires a non-empty rules object")
        elif not isinstance(value, (str, int, float, bool)):
            raise ValueError("exact_match requires a scalar expected value")
        return self


class BenchmarkDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    items_path: str = Field(min_length=1)
    current_question_count: int = Field(gt=0)
    target_question_count: int = Field(gt=0)
    current_difficulty_distribution: dict[Difficulty, int]
    difficulty_distribution: dict[Difficulty, int]
    order_rule: Literal["easy_to_hard"]
    scoring_methods: list[ScoringMethod] = Field(min_length=1)

    @model_validator(mode="after")
    def difficulty_counts_match_target(self) -> Self:
        if self.current_question_count > self.target_question_count:
            raise ValueError("current_question_count cannot exceed target_question_count")
        if set(self.current_difficulty_distribution) != {"easy", "medium", "hard"}:
            raise ValueError(
                "current_difficulty_distribution requires easy, medium, and hard"
            )
        if any(count < 1 for count in self.current_difficulty_distribution.values()):
            raise ValueError("current dataset must include every difficulty level")
        if (
            sum(self.current_difficulty_distribution.values())
            != self.current_question_count
        ):
            raise ValueError(
                "current_difficulty_distribution must sum to "
                "current_question_count"
            )
        if set(self.difficulty_distribution) != {"easy", "medium", "hard"}:
            raise ValueError("difficulty_distribution requires easy, medium, and hard")
        if any(count < 1 for count in self.difficulty_distribution.values()):
            raise ValueError("target dataset must include every difficulty level")
        if sum(self.difficulty_distribution.values()) != self.target_question_count:
            raise ValueError(
                "difficulty_distribution must sum to target_question_count"
            )
        return self


class SuiteManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    status: Literal["pilot", "frozen"]
    benchmark_files: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class BenchmarkSuite:
    manifest: SuiteManifest
    definitions: dict[str, BenchmarkDefinition]
    items: dict[str, list[DatasetItem]]


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


def load_dataset(path: Path) -> list[DatasetItem]:
    """Load a JSONL dataset and enforce stable IDs and easy-to-hard ordering."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise DatasetError(f"dataset file does not exist: {path}") from error
    except OSError as error:
        raise DatasetError(f"could not read dataset file {path}: {error}") from error

    items: list[DatasetItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_item = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(
                f"invalid JSON in {path} on line {line_number}: {error.msg}"
            ) from error
        try:
            item = DatasetItem.model_validate(raw_item)
        except ValidationError as error:
            raise DatasetError(
                f"invalid dataset item in {path} on line {line_number}:\n{error}"
            ) from error
        if item.id in seen_ids:
            raise DatasetError(f"duplicate item id {item.id!r} in {path}")
        seen_ids.add(item.id)
        items.append(item)

    if not items:
        raise DatasetError(f"dataset file contains no items: {path}")
    _validate_difficulty_progression(items, path)
    return items


def load_suite(path: Path) -> BenchmarkSuite:
    """Load a suite manifest, definitions, and all referenced benchmark items."""

    manifest = _load_yaml_model(path, SuiteManifest)
    definitions: dict[str, BenchmarkDefinition] = {}
    items_by_benchmark: dict[str, list[DatasetItem]] = {}
    all_item_ids: set[str] = set()

    for relative_definition_path in manifest.benchmark_files:
        definition_path = path.parent / relative_definition_path
        definition = _load_yaml_model(definition_path, BenchmarkDefinition)
        if definition.id in definitions:
            raise DatasetError(f"duplicate benchmark id {definition.id!r}")

        item_path = definition_path.parent / definition.items_path
        items = load_dataset(item_path)
        if len(items) != definition.current_question_count:
            raise DatasetError(
                f"{definition.id} expected {definition.current_question_count} current "
                f"items but loaded {len(items)}"
            )
        actual_difficulties = Counter(item.difficulty for item in items)
        if dict(actual_difficulties) != definition.current_difficulty_distribution:
            raise DatasetError(
                f"{definition.id} difficulty counts do not match "
                "current_difficulty_distribution"
            )
        for item in items:
            if item.benchmark != definition.id:
                raise DatasetError(
                    f"item {item.id!r} belongs to {item.benchmark!r}, expected "
                    f"{definition.id!r}"
                )
            if item.scoring.method not in definition.scoring_methods:
                raise DatasetError(
                    f"item {item.id!r} uses undeclared scoring method "
                    f"{item.scoring.method!r}"
                )
            if item.id in all_item_ids:
                raise DatasetError(f"duplicate item id across suite: {item.id!r}")
            all_item_ids.add(item.id)

        definitions[definition.id] = definition
        items_by_benchmark[definition.id] = items

    return BenchmarkSuite(
        manifest=manifest,
        definitions=definitions,
        items=items_by_benchmark,
    )


def score_answer(item: DatasetItem, answer: str) -> ScoreResult:
    """Score one raw model answer using the verifier declared by its item."""

    method = item.scoring.method
    if method == "numeric_tolerance":
        return _score_numeric(item, answer)
    if method == "exact_match":
        return _score_exact(item, answer)
    if method == "json_exact":
        return _score_json(item, answer)
    return _score_constraints(item, answer)


def _validate_difficulty_progression(items: list[DatasetItem], path: Path) -> None:
    levels = [DIFFICULTY_ORDER[item.difficulty] for item in items]
    if levels != sorted(levels):
        raise DatasetError(f"items in {path} must be ordered from easy to hard")
    if len(items) >= 3 and set(levels) != set(DIFFICULTY_ORDER.values()):
        raise DatasetError(
            f"dataset {path} must include easy, medium, and hard items"
        )


def _load_yaml_model(path: Path, model_type: type[BaseModel]) -> Any:
    try:
        raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"dataset metadata does not exist: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError(f"could not load dataset metadata {path}: {error}") from error
    try:
        return model_type.model_validate(raw_value)
    except ValidationError as error:
        raise DatasetError(f"invalid dataset metadata in {path}:\n{error}") from error


def _score_numeric(item: DatasetItem, answer: str) -> ScoreResult:
    try:
        actual = float(answer.strip().replace(",", ""))
    except ValueError:
        return ScoreResult(passed=False, score=0, details={"reason": "not_numeric"})

    expected = float(item.expected["value"])
    tolerance = float(item.scoring.parameters.get("absolute_tolerance", 0))
    difference = abs(actual - expected)
    passed = difference <= tolerance
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual, "difference": difference, "tolerance": tolerance},
    )


def _score_exact(item: DatasetItem, answer: str) -> ScoreResult:
    expected = str(item.expected["value"])
    actual = answer.strip() if item.scoring.parameters.get("strip", True) else answer
    if not item.scoring.parameters.get("case_sensitive", True):
        actual = actual.casefold()
        expected = expected.casefold()
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual, "expected": expected},
    )


def _score_json(item: DatasetItem, answer: str) -> ScoreResult:
    try:
        actual = json.loads(answer)
    except json.JSONDecodeError:
        return ScoreResult(passed=False, score=0, details={"reason": "invalid_json"})

    expected = item.expected["value"]
    expected_leaves = _flatten_json(expected)
    actual_leaves = _flatten_json(actual)
    all_paths = set(expected_leaves) | set(actual_leaves)
    matched_paths = {
        path
        for path in all_paths
        if path in expected_leaves
        and path in actual_leaves
        and expected_leaves[path] == actual_leaves[path]
    }
    score = len(matched_paths) / len(all_paths) if all_paths else 1.0
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=score,
        details={
            "valid_json": True,
            "leaf_accuracy": score,
            "missing_paths": sorted(set(expected_leaves) - set(actual_leaves)),
            "extra_paths": sorted(set(actual_leaves) - set(expected_leaves)),
        },
    )


def _flatten_json(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            leaves.update(_flatten_json(child, f"{path}.{key}"))
        return leaves or {path: value}
    if isinstance(value, list):
        leaves = {}
        for index, child in enumerate(value):
            leaves.update(_flatten_json(child, f"{path}[{index}]"))
        return leaves or {path: value}
    return {path: value}


def _score_constraints(item: DatasetItem, answer: str) -> ScoreResult:
    rules = item.scoring.parameters["rules"]
    checks: dict[str, bool] = {}
    words = re.findall(r"\b[\w'-]+\b", answer, flags=re.UNICODE)

    if "max_words" in rules:
        checks["max_words"] = len(words) <= rules["max_words"]
    if "exact_words" in rules:
        checks["exact_words"] = len(words) == rules["exact_words"]
    if "exact_sentences" in rules:
        sentence_count = len(re.findall(r"[.!?]+(?:\s|$)", answer.strip()))
        checks["exact_sentences"] = sentence_count == rules["exact_sentences"]
    if "required_terms" in rules:
        folded_answer = answer.casefold()
        checks["required_terms"] = all(
            term.casefold() in folded_answer for term in rules["required_terms"]
        )
    if "forbidden_terms" in rules:
        folded_answer = answer.casefold()
        checks["forbidden_terms"] = all(
            term.casefold() not in folded_answer for term in rules["forbidden_terms"]
        )
    if "prefix" in rules:
        checks["prefix"] = answer.startswith(rules["prefix"])
    if "suffix" in rules:
        checks["suffix"] = answer.endswith(rules["suffix"])

    passed_count = sum(checks.values())
    score = passed_count / len(checks) if checks else 0
    return ScoreResult(
        passed=bool(checks) and passed_count == len(checks),
        score=score,
        details={"checks": checks, "word_count": len(words)},
    )
