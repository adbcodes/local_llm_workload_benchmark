from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_workload_benchmark.evaluation import EvaluationResult

Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["dev", "test"]
ScoringMethod = Literal[
    "numeric_tolerance",
    "rational_value",
    "date_value",
    "exact_match",
    "json_exact",
    "constraint_rules",
    "executable_python",
    "llm_judge",
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

    type: Literal["number", "date", "text", "json", "code"]
    format: str | None = None


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: ScoringMethod
    parameters: dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["hand_authored", "synthetic", "adapted"]
    review_status: Literal["draft", "human_checked"]
    generator: str | None = None
    seed: int | None = None
    source: SourceReference | None = None


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
        elif method == "rational_value":
            if contract_type != "number":
                raise ValueError("rational_value requires a number contract")
            if not isinstance(value, str):
                raise ValueError("rational_value requires a string expected value")
            _parse_rational(value)
            tolerance = self.scoring.parameters.get("absolute_tolerance", 1e-9)
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                raise ValueError("absolute_tolerance must be numeric")
            if tolerance < 0:
                raise ValueError("absolute_tolerance cannot be negative")
        elif method == "date_value":
            if contract_type != "date":
                raise ValueError("date_value requires a date contract")
            if not isinstance(value, str):
                raise ValueError("date_value requires a string expected value")
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
        elif method == "executable_python":
            if (
                contract_type != "code"
                or self.response_contract.format != "python_function"
            ):
                raise ValueError(
                    "executable_python requires a python_function code contract"
                )
            _validate_python_specification(value)
        elif method == "llm_judge":
            if contract_type != "text":
                raise ValueError("llm_judge requires a text contract")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("llm_judge requires a non-empty reference summary")
        elif not isinstance(value, (str, int, float, bool)):
            raise ValueError("exact_match requires a scalar expected value")
        _validate_scoring_parameters(method, self.scoring.parameters)
        return self


class BenchmarkDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    items_path: str = Field(min_length=1)
    authoring_paths: list[str] = Field(default_factory=list)
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
        if any(count < 0 for count in self.current_difficulty_distribution.values()):
            raise ValueError("current difficulty counts cannot be negative")
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


class SuiteFilters(BaseModel):
    """Optional item selection applied after complete benchmark validation."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] | None = None
    subcategories: list[str] | None = None
    difficulties: list[Difficulty] | None = None
    splits: list[Split] | None = None
    review_statuses: list[Literal["draft", "human_checked"]] | None = None

    @model_validator(mode="after")
    def filters_are_not_empty(self) -> Self:
        for name in (
            "ids",
            "subcategories",
            "difficulties",
            "splits",
            "review_statuses",
        ):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError(f"{name} cannot be empty when provided")
        return self


class SuiteManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    status: Literal["pilot", "frozen"]
    benchmark_files: list[str] = Field(min_length=1)
    filters: SuiteFilters | None = None


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


def _validate_scoring_parameters(
    method: ScoringMethod,
    parameters: dict[str, Any],
) -> None:
    allowed_parameters: dict[ScoringMethod, set[str]] = {
        "numeric_tolerance": {"absolute_tolerance", "allow_surrounding_text"},
        "rational_value": {"absolute_tolerance", "allow_surrounding_text"},
        "date_value": set(),
        "exact_match": {
            "strip",
            "case_sensitive",
            "allow_surrounding_text",
            "answer_format",
        },
        "json_exact": {"allow_diagnostic_normalization"},
        "constraint_rules": {"rules", "content_requirements"},
        "executable_python": {
            "timeout_seconds",
            "memory_limit_mb",
            "max_output_characters",
        },
        "llm_judge": {
            "rubric",
            "pass_threshold",
            "minimum_faithfulness",
            "max_words",
        },
    }
    unknown = set(parameters) - allowed_parameters[method]
    if unknown:
        raise ValueError(f"unknown {method} parameters: {sorted(unknown)}")

    for name in ("allow_surrounding_text", "strip", "case_sensitive"):
        if name in parameters and not isinstance(parameters[name], bool):
            raise ValueError(f"{name} must be a boolean")

    if method in {"numeric_tolerance", "rational_value"}:
        return
    if method == "date_value":
        return
    if method == "exact_match":
        answer_format = parameters.get("answer_format")
        if answer_format not in {None, "comma_separated_labels"}:
            raise ValueError("answer_format must be comma_separated_labels")
        return
    if method == "json_exact":
        diagnostic = parameters.get("allow_diagnostic_normalization", True)
        if not isinstance(diagnostic, bool):
            raise ValueError("allow_diagnostic_normalization must be a boolean")
        return
    if method == "executable_python":
        timeout = parameters.get("timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0.05 <= timeout <= 10
        ):
            raise ValueError("timeout_seconds must be between 0.05 and 10")
        memory = parameters.get("memory_limit_mb")
        if (
            isinstance(memory, bool)
            or not isinstance(memory, int)
            or not 32 <= memory <= 1024
        ):
            raise ValueError("memory_limit_mb must be an integer from 32 to 1024")
        output = parameters.get("max_output_characters")
        if (
            isinstance(output, bool)
            or not isinstance(output, int)
            or not 256 <= output <= 100_000
        ):
            raise ValueError(
                "max_output_characters must be an integer from 256 to 100000"
            )
        return
    if method == "llm_judge":
        if parameters.get("rubric") != "grounded_summary_v1":
            raise ValueError("llm_judge rubric must be grounded_summary_v1")
        pass_threshold = parameters.get("pass_threshold")
        if (
            isinstance(pass_threshold, bool)
            or not isinstance(pass_threshold, (int, float))
            or not 0 <= pass_threshold <= 1
        ):
            raise ValueError("llm_judge pass_threshold must be between 0 and 1")
        minimum_faithfulness = parameters.get("minimum_faithfulness")
        if (
            isinstance(minimum_faithfulness, bool)
            or not isinstance(minimum_faithfulness, int)
            or not 0 <= minimum_faithfulness <= 4
        ):
            raise ValueError(
                "llm_judge minimum_faithfulness must be an integer from 0 to 4"
            )
        max_words = parameters.get("max_words")
        if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
            raise ValueError("llm_judge max_words must be a positive integer")
        return

    rules = parameters["rules"]
    allowed_rules = {
        "max_words",
        "exact_words",
        "exact_sentences",
        "required_terms",
        "forbidden_terms",
        "prefix",
        "suffix",
        "forbidden_punctuation",
    }
    unknown_rules = set(rules) - allowed_rules
    if unknown_rules:
        raise ValueError(f"unknown constraint rules: {sorted(unknown_rules)}")
    for name in ("max_words", "exact_words", "exact_sentences"):
        if name in rules:
            value = rules[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
    if "max_words" in rules and "exact_words" in rules:
        if rules["exact_words"] > rules["max_words"]:
            raise ValueError("exact_words cannot exceed max_words")
    for name in ("required_terms", "forbidden_terms", "forbidden_punctuation"):
        if name in rules:
            _validate_nonempty_strings(rules[name], name)
    if "forbidden_punctuation" in rules and any(
        len(value) != 1 for value in rules["forbidden_punctuation"]
    ):
        raise ValueError("forbidden_punctuation entries must be single characters")
    for name in ("prefix", "suffix"):
        if name in rules and (
            not isinstance(rules[name], str) or not rules[name].strip()
        ):
            raise ValueError(f"{name} must be a non-empty string")
    required = {term.casefold() for term in rules.get("required_terms", [])}
    forbidden = {term.casefold() for term in rules.get("forbidden_terms", [])}
    if required & forbidden:
        raise ValueError("the same term cannot be both required and forbidden")

    content = parameters.get("content_requirements")
    if not isinstance(content, dict) or set(content) != {"required_facts"}:
        raise ValueError(
            "constraint_rules requires content_requirements.required_facts"
        )
    facts = content["required_facts"]
    if not isinstance(facts, list) or not facts:
        raise ValueError("required_facts must be a non-empty list")
    seen_names: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != {"name", "any_of"}:
            raise ValueError("each required fact needs exactly name and any_of")
        name = fact["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("required fact names must use snake_case")
        if name in seen_names:
            raise ValueError(f"duplicate required fact name {name!r}")
        seen_names.add(name)
        _validate_nonempty_strings(fact["any_of"], f"required fact {name}")


def _validate_python_specification(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"entry_point", "tests"}:
        raise ValueError("executable_python expected value needs entry_point and tests")
    entry_point = value["entry_point"]
    if not isinstance(entry_point, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", entry_point
    ):
        raise ValueError("entry_point must use snake_case")
    tests = value["tests"]
    if not isinstance(tests, list) or not tests:
        raise ValueError("executable_python requires at least one test")
    for test in tests:
        if not isinstance(test, dict) or set(test) - {"args", "kwargs", "expected"}:
            raise ValueError("each Python test may contain args, kwargs, and expected")
        if "expected" not in test:
            raise ValueError("each Python test requires expected")
        if not isinstance(test.get("args", []), list):
            raise ValueError("Python test args must be a list")
        if not isinstance(test.get("kwargs", {}), dict):
            raise ValueError("Python test kwargs must be an object")
        try:
            json.dumps(test)
        except (TypeError, ValueError) as error:
            raise ValueError("Python tests must be JSON serializable") from error


def _validate_nonempty_strings(value: Any, name: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty list of non-empty strings")


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
        if item.scoring.method not in {"llm_judge", "executable_python"}:
            gold_answer = (
                json.dumps(item.expected["value"], separators=(",", ":"))
                if item.response_contract.type == "json"
                else str(item.expected["value"])
            )
            if not score_answer(item, gold_answer).passed:
                raise DatasetError(
                    f"expected answer for item {item.id!r} does not satisfy its scorer"
                )
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
    all_items: dict[str, DatasetItem] = {}

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
        difficulty_counts = Counter(item.difficulty for item in items)
        actual_difficulties = {
            difficulty: difficulty_counts[difficulty]
            for difficulty in DIFFICULTY_ORDER
        }
        if actual_difficulties != definition.current_difficulty_distribution:
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
            if item.id in all_items:
                raise DatasetError(f"duplicate item id across suite: {item.id!r}")
            all_items[item.id] = item

        definitions[definition.id] = definition
        items_by_benchmark[definition.id] = items

    _validate_variant_lineage(all_items)

    if manifest.filters is not None:
        requested_ids = set(manifest.filters.ids or [])
        unknown_ids = requested_ids - set(all_items)
        if unknown_ids:
            raise DatasetError(
                "suite filters reference unknown item ids: "
                + ", ".join(sorted(unknown_ids))
            )
        items_by_benchmark = {
            benchmark: selected
            for benchmark, benchmark_items in items_by_benchmark.items()
            if (
                selected := [
                    item
                    for item in benchmark_items
                    if _item_matches_filters(item, manifest.filters)
                ]
            )
        }
        definitions = {
            benchmark: definition
            for benchmark, definition in definitions.items()
            if benchmark in items_by_benchmark
        }
        if not items_by_benchmark:
            raise DatasetError("suite filters selected no dataset items")

    return BenchmarkSuite(
        manifest=manifest,
        definitions=definitions,
        items=items_by_benchmark,
    )


def _item_matches_filters(item: DatasetItem, filters: SuiteFilters) -> bool:
    return (
        (filters.ids is None or item.id in filters.ids)
        and (
            filters.subcategories is None
            or item.subcategory in filters.subcategories
        )
        and (
            filters.difficulties is None
            or item.difficulty in filters.difficulties
        )
        and (filters.splits is None or item.split in filters.splits)
        and (
            filters.review_statuses is None
            or item.provenance.review_status in filters.review_statuses
        )
    )


def _validate_variant_lineage(items: dict[str, DatasetItem]) -> None:
    for item in items.values():
        if item.variant_of is None:
            continue
        parent = items.get(item.variant_of)
        if parent is None:
            raise DatasetError(
                f"variant item {item.id!r} references unknown base item "
                f"{item.variant_of!r}"
            )
        if parent.benchmark != item.benchmark:
            raise DatasetError(
                f"variant item {item.id!r} must use a base item from the same benchmark"
            )
        if parent.variant_of is not None:
            raise DatasetError(
                f"variant item {item.id!r} must reference a base item, not another variant"
            )


def score_answer(item: DatasetItem, answer: str) -> EvaluationResult:
    """Score one raw model answer using the verifier declared by its item."""

    method = item.scoring.method
    if method == "llm_judge":
        raise DatasetError(
            f"item {item.id!r} requires an external LLM judge, not score_answer"
        )
    if method == "executable_python":
        raise DatasetError(
            f"item {item.id!r} requires restricted Python execution, not score_answer"
        )
    if method == "numeric_tolerance":
        score = _score_numeric(item, answer)
    elif method == "rational_value":
        score = _score_rational(item, answer)
    elif method == "date_value":
        score = _score_date(item, answer)
    elif method == "exact_match":
        score = _score_exact(item, answer)
    elif method == "json_exact":
        score = _score_json(item, answer)
    else:
        score = _score_constraints(item, answer)
    return EvaluationResult(
        type="deterministic",
        evaluator=method,
        version=1,
        passed=score.passed,
        score=score.score,
        details=score.details,
    )


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
    if item.scoring.parameters.get("allow_surrounding_text", False):
        matches = re.findall(
            r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            answer,
        )
        candidates = []
        for match in matches:
            try:
                value = float(match.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(value):
                candidates.append(value)
        unique_candidates = set(candidates)
        if len(unique_candidates) != 1:
            return ScoreResult(
                passed=False,
                score=0,
                details={
                    "reason": "missing_or_ambiguous_numeric_answer",
                    "candidates": candidates,
                },
            )
        actual = unique_candidates.pop()
    else:
        try:
            actual = float(answer.strip().replace(",", ""))
        except ValueError:
            return ScoreResult(passed=False, score=0, details={"reason": "not_numeric"})
        if not math.isfinite(actual):
            return ScoreResult(passed=False, score=0, details={"reason": "not_finite"})

    expected = float(item.expected["value"])
    tolerance = float(item.scoring.parameters.get("absolute_tolerance", 0))
    difference = abs(actual - expected)
    passed = difference <= tolerance
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual, "difference": difference, "tolerance": tolerance},
    )


def _score_rational(item: DatasetItem, answer: str) -> ScoreResult:
    expected = _parse_rational(str(item.expected["value"]))
    candidates = _extract_rationals(answer)
    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) != 1:
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "missing_or_ambiguous_rational_answer",
                "candidates": [str(value) for value in unique_candidates],
                "expected": str(expected),
            },
        )
    actual = unique_candidates[0]
    difference = abs(float(actual - expected))
    tolerance = float(item.scoring.parameters.get("absolute_tolerance", 1e-9))
    passed = actual == expected or difference <= tolerance
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={
            "actual": str(actual),
            "expected": str(expected),
            "difference": difference,
            "tolerance": tolerance,
        },
    )


def _parse_rational(value: str) -> Fraction:
    stripped = value.strip()
    latex_match = re.fullmatch(
        r"\\(?:d)?frac\{(-?\d+)\}\{([1-9]\d*)\}",
        stripped,
    )
    if latex_match:
        return Fraction(int(latex_match.group(1)), int(latex_match.group(2)))
    try:
        return Fraction(stripped)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("rational value must be an integer, decimal, or fraction") from error


def _extract_rationals(answer: str) -> list[Fraction]:
    candidates: list[Fraction] = []
    remaining = answer
    patterns = (
        r"\\(?:d)?frac\{(-?\d+)\}\{([1-9]\d*)\}",
        r"(?<![\d/])(-?\d+)\s*/\s*([1-9]\d*)(?![\d/])",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, remaining))
        for match in matches:
            candidates.append(Fraction(int(match.group(1)), int(match.group(2))))
        for match in reversed(matches):
            remaining = remaining[: match.start()] + " " + remaining[match.end() :]
    for value in re.findall(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])", remaining):
        candidates.append(Fraction(value))
    return candidates


def _score_exact(item: DatasetItem, answer: str) -> ScoreResult:
    expected = str(item.expected["value"])
    actual = answer.strip() if item.scoring.parameters.get("strip", True) else answer
    candidates: list[str] | None = None
    if item.scoring.parameters.get("allow_surrounding_text", False):
        answer_format = item.scoring.parameters.get("answer_format")
        if answer_format == "comma_separated_labels":
            label_count = len(expected.split(","))
            pattern = rf"(?<![\w])(?:[A-Za-z0-9]+\s*,\s*){{{label_count - 1}}}[A-Za-z0-9]+(?![\w])"
            candidates = re.findall(pattern, actual)
            candidates = [re.sub(r"\s*,\s*", ",", value) for value in candidates]
        if candidates is not None:
            unique_candidates = list(dict.fromkeys(candidates))
            if len(unique_candidates) != 1:
                return ScoreResult(
                    passed=False,
                    score=0,
                    details={
                        "reason": "missing_or_ambiguous_exact_answer",
                        "candidates": unique_candidates,
                        "expected": expected,
                    },
                )
            actual = unique_candidates[0]
    if not item.scoring.parameters.get("case_sensitive", True):
        actual = actual.casefold()
        expected = expected.casefold()
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual, "expected": expected, "candidates": candidates},
    )


def _score_date(item: DatasetItem, answer: str) -> ScoreResult:
    expected_candidates = _extract_dates(str(item.expected["value"]))
    candidates = _extract_dates(answer)
    if len(expected_candidates) != 1:
        return ScoreResult(
            passed=False,
            score=0,
            details={"reason": "invalid_expected_date"},
        )
    if len(candidates) != 1:
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "missing_or_ambiguous_date_answer",
                "candidates": [value.isoformat() for value in candidates],
                "expected": expected_candidates[0].isoformat(),
            },
        )

    actual = candidates[0]
    expected = expected_candidates[0]
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual.isoformat(), "expected": expected.isoformat()},
    )


def _extract_dates(answer: str) -> list[date]:
    candidates: list[date] = []

    for value in re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", answer):
        _append_parsed_date(candidates, value, "%Y-%m-%d")

    numeric_pattern = r"(?<!\d)(\d{1,2})([-/])(\d{1,2})\2(\d{4})(?!\d)"
    for first, _, second, year in re.findall(numeric_pattern, answer):
        first_number = int(first)
        second_number = int(second)
        if first_number > 12 >= second_number:
            _append_date_parts(candidates, int(year), second_number, first_number)
        elif second_number > 12 >= first_number:
            _append_date_parts(candidates, int(year), first_number, second_number)

    months = (
        "January|February|March|April|May|June|July|August|September|October|"
        "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )
    month_first_pattern = rf"(?i:\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b)"
    for month, day_value, year in re.findall(month_first_pattern, answer):
        _append_text_date(candidates, day_value, month, year)

    day_first_pattern = rf"(?i:\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\s+(\d{{4}})\b)"
    for day_value, month, year in re.findall(day_first_pattern, answer):
        _append_text_date(candidates, day_value, month, year)

    return list(dict.fromkeys(candidates))


def _append_parsed_date(candidates: list[date], value: str, date_format: str) -> None:
    try:
        candidates.append(datetime.strptime(value, date_format).date())
    except ValueError:
        return


def _append_date_parts(
    candidates: list[date],
    year: int,
    month: int,
    day_value: int,
) -> None:
    try:
        candidates.append(date(year, month, day_value))
    except ValueError:
        return


def _append_text_date(
    candidates: list[date],
    day_value: str,
    month: str,
    year: str,
) -> None:
    normalized_month = month.title()
    if normalized_month == "Sept":
        normalized_month = "Sep"
    full_months = {
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    month_format = "%B" if normalized_month in full_months else "%b"
    _append_parsed_date(
        candidates,
        f"{day_value} {normalized_month} {year}",
        f"%d {month_format} %Y",
    )


def _score_json(item: DatasetItem, answer: str) -> ScoreResult:
    protocol_compliant = True
    wrapper: str | None = None
    try:
        actual = json.loads(answer)
    except json.JSONDecodeError:
        protocol_compliant = False
        actual = None
        if item.scoring.parameters.get("allow_diagnostic_normalization", True):
            actual, wrapper = _extract_diagnostic_json(answer)
        if actual is None:
            return ScoreResult(
                passed=False,
                score=0,
                details={
                    "reason": "invalid_json",
                    "protocol_compliant": False,
                    "diagnostic_json_valid": False,
                },
            )

    expected = item.expected["value"]
    expected_leaves = _flatten_json(expected)
    actual_leaves = _flatten_json(actual)
    all_paths = set(expected_leaves) | set(actual_leaves)
    matched_paths = {
        path
        for path in all_paths
        if path in expected_leaves
        and path in actual_leaves
        and _json_values_equal(expected_leaves[path], actual_leaves[path])
    }
    score = len(matched_paths) / len(all_paths) if all_paths else 1.0
    content_exact = _json_values_equal(actual, expected)
    passed = protocol_compliant and content_exact
    return ScoreResult(
        passed=passed,
        score=score,
        details={
            "protocol_compliant": protocol_compliant,
            "content_exact": content_exact,
            "diagnostic_wrapper": wrapper,
            "leaf_accuracy": score,
            "missing_paths": sorted(set(expected_leaves) - set(actual_leaves)),
            "extra_paths": sorted(set(actual_leaves) - set(expected_leaves)),
        },
    )


def _extract_diagnostic_json(answer: str) -> tuple[Any | None, str | None]:
    stripped = answer.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "markdown_fence"
        except json.JSONDecodeError:
            return None, None

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        trailing = stripped[index + end :].strip()
        if index > 0 or trailing:
            return value, "surrounding_text"
    return None, None


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


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
        checks["required_terms"] = all(
            _contains_term(answer, term) for term in rules["required_terms"]
        )
    if "forbidden_terms" in rules:
        checks["forbidden_terms"] = all(
            not _contains_term(answer, term) for term in rules["forbidden_terms"]
        )
    if "prefix" in rules:
        checks["prefix"] = answer.startswith(rules["prefix"])
    if "suffix" in rules:
        checks["suffix"] = answer.endswith(rules["suffix"])
    if "forbidden_punctuation" in rules:
        checks["forbidden_punctuation"] = all(
            punctuation not in answer for punctuation in rules["forbidden_punctuation"]
        )

    passed_count = sum(checks.values())
    constraint_score = passed_count / len(checks) if checks else 0
    fact_checks = {
        fact["name"]: any(_contains_term(answer, phrase) for phrase in fact["any_of"])
        for fact in item.scoring.parameters["content_requirements"]["required_facts"]
    }
    content_score = sum(fact_checks.values()) / len(fact_checks)
    content_preserved = all(fact_checks.values())
    return ScoreResult(
        passed=(
            content_preserved
            and bool(checks)
            and passed_count == len(checks)
        ),
        score=content_score * constraint_score,
        details={
            "content_preserved": content_preserved,
            "content_score": content_score,
            "fact_checks": fact_checks,
            "constraint_score": constraint_score,
            "checks": checks,
            "word_count": len(words),
        },
    )


def _contains_term(answer: str, term: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(term)}(?!\w)",
        answer,
        flags=re.IGNORECASE,
    ) is not None
