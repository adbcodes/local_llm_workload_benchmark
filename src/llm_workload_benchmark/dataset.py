from __future__ import annotations

import json
import math
import re
import csv
import io
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_workload_benchmark.evaluation import EvaluationResult
from llm_workload_benchmark.answer_parser import ParsedAnswer, normalize_text, parse_answer

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
    "set_match",
    "behavior_rules",
    "tool_trace",
    "confidence_value",
]
BehaviorDecision = Literal[
    "fabricated_entity",
    "unanswerable",
    "clarify",
    "correct_false_premise",
    "flag_conflict",
    "benign_completion",
]
PrimaryOutcome = Literal["semantic", "protocol", "integration"]
PrimaryMetric = Literal[
    "semantic_pass_rate",
    "protocol_pass_rate",
    "integration_success_rate",
]
PartialCreditMetric = Literal[
    "mean_semantic_score",
    "mean_protocol_score",
    "mean_integration_score",
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


class EvaluationPolicy(BaseModel):
    """Declare which independent evaluation outcome represents benchmark success."""

    model_config = ConfigDict(extra="forbid")

    primary_outcome: PrimaryOutcome
    primary_metric: PrimaryMetric
    protocol_requirement: Literal["required", "diagnostic", "not_applicable"]
    partial_credit_metric: PartialCreditMetric

    @model_validator(mode="after")
    def metric_names_match_the_primary_outcome(self) -> Self:
        expected_metrics: dict[
            PrimaryOutcome,
            tuple[PrimaryMetric, PartialCreditMetric],
        ] = {
            "semantic": ("semantic_pass_rate", "mean_semantic_score"),
            "protocol": ("protocol_pass_rate", "mean_protocol_score"),
            "integration": ("integration_success_rate", "mean_integration_score"),
        }
        expected_primary, expected_partial = expected_metrics[self.primary_outcome]
        if self.primary_metric != expected_primary:
            raise ValueError(
                f"primary outcome {self.primary_outcome!r} requires "
                f"primary metric {expected_primary!r}"
            )
        if self.partial_credit_metric != expected_partial:
            raise ValueError(
                f"primary outcome {self.primary_outcome!r} requires "
                f"partial-credit metric {expected_partial!r}"
            )
        if (
            self.primary_outcome == "protocol"
            and self.protocol_requirement != "required"
        ):
            raise ValueError(
                "a protocol-primary benchmark must require protocol compliance"
            )
        return self


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


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class DatasetItem(BaseModel):
    """A common envelope shared by every benchmark item."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    benchmark: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    subcategory: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    difficulty: Difficulty
    split: Split
    visibility: Literal["public", "held_out"] = "public"
    prompt: str = Field(min_length=1)
    conversation: list[ChatMessage] | None = None
    response_contract: ResponseContract
    expected: dict[str, Any]
    scoring: ScoringSpec
    provenance: Provenance
    tags: list[str] = Field(default_factory=list)
    variant_of: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    source_item: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    @model_validator(mode="after")
    def scoring_contract_is_consistent(self) -> Self:
        if self.conversation is not None:
            if not self.conversation or self.conversation[-1].role != "user":
                raise ValueError("conversation must be non-empty and end with a user turn")
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
        elif method == "set_match":
            if contract_type not in {"text", "json"} or not isinstance(value, list):
                raise ValueError("set_match requires a text or json contract and list value")
            if not value or any(not isinstance(entry, str) for entry in value):
                raise ValueError("set_match expected value must contain strings")
        elif method == "behavior_rules":
            if contract_type != "text" or not isinstance(value, dict):
                raise ValueError("behavior_rules requires a text contract and object value")
            try:
                BehaviorDecisionSpec.model_validate(value)
            except ValidationError as error:
                raise ValueError(
                    "behavior_rules requires a structured decision contract"
                ) from error
        elif method == "tool_trace":
            if contract_type != "json" or not isinstance(value, dict):
                raise ValueError("tool_trace requires a json contract and object value")
            if set(value) - {"calls", "observations", "final_state"} or not isinstance(value.get("calls"), list):
                raise ValueError("tool_trace requires calls and optional final_state")
            if "observations" in value and not isinstance(value["observations"], list):
                raise ValueError("tool_trace observations must be a list")
            for call in value["calls"]:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"tool", "arguments"}
                    or not isinstance(call["tool"], str)
                    or not isinstance(call["arguments"], dict)
                ):
                    raise ValueError("tool_trace calls require tool and arguments")
        elif method == "confidence_value":
            if contract_type != "text" or not isinstance(value, dict):
                raise ValueError("confidence_value requires a text contract and object value")
            if set(value) != {"answer"} or not isinstance(
                value["answer"], (str, int, float, bool)
            ):
                raise ValueError("confidence_value expected value requires one scalar answer")
        elif not isinstance(value, (str, int, float, bool)):
            raise ValueError("exact_match requires a scalar expected value")
        _validate_scoring_parameters(method, self.scoring.parameters)
        return self


class BenchmarkDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    suite: Literal["A", "B", "C", "D", "E"] | None = None
    status: Literal["planned", "started", "complete", "redesigning"] = "started"
    execution_mode: Literal[
        "single_turn",
        "multi_turn",
        "tool_scenario",
        "paired_variants",
    ] = "single_turn"
    task_types: list[str] = Field(default_factory=list)
    reporting_dimensions: list[str] = Field(default_factory=list)
    redesign_notes: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    evaluation_policy: EvaluationPolicy
    score_formula: Literal[
        "mean_score",
        "accuracy_minus_hallucination",
        "clean_score_retained",
    ] = "mean_score"
    items_path: str = Field(min_length=1)
    authoring_paths: list[str] = Field(default_factory=list)
    current_question_count: int = Field(ge=0)
    target_question_count: int = Field(gt=0)
    current_difficulty_distribution: dict[Difficulty, int]
    difficulty_distribution: dict[Difficulty, int]
    target_visibility_distribution: dict[Literal["public", "held_out"], int] | None = None
    order_rule: Literal["easy_to_hard"]
    scoring_methods: list[ScoringMethod] = Field(min_length=1)

    @model_validator(mode="after")
    def difficulty_counts_match_target(self) -> Self:
        if (
            self.evaluation_policy.primary_outcome == "integration"
            and self.execution_mode != "tool_scenario"
        ):
            raise ValueError(
                "an integration-primary benchmark must use tool_scenario execution"
            )
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
        if any(count < 0 for count in self.difficulty_distribution.values()):
            raise ValueError("target difficulty counts cannot be negative")
        if sum(self.difficulty_distribution.values()) != self.target_question_count:
            raise ValueError(
                "difficulty_distribution must sum to target_question_count"
            )
        if self.target_visibility_distribution is not None:
            if set(self.target_visibility_distribution) != {"public", "held_out"}:
                raise ValueError(
                    "target_visibility_distribution requires public and held_out"
                )
            if sum(self.target_visibility_distribution.values()) != self.target_question_count:
                raise ValueError(
                    "target_visibility_distribution must sum to target_question_count"
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
    visibilities: list[Literal["public", "held_out"]] | None = None

    @model_validator(mode="after")
    def filters_are_not_empty(self) -> Self:
        for name in (
            "ids",
            "subcategories",
            "difficulties",
            "splits",
            "review_statuses",
            "visibilities",
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


class BehaviorDecisionSpec(BaseModel):
    """Deterministic semantic evidence for one behavior decision."""

    model_config = ConfigDict(extra="forbid")

    decision: BehaviorDecision
    reference_answer: str = Field(min_length=1)
    evidence_patterns: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def patterns_compile(self) -> Self:
        for pattern in [*self.evidence_patterns, *self.forbidden_patterns]:
            if not pattern:
                raise ValueError("behavior regex patterns cannot be empty")
            try:
                re.compile(pattern, flags=re.IGNORECASE)
            except re.error as error:
                raise ValueError(f"invalid behavior regex pattern: {pattern!r}") from error
        if self.decision == "benign_completion" and not self.evidence_patterns:
            raise ValueError("benign_completion requires evidence patterns")
        return self


def _validate_scoring_parameters(
    method: ScoringMethod,
    parameters: dict[str, Any],
) -> None:
    allowed_parameters: dict[ScoringMethod, set[str]] = {
        "numeric_tolerance": {
            "absolute_tolerance",
            "allow_surrounding_text",
            "answer_unit",
            "unit_aliases",
        },
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
        "set_match": {"separator", "case_sensitive"},
        "behavior_rules": {"case_sensitive"},
        "tool_trace": {"allow_diagnostic_normalization"},
        "confidence_value": {
            "answer_type",
            "absolute_tolerance",
            "case_sensitive",
            "answer_unit",
            "unit_aliases",
        },
    }
    unknown = set(parameters) - allowed_parameters[method]
    if unknown:
        raise ValueError(f"unknown {method} parameters: {sorted(unknown)}")

    for name in ("allow_surrounding_text", "strip", "case_sensitive"):
        if name in parameters and not isinstance(parameters[name], bool):
            raise ValueError(f"{name} must be a boolean")

    if method in {"numeric_tolerance", "confidence_value"}:
        answer_unit = parameters.get("answer_unit")
        if answer_unit is not None and (
            not isinstance(answer_unit, str) or not answer_unit.strip()
        ):
            raise ValueError("answer_unit must be a non-empty string")
        unit_aliases = parameters.get("unit_aliases", [])
        if (
            not isinstance(unit_aliases, list)
            or any(not isinstance(alias, str) or not alias.strip() for alias in unit_aliases)
        ):
            raise ValueError("unit_aliases must be a list of non-empty strings")
        if unit_aliases and answer_unit is None:
            raise ValueError("unit_aliases requires answer_unit")

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
    if method == "set_match":
        separator = parameters.get("separator", ",")
        if not isinstance(separator, str) or not separator:
            raise ValueError("set_match separator must be a non-empty string")
        return
    if method == "behavior_rules":
        return
    if method == "tool_trace":
        diagnostic = parameters.get("allow_diagnostic_normalization", False)
        if not isinstance(diagnostic, bool):
            raise ValueError("allow_diagnostic_normalization must be a boolean")
        return
    if method == "confidence_value":
        if parameters.get("answer_type", "exact") not in {"exact", "numeric"}:
            raise ValueError("confidence_value answer_type must be exact or numeric")
        tolerance = parameters.get("absolute_tolerance", 0)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("confidence_value absolute_tolerance must be non-negative")
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
        if parameters.get("rubric") not in {
            "grounded_summary_v1",
            "communication_quality_v1",
        }:
            raise ValueError(
                "llm_judge rubric must be grounded_summary_v1 or communication_quality_v1"
            )
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
        "comma_separated",
        "sorted_numeric",
        "excluded_values",
        "item_prefix",
        "numbered_list",
        "sorted_alphabetically",
        "max_words_per_line",
        "forbidden_item_character",
        "list_item_descriptions",
        "list_group_balance",
        "json_only",
        "exact_json_keys",
        "json_field_constraints",
        "json_key_order",
        "json_array_field_equals",
        "json_array_required_keys",
        "json_array_sorted_by",
        "json_derived_bands",
        "json_summary_counts",
        "csv_format",
        "csv_sorted_by",
        "csv_year_format",
        "csv_final_row",
        "csv_year_min",
        "csv_tie_sort",
        "word_range",
        "required_forbidden_terms",
        "exact_paragraphs",
        "json_label_array",
        "label_domain",
        "classification_order",
        "spam_count_consistent",
        "boundary",
        "yaml_only",
        "exact_top_level_keys",
        "required_top_level_keys",
        "yaml_field_constraints",
        "first_line_comment_prefix",
        "yaml_healthcheck",
        "sorted_by_points",
        "uppercase_items",
        "ranked_items",
        "ties_alphabetical",
    }
    unknown_rules = set(rules) - allowed_rules
    if unknown_rules:
        raise ValueError(f"unknown constraint rules: {sorted(unknown_rules)}")
    for name in (
        "max_words",
        "exact_words",
        "exact_sentences",
        "max_words_per_line",
        "exact_paragraphs",
    ):
        if name in rules:
            value = rules[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
    if "max_words" in rules and "exact_words" in rules:
        if rules["exact_words"] > rules["max_words"]:
            raise ValueError("exact_words cannot exceed max_words")
    for name in (
        "required_terms",
        "forbidden_terms",
        "forbidden_punctuation",
        "excluded_values",
        "exact_json_keys",
        "json_key_order",
        "json_array_required_keys",
        "csv_final_row",
        "label_domain",
        "exact_top_level_keys",
        "required_top_level_keys",
    ):
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
    if not isinstance(content, dict) or len(content) != 1:
        raise ValueError("constraint_rules requires exactly one content checker")
    content_kind, content_value = next(iter(content.items()))
    if content_kind == "required_facts":
        facts = content_value
        if not isinstance(facts, list) or not facts:
            raise ValueError("required_facts must be a non-empty list")
        seen_names: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) != {"name", "any_of"}:
                raise ValueError("each required fact needs exactly name and any_of")
            name = fact["name"]
            if not isinstance(name, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*", name
            ):
                raise ValueError("required fact names must use snake_case")
            if name in seen_names:
                raise ValueError(f"duplicate required fact name {name!r}")
            seen_names.add(name)
            _validate_nonempty_strings(fact["any_of"], f"required fact {name}")
    elif content_kind == "required_values":
        if not isinstance(content_value, dict):
            raise ValueError("required_values must be an object")
        if set(content_value) - {"values", "separator", "strip_prefix"}:
            raise ValueError("required_values has unknown fields")
        _validate_nonempty_strings(content_value.get("values"), "required values")
        separator = content_value.get("separator")
        if not isinstance(separator, str) or not separator:
            raise ValueError("required_values separator must be non-empty")
        strip_prefix = content_value.get("strip_prefix", "")
        if not isinstance(strip_prefix, str):
            raise ValueError("required_values strip_prefix must be a string")
    elif content_kind == "csv_records":
        if (
            not isinstance(content_value, list)
            or not content_value
            or any(not isinstance(row, list) or not row for row in content_value)
        ):
            raise ValueError("csv_records must be a non-empty list of rows")
    elif content_kind == "classification_labels":
        _validate_nonempty_strings(content_value, "classification labels")
    elif content_kind in {"exact_json", "exact_yaml"}:
        try:
            json.dumps(content_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{content_kind} must be JSON serializable") from error
        if content_kind == "exact_yaml" and not isinstance(content_value, dict):
            raise ValueError("exact_yaml must be an object")
    elif content_kind == "none":
        if content_value is not True:
            raise ValueError("the none content checker must be true")
    else:
        raise ValueError(f"unknown content checker: {content_kind!r}")


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


def load_dataset(path: Path, *, allow_empty: bool = False) -> list[DatasetItem]:
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
            gold_answer = gold_answer_text(item)
            if not score_answer(item, gold_answer).passed:
                raise DatasetError(
                    f"expected answer for item {item.id!r} does not satisfy its scorer"
                )
        if item.id in seen_ids:
            raise DatasetError(f"duplicate item id {item.id!r} in {path}")
        seen_ids.add(item.id)
        items.append(item)

    if not items and not allow_empty:
        raise DatasetError(f"dataset file contains no items: {path}")
    if items:
        _validate_difficulty_progression(items, path)
    return items


def gold_answer_text(item: DatasetItem) -> str:
    """Build a minimal answer that proves a deterministic item is satisfiable."""

    value = item.expected["value"]
    if item.scoring.method == "behavior_rules":
        return BehaviorDecisionSpec.model_validate(value).reference_answer
    if item.scoring.method == "confidence_value":
        return f"{value['answer']}\nconfidence: 100"
    if item.scoring.method == "set_match" and item.response_contract.type == "text":
        return item.scoring.parameters.get("separator", ",").join(value)
    if item.response_contract.type == "json":
        return json.dumps(value, separators=(",", ":"))
    return str(value)


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
        items = load_dataset(
            item_path,
            allow_empty=definition.current_question_count == 0,
        )
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
    _validate_source_lineage(all_items)

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
        and (
            filters.visibilities is None
            or item.visibility in filters.visibilities
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


def _validate_source_lineage(items: dict[str, DatasetItem]) -> None:
    for item in items.values():
        if item.source_item is None:
            continue
        if item.source_item == item.id:
            raise DatasetError(f"item {item.id!r} cannot reference itself as source_item")
        if item.source_item not in items:
            raise DatasetError(
                f"item {item.id!r} references unknown source_item {item.source_item!r}"
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
    extraction_details: dict[str, Any] = {}
    scoring_answer = answer
    if item.benchmark == "applied_reasoning":
        parsed = _parse_applied_reasoning_answer(item, answer)
        extraction_details = _answer_parse_details(parsed)
        extraction_details.update(
            {
                "answer_extraction": (
                    "final_marker"
                    if "extract_final_line" in parsed.normalization_steps
                    else "last_line_fallback"
                ),
                "final_marker_compliant": not parsed.protocol_violations,
                "final_answer": parsed.extracted_answer,
            }
        )
        if not parsed.parsed:
            extraction_details["reason"] = {
                "ambiguous": "multiple_final_answers",
                "missing": "missing_final_answer",
            }.get(parsed.status, "unparseable_final_answer")
            return EvaluationResult(
                type="deterministic",
                evaluator=method,
                version=2,
                passed=False,
                score=0,
                details=extraction_details,
            )
        scoring_answer = parsed.extracted_answer or ""
    if method == "numeric_tolerance":
        score = _score_numeric(item, scoring_answer)
    elif method == "rational_value":
        score = _score_rational(item, scoring_answer)
    elif method == "date_value":
        score = _score_date(item, scoring_answer)
    elif method == "exact_match":
        score = _score_exact(item, scoring_answer)
    elif method == "json_exact":
        score = _score_json(item, answer)
    elif method == "set_match":
        score = _score_set(item, answer)
    elif method == "behavior_rules":
        score = _score_behavior(item, answer)
    elif method == "tool_trace":
        score = _score_tool_trace(item, answer)
    elif method == "confidence_value":
        score = _score_confidence(item, answer)
    else:
        score = _score_constraints(item, answer)
    return EvaluationResult(
        type="deterministic",
        evaluator=method,
        version=2 if item.benchmark == "applied_reasoning" else 1,
        passed=score.passed,
        score=score.score,
        details={**score.details, **extraction_details},
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


def _parse_applied_reasoning_answer(item: DatasetItem, answer: str) -> ParsedAnswer:
    kind = "text"
    options: dict[str, str] | None = None
    date_formats: tuple[str, ...] = ()
    prompt_options = _prompt_options(item.prompt)
    if item.response_contract.format == "source_label" and prompt_options:
        kind = "option"
        options = prompt_options
    elif item.scoring.method == "numeric_tolerance":
        kind = "number"
    elif item.scoring.method == "date_value":
        kind = "date"
        date_formats = _declared_date_formats(item.response_contract.format)
    return parse_answer(
        answer,
        kind,
        require_final=True,
        recover_missing_final=True,
        option_text=options,
        date_formats=date_formats,
        **_number_parse_options(item.scoring.parameters),
    )


def _prompt_options(prompt: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for wrapped, bare, text in re.findall(
        r"(?m)^\s*(?:\(([A-Za-z])\)|([A-Za-z])[\).])\s*(.+?)\s*$",
        prompt,
    ):
        options[(wrapped or bare).upper()] = text.strip()
    return options


def _answer_parse_details(parsed: ParsedAnswer) -> dict[str, Any]:
    return {
        "answer_parse_status": parsed.status,
        "answer_extraction": (
            parsed.normalization_steps[0] if parsed.normalization_steps else None
        ),
        "normalization_steps": parsed.normalization_steps,
        "protocol_violations": parsed.protocol_violations,
        "parsed_value": parsed.value,
    }


def _number_parse_options(parameters: dict[str, Any]) -> dict[str, Any]:
    answer_unit = parameters.get("answer_unit")
    aliases = parameters.get("unit_aliases", [])
    return {
        "answer_unit": answer_unit if isinstance(answer_unit, str) else None,
        "unit_aliases": tuple(aliases) if isinstance(aliases, list) else (),
    }


def _extract_numeric_values(answer: str) -> list[float]:
    matches = re.findall(
        r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        answer,
    )
    candidates: list[float] = []
    for match in matches:
        try:
            value = float(match.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            candidates.append(value)
    return candidates


def _score_numeric(item: DatasetItem, answer: str) -> ScoreResult:
    expected = float(item.expected["value"])
    tolerance = float(item.scoring.parameters.get("absolute_tolerance", 0))
    parsed = parse_answer(
        answer,
        "number",
        **_number_parse_options(item.scoring.parameters),
    )
    if parsed.parsed:
        actual = float(parsed.value)
    elif item.scoring.parameters.get("allow_surrounding_text", False):
        candidates = _extract_numeric_values(answer)
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
        return ScoreResult(
            passed=False,
            score=0,
            details={"reason": "not_numeric", **_answer_parse_details(parsed)},
        )

    difference = abs(actual - expected)
    passed = difference <= tolerance
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={
            "actual": actual,
            "difference": difference,
            "tolerance": tolerance,
            **_answer_parse_details(parsed),
        },
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
    parse_details: dict[str, Any] = {}
    options = _prompt_options(item.prompt)
    if item.response_contract.format == "source_label" and options:
        parsed_actual = parse_answer(actual, "option", option_text=options)
        parsed_expected = parse_answer(expected, "option", option_text=options)
        parse_details = _answer_parse_details(parsed_actual)
        if not parsed_actual.parsed or not parsed_expected.parsed:
            return ScoreResult(
                passed=False,
                score=0,
                details={"reason": "invalid_option_answer", **parse_details},
            )
        actual = str(parsed_actual.value)
        expected = str(parsed_expected.value)
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
    if not item.scoring.parameters.get("case_sensitive", False):
        actual, actual_steps = normalize_text(actual)
        expected, _ = normalize_text(expected)
        parse_details.setdefault("normalization_steps", []).extend(
            step
            for step in actual_steps
            if step not in parse_details.get("normalization_steps", [])
        )
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={
            "actual": actual,
            "expected": expected,
            "candidates": candidates,
            **parse_details,
        },
    )


def _score_date(item: DatasetItem, answer: str) -> ScoreResult:
    formats = _declared_date_formats(item.response_contract.format)
    expected_parsed = parse_answer(str(item.expected["value"]), "date", date_formats=formats)
    parsed = parse_answer(answer, "date", date_formats=formats)
    if not expected_parsed.parsed:
        return ScoreResult(
            passed=False,
            score=0,
            details={"reason": "invalid_expected_date"},
        )
    if not parsed.parsed:
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "missing_or_ambiguous_date_answer",
                "expected": expected_parsed.value,
                **_answer_parse_details(parsed),
            },
        )

    actual = parsed.value
    expected = expected_parsed.value
    passed = actual == expected
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={"actual": actual, "expected": expected, **_answer_parse_details(parsed)},
    )


def _declared_date_formats(contract_format: str | None) -> tuple[str, ...]:
    if contract_format == "common_unambiguous_date":
        return (
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
        )
    if contract_format == "DD/MM/YYYY":
        return ("%d/%m/%Y",)
    if contract_format == "MM/DD/YYYY":
        return ("%m/%d/%Y",)
    return ("%Y-%m-%d",)


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
    parsed = parse_answer(
        answer,
        "json",
        allow_recovery=item.scoring.parameters.get(
            "allow_diagnostic_normalization", True
        ),
    )
    protocol_compliant = parsed.parsed and not parsed.protocol_violations
    if not parsed.parsed:
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "invalid_json",
                "protocol_compliant": False,
                "diagnostic_json_valid": False,
                **_answer_parse_details(parsed),
            },
        )
    actual = parsed.value

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
    leaf_accuracy = len(matched_paths) / len(all_paths) if all_paths else 1.0
    content_score = leaf_accuracy
    protocol_score = float(protocol_compliant)
    score = content_score
    content_exact = _json_values_equal(actual, expected)
    # Commit 5 will derive the headline verdict from each benchmark policy.
    passed = protocol_compliant and content_exact
    return ScoreResult(
        passed=passed,
        score=score,
        details={
            "protocol_compliant": protocol_compliant,
            "content_exact": content_exact,
            "diagnostic_wrapper": (
                parsed.protocol_violations[0] if parsed.protocol_violations else None
            ),
            "leaf_accuracy": leaf_accuracy,
            "content_score": content_score,
            "protocol_score": protocol_score,
            **_answer_parse_details(parsed),
            "missing_paths": sorted(set(expected_leaves) - set(actual_leaves)),
            "extra_paths": sorted(set(actual_leaves) - set(expected_leaves)),
        },
    )


def _score_set(item: DatasetItem, answer: str) -> ScoreResult:
    expected_values = [str(value).strip() for value in item.expected["value"]]
    if item.response_contract.type == "json":
        parsed_answer = parse_answer(answer, "json")
        if not parsed_answer.parsed:
            return ScoreResult(
                passed=False,
                score=0,
                details={"reason": "invalid_json", "protocol_compliant": False},
            )
        parsed = parsed_answer.value
        if not isinstance(parsed, list) or any(not isinstance(value, str) for value in parsed):
            return ScoreResult(
                passed=False,
                score=0,
                details={"reason": "expected_json_string_array", "protocol_compliant": True},
            )
        actual_values = [value.strip() for value in parsed]
    else:
        separator = item.scoring.parameters.get("separator", ",")
        raw_values = [value.strip() for value in answer.strip().split(separator) if value.strip()]
        actual_values = raw_values
    case_sensitive = item.scoring.parameters.get("case_sensitive", False)
    normalize = (lambda value: value) if case_sensitive else (lambda value: normalize_text(value)[0])
    expected = {normalize(value) for value in expected_values}
    actual = {normalize(value) for value in actual_values}
    intersection = expected & actual
    union = expected | actual
    jaccard = len(intersection) / len(union) if union else 1.0
    precision = len(intersection) / len(actual) if actual else float(not expected)
    recall = len(intersection) / len(expected) if expected else float(not actual)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact_match = actual == expected and len(actual_values) == len(expected_values)
    return ScoreResult(
        passed=exact_match,
        score=f1,
        details={
            "expected": sorted(expected),
            "actual": sorted(actual),
            "missing": sorted(expected - actual),
            "extra": sorted(actual - expected),
            "duplicate_count": len(actual_values) - len(actual),
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
        },
    )


def _score_behavior(item: DatasetItem, answer: str) -> ScoreResult:
    specification = BehaviorDecisionSpec.model_validate(item.expected["value"])
    decision_patterns: dict[BehaviorDecision, tuple[str, ...]] = {
        "fabricated_entity": (
            r"(?is)^\s*.{0,100}\b(?:does\s+not\s+exist|nonexistent|fictional|"
            r"made[\s-]?up|invented|imaginary|not\s+(?:a\s+)?(?:real|standard|valid))\b",
            r"(?is)^\s*.{0,100}\bthere\s+(?:is|was)\s+no\b",
        ),
        "unanswerable": (
            r"\b(?:cannot|can['’]?t|unable\s+to|no\s+way\s+to)\s+"
            r"(?:determine|know|identify|answer|infer|verify)\b",
            r"\bnot\s+(?:enough|sufficient)\s+(?:information|data|context)\b",
            r"\b(?:is|are|was|were)\s+not\s+(?:provided|given|specified|included)\b",
            r"\b(?:missing|omits?|omitted|redacted|unavailable)\b",
        ),
        "clarify": (
            r"(?is)^\s*.{0,80}\b(?:which|what|when|where|who|how)\b",
            r"\b(?:please\s+)?(?:clarify|specify|confirm)\b",
        ),
        "correct_false_premise": (
            r"\b(?:false\s+premise|incorrect|not\s+true|never|did\s+not|does\s+not|"
            r"no\s+year|is\s+odd|orbits?\s+the)\b",
        ),
        "flag_conflict": (
            r"\b(?:conflict(?:ing)?|inconsisten(?:t|cy)|contradict(?:s|ory|ion)?|"
            r"disagree|cannot\s+determine|need(?:s)?\s+clarification)\b",
        ),
        "benign_completion": (r"\S",),
    }
    evidence_matches = [
        pattern
        for pattern in specification.evidence_patterns
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
    ]
    decision_matches = [
        pattern
        for pattern in decision_patterns[specification.decision]
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
    ]
    evidence_near_start = [
        pattern
        for pattern in specification.evidence_patterns
        if re.search(pattern, answer[:160], flags=re.IGNORECASE | re.DOTALL)
    ]
    if specification.decision != "benign_completion":
        decision_matches.extend(
            pattern for pattern in evidence_near_start if pattern not in decision_matches
        )
    assertion_before_decision = False
    if specification.decision == "fabricated_entity" and decision_matches:
        correction = re.search(
            r"\b(?:does\s+not\s+exist|nonexistent|fictional|made[\s-]?up|"
            r"invented|imaginary|not\s+(?:a\s+)?(?:real|standard|valid)|"
            r"there\s+(?:is|was)\s+no)\b",
            answer,
            flags=re.IGNORECASE,
        )
        assertion = re.search(
            r"\b(?:accepts?|provides?|supports?|includes?|was\s+awarded|"
            r"received|founded|announced|ruled)\b",
            answer,
            flags=re.IGNORECASE,
        )
        assertion_before_decision = bool(
            assertion and correction and assertion.start() < correction.start()
        )
    global_forbidden = (
        (
            r"\b(?:cannot|can['’]?t|won['’]?t|unable\s+to)\s+"
            r"(?:help|assist|provide|comply)\b",
        )
        if specification.decision == "benign_completion"
        else ()
    )
    forbidden_matches = [
        pattern
        for pattern in [*specification.forbidden_patterns, *global_forbidden]
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
    ]
    checks = {
        "nonempty": bool(answer.strip()),
        "decision": bool(decision_matches),
        "evidence": (
            bool(evidence_matches)
            if specification.decision == "benign_completion"
            else True
        ),
        "no_forbidden_behavior": not forbidden_matches,
        "no_assertion_before_correction": not assertion_before_decision,
    }
    passed = all(checks.values())
    return ScoreResult(
        passed=passed,
        score=float(passed),
        details={
            "behavior_decision": specification.decision,
            "checks": checks,
            "matched_decision_patterns": decision_matches,
            "matched_evidence_patterns": evidence_matches,
            "matched_forbidden_patterns": forbidden_matches,
        },
    )


def _score_tool_trace(item: DatasetItem, answer: str) -> ScoreResult:
    parsed = parse_answer(answer, "tool", allow_recovery=True)
    protocol_violations = list(parsed.protocol_violations)
    protocol_compliant = parsed.parsed and not protocol_violations
    actual = parsed.value
    if isinstance(actual, dict) and isinstance(actual.get("tool"), str):
        actual = {"calls": [actual]}
        protocol_violations.append("single_tool_call_envelope")
        protocol_compliant = False
    elif isinstance(actual, dict) and "final_state" in actual and "calls" not in actual:
        actual = {"calls": [], "observations": [], **actual}
        protocol_violations.append("final_state_only_envelope")
        protocol_compliant = False
    if not isinstance(actual, dict) or not isinstance(actual.get("calls"), list):
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "invalid_tool_trace",
                **_answer_parse_details(parsed),
                "parseable": False,
                "parse_status": parsed.status,
                "protocol_compliant": False,
                "protocol_violations": protocol_violations,
                "diagnostic_wrapper": (
                    protocol_violations[0] if protocol_violations else None
                ),
            },
        )
    expected = item.expected["value"]
    expected_calls = expected["calls"]
    actual_calls = actual["calls"]
    arguments_well_formed = all(
        isinstance(call, dict) and isinstance(call.get("arguments"), dict)
        for call in actual_calls
    )
    call_scores: list[float] = []
    tool_choice_scores: list[float] = []
    argument_scores: list[float] = []
    for index in range(max(len(expected_calls), len(actual_calls))):
        if index >= len(expected_calls) or index >= len(actual_calls):
            call_scores.append(0.0)
            tool_choice_scores.append(0.0)
            argument_scores.append(0.0)
            continue
        expected_call = expected_calls[index]
        actual_call = actual_calls[index]
        if not isinstance(actual_call, dict):
            call_scores.append(0.0)
            tool_choice_scores.append(0.0)
            argument_scores.append(0.0)
            continue
        tool_ok = actual_call.get("tool") == expected_call["tool"]
        args_ok = _contract_value_matches(
            actual_call.get("arguments"), expected_call["arguments"]
        )
        tool_choice_scores.append(float(tool_ok))
        argument_scores.append(float(args_ok))
        call_scores.append((float(tool_ok) + float(args_ok)) / 2)
    calls_score = sum(call_scores) / len(call_scores) if call_scores else 1.0
    tool_choice_accuracy = (
        sum(tool_choice_scores) / len(tool_choice_scores)
        if tool_choice_scores
        else 1.0
    )
    argument_accuracy = (
        sum(argument_scores) / len(argument_scores) if argument_scores else 1.0
    )
    expected_order = [call["tool"] for call in expected_calls]
    actual_order = [
        call.get("tool") if isinstance(call, dict) else None for call in actual_calls
    ]
    order_ok = actual_order == expected_order
    call_count_ok = len(actual_calls) == len(expected_calls)
    unnecessary_calls_ok = len(actual_calls) <= len(expected_calls)
    final_state_required = "final_state" in expected
    final_state_ok = not final_state_required or _contract_value_matches(
        actual.get("final_state"), expected["final_state"]
    )
    observations_required = "observations" in expected
    observations_ok = not observations_required or _contract_value_matches(
        actual.get("observations"), expected["observations"]
    )
    score_parts = (
        [tool_choice_accuracy, argument_accuracy, float(order_ok), float(call_count_ok)]
        + ([float(observations_ok)] if observations_required else [])
        + ([float(final_state_ok)] if final_state_required else [])
    )
    score = sum(score_parts) / len(score_parts)
    integration_success = score == 1.0 and unnecessary_calls_ok
    passed = protocol_compliant and integration_success
    return ScoreResult(
        passed=passed,
        score=score,
        details={
            **_answer_parse_details(parsed),
            "protocol_compliant": protocol_compliant,
            "protocol_violations": protocol_violations,
            "diagnostic_wrapper": (
                protocol_violations[0] if protocol_violations else None
            ),
            "parseable": True,
            "parse_status": parsed.status,
            "call_scores": call_scores,
            "calls_score": calls_score,
            "tool_choice_accuracy": tool_choice_accuracy,
            "argument_accuracy": argument_accuracy,
            "arguments_well_formed": arguments_well_formed,
            "order_ok": order_ok,
            "call_count_ok": call_count_ok,
            "unnecessary_calls_ok": unnecessary_calls_ok,
            "call_count_expected": len(expected_calls),
            "call_count_actual": len(actual_calls),
            "final_state_ok": final_state_ok,
            "observations_ok": observations_ok,
            "observation_use_accuracy": float(observations_ok),
            "final_state_accuracy": float(final_state_ok),
            "integration_success": integration_success,
        },
    )


def _contract_value_matches(actual: Any, expected: Any) -> bool:
    """Compare tool values by declared expected type, allowing safe scalar cleanup."""

    if isinstance(expected, bool) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, str):
            parsed = parse_answer(actual, "number")
            actual = parsed.value if parsed.parsed else actual
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == float(expected)
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip() == expected.strip()
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contract_value_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _contract_value_matches(left, right)
            for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _score_confidence(item: DatasetItem, answer: str) -> ScoreResult:
    answer_type = item.scoring.parameters.get("answer_type", "exact")
    parsed = parse_answer(
        answer,
        "confidence",
        confidence_answer_kind="number" if answer_type == "numeric" else "text",
        **_number_parse_options(item.scoring.parameters),
    )
    if not parsed.parsed:
        return ScoreResult(
            passed=False,
            score=0,
            details={
                "reason": "missing_confidence_line",
                "confidence": None,
                **_answer_parse_details(parsed),
            },
        )
    confidence = int(parsed.value["confidence"])
    actual_value = parsed.value["answer"]
    expected = item.expected["value"]["answer"]
    if answer_type == "numeric":
        answer_correct = abs(float(actual_value) - float(expected)) <= float(
            item.scoring.parameters.get("absolute_tolerance", 0)
        )
    else:
        if item.scoring.parameters.get("case_sensitive", False):
            actual_value = answer.splitlines()[0].strip()
            expected_value = str(expected).strip()
        else:
            expected_value = normalize_text(str(expected))[0]
        answer_correct = actual_value == expected_value
    return ScoreResult(
        passed=answer_correct,
        score=float(answer_correct),
        details={
            "answer_correct": answer_correct,
            "confidence": confidence,
            "confidence_probability": confidence / 100,
            "brier_component": (confidence / 100 - float(answer_correct)) ** 2,
            **_answer_parse_details(parsed),
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
    comma_items = _comma_items(answer)
    numbered_items = _numbered_items(answer)
    parsed_json = _parse_json(answer)
    parsed_csv = _parse_csv(answer)
    parsed_yaml = _parse_yaml_mapping(answer)

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
    if "comma_separated" in rules:
        pattern = rules["comma_separated"]["item_pattern"]
        checks["comma_separated"] = (
            comma_items is not None
            and bool(comma_items)
            and all(re.fullmatch(pattern, value) for value in comma_items)
        )
    if "sorted_numeric" in rules:
        numeric_values = _numeric_suffixes(comma_items)
        checks["sorted_numeric"] = numeric_values is not None and numeric_values == sorted(
            numeric_values
        )
    if "excluded_values" in rules:
        normalized_items = _normalized_comma_values(comma_items)
        checks["excluded_values"] = normalized_items is not None and all(
            value.casefold() not in normalized_items
            for value in rules["excluded_values"]
        )
    if "item_prefix" in rules:
        checks["item_prefix"] = comma_items is not None and all(
            value.startswith(rules["item_prefix"]) for value in comma_items
        )
    if "numbered_list" in rules:
        checks["numbered_list"] = (
            numbered_items is not None
            and len(numbered_items) == rules["numbered_list"]["count"]
            and [number for number, _ in numbered_items]
            == list(range(1, len(numbered_items) + 1))
        )
    if "sorted_alphabetically" in rules:
        list_values = _list_values(numbered_items)
        checks["sorted_alphabetically"] = (
            list_values is not None
            and list_values == sorted(list_values, key=str.casefold)
        )
    if "max_words_per_line" in rules:
        nonempty_lines = [line for line in answer.splitlines() if line.strip()]
        checks["max_words_per_line"] = bool(nonempty_lines) and all(
            len(re.findall(r"\b[\w'-]+\b", line, flags=re.UNICODE))
            <= rules["max_words_per_line"]
            for line in nonempty_lines
        )
    if "forbidden_item_character" in rules:
        list_values = _list_values(numbered_items)
        character = rules["forbidden_item_character"].casefold()
        checks["forbidden_item_character"] = (
            list_values is not None
            and all(character not in value.casefold() for value in list_values)
        )
    if "list_item_descriptions" in rules:
        list_values = _list_values(numbered_items)
        checks["list_item_descriptions"] = list_values is not None and all(
            len(value.split(" — ", 1)) == 2
            and len(re.findall(r"\b[\w'-]+\b", value.split(" — ", 1)[1])) >= 2
            for value in list_values
        )
    if "list_group_balance" in rules:
        list_values = _list_values(numbered_items)
        specification = rules["list_group_balance"]
        groups = specification["groups"]
        expected_count = specification["count_per_group"]
        names = (
            [value.split(" — ", 1)[0] for value in list_values]
            if list_values is not None
            else []
        )
        group_counts = Counter(groups.get(name) for name in names)
        checks["list_group_balance"] = (
            bool(names)
            and None not in group_counts
            and set(group_counts) == set(groups.values())
            and all(count == expected_count for count in group_counts.values())
        )
    if "json_only" in rules:
        expected_type = rules["json_only"]
        checks["json_only"] = parsed_json is not None and (
            (expected_type == "object" and isinstance(parsed_json, dict))
            or (expected_type == "array" and isinstance(parsed_json, list))
        )
    if "exact_json_keys" in rules:
        checks["exact_json_keys"] = isinstance(parsed_json, dict) and set(
            parsed_json
        ) == set(rules["exact_json_keys"])
    if "json_field_constraints" in rules:
        checks["json_field_constraints"] = isinstance(
            parsed_json, dict
        ) and _mapping_fields_satisfy(parsed_json, rules["json_field_constraints"])
    if "json_key_order" in rules:
        checks["json_key_order"] = isinstance(parsed_json, dict) and list(
            parsed_json
        ) == rules["json_key_order"]
    json_records = _json_array_records(
        parsed_json, allow_summary="json_summary_counts" in rules
    )
    if "json_array_field_equals" in rules:
        specification = rules["json_array_field_equals"]
        checks["json_array_field_equals"] = bool(json_records) and all(
            record.get(specification["field"]) == specification["equals"]
            for record in json_records
        )
    if "json_array_required_keys" in rules:
        required_keys = set(rules["json_array_required_keys"])
        checks["json_array_required_keys"] = bool(json_records) and all(
            required_keys <= set(record) for record in json_records
        )
    if "json_array_sorted_by" in rules:
        checks["json_array_sorted_by"] = _json_records_are_sorted(
            json_records, rules["json_array_sorted_by"]
        )
    if "json_derived_bands" in rules:
        checks["json_derived_bands"] = _json_records_have_derived_bands(
            json_records, rules["json_derived_bands"]
        )
    if "json_summary_counts" in rules:
        checks["json_summary_counts"] = _json_summary_counts_match(
            parsed_json, rules["json_summary_counts"]
        )
    if "csv_format" in rules:
        specification = rules["csv_format"]
        data_row_count = len(parsed_csv) - 1 if parsed_csv is not None else 0
        checks["csv_format"] = (
            parsed_csv is not None
            and parsed_csv[0] == specification["header"]
            and all(len(row) == len(specification["header"]) for row in parsed_csv)
            and (
                "data_rows" not in specification
                or data_row_count
                == specification["data_rows"]
                + (1 if "csv_final_row" in rules else 0)
            )
            and data_row_count >= specification.get("minimum_data_rows", 1)
        )
    if "csv_sorted_by" in rules:
        checks["csv_sorted_by"] = _csv_column_is_sorted(
            parsed_csv,
            (
                rules["csv_sorted_by"]["column"]
                if isinstance(rules["csv_sorted_by"], dict)
                else rules["csv_sorted_by"]
            ),
            final_row=rules.get("csv_final_row"),
            direction=(
                rules["csv_sorted_by"].get("direction", "ascending")
                if isinstance(rules["csv_sorted_by"], dict)
                else "ascending"
            ),
        )
    if "csv_year_min" in rules:
        checks["csv_year_min"] = _csv_column_meets_minimum(
            parsed_csv, "year", rules["csv_year_min"]
        )
    if "csv_tie_sort" in rules:
        checks["csv_tie_sort"] = _csv_ties_are_sorted(
            parsed_csv, rules["csv_tie_sort"]
        )
    if "csv_year_format" in rules:
        lines = [line for line in answer.strip().splitlines() if line]
        data_lines = lines[1:]
        if "csv_final_row" in rules and data_lines:
            data_lines = data_lines[:-1]
        checks["csv_year_format"] = bool(data_lines) and all(
            re.fullmatch(r".*,(\d{4})", line) is not None
            and not re.search(r",\s*[\"']\d{4}[\"']\s*$", line)
            for line in data_lines
        )
    if "csv_final_row" in rules:
        checks["csv_final_row"] = (
            parsed_csv is not None
            and bool(parsed_csv)
            and parsed_csv[-1] == [str(value) for value in rules["csv_final_row"]]
        )
    if "word_range" in rules:
        checks["word_range"] = (
            rules["word_range"]["min"]
            <= len(words)
            <= rules["word_range"]["max"]
        )
    if "required_forbidden_terms" in rules:
        term_rule = rules["required_forbidden_terms"]
        checks["required_forbidden_terms"] = all(
            _contains_term(answer, term) for term in term_rule["required"]
        ) and all(
            not _contains_term(answer, term) for term in term_rule["forbidden"]
        )
    if "exact_paragraphs" in rules:
        paragraphs = re.split(r"\n[ \t]*\n", answer.strip())
        checks["exact_paragraphs"] = (
            len(paragraphs) == rules["exact_paragraphs"]
            and all(paragraph.strip() for paragraph in paragraphs)
        )
    if "json_label_array" in rules:
        checks["json_label_array"] = _is_label_array(
            parsed_json,
            allow_summary="spam_count_consistent" in rules,
        )
    if "label_domain" in rules:
        labels, _ = _classification_parts(parsed_json)
        checks["label_domain"] = labels is not None and all(
            label in rules["label_domain"] for label in labels
        )
    if "classification_order" in rules:
        labels, _ = _classification_parts(parsed_json)
        checks["classification_order"] = (
            labels is not None
            and [_normalize_label(label) for label in labels]
            == rules["classification_order"]
        )
    if "spam_count_consistent" in rules:
        labels, summary = _classification_parts(parsed_json)
        reported_count = summary.get("spam_count") if summary is not None else None
        checks["spam_count_consistent"] = (
            labels is not None
            and summary is not None
            and isinstance(reported_count, int)
            and not isinstance(reported_count, bool)
            and reported_count
            == sum(_normalize_label(label) == "spam" for label in labels)
        )
    if "boundary" in rules:
        checks["boundary"] = answer.startswith(
            rules["boundary"]["prefix"]
        ) and answer.endswith(rules["boundary"]["suffix"])
    if "yaml_only" in rules:
        checks["yaml_only"] = parsed_yaml is not None
    if "exact_top_level_keys" in rules:
        checks["exact_top_level_keys"] = parsed_yaml is not None and set(
            parsed_yaml
        ) == set(rules["exact_top_level_keys"])
    if "required_top_level_keys" in rules:
        checks["required_top_level_keys"] = parsed_yaml is not None and set(
            rules["required_top_level_keys"]
        ) <= set(parsed_yaml)
    if "yaml_field_constraints" in rules:
        checks["yaml_field_constraints"] = parsed_yaml is not None and (
            _mapping_fields_satisfy(parsed_yaml, rules["yaml_field_constraints"])
        )
    if "first_line_comment_prefix" in rules:
        first_line = answer.splitlines()[0] if answer.splitlines() else ""
        checks["first_line_comment_prefix"] = first_line.startswith(
            rules["first_line_comment_prefix"]
        )
    if "yaml_healthcheck" in rules:
        healthcheck = parsed_yaml.get("healthcheck") if parsed_yaml else None
        checks["yaml_healthcheck"] = (
            isinstance(healthcheck, dict)
            and healthcheck == rules["yaml_healthcheck"]
        )
    if "sorted_by_points" in rules:
        ranked_values = _ranked_comma_values(comma_items)
        points = rules["sorted_by_points"]
        checks["sorted_by_points"] = (
            ranked_values is not None
            and all(value.casefold() in points for value in ranked_values)
            and [points[value.casefold()] for value in ranked_values]
            == sorted(
                [points[value.casefold()] for value in ranked_values], reverse=True
            )
        )
    if "uppercase_items" in rules:
        ranked_values = _ranked_comma_values(comma_items)
        checks["uppercase_items"] = ranked_values is not None and all(
            value == value.upper() for value in ranked_values
        )
    if "ranked_items" in rules:
        checks["ranked_items"] = comma_items is not None and all(
            re.fullmatch(rf"{index}:[A-Za-z]+", value) is not None
            for index, value in enumerate(comma_items, start=1)
        )
    if "ties_alphabetical" in rules:
        ranked_values = _ranked_comma_values(comma_items)
        points = rules["ties_alphabetical"]
        checks["ties_alphabetical"] = (
            ranked_values is not None
            and all(value.casefold() in points for value in ranked_values)
            and all(
                points[left.casefold()] != points[right.casefold()]
                or left.casefold() <= right.casefold()
                for left, right in zip(ranked_values, ranked_values[1:])
            )
        )

    passed_count = sum(checks.values())
    constraint_score = passed_count / len(checks) if checks else 0
    content_preserved, content_score, content_details = _score_constraint_content(
        item,
        answer,
        parsed_csv=parsed_csv,
        parsed_json=parsed_json,
        comma_items=comma_items,
    )
    classification_details: dict[str, Any] = {}
    content = item.scoring.parameters["content_requirements"]
    if "classification_labels" in content:
        labels, summary = _classification_parts(parsed_json)
        normalized = (
            [_normalize_label(label) for label in labels] if labels is not None else []
        )
        gold = content["classification_labels"]
        model_spam_count = sum(label == "spam" for label in normalized)
        reported_count = summary.get("spam_count") if summary else None
        reported_count_is_integer = isinstance(reported_count, int) and not isinstance(
            reported_count, bool
        )
        classification_details = {
            "gold_label_accuracy": content_score,
            "model_spam_count": model_spam_count,
            "gold_spam_count": sum(label == "spam" for label in gold),
            "reported_spam_count": reported_count,
            "count_consistent_with_labels": (
                reported_count_is_integer and reported_count == model_spam_count
            ),
            "count_matches_gold": (
                reported_count_is_integer
                and reported_count
                == sum(label == "spam" for label in gold)
            ),
        }
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
            **content_details,
            "constraint_score": constraint_score,
            "checks": checks,
            "word_count": len(words),
            **classification_details,
        },
    )


def _score_constraint_content(
    item: DatasetItem,
    answer: str,
    *,
    parsed_csv: list[list[str]] | None,
    parsed_json: Any | None,
    comma_items: list[str] | None,
) -> tuple[bool, float, dict[str, Any]]:
    content = item.scoring.parameters["content_requirements"]
    kind, specification = next(iter(content.items()))
    if kind == "none":
        return True, 1.0, {"content_checker": "none"}
    if kind == "required_facts":
        fact_checks = {
            fact["name"]: any(
                _contains_term(answer, phrase) for phrase in fact["any_of"]
            )
            for fact in specification
        }
        score = sum(fact_checks.values()) / len(fact_checks)
        return all(fact_checks.values()), score, {"fact_checks": fact_checks}
    if kind == "required_values":
        actual = _normalized_required_values(comma_items, specification)
        expected = [value.casefold() for value in specification["values"]]
        matches = sum((Counter(actual) & Counter(expected)).values()) if actual else 0
        score = matches / max(len(actual or []), len(expected))
        return Counter(actual or []) == Counter(expected), score, {
            "actual_values": actual,
            "expected_values": expected,
        }
    if kind == "csv_records":
        actual_rows = parsed_csv[1:] if parsed_csv else []
        if actual_rows and actual_rows[-1] == ["END", "END", "0"]:
            actual_rows = actual_rows[:-1]
        expected_rows = [[str(value) for value in row] for row in specification]
        actual_counter = Counter(tuple(row) for row in actual_rows)
        expected_counter = Counter(tuple(row) for row in expected_rows)
        matches = sum((actual_counter & expected_counter).values())
        score = matches / max(len(actual_rows), len(expected_rows))
        return actual_counter == expected_counter, score, {
            "actual_records": actual_rows,
            "expected_records": expected_rows,
        }
    if kind == "exact_json":
        return _score_exact_structure(parsed_json, specification, "json")
    if kind == "exact_yaml":
        return _score_exact_structure(_parse_yaml_mapping(answer), specification, "yaml")
    labels, _ = _classification_parts(parsed_json)
    actual_labels = (
        [_normalize_label(label) for label in labels] if labels is not None else []
    )
    matches = sum(
        actual == expected
        for actual, expected in zip(actual_labels, specification)
    )
    score = matches / max(len(actual_labels), len(specification))
    return actual_labels == specification, score, {
        "actual_labels": actual_labels,
        "expected_labels": specification,
    }


def _comma_items(answer: str) -> list[str] | None:
    if "\n" in answer or "\r" in answer:
        return None
    values = [value.strip() for value in answer.strip().split(",")]
    return values if values and all(values) else None


def _numbered_items(answer: str) -> list[tuple[int, str]] | None:
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    parsed: list[tuple[int, str]] = []
    for line in lines:
        match = re.fullmatch(r"(\d+)\.\s+(.+)", line)
        if match is None:
            return None
        parsed.append((int(match.group(1)), match.group(2).strip()))
    return parsed or None


def _list_values(items: list[tuple[int, str]] | None) -> list[str] | None:
    return [value for _, value in items] if items is not None else None


def _numeric_suffixes(items: list[str] | None) -> list[int] | None:
    if items is None:
        return None
    matches = [re.search(r"(\d+)$", value) for value in items]
    if any(match is None for match in matches):
        return None
    return [int(match.group(1)) for match in matches if match is not None]


def _normalized_comma_values(items: list[str] | None) -> set[str] | None:
    if items is None:
        return None
    return {re.sub(r"^[A-Za-z]+-", "", value).casefold() for value in items}


def _normalized_required_values(
    items: list[str] | None,
    specification: dict[str, Any],
) -> list[str] | None:
    if items is None:
        return None
    prefix = specification.get("strip_prefix", "")
    values: list[str] = []
    for item in items:
        value = re.sub(r"^\d+:", "", item).strip()
        if prefix and value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix) :]
        values.append(value.casefold())
    return values


def _parse_json(answer: str) -> Any | None:
    try:
        return json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_csv(answer: str) -> list[list[str]] | None:
    try:
        rows = list(csv.reader(io.StringIO(answer, newline=""), strict=True))
    except csv.Error:
        return None
    return rows if rows and all(row for row in rows) else None


def _parse_yaml_mapping(answer: str) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(answer)
    except yaml.YAMLError:
        return None
    return value if isinstance(value, dict) else None


def _score_exact_structure(
    actual: Any | None, expected: Any, format_name: str
) -> tuple[bool, float, dict[str, Any]]:
    if actual is None:
        return False, 0.0, {
            "content_checker": f"exact_{format_name}",
            "reason": f"invalid_{format_name}",
        }
    actual_leaves = _flatten_json(actual)
    expected_leaves = _flatten_json(expected)
    matching = sum(
        path in actual_leaves
        and _json_values_equal(actual_leaves[path], expected_value)
        for path, expected_value in expected_leaves.items()
    )
    denominator = max(len(actual_leaves), len(expected_leaves))
    return _json_values_equal(actual, expected), matching / denominator, {
        "content_checker": f"exact_{format_name}",
        "matching_leaves": matching,
        "expected_leaves": len(expected_leaves),
        "actual_leaves": len(actual_leaves),
    }


def _json_array_records(
    value: Any | None, *, allow_summary: bool
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    records = value[:-1] if allow_summary else value
    if not records or any(not isinstance(record, dict) for record in records):
        return None
    return records


def _json_records_are_sorted(
    records: list[dict[str, Any]] | None, specifications: list[dict[str, Any]]
) -> bool:
    if not records:
        return False

    def in_order(left: dict[str, Any], right: dict[str, Any]) -> bool:
        for specification in specifications:
            field = specification["field"]
            if field not in left or field not in right:
                return False
            left_value = left[field]
            right_value = right[field]
            if "order" in specification:
                positions = {
                    value: index for index, value in enumerate(specification["order"])
                }
                if left_value not in positions or right_value not in positions:
                    return False
                left_value = positions[left_value]
                right_value = positions[right_value]
            if left_value == right_value:
                continue
            if specification.get("direction", "ascending") == "descending":
                return left_value > right_value
            return left_value < right_value
        return True

    return all(in_order(left, right) for left, right in zip(records, records[1:]))


def _json_records_have_derived_bands(
    records: list[dict[str, Any]] | None, specification: dict[str, Any]
) -> bool:
    if not records:
        return False
    source_field = specification["source_field"]
    target_field = specification["target_field"]
    for record in records:
        source_value = record.get(source_field)
        if isinstance(source_value, bool) or not isinstance(source_value, (int, float)):
            return False
        expected = specification["otherwise"]
        for band in specification["bands"]:
            if source_value <= band["maximum"]:
                expected = band["value"]
                break
        if record.get(target_field) != expected:
            return False
    return True


def _json_summary_counts_match(
    value: Any | None, specification: dict[str, Any]
) -> bool:
    if not isinstance(value, list) or len(value) < 2 or not isinstance(value[-1], dict):
        return False
    summary_key = specification["summary_key"]
    summary = value[-1].get(summary_key)
    records = value[:-1]
    if not isinstance(summary, dict) or any(not isinstance(record, dict) for record in records):
        return False
    counts = Counter(record.get(specification["field"]) for record in records)
    return None not in counts and dict(sorted(counts.items())) == dict(sorted(summary.items()))


def _mapping_fields_satisfy(
    value: dict[str, Any],
    constraints: dict[str, dict[str, Any]],
) -> bool:
    for field, rules in constraints.items():
        if field not in value:
            return False
        field_value = value[field]
        expected_type = rules.get("type")
        if expected_type == "integer" and (
            isinstance(field_value, bool) or not isinstance(field_value, int)
        ):
            return False
        if expected_type == "string" and not isinstance(field_value, str):
            return False
        if "enum" in rules and field_value not in rules["enum"]:
            return False
        if "equals" in rules and field_value != rules["equals"]:
            return False
        if "minimum" in rules and (
            not isinstance(field_value, (int, float))
            or field_value < rules["minimum"]
        ):
            return False
        if "maximum" in rules and (
            not isinstance(field_value, (int, float))
            or field_value > rules["maximum"]
        ):
            return False
    return True


def _csv_column_is_sorted(
    rows: list[list[str]] | None,
    column: str,
    *,
    final_row: list[Any] | None,
    direction: str = "ascending",
) -> bool:
    if not rows or column not in rows[0]:
        return False
    data_rows = rows[1:]
    if final_row is not None and data_rows:
        data_rows = data_rows[:-1]
    index = rows[0].index(column)
    try:
        values = [int(row[index]) for row in data_rows]
    except (IndexError, ValueError):
        return False
    return values == sorted(values, reverse=direction == "descending")


def _csv_column_meets_minimum(
    rows: list[list[str]] | None, column: str, minimum: int
) -> bool:
    if not rows or column not in rows[0] or len(rows) < 2:
        return False
    index = rows[0].index(column)
    try:
        values = [int(row[index]) for row in rows[1:]]
    except (IndexError, ValueError):
        return False
    return all(value >= minimum for value in values)


def _csv_ties_are_sorted(
    rows: list[list[str]] | None, specification: dict[str, str]
) -> bool:
    if not rows:
        return False
    header = rows[0]
    if specification["primary"] not in header or specification["secondary"] not in header:
        return False
    primary_index = header.index(specification["primary"])
    secondary_index = header.index(specification["secondary"])
    for left, right in zip(rows[1:], rows[2:]):
        if left[primary_index] == right[primary_index] and (
            left[secondary_index].casefold() > right[secondary_index].casefold()
        ):
            return False
    return True


def _classification_parts(
    value: Any | None,
) -> tuple[list[str] | None, dict[str, Any] | None]:
    if not isinstance(value, list):
        return None, None
    summary = value[-1] if value and isinstance(value[-1], dict) else None
    labels = value[:-1] if summary is not None else value
    if any(not isinstance(label, str) for label in labels):
        return None, summary
    return labels, summary


def _is_label_array(value: Any | None, *, allow_summary: bool) -> bool:
    labels, summary = _classification_parts(value)
    if labels is None or not labels:
        return False
    if summary is None:
        return True
    return allow_summary and set(summary) == {"spam_count"}


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[\s_-]+", "-", label.strip().casefold())
    return normalized


def _ranked_comma_values(items: list[str] | None) -> list[str] | None:
    if items is None:
        return None
    return [re.sub(r"^\d+:", "", item).strip() for item in items]


def _contains_term(answer: str, term: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(term)}(?!\w)",
        answer,
        flags=re.IGNORECASE,
    ) is not None
