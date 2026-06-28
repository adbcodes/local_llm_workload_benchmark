"""Temporary Phase 2 saved-response replay support; remove after Phase 3."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from llm_workload_benchmark.dataset import DatasetItem, load_suite, score_answer
from llm_workload_benchmark.executable import evaluate_python
from llm_workload_benchmark.runner import integration_outcome


class RegressionCorpusError(ValueError):
    """Raised when saved-response regression data is missing or inconsistent."""


class RegressionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    benchmark: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    item_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scoring_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SavedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    evaluated: str | None = None
    cleanup_applied: str | None = None
    finish_reason: str | None = None

    @model_validator(mode="after")
    def cleanup_has_an_evaluated_response(self) -> Self:
        if self.cleanup_applied is not None and self.evaluated is None:
            raise ValueError("cleanup_applied requires a distinct evaluated response")
        return self

    @property
    def legacy_input(self) -> str:
        """Return the saved scorer input; null means it was identical to raw."""
        return self.evaluated if self.evaluated is not None else self.raw


class LegacyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator: str = Field(min_length=1)
    passed: bool
    score: float = Field(ge=0, le=1)
    integration_outcome: str = Field(min_length=1)


class ExpectedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_outcome: Literal["correct", "incorrect", "not_scored"]
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    protocol_outcome: Literal["compliant", "noncompliant", "not_applicable"]
    integration_outcome: Literal[
        "scored_cleanly",
        "scored_after_recovery",
        "unparseable",
        "execution_failed",
        "evaluation_error",
    ]
    components: dict[str, bool] = Field(default_factory=dict)


class RegressionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    failure_class: Literal[
        "false_reject",
        "false_accept",
        "partial_answer",
        "protocol_wrapper",
        "malformed_output",
        "truncated_output",
        "genuine_model_error",
        "passing_control",
    ]
    source: RegressionSource
    response: SavedResponse
    legacy_result: LegacyResult
    expected_result: ExpectedResult
    expected_change: Literal[
        "flip_to_pass",
        "flip_to_fail",
        "remain_pass",
        "remain_fail",
        "diagnostics_only",
    ]
    rationale: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def change_matches_legacy_pass(self) -> Self:
        required_legacy_pass = {
            "flip_to_pass": False,
            "flip_to_fail": True,
            "remain_pass": True,
            "remain_fail": False,
        }
        expected = required_legacy_pass.get(self.expected_change)
        if expected is not None and self.legacy_result.passed is not expected:
            raise ValueError(
                f"{self.expected_change} requires legacy passed={expected}"
            )
        return self


@dataclass(frozen=True)
class ReplayCaseResult:
    case_id: str
    baseline_matches: bool
    actual_passed: bool
    actual_score: float
    actual_integration_outcome: str


@dataclass(frozen=True)
class ReplaySummary:
    total: int
    baseline_reproduced: int
    known_target_gaps: int
    unexpected_case_ids: tuple[str, ...]
    cases: tuple[ReplayCaseResult, ...]


def item_prompt_sha256(item: DatasetItem) -> str:
    value = (
        json.dumps(
            [message.model_dump(mode="json") for message in item.conversation],
            sort_keys=True,
        )
        if item.conversation is not None
        else item.prompt
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def item_scoring_contract_sha256(item: DatasetItem) -> str:
    value = {
        "response_contract": item.response_contract.model_dump(mode="json"),
        "expected": item.expected,
        "scoring": item.scoring.model_dump(mode="json"),
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_regression_corpus(path: Path) -> list[RegressionCase]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RegressionCorpusError(
            f"cannot read regression corpus {path}: {error}"
        ) from error

    cases: list[RegressionCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = RegressionCase.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as error:
            raise RegressionCorpusError(
                f"invalid regression case in {path} on line {line_number}: {error}"
            ) from error
        if case.id in seen_ids:
            raise RegressionCorpusError(f"duplicate regression case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise RegressionCorpusError(f"regression corpus contains no cases: {path}")
    return cases


def replay_regression_corpus(corpus_path: Path, suite_path: Path) -> ReplaySummary:
    cases = load_regression_corpus(corpus_path)
    suite = load_suite(suite_path)
    items = {
        (benchmark, item.id): item
        for benchmark, benchmark_items in suite.items.items()
        for item in benchmark_items
    }
    results: list[ReplayCaseResult] = []
    unexpected: list[str] = []

    for case in cases:
        key = (case.source.benchmark, case.source.item_id)
        item = items.get(key)
        if item is None:
            raise RegressionCorpusError(
                f"case {case.id!r} references missing item {key[1]!r} in {key[0]!r}"
            )
        _validate_item_snapshot(case, item)
        if item.scoring.method == "llm_judge":
            raise RegressionCorpusError(
                f"case {case.id!r} uses non-deterministic llm_judge scoring"
            )
        evaluation = (
            evaluate_python(item, case.response.legacy_input)
            if item.scoring.method == "executable_python"
            else score_answer(item, case.response.legacy_input)
        )
        actual_integration = integration_outcome(
            item,
            case.response.legacy_input,
            evaluation.details,
        )
        baseline_matches = (
            evaluation.evaluator == case.legacy_result.evaluator
            and evaluation.passed is case.legacy_result.passed
            and math.isclose(
                evaluation.score,
                case.legacy_result.score,
                rel_tol=0,
                abs_tol=1e-12,
            )
            and actual_integration == case.legacy_result.integration_outcome
        )
        if not baseline_matches:
            unexpected.append(case.id)
        results.append(
            ReplayCaseResult(
                case_id=case.id,
                baseline_matches=baseline_matches,
                actual_passed=evaluation.passed,
                actual_score=evaluation.score,
                actual_integration_outcome=actual_integration,
            )
        )

    known_target_gaps = sum(
        case.expected_change
        in {"flip_to_pass", "flip_to_fail", "diagnostics_only"}
        for case in cases
    )
    return ReplaySummary(
        total=len(cases),
        baseline_reproduced=len(cases) - len(unexpected),
        known_target_gaps=known_target_gaps,
        unexpected_case_ids=tuple(unexpected),
        cases=tuple(results),
    )


def _validate_item_snapshot(case: RegressionCase, item: DatasetItem) -> None:
    prompt_hash = item_prompt_sha256(item)
    if prompt_hash != case.source.prompt_sha256:
        raise RegressionCorpusError(
            f"case {case.id!r} prompt changed: expected "
            f"{case.source.prompt_sha256}, found {prompt_hash}"
        )
    contract_hash = item_scoring_contract_sha256(item)
    if contract_hash != case.source.scoring_contract_sha256:
        raise RegressionCorpusError(
            f"case {case.id!r} scoring contract changed: expected "
            f"{case.source.scoring_contract_sha256}, found {contract_hash}"
        )
