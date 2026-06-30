from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationType = Literal[
    "deterministic",
    "executable",
    "llm_judge",
    "human",
]
SemanticOutcome = Literal["correct", "incorrect", "not_scored"]
ProtocolOutcome = Literal["compliant", "noncompliant", "not_applicable"]
IntegrationOutcome = Literal[
    "scored_cleanly",
    "scored_after_recovery",
    "unparseable",
    "execution_failed",
    "evaluation_error",
]


class EvaluationResult(BaseModel):
    """Common result envelope shared by every evaluator type."""

    model_config = ConfigDict(extra="forbid")

    type: EvaluationType
    evaluator: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(default=1, ge=1)
    passed: bool
    score: float = Field(ge=0, le=1)
    semantic_outcome: SemanticOutcome | None = None
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    protocol_outcome: ProtocolOutcome | None = None
    protocol_score: float | None = Field(default=None, ge=0, le=1)
    protocol_violations: list[str] = Field(default_factory=list)
    integration_outcome: IntegrationOutcome | None = None
    integration_score: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


def finalize_evaluation(
    evaluation: EvaluationResult,
    *,
    primary_outcome: Literal["semantic", "protocol", "integration"],
    scoring_method: str,
    raw_response: str,
    finish_reason: str | None,
) -> EvaluationResult:
    """Populate independent outcomes and derive the policy headline verdict."""

    details = dict(evaluation.details)
    violations = list(details.get("protocol_violations") or [])
    wrapper = details.get("diagnostic_wrapper")
    if isinstance(wrapper, str) and wrapper and wrapper not in violations:
        violations.append(wrapper)
    if details.get("protocol_compliant") is False and not violations:
        violations.append("unparseable_output")
    complete_final_answer = (
        details.get("final_marker_compliant") is True
        and details.get("answer_parse_status") in {"parsed", "recovered"}
    )
    if (
        finish_reason == "length"
        and not complete_final_answer
        and "output_truncated" not in violations
    ):
        violations.append("output_truncated")
    if not raw_response.strip() and "missing_answer" not in violations:
        violations.append("missing_answer")

    protocol_outcome: ProtocolOutcome = (
        "noncompliant" if violations else "compliant"
    )
    protocol_score = float(protocol_outcome == "compliant")

    truncation_invalidates_answer = (
        finish_reason == "length"
        and not complete_final_answer
        and (evaluation.passed or "final_marker_compliant" in details)
    )
    parse_failed = (
        not raw_response.strip()
        or truncation_invalidates_answer
        or details.get("parseable") is False
        or details.get("answer_parse_status")
        in {"missing", "ambiguous", "unparseable", "truncated"}
        or details.get("reason")
        in {"invalid_json", "invalid_tool_call", "invalid_tool_trace"}
    )
    execution_failed = evaluation.type == "executable" and details.get("reason") in {
        "timeout",
        "resource_limit",
        "output_limit",
        "worker_failure",
        "invalid_worker_output",
        "candidate_load_error",
    }
    if parse_failed:
        integration_outcome: IntegrationOutcome = "unparseable"
        integration_score = 0.0
    elif execution_failed:
        integration_outcome = "execution_failed"
        integration_score = 0.0
    else:
        recovery_violations = [
            violation for violation in violations if violation != "output_truncated"
        ]
        integration_outcome = (
            "scored_after_recovery" if recovery_violations else "scored_cleanly"
        )
        integration_score = (
            evaluation.score if scoring_method == "tool_trace" else 1.0
        )

    if parse_failed:
        semantic_outcome: SemanticOutcome = "not_scored"
        semantic_score: float | None = None
    elif execution_failed:
        semantic_outcome = "incorrect"
        semantic_score = evaluation.score
    elif scoring_method == "json_exact":
        semantic_score = float(details.get("content_score", evaluation.score))
        semantic_outcome = (
            "correct" if details.get("content_exact") is True else "incorrect"
        )
    elif scoring_method == "tool_trace":
        semantic_outcome = (
            "correct" if details.get("integration_success") is True else "incorrect"
        )
        semantic_score = (
            0.0 if details.get("unnecessary_calls_ok") is False else None
        )
    elif scoring_method == "confidence_value":
        semantic_score = float(details.get("answer_correct") is True)
        semantic_outcome = "correct" if semantic_score == 1 else "incorrect"
    else:
        semantic_score = evaluation.score
        semantic_outcome = "correct" if evaluation.passed else "incorrect"

    if primary_outcome == "semantic":
        passed = semantic_outcome == "correct"
        score = semantic_score or 0.0
    elif primary_outcome == "protocol":
        passed = semantic_outcome == "correct" and protocol_outcome == "compliant"
        score = protocol_score
    else:
        passed = integration_score == 1.0
        score = integration_score

    details.update(
        {
            "primary_outcome": primary_outcome,
            "parse_failure": parse_failed,
            "recovered": integration_outcome == "scored_after_recovery",
        }
    )
    return evaluation.model_copy(
        update={
            "passed": passed,
            "score": score,
            "semantic_outcome": semantic_outcome,
            "semantic_score": semantic_score,
            "protocol_outcome": protocol_outcome,
            "protocol_score": protocol_score,
            "protocol_violations": list(dict.fromkeys(violations)),
            "integration_outcome": integration_outcome,
            "integration_score": integration_score,
            "details": details,
        }
    )
