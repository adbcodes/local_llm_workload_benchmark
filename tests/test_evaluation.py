import pytest
from pydantic import ValidationError

from llm_workload_benchmark.evaluation import EvaluationResult, finalize_evaluation


@pytest.mark.parametrize(
    "evaluation_type",
    ["deterministic", "executable", "llm_judge", "human"],
)
def test_common_evaluation_result_supports_planned_evaluator_types(
    evaluation_type: str,
) -> None:
    result = EvaluationResult(
        type=evaluation_type,
        evaluator="example_evaluator",
        passed=True,
        score=0.75,
        details={"reason": "test fixture"},
    )

    assert result.version == 1
    assert result.model_dump(mode="json") == {
        "type": evaluation_type,
        "evaluator": "example_evaluator",
        "version": 1,
        "passed": True,
        "score": 0.75,
        "semantic_outcome": None,
        "semantic_score": None,
        "protocol_outcome": None,
        "protocol_score": None,
        "protocol_violations": [],
        "integration_outcome": None,
        "integration_score": None,
        "details": {"reason": "test fixture"},
    }


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"type": "unknown"},
        {"evaluator": "Bad evaluator name"},
        {"version": 0},
        {"score": 1.5},
        {"unexpected": True},
    ],
)
def test_common_evaluation_result_rejects_invalid_contracts(
    invalid_field: dict[str, object],
) -> None:
    value = {
        "type": "deterministic",
        "evaluator": "example_evaluator",
        "version": 1,
        "passed": True,
        "score": 1.0,
        "details": {},
    }
    value.update(invalid_field)

    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(value)


def test_semantic_policy_keeps_correct_recovered_content_as_a_pass() -> None:
    legacy = EvaluationResult(
        type="deterministic",
        evaluator="json_exact",
        passed=False,
        score=1.0,
        details={
            "content_exact": True,
            "content_score": 1.0,
            "protocol_compliant": False,
            "protocol_violations": ["markdown_fence"],
        },
    )

    result = finalize_evaluation(
        legacy,
        primary_outcome="semantic",
        scoring_method="json_exact",
        raw_response='```json\n{"answer": 7}\n```',
        finish_reason="stop",
    )

    assert result.passed
    assert result.semantic_outcome == "correct"
    assert result.protocol_outcome == "noncompliant"
    assert result.integration_outcome == "scored_after_recovery"


def test_protocol_policy_requires_correct_content_and_compliant_delivery() -> None:
    recovered = EvaluationResult(
        type="deterministic",
        evaluator="json_exact",
        passed=False,
        score=1.0,
        details={
            "content_exact": True,
            "content_score": 1.0,
            "protocol_compliant": False,
            "protocol_violations": ["markdown_fence"],
        },
    )

    result = finalize_evaluation(
        recovered,
        primary_outcome="protocol",
        scoring_method="json_exact",
        raw_response='```json\n{"answer": 7}\n```',
        finish_reason="stop",
    )

    assert not result.passed
    assert result.score == 0.0


def test_truncation_invalidates_a_gold_looking_unfinished_answer() -> None:
    legacy = EvaluationResult(
        type="deterministic",
        evaluator="exact_match",
        passed=True,
        score=1.0,
        details={"final_marker_compliant": False},
    )

    result = finalize_evaluation(
        legacy,
        primary_outcome="semantic",
        scoring_method="exact_match",
        raw_response="unfinished working with 98",
        finish_reason="length",
    )

    assert not result.passed
    assert result.semantic_outcome == "not_scored"
    assert result.protocol_violations == ["output_truncated"]
    assert result.integration_outcome == "unparseable"


def test_truncation_keeps_a_complete_parsed_final_answer() -> None:
    legacy = EvaluationResult(
        type="deterministic",
        evaluator="rational_value",
        passed=True,
        score=1.0,
        details={
            "answer_parse_status": "parsed",
            "final_marker_compliant": True,
        },
    )

    result = finalize_evaluation(
        legacy,
        primary_outcome="semantic",
        scoring_method="rational_value",
        raw_response="working\nFINAL: 27/128",
        finish_reason="length",
    )

    assert result.passed
    assert result.semantic_outcome == "correct"
    assert result.protocol_outcome == "compliant"
    assert result.integration_outcome == "scored_cleanly"


def test_integration_policy_uses_partial_tool_score_and_strict_success() -> None:
    partial = EvaluationResult(
        type="deterministic",
        evaluator="tool_trace",
        passed=False,
        score=0.5,
        details={
            "parseable": True,
            "integration_success": False,
            "unnecessary_calls_ok": True,
        },
    )

    result = finalize_evaluation(
        partial,
        primary_outcome="integration",
        scoring_method="tool_trace",
        raw_response='{"calls": []}',
        finish_reason="stop",
    )

    assert not result.passed
    assert result.score == 0.5
    assert result.integration_score == 0.5
    assert result.semantic_outcome == "incorrect"
