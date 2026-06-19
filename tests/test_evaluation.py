import pytest
from pydantic import ValidationError

from llm_workload_benchmark.evaluation import EvaluationResult


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
