from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationType = Literal[
    "deterministic",
    "executable",
    "llm_judge",
    "human",
]


class EvaluationResult(BaseModel):
    """Common result envelope shared by every evaluator type."""

    model_config = ConfigDict(extra="forbid")

    type: EvaluationType
    evaluator: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(default=1, ge=1)
    passed: bool
    score: float = Field(ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)
