from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_workload_benchmark.answer_parser import parse_answer
from llm_workload_benchmark.config import JudgeConfig
from llm_workload_benchmark.dataset import DatasetItem, score_answer
from llm_workload_benchmark.judge import (
    BLIND_EXTRACTION_RUBRIC_VERSION,
    BLIND_EXTRACTION_SYSTEM_PROMPT,
    SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
    SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
    BlindExtractionDecision,
    JudgeBackend,
    JudgeError,
    SemanticJudgeDecision,
    create_judge_backend,
    evaluate_semantic_requirements,
    extract_claimed_answer,
)


class JudgeEvaluationError(RuntimeError):
    """Raised when the standalone judge calibration cannot be completed."""


class RequirementGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    satisfied: bool
    contradicted: bool


class SemanticGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_correct: bool
    ambiguous: bool
    requirements: dict[str, RequirementGold]


class ExtractionGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["extracted", "no_answer", "ambiguous"]
    accepted_answers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answer_matches_status(self) -> "ExtractionGold":
        if self.status == "extracted" and not self.accepted_answers:
            raise ValueError("extracted gold requires at least one accepted answer")
        if self.status != "extracted" and self.accepted_answers:
            raise ValueError("unresolved gold cannot contain accepted answers")
        if any(not answer.strip() for answer in self.accepted_answers):
            raise ValueError("accepted answers must be non-empty strings")
        if len(self.accepted_answers) != len(set(self.accepted_answers)):
            raise ValueError("accepted answers must be unique")
        return self


class JudgeEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    route: Literal["semantic_requirements", "blind_extraction"]
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    candidate: str = Field(min_length=1)
    task: str | None = None
    requirements: dict[str, str] = Field(default_factory=dict)
    tool_explanation: bool = False
    response_type: Literal["number", "date", "text", "json"] = "text"
    response_format: str | None = None
    scoring_method: Literal[
        "numeric_tolerance",
        "rational_value",
        "date_value",
        "exact_match",
        "json_exact",
        "set_match",
    ] = "exact_match"
    gold_value: Any = None
    scoring_parameters: dict[str, Any] = Field(default_factory=dict)
    semantic_gold: SemanticGold | None = None
    extraction_gold: ExtractionGold | None = None

    @model_validator(mode="after")
    def route_fields_are_consistent(self) -> "JudgeEvaluationCase":
        if self.route == "semantic_requirements":
            if not self.task or not self.requirements:
                raise ValueError("semantic cases require task and requirements")
            if self.semantic_gold is None or self.extraction_gold is not None:
                raise ValueError("semantic cases require only semantic_gold")
            if set(self.requirements) != set(self.semantic_gold.requirements):
                raise ValueError("semantic gold must cover exactly the declared requirements")
        elif self.extraction_gold is None or self.semantic_gold is not None:
            raise ValueError("extraction cases require only extraction_gold")
        return self


class JudgeEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str = Field(min_length=1)
    cases: list[JudgeEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> "JudgeEvaluationDataset":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("judge evaluation case ids must be unique")
        return self


class JudgeEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    dataset_path: Path
    output_root: Path = Path("runs/judge-evaluation")
    seed: int = 42
    judge: JudgeConfig


ProgressCallback = Callable[[int, int, str], None]


def load_judge_evaluation_config(path: Path) -> JudgeEvaluationConfig:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return JudgeEvaluationConfig.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise JudgeEvaluationError(f"invalid judge evaluation config {path}: {error}") from error


def load_judge_evaluation_dataset(path: Path) -> JudgeEvaluationDataset:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return JudgeEvaluationDataset.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise JudgeEvaluationError(f"invalid judge evaluation dataset {path}: {error}") from error


def run_judge_evaluation(
    config: JudgeEvaluationConfig,
    *,
    project_root: Path | None = None,
    output_directory: Path | None = None,
    backend_factory: Callable[[JudgeConfig], JudgeBackend] = create_judge_backend,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run the human-labelled fixture separately from benchmark inference."""

    root = (project_root or Path.cwd()).resolve()
    dataset_path = _resolve(root, config.dataset_path)
    dataset = load_judge_evaluation_dataset(dataset_path)
    output = output_directory or _new_output_directory(
        _resolve(root, config.output_root), dataset.name
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    prior = _read_jsonl(results_path)
    records = {str(record["case_id"]): record for record in prior}
    backend = backend_factory(config.judge)

    for position, case in enumerate(dataset.cases, start=1):
        existing = records.get(case.id)
        if existing is None or existing.get("status") != "completed":
            records[case.id] = _evaluate_case(case, config, backend)
            _write_jsonl(
                results_path,
                [records[item.id] for item in dataset.cases if item.id in records],
            )
        if progress_callback is not None:
            progress_callback(position, len(dataset.cases), case.id)

    ordered = [records[case.id] for case in dataset.cases]
    summary = summarize_judge_evaluation(ordered)
    _write_json(output / "summary.json", summary)
    _write_case_csv(output / "cases.csv", ordered)
    _write_metric_csv(output / "metrics.csv", summary)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "judge": config.judge.model_dump(mode="json", exclude={"cache_path"}),
        "judge_contract_sha256": _judge_contract_hash(config.judge),
        "case_count": len(dataset.cases),
        "completed": summary["overall"]["completed"],
        "api_failures": summary["overall"]["api_failures"],
        "artifacts": {
            "results": "results.jsonl",
            "summary": "summary.json",
            "cases": "cases.csv",
            "metrics": "metrics.csv",
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output


def _evaluate_case(
    case: JudgeEvaluationCase,
    config: JudgeEvaluationConfig,
    backend: JudgeBackend,
) -> dict[str, Any]:
    item = _case_item(case)
    scenario = _case_scenario(case, item)
    try:
        if case.route == "semantic_requirements":
            evaluation = evaluate_semantic_requirements(
                item,
                case.candidate,
                backend=backend,
                config=config.judge,
                seed=config.seed,
            )
            comparison = _compare_semantic(case, evaluation.model_dump(mode="json"))
        else:
            evaluation = extract_claimed_answer(
                item,
                case.candidate,
                backend=backend,
                config=config.judge,
                seed=config.seed,
            )
            comparison = _compare_extraction(
                case, item, evaluation.model_dump(mode="json")
            )
        judge_metadata = evaluation.details["judge"]
        return {
            "schema_version": 1,
            "status": "completed",
            "case_id": case.id,
            "route": case.route,
            "category": case.category,
            "scenario": scenario,
            "candidate": case.candidate,
            "human_gold": _gold_dump(case),
            "judge_evaluation": evaluation.model_dump(mode="json"),
            "comparison": comparison,
            "telemetry": judge_metadata,
            "error": None,
        }
    except JudgeError as error:
        return {
            "schema_version": 1,
            "status": "judge_error",
            "case_id": case.id,
            "route": case.route,
            "category": case.category,
            "scenario": scenario,
            "candidate": case.candidate,
            "human_gold": _gold_dump(case),
            "judge_evaluation": None,
            "comparison": None,
            "telemetry": None,
            "error": str(error),
        }


def _case_item(case: JudgeEvaluationCase) -> DatasetItem:
    common = {
        "id": case.id,
        "benchmark": "judge_evaluation",
        "subcategory": case.category,
        "difficulty": "medium",
        "split": "dev",
        "visibility": "public",
        "prompt": case.task or "Return the requested answer.",
        "provenance": {"kind": "hand_authored", "review_status": "human_checked"},
        "tags": ["judge_evaluation", case.route, case.category],
    }
    if case.route == "semantic_requirements":
        requirement_items = list(case.requirements.items())
        core_id, core_description = requirement_items[0]
        if case.tool_explanation:
            if set(case.requirements) != {"tool_result_explanation"}:
                raise JudgeEvaluationError(
                    f"tool case {case.id} must declare tool_result_explanation"
                )
            return DatasetItem.model_validate(
                {
                    **common,
                    "conversation": [{"role": "user", "content": case.task}],
                    "response_contract": {"type": "json", "format": "next_tool_call_or_answer"},
                    "expected": {
                        "value": {
                            "tool_call": None,
                            "arguments": {},
                            "answer": core_description,
                        }
                    },
                    "scoring": {
                        "method": "tool_call",
                        "parameters": {"direct_answer_patterns": ["(?s).+"]},
                    },
                }
            )
        if not core_id.startswith("core_fact:"):
            raise JudgeEvaluationError(
                f"semantic case {case.id} must declare a core_fact requirement first"
            )
        semantic_requirements = [
            {"id": key, "description": description}
            for key, description in requirement_items[1:]
        ]
        return DatasetItem.model_validate(
            {
                **common,
                "response_contract": {"type": "text", "format": "prose"},
                "expected": {"value": core_description},
                "scoring": {
                    "method": "constraint_rules",
                    "parameters": {
                        "content_requirements": {
                            "required_facts": [
                                {
                                    "name": core_id.removeprefix("core_fact:"),
                                    "any_of": [core_description],
                                }
                            ]
                        },
                        "rules": {"max_words": 10000},
                        "semantic_requirements": semantic_requirements,
                    },
                },
            }
        )
    return DatasetItem.model_validate(
        {
            **common,
            "response_contract": {
                "type": case.response_type,
                "format": case.response_format,
            },
            "expected": {"value": case.gold_value},
            "scoring": {
                "method": case.scoring_method,
                "parameters": case.scoring_parameters,
            },
        }
    )


def _compare_semantic(
    case: JudgeEvaluationCase, evaluation: dict[str, Any]
) -> dict[str, Any]:
    assert case.semantic_gold is not None
    details = evaluation["details"]
    decision = SemanticJudgeDecision.model_validate(
        {
            "requirements": details["requirements"],
            "ambiguous": details["ambiguous"],
            "overall_correct": details["overall_correct"],
            "overall_reason": details["overall_reason"],
        }
    )
    predicted_requirements = {requirement.id: requirement for requirement in decision.requirements}
    requirement_agreement = {
        requirement_id: (
            predicted_requirements[requirement_id].satisfied == gold.satisfied
            and predicted_requirements[requirement_id].contradicted == gold.contradicted
        )
        for requirement_id, gold in case.semantic_gold.requirements.items()
    }
    exact = (
        all(requirement_agreement.values())
        and decision.ambiguous == case.semantic_gold.ambiguous
        and decision.overall_correct == case.semantic_gold.overall_correct
    )
    return {
        "gold_positive": case.semantic_gold.overall_correct,
        "predicted_positive": bool(evaluation["passed"]),
        "exact_agreement": exact,
        "status_agreement": decision.ambiguous == case.semantic_gold.ambiguous,
        "answer_agreement": None,
        "requirement_agreement": requirement_agreement,
    }


def _compare_extraction(
    case: JudgeEvaluationCase,
    item: DatasetItem,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    assert case.extraction_gold is not None
    details = evaluation["details"]
    decision = BlindExtractionDecision.model_validate(
        {
            "status": details["status"],
            "extracted_answer": details["extracted_answer"],
            "reason": details["reason"],
        }
    )
    gold_positive = case.extraction_gold.status == "extracted"
    answer_agreement = None
    predicted_positive = False
    if decision.status == "extracted":
        answer_agreement = _extraction_answer_agrees(
            case, item, decision.extracted_answer
        )
        predicted_positive = answer_agreement
    exact = decision.status == case.extraction_gold.status
    if gold_positive:
        exact = exact and bool(answer_agreement)
    return {
        "gold_positive": gold_positive,
        "predicted_positive": predicted_positive,
        "exact_agreement": exact,
        "status_agreement": decision.status == case.extraction_gold.status,
        "answer_agreement": answer_agreement,
        "requirement_agreement": None,
    }


def _extraction_answer_agrees(
    case: JudgeEvaluationCase,
    item: DatasetItem,
    predicted_answer: str,
) -> bool:
    assert case.extraction_gold is not None
    accepted_answers = case.extraction_gold.accepted_answers
    if not accepted_answers:
        return False
    return any(
        _answer_matches_accepted_gold(case, item, predicted_answer, expected_answer)
        for expected_answer in accepted_answers
    )


def _answer_matches_accepted_gold(
    case: JudgeEvaluationCase,
    item: DatasetItem,
    predicted_answer: str,
    expected_answer: str,
) -> bool:
    expected_value: Any
    if case.scoring_method == "numeric_tolerance":
        parsed = parse_answer(
            expected_answer,
            "number",
            answer_unit=case.scoring_parameters.get("answer_unit"),
            unit_aliases=case.scoring_parameters.get("unit_aliases", []),
        )
        if not parsed.parsed:
            raise JudgeEvaluationError(
                f"invalid numeric extraction gold for {case.id}"
            )
        expected_value = parsed.value
    elif case.scoring_method == "json_exact":
        try:
            expected_value = json.loads(expected_answer)
        except json.JSONDecodeError as error:
            raise JudgeEvaluationError(
                f"invalid JSON extraction gold for {case.id}"
            ) from error
    elif case.scoring_method == "set_match":
        separator = str(case.scoring_parameters.get("separator", ","))
        expected_value = [
            value.strip() for value in expected_answer.split(separator) if value.strip()
        ]
    else:
        expected_value = expected_answer
    comparison_item = item.model_copy(update={"expected": {"value": expected_value}})
    return score_answer(comparison_item, predicted_answer).passed


def summarize_judge_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    groups[("overall", "all")].extend(records)
    for record in records:
        groups[("route", str(record["route"]))].append(record)
        groups[("category", str(record["category"]))].append(record)
        groups[("scenario", str(record["scenario"]))].append(record)
    metrics = {
        scope: {name: _group_metrics(group) for (kind, name), group in groups.items() if kind == scope}
        for scope in ("route", "category", "scenario")
    }
    return {
        "schema_version": 1,
        "overall": _group_metrics(groups[("overall", "all")]),
        **metrics,
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    comparisons = [record["comparison"] for record in completed]
    tp = sum(item["gold_positive"] and item["predicted_positive"] for item in comparisons)
    tn = sum(not item["gold_positive"] and not item["predicted_positive"] for item in comparisons)
    fp = sum(not item["gold_positive"] and item["predicted_positive"] for item in comparisons)
    fn = sum(item["gold_positive"] and not item["predicted_positive"] for item in comparisons)
    exact = sum(item["exact_agreement"] for item in comparisons)
    telemetry = [record["telemetry"] for record in completed]
    token = lambda key: sum(value.get(key) or 0 for value in telemetry)
    cost = sum(value.get("estimated_cost_usd") or 0.0 for value in telemetry)
    return {
        "total": len(records),
        "completed": len(completed),
        "api_failures": len(records) - len(completed),
        "api_failure_rate": _ratio(len(records) - len(completed), len(records)),
        "exact_agreement": _ratio(exact, len(completed)),
        "accuracy": _ratio(tp + tn, len(completed)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "latency_seconds": sum(value.get("latency_seconds") or 0.0 for value in telemetry),
        "prompt_tokens": token("prompt_tokens"),
        "cached_prompt_tokens": token("cached_prompt_tokens"),
        "output_tokens": token("output_tokens"),
        "reasoning_tokens": token("reasoning_tokens"),
        "estimated_cost_usd": cost,
    }


def _write_case_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "route", "category", "scenario", "status", "gold_positive",
        "predicted_positive", "exact_agreement", "latency_seconds",
        "prompt_tokens", "output_tokens", "reasoning_tokens", "estimated_cost_usd", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for record in records:
            comparison = record.get("comparison") or {}
            telemetry = record.get("telemetry") or {}
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "route": record["route"],
                    "category": record["category"],
                    "scenario": record["scenario"],
                    "status": record["status"],
                    "gold_positive": comparison.get("gold_positive"),
                    "predicted_positive": comparison.get("predicted_positive"),
                    "exact_agreement": comparison.get("exact_agreement"),
                    "latency_seconds": telemetry.get("latency_seconds"),
                    "prompt_tokens": telemetry.get("prompt_tokens"),
                    "output_tokens": telemetry.get("output_tokens"),
                    "reasoning_tokens": telemetry.get("reasoning_tokens"),
                    "estimated_cost_usd": telemetry.get("estimated_cost_usd"),
                    "error": record.get("error"),
                }
            )


def _write_metric_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "scope", "name", "total", "completed", "api_failures", "exact_agreement",
        "accuracy", "precision", "recall", "specificity", "f1", "latency_seconds",
        "prompt_tokens", "output_tokens", "reasoning_tokens", "estimated_cost_usd",
    ]
    rows = [("overall", "all", summary["overall"])]
    rows.extend(("route", name, value) for name, value in summary["route"].items())
    rows.extend(("category", name, value) for name, value in summary["category"].items())
    rows.extend(("scenario", name, value) for name, value in summary["scenario"].items())
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for scope, name, value in rows:
            writer.writerow({"scope": scope, "name": name, **{key: value[key] for key in fields[2:]}})


def _judge_contract_hash(config: JudgeConfig) -> str:
    payload = {
        "judge": config.model_dump(mode="json", exclude={"cache_path"}),
        "semantic": {
            "version": SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
            "prompt": SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
            "schema": SemanticJudgeDecision.model_json_schema(),
        },
        "extraction": {
            "version": BLIND_EXTRACTION_RUBRIC_VERSION,
            "prompt": BLIND_EXTRACTION_SYSTEM_PROMPT,
            "schema": BlindExtractionDecision.model_json_schema(),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _gold_dump(case: JudgeEvaluationCase) -> dict[str, Any]:
    gold = case.semantic_gold or case.extraction_gold
    assert gold is not None
    return gold.model_dump(mode="json")


def _case_scenario(case: JudgeEvaluationCase, item: DatasetItem) -> str:
    if case.semantic_gold is not None:
        if case.semantic_gold.overall_correct:
            return "semantic_pass"
        if case.semantic_gold.ambiguous:
            return "ambiguous"
        if any(
            requirement.contradicted
            for requirement in case.semantic_gold.requirements.values()
        ):
            return "contradiction"
        return "partial_or_missing"
    assert case.extraction_gold is not None
    if case.extraction_gold.status != "extracted":
        return case.extraction_gold.status
    return (
        "correct_misformatted"
        if any(
            score_answer(item, answer).passed
            for answer in case.extraction_gold.accepted_answers
        )
        else "wrong_answer"
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _new_output_directory(root: Path, name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    return root / f"{timestamp}-{name}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
