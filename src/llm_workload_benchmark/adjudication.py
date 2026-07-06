from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from llm_workload_benchmark.config import BenchmarkConfig, JudgeConfig
from llm_workload_benchmark.dataset import DatasetError, DatasetItem, load_suite, score_answer
from llm_workload_benchmark.judge import (
    BLIND_EXTRACTION_RUBRIC_VERSION,
    BLIND_EXTRACTION_SYSTEM_PROMPT,
    JudgeBackend,
    JudgeError,
    SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
    SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
    create_judge_backend,
    evaluate_semantic_requirements,
    extract_claimed_answer,
    semantic_requirements_for_item,
)


class AdjudicationError(RuntimeError):
    """Raised when saved inference evidence cannot be adjudicated safely."""


JudgeBackendFactory = Callable[[JudgeConfig], JudgeBackend]
ProgressCallback = Callable[[int, int, str], None]

RouteKind = Literal["semantic_requirements", "blind_extraction", "unresolved"]
_BLIND_EXTRACTION_METHODS = {
    "numeric_tolerance",
    "rational_value",
    "date_value",
    "exact_match",
    "json_exact",
    "set_match",
}
_UNKNOWN_PARSE_REASONS = {
    "invalid_json",
    "invalid_option_answer",
    "missing_or_ambiguous_date_answer",
    "missing_or_ambiguous_exact_answer",
    "missing_or_ambiguous_numeric_answer",
    "missing_or_ambiguous_rational_answer",
    "not_numeric",
}
_ROUTING_VERSION = 2


@dataclass(frozen=True)
class AdjudicationRoute:
    kind: RouteKind
    reason: str


def adjudicate_experiment(
    experiment: Path,
    config: BenchmarkConfig,
    *,
    project_root: Path | None = None,
    workers: int = 1,
    judge_backend_factory: JudgeBackendFactory = create_judge_backend,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Judge meaning-level requirements and append immutable sidecar records."""

    if config.judge is None or config.judge_panel is not None:
        raise AdjudicationError("adjudication requires exactly one configured judge")
    if workers < 1:
        raise AdjudicationError("adjudication workers must be at least 1")

    root = (project_root or Path.cwd()).resolve()
    experiment = experiment.resolve()
    index = _read_json(experiment / "experiment.json")
    suite_path = _resolve(root, config.benchmark.workload_path)
    suite = load_suite(suite_path)
    items = {
        (benchmark, item.id): item
        for benchmark, benchmark_items in suite.items.items()
        for item in benchmark_items
    }
    contract_hash = _judge_contract_hash(config.judge)
    output = experiment / "adjudications" / contract_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    existing = _read_jsonl(results_path) if results_path.exists() else []
    records_by_key = {
        record["adjudication_key"]: record
        for record in existing
        if isinstance(record.get("adjudication_key"), str)
    }
    completed_keys = {
        record["adjudication_key"]
        for record in existing
        if record.get("status") in {"completed", "unresolved"}
        and isinstance(record.get("adjudication_key"), str)
    }

    queue: list[tuple[dict[str, Any], DatasetItem, AdjudicationRoute]] = []
    for model in index.get("models", []):
        if not isinstance(model, dict) or model.get("status") != "completed":
            continue
        records_path = experiment / str(model["run_directory"]) / "results.jsonl"
        for record in _read_jsonl(records_path):
            item = items.get((str(record.get("benchmark")), str(record.get("item_id"))))
            answer = record.get("evaluated_response")
            if item is None or record.get("status") != "completed":
                continue
            if not isinstance(answer, str) or not answer.strip():
                continue
            route = safe_adjudication_route(
                record,
                item,
                answer,
                max_candidate_characters=config.judge.max_candidate_characters,
            )
            if route is None:
                continue
            key = _adjudication_key(record, item, route, answer, contract_hash)
            if key not in completed_keys:
                queue.append(({**record, "adjudication_key": key}, item, route))

    backend = (
        judge_backend_factory(config.judge)
        if any(route.kind != "unresolved" for _, _, route in queue)
        else None
    )

    def judge_one(
        entry: tuple[dict[str, Any], DatasetItem, AdjudicationRoute]
    ) -> dict[str, Any]:
        record, item, route = entry
        answer = str(record["evaluated_response"])
        if route.kind == "unresolved":
            return _unresolved_sidecar(record, item, route)
        assert backend is not None
        try:
            seed = int(record.get("seed", config.benchmark.seed))
            deterministic_rescore = None
            if route.kind == "semantic_requirements":
                judgment = evaluate_semantic_requirements(
                    item,
                    answer,
                    backend=backend,
                    config=config.judge,
                    seed=seed,
                )
            else:
                judgment = extract_claimed_answer(
                    item,
                    answer,
                    backend=backend,
                    config=config.judge,
                    seed=seed,
                )
                extracted_answer = judgment.details["extracted_answer"]
                if judgment.details["status"] == "extracted":
                    deterministic_rescore = score_answer(
                        item, str(extracted_answer)
                    ).model_dump(mode="json")
            return _sidecar_record(
                record,
                item,
                route,
                judgment.model_dump(mode="json"),
                deterministic_rescore=deterministic_rescore,
            )
        except (DatasetError, JudgeError) as error:
            return _sidecar_record(record, item, route, None, error=str(error))

    total = len(queue)
    if workers == 1:
        judged = map(judge_one, queue)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        judged = executor.map(judge_one, queue)
    try:
        for completed, sidecar in enumerate(judged, start=1):
            records_by_key[str(sidecar["adjudication_key"])] = sidecar
            _write_jsonl(results_path, list(records_by_key.values()))
            if progress_callback is not None:
                progress_callback(completed, total, str(sidecar["model_id"]))
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    all_records = _read_jsonl(results_path) if results_path.exists() else []
    manifest = {
        "schema_version": 2,
        "source_experiment": str(experiment),
        "suite": str(suite_path),
        "judge_contract_sha256": contract_hash,
        "judge": config.judge.model_dump(mode="json", exclude={"cache_path"}),
        "records": len(all_records),
        "completed": sum(record.get("status") == "completed" for record in all_records),
        "unresolved": sum(record.get("status") == "unresolved" for record in all_records),
        "judge_errors": sum(record.get("status") == "judge_error" for record in all_records),
        "routes": {
            route: sum(record.get("judge_route") == route for record in all_records)
            for route in ("semantic_requirements", "blind_extraction", "unresolved")
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output


def adjudication_route(
    record: dict[str, Any], item: DatasetItem
) -> AdjudicationRoute | None:
    """Select a judge route only when deterministic evidence is inconclusive."""

    evaluation = record.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    if (
        record.get("finish_reason") == "length"
        and evaluation.get("semantic_outcome") == "not_scored"
    ):
        return AdjudicationRoute(
            kind="unresolved",
            reason="output_truncated_without_complete_answer",
        )

    requirements = semantic_requirements_for_item(item)
    if requirements:
        details = evaluation.get("details")
        details = details if isinstance(details, dict) else {}
        if (
            item.scoring.method == "tool_call"
            and details.get("no_tool_expected") is True
            and isinstance(details.get("actual_tool"), str)
        ):
            return None
        return AdjudicationRoute(
            kind="semantic_requirements",
            reason="meaning_requires_semantic_evaluation",
        )

    method = item.scoring.method
    if method not in _BLIND_EXTRACTION_METHODS:
        return None
    if not evaluation or evaluation.get("passed") is True:
        return None

    semantic_outcome = evaluation.get("semantic_outcome")
    details = evaluation.get("details")
    details = details if isinstance(details, dict) else {}
    reason = details.get("reason")

    if semantic_outcome == "not_scored":
        return AdjudicationRoute(
            kind="blind_extraction",
            reason="deterministic_semantics_not_scored",
        )
    if isinstance(reason, str) and reason in _UNKNOWN_PARSE_REASONS:
        return AdjudicationRoute(
            kind="blind_extraction",
            reason=f"inconclusive_parser:{reason}",
        )
    if method == "exact_match":
        return AdjudicationRoute(
            kind="blind_extraction",
            reason="exact_string_mismatch_does_not_establish_claimed_meaning",
        )
    return None


def safe_adjudication_route(
    record: dict[str, Any],
    item: DatasetItem,
    answer: str,
    *,
    max_candidate_characters: int,
) -> AdjudicationRoute | None:
    """Apply non-LLM safety limits after ordinary eligibility routing."""

    route = adjudication_route(record, item)
    if (
        route is not None
        and route.kind != "unresolved"
        and len(answer) > max_candidate_characters
    ):
        return AdjudicationRoute(
            kind="unresolved",
            reason="candidate_exceeds_judge_character_limit",
        )
    return route


def _unresolved_sidecar(
    record: dict[str, Any],
    item: DatasetItem,
    route: AdjudicationRoute,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "unresolved",
        "adjudication_key": record["adjudication_key"],
        "model_id": record.get("model_id"),
        "benchmark": item.benchmark,
        "item_id": item.id,
        "repetition": record.get("repetition"),
        "response_sha256": hashlib.sha256(
            str(record["evaluated_response"]).encode("utf-8")
        ).hexdigest(),
        "deterministic_evaluation": record.get("evaluation"),
        "judge_route": route.kind,
        "route_reason": route.reason,
        "judge_evaluation": None,
        "deterministic_rescore": None,
        "derived": {
            "semantic_status": "unknown",
            "semantic_correct": None,
            "instruction_compliant": None,
            "format_correct": False,
            "strict_pass": False,
            "loose_pass": False,
            "format_tax": False,
            "unresolved_reason": route.reason,
        },
        "error": None,
    }


def _sidecar_record(
    record: dict[str, Any],
    item: DatasetItem,
    route: AdjudicationRoute,
    judgment: dict[str, Any] | None,
    *,
    deterministic_rescore: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    derived = (
        _derive_outcomes(
            record,
            judgment,
            route_kind=route.kind,
            deterministic_rescore=deterministic_rescore,
        )
        if judgment is not None
        else None
    )
    return {
        "schema_version": 2,
        "status": "completed" if judgment is not None else "judge_error",
        "adjudication_key": record["adjudication_key"],
        "model_id": record.get("model_id"),
        "benchmark": item.benchmark,
        "item_id": item.id,
        "repetition": record.get("repetition"),
        "response_sha256": hashlib.sha256(
            str(record["evaluated_response"]).encode("utf-8")
        ).hexdigest(),
        "deterministic_evaluation": record.get("evaluation"),
        "judge_route": route.kind,
        "route_reason": route.reason,
        "judge_evaluation": judgment,
        "deterministic_rescore": deterministic_rescore,
        "derived": derived,
        "error": error,
    }


def _derive_outcomes(
    record: dict[str, Any],
    judgment: dict[str, Any],
    *,
    route_kind: RouteKind | None = None,
    deterministic_rescore: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_kind = route_kind or "semantic_requirements"
    if route_kind == "blind_extraction":
        return _derive_extraction_outcomes(
            record, judgment, deterministic_rescore=deterministic_rescore
        )

    judge_details = judgment["details"]
    assessments = {
        assessment["id"]: assessment for assessment in judge_details["requirements"]
    }
    requirement_pass = {
        requirement_id: bool(value["satisfied"] and not value["contradicted"])
        for requirement_id, value in assessments.items()
    }
    deterministic = record.get("evaluation") or {}
    details = deterministic.get("details") or {}
    judge_ambiguous = judgment["details"].get("ambiguous") is True

    if record.get("scoring_method") == "constraint_rules":
        core_ids = [key for key in requirement_pass if key.startswith("core_fact:")]
        instruction_ids = [key for key in requirement_pass if not key.startswith("core_fact:")]
        core_correct = (
            all(requirement_pass[key] for key in core_ids) and not judge_ambiguous
        )
        semantic_rules_correct = (
            all(requirement_pass[key] for key in instruction_ids)
            and not judge_ambiguous
        )
        mechanical_rules_correct = bool(details.get("checks")) and all(
            details["checks"].values()
        )
        semantic_correct: bool | None = core_correct
        instruction_compliant: bool | None = (
            semantic_rules_correct and mechanical_rules_correct
        )
        format_correct = mechanical_rules_correct
    else:
        explanation_correct = (
            all(requirement_pass.values())
            and not judge_ambiguous
            and judgment["details"].get("overall_correct") is True
        )
        tool_correct = details.get("tool_choice_accuracy") == 1.0
        arguments_correct = details.get("argument_accuracy") == 1.0
        semantic_correct = explanation_correct and tool_correct and arguments_correct
        instruction_compliant = semantic_correct
        format_correct = details.get("format_compliant") is True

    if judge_ambiguous:
        semantic_correct = None
        semantic_status = "unknown"
        instruction_compliant = None
    else:
        semantic_status = "correct" if semantic_correct else "incorrect"
    strict_pass = (
        semantic_correct is True
        and instruction_compliant is True
        and format_correct
    )
    loose_pass = semantic_correct is True
    return {
        "semantic_status": semantic_status,
        "semantic_correct": semantic_correct,
        "instruction_compliant": instruction_compliant,
        "format_correct": format_correct,
        "strict_pass": strict_pass,
        "loose_pass": loose_pass,
        "format_tax": bool(loose_pass and not strict_pass),
        "requirement_pass": requirement_pass,
    }


def _derive_extraction_outcomes(
    record: dict[str, Any],
    judgment: dict[str, Any],
    *,
    deterministic_rescore: dict[str, Any] | None,
) -> dict[str, Any]:
    extraction_status = judgment["details"]["status"]
    if extraction_status == "extracted" and deterministic_rescore is not None:
        semantic_correct = deterministic_rescore.get("passed") is True
        semantic_status = "correct" if semantic_correct else "incorrect"
    elif extraction_status == "no_answer":
        semantic_correct = False
        semantic_status = "incorrect"
    else:
        semantic_correct = None
        semantic_status = "unknown"

    deterministic = record.get("evaluation")
    deterministic = deterministic if isinstance(deterministic, dict) else {}
    protocol_outcome = deterministic.get("protocol_outcome")
    details = deterministic.get("details")
    details = details if isinstance(details, dict) else {}
    format_correct = (
        protocol_outcome == "compliant"
        if protocol_outcome is not None
        else details.get("protocol_compliant") is not False
    )
    if extraction_status == "extracted" and record.get("scoring_method") not in {
        "json_exact",
        "set_match",
    }:
        original = str(record.get("evaluated_response") or "").strip()
        extracted = str(judgment["details"]["extracted_answer"]).strip()
        format_correct = format_correct and original == extracted
    loose_pass = semantic_correct is True
    strict_pass = loose_pass and format_correct
    return {
        "semantic_status": semantic_status,
        "semantic_correct": semantic_correct,
        "instruction_compliant": None,
        "format_correct": format_correct,
        "strict_pass": strict_pass,
        "loose_pass": loose_pass,
        "format_tax": bool(loose_pass and not strict_pass),
        "extraction_status": extraction_status,
    }


def _adjudication_key(
    record: dict[str, Any],
    item: DatasetItem,
    route: AdjudicationRoute,
    answer: str,
    contract_hash: str,
) -> str:
    item_contract = item.model_dump(mode="json")
    payload = {
        "model_id": record.get("model_id"),
        "benchmark": record.get("benchmark"),
        "item_id": record.get("item_id"),
        "repetition": record.get("repetition"),
        "route": route.kind,
        "response_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "judge_contract_sha256": contract_hash,
        "item_contract_sha256": hashlib.sha256(
            json.dumps(item_contract, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _judge_contract_hash(config: JudgeConfig) -> str:
    payload = {
        "judge": config.model_dump(mode="json", exclude={"cache_path"}),
        "routing_version": _ROUTING_VERSION,
        "rubrics": {
            "semantic_requirements": {
                "version": SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
                "system_prompt": SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
            },
            "blind_extraction": {
                "version": BLIND_EXTRACTION_RUBRIC_VERSION,
                "system_prompt": BLIND_EXTRACTION_SYSTEM_PROMPT,
            },
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdjudicationError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"cannot read {path}: {error}") from error
    if any(not isinstance(value, dict) for value in values):
        raise AdjudicationError(f"expected JSON objects in {path}")
    return values


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
