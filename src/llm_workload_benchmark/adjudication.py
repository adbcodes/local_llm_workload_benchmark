from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from llm_workload_benchmark.config import BenchmarkConfig, JudgeConfig
from llm_workload_benchmark.dataset import DatasetItem, load_suite
from llm_workload_benchmark.judge import (
    JudgeBackend,
    JudgeError,
    SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
    SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
    create_judge_backend,
    evaluate_semantic_requirements,
    semantic_requirements_for_item,
)


class AdjudicationError(RuntimeError):
    """Raised when saved inference evidence cannot be adjudicated safely."""


JudgeBackendFactory = Callable[[JudgeConfig], JudgeBackend]
ProgressCallback = Callable[[int, int, str], None]


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
        if record.get("status") == "completed"
        and isinstance(record.get("adjudication_key"), str)
    }

    queue: list[tuple[dict[str, Any], DatasetItem]] = []
    for model in index.get("models", []):
        if not isinstance(model, dict) or model.get("status") != "completed":
            continue
        records_path = experiment / str(model["run_directory"]) / "results.jsonl"
        for record in _read_jsonl(records_path):
            item = items.get((str(record.get("benchmark")), str(record.get("item_id"))))
            answer = record.get("evaluated_response")
            if (
                item is None
                or record.get("status") != "completed"
                or not isinstance(answer, str)
                or not answer.strip()
                or not semantic_requirements_for_item(item)
            ):
                continue
            key = _adjudication_key(record, answer, contract_hash)
            if key not in completed_keys:
                queue.append(({**record, "adjudication_key": key}, item))

    backend = judge_backend_factory(config.judge)

    def judge_one(entry: tuple[dict[str, Any], DatasetItem]) -> dict[str, Any]:
        record, item = entry
        answer = str(record["evaluated_response"])
        try:
            judgment = evaluate_semantic_requirements(
                item,
                answer,
                backend=backend,
                config=config.judge,
                seed=int(record.get("seed", config.benchmark.seed)),
            )
            return _sidecar_record(record, item, judgment.model_dump(mode="json"))
        except JudgeError as error:
            return _sidecar_record(record, item, None, error=str(error))

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
        "schema_version": 1,
        "source_experiment": str(experiment),
        "suite": str(suite_path),
        "judge_contract_sha256": contract_hash,
        "judge": config.judge.model_dump(mode="json", exclude={"cache_path"}),
        "records": len(all_records),
        "completed": sum(record.get("status") == "completed" for record in all_records),
        "judge_errors": sum(record.get("status") == "judge_error" for record in all_records),
    }
    _write_json(output / "manifest.json", manifest)
    return output


def _sidecar_record(
    record: dict[str, Any],
    item: DatasetItem,
    judgment: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    derived = _derive_outcomes(record, judgment) if judgment is not None else None
    return {
        "schema_version": 1,
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
        "judge_evaluation": judgment,
        "derived": derived,
        "error": error,
    }


def _derive_outcomes(
    record: dict[str, Any], judgment: dict[str, Any]
) -> dict[str, Any]:
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
        semantic_correct = core_correct
        instruction_compliant = semantic_rules_correct and mechanical_rules_correct
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

    strict_pass = semantic_correct and instruction_compliant and format_correct
    loose_pass = semantic_correct
    return {
        "semantic_correct": semantic_correct,
        "instruction_compliant": instruction_compliant,
        "format_correct": format_correct,
        "strict_pass": strict_pass,
        "loose_pass": loose_pass,
        "format_tax": bool(loose_pass and not strict_pass),
        "requirement_pass": requirement_pass,
    }


def _adjudication_key(
    record: dict[str, Any], answer: str, contract_hash: str
) -> str:
    payload = {
        "model_id": record.get("model_id"),
        "benchmark": record.get("benchmark"),
        "item_id": record.get("item_id"),
        "repetition": record.get("repetition"),
        "response_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "judge_contract_sha256": contract_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _judge_contract_hash(config: JudgeConfig) -> str:
    payload = {
        "judge": config.model_dump(mode="json", exclude={"cache_path"}),
        "rubric_version": SEMANTIC_REQUIREMENTS_RUBRIC_VERSION,
        "system_prompt": SEMANTIC_REQUIREMENTS_SYSTEM_PROMPT,
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
