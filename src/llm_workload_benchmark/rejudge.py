from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from llm_workload_benchmark.config import (
    BenchmarkConfig,
    JudgeConfig,
    ModelConfig,
    load_config,
)
from llm_workload_benchmark.dataset import DatasetItem, load_suite
from llm_workload_benchmark.judge import (
    JudgeBackend,
    create_judge_backend,
    evaluate_summary,
)
from llm_workload_benchmark.evaluation import finalize_evaluation
from llm_workload_benchmark.runner import rebuild_run_summary


class RejudgeError(RuntimeError):
    """Raised when saved answers cannot be safely re-judged."""


JudgeBackendFactory = Callable[[JudgeConfig], JudgeBackend]
ProgressCallback = Callable[[str, int, int], None]


def rejudge_experiment(
    source_experiment: Path,
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    judge_backend_factory: JudgeBackendFactory = create_judge_backend,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Copy completed model runs and evaluate their saved answers with one judge."""

    root = (project_root or Path.cwd()).resolve()
    source = source_experiment.resolve()
    index_path = source / "experiment.json"
    try:
        source_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RejudgeError(f"cannot read source experiment {index_path}: {error}") from error

    if config.judge is None or config.judge_panel is not None:
        raise RejudgeError("rejudge requires exactly one configured judge")
    _validate_reusable_runs(source, source_index, config, config_path, root)

    suite_path = _resolve(root, config.benchmark.workload_path)
    suite = load_suite(suite_path)
    items = {
        (benchmark, item.id): item
        for benchmark, benchmark_items in suite.items.items()
        for item in benchmark_items
    }
    backend = judge_backend_factory(config.judge)

    output_root = _resolve(root, config.benchmark.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_id = _new_rejudged_experiment_id()
    target = output_root / experiment_id
    (target / "models").mkdir(parents=True)

    completed = [
        entry
        for entry in source_index.get("models", [])
        if isinstance(entry, dict) and entry.get("status") == "completed"
    ]
    total_judgments = sum(_judged_record_count(source, entry) for entry in completed)
    completed_judgments = 0
    copied_entries: list[dict[str, object]] = []
    models_by_id = {model.id: model for model in config.models if model.enabled}

    try:
        for entry in completed:
            model_id = str(entry["model_id"])
            if model_id not in models_by_id:
                raise RejudgeError(f"source model {model_id!r} is absent from target config")
            source_run = source / str(entry["run_directory"])
            target_run = target / "models" / model_id
            shutil.copytree(source_run, target_run)
            records_path = target_run / "results.jsonl"
            records = _read_jsonl(records_path)
            for record in records:
                if record.get("scoring_method") != "llm_judge":
                    continue
                item = _find_item(items, record)
                answer = record.get("evaluated_response")
                if not isinstance(answer, str):
                    raise RejudgeError(
                        f"saved answer for {record.get('item_id')!r} is not text"
                    )
                result = evaluate_summary(
                    item,
                    answer,
                    backend=backend,
                    config=config.judge,
                    seed=int(record["seed"]),
                )
                result = finalize_evaluation(
                    result,
                    primary_outcome=suite.definitions[
                        item.benchmark
                    ].evaluation_policy.primary_outcome,
                    scoring_method=item.scoring.method,
                    raw_response=answer,
                    finish_reason=record.get("finish_reason"),
                )
                record["evaluation"] = result.model_dump(mode="json")
                record["integration_outcome"] = result.integration_outcome
                record["schema_version"] = 3
                completed_judgments += 1
                if progress_callback is not None:
                    progress_callback(model_id, completed_judgments, total_judgments)
            _write_jsonl(records_path, records)
            summary_path = rebuild_run_summary(
                target_run,
                model=models_by_id[model_id],
                suite_path=suite_path,
                definitions=suite.definitions,
            )
            copied_entries.append(
                {
                    "model_id": model_id,
                    "status": "completed",
                    "run_directory": str(target_run.relative_to(target)),
                    "summary": str(summary_path.relative_to(target)),
                    "error": None,
                }
            )

        status = (
            "completed"
            if len(copied_entries) == len(models_by_id)
            else "interrupted"
        )
        target_index = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": status,
            "config_source": {
                "path": str(config_path.resolve()),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            },
            "dataset": str(suite_path),
            "elapsed_seconds": 0.0,
            "resume_count": 0,
            "models_total": len(models_by_id),
            "models_completed": len(copied_entries),
            "models_failed": 0,
            "models": copied_entries,
            "rejudged_from": str(source),
            "judge": {
                "provider": config.judge.provider,
                "model": config.judge.model,
            },
        }
        _write_json(target / "experiment.json", target_index)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def _validate_reusable_runs(
    source: Path,
    source_index: dict[str, object],
    target_config: BenchmarkConfig,
    target_config_path: Path,
    root: Path,
) -> None:
    source_info = source_index.get("config_source")
    if not isinstance(source_info, dict) or not isinstance(source_info.get("path"), str):
        raise RejudgeError("source experiment does not record its configuration path")
    source_path = Path(source_info["path"])
    if not source_path.is_file():
        raise RejudgeError(f"source configuration no longer exists: {source_path}")
    recorded_hash = source_info.get("sha256")
    current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if recorded_hash == current_hash:
        source_config = load_config(source_path)
        source_data = source_config.model_dump(
            mode="json", exclude={"judge", "judge_panel"}
        )
        target_data = target_config.model_dump(
            mode="json", exclude={"judge", "judge_panel"}
        )
        if source_data != target_data:
            raise RejudgeError(
                "source and target configs differ outside judge settings; refusing "
                f"to combine their results ({target_config_path})"
            )
        return

    expected_dataset = _resolve(root, target_config.benchmark.workload_path).resolve()
    saved_dataset = Path(str(source_index.get("dataset", ""))).resolve()
    if saved_dataset != expected_dataset:
        raise RejudgeError("target config uses a different dataset from the saved run")
    models = {model.id: model for model in target_config.models if model.enabled}
    for entry in source_index.get("models", []):
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            continue
        model_id = str(entry.get("model_id"))
        model = models.get(model_id)
        if model is None:
            raise RejudgeError(f"saved model {model_id!r} is absent from target config")
        summary_path = source / str(entry.get("summary"))
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RejudgeError(f"cannot validate saved summary {summary_path}") from error
        _validate_saved_model(summary.get("model"), model, root)


def _validate_saved_model(saved: object, model: ModelConfig, root: Path) -> None:
    if not isinstance(saved, dict):
        raise RejudgeError("saved run has no model metadata")
    expected = {
        "id": model.id,
        "backend": model.backend,
        "architecture": model.architecture,
        "family": model.family,
        "quantization": model.quantization,
        "context_window": model.context_window,
        "gpu_layers": model.gpu_layers,
        "threads": model.threads,
        "batch_size": model.batch_size,
        "flash_attention": model.flash_attention,
        "kv_cache_type": model.kv_cache_type,
        "chat_format": model.chat_format,
        "response_cleanup": model.response_cleanup,
        "generation": model.generation.model_dump(mode="json"),
        "path": str(_resolve(root, model.model_path).resolve()),
        "system_prompt_sha256": hashlib.sha256(
            model.system_prompt.encode("utf-8")
        ).hexdigest(),
    }
    mismatches = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatches:
        raise RejudgeError(
            f"saved model {model.id!r} differs from target config: "
            + ", ".join(mismatches)
        )


def _judged_record_count(source: Path, entry: dict[str, object]) -> int:
    path = source / str(entry["run_directory"]) / "results.jsonl"
    return sum(
        record.get("scoring_method") == "llm_judge" for record in _read_jsonl(path)
    )


def _find_item(
    items: dict[tuple[str, str], DatasetItem], record: dict[str, object]
) -> DatasetItem:
    key = (str(record.get("benchmark")), str(record.get("item_id")))
    try:
        return items[key]
    except KeyError as error:
        raise RejudgeError(f"saved result references unknown dataset item {key}") from error


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RejudgeError(f"cannot read saved results {path}: {error}") from error


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _new_rejudged_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-rejudged-{uuid4().hex[:8]}"
