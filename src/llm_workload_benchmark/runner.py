from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from llm_workload_benchmark.config import (
    BenchmarkConfig,
    GenerationConfig,
    JudgeConfig,
    ModelConfig,
)
from llm_workload_benchmark.dataset import (
    BenchmarkSuite,
    DatasetItem,
    load_suite,
    score_answer,
)
from llm_workload_benchmark.executable import evaluate_python
from llm_workload_benchmark.judge import (
    GroqJudgeBackend,
    JudgeBackend,
    evaluate_summary,
)
from llm_workload_benchmark.manifest import create_run


class EvaluationError(RuntimeError):
    """Raised when a benchmark run cannot be started or completed."""


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    time_to_first_token_seconds: float | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class RunProgress:
    model_id: str
    model_number: int
    model_count: int
    benchmark: str
    completed_items: int
    total_items: int
    elapsed_seconds: float


class ModelBackend(Protocol):
    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationOutput: ...


BackendFactory = Callable[[ModelConfig, Path, int], ModelBackend]
JudgeBackendFactory = Callable[[JudgeConfig], JudgeBackend]


def _default_judge_backend_factory(config: JudgeConfig) -> JudgeBackend:
    return GroqJudgeBackend(config)


class LlamaCppBackend:
    """In-process GGUF inference through the optional llama-cpp-python package."""

    def __init__(self, model: ModelConfig, model_path: Path, seed: int) -> None:
        if not model_path.is_file():
            raise EvaluationError(f"model file does not exist: {model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise EvaluationError(
                "llama-cpp-python is required for inference; install the "
                "project with the llama-cpp extra"
            ) from error

        arguments: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": model.context_window,
            "n_gpu_layers": model.gpu_layers,
            "seed": seed,
            "verbose": model.verbose,
        }
        if model.threads is not None:
            arguments["n_threads"] = model.threads
        if model.chat_format is not None:
            arguments["chat_format"] = model.chat_format
        self._model = Llama(**arguments)
        self._system_prompt = model.system_prompt

    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationOutput:
        started = time.perf_counter()
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=generation.max_output_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            seed=seed,
            stream=True,
        )
        if isinstance(response, dict):
            raise EvaluationError("llama.cpp did not return a completion stream")

        content_parts: list[str] = []
        prompt_tokens: int | None = None
        time_to_first_token_seconds: float | None = None
        finish_reason: str | None = None
        saw_choice = False

        for chunk in response:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            saw_choice = True
            if prompt_tokens is None:
                prompt_tokens = _optional_int(getattr(self._model, "n_tokens", None))
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                if time_to_first_token_seconds is None:
                    time_to_first_token_seconds = time.perf_counter() - started
                content_parts.append(content)
            chunk_finish_reason = choice.get("finish_reason")
            if isinstance(chunk_finish_reason, str):
                finish_reason = chunk_finish_reason

        if not saw_choice:
            raise EvaluationError("llama.cpp returned no completion choices")

        text = "".join(content_parts)
        output_tokens = len(
            self._model.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            time_to_first_token_seconds=time_to_first_token_seconds,
            finish_reason=finish_reason,
        )

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()


def run_benchmark(
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    backend_factory: BackendFactory = LlamaCppBackend,
    judge_backend_factory: JudgeBackendFactory = _default_judge_backend_factory,
    peak_memory_reader: Callable[[], int | None] | None = None,
    run_directory: Path | None = None,
    progress_callback: Callable[[RunProgress], None] | None = None,
    model_number: int = 1,
    model_count: int = 1,
) -> Path:
    """Evaluate one enabled model and save per-item results plus a summary."""

    root = (project_root or Path.cwd()).resolve()
    memory_reader = peak_memory_reader or _process_peak_memory_bytes
    suite_path = _resolve_from_root(root, config.benchmark.workload_path)
    suite = load_suite(suite_path)
    enabled_models = [model for model in config.models if model.enabled]
    if len(enabled_models) != 1:
        raise EvaluationError(
            "the schema pilot requires exactly one enabled model; found "
            f"{len(enabled_models)}"
        )
    requires_judge = _suite_requires_judge(suite)
    if requires_judge and config.judge is None:
        raise EvaluationError(
            "the workload contains llm_judge items but no judge is configured"
        )
    judge_backend: JudgeBackend | None = None
    if requires_judge:
        try:
            judge_backend = judge_backend_factory(config.judge)
        except Exception as error:
            raise EvaluationError(f"could not initialize LLM judge: {error}") from error
    model = enabled_models[0]
    model_path = _resolve_from_root(root, model.model_path)

    run_directory = create_run(
        config,
        config_path,
        project_root=root,
        run_directory=run_directory,
    )
    results_path = run_directory / "results.jsonl"
    summary_path = run_directory / "summary.json"

    load_started = time.perf_counter()
    try:
        backend = backend_factory(model, model_path, config.benchmark.seed)
    except Exception as error:
        load_seconds = time.perf_counter() - load_started
        _write_json(
            summary_path,
            _failed_summary(
                model=model,
                model_path=model_path,
                suite_path=suite_path,
                load_seconds=load_seconds,
                peak_memory_after_model_load_bytes=memory_reader(),
                error=error,
            ),
        )
        raise EvaluationError(
            f"could not load model {model.id!r}: {error}; failure summary: "
            f"{summary_path}"
        ) from error
    load_seconds = time.perf_counter() - load_started
    peak_memory_after_model_load_bytes = memory_reader()

    records: list[dict[str, Any]] = []
    total_items = (
        sum(len(items) for items in suite.items.values())
        * config.benchmark.repetitions
    )
    run_started = time.perf_counter()
    try:
        with results_path.open("w", encoding="utf-8") as results_file:
            for repetition in range(1, config.benchmark.repetitions + 1):
                seed = config.benchmark.seed + repetition - 1
                for benchmark_items in suite.items.values():
                    for item in benchmark_items:
                        record = _evaluate_item(
                            item,
                            model=model,
                            backend=backend,
                            repetition=repetition,
                            seed=seed,
                            judge_config=config.judge,
                            judge_backend=judge_backend,
                            peak_memory_reader=memory_reader,
                        )
                        records.append(record)
                        results_file.write(json.dumps(record, sort_keys=True) + "\n")
                        results_file.flush()
                        if progress_callback is not None:
                            progress_callback(
                                RunProgress(
                                    model_id=model.id,
                                    model_number=model_number,
                                    model_count=model_count,
                                    benchmark=item.benchmark,
                                    completed_items=len(records),
                                    total_items=total_items,
                                    elapsed_seconds=time.perf_counter() - run_started,
                                )
                            )
    finally:
        _release_backend(backend)

    _write_json(
        summary_path,
        _build_summary(
            records,
            model=model,
            model_path=model_path,
            suite_path=suite_path,
            load_seconds=load_seconds,
            peak_memory_after_model_load_bytes=peak_memory_after_model_load_bytes,
        ),
    )
    return run_directory


def run_matrix(
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    backend_factory: BackendFactory = LlamaCppBackend,
    judge_backend_factory: JudgeBackendFactory = _default_judge_backend_factory,
    peak_memory_reader: Callable[[], int | None] | None = None,
    progress_callback: Callable[[RunProgress], None] | None = None,
) -> Path:
    """Run all enabled models sequentially under one experiment directory."""

    root = (project_root or Path.cwd()).resolve()
    enabled_models = [model for model in config.models if model.enabled]
    if not enabled_models:
        raise EvaluationError("the model matrix has no enabled models")

    suite = load_suite(_resolve_from_root(root, config.benchmark.workload_path))
    effective_judge_backend_factory = judge_backend_factory
    if _suite_requires_judge(suite):
        if config.judge is None:
            raise EvaluationError(
                "the workload contains llm_judge items but no judge is configured"
            )
        try:
            shared_judge_backend = judge_backend_factory(config.judge)
        except Exception as error:
            raise EvaluationError(f"could not initialize LLM judge: {error}") from error

        def use_shared_judge_backend(_: JudgeConfig) -> JudgeBackend:
            return shared_judge_backend

        effective_judge_backend_factory = use_shared_judge_backend
    output_root = _resolve_from_root(root, config.benchmark.output_root)
    experiment_id = _new_experiment_id()
    experiment_directory = output_root / experiment_id
    experiment_directory.mkdir(parents=True, exist_ok=False)
    index_path = experiment_directory / "experiment.json"
    started = time.perf_counter()
    model_results: list[dict[str, Any]] = []

    def write_index(status: str) -> None:
        completed = sum(result["status"] == "completed" for result in model_results)
        _write_json(
            index_path,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "status": status,
                "config_source": str(config_path.resolve()),
                "dataset": str(
                    _resolve_from_root(root, config.benchmark.workload_path)
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "models_total": len(enabled_models),
                "models_completed": completed,
                "models_failed": len(model_results) - completed,
                "models": model_results,
            },
        )

    write_index("running")
    for model_number, model in enumerate(enabled_models, start=1):
        child_directory = experiment_directory / "models" / model.id
        single_model_config = config.model_copy(
            update={"models": [model.model_copy(update={"enabled": True})]},
            deep=True,
        )
        try:
            run_directory = run_benchmark(
                single_model_config,
                config_path,
                project_root=root,
                backend_factory=backend_factory,
                judge_backend_factory=effective_judge_backend_factory,
                peak_memory_reader=peak_memory_reader,
                run_directory=child_directory,
                progress_callback=progress_callback,
                model_number=model_number,
                model_count=len(enabled_models),
            )
            model_results.append(
                {
                    "model_id": model.id,
                    "status": "completed",
                    "run_directory": str(
                        run_directory.relative_to(experiment_directory)
                    ),
                    "summary": str(
                        (run_directory / "summary.json").relative_to(
                            experiment_directory
                        )
                    ),
                    "error": None,
                }
            )
        except Exception as error:
            model_results.append(
                {
                    "model_id": model.id,
                    "status": "failed",
                    "run_directory": str(
                        child_directory.relative_to(experiment_directory)
                    ),
                    "summary": (
                        str(
                            (child_directory / "summary.json").relative_to(
                                experiment_directory
                            )
                        )
                        if (child_directory / "summary.json").is_file()
                        else None
                    ),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        write_index("running")

    completed = sum(result["status"] == "completed" for result in model_results)
    final_status = (
        "completed"
        if completed == len(enabled_models)
        else "partial_failure"
        if completed
        else "failed"
    )
    write_index(final_status)
    return experiment_directory


def _evaluate_item(
    item: DatasetItem,
    *,
    model: ModelConfig,
    backend: ModelBackend,
    repetition: int,
    seed: int,
    judge_config: JudgeConfig | None,
    judge_backend: JudgeBackend | None,
    peak_memory_reader: Callable[[], int | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    output: GenerationOutput | None = None
    latency_seconds: float | None = None
    peak_process_memory_bytes: int | None = None
    evaluated_response: str | None = None
    cleanup_applied: str | None = None
    try:
        output = backend.generate(item.prompt, model.generation, seed=seed)
        latency_seconds = time.perf_counter() - started
        peak_process_memory_bytes = peak_memory_reader()
        evaluated_response, cleanup_applied = _prepare_response_for_scoring(
            output.text,
            model.response_cleanup,
        )
        if item.scoring.method == "llm_judge":
            if judge_config is None or judge_backend is None:
                raise EvaluationError(
                    f"item {item.id!r} requires a configured LLM judge"
                )
            evaluation = evaluate_summary(
                item,
                evaluated_response,
                backend=judge_backend,
                config=judge_config,
                seed=seed,
            )
        elif item.scoring.method == "executable_python":
            evaluation = evaluate_python(item, evaluated_response)
        else:
            evaluation = score_answer(item, evaluated_response)
        output_tokens_per_second = (
            output.output_tokens / latency_seconds
            if output.output_tokens is not None and latency_seconds > 0
            else None
        )
        return {
            "schema_version": 2,
            "status": "completed",
            "model_id": model.id,
            "benchmark": item.benchmark,
            "item_id": item.id,
            "subcategory": item.subcategory,
            "difficulty": item.difficulty,
            "split": item.split,
            "repetition": repetition,
            "seed": seed,
            "raw_response": output.text,
            "evaluated_response": evaluated_response,
            "response_cleanup": cleanup_applied,
            "evaluation": evaluation.model_dump(mode="json"),
            "latency_seconds": latency_seconds,
            "time_to_first_token_seconds": output.time_to_first_token_seconds,
            "prompt_tokens": output.prompt_tokens,
            "output_tokens": output.output_tokens,
            "output_characters": len(output.text),
            "output_tokens_per_second_end_to_end": output_tokens_per_second,
            "peak_process_memory_bytes": peak_process_memory_bytes,
            "finish_reason": output.finish_reason,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": 2,
            "status": "error",
            "model_id": model.id,
            "benchmark": item.benchmark,
            "item_id": item.id,
            "subcategory": item.subcategory,
            "difficulty": item.difficulty,
            "split": item.split,
            "repetition": repetition,
            "seed": seed,
            "raw_response": output.text if output is not None else None,
            "evaluated_response": evaluated_response,
            "response_cleanup": cleanup_applied,
            "evaluation": None,
            "latency_seconds": (
                latency_seconds
                if latency_seconds is not None
                else time.perf_counter() - started
            ),
            "time_to_first_token_seconds": (
                output.time_to_first_token_seconds if output is not None else None
            ),
            "prompt_tokens": output.prompt_tokens if output is not None else None,
            "output_tokens": output.output_tokens if output is not None else None,
            "output_characters": len(output.text) if output is not None else None,
            "output_tokens_per_second_end_to_end": (
                output.output_tokens / latency_seconds
                if output is not None
                and output.output_tokens is not None
                and latency_seconds is not None
                and latency_seconds > 0
                else None
            ),
            "peak_process_memory_bytes": (
                peak_process_memory_bytes
                if peak_process_memory_bytes is not None
                else peak_memory_reader()
            ),
            "finish_reason": output.finish_reason if output is not None else None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def _build_summary(
    records: list[dict[str, Any]],
    *,
    model: ModelConfig,
    model_path: Path,
    suite_path: Path,
    load_seconds: float,
    peak_memory_after_model_load_bytes: int | None,
) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record["difficulty"])].append(record)

    benchmark_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    difficulty_groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in records:
        benchmark_records[record["benchmark"]].append(record)
    for (benchmark, difficulty), group_records in groups.items():
        difficulty_groups[benchmark][difficulty] = _aggregate(group_records)

    benchmark_groups = {
        benchmark: {
            "overall": _aggregate(group_records),
            "by_difficulty": difficulty_groups[benchmark],
        }
        for benchmark, group_records in benchmark_records.items()
    }

    return {
        "schema_version": 1,
        "status": "completed",
        "model": _model_summary(model, model_path),
        "dataset": {
            "path": str(suite_path),
            "sha256": _suite_hash(suite_path),
        },
        "model_load_seconds": load_seconds,
        "peak_process_memory_after_model_load_bytes": (
            peak_memory_after_model_load_bytes
        ),
        "totals": _aggregate(records),
        "total_prompt_tokens": sum(
            record["prompt_tokens"] or 0 for record in completed
        ),
        "total_output_tokens": sum(
            record["output_tokens"] or 0 for record in completed
        ),
        "judge": _aggregate_judge_usage(completed),
        "benchmarks": benchmark_groups,
    }


def _failed_summary(
    *,
    model: ModelConfig,
    model_path: Path,
    suite_path: Path,
    load_seconds: float,
    peak_memory_after_model_load_bytes: int | None,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "model_load_error",
        "model": _model_summary(model, model_path),
        "dataset": {
            "path": str(suite_path),
            "sha256": _suite_hash(suite_path),
        },
        "model_load_seconds": load_seconds,
        "peak_process_memory_after_model_load_bytes": (
            peak_memory_after_model_load_bytes
        ),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    scores = [record["evaluation"]["score"] for record in completed]
    latencies = [record["latency_seconds"] for record in completed]
    time_to_first_token_values = [
        record["time_to_first_token_seconds"]
        for record in completed
        if record["time_to_first_token_seconds"] is not None
    ]
    output_rates = [
        record["output_tokens_per_second_end_to_end"]
        for record in completed
        if record["output_tokens_per_second_end_to_end"] is not None
    ]
    peak_memory_values = [
        record["peak_process_memory_bytes"]
        for record in records
        if record["peak_process_memory_bytes"] is not None
    ]
    passed = sum(record["evaluation"]["passed"] is True for record in completed)
    return {
        "attempted": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "passed": passed,
        "pass_rate": passed / len(completed) if completed else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "latency_seconds": sum(record["latency_seconds"] for record in records),
        "mean_latency_seconds": _mean(latencies),
        "mean_time_to_first_token_seconds": _mean(time_to_first_token_values),
        "mean_output_tokens_per_second_end_to_end": _mean(output_rates),
        "peak_process_memory_bytes": max(peak_memory_values, default=None),
    }


def _aggregate_judge_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    judge_details = [
        record["evaluation"]["details"]["judge"]
        for record in records
        if record["evaluation"]["type"] == "llm_judge"
    ]
    if not judge_details:
        return None
    costs = [
        details["estimated_cost_usd"]
        for details in judge_details
        if details["estimated_cost_usd"] is not None
    ]
    return {
        "evaluations": len(judge_details),
        "prompt_tokens": sum(details["prompt_tokens"] or 0 for details in judge_details),
        "cached_prompt_tokens": sum(
            details["cached_prompt_tokens"] or 0 for details in judge_details
        ),
        "output_tokens": sum(details["output_tokens"] or 0 for details in judge_details),
        "reasoning_tokens": sum(
            details["reasoning_tokens"] or 0 for details in judge_details
        ),
        "latency_seconds": sum(
            details["latency_seconds"] for details in judge_details
        ),
        "estimated_cost_usd": sum(costs) if costs else None,
    }


def _suite_requires_judge(suite: BenchmarkSuite) -> bool:
    return any(
        item.scoring.method == "llm_judge"
        for benchmark_items in suite.items.values()
        for item in benchmark_items
    )


def _model_summary(model: ModelConfig, model_path: Path) -> dict[str, Any]:
    return {
        "id": model.id,
        "backend": model.backend,
        "path": str(model_path),
        "quantization": model.quantization,
        "context_window": model.context_window,
        "gpu_layers": model.gpu_layers,
        "threads": model.threads,
        "chat_format": model.chat_format,
        "response_cleanup": model.response_cleanup,
        "generation": model.generation.model_dump(mode="json"),
    }


def _suite_hash(suite_path: Path) -> str:
    digest = hashlib.sha256()
    suite_root = suite_path.parent
    manifest = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    active_paths = [suite_path]
    for relative_definition_path in manifest["benchmark_files"]:
        definition_path = suite_root / relative_definition_path
        active_paths.append(definition_path)
        definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
        active_paths.append(definition_path.parent / definition["items_path"])

    for path in sorted(active_paths, key=lambda value: str(value.relative_to(suite_root))):
        relative_path = path.relative_to(suite_root)
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _new_experiment_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-matrix-{uuid.uuid4().hex[:8]}"


def _release_backend(backend: ModelBackend) -> None:
    close = getattr(backend, "close", None)
    try:
        if callable(close):
            close()
    finally:
        del backend
        gc.collect()


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _prepare_response_for_scoring(
    response: str,
    cleanup: str,
) -> tuple[str, str | None]:
    if cleanup == "strip_empty_think":
        cleaned = re.sub(
            r"^\s*<think>\s*</think>\s*",
            "",
            response,
            count=1,
            flags=re.IGNORECASE,
        )
        if cleaned != response:
            return cleaned, "strip_empty_think"
    return response, None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _process_peak_memory_bytes() -> int | None:
    """Return this process's lifetime peak resident memory when available."""
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(peak_rss, int | float) or peak_rss < 0:
        return None
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(peak_rss * multiplier)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
