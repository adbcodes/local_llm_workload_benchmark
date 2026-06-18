from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_workload_benchmark.config import (
    BenchmarkConfig,
    GenerationConfig,
    ModelConfig,
)
from llm_workload_benchmark.dataset import DatasetItem, load_suite, score_answer
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


class ModelBackend(Protocol):
    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationOutput: ...


BackendFactory = Callable[[ModelConfig, Path, int], ModelBackend]


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


def run_benchmark(
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    backend_factory: BackendFactory = LlamaCppBackend,
    peak_memory_reader: Callable[[], int | None] | None = None,
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
    model = enabled_models[0]
    model_path = _resolve_from_root(root, model.model_path)

    run_directory = create_run(
        config,
        config_path,
        project_root=root,
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
                        peak_memory_reader=memory_reader,
                    )
                    records.append(record)
                    results_file.write(json.dumps(record, sort_keys=True) + "\n")
                    results_file.flush()

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


def _evaluate_item(
    item: DatasetItem,
    *,
    model: ModelConfig,
    backend: ModelBackend,
    repetition: int,
    seed: int,
    peak_memory_reader: Callable[[], int | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = backend.generate(item.prompt, model.generation, seed=seed)
        latency_seconds = time.perf_counter() - started
        peak_process_memory_bytes = peak_memory_reader()
        score = score_answer(item, output.text)
        output_tokens_per_second = (
            output.output_tokens / latency_seconds
            if output.output_tokens is not None and latency_seconds > 0
            else None
        )
        return {
            "schema_version": 1,
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
            "passed": score.passed,
            "score": score.score,
            "score_details": score.details,
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
            "schema_version": 1,
            "status": "error",
            "model_id": model.id,
            "benchmark": item.benchmark,
            "item_id": item.id,
            "subcategory": item.subcategory,
            "difficulty": item.difficulty,
            "split": item.split,
            "repetition": repetition,
            "seed": seed,
            "raw_response": None,
            "passed": None,
            "score": None,
            "score_details": None,
            "latency_seconds": time.perf_counter() - started,
            "time_to_first_token_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "output_characters": None,
            "output_tokens_per_second_end_to_end": None,
            "peak_process_memory_bytes": peak_memory_reader(),
            "finish_reason": None,
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
    scores = [record["score"] for record in completed]
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
    passed = sum(record["passed"] is True for record in completed)
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
        "generation": model.generation.model_dump(mode="json"),
    }


def _suite_hash(suite_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(suite_path.parent.rglob("*")):
        if path.is_file() and path.suffix in {".yaml", ".jsonl"}:
            relative_path = path.relative_to(suite_path.parent)
            digest.update(str(relative_path).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _resolve_from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
