from __future__ import annotations

import hashlib
import json
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
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=generation.max_output_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            seed=seed,
        )
        choices = response.get("choices", [])
        if not choices:
            raise EvaluationError("llama.cpp returned no completion choices")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise EvaluationError("llama.cpp returned a completion without text")
        usage = response.get("usage", {})
        return GenerationOutput(
            text=content,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=choice.get("finish_reason"),
        )


def run_benchmark(
    config: BenchmarkConfig,
    config_path: Path,
    *,
    project_root: Path | None = None,
    backend_factory: BackendFactory = LlamaCppBackend,
) -> Path:
    """Evaluate one enabled model and save per-item results plus a summary."""

    root = (project_root or Path.cwd()).resolve()
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
                error=error,
            ),
        )
        raise EvaluationError(
            f"could not load model {model.id!r}: {error}; failure summary: "
            f"{summary_path}"
        ) from error
    load_seconds = time.perf_counter() - load_started

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
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = backend.generate(item.prompt, model.generation, seed=seed)
        latency_seconds = time.perf_counter() - started
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
            "prompt_tokens": output.prompt_tokens,
            "output_tokens": output.output_tokens,
            "output_tokens_per_second_end_to_end": output_tokens_per_second,
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
            "prompt_tokens": None,
            "output_tokens": None,
            "output_tokens_per_second_end_to_end": None,
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
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    scores = [record["score"] for record in completed]
    passed = sum(record["passed"] is True for record in completed)
    return {
        "attempted": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "passed": passed,
        "pass_rate": passed / len(completed) if completed else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "latency_seconds": sum(record["latency_seconds"] for record in records),
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
