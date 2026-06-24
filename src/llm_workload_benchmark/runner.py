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
    CachedJudgeBackend,
    GroqJudgeBackend,
    JudgeBackend,
    evaluate_summary,
    evaluate_summary_panel,
)
from llm_workload_benchmark.manifest import create_run
from llm_workload_benchmark.telemetry import RuntimeTelemetry


class EvaluationError(RuntimeError):
    """Raised when a benchmark run cannot be started or completed."""


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    time_to_first_token_seconds: float | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None


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
    backend: JudgeBackend = GroqJudgeBackend(config)
    return CachedJudgeBackend(backend, config) if config.cache_path else backend


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
            "n_batch": model.batch_size,
            "flash_attn": model.flash_attention,
        }
        if model.kv_cache_type is not None:
            arguments["type_k"] = model.kv_cache_type
            arguments["type_v"] = model.kv_cache_type
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
        return self._generate_chat(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            generation,
            seed=seed,
        )

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationOutput:
        effective_messages = list(messages)
        if not effective_messages or effective_messages[0]["role"] != "system":
            effective_messages.insert(
                0, {"role": "system", "content": self._system_prompt}
            )
        return self._generate_chat(effective_messages, generation, seed=seed)

    def _generate_chat(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationOutput:
        started = time.perf_counter()
        arguments: dict[str, Any] = {
            "messages": messages,
            "max_tokens": generation.max_output_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation.top_k,
            "repeat_penalty": generation.repeat_penalty,
            "seed": seed,
            "stream": True,
        }
        if generation.constrained_decoding == "json":
            arguments["response_format"] = {"type": "json_object"}
        response = self._model.create_chat_completion(**arguments)
        return self._consume_stream(response, started)

    def _consume_stream(self, response: Any, started: float) -> GenerationOutput:
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
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str) and content:
                if time_to_first_token_seconds is None:
                    time_to_first_token_seconds = time.perf_counter() - started
                content_parts.append(content)
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
        if not saw_choice:
            raise EvaluationError("llama.cpp returned no completion choices")
        text = "".join(content_parts)
        output_tokens = len(
            self._model.tokenize(text.encode("utf-8"), add_bos=False, special=True)
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
        if config.judge_panel is None:
            raise EvaluationError(
                "the workload contains llm_judge items but no judge is configured; "
                "configure either one judge or a three-judge panel"
            )
    judge_backend: JudgeBackend | None = None
    judge_panel_backends: list[JudgeBackend] | None = None
    if requires_judge and config.judge_panel is not None:
        try:
            judge_panel_backends = [
                judge_backend_factory(judge_config)
                for judge_config in config.judge_panel.judges
            ]
        except Exception as error:
            raise EvaluationError(f"could not initialize judge panel: {error}") from error
    elif requires_judge:
        try:
            judge_backend = judge_backend_factory(config.judge)
        except Exception as error:
            raise EvaluationError(f"could not initialize LLM judge: {error}") from error
    model = enabled_models[0]
    if config.judge_panel is not None and model.family is not None:
        judge_families = {
            judge.family.casefold() for judge in config.judge_panel.judges
        }
        if model.family.casefold() in judge_families:
            raise EvaluationError(
                f"candidate family {model.family!r} cannot judge its own family"
            )
    model_path = _resolve_from_root(root, model.model_path)

    run_directory = create_run(
        config,
        config_path,
        project_root=root,
        run_directory=run_directory,
    )
    telemetry: RuntimeTelemetry | None = None
    if peak_memory_reader is None:
        telemetry = RuntimeTelemetry(run_directory / "telemetry.jsonl")
        telemetry.start()
        memory_reader = telemetry.peak_rss_bytes
    else:
        memory_reader = peak_memory_reader
    results_path = run_directory / "results.jsonl"
    summary_path = run_directory / "summary.json"

    load_started = time.perf_counter()
    try:
        backend = backend_factory(model, model_path, config.benchmark.seed)
    except Exception as error:
        load_seconds = time.perf_counter() - load_started
        telemetry_summary = telemetry.stop() if telemetry is not None else None
        _write_json(
            summary_path,
            _failed_summary(
                model=model,
                model_path=model_path,
                suite_path=suite_path,
                load_seconds=load_seconds,
                peak_memory_after_model_load_bytes=memory_reader(),
                telemetry=telemetry_summary,
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
    telemetry_summary: dict[str, Any] | None = None
    try:
        with results_path.open("w", encoding="utf-8") as results_file:
            for repetition in range(1, config.benchmark.repetitions + 1):
                seed = config.benchmark.seed + repetition - 1
                for benchmark_id, benchmark_items in suite.items.items():
                    suite_id = suite.definitions[benchmark_id].suite
                    for item in benchmark_items:
                        source_record = next(
                            (
                                record
                                for record in reversed(records)
                                if record["repetition"] == repetition
                                and record["item_id"] == item.source_item
                            ),
                            None,
                        )
                        record = _evaluate_item(
                            item,
                            suite_id=suite_id,
                            source_response=(
                                source_record.get("evaluated_response")
                                if source_record is not None
                                else None
                            ),
                            model=model,
                            backend=backend,
                            repetition=repetition,
                            seed=seed,
                            judge_config=config.judge,
                            judge_backend=judge_backend,
                            judge_panel_configs=(
                                config.judge_panel.judges
                                if config.judge_panel is not None
                                else None
                            ),
                            judge_panel_backends=judge_panel_backends,
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
        try:
            _release_backend(backend)
        finally:
            if telemetry is not None:
                telemetry_summary = telemetry.stop()

    _write_json(
        summary_path,
        _build_summary(
            records,
            model=model,
            model_path=model_path,
            suite_path=suite_path,
            definitions=suite.definitions,
            load_seconds=load_seconds,
            peak_memory_after_model_load_bytes=peak_memory_after_model_load_bytes,
            telemetry=telemetry_summary,
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
        if config.judge is None and config.judge_panel is None:
            raise EvaluationError(
                "the workload contains llm_judge items but no judge is configured; "
                "configure either one judge or a three-judge panel"
            )
        if config.judge_panel is None:
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
    suite_id: str | None,
    source_response: str | None,
    model: ModelConfig,
    backend: ModelBackend,
    repetition: int,
    seed: int,
    judge_config: JudgeConfig | None,
    judge_backend: JudgeBackend | None,
    judge_panel_configs: list[JudgeConfig] | None,
    judge_panel_backends: list[JudgeBackend] | None,
    peak_memory_reader: Callable[[], int | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    output: GenerationOutput | None = None
    latency_seconds: float | None = None
    peak_process_memory_bytes: int | None = None
    evaluated_response: str | None = None
    cleanup_applied: str | None = None
    try:
        generation = _generation_for_item(model.generation, item)
        if item.scoring.method == "tool_trace":
            output = _run_tool_scenario(
                item,
                backend=backend,
                generation=generation,
                seed=seed,
                response_cleanup=model.response_cleanup,
            )
        elif item.conversation is not None:
            generate_messages = getattr(backend, "generate_messages", None)
            if not callable(generate_messages):
                raise EvaluationError(
                    f"item {item.id!r} requires a backend with multi-turn support"
                )
            messages = [message.model_dump(mode="json") for message in item.conversation]
            if source_response is None and any(
                "{{source_response}}" in message["content"] for message in messages
            ):
                raise EvaluationError(
                    f"item {item.id!r} requires its source_item response first"
                )
            if source_response is not None:
                messages = [
                    {
                        **message,
                        "content": message["content"].replace(
                            "{{source_response}}", source_response
                        ),
                    }
                    for message in messages
                ]
            output = generate_messages(
                messages,
                generation,
                seed=seed,
            )
        else:
            output = backend.generate(item.prompt, generation, seed=seed)
        latency_seconds = time.perf_counter() - started
        peak_process_memory_bytes = peak_memory_reader()
        evaluated_response, cleanup_applied = _prepare_response_for_scoring(
            output.text,
            model.response_cleanup,
        )
        if item.scoring.method == "llm_judge":
            if judge_panel_configs is not None and judge_panel_backends is not None:
                evaluation = evaluate_summary_panel(
                    item,
                    evaluated_response,
                    backends=judge_panel_backends,
                    configs=judge_panel_configs,
                    seed=seed,
                )
            elif judge_config is None or judge_backend is None:
                raise EvaluationError(
                    f"item {item.id!r} requires a configured LLM judge"
                )
            else:
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
        integration_outcome = _integration_outcome(
            item, evaluated_response, evaluation.details
        )
        output_tokens_per_second = (
            output.output_tokens / latency_seconds
            if output.output_tokens is not None and latency_seconds > 0
            else None
        )
        wall_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        return {
            "schema_version": 2,
            "status": "completed",
            "model_id": model.id,
            "benchmark": item.benchmark,
            "suite": suite_id,
            "item_id": item.id,
            "source_item": item.source_item,
            "subcategory": item.subcategory,
            "difficulty": item.difficulty,
            "split": item.split,
            "visibility": item.visibility,
            "dataset_origin": _dataset_origin(item),
            "repetition": repetition,
            "seed": seed,
            "raw_response": output.text,
            "evaluated_response": evaluated_response,
            "response_cleanup": cleanup_applied,
            "prompt_sha256": _item_prompt_hash(item),
            "system_prompt_sha256": hashlib.sha256(
                model.system_prompt.encode("utf-8")
            ).hexdigest(),
            "evaluation": evaluation.model_dump(mode="json"),
            "integration_outcome": integration_outcome,
            "latency_seconds": latency_seconds,
            "time_to_first_token_seconds": output.time_to_first_token_seconds,
            "prompt_tokens": output.prompt_tokens,
            "output_tokens": output.output_tokens,
            "reasoning_tokens": output.reasoning_tokens,
            "output_characters": len(output.text),
            "output_tokens_per_second_end_to_end": output_tokens_per_second,
            "peak_process_memory_bytes": peak_process_memory_bytes,
            "process_wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_utilization_percent": (
                cpu_seconds / wall_seconds * 100 if wall_seconds > 0 else None
            ),
            "finish_reason": output.finish_reason,
            "error": None,
        }
    except Exception as error:
        wall_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        return {
            "schema_version": 2,
            "status": "error",
            "model_id": model.id,
            "benchmark": item.benchmark,
            "suite": suite_id,
            "item_id": item.id,
            "source_item": item.source_item,
            "subcategory": item.subcategory,
            "difficulty": item.difficulty,
            "split": item.split,
            "visibility": item.visibility,
            "dataset_origin": _dataset_origin(item),
            "repetition": repetition,
            "seed": seed,
            "raw_response": output.text if output is not None else None,
            "evaluated_response": evaluated_response,
            "response_cleanup": cleanup_applied,
            "prompt_sha256": _item_prompt_hash(item),
            "system_prompt_sha256": hashlib.sha256(
                model.system_prompt.encode("utf-8")
            ).hexdigest(),
            "evaluation": None,
            "integration_outcome": "evaluation_error",
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
            "reasoning_tokens": output.reasoning_tokens if output is not None else None,
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
            "process_wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_utilization_percent": (
                cpu_seconds / wall_seconds * 100 if wall_seconds > 0 else None
            ),
            "finish_reason": output.finish_reason if output is not None else None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def _run_tool_scenario(
    item: DatasetItem,
    *,
    backend: ModelBackend,
    generation: GenerationConfig,
    seed: int,
    response_cleanup: str = "none",
) -> GenerationOutput:
    generate_messages = getattr(backend, "generate_messages", None)
    if not callable(generate_messages):
        raise EvaluationError(
            f"tool item {item.id!r} requires a backend with message support"
        )
    expected = item.expected["value"]
    expected_calls = expected["calls"]
    fixture_observations = expected.get("observations", [])
    messages = [
        {
            "role": "user",
            "content": (
                item.prompt
                + "\n\nTool protocol: return raw JSON only. To call a tool, return "
                '{"tool":"name","arguments":{...}}. After receiving tool results, '
                'continue or finish with {"final_state":{...}}.'
            ),
        }
    ]
    calls: list[dict[str, Any]] = []
    observations: list[Any] = []
    outputs: list[GenerationOutput] = []
    final_state: Any | None = None
    for turn in range(max(2, len(expected_calls) + 2)):
        output = generate_messages(messages, generation, seed=seed + turn)
        outputs.append(output)
        action_text, _ = _prepare_response_for_scoring(
            output.text,
            response_cleanup,
        )
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            return _combine_generation_outputs(outputs, output.text)
        if not isinstance(action, dict):
            return _combine_generation_outputs(outputs, output.text)
        if isinstance(action.get("tool"), str) and isinstance(
            action.get("arguments"), dict
        ):
            call = {"tool": action["tool"], "arguments": action["arguments"]}
            calls.append(call)
            index = len(calls) - 1
            if index < len(expected_calls) and _json_like_equal(
                call, expected_calls[index]
            ):
                observation = (
                    fixture_observations[index]
                    if index < len(fixture_observations)
                    else {"ok": True}
                )
            else:
                observation = {"error": "unexpected_tool_call"}
            observations.append(observation)
            messages.extend(
                [
                    {"role": "assistant", "content": action_text},
                    {
                        "role": "user",
                        "content": "Tool result: " + json.dumps(observation),
                    },
                ]
            )
            continue
        if "final_state" in action:
            final_state = action["final_state"]
            break
        return _combine_generation_outputs(outputs, output.text)
    trace: dict[str, Any] = {"calls": calls, "observations": observations}
    if final_state is not None:
        trace["final_state"] = final_state
    return _combine_generation_outputs(
        outputs, json.dumps(trace, separators=(",", ":"))
    )


def _combine_generation_outputs(
    outputs: list[GenerationOutput], text: str
) -> GenerationOutput:
    return GenerationOutput(
        text=text,
        prompt_tokens=sum(output.prompt_tokens or 0 for output in outputs) or None,
        output_tokens=sum(output.output_tokens or 0 for output in outputs) or None,
        time_to_first_token_seconds=(
            outputs[0].time_to_first_token_seconds if outputs else None
        ),
        finish_reason=outputs[-1].finish_reason if outputs else None,
        reasoning_tokens=sum(output.reasoning_tokens or 0 for output in outputs) or None,
    )


def _json_like_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _build_summary(
    records: list[dict[str, Any]],
    *,
    model: ModelConfig,
    model_path: Path,
    suite_path: Path,
    definitions: dict[str, Any],
    load_seconds: float,
    peak_memory_after_model_load_bytes: int | None,
    telemetry: dict[str, Any] | None,
) -> dict[str, Any]:
    _attach_paired_metrics(records)
    completed = [record for record in records if record["status"] == "completed"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record["difficulty"])].append(record)

    benchmark_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    difficulty_groups: dict[str, dict[str, Any]] = defaultdict(dict)
    benchmark_origin_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    origin_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    visibility_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    suite_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        benchmark_records[record["benchmark"]].append(record)
        origin = record["dataset_origin"]
        origin_records[origin].append(record)
        visibility_records[record.get("visibility", "public")].append(record)
        benchmark_origin_groups[record["benchmark"]][origin].append(record)
        if record.get("suite") is not None:
            suite_records[record["suite"]].append(record)
    for (benchmark, difficulty), group_records in groups.items():
        difficulty_groups[benchmark][difficulty] = _aggregate(group_records)

    benchmark_groups = {
        benchmark: {
            "overall": (
                overall := _aggregate(group_records)
            ),
            "reported_score": _reported_benchmark_score(
                definitions[benchmark].score_formula,
                overall,
                group_records,
            ),
            "score_formula": definitions[benchmark].score_formula,
            "by_difficulty": difficulty_groups[benchmark],
            "by_origin": {
                origin: _aggregate(origin_group)
                for origin, origin_group in benchmark_origin_groups[benchmark].items()
            },
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
        "telemetry": telemetry,
        "totals": _aggregate(records),
        "suites": {
            suite_id: _aggregate(suite_group)
            for suite_id, suite_group in sorted(suite_records.items())
        },
        "headline_scores": _headline_scores(benchmark_groups, definitions),
        "by_origin": {
            origin: _aggregate(origin_group)
            for origin, origin_group in origin_records.items()
        },
        "by_visibility": {
            visibility: _aggregate(group)
            for visibility, group in visibility_records.items()
        },
        "total_prompt_tokens": sum(
            record["prompt_tokens"] or 0 for record in completed
        ),
        "total_output_tokens": sum(
            record["output_tokens"] or 0 for record in completed
        ),
        "total_reasoning_tokens": sum(
            record["reasoning_tokens"] or 0 for record in completed
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
    telemetry: dict[str, Any] | None,
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
        "telemetry": telemetry,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    scored = [record for record in completed if record.get("evaluation") is not None]
    scores = [record["evaluation"]["score"] for record in scored]
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
    passed = sum(record["evaluation"]["passed"] is True for record in scored)
    confidence_interval = _wilson_interval(passed, len(scored))
    integration_outcomes: dict[str, int] = defaultdict(int)
    for record in completed:
        integration_outcomes[record.get("integration_outcome", "scored")] += 1
    confidence_records = [
        record
        for record in scored
        if isinstance(record["evaluation"]["details"].get("confidence"), int)
    ]
    return {
        "attempted": len(records),
        "completed": len(completed),
        "scored": len(scored),
        "errors": len(records) - len(completed),
        "integration_failures": sum(
            record.get("integration_outcome", "scored") != "scored"
            for record in completed
        ),
        "integration_friction_rate": (
            sum(
                record.get("integration_outcome", "scored") != "scored"
                for record in completed
            )
            / len(completed)
            if completed
            else None
        ),
        "integration_outcomes": dict(sorted(integration_outcomes.items())),
        "passed": passed,
        "pass_rate": passed / len(scored) if scored else None,
        "pass_rate_ci_95": confidence_interval,
        "brier_score": _mean(
            [
                record["evaluation"]["details"]["brier_component"]
                for record in confidence_records
            ]
        ),
        "expected_calibration_error": _expected_calibration_error(
            confidence_records
        ),
        "run_to_run_flip_rate": _run_to_run_flip_rate(scored),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "latency_seconds": sum(record["latency_seconds"] for record in records),
        "mean_latency_seconds": _mean(latencies),
        "mean_time_to_first_token_seconds": _mean(time_to_first_token_values),
        "mean_output_tokens_per_second_end_to_end": _mean(output_rates),
        "mean_process_wall_seconds": _mean(
            [record.get("process_wall_seconds") for record in records]
        ),
        "mean_process_cpu_seconds": _mean(
            [record["process_cpu_seconds"] for record in records]
        ),
        "mean_process_cpu_utilization_percent": _mean(
            [record.get("process_cpu_utilization_percent") for record in records]
        ),
        "peak_process_memory_bytes": max(peak_memory_values, default=None),
    }


def _reported_benchmark_score(
    formula: str,
    aggregate: dict[str, Any],
    records: list[dict[str, Any]],
) -> float | None:
    if formula == "mean_score":
        return aggregate["mean_score"]
    if formula == "clean_score_retained":
        values = [
            record["evaluation"]["details"]["retained_score"]
            for record in records
            if record.get("evaluation") is not None
            and isinstance(record["evaluation"]["details"].get("retained_score"), int | float)
        ]
        return _mean(values)
    if formula == "accuracy_minus_hallucination":
        if aggregate["pass_rate"] is None:
            return None
        risky_labels = {
            "abstain", "unanswerable", "fabricated_entity",
            "correct_false_premise", "flag_conflict",
        }
        behavior_records = [
            record
            for record in records
            if record.get("evaluation") is not None
            and record["evaluation"]["details"].get("behavior_label") in risky_labels
        ]
        hallucinations = sum(
            not record["evaluation"]["passed"] for record in behavior_records
        )
        penalty = hallucinations / len(behavior_records) if behavior_records else 0.0
        return aggregate["pass_rate"] - penalty
    raise EvaluationError(f"unsupported benchmark score formula: {formula}")


def _integration_outcome(
    item: DatasetItem,
    answer: str,
    details: dict[str, Any],
) -> str:
    if not answer.strip():
        return "missing_answer"
    if item.benchmark == "applied_reasoning":
        if details.get("reason") == "multiple_final_answers":
            return "ambiguous_final_marker"
        if details.get("final_marker_compliant") is False:
            return "missing_final_marker"
    if item.response_contract.type == "json":
        if details.get("protocol_compliant") is False:
            wrapper = details.get("diagnostic_wrapper")
            if wrapper == "markdown_fence":
                return "markdown_fence"
            if wrapper == "surrounding_text":
                return "surrounding_text"
            return "unparseable_output"
    return "scored"


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * (
        (proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5
    ) / denominator
    return {"low": max(0.0, centre - margin), "high": min(1.0, centre + margin)}


def _expected_calibration_error(records: list[dict[str, Any]], bins: int = 10) -> float | None:
    if not records:
        return None
    total = len(records)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            record
            for record in records
            if lower <= record["evaluation"]["details"]["confidence_probability"]
            <= (upper if bin_index == bins - 1 else upper - 1e-12)
        ]
        if not selected:
            continue
        mean_confidence = _mean(
            [
                record["evaluation"]["details"]["confidence_probability"]
                for record in selected
            ]
        )
        accuracy = _mean(
            [float(record["evaluation"]["details"]["answer_correct"]) for record in selected]
        )
        error += len(selected) / total * abs(mean_confidence - accuracy)
    return error


def _run_to_run_flip_rate(records: list[dict[str, Any]]) -> float | None:
    responses: dict[str, list[str]] = defaultdict(list)
    for record in records:
        responses[record["item_id"]].append(record.get("evaluated_response") or "")
    repeated = [values for values in responses.values() if len(values) > 1]
    if not repeated:
        return None
    return sum(len(set(values)) > 1 for values in repeated) / len(repeated)


def _headline_scores(
    benchmark_groups: dict[str, dict[str, Any]],
    definitions: dict[str, Any],
) -> dict[str, Any]:
    capability = [
        group["reported_score"]
        for benchmark, group in benchmark_groups.items()
        if definitions[benchmark].suite in {"A", "B"}
        and group["reported_score"] is not None
    ]
    control = [
        group["reported_score"]
        for benchmark, group in benchmark_groups.items()
        if definitions[benchmark].suite in {"C", "D"}
        and group["reported_score"] is not None
    ]
    trust_values = [
        group["reported_score"]
        for benchmark, group in benchmark_groups.items()
        if definitions[benchmark].suite == "E"
        and group["reported_score"] is not None
    ]
    return {
        "capability": _mean(capability),
        "control": _mean(control),
        "trust_clean_score_retained": (
            sum(trust_values) / len(trust_values) if trust_values else None
        ),
        "note": "Trust is a retained-score delta and is never averaged with absolute scores.",
    }


def _attach_paired_metrics(records: list[dict[str, Any]]) -> None:
    by_key = {
        (record["item_id"], record["repetition"]): record
        for record in records
    }
    for record in records:
        source_item = record.get("source_item")
        if source_item is None or record.get("evaluation") is None:
            continue
        source = by_key.get((source_item, record["repetition"]))
        if source is None or source.get("evaluation") is None:
            continue
        clean_score = source["evaluation"]["score"]
        changed_score = record["evaluation"]["score"]
        retained_score = (
            min(1.0, changed_score / clean_score) if clean_score > 0 else None
        )
        record["evaluation"]["details"].update(
            {
                "source_item": source_item,
                "clean_score": clean_score,
                "changed_score": changed_score,
                "retained_score": retained_score,
                "transition": _correctness_transition(
                    bool(source["evaluation"]["passed"]),
                    bool(record["evaluation"]["passed"]),
                ),
            }
        )


def _correctness_transition(source_passed: bool, changed_passed: bool) -> str:
    if source_passed and changed_passed:
        return "stood_by_correct"
    if source_passed and not changed_passed:
        return "flipped_correct_to_wrong"
    if not source_passed and changed_passed:
        return "flipped_wrong_to_correct"
    return "remained_wrong"


def _aggregate_judge_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    judge_details: list[dict[str, Any]] = []
    for record in records:
        if record["evaluation"]["type"] != "llm_judge":
            continue
        details = record["evaluation"]["details"]
        if isinstance(details.get("judge"), dict):
            judge_details.append(details["judge"])
        for verdict in details.get("verdicts", []):
            nested = verdict.get("details", {}).get("judge")
            if isinstance(nested, dict):
                judge_details.append(nested)
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


def _dataset_origin(item: DatasetItem) -> str:
    if item.provenance.source is not None:
        return "licensed_anchor"
    if item.provenance.kind == "synthetic":
        return "fresh_generated"
    return "hand_authored"


def _model_summary(model: ModelConfig, model_path: Path) -> dict[str, Any]:
    return {
        "id": model.id,
        "backend": model.backend,
        "path": str(model_path),
        "quantization": model.quantization,
        "architecture": model.architecture,
        "family": model.family,
        "role": model.role,
        "file_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "context_window": model.context_window,
        "gpu_layers": model.gpu_layers,
        "threads": model.threads,
        "batch_size": model.batch_size,
        "flash_attention": model.flash_attention,
        "kv_cache_type": model.kv_cache_type,
        "chat_format": model.chat_format,
        "response_cleanup": model.response_cleanup,
        "generation": model.generation.model_dump(mode="json"),
        "system_prompt_sha256": hashlib.sha256(
            model.system_prompt.encode("utf-8")
        ).hexdigest(),
        "chat_template_sha256": (
            hashlib.sha256(model.chat_format.encode("utf-8")).hexdigest()
            if model.chat_format is not None
            else None
        ),
    }


def _item_prompt_hash(item: DatasetItem) -> str:
    value = (
        json.dumps(
            [message.model_dump(mode="json") for message in item.conversation],
            sort_keys=True,
        )
        if item.conversation is not None
        else item.prompt
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _generation_for_item(
    generation: GenerationConfig,
    item: DatasetItem,
) -> GenerationConfig:
    if generation.constrained_decoding != "json_when_requested":
        return generation
    mode = "json" if item.response_contract.type == "json" else "none"
    return generation.model_copy(update={"constrained_decoding": mode})


def _prepare_response_for_scoring(
    response: str,
    cleanup: str,
) -> tuple[str, str | None]:
    if cleanup == "strip_think":
        cleaned = re.sub(
            r"^(?:\s*<think\b[^>]*>.*?</think>\s*)+",
            "",
            response,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cleaned != response:
            return cleaned, "strip_think"
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
