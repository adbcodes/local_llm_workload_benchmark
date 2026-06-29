from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from llm_workload_benchmark.dataset import DatasetItem
from llm_workload_benchmark.evaluation import EvaluationResult
from llm_workload_benchmark.answer_parser import parse_answer


class ExecutableEvaluationError(RuntimeError):
    """Raised when an executable task or its restricted runner is invalid."""


EXECUTABLE_EVALUATOR_VERSION = 2
_BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
)
_BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "print",
    "setattr",
    "vars",
    "__import__",
}

_WORKER = r"""
import copy
import json
import sys

payload = json.loads(sys.stdin.read())
allowed_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "ArithmeticError": ArithmeticError, "IndexError": IndexError,
    "KeyError": KeyError, "TypeError": TypeError, "ValueError": ValueError,
}
namespace = {"__builtins__": allowed_builtins}
try:
    exec(compile(payload["source"], "<candidate>", "exec"), namespace, namespace)
    function = namespace[payload["entry_point"]]
    results = []
    for case in payload["tests"]:
        try:
            args = case.get("args", [])
            kwargs = case.get("kwargs", {})
            preserved = {
                str(index): copy.deepcopy(args[index])
                for index in case.get("preserve_args", [])
            }
            actual = function(*args, **kwargs)
            mutated_arguments = [
                int(index)
                for index, before in preserved.items()
                if type(args[int(index)]) is not type(before) or args[int(index)] != before
            ]
            value_passed = type(actual) is type(case["expected"]) and actual == case["expected"]
            passed = value_passed and not mutated_arguments
            results.append({
                "passed": passed,
                "actual": actual,
                "value_passed": value_passed,
                "mutated_arguments": mutated_arguments,
            })
        except BaseException as error:
            results.append({
                "passed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            })
    print(json.dumps({"status": "completed", "results": results}, ensure_ascii=False))
except BaseException as error:
    print(json.dumps({
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
    }, ensure_ascii=False))
"""


def evaluate_python(item: DatasetItem, answer: str) -> EvaluationResult:
    """Run one function-only Python answer against JSON-serializable tests."""

    if item.scoring.method != "executable_python":
        raise ExecutableEvaluationError(
            f"item {item.id!r} is not configured for Python execution"
        )

    source, wrapper = _extract_source(answer)
    parameters = item.scoring.parameters
    specification = item.expected["value"]
    entry_point = specification["entry_point"]
    tests = specification["tests"]
    validation_error = _validate_candidate(source, entry_point)
    if validation_error is not None:
        return EvaluationResult(
            type="executable",
            evaluator="restricted_python_tests",
            version=EXECUTABLE_EVALUATOR_VERSION,
            passed=False,
            score=0,
            details={
                "reason": "rejected_source",
                "validation_error": validation_error,
                "test_count": len(tests),
                "tests_passed": 0,
                "diagnostic_wrapper": wrapper,
            },
        )

    payload = json.dumps(
        {"source": source, "entry_point": entry_point, "tests": tests},
        ensure_ascii=False,
    )
    timeout_seconds = float(parameters["timeout_seconds"])
    output_limit = int(parameters["max_output_characters"])
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="llm-benchmark-python-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _WORKER],
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(directory),
                env={"PATH": os.environ.get("PATH", "")},
                timeout=timeout_seconds,
                check=False,
                start_new_session=True,
                preexec_fn=_resource_limits(
                    timeout_seconds,
                    int(parameters["memory_limit_mb"]),
                    output_limit,
                ),
            )
    except subprocess.TimeoutExpired:
        return _execution_failure(
            "timeout",
            len(tests),
            time.perf_counter() - started,
            wrapper,
        )
    except OSError as error:
        raise ExecutableEvaluationError(
            f"could not start restricted Python process: {error}"
        ) from error

    latency = time.perf_counter() - started
    if len(completed.stdout) > output_limit or len(completed.stderr) > output_limit:
        return _execution_failure("output_limit", len(tests), latency, wrapper)
    if completed.returncode != 0:
        reason = (
            "resource_limit"
            if completed.returncode < 0
            and -completed.returncode in {signal.SIGKILL, signal.SIGXCPU}
            else "worker_failure"
        )
        return _execution_failure(reason, len(tests), latency, wrapper)

    try:
        worker_result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return _execution_failure("invalid_worker_output", len(tests), latency, wrapper)
    if worker_result.get("status") != "completed":
        return _execution_failure(
            "candidate_load_error",
            len(tests),
            latency,
            wrapper,
            error=worker_result.get("error"),
        )

    results = worker_result["results"]
    tests_passed = sum(result.get("passed") is True for result in results)
    score = tests_passed / len(tests)
    failures = [
        {
            "test_index": index,
            **(
                {"mutated_arguments": result["mutated_arguments"]}
                if result.get("mutated_arguments")
                else {}
            ),
            **({"error": result["error"]} if "error" in result else {}),
        }
        for index, result in enumerate(results, start=1)
        if result.get("passed") is not True
    ]
    return EvaluationResult(
        type="executable",
        evaluator="restricted_python_tests",
        version=EXECUTABLE_EVALUATOR_VERSION,
        passed=tests_passed == len(tests),
        score=score,
        details={
            "reason": "all_tests_passed" if score == 1 else "tests_failed",
            "entry_point": entry_point,
            "test_count": len(tests),
            "tests_passed": tests_passed,
            "failures": failures,
            "latency_seconds": latency,
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": parameters["memory_limit_mb"],
            "diagnostic_wrapper": wrapper,
        },
    )


def _extract_source(answer: str) -> tuple[str, str | None]:
    parsed = parse_answer(answer, "code")
    if not parsed.parsed:
        return answer.strip(), None
    wrapper = "markdown_fence" if "markdown_fence" in parsed.protocol_violations else None
    return str(parsed.value), wrapper


def _validate_candidate(source: str, entry_point: str) -> str | None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        return f"syntax_error on line {error.lineno}: {error.msg}"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return "source must contain exactly one function definition"
    function = tree.body[0]
    if function.name != entry_point:
        return f"expected function {entry_point!r}, found {function.name!r}"
    if function.decorator_list:
        return "function decorators are not allowed"
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            return f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "dunder names are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "private and dunder attributes are not allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                return f"call to {node.func.id!r} is not allowed"
    return None


def _resource_limits(timeout_seconds: float, memory_limit_mb: int, output_limit: int):
    def apply_limits() -> None:
        try:
            import resource

            cpu_seconds = max(1, math.ceil(timeout_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_limit_mb * 1024 * 1024, memory_limit_mb * 1024 * 1024),
            )
            resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit, output_limit))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except (ImportError, OSError, ValueError):
            pass

    return apply_limits


def _execution_failure(
    reason: str,
    test_count: int,
    latency_seconds: float,
    wrapper: str | None,
    *,
    error: Any = None,
) -> EvaluationResult:
    details: dict[str, Any] = {
        "reason": reason,
        "test_count": test_count,
        "tests_passed": 0,
        "latency_seconds": latency_seconds,
        "diagnostic_wrapper": wrapper,
    }
    if error is not None:
        details["error"] = error
    return EvaluationResult(
        type="executable",
        evaluator="restricted_python_tests",
        version=EXECUTABLE_EVALUATOR_VERSION,
        passed=False,
        score=0,
        details=details,
    )
