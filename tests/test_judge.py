import json
from pathlib import Path

import pytest

from llm_workload_benchmark.config import JudgeConfig, load_config
from llm_workload_benchmark.dataset import DatasetError, load_suite, score_answer
from llm_workload_benchmark.judge import (
    GroqJudgeBackend,
    JudgeCallResult,
    JudgeError,
    SummaryJudgeDecision,
    evaluate_summary,
)
from llm_workload_benchmark.runner import (
    EvaluationError,
    GenerationOutput,
    run_benchmark,
)

JUDGED_SUITE_PATH = Path("data/benchmarks/v1/judged_suite.yaml").resolve()


def _decision(*, critical_error: bool = False) -> SummaryJudgeDecision:
    return SummaryJudgeDecision.model_validate(
        {
            "faithfulness": {"score": 4, "reason": "All claims are supported."},
            "coverage": {"score": 3, "reason": "One secondary detail is omitted."},
            "relevance": {"score": 4, "reason": "The emphasis fits the audience."},
            "clarity": {"score": 4, "reason": "The summary is unambiguous."},
            "concision": {"score": 4, "reason": "No low-value detail is present."},
            "critical_error": critical_error,
            "unsupported_claims": [],
            "missing_required_facts": ["customer complaint count"],
            "overall_reason": "The summary is faithful and decision-oriented.",
        }
    )


class FakeJudgeBackend:
    def __init__(self, decision: SummaryJudgeDecision | None = None) -> None:
        self.decision = decision or _decision()
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **arguments) -> JudgeCallResult:
        self.calls.append(arguments)
        return JudgeCallResult(
            decision=self.decision,
            response_id="judge-response-1",
            model="openai/gpt-oss-120b",
            system_fingerprint="groq-test-fingerprint",
            prompt_tokens=1000,
            cached_prompt_tokens=200,
            output_tokens=100,
            reasoning_tokens=40,
            latency_seconds=0.2,
            finish_reason="stop",
        )


def _judged_item():
    suite = load_suite(JUDGED_SUITE_PATH)
    return suite.items["grounded_compression"][0]


def test_judged_summary_suite_loads_one_incremental_pilot_item() -> None:
    suite = load_suite(JUDGED_SUITE_PATH)
    item = _judged_item()

    assert suite.manifest.status == "pilot"
    assert item.scoring.method == "llm_judge"
    assert item.scoring.parameters["rubric"] == "grounded_summary_v1"
    assert item.expected["value"]
    with pytest.raises(DatasetError, match="requires an external LLM judge"):
        score_answer(item, item.expected["value"])


def test_judge_defaults_leave_room_for_reasoning_and_structured_output() -> None:
    config = JudgeConfig()

    assert config.model == "openai/gpt-oss-120b"
    assert config.reasoning_effort == "medium"
    assert config.max_completion_tokens == 4096


def test_pointwise_judge_builds_anonymous_prompt_and_computes_score() -> None:
    item = _judged_item()
    backend = FakeJudgeBackend()
    config = JudgeConfig()

    result = evaluate_summary(
        item,
        item.expected["value"],
        backend=backend,
        config=config,
        seed=42,
    )

    assert result.type == "llm_judge"
    assert result.evaluator == "groq_gpt_oss_summary_rubric"
    assert result.passed
    assert result.score == pytest.approx(0.925)
    assert result.details["deterministic_checks"]["within_word_limit"] is True
    assert result.details["judge"]["estimated_cost_usd"] == pytest.approx(0.000195)
    assert len(result.details["judge"]["prompt_sha256"]) == 64

    call = backend.calls[0]
    prompt_payload = json.loads(call["user_prompt"])
    assert prompt_payload == {
        "task_and_source": item.prompt,
        "candidate_summary": item.expected["value"],
    }
    assert "qwen" not in call["user_prompt"].casefold()
    assert call["response_schema"]["additionalProperties"] is False
    assert set(call["response_schema"]["required"]) == {
        "faithfulness",
        "coverage",
        "relevance",
        "clarity",
        "concision",
        "critical_error",
        "unsupported_claims",
        "missing_required_facts",
        "overall_reason",
    }


def test_pointwise_judge_enforces_mechanical_gate_and_critical_error() -> None:
    item = _judged_item()
    config = JudgeConfig()

    too_long = " ".join(["word"] * 61)
    length_failure = evaluate_summary(
        item,
        too_long,
        backend=FakeJudgeBackend(),
        config=config,
        seed=42,
    )
    assert not length_failure.passed
    assert length_failure.score == pytest.approx(0.925)
    assert length_failure.details["deterministic_checks"]["word_count"] == 61

    critical_failure = evaluate_summary(
        item,
        item.expected["value"],
        backend=FakeJudgeBackend(_decision(critical_error=True)),
        config=config,
        seed=42,
    )
    assert not critical_failure.passed


def test_runner_dispatches_judged_item_and_records_usage(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake model")
    config_path = tmp_path / "judge-run.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: judged-runner-test
  workload_path: {JUDGED_SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  repetitions: 1
  seed: 42
judge:
  provider: groq
  model: openai/gpt-oss-120b
models:
  - id: fake-local-model
    backend: llama_cpp
    model_path: {model_path}
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    item = _judged_item()

    class SummaryBackend:
        def generate(self, prompt, generation, *, seed):
            assert prompt == item.prompt
            return GenerationOutput(
                text=item.expected["value"],
                prompt_tokens=100,
                output_tokens=50,
                time_to_first_token_seconds=0.01,
                finish_reason="stop",
            )

    fake_judge = FakeJudgeBackend()
    run_directory = run_benchmark(
        config,
        config_path,
        project_root=tmp_path,
        backend_factory=lambda model, path, seed: SummaryBackend(),
        judge_backend_factory=lambda judge_config: fake_judge,
    )

    record = json.loads((run_directory / "results.jsonl").read_text())
    assert record["evaluation"]["type"] == "llm_judge"
    assert record["evaluation"]["passed"] is True
    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["judge"] == {
        "cached_prompt_tokens": 200,
        "estimated_cost_usd": pytest.approx(0.000195),
        "evaluations": 1,
        "latency_seconds": 0.2,
        "output_tokens": 100,
        "prompt_tokens": 1000,
        "reasoning_tokens": 40,
    }


def test_runner_rejects_judged_workload_without_judge_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake model")
    config_path = tmp_path / "missing-judge.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: missing-judge
  workload_path: {JUDGED_SUITE_PATH}
  output_root: {tmp_path / 'runs'}
models:
  - id: fake-local-model
    backend: llama_cpp
    model_path: {model_path}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="no judge is configured"):
        run_benchmark(load_config(config_path), config_path, project_root=tmp_path)


def test_groq_backend_requires_key_without_exposing_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(JudgeError, match="GROQ_API_KEY.*not set"):
        GroqJudgeBackend(JudgeConfig())
