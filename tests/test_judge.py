import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from llm_workload_benchmark.config import JudgeConfig, load_config
from llm_workload_benchmark.dataset import (
    DatasetError,
    DatasetItem,
    load_suite,
    score_answer,
)
from llm_workload_benchmark.judge import (
    CachedJudgeBackend,
    CerebrasJudgeBackend,
    GroqJudgeBackend,
    JudgeCallResult,
    JudgeError,
    SummaryJudgeDecision,
    SemanticJudgeDecision,
    _blind_extraction_instructions,
    _is_rate_limit_error,
    _rate_limit_retry_after_seconds,
    evaluate_summary,
    evaluate_summary_panel,
    evaluate_semantic_requirements,
    semantic_requirements_for_item,
)
from llm_workload_benchmark.runner import (
    EvaluationError,
    GenerationOutput,
    run_benchmark,
)

JUDGED_SUITE_PATH = Path("data/suites/judged.yaml").resolve()


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
            decision=self.decision.model_dump(mode="json"),
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


def test_judged_summary_suite_loads_complete_first_pass_set() -> None:
    suite = load_suite(JUDGED_SUITE_PATH)
    item = _judged_item()

    assert suite.manifest.status == "pilot"
    assert len(suite.items["grounded_compression"]) == 30
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
    assert config.cache_path is None
    assert config.rate_limit_cooldown_retries == 1
    assert config.rate_limit_fallback_wait_seconds == 60
    assert config.rate_limit_max_wait_seconds == 3600


def test_cerebras_judge_uses_provider_specific_defaults() -> None:
    config = JudgeConfig(provider="cerebras")

    assert config.model == "gpt-oss-120b"
    assert config.api_key_env == "CEREBRAS_API_KEY"
    assert config.input_price_per_million_tokens == 0.35
    assert config.cached_input_price_per_million_tokens == 0.35
    assert config.output_price_per_million_tokens == 0.75


class FakeRateLimitError(Exception):
    status_code = 429

    def __init__(self, message: str, *, retry_after: str | None = None) -> None:
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)


def test_groq_judge_waits_for_rate_limit_reset_then_retries() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=_decision().model_dump_json()),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        id="judge-after-cooldown",
        model="openai/gpt-oss-120b",
        system_fingerprint="test-fingerprint",
    )

    class Completions:
        calls = 0

        def create(self, **arguments):
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError("rate limited", retry_after="2.5")
            return completion

    completions = Completions()
    backend = GroqJudgeBackend.__new__(GroqJudgeBackend)
    backend._config = JudgeConfig()
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    sleeps: list[float] = []
    notices: list[str] = []
    backend._sleep = sleeps.append
    backend._retry_notifier = notices.append

    result = backend.evaluate(
        system_prompt="judge policy",
        user_prompt="candidate answer",
        response_schema=SummaryJudgeDecision.model_json_schema(),
        seed=42,
    )

    assert completions.calls == 2
    assert sleeps == [2.5]
    assert notices and "waiting 2.5s" in notices[0]
    assert result.rate_limit_retries == 1
    assert result.rate_limit_wait_seconds == 2.5


def test_groq_judge_parses_long_reset_and_respects_wait_cap() -> None:
    error = FakeRateLimitError("Please try again in 25m11.568s.")
    assert _rate_limit_retry_after_seconds(error) == pytest.approx(1511.568)

    class Completions:
        def create(self, **arguments):
            raise error

    backend = GroqJudgeBackend.__new__(GroqJudgeBackend)
    backend._config = JudgeConfig(rate_limit_max_wait_seconds=60)
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    sleeps: list[float] = []
    backend._sleep = sleeps.append
    backend._retry_notifier = lambda message: None

    with pytest.raises(JudgeError, match="above the configured maximum"):
        backend.evaluate(
            system_prompt="judge policy",
            user_prompt="candidate answer",
            response_schema=SummaryJudgeDecision.model_json_schema(),
            seed=42,
        )
    assert not sleeps


def test_rate_limit_detection_accepts_cerebras_rpm_message_without_status() -> None:
    error = RuntimeError("Requests per minute limit exceeded - too many requests sent.")

    assert _is_rate_limit_error(error)
    assert _rate_limit_retry_after_seconds(error) is None


def test_blind_extraction_instructions_are_type_specific() -> None:
    suite = load_suite(Path("data/suites/final_six.yaml"))
    date_item = next(
        item
        for items in suite.items.values()
        for item in items
        if item.scoring.method == "date_value"
    )
    json_item = suite.items["messy_text_to_schema"][0]
    set_item = DatasetItem.model_validate(
        {
            "id": "set_instruction_test",
            "benchmark": "judge_test",
            "subcategory": "set",
            "difficulty": "easy",
            "split": "dev",
            "prompt": "Return the matching labels.",
            "response_contract": {
                "type": "text",
                "format": "comma_separated_labels",
            },
            "expected": {"value": ["north", "west"]},
            "scoring": {
                "method": "set_match",
                "parameters": {"separator": ",", "case_sensitive": False},
            },
            "provenance": {
                "kind": "hand_authored",
                "review_status": "human_checked",
            },
        }
    )

    date_format = _blind_extraction_instructions(date_item)[
        "extracted_answer_format"
    ]
    json_format = _blind_extraction_instructions(json_item)[
        "extracted_answer_format"
    ]
    set_format = _blind_extraction_instructions(set_item)[
        "extracted_answer_format"
    ]

    assert "YYYY-MM-DD" in date_format
    assert "ISO 8601" in date_format
    assert "valid compact JSON" in json_format
    assert "no Markdown fence or prose" in json_format
    assert "distinct claimed set member once" in set_format
    date_instructions = _blind_extraction_instructions(date_item)
    assert date_instructions["valid_extracted_answer_example"] == "2026-07-05"
    assert "05/07/2026" in date_instructions["invalid_extracted_answer_examples"]


def test_cerebras_judge_sends_supported_structured_output_parameters() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=_decision().model_dump_json()),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        id="cerebras-judge",
        model="gpt-oss-120b",
        system_fingerprint="cerebras-test-fingerprint",
    )

    class Completions:
        arguments = None

        def create(self, **arguments):
            self.arguments = arguments
            return completion

    completions = Completions()
    backend = CerebrasJudgeBackend.__new__(CerebrasJudgeBackend)
    backend._config = JudgeConfig(provider="cerebras")
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend._sleep = lambda seconds: None
    backend._retry_notifier = lambda message: None

    result = backend.evaluate(
        system_prompt="judge policy",
        user_prompt="candidate answer",
        response_schema=SummaryJudgeDecision.model_json_schema(),
        seed=42,
    )

    assert result.model == "gpt-oss-120b"
    assert completions.arguments["reasoning_effort"] == "medium"
    response_format = completions.arguments["response_format"]
    assert response_format["type"] == "json_schema"
    cerebras_schema = response_format["json_schema"]["schema"]
    encoded_schema = json.dumps(cerebras_schema)
    assert "minLength" not in encoded_schema
    assert "title" not in encoded_schema
    assert "description" not in encoded_schema
    assert cerebras_schema["additionalProperties"] is False
    assert "include_reasoning" not in completions.arguments


def test_judge_cache_reuses_exact_decisions_without_request_cost(tmp_path: Path) -> None:
    config = JudgeConfig(cache_path=tmp_path / "judge-cache.jsonl")
    delegate = FakeJudgeBackend()
    backend = CachedJudgeBackend(delegate, config)
    arguments = {
        "system_prompt": "judge policy",
        "user_prompt": "candidate answer",
        "response_schema": SummaryJudgeDecision.model_json_schema(),
        "seed": 42,
    }

    first = backend.evaluate(**arguments)
    second = backend.evaluate(**arguments)

    assert len(delegate.calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.latency_seconds == 0.0
    assert second.prompt_tokens == second.output_tokens == 0

    fresh_delegate = FakeJudgeBackend()
    disk_backed = CachedJudgeBackend(fresh_delegate, config)
    from_disk = disk_backed.evaluate(**arguments)
    assert from_disk.cache_hit is True
    assert not fresh_delegate.calls


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
        "reference_answer": item.expected["value"],
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


def test_semantic_judge_detects_negated_instruction_fact() -> None:
    item = {
        candidate.id: candidate
        for candidate in load_suite(Path("data/suites/instruction.yaml")).items[
            "constraint_load_curve"
        ]
    }["constraint_api_rate_limiting_001"]
    requirements = semantic_requirements_for_item(item)
    requirement_id = requirements[0]["id"]
    decision = SemanticJudgeDecision.model_validate(
        {
            "requirements": [
                {
                    "id": requirement_id,
                    "satisfied": False,
                    "contradicted": True,
                    "reason": "The candidate states the opposite.",
                }
            ],
            "ambiguous": False,
            "overall_correct": False,
            "overall_reason": "The required fact is contradicted.",
        }
    )
    result = evaluate_semantic_requirements(
        item,
        "Rate limiting does not prevent overload.",
        backend=FakeJudgeBackend(decision),
        config=JudgeConfig(),
        seed=42,
    )

    assert not result.passed
    assert result.score == 0
    assert result.details["requirements"][0]["contradicted"] is True


def test_tool_semantic_judge_receives_only_the_direct_answer_text() -> None:
    item = {
        candidate.id: candidate
        for candidate in load_suite(Path("data/suites/final_six.yaml")).items["tool_use"]
    }["tool_use_027"]
    requirement = semantic_requirements_for_item(item)[0]
    decision = SemanticJudgeDecision.model_validate(
        {
            "requirements": [
                {
                    "id": requirement["id"],
                    "satisfied": True,
                    "contradicted": False,
                    "reason": "The answer explains why no event was created.",
                }
            ],
            "ambiguous": False,
            "overall_correct": True,
            "overall_reason": "The explanation matches the observation.",
        }
    )
    backend = FakeJudgeBackend(decision)
    candidate = {
        "tool_call": None,
        "arguments": {},
        "answer": "Rain is expected, so I did not create the event.",
    }

    result = evaluate_semantic_requirements(
        item,
        json.dumps(candidate),
        backend=backend,
        config=JudgeConfig(),
        seed=42,
    )

    assert result.passed
    prompt = json.loads(backend.calls[0]["user_prompt"])
    assert prompt["candidate"] == candidate["answer"]
    assert "reference_answer" not in prompt
    assert set(prompt) == {
        "task",
        "conversation",
        "semantic_requirements",
        "candidate",
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


def test_judge_panel_uses_third_family_only_on_disagreement() -> None:
    item = _judged_item()
    backends = [
        FakeJudgeBackend(_decision()),
        FakeJudgeBackend(_decision(critical_error=True)),
        FakeJudgeBackend(_decision()),
    ]
    configs = [
        JudgeConfig(model="judge-a", family="family-a"),
        JudgeConfig(model="judge-b", family="family-b"),
        JudgeConfig(model="judge-c", family="family-c"),
    ]

    result = evaluate_summary_panel(
        item, item.expected["value"], backends=backends, configs=configs, seed=42
    )

    assert result.passed
    assert result.evaluator == "summary_judge_panel"
    assert result.details["tie_break_used"] is True
    assert result.details["judges_used"] == 3
    assert [len(backend.calls) for backend in backends] == [1, 1, 1]


def test_runner_dispatches_judged_item_and_records_usage(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake model")
    item = _judged_item()
    benchmark_directory = tmp_path / "grounded_compression"
    shutil.copytree(Path("data/grounded_compression"), benchmark_directory)
    suite_path = tmp_path / "single-judged-item.yaml"
    benchmark_path = benchmark_directory / "benchmark.yaml"
    suite_path.write_text(
        f"""
schema_version: 1
name: single-judged-item
version: 1
status: pilot
benchmark_files:
  - {benchmark_path}
filters:
  ids: [{item.id}]
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "judge-run.yaml"
    config_path.write_text(
        f"""
schema_version: 1
benchmark:
  name: judged-runner-test
  workload_path: {suite_path}
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


def test_cerebras_backend_requires_key_without_exposing_credentials(monkeypatch) -> None:
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    with pytest.raises(JudgeError, match="CEREBRAS_API_KEY.*not set"):
        CerebrasJudgeBackend(JudgeConfig(provider="cerebras"))
