import json
from pathlib import Path

from llm_workload_benchmark.adjudication import (
    _derive_outcomes,
    adjudicate_experiment,
    adjudication_route,
)
from llm_workload_benchmark.config import BenchmarkConfig
from llm_workload_benchmark.dataset import load_suite, score_answer
from llm_workload_benchmark.evaluation import finalize_evaluation
from llm_workload_benchmark.judge import (
    BlindExtractionDecision,
    JudgeCallResult,
    SemanticJudgeDecision,
)


class FakeSemanticBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **arguments) -> JudgeCallResult:
        self.calls.append(arguments)
        payload = json.loads(str(arguments["user_prompt"]))
        decision = SemanticJudgeDecision.model_validate(
            {
                "requirements": [
                    {
                        "id": requirement["id"],
                        "satisfied": True,
                        "contradicted": False,
                        "reason": "The candidate conveys the requirement.",
                    }
                    for requirement in payload["semantic_requirements"]
                ],
                "ambiguous": False,
                "overall_correct": True,
                "overall_reason": "All meaning-level requirements are satisfied.",
            }
        )
        return JudgeCallResult(
            decision=decision.model_dump(mode="json"),
            response_id="judge-1",
            model="gpt-oss-120b",
            system_fingerprint="test",
            prompt_tokens=100,
            cached_prompt_tokens=0,
            output_tokens=40,
            reasoning_tokens=10,
            latency_seconds=0.1,
            finish_reason="stop",
        )


class FakeExtractionBackend:
    def __init__(self, extracted_answer: str) -> None:
        self.extracted_answer = extracted_answer
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **arguments) -> JudgeCallResult:
        self.calls.append(arguments)
        decision = BlindExtractionDecision(
            status="extracted",
            extracted_answer=self.extracted_answer,
            reason="The candidate clearly states one final answer.",
        )
        return JudgeCallResult(
            decision=decision.model_dump(mode="json"),
            response_id="extract-1",
            model="gpt-oss-120b",
            system_fingerprint="test",
            prompt_tokens=50,
            cached_prompt_tokens=0,
            output_tokens=20,
            reasoning_tokens=5,
            latency_seconds=0.1,
            finish_reason="stop",
        )


def _finalized_record(item, answer: str) -> dict[str, object]:
    evaluation = finalize_evaluation(
        score_answer(item, answer),
        primary_outcome="semantic",
        scoring_method=item.scoring.method,
        raw_response=answer,
        finish_reason="stop",
    )
    return {
        "status": "completed",
        "scoring_method": item.scoring.method,
        "evaluated_response": answer,
        "evaluation": evaluation.model_dump(mode="json"),
    }


def test_adjudication_writes_resumable_sidecars_without_mutating_inference(
    tmp_path: Path,
) -> None:
    item = {
        candidate.id: candidate
        for candidate in load_suite(Path("data/suites/instruction.yaml")).items[
            "constraint_load_curve"
        ]
    }["constraint_api_rate_limiting_001"]
    answer = str(item.expected["value"])
    evaluation = score_answer(item, answer).model_dump(mode="json")
    experiment = tmp_path / "experiment"
    run = experiment / "models" / "fake-model"
    run.mkdir(parents=True)
    record = {
        "schema_version": 4,
        "status": "completed",
        "model_id": "fake-model",
        "benchmark": item.benchmark,
        "item_id": item.id,
        "repetition": 0,
        "seed": 42,
        "scoring_method": item.scoring.method,
        "evaluated_response": answer,
        "evaluation": evaluation,
    }
    results_path = run / "results.jsonl"
    original = json.dumps(record, sort_keys=True) + "\n"
    results_path.write_text(original, encoding="utf-8")
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "fake-model",
                        "status": "completed",
                        "run_directory": "models/fake-model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = BenchmarkConfig.model_validate(
        {
            "schema_version": 1,
            "benchmark": {
                "name": "adjudication-test",
                "workload_path": "data/suites/instruction.yaml",
                "output_root": str(tmp_path / "runs"),
                "seed": 42,
            },
            "judge": {"provider": "cerebras"},
            "models": [
                {
                    "id": "fake-model",
                    "backend": "llama_cpp",
                    "model_path": str(tmp_path / "fake.gguf"),
                }
            ],
        }
    )
    backend = FakeSemanticBackend()

    output = adjudicate_experiment(
        experiment,
        config,
        project_root=Path.cwd(),
        judge_backend_factory=lambda _: backend,
    )
    sidecars = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(sidecars) == 1
    assert sidecars[0]["derived"]["semantic_correct"] is True
    assert sidecars[0]["derived"]["strict_pass"] is True
    assert results_path.read_text(encoding="utf-8") == original

    adjudicate_experiment(
        experiment,
        config,
        project_root=Path.cwd(),
        judge_backend_factory=lambda _: backend,
    )
    assert len(backend.calls) == 1


def test_constraint_derivation_separates_core_tone_and_mechanical_rules() -> None:
    record = {
        "scoring_method": "constraint_rules",
        "evaluation": {"details": {"checks": {"exact_sentences": True}}},
    }
    judgment = {
        "passed": False,
        "details": {
            "ambiguous": False,
            "overall_correct": False,
            "requirements": [
                {
                    "id": "core_fact:refund",
                    "satisfied": True,
                    "contradicted": False,
                },
                {
                    "id": "semantic_tone",
                    "satisfied": False,
                    "contradicted": False,
                },
            ],
        },
    }

    derived = _derive_outcomes(record, judgment)

    assert derived["semantic_correct"] is True
    assert derived["format_correct"] is True
    assert derived["instruction_compliant"] is False
    assert derived["loose_pass"] is True
    assert derived["strict_pass"] is False


def test_routing_uses_unknown_semantics_not_format_failure() -> None:
    item = load_suite(Path("data/suites/reasoning.yaml")).items[
        "applied_reasoning"
    ][0]

    wrong_but_parseable = _finalized_record(item, "FINAL: 154 kg")
    assert wrong_but_parseable["evaluation"]["semantic_outcome"] == "incorrect"
    assert adjudication_route(wrong_but_parseable, item) is None

    correct_but_misformatted = _finalized_record(item, "112 kg")
    assert correct_but_misformatted["evaluation"]["semantic_outcome"] == "correct"
    assert correct_but_misformatted["evaluation"]["protocol_outcome"] == "noncompliant"
    assert adjudication_route(correct_but_misformatted, item) is None

    unknown = _finalized_record(
        item, "I get one hundred and twelve kilograms."
    )
    route = adjudication_route(unknown, item)
    assert unknown["evaluation"]["semantic_outcome"] == "not_scored"
    assert route is not None
    assert route.kind == "blind_extraction"


def test_routing_keeps_authoritative_failures_out_of_judging() -> None:
    suite = load_suite(Path("data/suites/final_six.yaml"))
    json_item = suite.items["messy_text_to_schema"][0]
    wrong_json = json.dumps({**json_item.expected["value"], "vendor": "Wrong"})
    assert adjudication_route(_finalized_record(json_item, wrong_json), json_item) is None

    invalid_json = _finalized_record(json_item, "not valid json")
    route = adjudication_route(invalid_json, json_item)
    assert route is not None
    assert route.kind == "blind_extraction"

    code_item = suite.items["code_debug_repair"][0]
    code_record = {
        "evaluation": {
            "passed": False,
            "semantic_outcome": "incorrect",
            "details": {"tests_passed": 0},
        }
    }
    assert adjudication_route(code_record, code_item) is None


def test_blind_extraction_is_rescored_without_exposing_task_or_gold(
    tmp_path: Path,
) -> None:
    item = load_suite(Path("data/suites/final_six.yaml")).items[
        "long_text_retrieval"
    ][0]
    answer = f"The required path is {item.expected['value']}."
    record = {
        "schema_version": 4,
        "model_id": "fake-model",
        "benchmark": item.benchmark,
        "item_id": item.id,
        "repetition": 0,
        "seed": 42,
        **_finalized_record(item, answer),
    }
    experiment = tmp_path / "experiment"
    run = experiment / "models" / "fake-model"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "fake-model",
                        "status": "completed",
                        "run_directory": "models/fake-model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = BenchmarkConfig.model_validate(
        {
            "schema_version": 1,
            "benchmark": {
                "name": "extraction-test",
                "workload_path": "data/suites/final_six.yaml",
                "output_root": str(tmp_path / "runs"),
                "seed": 42,
            },
            "judge": {"provider": "cerebras"},
            "models": [
                {
                    "id": "fake-model",
                    "backend": "llama_cpp",
                    "model_path": str(tmp_path / "fake.gguf"),
                }
            ],
        }
    )
    backend = FakeExtractionBackend(str(item.expected["value"]))

    output = adjudicate_experiment(
        experiment,
        config,
        project_root=Path.cwd(),
        judge_backend_factory=lambda _: backend,
    )
    sidecar = json.loads(
        (output / "results.jsonl").read_text(encoding="utf-8").strip()
    )

    assert sidecar["judge_route"] == "blind_extraction"
    assert sidecar["deterministic_rescore"]["passed"] is True
    assert sidecar["derived"]["semantic_status"] == "correct"
    assert sidecar["derived"]["format_correct"] is False
    assert sidecar["derived"]["loose_pass"] is True
    assert sidecar["derived"]["strict_pass"] is False

    judge_payload = json.loads(str(backend.calls[0]["user_prompt"]))
    assert set(judge_payload) == {
        "response_contract",
        "scoring_method",
        "normalization_hints",
        "candidate",
    }
    assert item.prompt not in str(backend.calls[0]["user_prompt"])
    assert "reference_answer" not in judge_payload


def test_ambiguous_extraction_remains_unknown() -> None:
    judgment = {
        "details": {
            "status": "ambiguous",
            "extracted_answer": "",
        }
    }
    record = {
        "scoring_method": "numeric_tolerance",
        "evaluated_response": "It may be 145 or 154.",
        "evaluation": {"protocol_outcome": "noncompliant", "details": {}},
    }

    derived = _derive_outcomes(
        record,
        judgment,
        route_kind="blind_extraction",
        deterministic_rescore=None,
    )

    assert derived["semantic_status"] == "unknown"
    assert derived["semantic_correct"] is None
    assert derived["strict_pass"] is False
    assert derived["loose_pass"] is False


def test_ambiguous_semantic_judgment_remains_unknown() -> None:
    judgment = {
        "details": {
            "ambiguous": True,
            "overall_correct": False,
            "requirements": [
                {
                    "id": "core_fact:refund",
                    "satisfied": False,
                    "contradicted": False,
                }
            ],
        }
    }
    record = {
        "scoring_method": "constraint_rules",
        "evaluation": {"details": {"checks": {"exact_sentences": True}}},
    }

    derived = _derive_outcomes(record, judgment)

    assert derived["semantic_status"] == "unknown"
    assert derived["semantic_correct"] is None
    assert derived["instruction_compliant"] is None
    assert derived["strict_pass"] is False
    assert derived["loose_pass"] is False
