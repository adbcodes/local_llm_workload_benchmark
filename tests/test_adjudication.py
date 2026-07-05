import json
from pathlib import Path

from llm_workload_benchmark.adjudication import _derive_outcomes, adjudicate_experiment
from llm_workload_benchmark.config import BenchmarkConfig
from llm_workload_benchmark.dataset import load_suite, score_answer
from llm_workload_benchmark.judge import JudgeCallResult, SemanticJudgeDecision


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
