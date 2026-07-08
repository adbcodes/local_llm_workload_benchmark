import json
from pathlib import Path

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite
from llm_workload_benchmark.judge import JudgeCallResult, SummaryJudgeDecision
from llm_workload_benchmark.rejudge import rejudge_experiment
from llm_workload_benchmark.runner import GenerationOutput, run_matrix


JUDGED_SUITE_PATH = Path("data/suites/grounded_compression.yaml").resolve()


def _decision() -> SummaryJudgeDecision:
    assessment = {"score": 4, "reason": "The answer satisfies the criterion."}
    return SummaryJudgeDecision.model_validate(
        {
            "faithfulness": assessment,
            "coverage": assessment,
            "relevance": assessment,
            "clarity": assessment,
            "concision": assessment,
            "critical_error": False,
            "unsupported_claims": [],
            "missing_required_facts": [],
            "overall_reason": "The answer is correct.",
        }
    )


class FakeJudgeBackend:
    def evaluate(self, **arguments) -> JudgeCallResult:
        return JudgeCallResult(
            decision=_decision().model_dump(mode="json"),
            response_id="fake-judge-response",
            model="gpt-oss-120b",
            system_fingerprint="fake-fingerprint",
            prompt_tokens=100,
            cached_prompt_tokens=0,
            output_tokens=50,
            reasoning_tokens=20,
            latency_seconds=0.01,
            finish_reason="stop",
        )


def test_rejudge_copies_saved_answers_and_replaces_only_judge_results(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    source_config_path = tmp_path / "source.yaml"
    target_config_path = tmp_path / "target.yaml"
    common = f"""
schema_version: 1
benchmark:
  name: rejudge-test
  workload_path: {JUDGED_SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  repetitions: 1
  seed: 42
models:
  - id: fake-local-model
    backend: llama_cpp
    model_path: {model_path}
""".strip()
    source_config_path.write_text(
        common
        + "\njudge:\n  provider: groq\n  model: openai/gpt-oss-120b\n",
        encoding="utf-8",
    )
    target_config_path.write_text(
        common + "\njudge:\n  provider: cerebras\n  model: gpt-oss-120b\n",
        encoding="utf-8",
    )

    answers = {
        item.prompt: item.expected["value"]
        for item in load_suite(JUDGED_SUITE_PATH).items["grounded_compression"]
    }

    class SavedAnswerBackend:
        def generate(self, prompt, generation, *, seed):
            return GenerationOutput(
                text=answers[prompt],
                prompt_tokens=10,
                output_tokens=10,
                time_to_first_token_seconds=0.01,
                finish_reason="stop",
            )

    source = run_matrix(
        load_config(source_config_path),
        source_config_path,
        project_root=tmp_path,
        backend_factory=lambda model, path, seed: SavedAnswerBackend(),
        judge_backend_factory=lambda config: FakeJudgeBackend(),
    )
    source_config_path.write_text(
        source_config_path.read_text(encoding="utf-8")
        + "  reasoning_effort: low\n",
        encoding="utf-8",
    )
    source_results = source / "models" / "fake-local-model" / "results.jsonl"
    original_records = [json.loads(line) for line in source_results.read_text().splitlines()]

    target = rejudge_experiment(
        source,
        load_config(target_config_path),
        target_config_path,
        project_root=tmp_path,
        judge_backend_factory=lambda config: FakeJudgeBackend(),
    )

    target_results = target / "models" / "fake-local-model" / "results.jsonl"
    rejudged_records = [json.loads(line) for line in target_results.read_text().splitlines()]
    assert [record["raw_response"] for record in rejudged_records] == [
        record["raw_response"] for record in original_records
    ]
    assert all(
        record["evaluation"]["details"]["judge"]["provider"] == "cerebras"
        for record in rejudged_records
    )
    target_index = json.loads((target / "experiment.json").read_text())
    assert target_index["status"] == "completed"
    assert target_index["rejudged_from"] == str(source.resolve())
