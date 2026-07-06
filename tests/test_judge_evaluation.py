import json
from pathlib import Path

from typer.testing import CliRunner

from llm_workload_benchmark.config import JudgeConfig
from llm_workload_benchmark.judge import JudgeCallResult
from llm_workload_benchmark.judge_evaluation import (
    JudgeEvaluationConfig,
    load_judge_evaluation_dataset,
    run_judge_evaluation,
)
from llm_workload_benchmark.judge_evaluation_cli import (
    _ProgressDisplay,
    _progress_line,
    app,
)


FIXTURE = Path("data/judge_evaluation/cases.yaml")


class GoldFixtureBackend:
    def __init__(self, cases) -> None:
        self.by_candidate = {}
        for case in cases:
            self.by_candidate[case.candidate] = case
            if case.tool_explanation:
                try:
                    parsed_candidate = json.loads(case.candidate)
                except json.JSONDecodeError:
                    parsed_candidate = None
                if isinstance(parsed_candidate, dict) and isinstance(
                    parsed_candidate.get("answer"), str
                ):
                    self.by_candidate[parsed_candidate["answer"]] = case
        self.calls = 0

    def evaluate(self, **arguments) -> JudgeCallResult:
        self.calls += 1
        payload = json.loads(arguments["user_prompt"])
        case = self.by_candidate[payload["candidate"]]
        if case.route == "semantic_requirements":
            gold = case.semantic_gold
            decision = {
                "requirements": [
                    {
                        "id": requirement_id,
                        "satisfied": requirement.satisfied,
                        "contradicted": requirement.contradicted,
                        "reason": "Matches the reviewed calibration label.",
                    }
                    for requirement_id, requirement in gold.requirements.items()
                ],
                "ambiguous": gold.ambiguous,
                "overall_correct": gold.overall_correct,
                "overall_reason": "Matches the reviewed calibration label.",
            }
        else:
            gold = case.extraction_gold
            decision = {
                "status": gold.status,
                "extracted_answer": (
                    gold.accepted_answers[-1] if gold.accepted_answers else ""
                ),
                "reason": "Matches the reviewed calibration label.",
            }
        return JudgeCallResult(
            decision=decision,
            response_id=f"fake-{self.calls}",
            model="gpt-oss-120b",
            system_fingerprint="fixture-test",
            prompt_tokens=20,
            cached_prompt_tokens=0,
            output_tokens=10,
            reasoning_tokens=5,
            latency_seconds=0.01,
            finish_reason="stop",
        )


def test_human_fixture_is_generous_and_covers_production_routes() -> None:
    dataset = load_judge_evaluation_dataset(FIXTURE)

    assert len(dataset.cases) == 60
    assert {case.route for case in dataset.cases} == {
        "semantic_requirements",
        "blind_extraction",
    }
    assert {
        "paraphrase",
        "negation",
        "partial",
        "contradiction",
        "ambiguity",
        "tone",
        "required_idea",
        "tool_stop",
        "numeric",
        "date",
        "identifier",
        "json",
        "set",
        "rational",
        "no_answer",
        "adversarial",
    } <= {case.category for case in dataset.cases}
    assert any(
        case.semantic_gold and not case.semantic_gold.overall_correct
        for case in dataset.cases
    )
    assert any(
        case.extraction_gold and case.extraction_gold.status == "ambiguous"
        for case in dataset.cases
    )
    hard_semantic_cases = [
        case
        for case in dataset.cases
        if case.route == "semantic_requirements" and len(case.candidate) >= 400
    ]
    assert len(hard_semantic_cases) >= 5
    hard_extraction_cases = [
        case
        for case in dataset.cases
        if case.route == "blind_extraction" and len(case.candidate) >= 300
    ]
    assert len(hard_extraction_cases) >= 5


def test_standalone_pipeline_records_metrics_plots_and_resumes(tmp_path: Path) -> None:
    dataset = load_judge_evaluation_dataset(FIXTURE)
    backend = GoldFixtureBackend(dataset.cases)
    output = tmp_path / "calibration"
    config = JudgeEvaluationConfig(
        schema_version=1,
        dataset_path=FIXTURE,
        output_root=tmp_path,
        judge=JudgeConfig(provider="cerebras"),
    )

    result = run_judge_evaluation(
        config,
        project_root=Path.cwd(),
        output_directory=output,
        backend_factory=lambda _: backend,
    )

    assert result == output.resolve()
    assert backend.calls == len(dataset.cases)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["exact_agreement"] == 1.0
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["overall"]["precision"] == 1.0
    assert summary["overall"]["recall"] == 1.0
    assert summary["overall"]["api_failures"] == 0
    assert summary["overall"]["prompt_tokens"] == 20 * len(dataset.cases)
    assert summary["overall"]["estimated_cost_usd"] > 0
    for artifact in (
        "results.jsonl",
        "summary.json",
        "cases.csv",
        "metrics.csv",
        "manifest.json",
    ):
        assert (output / artifact).is_file()
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.svg"))
    assert {
        "semantic_pass",
        "correct_misformatted",
        "wrong_answer",
        "partial_or_missing",
        "contradiction",
        "ambiguous",
        "no_answer",
    } <= set(summary["scenario"])

    run_judge_evaluation(
        config,
        project_root=Path.cwd(),
        output_directory=output,
        backend_factory=lambda _: backend,
    )
    assert backend.calls == len(dataset.cases)


def test_cli_is_a_separate_entrypoint() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "report reliability" in result.output


def test_judge_evaluation_progress_updates_one_terminal_line() -> None:
    from io import StringIO

    stream = StringIO()
    display = _ProgressDisplay(stream)

    display.update(1, 4, "first_case")
    display.update(4, 4, "last_case")

    output = stream.getvalue()
    assert output.count("\r\033[2K") == 2
    assert output.count("\n") == 1
    assert "25.0% | 1/4 | first_case" in output
    assert "100.0% | 4/4 | last_case" in output
    assert _progress_line(0, 0, "waiting").startswith("[----------------------------]")
