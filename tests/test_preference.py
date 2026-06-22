import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_workload_benchmark.cli import app
from llm_workload_benchmark.preference import (
    BlindCandidate,
    BlindComparison,
    PreferenceError,
    append_human_preference,
    collect_human_preferences,
    default_preference_path,
    prepare_preference_ballot,
    summarize_human_preferences,
)
from llm_workload_benchmark.preference_terminal import (
    TerminalPreferenceResult,
    normalize_human_selection,
    render_terminal_comparison,
    run_terminal_preferences,
)

SUITE_PATH = Path("data/suites/core.yaml").resolve()


def _write_experiment(
    tmp_path: Path,
    model_ids=("first-model", "second-model", "third-model"),
) -> Path:
    experiment = tmp_path / "experiment"
    model_entries = []
    for model_number, model_id in enumerate(model_ids, start=1):
        run_directory = experiment / "models" / model_id
        run_directory.mkdir(parents=True)
        records = [
            {
                "status": "completed",
                "model_id": model_id,
                "benchmark": "applied_reasoning",
                "item_id": "reason_percentage_001",
                "repetition": 1,
                "evaluated_response": f"Candidate response {model_number}",
            },
            {
                "status": "completed",
                "model_id": model_id,
                "benchmark": "applied_reasoning",
                "item_id": "reason_calendar_001",
                "repetition": 1,
                "evaluated_response": (
                    f"Second candidate response {model_number}\nwith another line"
                ),
            },
        ]
        (run_directory / "results.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        model_entries.append(
            {
                "model_id": model_id,
                "status": "completed",
                "run_directory": f"models/{model_id}",
            }
        )
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "test-experiment",
                "dataset": str(SUITE_PATH),
                "models": model_entries,
            }
        ),
        encoding="utf-8",
    )
    return experiment


def test_arena_ballot_anonymizes_three_model_answers_together(tmp_path: Path) -> None:
    ballot = prepare_preference_ballot(_write_experiment(tmp_path), seed=7)
    comparison = ballot.items[0].comparison

    assert ballot.model_ids == ("first-model", "second-model", "third-model")
    assert [candidate.label for candidate in comparison.candidates] == ["a", "b", "c"]
    assert len(comparison.candidates) == 3
    assert not hasattr(comparison, "candidate_model_ids")
    assert set(ballot.items[0].candidate_model_ids.values()) == set(ballot.model_ids)


def test_multiway_votes_use_human_contract_and_preserve_hidden_mapping(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    choices = iter([("a", "c"), "none"])
    output = collect_human_preferences(
        experiment,
        chooser=lambda comparison: choices.__next__(),
        seed=7,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert records[0]["schema_version"] == 3
    assert records[0]["evaluation"]["type"] == "human"
    assert records[0]["evaluation"]["evaluator"] == "blind_multiway_preference"
    assert records[0]["evaluation"]["details"]["selected_labels"] == ["a", "c"]
    assert len(records[0]["evaluation"]["details"]["selected_model_ids"]) == 2
    assert len(records[0]["candidates"]) == 3
    assert records[1]["evaluation"]["details"]["selected_labels"] == []
    assert records[1]["evaluation"]["details"]["none_of_above"] is True
    assert records[1]["evaluation"]["passed"] is False

    summary = summarize_human_preferences(output)
    assert summary["votes"] == 2
    assert sum(summary["wins"].values()) == 2
    assert summary["none_of_above"] == 1


@pytest.mark.parametrize("model_count", [1, 4])
def test_human_arena_requires_two_or_three_completed_models(
    tmp_path: Path,
    model_count: int,
) -> None:
    model_ids = tuple(f"model-{number}" for number in range(model_count))
    experiment = _write_experiment(tmp_path, model_ids)

    with pytest.raises(PreferenceError, match="two or three completed models"):
        prepare_preference_ballot(experiment)


def test_human_preference_never_overwrites_existing_votes(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    collect_human_preferences(experiment, chooser=lambda comparison: ("a",))

    with pytest.raises(PreferenceError, match="already exists"):
        collect_human_preferences(experiment, chooser=lambda comparison: ("b",))


def test_terminal_renderer_places_three_anonymous_answers_side_by_side(
    tmp_path: Path,
) -> None:
    ballot = prepare_preference_ballot(_write_experiment(tmp_path))

    rendered = render_terminal_comparison(ballot.items[0].comparison, width=120)
    lines = rendered.splitlines()

    header = next(line for line in lines if "ANSWER A" in line)
    assert "ANSWER B" in header
    assert "ANSWER C" in header
    assert "[A] Answer A" in rendered
    assert "[C] Answer C" in rendered
    assert "[N] None of the above" in rendered
    assert "first-model" not in rendered
    assert "second-model" not in rendered
    assert "third-model" not in rendered
    assert all(len(line) <= 120 for line in lines)


def test_terminal_renderer_colours_python_without_breaking_three_columns() -> None:
    comparison = BlindComparison(
        number=1,
        total=1,
        benchmark="code_debug_repair",
        item_id="code_colour_test",
        repetition=1,
        prompt="Implement add_one.",
        candidates=tuple(
            BlindCandidate(label, "def add_one(value):\n    return int(value) + 1")
            for label in ("a", "b", "c")
        ),
    )

    rendered = render_terminal_comparison(comparison, width=120, color=True)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert "\x1b[1;36mdef\x1b[0m" in rendered
    assert "\x1b[35madd_one\x1b[0m" in rendered
    assert "\x1b[33mint\x1b[0m" in rendered
    assert all(len(line) <= 120 for line in plain.splitlines())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("A", "a"),
        ("a", "a"),
        ("B", "b"),
        ("b", "b"),
        ("C", "c"),
        ("c", "c"),
        ("A B", ("a", "b")),
        ("a,b", ("a", "b")),
        ("AC", ("a", "c")),
        ("a and c", ("a", "c")),
        ("N", "none"),
        ("n", "none"),
        ("NONE", "none"),
        ("none of above", "none"),
        ("None of the Above", "none"),
        ("tie", None),
    ],
)
def test_terminal_choice_normalization(value: str, expected: object) -> None:
    normalized_expected = (
        (expected,)
        if isinstance(expected, str) and expected in {"a", "b", "c"}
        else expected
    )
    assert normalize_human_selection(value) == normalized_expected


def test_terminal_arena_validates_case_insensitive_input_and_saves(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    answers = iter(["invalid", "C", "NONE OF THE ABOVE"])
    output_lines: list[str] = []

    result = run_terminal_preferences(
        experiment,
        input_fn=lambda prompt: answers.__next__(),
        output_fn=output_lines.append,
        terminal_width=120,
    )
    records = [
        json.loads(line) for line in result.output_path.read_text().splitlines()
    ]

    assert result.is_complete
    assert [
        record["evaluation"]["details"]["selected_labels"] for record in records
    ] == [
        ["c"],
        [],
    ]
    assert any("Invalid choice" in line for line in output_lines)


def test_terminal_arena_resumes_from_durable_vote_file(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    ballot = prepare_preference_ballot(experiment, seed=19)
    output = default_preference_path(ballot)
    append_human_preference(
        output,
        ballot=ballot,
        item=ballot.items[0],
        choice=("b",),
    )
    output_lines: list[str] = []

    result = run_terminal_preferences(
        experiment,
        seed=19,
        input_fn=lambda prompt: "A",
        output_fn=output_lines.append,
    )

    assert result.is_complete
    assert len(output.read_text().splitlines()) == 2
    assert output_lines[0] == "Resuming saved ballot at comparison 2/2."


def test_prefer_cli_shows_one_arena_and_never_reveals_model_ids(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    output = tmp_path / "terminal-votes.jsonl"

    result = CliRunner().invoke(
        app,
        ["prefer", "--experiment", str(experiment), "--output", str(output)],
        input="A B\nN\n",
        color=True,
        env={"TERM": "xterm-256color"},
    )

    assert result.exit_code == 0
    assert "HUMAN PREFERENCE  3 anonymous answers" in result.output
    assert result.output.count("ANSWER A") >= 2
    assert "ANSWER C" in result.output
    assert "first-model" not in result.output
    assert "second-model" not in result.output
    assert "third-model" not in result.output
    assert "wins" not in result.output.casefold()
    assert "\x1b[" in result.output
    first_vote = json.loads(output.read_text().splitlines()[0])
    assert first_vote["evaluation"]["details"]["selected_labels"] == ["a", "b"]


def _write_matrix_config(tmp_path: Path, model_ids: tuple[str, ...]) -> Path:
    models = "\n".join(
        f"""  - id: {model_id}
    backend: llama_cpp
    model_path: {tmp_path / f'{model_id}.gguf'}"""
        for model_id in model_ids
    )
    path = tmp_path / "matrix.yaml"
    path.write_text(
        f"""
schema_version: 1
benchmark:
  name: cli-matrix-test
  workload_path: {SUITE_PATH}
  output_root: {tmp_path / 'runs'}
  seed: 42
models:
{models}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_benchmark_cli_runs_one_three_answer_arena_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ids = ("first", "second", "third")
    experiment = _write_experiment(tmp_path, model_ids)
    config_path = _write_matrix_config(tmp_path, model_ids)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "llm_workload_benchmark.cli.run_matrix",
        lambda *args, **kwargs: experiment,
    )

    def fake_terminal_ballot(experiment_directory, *, model_ids, **kwargs):
        calls.append(model_ids)
        return TerminalPreferenceResult(tmp_path / "vote.jsonl", 1, 1)

    monkeypatch.setattr(
        "llm_workload_benchmark.cli.run_terminal_preferences",
        fake_terminal_ballot,
    )

    result = CliRunner().invoke(app, ["benchmark", "--config", str(config_path)])

    assert result.exit_code == 0
    assert calls == [model_ids]
    assert "HUMAN PREFERENCE  3 anonymous answers" in result.output
    assert "PAIR" not in result.output


def test_benchmark_cli_can_skip_interactive_human_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ids = ("first", "second", "third")
    experiment = _write_experiment(tmp_path, model_ids)
    config_path = _write_matrix_config(tmp_path, model_ids)
    monkeypatch.setattr(
        "llm_workload_benchmark.cli.run_matrix",
        lambda *args, **kwargs: experiment,
    )

    result = CliRunner().invoke(
        app,
        ["benchmark", "--config", str(config_path), "--skip-human-eval"],
    )

    assert result.exit_code == 0
    assert "HUMAN PREFERENCE" not in result.output
