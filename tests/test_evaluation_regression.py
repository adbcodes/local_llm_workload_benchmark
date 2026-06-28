import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_workload_benchmark.cli import app
from llm_workload_benchmark.evaluation_regression import (
    RegressionCorpusError,
    load_regression_corpus,
    replay_regression_corpus,
)


CORPUS_PATH = Path("tests/fixtures/evaluation/failure_regressions_v1.jsonl")
SUITE_PATH = Path("data/suites/all.yaml")
SOURCE_EXPERIMENT = "2026-07-27_14-27-42-rejudged-1e039cd6"


def test_failure_corpus_covers_real_failures_and_controls() -> None:
    cases = load_regression_corpus(CORPUS_PATH)

    assert len(cases) == 30
    assert {case.source.experiment_id for case in cases} == {SOURCE_EXPERIMENT}
    assert {case.failure_class for case in cases} >= {
        "false_reject",
        "false_accept",
        "partial_answer",
        "protocol_wrapper",
        "malformed_output",
        "truncated_output",
        "genuine_model_error",
        "passing_control",
    }
    tags = {tag for case in cases for tag in case.tags}
    assert tags >= {
        "option_label",
        "casefold",
        "currency",
        "confidence",
        "lambda",
        "paraphrase",
        "hallucination",
        "truncation",
        "model_cleanup",
        "markdown_fence",
        "tool_trace",
        "partial_credit",
        "option_text",
        "set_match",
        "surrounding_text",
        "raw_protocol",
        "timeout",
        "false_premise",
        "over_refusal",
    }
    assert {case.expected_result.semantic_outcome for case in cases} == {
        "correct",
        "incorrect",
        "not_scored",
    }
    assert {case.expected_result.protocol_outcome for case in cases} == {
        "compliant",
        "noncompliant",
    }
    assert {case.expected_result.integration_outcome for case in cases} >= {
        "scored_cleanly",
        "scored_after_recovery",
        "unparseable",
    }


def test_failure_corpus_reproduces_the_frozen_legacy_baseline() -> None:
    summary = replay_regression_corpus(CORPUS_PATH, SUITE_PATH)

    assert summary.total == 30
    assert summary.baseline_reproduced == 30
    assert summary.known_target_gaps == 24
    assert summary.unexpected_case_ids == ()
    assert all(case.baseline_matches for case in summary.cases)


def test_failure_corpus_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    first = CORPUS_PATH.read_text(encoding="utf-8").splitlines()[0]
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(first + "\n" + first + "\n", encoding="utf-8")

    with pytest.raises(RegressionCorpusError, match="duplicate regression case id"):
        load_regression_corpus(duplicate_path)


def test_replay_stops_when_a_referenced_item_contract_changes(tmp_path: Path) -> None:
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8").splitlines()[0])
    raw["source"]["scoring_contract_sha256"] = "0" * 64
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(RegressionCorpusError, match="scoring contract changed"):
        replay_regression_corpus(changed_path, SUITE_PATH)


def test_replay_cli_is_read_only() -> None:
    before = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "regression",
            "replay",
            "--corpus",
            str(CORPUS_PATH),
            "--suite",
            str(SUITE_PATH),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Cases: 30" in result.output
    assert "Baseline reproduced: 30" in result.output
    assert "Known target gaps: 24" in result.output
    assert "Unexpected results: 0" in result.output
    assert hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() == before


def test_temporary_corpus_is_not_registered_as_benchmark_data() -> None:
    catalog = yaml.safe_load(Path("data/catalog.yaml").read_text(encoding="utf-8"))

    assert CORPUS_PATH.is_relative_to(Path("tests/fixtures"))
    assert "failure_regressions" not in catalog["evaluation_files"]
