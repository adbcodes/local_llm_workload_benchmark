import json
import subprocess
from pathlib import Path

import yaml

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite


ROOT = Path(__file__).resolve().parents[1]


def test_matrix_launchers_expose_run_resume_and_status() -> None:
    launchers = {
        "run_workloads.sh": "configs/final_workloads_matrix.yaml",
        "run_retrieval.sh": "configs/final_retrieval_matrix.yaml",
        "run_grounded_compression.sh": "configs/final_grounded_compression_matrix.yaml",
    }
    for name, config_path in launchers.items():
        path = ROOT / "scripts" / "matrices" / name
        assert path.stat().st_mode & 0o111
        assert config_path in path.read_text(encoding="utf-8")
        completed = subprocess.run(
            ["bash", str(path), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--resume [EXPERIMENT]" in completed.stdout
        assert "--status [EXPERIMENT]" in completed.stdout


def test_matrix_status_prints_every_model_quantization_bar(tmp_path: Path) -> None:
    (tmp_path / "experiment.json").write_text(
        json.dumps({"status": "running", "models": []}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "matrices" / "run_workloads.sh"),
            "--status",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    model_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("[")
    ]
    assert len(model_lines) == 20
    assert all("[------------------------]" in line for line in model_lines)
    assert "llama-3.1-8b           Q8_0" in completed.stdout
    assert "qwen2.5-coder-7b       Q3_K_M" in completed.stdout
    assert "Overall:" in completed.stdout


def test_final_quantization_profiles_are_independent_and_complete() -> None:
    workloads = load_config(ROOT / "configs" / "final_workloads_matrix.yaml")
    retrieval = load_config(ROOT / "configs" / "final_retrieval_matrix.yaml")
    compression = load_config(
        ROOT / "configs" / "final_grounded_compression_matrix.yaml"
    )

    assert len(workloads.models) == len(retrieval.models) == len(compression.models) == 20
    assert workloads.benchmark.workload_path == Path("data/suites/final_workloads.yaml")
    assert retrieval.benchmark.workload_path == Path(
        "data/suites/final_retrieval.yaml"
    )
    assert compression.benchmark.workload_path == Path(
        "data/suites/grounded_compression.yaml"
    )
    assert workloads.benchmark.repetitions == retrieval.benchmark.repetitions == (
        compression.benchmark.repetitions
    ) == 1
    for profile in (workloads, retrieval, compression):
        assert profile.judge is not None
        assert profile.judge.provider == "cerebras"
        assert profile.judge.requests_per_minute == 5
        assert profile.judge.requests_per_hour == 150

    expected_models = {
        (architecture, quantization)
        for architecture in (
            "llama-3.1-8b",
            "qwen3-8b",
            "mistral-7b-v0.3",
            "granite-3.3-8b",
            "qwen2.5-coder-7b",
        )
        for quantization in ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M")
    }
    assert {(model.architecture, model.quantization) for model in workloads.models} == (
        expected_models
    )
    assert [model.model_dump() for model in retrieval.models] == [
        model.model_dump() for model in workloads.models
    ]
    assert [model.model_dump() for model in compression.models] == [
        model.model_dump() for model in workloads.models
    ]


def test_final_profiles_match_pinned_model_sources() -> None:
    config = load_config(ROOT / "configs" / "final_workloads_matrix.yaml")
    sources = yaml.safe_load(
        (ROOT / "data" / "model_sources.yaml").read_text(encoding="utf-8")
    )
    pinned = {
        (family["architecture"], file["quantization"], file["local_path"])
        for family in sources["families"].values()
        for file in family["files"]
    }

    assert len(sources["families"]) == 5
    assert len(pinned) == 20
    assert {
        (model.architecture, model.quantization, str(model.model_path))
        for model in config.models
    } == pinned


def test_split_suites_partition_the_frozen_deterministic_seven() -> None:
    frozen = load_suite(ROOT / "data" / "suites" / "final_deterministic.yaml")
    workloads = load_suite(ROOT / "data" / "suites" / "final_workloads.yaml")
    retrieval = load_suite(ROOT / "data" / "suites" / "final_retrieval.yaml")

    assert set(workloads.items) == {
        "applied_reasoning",
        "code_debug_repair",
        "messy_text_to_schema",
        "constraint_load_curve",
        "tool_use",
        "email_to_action",
    }
    assert set(retrieval.items) == {"long_text_retrieval"}
    assert set(workloads.items).isdisjoint(retrieval.items)
    assert set(frozen.items) == set(workloads.items) | set(retrieval.items)
    assert sum(map(len, workloads.items.values())) == 312
    assert sum(map(len, retrieval.items.values())) == 48
    assert sum(map(len, frozen.items.values())) == 360


def test_three_final_profiles_define_7800_generations() -> None:
    attempts = 0
    for config_name, suite_name in (
        ("final_workloads_matrix.yaml", "final_workloads.yaml"),
        ("final_retrieval_matrix.yaml", "final_retrieval.yaml"),
        ("final_grounded_compression_matrix.yaml", "grounded_compression.yaml"),
    ):
        config = load_config(ROOT / "configs" / config_name)
        suite = load_suite(ROOT / "data" / "suites" / suite_name)
        attempts += len(config.models) * sum(map(len, suite.items.values()))

    assert attempts == 7_800
