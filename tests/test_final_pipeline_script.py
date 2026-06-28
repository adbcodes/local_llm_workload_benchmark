from pathlib import Path
import json
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_final_pipeline.sh"


def test_final_pipeline_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_final_pipeline_script_runs_all_profiles_and_figures() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for config in (
        "final_default_matrix.yaml",
        "final_temperature_matrix.yaml",
        "final_constrained_matrix.yaml",
        "final_repetition_matrix.yaml",
        "final_context_matrix.yaml",
    ):
        assert config in source
    for option in (
        "--default-experiment",
        "--context-experiment",
    ):
        assert option in source
    assert "--skip-human-eval" in source
    assert "latest_compatible_experiment" in source
    assert "config_identity" in source
    assert "acquire_lock" in source
    assert "another final pipeline is already running" in source
    assert "progress_bar" in source
    assert "show_pipeline_progress" in source
    assert "events.log" in source
    assert "stderr.log" in source
    assert "Keep stdout attached to the terminal" in source
    assert "--resume-experiment" in source
    assert "REJUDGE_SOURCE_EXPERIMENT" in source
    assert "llm-benchmark rejudge" in source
    assert "--fresh" in source
    assert "--status" in source


def test_config_source_lookup_supports_string_and_object_formats(tmp_path: Path) -> None:
    jq_filter = """
      if (.config_source | type) == "object" then
        .config_source.path // ""
      elif (.config_source | type) == "string" then
        .config_source
      else
        ""
      end
    """
    for value in (
        "configs/final_default_matrix.yaml",
        {"path": "configs/final_default_matrix.yaml", "sha256": "abc"},
    ):
        index = tmp_path / "experiment.json"
        index.write_text(json.dumps({"config_source": value}), encoding="utf-8")
        result = subprocess.run(
            ["jq", "-r", jq_filter, str(index)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "configs/final_default_matrix.yaml"


def test_status_finds_a_compatible_rejudged_experiment(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "2026-01-01_00-00-00-rejudged-test"
    run.mkdir(parents=True)
    config = ROOT / "configs" / "final_default_matrix.yaml"
    config_hash = subprocess.run(
        ["shasum", "-a", "256", str(config)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "status": "interrupted",
                "models_completed": 3,
                "models_total": 12,
                "config_source": {
                    "path": str(config),
                    "sha256": config_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "RUNS_DIR": str(tmp_path / "runs"),
            "ENV_FILE": str(tmp_path / "no-project-env"),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--status"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "default       interrupted      3/12" in result.stdout
    assert str(run) in result.stdout


def test_final_pipeline_completes_with_tiny_fake_benchmarks(tmp_path: Path) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    fake_uv = bin_directory / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_UV_CALLS"

if [[ " $* " == *" llm-benchmark benchmark "* ]]; then
  config=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--config" ]]; then config="$argument"; break; fi
    previous="$argument"
  done
  name="$(basename "$config" .yaml)"
  directory="$RUNS_DIR/2026-01-01_00-00-00-matrix-$name"
  mkdir -p "$directory"
  hash="$(shasum -a 256 "$config" | awk '{print $1}')"
  jq -n --arg path "$config" --arg hash "$hash" \
    '{status:"completed", models_completed:1, models_total:1,
      elapsed_seconds:1, config_source:{path:$path, sha256:$hash}}' \
    > "$directory/experiment.json"
  exit 0
fi

if [[ " $* " == *" llm-benchmark figures "* ]]; then
  default_run=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--default-experiment" ]]; then
      default_run="$argument"; break
    fi
    previous="$argument"
  done
  mkdir -p "$default_run/artifacts/final_figures"
  printf '{}\n' > "$default_run/artifacts/final_figures/manifest.json"
fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "CEREBRAS_API_KEY": "test-key",
            "ENV_FILE": str(tmp_path / "no-project-env"),
            "PATH": f"{bin_directory}:{environment['PATH']}",
            "RUNS_DIR": str(tmp_path / "runs"),
            "LOCK_DIR": str(tmp_path / "pipeline.lock"),
            "FAKE_UV_CALLS": str(tmp_path / "uv-calls.log"),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "PIPELINE [########################] 100%" in result.stdout
    assert "Whole pipeline wall time:" in result.stdout
    state = tmp_path / "runs" / "final-pipeline"
    assert (state / "events.log").is_file()
    assert (state / "figures.path").is_file()
    for label in ("default", "temperature", "constrained", "repetition", "context"):
        assert (state / f"{label}.path").is_file()
    assert not Path(environment["LOCK_DIR"]).exists()

    temperature_run = Path((state / "temperature.path").read_text().strip())
    temperature_index = temperature_run / "experiment.json"
    interrupted = json.loads(temperature_index.read_text(encoding="utf-8"))
    interrupted.update(status="running", models_completed=1, models_total=2)
    temperature_index.write_text(json.dumps(interrupted), encoding="utf-8")

    resumed = subprocess.run(
        ["bash", str(SCRIPT), "--resume"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "Resuming running run" in resumed.stdout
    assert "--resume-experiment" in Path(environment["FAKE_UV_CALLS"]).read_text()

    status = subprocess.run(
        ["bash", str(SCRIPT), "--status"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert status.returncode == 0, status.stderr
    assert "default       completed" in status.stdout

    runs = Path(environment["RUNS_DIR"])
    (runs / ".gitkeep").touch()
    stale = runs / "stale-data"
    stale.mkdir()
    fresh = subprocess.run(
        ["bash", str(SCRIPT), "--fresh"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert fresh.returncode == 0, fresh.stderr
    assert "Fresh mode: deleting generated run data" in fresh.stdout
    assert not stale.exists()
    assert (runs / ".gitkeep").is_file()
