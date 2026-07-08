#!/usr/bin/env bash
set -euo pipefail

matrix_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

matrix_find_experiment() {
  local config_path="$1"
  "$matrix_python" - "$config_path" "$matrix_project_root/runs" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1]).resolve()
runs_root = Path(sys.argv[2]).resolve()
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
matches: list[Path] = []
for index_path in runs_root.glob("*/experiment.json"):
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    source = index.get("config_source")
    if isinstance(source, dict) and source.get("sha256") == config_hash:
        matches.append(index_path.parent)
if not matches:
    raise SystemExit(
        f"No experiment matches the current config hash for {config_path}."
    )
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
}

matrix_show_status() {
  local config_path="$1"
  local experiment="$2"
  "$matrix_python" - "$config_path" "$experiment" "$matrix_project_root" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from llm_workload_benchmark.config import load_config
from llm_workload_benchmark.dataset import load_suite

config_path = Path(sys.argv[1]).resolve()
experiment = Path(sys.argv[2]).resolve()
root = Path(sys.argv[3]).resolve()
index_path = experiment / "experiment.json"
try:
    index = json.loads(index_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Invalid experiment index {index_path}: {error}")

config = load_config(config_path)
suite_path = config.benchmark.workload_path
if not suite_path.is_absolute():
    suite_path = root / suite_path
suite = load_suite(suite_path)
items_per_model = sum(len(items) for items in suite.items.values()) * config.benchmark.repetitions
indexed = {
    entry.get("model_id"): entry
    for entry in index.get("models", [])
    if isinstance(entry, dict)
}

def result_count(model_id: str) -> int:
    path = experiment / "models" / model_id / "results.jsonl"
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())

def bar(completed: int, total: int, width: int = 24) -> str:
    fraction = completed / total if total else 0.0
    filled = round(width * min(1.0, max(0.0, fraction)))
    return "#" * filled + "-" * (width - filled)

print(f"Experiment: {experiment}")
print(f"State: {index.get('status', 'unknown')} | {config.benchmark.name}")
overall_completed = 0
for model in config.models:
    if not model.enabled:
        continue
    entry = indexed.get(model.id, {})
    completed = min(result_count(model.id), items_per_model)
    state = str(entry.get("status") or ("running" if completed else "pending"))
    if state == "completed":
        completed = items_per_model
    overall_completed += completed
    fraction = completed / items_per_model if items_per_model else 0.0
    print(
        f"[{bar(completed, items_per_model)}] {fraction:6.1%}  "
        f"{model.architecture or model.id:<22} {model.quantization or '-':<7} "
        f"{completed:>3}/{items_per_model:<3} {state}"
    )
overall_total = items_per_model * sum(model.enabled for model in config.models)
overall_fraction = overall_completed / overall_total if overall_total else 0.0
print(
    f"Overall: [{bar(overall_completed, overall_total)}] "
    f"{overall_fraction:6.1%} {overall_completed}/{overall_total} generations"
)
PY
}

run_matrix_profile() {
  local config_relative="$1"
  shift
  cd "$matrix_project_root"

  matrix_python="$matrix_project_root/.venv/bin/python"
  local benchmark_bin="$matrix_project_root/.venv/bin/llm-benchmark"
  if [[ ! -x "$matrix_python" || ! -x "$benchmark_bin" ]]; then
    echo "Error: project virtual environment is not ready; run uv sync first." >&2
    exit 1
  fi

  if [[ -f "$matrix_project_root/.env" ]]; then
    set -a
    source "$matrix_project_root/.env"
    set +a
  fi

  local config_path="$matrix_project_root/$config_relative"
  local action="${1:---run}"
  local experiment=""
  case "$action" in
    --run)
      if [[ $# -ne 0 && $# -ne 1 ]]; then
        echo "Usage: $0 [--run | --resume [EXPERIMENT] | --status [EXPERIMENT]]" >&2
        exit 2
      fi
      exec "$benchmark_bin" benchmark --config "$config_path"
      ;;
    --resume|--status)
      if [[ $# -gt 2 ]]; then
        echo "Usage: $0 [--run | --resume [EXPERIMENT] | --status [EXPERIMENT]]" >&2
        exit 2
      fi
      if [[ $# -eq 2 ]]; then
        experiment="$2"
      else
        experiment="$(matrix_find_experiment "$config_path")"
      fi
      if [[ "$action" == "--status" ]]; then
        matrix_show_status "$config_path" "$experiment"
      else
        matrix_show_status "$config_path" "$experiment"
        exec "$benchmark_bin" benchmark \
          --config "$config_path" \
          --resume-experiment "$experiment"
      fi
      ;;
    --help|-h)
      echo "Usage: $0 [--run | --resume [EXPERIMENT] | --status [EXPERIMENT]]"
      echo "  --run                 Start a new matrix (default)."
      echo "  --resume [EXPERIMENT] Resume the newest matching matrix or the given path."
      echo "  --status [EXPERIMENT] Show per-model and per-quantization progress bars."
      ;;
    *)
      echo "Unknown option: $action" >&2
      exit 2
      ;;
  esac
}
