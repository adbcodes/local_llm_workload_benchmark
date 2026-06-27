#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${RUNS_DIR:-$ROOT/runs}"
STATE="$RUNS/final-pipeline"
LOCK="${LOCK_DIR:-$ROOT/.final-pipeline.lock}"
EVENT_LOG="$STATE/events.log"
PIPELINE_TOTAL=6
PIPELINE_STARTED="$(date +%s)"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-configs/final_default_matrix.yaml}"
TEMPERATURE_CONFIG="${TEMPERATURE_CONFIG:-configs/final_temperature_matrix.yaml}"
CONSTRAINED_CONFIG="${CONSTRAINED_CONFIG:-configs/final_constrained_matrix.yaml}"
REPETITION_CONFIG="${REPETITION_CONFIG:-configs/final_repetition_matrix.yaml}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/final_context_matrix.yaml}"
MODE="resume"

for argument in "$@"; do
  case "$argument" in
    --resume) MODE="resume" ;;
    --fresh) MODE="fresh" ;;
    --status) MODE="status" ;;
    -h|--help)
      echo "Usage: bash scripts/run_final_pipeline.sh [--resume|--fresh|--status]"
      echo "  --resume  Reuse completed matrices and resume interrupted models (default)."
      echo "  --fresh   Delete generated runs and rerun the entire pipeline."
      echo "  --status  Show saved progress without running anything."
      exit 0
      ;;
    *)
      echo "Error: unknown option '$argument'" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
mkdir -p "$RUNS" "$STATE"

format_duration() {
  local total="${1%.*}"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  if (( hours > 0 )); then
    printf '%d:%02d:%02d' "$hours" "$minutes" "$seconds"
  else
    printf '%02d:%02d' "$minutes" "$seconds"
  fi
}

progress_bar() {
  local current="$1"
  local total="$2"
  local width="${3:-24}"
  local filled=0 percent=0 empty
  if (( total > 0 )); then
    filled=$((current * width / total))
    percent=$((current * 100 / total))
  fi
  (( filled > width )) && filled="$width"
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%%' "$percent"
}

log_event() {
  local level="$1"
  shift
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$level" "$*" \
    >> "$EVENT_LOG"
}

die() {
  log_event ERROR "$*"
  echo "Error: $*" >&2
  exit 1
}

release_lock() {
  local exit_code=$?
  rm -f "$LOCK/pid"
  rmdir "$LOCK" 2>/dev/null || true
  if (( exit_code != 0 )); then
    log_event STOPPED "pipeline exited with status $exit_code"
  fi
}

acquire_lock() {
  if ! mkdir "$LOCK" 2>/dev/null; then
    local owner="unknown"
    [[ -f "$LOCK/pid" ]] && owner="$(<"$LOCK/pid")"
    if [[ "$owner" =~ ^[0-9]+$ ]] && ! kill -0 "$owner" 2>/dev/null; then
      rm -f "$LOCK/pid"
      rmdir "$LOCK" 2>/dev/null || true
      mkdir "$LOCK" || die "could not replace stale pipeline lock"
    else
      die "another final pipeline is already running (PID $owner)"
    fi
  fi
  printf '%d\n' "$$" > "$LOCK/pid"
  trap release_lock EXIT
  trap 'exit 130' INT TERM
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command '$1' was not found"
}

config_identity() {
  jq -r '
    if (.config_source | type) == "object" then
      [(.config_source.path // ""), (.config_source.sha256 // "")]
    elif (.config_source | type) == "string" then
      [.config_source, ""]
    else
      ["", ""]
    end | @tsv
  ' "$1"
}

latest_compatible_experiment() {
  local config="$1"
  local config_name="${config##*/}"
  local expected_hash directory index source saved_hash
  expected_hash="$(shasum -a 256 "$config" | awk '{print $1}')"

  while IFS= read -r directory; do
    index="$directory/experiment.json"
    [[ -f "$index" ]] || continue
    IFS=$'\t' read -r source saved_hash < <(config_identity "$index")
    [[ "$source" == */"$config_name" ]] || continue
    # A hashless legacy manifest cannot prove that its config still matches.
    [[ -n "$saved_hash" && "$saved_hash" == "$expected_hash" ]] || continue
    printf '%s\n' "$directory"
    return 0
  done < <(find "$RUNS" -mindepth 1 -maxdepth 1 -type d -name '*-matrix-*' \
    -print | sort -r)

  return 1
}

experiment_status() {
  jq -r '.status // "unknown"' "$1/experiment.json"
}

show_status() {
  local label config directory status completed total
  printf '%-13s %-16s %-10s %s\n' "MATRIX" "STATUS" "MODELS" "RUN"
  while IFS=$'\t' read -r label config; do
    if directory="$(latest_compatible_experiment "$config")"; then
      status="$(experiment_status "$directory")"
      completed="$(jq -r '.models_completed // 0' "$directory/experiment.json")"
      total="$(jq -r '.models_total // 0' "$directory/experiment.json")"
      printf '%-13s %-16s %-10s %s\n' \
        "$label" "$status" "$completed/$total" "$directory"
    else
      printf '%-13s %-16s %-10s %s\n' "$label" "not_started" "0/?" "-"
    fi
  done < <(
    printf '%s\t%s\n' \
      default "$DEFAULT_CONFIG" \
      temperature "$TEMPERATURE_CONFIG" \
      constrained "$CONSTRAINED_CONFIG" \
      repetition "$REPETITION_CONFIG" \
      context "$CONTEXT_CONFIG"
  )
}

fresh_reset() {
  echo "Fresh mode: deleting generated run data and restarting every matrix."
  find "$RUNS" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
  mkdir -p "$STATE"
}

show_pipeline_progress() {
  local current="$1"
  local label="$2"
  local elapsed=$(( $(date +%s) - PIPELINE_STARTED ))
  printf '\nPIPELINE '
  progress_bar "$current" "$PIPELINE_TOTAL"
  printf '  %d/%d complete | elapsed %s | %s\n' \
    "$current" "$PIPELINE_TOTAL" "$(format_duration "$elapsed")" "$label"
}

run_experiment() {
  local label="$1"
  local config="$2"
  local directory status started elapsed

  local resume_directory=""
  if directory="$(latest_compatible_experiment "$config")"; then
    status="$(experiment_status "$directory")"
    case "$status" in
      completed)
        echo "[$label] Reusing matching completed run: $directory"
        log_event REUSED "$label $directory"
        ;;
      running|failed|partial_failure|interrupted)
        resume_directory="$directory"
        echo "[$label] Resuming $status run at the next unfinished model: $directory"
        log_event RESUMING "$label status=$status path=$directory"
        ;;
      *) die "$label has an unsupported saved status '$status': $directory" ;;
    esac
  fi

  if [[ -z "${directory:-}" || -n "$resume_directory" ]]; then
    local benchmark_args=(
      run --offline llm-benchmark benchmark
      --config "$config"
      --skip-human-eval
    )
    if [[ -n "$resume_directory" ]]; then
      benchmark_args+=(--resume-experiment "$resume_directory")
    fi
    if [[ -z "$resume_directory" ]]; then
      directory=""
    fi
    echo "[$label] Starting $config"
    log_event STARTED "$label $config"
    started="$(date +%s)"

    # Keep stdout attached to the terminal so the Python progress renderer can
    # update one line in place. Save stderr separately for diagnostics.
    if ! UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
      uv "${benchmark_args[@]}" 2> "$STATE/$label.stderr.log"; then
      cat "$STATE/$label.stderr.log" >&2
      die "$label benchmark command failed; see $STATE/$label.stderr.log"
    fi

    if [[ -z "$directory" ]]; then
      directory="$(latest_compatible_experiment "$config")" || \
        die "$label finished without a compatible saved experiment"
    fi
    status="$(experiment_status "$directory")"
    [[ "$status" == "completed" ]] || \
      die "$label command returned but experiment status is '$status': $directory"
    elapsed=$(( $(date +%s) - started ))
    log_event COMPLETED "$label elapsed_seconds=$elapsed path=$directory"
  fi

  printf '%s\n' "$directory" > "$STATE/$label.path"
  RUN_RESULT="$directory"
  echo "[$label] Completed: $directory"
}

show_timing() {
  local label="$1"
  local directory="$2"
  local seconds
  seconds="$(jq -r '.elapsed_seconds // 0' "$directory/experiment.json")"
  printf '%-12s %10s  %s\n' "$label" "$(format_duration "$seconds")" "$directory"
}

require_command jq
require_command shasum

if [[ "$MODE" == "status" ]]; then
  show_status
  exit 0
fi

require_command uv
acquire_lock
if [[ "$MODE" == "fresh" ]]; then
  fresh_reset
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

[[ -n "${GROQ_API_KEY:-}" ]] || die "GROQ_API_KEY is missing; add it to .env"

: > "$EVENT_LOG"
log_event STARTED "final pipeline pid=$$"

echo "Preflight: validating datasets and final profile tests"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  uv run --offline llm-benchmark dataset validate --catalog data/catalog.yaml
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  uv run --offline pytest -q \
  tests/test_final_profiles.py tests/test_artifacts.py tests/test_final_figures.py
log_event COMPLETED "preflight"

RUN_RESULT=""
show_pipeline_progress 0 "starting default matrix"
run_experiment default "$DEFAULT_CONFIG"
default_run="$RUN_RESULT"

show_pipeline_progress 1 "default complete; starting temperature"
run_experiment temperature "$TEMPERATURE_CONFIG"
temperature_run="$RUN_RESULT"

show_pipeline_progress 2 "temperature complete; starting constrained decoding"
run_experiment constrained "$CONSTRAINED_CONFIG"
constrained_run="$RUN_RESULT"

show_pipeline_progress 3 "constrained decoding complete; starting repetition penalty"
run_experiment repetition "$REPETITION_CONFIG"
repetition_run="$RUN_RESULT"

show_pipeline_progress 4 "repetition penalty complete; starting context"
run_experiment context "$CONTEXT_CONFIG"
context_run="$RUN_RESULT"

show_pipeline_progress 5 "all matrices complete; generating figures"
echo "Generating final figures"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  uv run --offline llm-benchmark figures \
  --default-experiment "$default_run" \
  --temperature-experiment "$temperature_run" \
  --constrained-experiment "$constrained_run" \
  --repetition-experiment "$repetition_run" \
  --context-experiment "$context_run"

figure_manifest="$default_run/artifacts/final_figures/manifest.json"
[[ -f "$figure_manifest" ]] || die "figure command did not create $figure_manifest"
printf '%s\n' "$figure_manifest" > "$STATE/figures.path"
log_event COMPLETED "figures path=$figure_manifest"
show_pipeline_progress 6 "figures generated"

echo
echo "Completed experiment timings"
show_timing default "$default_run"
show_timing temperature "$temperature_run"
show_timing constrained "$constrained_run"
show_timing repetition "$repetition_run"
show_timing context "$context_run"
echo
echo "Final figure manifest: $figure_manifest"
echo "Event log: $EVENT_LOG"
echo "Whole pipeline wall time: $(format_duration "$(( $(date +%s) - PIPELINE_STARTED ))")"
log_event COMPLETED "final pipeline"
