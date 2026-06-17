# Local LLM Workload Benchmark

A reproducible benchmark for comparing local, open-weight language models on a
fixed practical workload. The project will measure output quality, structured
output reliability, latency, memory use, and serving behavior.

Benchmark results are specific to the workload, model files, model settings,
and hardware used for a run. The benchmark method is reusable across machines;
the measured numbers are not universal model rankings.

## Current milestone

The repository currently provides the project foundation and experiment run
tracking. It can:

- load and validate a YAML benchmark configuration;
- create a unique directory for an experiment run; and
- write a `manifest.json` receipt containing the resolved configuration and
  environment details.

Model inference, workload examples, and scoring are intentionally added in
later milestones.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the project environment and install its dependencies:

```bash
uv sync
```

## Create a run

```bash
uv run llm-benchmark create-run --config configs/benchmark.example.yaml
```

The command prints the new run directory. Its manifest is stored at:

```text
runs/<timestamp>-<short-id>/manifest.json
```

This command only records the planned experiment. It does not run a model yet.

## Run tests

```bash
uv run pytest
```

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the locked project
scope and incremental build plan.

See [`docs/CURRENT_IMPLEMENTATION.md`](docs/CURRENT_IMPLEMENTATION.md) for a
detailed explanation of the code, project files, and Python concepts currently
implemented.
