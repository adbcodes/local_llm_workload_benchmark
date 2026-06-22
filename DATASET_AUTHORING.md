# Dataset authoring

Questions are edited in YAML under each benchmark's `authoring/` directory.
The benchmark runtime continues to read the generated `items.jsonl` beside it.

```text
data/benchmarks/v1/
  applied_reasoning/
    benchmark.yaml
    authoring/
      arithmetic.yaml
      calendar_math.yaml
    items.jsonl
```

`benchmark.yaml` lists its `authoring_paths`. A benchmark may use as many
subcategory shards as needed.

## Build and validate

```bash
uv run llm-benchmark dataset build --suite data/benchmarks/v1/all_suite.yaml
```

The command rewrites only JSONL files whose YAML source changed. `run` and
`benchmark` perform the same build automatically before model inference.

CI can verify that committed JSONL is current without modifying it:

```bash
uv run llm-benchmark dataset build \
  --suite data/benchmarks/v1/all_suite.yaml \
  --check
```

## Choose what runs

Set `benchmark.workload_path` in a model config to one of the suite manifests:

- `reasoning_suite.yaml`
- `structured_suite.yaml`
- `instruction_suite.yaml`
- `coding_suite.yaml`
- `judged_suite.yaml`
- `smoke_suite.yaml`
- `all_suite.yaml`

A suite selects benchmark files and may optionally filter item IDs,
subcategories, difficulties, splits, or review statuses. The smoke suite shows
an explicit-ID example.

Run every implemented benchmark with:

```bash
uv run llm-benchmark benchmark \
  --config configs/all_matrix.yaml \
  --skip-human-eval
```

The all-suite includes an LLM-judged benchmark, so it requires `GROQ_API_KEY`.
