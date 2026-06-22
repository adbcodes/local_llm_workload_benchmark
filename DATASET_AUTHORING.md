# Dataset authoring

Questions are edited in YAML directly inside each benchmark directory. The
benchmark runtime continues to read the generated `items.jsonl` beside them.

```text
data/
  suites/
    reasoning.yaml
  applied_reasoning/
    benchmark.yaml
    external.yaml
    generated.yaml
    items.jsonl
```

`benchmark.yaml` lists its `authoring_paths`. A benchmark may use as many
subcategory shards as needed.

## Build and validate

```bash
uv run llm-benchmark dataset build --suite data/suites/all.yaml
```

The command rewrites only JSONL files whose YAML source changed. `run` and
`benchmark` perform the same build automatically before model inference.

CI can verify that committed JSONL is current without modifying it:

```bash
uv run llm-benchmark dataset build \
  --suite data/suites/all.yaml \
  --check
```

## Choose what runs

Set `benchmark.workload_path` in a model config to one of the suite manifests:

- `data/suites/reasoning.yaml`
- `data/suites/structured.yaml`
- `data/suites/instruction.yaml`
- `data/suites/coding.yaml`
- `data/suites/judged.yaml`
- `data/suites/smoke.yaml`
- `data/suites/all.yaml`

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

## Planned benchmarks

`data/catalog.yaml` is the navigation index for all 20 planned
benchmarks. Categories whose evaluators do not exist yet use a `draft.yaml`
with one starter item. These drafts are intentionally excluded from runnable
suites until their evaluator and runtime schema are implemented.

The quantization and laptop-value benchmarks are derived comparisons, so their
YAML files reference `data/suites/all.yaml` instead of duplicating questions.

## Applied Reasoning sources

Applied Reasoning combines 24 MIT-licensed anchor questions with 24 fresh,
seeded generator outputs. Item provenance records source IDs, licences, and
content hashes. Third-party notices live beside the benchmark.

Regenerate the fresh half with:

```bash
uv run python scripts/generate_applied_reasoning.py
```

The licensed half is materialized in YAML. Its importer accepts a MATH parquet
file and a local BIG-Bench Hard checkout so the exact selected source records
can be audited and reproduced.
