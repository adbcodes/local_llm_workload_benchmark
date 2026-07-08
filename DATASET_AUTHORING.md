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
uv run llm-benchmark dataset build --suite data/suites/final_deterministic.yaml
uv run llm-benchmark dataset build --suite data/suites/grounded_compression.yaml
```

The command rewrites only JSONL files whose YAML source changed. `run` and
`benchmark` perform the same build automatically before model inference.

CI can verify that committed JSONL is current without modifying it:

```bash
uv run llm-benchmark dataset build \
  --suite data/suites/final_deterministic.yaml \
  --check
```

## Choose what runs

Set `benchmark.workload_path` in a model config to one of the suite manifests:

- `data/suites/reasoning.yaml`
- `data/suites/structured.yaml`
- `data/suites/instruction.yaml`
- `data/suites/coding.yaml`
- `data/suites/grounded_compression.yaml`
- `data/suites/smoke.yaml`
- `data/suites/final_workloads.yaml`
- `data/suites/final_retrieval.yaml`
- `data/suites/final_deterministic.yaml`

A suite selects benchmark files and may optionally filter item IDs,
subcategories, difficulties, splits, or review statuses. The smoke suite shows
an explicit-ID example.

Run the final evidence as three independent matrices:

```bash
uv run llm-benchmark benchmark \
  --config configs/final_workloads_matrix.yaml
uv run llm-benchmark benchmark \
  --config configs/final_retrieval_matrix.yaml
uv run llm-benchmark benchmark \
  --config configs/final_grounded_compression_matrix.yaml
```

The main workload matrix contains 312 questions across six benchmarks,
including email-to-action inbox routing. Retrieval contains 48 questions and
grounded compression contains 30. All three use the same 20 model/quantization
configurations; grounded compression additionally uses the configured Cerebras
judge.

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
