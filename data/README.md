# Benchmark data

The repository retains only the evidence tracks in the implementation plan:

- six frozen deterministic Stage 1 benchmarks (320 questions total); and
- `grounded_compression`, which will be audited before the separate Stage 2
  judged experiment.

Generated `items.jsonl` files come from their authoritative generator or
question source. Do not edit generated questions or gold answers directly.

## Build and validate

```bash
uv run llm-benchmark dataset build --suite data/suites/final_six.yaml
uv run llm-benchmark dataset build --suite data/suites/judged.yaml
uv run llm-benchmark dataset validate --catalog data/catalog.yaml
```

Stage 1 execution is intentionally split:

- `configs/final_default_matrix.yaml`: five non-retrieval workloads, 5,440
  generations;
- `configs/final_retrieval_matrix.yaml`: long-text retrieval, 960 generations.

Both use the same five model families, four quantizations, generation settings,
and evaluator freeze. The split prevents a retrieval-only correction or failure
from invalidating the other five workloads.
