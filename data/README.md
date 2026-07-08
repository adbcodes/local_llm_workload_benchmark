# Benchmark data

The repository retains only the evidence tracks in the implementation plan:

- seven frozen deterministic Stage 1 benchmarks (360 questions total); and
- a 30-item `grounded_compression` set for the separate Stage 2 judged
  experiment.

Generated `items.jsonl` files come from their authoritative generator or
question source. Do not edit generated questions or gold answers directly.

## Build and validate

```bash
uv run llm-benchmark dataset build --suite data/suites/final_six.yaml
uv run llm-benchmark dataset build --suite data/suites/judged.yaml
uv run llm-benchmark dataset validate --catalog data/catalog.yaml
```

Stage 1 execution is intentionally split:

- `configs/final_default_matrix.yaml`: six non-retrieval workloads, 6,240
  generations;
- `configs/final_retrieval_matrix.yaml`: long-text retrieval, 960 generations.

Both use the same five model families, four quantizations, generation settings,
and evaluator freeze. The split prevents a retrieval-only correction or failure
from invalidating the other six workloads. Together they contain 7,200
generations.
