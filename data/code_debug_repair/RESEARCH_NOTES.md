# Code, Debug, and Repair research notes

The active benchmark is generated from locally authored scenario specifications,
reference implementations, and code-computed expected values. The former classic
algorithm set remains inactive under `archive/` as a contamination reference;
the active authoring source is `generated_questions.yaml`.

## Design

Implementation questions combine ordinary programming operations inside fresh
record-processing, reconciliation, scheduling, dependency, and event-processing
contracts. Changing names alone is not considered a new task. Each active
implementation changes the representation, operation path, boundary rules, or
tie-breaking behavior from recognizable practice-platform templates.

`scripts/generate_coding_benchmark.py` computes every executable gold by running
the stored reference solution. Mutable inputs are marked with `preserve_args`,
so a correct return value cannot hide an input-mutation violation.

## Composition

| Task family | Count | Evaluation |
|---|---:|---|
| Fresh practical implementation | 30 | Restricted Python tests and postconditions |
| Failure-trace diagnosis | 10 | Exact diagnostic category |
| Generated code repair | 8 | Restricted Python tests and mutant killing |
| **Total** | **48** | |

Public and held-out visibility are balanced 24/24. Development scenarios are
used for calibration; held-out scenarios must not be selected because a
particular model or quantization happened to fail them.
