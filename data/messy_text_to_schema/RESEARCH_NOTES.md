# Messy Text to Schema: Phase 4 Review

## Capability and output contract

This benchmark measures whether a model can recover typed, nested records from
clean and messy business text. The semantic target is the declared JSON value.
Every prompt now shows the exact key and nesting skeleton, type annotations,
date/time formats, null behavior, object-versus-array cardinality, and numeric
normalization rules. Exact semantic equality is the pass criterion; leaf
accuracy remains the partial-credit diagnostic, while wrappers are reported as
protocol friction.

## Item decisions

The 48-item set retains the useful clean anchors and stronger noisy/nested
scenarios from the earlier review. Six weaker scenarios were replaced with
small operational artifacts: CLI output, an access-request email, deployment
logs with an operator comment, a support-ticket history, similar CMDB records
requiring an exact identifier match, and a revised purchase-order email/table
containing an untrusted instruction.

| Items | Decision | Phase 4 repair |
| --- | --- | --- |
| 8 easy records | retain/replace | Kept six clean sanity anchors and replaced two generic records with realistic CLI and access-request artifacts. |
| 25 medium records | retain/replace | Replaced three weak clean-form tasks with a deployment log, support history, and exact-identifier reconciliation case. |
| 15 hard records | retain/replace | Replaced one compact synthetic QC row with an email thread and table combining revision selection, nesting, a missing value, arithmetic, and an untrusted instruction. |

No item is a public benchmark control, and no public-source content is used.
The deterministic generator assigns 24 public development items and 24
held-out test items by alternating within the easy-to-hard ordering so both
halves contain multiple difficulty tiers.

## Corrections made during review

- The incident prompt now defines the existing `mitigated_at: 09:41` gold as
  the time errors returned below 1%; the generator-owned gold was not changed.
- Lakh conversion and percentage-to-decimal normalization are stated where
  required.
- Purchase-order and manifest totals are checked against their component rows.
- Project completion percentage, contract milestone years, and the ViewMax
  revision are now explicit in their source text instead of being inferred.
- Nullable tyre pressure is declared as `number or null`, fixing the prior
  generated-schema type mismatch.
- Every item records the deterministic generator version and seed.

## Difficulty policy

Difficulty is tied to visible task properties rather than item position. Easy
items are clean single records. Medium items introduce missing values,
distractors, normalization, revisions, or mixed layouts. Hard items combine at
least three observable features such as nesting, OCR damage, multiple records,
conflicting revisions, unit conversion, and derived totals.
