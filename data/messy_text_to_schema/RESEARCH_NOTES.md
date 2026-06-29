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

All 30 original scenarios were retained and repaired. Their domains and operation paths
were already distinct enough to avoid template substitution; the shared defect
was an underspecified prompt contract and incomplete review metadata.
Eighteen additional scenarios were then authored at the medium/hard boundary to
improve quantization sensitivity without increasing the easy sanity tier.

| Items | Decision | Phase 4 repair |
| --- | --- | --- |
| 8 easy clean records | repair | Added typed schemas and explicit scalar/date rules; kept as local-model sanity anchors. |
| 15 original medium noisy records | repair | Declared missing values, normalized dates/numbers, mixed layouts, distractors, state selection, lists, and revisions explicitly. |
| 7 original hard nested records | repair | Declared nested array/object shapes and source order; retained OCR, multi-record, revision, unit-conversion, and derived-total interactions. |
| 10 new medium scenarios | generate | Added corrections, mixed layouts, timelines, filtering, OCR, missing values, and numeric distractors. |
| 8 new approachable hard scenarios | generate | Combined two or three extraction complications across contracts, freight, rosters, usage, quality, travel, quotes, and energy records. |

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
- Every retained item is marked `human_checked` and records generator version
  and seed.

## Difficulty policy

Difficulty is tied to visible task properties rather than item position. Easy
items are clean single records. Medium items introduce missing values,
distractors, normalization, revisions, or mixed layouts. Hard items combine at
least three observable features such as nesting, OCR damage, multiple records,
conflicting revisions, unit conversion, and derived totals.
