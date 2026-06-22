from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from llm_workload_benchmark.dataset import DatasetItem, load_suite


DEFAULT_SUITE = Path("data/suites/instruction.yaml")
DEFAULT_SCHEMA_SUITE = Path("data/suites/structured.yaml")
DEFAULT_SUMMARY_SUITE = Path("data/suites/judged.yaml")
DEFAULT_OUTPUT = Path("docs/TEMP_CONSTRAINT_LOAD_CURVE_REVIEW.md")

TASK_METADATA = {
    "api_rate_limiting": ("Explain API rate limiting", "Prose"),
    "order_extraction": ("Extract order IDs", "Extraction"),
    "language_list": ("Generate a programming-language list", "List"),
    "employee_json": ("Create employee JSON", "Structured: JSON"),
    "book_csv": ("Extract book data as CSV", "Structured: CSV"),
    "paragraph_rewrite": ("Rewrite a clunky paragraph", "Prose"),
    "message_classification": ("Route support tickets", "Classification"),
    "vendor_email": ("Decline a vendor renewal", "Prose"),
    "service_yaml": ("Create a service YAML config", "Structured: YAML"),
    "point_ordering": ("Order and transform scored words", "Ordering"),
}


def _slug(item: DatasetItem) -> str:
    return item.id.removeprefix("constraint_").rsplit("_", 1)[0]


def _title(item: DatasetItem) -> str:
    return TASK_METADATA[_slug(item)][0]


def _carrier(item: DatasetItem) -> str:
    return TASK_METADATA[_slug(item)][1]


def _anchor(title: str) -> str:
    return title.casefold().replace(" ", "-").replace("/", "")


def _rule_label(name: str, value: Any) -> str:
    if name == "required_terms":
        return "Include " + ", ".join(map(str, value))
    if name == "forbidden_terms":
        return "Exclude " + ", ".join(map(str, value))
    if name == "excluded_values":
        return "Exclude " + ", ".join(map(str, value))
    if name in {"exact_json_keys", "json_key_order", "exact_top_level_keys", "required_top_level_keys"}:
        prefix = {
            "exact_json_keys": "Exact JSON keys: ",
            "json_key_order": "JSON key order: ",
            "exact_top_level_keys": "Exact YAML keys: ",
            "required_top_level_keys": "Required YAML keys: ",
        }[name]
        return prefix + ", ".join(value)
    if name == "required_forbidden_terms":
        return (
            "Include "
            + ", ".join(value["required"])
            + "; exclude "
            + ", ".join(value["forbidden"])
        )
    if name == "json_array_field_equals":
        return f"Keep only `{value['field']}={value['equals']}` records"
    if name == "json_array_required_keys":
        return "Every JSON record adds " + ", ".join(value)
    if name == "json_array_sorted_by":
        fields = [specification["field"] for specification in value]
        return "Sort JSON records by " + ", then ".join(fields)
    if name == "json_derived_bands":
        return f"Add derived `{value['target_field']}` values"
    if name == "json_summary_counts":
        return f"Append counts grouped by `{value['field']}`"
    if name == "csv_year_min":
        return f"Keep books from {value} onward"
    if name == "csv_tie_sort":
        return f"Sort `{value['secondary']}` inside `{value['primary']}` ties"
    if name == "yaml_healthcheck":
        return f"Add healthcheck `{value['path']}` every {value['interval_seconds']} seconds"
    dynamic_labels = {
        "exact_sentences": lambda: f"Exactly {value} sentences",
        "suffix": lambda: f"End with `{value}`",
        "item_prefix": lambda: f"Prefix every item with `{value}`",
        "numbered_list": lambda: f"Numbered list with {value['count']} lines",
        "max_words_per_line": lambda: f"Fewer than {value + 1} words per line",
        "forbidden_item_character": lambda: f"No item containing `{value}`",
        "csv_sorted_by": lambda: (
            f"Rows sorted by `{value['column']}` {value.get('direction', 'ascending')}"
            if isinstance(value, dict)
            else f"Rows sorted by `{value}` ascending"
        ),
        "max_words": lambda: f"At most {value} words",
        "exact_paragraphs": lambda: f"Exactly {value} paragraphs",
        "word_range": lambda: f"Between {value['min']} and {value['max']} words",
        "first_line_comment_prefix": lambda: f"First line starts `{value}`",
    }
    if name in dynamic_labels:
        return dynamic_labels[name]()
    labels = {
        "comma_separated": "Comma-separated values only",
        "sorted_numeric": "Ascending numeric order",
        "sorted_alphabetically": "Alphabetical order",
        "json_only": "Valid JSON only",
        "json_field_constraints": "Typed and allowed JSON field values",
        "csv_format": "Valid CSV with the required header and enough data rows",
        "csv_year_min": "Only books from the required year onward",
        "csv_tie_sort": "Alphabetical author order inside year ties",
        "csv_year_format": "Unquoted 4-digit years",
        "csv_final_row": "Final row `END,END,0`",
        "json_label_array": "JSON array of labels only",
        "label_domain": "Labels limited to `spam` and `not-spam`",
        "classification_order": "One correct label per message in input order",
        "spam_count_consistent": "Final spam count consistent with output labels",
        "boundary": "Required greeting and sign-off",
        "yaml_only": "Valid YAML only",
        "yaml_field_constraints": "Required YAML values and types",
        "yaml_healthcheck": "Required healthcheck settings",
        "json_array_field_equals": "Only records matching the required field value",
        "json_array_required_keys": "Required keys on every JSON record",
        "json_array_sorted_by": "Required JSON record order",
        "json_derived_bands": "Derived seniority field",
        "json_summary_counts": "Final category counts match the records",
        "list_item_descriptions": "Practical use beside every list item",
        "list_group_balance": "Equal coverage of all supplied focus areas",
        "sorted_by_points": "Descending point-value order",
        "uppercase_items": "All words uppercase",
        "ranked_items": "Every word prefixed with its rank",
        "ties_alphabetical": "Alphabetical tie-breaking",
    }
    return labels.get(name, f"{name}: {value}")


def _task_parts(prompt: str) -> tuple[str, str | None]:
    before_constraints = prompt.split(" Mandatory constraints:", 1)[0]
    if " Source data: " not in before_constraints:
        return before_constraints, None
    objective, source = before_constraints.split(" Source data: ", 1)
    return objective, source


def _render_answer(answer: str) -> str:
    return f"```text\n{answer}\n```"


def _render_source(source: str) -> str:
    separator = " | " if " | " in source else "; " if "; " in source else None
    parts = source.split(separator) if separator else [source]
    return "\n".join(f"> {part}" for part in parts)


def _render_task(number: int, variants: list[DatasetItem]) -> str:
    variants.sort(key=lambda item: len(item.scoring.parameters["rules"]))
    base = variants[0]
    title = _title(base)
    objective, source = _task_parts(base.prompt)
    source_section = (
        f"\n**Frozen source data**\n\n{_render_source(source)}\n"
        if source is not None
        else ""
    )
    progression: list[str] = []
    previous_names: set[str] = set()
    for item in variants:
        rules = item.scoring.parameters["rules"]
        new_names = [name for name in rules if name not in previous_names]
        added = "; ".join(_rule_label(name, rules[name]) for name in new_names)
        progression.append(
            f"| {len(rules)} | `{item.id}` | {item.difficulty} | {added} |"
        )
        previous_names = set(rules)

    references = "\n\n".join(
        f"**L{level} reference**\n\n{_render_answer(item.expected['value'])}"
        for level, item in enumerate(variants, start=1)
    )
    hotspot = next(
        (
            tag
            for tag in base.tags
            if tag
            in {
                "short_rewrite_with_banned_verbs",
                "word_range_with_paragraph_structure",
            }
        ),
        None,
    )
    hotspot_line = (
        f"\n**Interaction hotspot:** `{hotspot}`\n" if hotspot is not None else ""
    )
    return f"""### {number}. {title}

**Carrier:** {_carrier(base)}  
**Task:** {objective}
{source_section}{hotspot_line}
**Constraint progression**

| Level | Item ID | Difficulty | Newly added constraint |
|---:|---|---|---|
{chr(10).join(progression)}

{references}
"""


def _render_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


def _render_dataset_items(items: list[DatasetItem], *, json_answers: bool) -> str:
    sections: list[str] = []
    for item in items:
        answer = (
            _render_json(item.expected["value"])
            if json_answers
            else _render_answer(str(item.expected["value"]))
        )
        max_words = item.scoring.parameters.get("max_words")
        limit_line = f"  \n**Word limit:** {max_words}" if max_words else ""
        sections.append(
            f"""<details>
<summary><code>{item.id}</code> — {item.subcategory} ({item.difficulty})</summary>

**Prompt:** {item.prompt}{limit_line}

**Reference answer**

{answer}

</details>"""
        )
    return "\n\n".join(sections)


def _difficulty_table(items: list[DatasetItem]) -> str:
    counts = Counter(item.difficulty for item in items)
    return f"""| Questions | Easy | Medium | Hard |
|---:|---:|---:|---:|
| {len(items)} | {counts['easy']} | {counts['medium']} | {counts['hard']} |"""


def generate_review(
    suite_path: Path,
    output_path: Path,
    schema_suite_path: Path = DEFAULT_SCHEMA_SUITE,
    summary_suite_path: Path = DEFAULT_SUMMARY_SUITE,
) -> None:
    items = load_suite(suite_path.resolve()).items["constraint_load_curve"]
    schema_items = load_suite(schema_suite_path.resolve()).items["messy_text_to_schema"]
    summary_items = load_suite(summary_suite_path.resolve()).items["grounded_compression"]
    groups: dict[str, list[DatasetItem]] = {}
    for item in items:
        groups.setdefault(item.variant_of or item.id, []).append(item)
    ordered_groups = list(groups.values())
    load_counts = Counter(len(item.scoring.parameters["rules"]) for item in items)
    carrier_counts = Counter(_carrier(group[0]) for group in ordered_groups)
    task_index = "\n".join(
        f"{number}. [{_title(group[0])}](#{_anchor(_title(group[0]))}) — {_carrier(group[0])}"
        for number, group in enumerate(ordered_groups, start=1)
    )
    sections = "\n\n".join(
        _render_task(number, group)
        for number, group in enumerate(ordered_groups, start=1)
    )
    carrier_rows = "\n".join(
        f"| {carrier} | {count} |" for carrier, count in carrier_counts.items()
    )
    document = f"""# Dataset Benchmarks — Temporary Review

> Generated from the instruction, structured extraction, and judged summary
> suites. Edit the question generators, then regenerate this file; do not edit
> the generated review by hand.

## Benchmark navigation

- [Following Multiple Rules](#following-multiple-rules) — 40 questions
- [Messy Text to Schema](#messy-text-to-schema) — 30 questions
- [Fact-Safe Summaries](#fact-safe-summaries) — 20 questions

<a id="following-multiple-rules"></a>
<details open>
<summary><big><big><strong>Following Multiple Rules — 40 questions</strong></big></big></summary>

### At a glance

| Measure | Value |
|---|---:|
| Base tasks | {len(groups)} |
| Variants per task | 4 |
| Total questions | {len(items)} |
| One-constraint questions | {load_counts[1]} |
| Two-constraint questions | {load_counts[2]} |
| Three-constraint questions | {load_counts[3]} |
| Four-constraint questions | {load_counts[4]} |

| Carrier | Base tasks |
|---|---:|
{carrier_rows}

Every group keeps the same underlying task while adding one cumulative
constraint per level. Deterministic checks record content correctness and each
constraint separately; the score is content accuracy multiplied by constraint
compliance.

The data-heavy tasks use 12 to 16 source records. Their references change as
filtering, sorting, derived fields, priorities, or summaries are added.

### Task index

{task_index}

{sections}

### Expected pressure points

- `short_rewrite_with_banned_verbs`: fewer than 40 words while retaining key
  meaning and avoiding common verbs.
- `word_range_with_paragraph_structure`: 45–80 words while preserving exactly
  four blank-line-separated paragraphs.

If tested models remain near-perfect at four constraints, a fifth variant can
be added later without changing these 40 comparisons.

</details>

<a id="messy-text-to-schema"></a>
<details>
<summary><big><big><strong>Messy Text to Schema — 30 questions</strong></big></big></summary>

This benchmark checks whether a model can turn realistic text into exact JSON.
It includes clean records, missing values, distracting numbers, OCR-like
mistakes, conflicting values, lists, nested objects, multiple records, and unit
conversion.

**Evaluation:** parse the answer as JSON, compare every value and type with the
reference, and report partial leaf accuracy when the full object is wrong.

{_difficulty_table(schema_items)}

### Extraction questions

{_render_dataset_items(schema_items, json_answers=True)}

</details>

<a id="fact-safe-summaries"></a>
<details>
<summary><big><big><strong>Fact-Safe Summaries — 20 questions</strong></big></big></summary>

This benchmark checks whether a model can shorten source text without changing
facts or hiding uncertainty. Tasks cover project updates, customer messages,
policies, incidents, experiments, research, contracts, security, launches, and
audits for different readers.

**Evaluation:** enforce the word limit, then use the summary judge to score
faithfulness, fact coverage, relevance, clarity, and concision. Fabricated facts
or missing decision-critical information can fail the answer.

{_difficulty_table(summary_items)}

### Summary questions

{_render_dataset_items(summary_items, json_answers=False)}

</details>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--schema-suite", type=Path, default=DEFAULT_SCHEMA_SUITE)
    parser.add_argument("--summary-suite", type=Path, default=DEFAULT_SUMMARY_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_review(
        args.suite,
        args.output,
        schema_suite_path=args.schema_suite,
        summary_suite_path=args.summary_suite,
    )


if __name__ == "__main__":
    main()
