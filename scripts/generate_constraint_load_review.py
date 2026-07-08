from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from typing import Any

from llm_workload_benchmark.dataset import BenchmarkDefinition, DatasetItem, load_suite


DEFAULT_SUITE = Path("data/suites/final_deterministic.yaml")
DEFAULT_OUTPUT = Path("docs/TEMP_CONSTRAINT_LOAD_CURVE_REVIEW.md")

SUITE_TITLES = {
    "A": "Suite A — Core Capability",
    "B": "Suite B — Structured Work",
    "C": "Suite C — Instruction Control",
    "D": "Suite D — Communication",
    "E": "Suite E — Reliability and Trust",
}


def _anchor(benchmark_id: str) -> str:
    return benchmark_id.replace("_", "-")


def _display(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _code_block(value: Any) -> str:
    escaped = html.escape(_display(value))
    return f'<pre style="white-space: pre-wrap"><code>{escaped}</code></pre>'


def _render_conversation(item: DatasetItem) -> str:
    if item.conversation is None:
        return f"**Prompt**\n\n{_code_block(item.prompt)}"
    turns = []
    for message in item.conversation:
        turns.append(
            f"**{message.role.title()}**\n\n{_code_block(message.content)}"
        )
    return "**Conversation shown to the model**\n\n" + "\n\n".join(turns)


def _render_item(item: DatasetItem, number: int) -> str:
    lineage = []
    if item.source_item:
        lineage.append(f"Source: `{item.source_item}`")
    if item.variant_of:
        lineage.append(f"Variant of: `{item.variant_of}`")
    lineage_text = " · ".join(lineage)
    lineage_line = f"  \n{lineage_text}" if lineage_text else ""
    return f"""<details>
<summary><code>{item.id}</code> — {item.subcategory} ({item.difficulty})</summary>

**Question {number}** · Split: `{item.split}` · Visibility: `{item.visibility}`{lineage_line}

{_render_conversation(item)}

**Expected answer**

{_code_block(item.expected["value"])}

**Evaluation:** `{item.scoring.method}`

</details>"""


def _difficulty_table(items: list[DatasetItem]) -> str:
    counts = Counter(item.difficulty for item in items)
    return f"""| Questions | Easy | Medium | Hard | Public | Held out |
|---:|---:|---:|---:|---:|---:|
| {len(items)} | {counts['easy']} | {counts['medium']} | {counts['hard']} | {sum(item.visibility == 'public' for item in items)} | {sum(item.visibility == 'held_out' for item in items)} |"""


def _render_benchmark(
    definition: BenchmarkDefinition,
    items: list[DatasetItem],
) -> str:
    methods = ", ".join(f"`{method}`" for method in definition.scoring_methods)
    task_types = "\n".join(f"- {task_type}" for task_type in definition.task_types)
    questions = "\n\n".join(
        _render_item(item, number) for number, item in enumerate(items, start=1)
    )
    anchor = _anchor(definition.id)
    return f"""<a id="{anchor}"></a>
<details>
<summary><span style="font-size: 1.5em; font-weight: 700;">{html.escape(definition.title)} — {len(items)} questions</span></summary>

{html.escape(definition.description)}

{_difficulty_table(items)}

**Task types**

{task_types}

**Evaluation:** {methods}

### Questions

{questions}

</details>"""


def _navigation(
    definitions: dict[str, BenchmarkDefinition],
    items_by_benchmark: dict[str, list[DatasetItem]],
) -> str:
    grouped: dict[str, list[str]] = {suite: [] for suite in SUITE_TITLES}
    for benchmark_id, definition in definitions.items():
        if definition.suite is None:
            continue
        count = len(items_by_benchmark[benchmark_id])
        grouped[definition.suite].append(
            f"- [{definition.title}](#{_anchor(benchmark_id)}) — {count} questions"
        )
    sections = []
    for suite, title in SUITE_TITLES.items():
        if not grouped[suite]:
            continue
        sections.append(f"### {title}\n\n" + "\n".join(grouped[suite]))
    return "\n\n".join(sections)


def generate_review(
    suite_path: Path = DEFAULT_SUITE,
    output_path: Path = DEFAULT_OUTPUT,
    schema_suite_path: Path | None = None,
    summary_suite_path: Path | None = None,
) -> None:
    del schema_suite_path, summary_suite_path  # Kept for old callers.
    suite = load_suite(suite_path.resolve())
    total = sum(len(items) for items in suite.items.values())
    benchmark_sections = "\n\n".join(
        _render_benchmark(suite.definitions[benchmark_id], items)
        for benchmark_id, items in suite.items.items()
    )
    document = f"""# Dataset Benchmarks — Temporary Review

> Generated from `data/suites/final_deterministic.yaml`. Edit the question YAML or generators,
> then regenerate this file. Do not edit this review by hand.

**{total} questions across {len(suite.items)} benchmarks.**

## Benchmark navigation

{_navigation(suite.definitions, suite.items)}

---

{benchmark_sections}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_review(args.suite, args.output)


if __name__ == "__main__":
    main()
