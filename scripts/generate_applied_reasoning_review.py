from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from llm_workload_benchmark.dataset import load_suite


DEFAULT_SUITE = Path("data/suites/reasoning.yaml")
DEFAULT_OUTPUT = Path("docs/TEMP_APPLIED_REASONING_REVIEW.html")


def generate_review(suite_path: Path, output_path: Path) -> None:
    suite = load_suite(suite_path.resolve())
    items = suite.items["applied_reasoning"]
    payload = json.dumps(
        [item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    static_dataset = _render_static_dataset(
        [item.model_dump(mode="json") for item in items]
    )
    rendered = (
        HTML_TEMPLATE.replace("__DATASET_JSON__", payload)
        .replace("__STATIC_DATASET__", static_dataset)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


CATEGORY_LABELS = {
    "arithmetic_percentages": "Arithmetic & percentages",
    "ratios_rates_work": "Ratios, rates & work",
    "algebra_word_problems": "Algebraic word problems",
    "number_properties_sequences": "Number properties & sequences",
    "calendar_time": "Calendar & time",
    "probability_counting": "Probability & counting",
    "deductive_logic": "Deductive logic",
    "ordering_constraint_puzzles": "Ordering & constraint puzzles",
}


def _render_static_dataset(items: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for category_number, (category, label) in enumerate(CATEGORY_LABELS.items(), start=1):
        category_items = [item for item in items if item["subcategory"] == category]
        records = "".join(
            _render_static_record(item, category_number, item_number)
            for item_number, item in enumerate(category_items, start=1)
        )
        sections.append(
            f'''<section class="category" id="{category}">
              <header class="category-heading">
                <div><p class="category-kicker">Section {category_number:02d}</p><h2>{html.escape(label)}</h2></div>
                <span class="category-count">{len(category_items)} questions</span>
              </header>
              <div class="records">{records}</div>
            </section>'''
        )
    return "\n".join(sections)


def _render_static_record(
    item: dict[str, object], category_number: int, item_number: int
) -> str:
    provenance = item["provenance"]
    assert isinstance(provenance, dict)
    source = provenance.get("source")
    origin = "licensed_anchor" if source else "fresh_generated"
    origin_label = "Licensed anchor" if source else "Fresh generated"
    scoring = item["scoring"]
    contract = item["response_contract"]
    expected = item["expected"]
    assert isinstance(scoring, dict)
    assert isinstance(contract, dict)
    assert isinstance(expected, dict)
    tags = "".join(
        f'<span class="tag">{html.escape(str(tag))}</span>' for tag in item["tags"]
    )
    if isinstance(source, dict):
        source_html = f'''<div class="source-block">
          <div class="source-line"><strong>{html.escape(str(source["dataset"]))}</strong> · {html.escape(str(source["record_id"]))} · {html.escape(str(source["license"]))}</div>
          <div class="source-line"><a href="{html.escape(str(source["url"]))}" target="_blank" rel="noreferrer">Open source record</a></div>
          <div class="source-line">SHA-256 · {html.escape(str(source["content_sha256"]))}</div>
        </div>'''
    else:
        source_html = f'''<div class="source-block">
          <div class="source-line"><strong>{html.escape(str(provenance.get("generator") or "Hand authored"))}</strong></div>
          <div class="source-line">Seed · {html.escape(str(provenance.get("seed") or "—"))}</div>
          <div class="source-line">Kind · {html.escape(str(provenance["kind"]))}</div>
        </div>'''
    contract_value = str(contract["type"])
    if contract.get("format"):
        contract_value += " · " + str(contract["format"])
    parameters = json.dumps(scoring["parameters"], ensure_ascii=False, indent=2)
    answer = expected["value"]
    answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, indent=2)
    return f'''<article class="record" data-origin="{origin}">
      <div class="evidence-strip" aria-hidden="true"></div>
      <div class="record-body">
        <div class="record-head">
          <span class="record-id">{category_number:02d}.{item_number:02d} · {html.escape(str(item["id"]))}</span>
          <div class="chips">
            <span class="chip difficulty-{item["difficulty"]}">{html.escape(str(item["difficulty"]))}</span>
            <span class="chip origin">{origin_label}</span>
            <span class="chip">{html.escape(str(scoring["method"]))}</span>
          </div>
        </div>
        <p class="question">{html.escape(str(item["prompt"]))}</p>
        <div class="answer"><span class="answer-label">Gold answer</span><pre class="answer-value">{html.escape(str(answer_text))}</pre></div>
        <button class="details-toggle" type="button" aria-expanded="false">Show metadata</button>
        <div class="details">
          <dl class="metadata">
            <dt>Subcategory</dt><dd>{html.escape(str(item["subcategory"]))}</dd>
            <dt>Split</dt><dd>{html.escape(str(item["split"]))}</dd>
            <dt>Contract</dt><dd>{html.escape(contract_value)}</dd>
            <dt>Scorer</dt><dd>{html.escape(str(scoring["method"]))}</dd>
            <dt>Parameters</dt><dd>{html.escape(parameters)}</dd>
            <dt>Review</dt><dd>{html.escape(str(provenance["review_status"]))}</dd>
          </dl>
          {source_html}
        </div>
        <div class="tags">{tags}</div>
      </div>
    </article>'''


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Applied Reasoning · Dataset Review</title>
  <style>
    :root {
      --paper: #f2eee5;
      --paper-raised: #f8f5ee;
      --paper-inset: #eae4d8;
      --ink: #25231f;
      --ink-secondary: #555148;
      --ink-tertiary: #777166;
      --ink-muted: #9a9387;
      --rule: rgba(48, 44, 37, 0.17);
      --rule-soft: rgba(48, 44, 37, 0.09);
      --rule-strong: rgba(48, 44, 37, 0.34);
      --ochre: #9a6719;
      --ochre-soft: #e8dcc5;
      --positive: #3f6b57;
      --warning: #945743;
      --control: #e8e1d4;
      --focus: #bd852d;
      --radius-small: 5px;
      --radius-medium: 9px;
      --space: 4px;
      --rail-width: 288px;
      color-scheme: light;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: 16px;
      line-height: 1.55;
    }
    button, input { font: inherit; }
    button { color: inherit; }

    .shell {
      display: grid;
      grid-template-columns: var(--rail-width) minmax(0, 1fr);
      min-height: 100vh;
    }
    .rail {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 32px 24px;
      border-right: 1px solid var(--rule);
      overflow-y: auto;
    }
    .eyebrow, .label, .chip, .record-id, .result-count, .metadata dt,
    .source-line, .answer-label {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--ochre);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-size: clamp(27px, 3vw, 38px);
      font-weight: 600;
      letter-spacing: -.035em;
      line-height: 1.05;
    }
    .lede {
      margin: 14px 0 24px;
      color: var(--ink-secondary);
      font-size: 14px;
      line-height: 1.5;
    }
    .ledger {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 16px;
      padding: 16px 0;
      border-block: 1px solid var(--rule);
    }
    .ledger span { color: var(--ink-tertiary); font-size: 13px; }
    .ledger strong {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .control-group { margin-top: 24px; }
    .label {
      display: block;
      margin-bottom: 8px;
      color: var(--ink-tertiary);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
    }
    .search {
      width: 100%;
      padding: 10px 11px;
      border: 1px solid var(--rule);
      border-radius: var(--radius-small);
      outline: none;
      background: var(--paper-inset);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
    }
    .search:focus { border-color: var(--focus); box-shadow: 0 0 0 2px rgba(189, 133, 45, .14); }
    .search::placeholder { color: var(--ink-muted); }
    .filter-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .filter {
      padding: 6px 9px;
      border: 1px solid var(--rule);
      border-radius: 999px;
      background: transparent;
      cursor: pointer;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
      transition: background 120ms ease, border-color 120ms ease;
    }
    .filter:hover { border-color: var(--rule-strong); }
    .filter[aria-pressed="true"] {
      border-color: rgba(154, 103, 25, .42);
      background: var(--ochre-soft);
      color: #68440f;
    }
    .rail-actions { display: grid; gap: 8px; margin-top: 24px; }
    .action {
      padding: 9px 11px;
      border: 1px solid var(--rule);
      border-radius: var(--radius-small);
      background: transparent;
      cursor: pointer;
      text-align: left;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
    }
    .action:hover { background: var(--paper-raised); }
    .action:focus-visible, .filter:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

    main { min-width: 0; padding: 32px clamp(24px, 5vw, 72px) 80px; }
    .topline {
      position: sticky;
      top: 0;
      z-index: 4;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 48px;
      margin: -32px 0 36px;
      padding: 12px 0 10px;
      border-bottom: 1px solid var(--rule);
      background: rgba(242, 238, 229, .96);
      backdrop-filter: blur(8px);
    }
    .breadcrumb, .result-count {
      color: var(--ink-tertiary);
      font-size: 11px;
    }
    .result-count { font-variant-numeric: tabular-nums; }

    .category { margin: 0 auto 52px; max-width: 980px; scroll-margin-top: 72px; }
    .category-heading {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 20px;
      margin-bottom: 16px;
      padding-bottom: 11px;
      border-bottom: 2px solid var(--ink);
    }
    .category-kicker {
      margin: 0 0 3px;
      color: var(--ochre);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    h2 { margin: 0; font-size: 24px; font-weight: 600; letter-spacing: -.02em; }
    .category-count {
      color: var(--ink-tertiary);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
    }
    .records { display: grid; gap: 12px; }
    .record {
      position: relative;
      display: grid;
      grid-template-columns: 7px minmax(0, 1fr);
      border: 1px solid var(--rule);
      border-radius: var(--radius-medium);
      background: var(--paper-raised);
      overflow: hidden;
    }
    .evidence-strip { background: var(--rule-strong); }
    .record[data-origin="licensed_anchor"] .evidence-strip { background: var(--ochre); }
    .record[data-origin="fresh_generated"] .evidence-strip { background: var(--positive); }
    .record-body { min-width: 0; padding: 18px 20px 16px; }
    .record-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .record-id {
      overflow: hidden;
      color: var(--ink-tertiary);
      font-size: 10px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chips { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 5px; }
    .chip {
      padding: 3px 6px;
      border: 1px solid var(--rule-soft);
      border-radius: 3px;
      color: var(--ink-secondary);
      background: var(--paper);
      font-size: 9px;
      line-height: 1.2;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .chip.difficulty-hard { color: var(--warning); }
    .chip.origin { color: var(--positive); }
    .record[data-origin="licensed_anchor"] .chip.origin { color: var(--ochre); }
    .question {
      margin: 0;
      color: var(--ink);
      font-size: clamp(16px, 1.5vw, 19px);
      line-height: 1.58;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .answer {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      gap: 12px;
      margin-top: 17px;
      padding: 12px 14px;
      border-left: 2px solid var(--ochre);
      background: var(--ochre-soft);
    }
    .answer-label {
      padding-top: 2px;
      color: var(--ochre);
      font-size: 9px;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
    }
    .answer-value {
      margin: 0;
      color: #3c2c16;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    body.answers-hidden .answer-value { filter: blur(7px); user-select: none; }
    .details-toggle {
      margin-top: 14px;
      padding: 0;
      border: 0;
      border-bottom: 1px solid transparent;
      background: transparent;
      color: var(--ink-tertiary);
      cursor: pointer;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 11px;
    }
    .details-toggle:hover { border-color: currentColor; color: var(--ink); }
    .details {
      display: none;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px 28px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--rule-soft);
    }
    .record.details-open .details { display: grid; }
    .metadata { display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 5px 12px; margin: 0; }
    .metadata dt {
      color: var(--ink-muted);
      font-size: 9px;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .metadata dd {
      min-width: 0;
      margin: 0;
      color: var(--ink-secondary);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      overflow-wrap: anywhere;
    }
    .source-block {
      padding: 11px 12px;
      border: 1px solid var(--rule-soft);
      border-radius: var(--radius-small);
      background: var(--paper);
    }
    .source-line { color: var(--ink-secondary); font-size: 10px; overflow-wrap: anywhere; }
    .source-line + .source-line { margin-top: 5px; }
    .source-line a { color: var(--ochre); text-underline-offset: 2px; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; }
    .tag {
      color: var(--ink-tertiary);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 9px;
    }
    .tag::before { content: "#"; color: var(--ink-muted); }
    .empty {
      display: none;
      max-width: 680px;
      margin: 18vh auto 0;
      padding: 28px 0;
      border-block: 1px solid var(--rule);
      text-align: center;
    }
    .empty strong { display: block; font-size: 21px; }
    .empty span { color: var(--ink-tertiary); font-size: 13px; }
    .footer-note {
      max-width: 980px;
      margin: 72px auto 0;
      padding-top: 16px;
      border-top: 1px solid var(--rule);
      color: var(--ink-muted);
      font-size: 11px;
    }

    @media (max-width: 840px) {
      .shell { display: block; }
      .rail {
        position: relative;
        width: auto;
        height: auto;
        padding: 24px 20px 20px;
        border-right: 0;
        border-bottom: 1px solid var(--rule);
      }
      .lede { max-width: 620px; }
      .ledger { grid-template-columns: repeat(4, auto); justify-content: start; gap: 6px 18px; }
      .ledger span { display: none; }
      .control-group { margin-top: 16px; }
      .rail-actions { display: flex; flex-wrap: wrap; margin-top: 16px; }
      main { padding: 28px 16px 64px; }
      .topline { margin-top: -28px; }
    }

    @media (max-width: 560px) {
      .ledger { grid-template-columns: repeat(2, auto); }
      .category-heading { grid-template-columns: 1fr; gap: 2px; }
      .category-count { order: -1; }
      .record-body { padding: 15px 14px 14px; }
      .record-head { align-items: flex-start; flex-direction: column; }
      .chips { justify-content: flex-start; }
      .answer { grid-template-columns: 1fr; gap: 4px; }
      .details { grid-template-columns: 1fr; }
      .metadata { grid-template-columns: 92px minmax(0, 1fr); }
      .breadcrumb { display: none; }
    }

    @media print {
      :root { --paper: #fff; --paper-raised: #fff; --paper-inset: #fff; }
      .shell { display: block; }
      .rail { position: static; height: auto; border: 0; padding: 0 0 20px; }
      .control-group, .rail-actions, .topline, .details-toggle, .empty { display: none !important; }
      main { padding: 0; }
      .category { break-before: page; max-width: none; }
      .category:first-of-type { break-before: auto; }
      .record { break-inside: avoid; }
      .details { display: grid !important; }
      body.answers-hidden .answer-value { filter: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="rail">
      <p class="eyebrow">Dataset review · temporary</p>
      <h1>Applied<br>Reasoning</h1>
      <p class="lede">A reading-first ledger of every prompt, answer, scorer, source, and provenance record in the current 48-question set.</p>
      <div class="ledger" aria-label="Dataset summary">
        <span>Questions</span><strong id="total-count">48</strong>
        <span>Licensed</span><strong id="licensed-count">24</strong>
        <span>Generated</span><strong id="generated-count">24</strong>
        <span>Distribution</span><strong>12 · 24 · 12</strong>
      </div>

      <div class="control-group">
        <label class="label" for="search">Find a question</label>
        <input class="search" id="search" type="search" placeholder="Prompt, ID, tag, answer…" autocomplete="off">
      </div>
      <div class="control-group">
        <span class="label">Difficulty</span>
        <div class="filter-row" id="difficulty-filters"></div>
      </div>
      <div class="control-group">
        <span class="label">Origin</span>
        <div class="filter-row" id="origin-filters"></div>
      </div>
      <div class="rail-actions">
        <button class="action" id="toggle-answers" type="button">Hide answers</button>
        <button class="action" id="expand-details" type="button">Expand all metadata</button>
        <button class="action" id="clear-filters" type="button">Clear filters</button>
      </div>
    </aside>

    <main>
      <div class="topline">
        <span class="breadcrumb">data / applied_reasoning / review</span>
        <span class="result-count" id="result-count" aria-live="polite"></span>
      </div>
      <div id="dataset">__STATIC_DATASET__</div>
      <div class="empty" id="empty-state">
        <strong>No matching questions</strong>
        <span>Clear a filter or try a broader search.</span>
      </div>
      <p class="footer-note">Generated from <code>data/suites/reasoning.yaml</code>. This temporary review file should be regenerated whenever the YAML dataset changes.</p>
    </main>
  </div>

  <script id="dataset-json" type="application/json">__DATASET_JSON__</script>
  <script>
    const items = JSON.parse(document.getElementById('dataset-json').textContent);
    const categoryLabels = {
      arithmetic_percentages: 'Arithmetic & percentages',
      ratios_rates_work: 'Ratios, rates & work',
      algebra_word_problems: 'Algebraic word problems',
      number_properties_sequences: 'Number properties & sequences',
      calendar_time: 'Calendar & time',
      probability_counting: 'Probability & counting',
      deductive_logic: 'Deductive logic',
      ordering_constraint_puzzles: 'Ordering & constraint puzzles'
    };
    const categoryOrder = Object.keys(categoryLabels);
    const state = { difficulty: 'all', origin: 'all', query: '', expanded: false };

    const originOf = item => item.provenance.source ? 'licensed_anchor' :
      item.provenance.kind === 'synthetic' ? 'fresh_generated' : 'hand_authored';
    const originLabel = value => ({
      licensed_anchor: 'Licensed anchor',
      fresh_generated: 'Fresh generated',
      hand_authored: 'Hand authored'
    }[value] || value);
    const titleCase = value => value.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
    const escapeHtml = value => String(value)
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
    const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);

    function filterButton(value, label, group) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'filter';
      button.textContent = label;
      button.dataset.value = value;
      button.setAttribute('aria-pressed', value === 'all' ? 'true' : 'false');
      button.addEventListener('click', () => {
        state[group] = value;
        button.parentElement.querySelectorAll('.filter').forEach(candidate =>
          candidate.setAttribute('aria-pressed', String(candidate === button)));
        render();
      });
      return button;
    }

    function setupFilters() {
      const difficulty = document.getElementById('difficulty-filters');
      [['all', 'All'], ['easy', 'Easy'], ['medium', 'Medium'], ['hard', 'Hard']]
        .forEach(([value, label]) => difficulty.append(filterButton(value, label, 'difficulty')));
      const origin = document.getElementById('origin-filters');
      [['all', 'All'], ['licensed_anchor', 'Licensed'], ['fresh_generated', 'Generated']]
        .forEach(([value, label]) => origin.append(filterButton(value, label, 'origin')));
    }

    function matches(item) {
      if (state.difficulty !== 'all' && item.difficulty !== state.difficulty) return false;
      if (state.origin !== 'all' && originOf(item) !== state.origin) return false;
      if (!state.query) return true;
      const haystack = JSON.stringify(item).toLowerCase();
      return haystack.includes(state.query);
    }

    function metadata(item) {
      const scoring = item.scoring;
      const contract = item.response_contract;
      return `
        <dl class="metadata">
          <dt>Subcategory</dt><dd>${escapeHtml(item.subcategory)}</dd>
          <dt>Split</dt><dd>${escapeHtml(item.split)}</dd>
          <dt>Contract</dt><dd>${escapeHtml(contract.type)}${contract.format ? ' · ' + escapeHtml(contract.format) : ''}</dd>
          <dt>Scorer</dt><dd>${escapeHtml(scoring.method)}</dd>
          <dt>Parameters</dt><dd>${escapeHtml(pretty(scoring.parameters))}</dd>
          <dt>Review</dt><dd>${escapeHtml(item.provenance.review_status)}</dd>
        </dl>`;
    }

    function sourceBlock(item) {
      const source = item.provenance.source;
      if (source) {
        return `<div class="source-block">
          <div class="source-line"><strong>${escapeHtml(source.dataset)}</strong> · ${escapeHtml(source.record_id)} · ${escapeHtml(source.license)}</div>
          <div class="source-line"><a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Open source record</a></div>
          <div class="source-line">SHA-256 · ${escapeHtml(source.content_sha256)}</div>
        </div>`;
      }
      return `<div class="source-block">
        <div class="source-line"><strong>${escapeHtml(item.provenance.generator || 'Hand authored')}</strong></div>
        <div class="source-line">Seed · ${escapeHtml(item.provenance.seed ?? '—')}</div>
        <div class="source-line">Kind · ${escapeHtml(item.provenance.kind)}</div>
      </div>`;
    }

    function record(item, categoryNumber, itemNumber) {
      const origin = originOf(item);
      const article = document.createElement('article');
      article.className = 'record' + (state.expanded ? ' details-open' : '');
      article.dataset.origin = origin;
      article.innerHTML = `
        <div class="evidence-strip" aria-hidden="true"></div>
        <div class="record-body">
          <div class="record-head">
            <span class="record-id">${String(categoryNumber).padStart(2, '0')}.${String(itemNumber).padStart(2, '0')} · ${escapeHtml(item.id)}</span>
            <div class="chips">
              <span class="chip difficulty-${escapeHtml(item.difficulty)}">${escapeHtml(item.difficulty)}</span>
              <span class="chip origin">${escapeHtml(originLabel(origin))}</span>
              <span class="chip">${escapeHtml(item.scoring.method)}</span>
            </div>
          </div>
          <p class="question">${escapeHtml(item.prompt)}</p>
          <div class="answer">
            <span class="answer-label">Gold answer</span>
            <pre class="answer-value">${escapeHtml(pretty(item.expected.value))}</pre>
          </div>
          <button class="details-toggle" type="button" aria-expanded="${state.expanded}">${state.expanded ? 'Hide' : 'Show'} metadata</button>
          <div class="details">
            ${metadata(item)}
            ${sourceBlock(item)}
          </div>
          <div class="tags">${item.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div>
        </div>`;
      const toggle = article.querySelector('.details-toggle');
      toggle.addEventListener('click', () => {
        const open = article.classList.toggle('details-open');
        toggle.textContent = open ? 'Hide metadata' : 'Show metadata';
        toggle.setAttribute('aria-expanded', String(open));
      });
      return article;
    }

    function render() {
      const root = document.getElementById('dataset');
      root.textContent = '';
      const visible = items.filter(matches);
      categoryOrder.forEach((category, categoryIndex) => {
        const categoryItems = visible.filter(item => item.subcategory === category);
        if (!categoryItems.length) return;
        const section = document.createElement('section');
        section.className = 'category';
        section.id = category;
        section.innerHTML = `<header class="category-heading">
          <div><p class="category-kicker">Section ${String(categoryIndex + 1).padStart(2, '0')}</p><h2>${categoryLabels[category]}</h2></div>
          <span class="category-count">${categoryItems.length} question${categoryItems.length === 1 ? '' : 's'}</span>
        </header><div class="records"></div>`;
        const records = section.querySelector('.records');
        categoryItems.forEach((item, itemIndex) => records.append(record(item, categoryIndex + 1, itemIndex + 1)));
        root.append(section);
      });
      document.getElementById('result-count').textContent = `${visible.length} / ${items.length} visible`;
      document.getElementById('empty-state').style.display = visible.length ? 'none' : 'block';
    }

    document.getElementById('search').addEventListener('input', event => {
      state.query = event.target.value.trim().toLowerCase();
      render();
    });
    document.getElementById('toggle-answers').addEventListener('click', event => {
      const hidden = document.body.classList.toggle('answers-hidden');
      event.currentTarget.textContent = hidden ? 'Reveal answers' : 'Hide answers';
    });
    document.getElementById('expand-details').addEventListener('click', event => {
      state.expanded = !state.expanded;
      event.currentTarget.textContent = state.expanded ? 'Collapse all metadata' : 'Expand all metadata';
      render();
    });
    document.getElementById('clear-filters').addEventListener('click', () => {
      state.difficulty = 'all'; state.origin = 'all'; state.query = '';
      document.getElementById('search').value = '';
      document.querySelectorAll('.filter').forEach(button =>
        button.setAttribute('aria-pressed', String(button.dataset.value === 'all')));
      render();
    });

    setupFilters();
    document.getElementById('total-count').textContent = items.length;
    document.getElementById('licensed-count').textContent = items.filter(item => originOf(item) === 'licensed_anchor').length;
    document.getElementById('generated-count').textContent = items.filter(item => originOf(item) === 'fresh_generated').length;
    render();
  </script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_review(args.suite, args.output)


if __name__ == "__main__":
    main()
