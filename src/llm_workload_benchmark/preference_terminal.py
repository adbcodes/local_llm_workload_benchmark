from __future__ import annotations

import builtins
import io
import keyword
import os
import re
import shutil
import sys
import textwrap
import token
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llm_workload_benchmark.preference import (
    BlindComparison,
    HumanSelection,
    PreferenceError,
    append_human_preference,
    completed_preference_count,
    default_preference_path,
    prepare_preference_ballot,
)

_NONE_CHOICES = {"n", "none", "none of above", "none of the above"}
_BUILTIN_NAMES = set(dir(builtins))
_RESET = "\033[0m"
_CYAN = "\033[1;36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_DIM = "\033[2m"


@dataclass(frozen=True)
class TerminalPreferenceResult:
    output_path: Path
    completed: int
    total: int

    @property
    def is_complete(self) -> bool:
        return self.completed == self.total


def run_terminal_preferences(
    experiment_directory: Path,
    *,
    model_ids: tuple[str, str] | None = None,
    seed: int = 42,
    output_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    terminal_width: int | None = None,
    color: bool | None = None,
) -> TerminalPreferenceResult:
    """Display, collect, persist, and resume a blind terminal ballot."""

    ballot = prepare_preference_ballot(
        experiment_directory,
        model_ids=model_ids,
        seed=seed,
    )
    destination = (output_path or default_preference_path(ballot)).resolve()
    completed = completed_preference_count(destination, ballot)
    if completed:
        output_fn(f"Resuming saved ballot at comparison {completed + 1}/{len(ballot.items)}.")
    if completed == len(ballot.items):
        output_fn("This ballot is already complete.")
        return TerminalPreferenceResult(destination, completed, len(ballot.items))

    for index in range(completed, len(ballot.items)):
        item = ballot.items[index]
        output_fn(
            render_terminal_comparison(
                item.comparison,
                width=terminal_width,
                color=_supports_color() if color is None else color,
            )
        )
        try:
            selection = _prompt_for_selection(
                input_fn,
                output_fn,
                {candidate.label for candidate in item.comparison.candidates},
            )
        except (EOFError, KeyboardInterrupt):
            output_fn("\nVoting paused. Run the same command to resume.")
            return TerminalPreferenceResult(destination, index, len(ballot.items))
        append_human_preference(
            destination,
            ballot=ballot,
            item=item,
            choice=selection,
        )
        output_fn(f"Vote saved [{index + 1}/{len(ballot.items)}].")
    return TerminalPreferenceResult(destination, len(ballot.items), len(ballot.items))


def render_terminal_comparison(
    comparison: BlindComparison,
    *,
    width: int | None = None,
    color: bool = False,
) -> str:
    """Render one prompt and two or three anonymous answers in columns."""

    detected_width = width or shutil.get_terminal_size((120, 24)).columns
    candidate_count = len(comparison.candidates)
    total_width = max(66, min(detected_width, 180))
    gap = "  "
    column_width = (
        total_width - len(gap) * (candidate_count - 1)
    ) // candidate_count
    rule = "─" * total_width
    question_width = max(20, total_width - 2)
    question_lines = _wrap_text(comparison.prompt, question_width)
    boxes = [
        _answer_box(
            candidate.label.upper(),
            candidate.response,
            column_width,
            color=color,
        )
        for candidate in comparison.candidates
    ]
    box_height = max(len(box) for box in boxes)
    for box in boxes:
        _pad_box(box, box_height, column_width)
    rows = [
        "",
        f"BENCHMARK ITEM  {comparison.number}/{comparison.total}",
        f"{comparison.benchmark} / {comparison.item_id}",
        rule,
        "QUESTION",
        *question_lines,
        "",
    ]
    rows.extend(gap.join(parts) for parts in zip(*boxes, strict=True))
    answer_options = "    ".join(
        f"[{candidate.label.upper()}] Answer {candidate.label.upper()}"
        for candidate in comparison.candidates
    )
    rows.extend(
        [
            "",
            rule,
            f"{answer_options}    [N] None of the above",
            "Choose one or more answers, for example: A, a b, or A,C",
        ]
    )
    return "\n".join(rows)


def normalize_human_selection(value: str) -> HumanSelection | None:
    """Map case-insensitive single, multiple, or none input to a selection."""

    normalized = " ".join(value.strip().casefold().split())
    if normalized in _NONE_CHOICES:
        return "none"
    normalized = re.sub(r"\band\b", " ", normalized)
    compact = re.sub(r"[\s,;/+&]+", "", normalized)
    if not compact or any(label not in "abc" for label in compact):
        return None
    labels = tuple(label for label in "abc" if label in compact)
    if len(labels) != len(compact):
        return None
    return labels


def _prompt_for_selection(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    candidate_labels: set[str],
) -> HumanSelection:
    keys = "/".join(label.upper() for label in sorted(candidate_labels)) + "/N"
    named_choices = ", ".join(
        label.upper() for label in sorted(candidate_labels)
    )
    while True:
        selection = normalize_human_selection(
            input_fn(f"Your vote [{keys}; multiple allowed]")
        )
        if selection == "none" or (
            selection is not None
            and all(label in candidate_labels for label in selection)
        ):
            return selection
        output_fn(
            f"Invalid choice. Enter one or more of {named_choices}, "
            "or none of the above."
        )


def _answer_box(
    label: str,
    response: str,
    width: int,
    *,
    color: bool,
) -> list[str]:
    content_width = max(1, width - 4)
    content = _wrap_code(response, content_width)
    title = f" ANSWER {label} · IDENTITY SEALED "
    inner_width = width - 2
    if len(title) > inner_width:
        title = f" ANSWER {label} "
    top = "┌" + title + "─" * (inner_width - len(title)) + "┐"
    if color:
        top = top.replace(f"ANSWER {label}", f"{_CYAN}ANSWER {label}{_RESET}")
    rows = [top]
    padded_content = [line.ljust(content_width) for line in content]
    if color:
        padded_content = _highlight_python_lines(padded_content)
    rows.extend(f"│ {line} │" for line in padded_content)
    rows.append("└" + "─" * inner_width + "┘")
    return rows


def _wrap_text(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in value.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=True,
            )
            or [""]
        )
    return lines


def _pad_box(box: list[str], height: int, width: int) -> None:
    blank = f"│ {' ' * (width - 4)} │"
    box[-1:-1] = [blank] * (height - len(box))


def _wrap_code(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in value.splitlines() or [""]:
        if not raw_line:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                subsequent_indent="↪ ",
                replace_whitespace=False,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return lines


def _supports_color() -> bool:
    return (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


def _highlight_python_lines(lines: list[str]) -> list[str]:
    highlighted: list[str] = []
    triple_quote: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            highlighted.append(f"{_DIM}{_CYAN}{line}{_RESET}")
            continue
        delimiter = triple_quote or _triple_quote_in(line)
        if delimiter is not None:
            highlighted.append(f"{_GREEN}{line}{_RESET}")
            if line.count(delimiter) % 2 == 1:
                triple_quote = None if triple_quote else delimiter
            continue
        if line.startswith("↪ "):
            highlighted.append(
                f"{_DIM}↪{_RESET} " + _highlight_python_line(line[2:])
            )
        else:
            highlighted.append(_highlight_python_line(line))
    return highlighted


def _triple_quote_in(line: str) -> str | None:
    for delimiter in ('"""', "'''"):
        if delimiter in line:
            return delimiter
    return None


def _highlight_python_line(line: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line + "\n").readline))
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return line
    parts: list[str] = []
    cursor = 0
    after_def = False
    for value in tokens:
        if value.type in {token.NEWLINE, token.NL, token.ENDMARKER, token.INDENT, token.DEDENT}:
            continue
        start = value.start[1]
        end = value.end[1]
        if start < cursor or start > len(line):
            continue
        parts.append(line[cursor:start])
        style = ""
        if value.type == token.NAME:
            if keyword.iskeyword(value.string):
                style = _CYAN
                after_def = value.string == "def"
            elif after_def:
                style = _MAGENTA
                after_def = False
            elif value.string in _BUILTIN_NAMES:
                style = _YELLOW
        elif value.type == token.STRING:
            style = _GREEN
        elif value.type == token.NUMBER:
            style = _BLUE
        elif value.type == token.COMMENT:
            style = _DIM
        parts.append(f"{style}{value.string}{_RESET}" if style else value.string)
        cursor = min(end, len(line))
    parts.append(line[cursor:])
    return "".join(parts)
