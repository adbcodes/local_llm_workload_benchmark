from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerKind = Literal[
    "text",
    "option",
    "number",
    "date",
    "json",
    "set",
    "confidence",
    "tool",
    "code",
]
ParseStatus = Literal[
    "parsed",
    "recovered",
    "missing",
    "ambiguous",
    "unparseable",
    "truncated",
]


class ParsedAnswer(BaseModel):
    """Auditable result of extracting and normalizing one model response."""

    model_config = ConfigDict(extra="forbid")

    raw_response: str
    extracted_answer: str | None = None
    value: Any = None
    value_type: AnswerKind | None = None
    normalization_steps: list[str] = Field(default_factory=list)
    protocol_violations: list[str] = Field(default_factory=list)
    status: ParseStatus

    @property
    def parsed(self) -> bool:
        return self.status in {"parsed", "recovered"}


def parse_answer(
    response: str,
    kind: AnswerKind,
    *,
    require_final: bool = False,
    recover_missing_final: bool = False,
    allow_recovery: bool = True,
    option_text: dict[str, str] | None = None,
    separator: str = ",",
    date_formats: tuple[str, ...] = (),
    confidence_answer_kind: Literal["text", "number"] = "text",
    finish_reason: str | None = None,
) -> ParsedAnswer:
    """Extract one unambiguous typed answer under an explicit contract."""

    if finish_reason == "length":
        return _result(
            response,
            kind,
            status="truncated",
            violations=["output_truncated"],
        )
    if not response.strip():
        return _result(response, kind, status="missing", violations=["missing_answer"])

    extracted, steps, violations, error = _extract_answer_slot(
        response,
        require_final=require_final,
        recover_missing_final=recover_missing_final,
    )
    if error is not None:
        return _result(
            response,
            kind,
            extracted=extracted,
            steps=steps,
            violations=violations,
            status=error,
        )
    assert extracted is not None

    if kind in {"json", "tool"}:
        value, wrapper, json_error = _parse_json_value(extracted, allow_recovery)
        if json_error is not None:
            return _result(
                response,
                kind,
                extracted=extracted,
                steps=steps,
                violations=violations,
                status=json_error,
            )
        if wrapper is not None:
            steps.append(f"strip_{wrapper}")
            violations.append(wrapper)
        return _success(response, extracted, value, kind, steps, violations)

    unfenced, fence = _strip_fence(extracted, kind)
    if fence is not None:
        if not allow_recovery:
            return _result(
                response,
                kind,
                extracted=extracted,
                steps=steps,
                violations=violations,
                status="unparseable",
            )
        extracted = unfenced
        steps.append("strip_markdown_fence")
        violations.append("markdown_fence")

    try:
        if kind == "text":
            value, applied = normalize_text(extracted)
        elif kind == "option":
            value, applied = _parse_option(extracted, option_text or {})
        elif kind == "number":
            value, applied = _parse_number(extracted)
        elif kind == "date":
            value, applied = _parse_date(extracted, date_formats)
        elif kind == "set":
            value, applied = _parse_set(extracted, separator)
        elif kind == "confidence":
            value, applied, confidence_violations = _parse_confidence(
                extracted,
                confidence_answer_kind,
            )
            violations.extend(confidence_violations)
        elif kind == "code":
            value, applied = extracted.strip(), ["strip_whitespace"]
        else:  # pragma: no cover - Literal keeps callers out of this branch.
            raise ValueError(f"unsupported answer kind: {kind}")
    except ValueError:
        return _result(
            response,
            kind,
            extracted=extracted,
            steps=steps,
            violations=violations,
            status="unparseable",
        )
    steps.extend(step for step in applied if step not in steps)
    return _success(response, extracted, value, kind, steps, violations)


def normalize_text(value: str) -> tuple[str, list[str]]:
    """Normalize semantic text while preserving the raw response elsewhere."""

    steps: list[str] = []
    normalized = value.strip()
    if normalized != value:
        steps.append("strip_whitespace")
    folded = normalized.casefold()
    if folded != normalized:
        steps.append("casefold")
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in folded
    )
    if without_punctuation != folded:
        steps.append("remove_punctuation")
    collapsed = " ".join(without_punctuation.split())
    if collapsed != without_punctuation:
        steps.append("collapse_whitespace")
    return collapsed, steps


def _extract_answer_slot(
    response: str,
    *,
    require_final: bool,
    recover_missing_final: bool,
) -> tuple[str | None, list[str], list[str], ParseStatus | None]:
    if not require_final:
        return response.strip(), ["strip_whitespace"], [], None

    final_candidates = re.findall(r"(?im)^\s*FINAL\s*:\s*(.*?)\s*$", response)
    inline_candidates = re.findall(r"(?i)(?<!\w)FINAL\s*:\s*([^\r\n]+)", response)
    if len(inline_candidates) > 1 or len(final_candidates) > 1:
        return None, [], ["multiple_final_answers"], "ambiguous"
    if final_candidates:
        candidate = final_candidates[0].strip()
        if not candidate:
            return None, [], ["empty_final_answer"], "missing"
        return candidate, ["extract_final_line"], [], None
    if len(inline_candidates) == 1:
        candidate = inline_candidates[0].strip()
        if not candidate:
            return None, [], ["empty_final_answer"], "missing"
        return candidate, ["extract_inline_final"], ["final_not_own_line"], None
    if recover_missing_final:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            candidate = re.sub(
                r"(?i)^(?:the\s+)?(?:final\s+)?(?:answer|date|value)\s*(?::|=|is)\s*",
                "",
                lines[-1],
            ).strip()
            if candidate:
                return (
                    candidate,
                    ["recover_last_nonempty_line"],
                    ["missing_final_marker"],
                    None,
                )
    return None, [], ["missing_final_marker"], "missing"


def _parse_option(value: str, option_text: dict[str, str]) -> tuple[str, list[str]]:
    candidate = value.strip().rstrip(".!?")
    steps = ["strip_whitespace"] if candidate != value else []
    wrapped = re.fullmatch(r"[\(\[\{]?\s*([A-Za-z])\s*[\)\]\}]?[\.!?]?", candidate)
    if wrapped:
        label = wrapped.group(1).upper()
        if candidate != label:
            steps.append("normalize_option_label")
        return label, steps

    normalized_candidate, text_steps = normalize_text(candidate)
    matches = [
        label.upper()
        for label, text in option_text.items()
        if normalize_text(text)[0] == normalized_candidate
    ]
    if len(matches) != 1:
        raise ValueError("option text is missing or ambiguous")
    return matches[0], [*steps, *text_steps, "map_option_text_to_label"]


def _parse_number(value: str) -> tuple[int | float, list[str]]:
    candidate = value.strip()
    steps = ["strip_whitespace"] if candidate != value else []
    quoted = re.fullmatch(r"(['\"])(.*?)\1", candidate)
    if quoted:
        candidate = quoted.group(2).strip()
        steps.append("remove_scalar_quotes")
    without_terminal = candidate.rstrip("!?")
    if re.fullmatch(r"[-+]?(?:\d[\d,]*)\.", without_terminal):
        without_terminal = without_terminal[:-1]
    if without_terminal != candidate:
        candidate = without_terminal
        steps.append("remove_terminal_punctuation")
    currency = re.fullmatch(r"([₹$£€])?\s*(.*?)\s*", candidate)
    assert currency is not None
    if currency.group(1):
        steps.append("remove_currency_symbol")
    numeric = currency.group(2)
    if not re.fullmatch(r"[-+]?(?:\d+(?:,\d{2,3})*|\d+)(?:\.\d+)?", numeric):
        raise ValueError("not one numeric value")
    if "," in numeric:
        numeric = numeric.replace(",", "")
        steps.append("remove_grouping_separators")
    parsed = float(numeric) if "." in numeric else int(numeric)
    return parsed, steps


def _parse_date(value: str, formats: tuple[str, ...]) -> tuple[str, list[str]]:
    if not formats:
        raise ValueError("date parsing requires declared formats")
    candidate = value.strip().rstrip(".!?")
    without_ordinal = re.sub(r"(?i)(?<=\d)(st|nd|rd|th)\b", "", candidate)
    steps = ["remove_date_ordinal"] if without_ordinal != candidate else []
    matches: set[str] = set()
    for date_format in formats:
        try:
            matches.add(datetime.strptime(without_ordinal, date_format).date().isoformat())
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError("date is unsupported or ambiguous")
    return matches.pop(), [*steps, "parse_declared_date_format", "normalize_date_iso8601"]


def _parse_set(value: str, separator: str) -> tuple[list[str], list[str]]:
    members = [part.strip() for part in value.split(separator)]
    if not members or any(not member for member in members):
        raise ValueError("set contains an empty member")
    normalized = [normalize_text(member)[0] for member in members]
    if len(set(normalized)) != len(normalized):
        raise ValueError("set contains duplicate normalized members")
    return sorted(normalized), ["split_set", "normalize_set_members", "sort_set"]


def _parse_confidence(
    value: str,
    answer_kind: Literal["text", "number"],
) -> tuple[dict[str, Any], list[str], list[str]]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("confidence response must contain answer and confidence")
    labelled = re.fullmatch(r"(?i)confidence\s*:\s*(100|[1-9]?\d)\s*%?", lines[1])
    violations: list[str] = []
    if labelled:
        confidence = int(labelled.group(1))
        steps = ["parse_confidence_label"]
    else:
        unlabelled = re.fullmatch(r"(100|[1-9]?\d)\s*%?", lines[1])
        if not unlabelled:
            raise ValueError("confidence value is missing or ambiguous")
        confidence = int(unlabelled.group(1))
        steps = ["recover_unlabelled_confidence"]
        violations.append("missing_confidence_label")
    if answer_kind == "number":
        answer, answer_steps = _parse_number(lines[0])
    else:
        answer, answer_steps = normalize_text(lines[0])
    return {"answer": answer, "confidence": confidence}, [*answer_steps, *steps], violations


def _parse_json_value(
    value: str,
    allow_recovery: bool,
) -> tuple[Any | None, str | None, ParseStatus | None]:
    stripped = value.strip()
    try:
        return json.loads(stripped), None, None
    except json.JSONDecodeError:
        pass
    if not allow_recovery:
        return None, None, "unparseable"

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "markdown_fence", None
        except json.JSONDecodeError:
            return None, None, "unparseable"

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(parsed)
        if stripped[index + end :].lstrip().startswith(("{", "[")):
            return None, None, "ambiguous"
    if len(candidates) != 1:
        return None, None, "ambiguous" if candidates else "unparseable"
    return candidates[0], "surrounding_text", None


def _strip_fence(value: str, kind: AnswerKind) -> tuple[str, str | None]:
    language = "python|py" if kind == "code" else "text|txt"
    match = re.fullmatch(
        rf"```(?:{language})?\s*(.*?)\s*```",
        value.strip(),
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (match.group(1).strip(), "markdown_fence") if match else (value, None)


def _success(
    raw: str,
    extracted: str,
    value: Any,
    kind: AnswerKind,
    steps: list[str],
    violations: list[str],
) -> ParsedAnswer:
    status: ParseStatus = "recovered" if violations else "parsed"
    return _result(
        raw,
        kind,
        extracted=extracted,
        value=value,
        steps=steps,
        violations=list(dict.fromkeys(violations)),
        status=status,
    )


def _result(
    raw: str,
    kind: AnswerKind,
    *,
    extracted: str | None = None,
    value: Any = None,
    steps: list[str] | None = None,
    violations: list[str] | None = None,
    status: ParseStatus,
) -> ParsedAnswer:
    return ParsedAnswer(
        raw_response=raw,
        extracted_answer=extracted,
        value=value,
        value_type=kind,
        normalization_steps=steps or [],
        protocol_violations=violations or [],
        status=status,
    )
