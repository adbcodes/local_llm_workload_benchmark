import pytest

from llm_workload_benchmark.answer_parser import normalize_text, parse_answer


@pytest.mark.parametrize("response", ["B", "b", "(B)", "[b]", "{B}", "B."])
def test_option_labels_normalize_to_one_canonical_value(response: str) -> None:
    parsed = parse_answer(response, "option")

    assert parsed.parsed
    assert parsed.value == "B"


def test_exact_option_text_maps_to_its_label() -> None:
    parsed = parse_answer(
        "  Twenty-nine days! ",
        "option",
        option_text={"A": "28 days", "B": "Twenty nine days"},
    )

    assert parsed.value == "B"
    assert "map_option_text_to_label" in parsed.normalization_steps


@pytest.mark.parametrize(
    ("response", "status", "violation"),
    [
        ("work\nFINAL: B", "parsed", None),
        ("work FINAL: B", "recovered", "final_not_own_line"),
        ("work without marker", "missing", "missing_final_marker"),
        ("FINAL: A\nFINAL: B", "ambiguous", "multiple_final_answers"),
    ],
)
def test_final_answer_extraction_distinguishes_protocol_and_ambiguity(
    response: str,
    status: str,
    violation: str | None,
) -> None:
    parsed = parse_answer(response, "option", require_final=True)

    assert parsed.status == status
    if violation is not None:
        assert violation in parsed.protocol_violations


def test_text_normalization_is_case_and_punctuation_tolerant() -> None:
    assert normalize_text(" Window-Seats! ")[0] == "window seats"


@pytest.mark.parametrize(
    ("response", "expected"),
    [("₹300", 300), ("$1,234.50", 1234.5), ("12,34,567", 1234567)],
)
def test_number_parser_handles_declared_currency_and_grouping(
    response: str,
    expected: int | float,
) -> None:
    assert parse_answer(response, "number").value == expected


def test_number_parser_rejects_surrounding_explanation() -> None:
    assert parse_answer("The answer is 300", "number").status == "unparseable"


@pytest.mark.parametrize(
    ("response", "answer_unit", "unit_aliases", "expected"),
    [
        ("x = 8", None, (), 8),
        ("$30", "$", (), 30),
        ("30$", "$", (), 30),
        ("45km", "km", ("kilometres",), 45),
        ("45 kilometres", "km", ("kilometres",), 45),
    ],
)
def test_number_parser_handles_generic_wrappers_and_declared_units(
    response: str,
    answer_unit: str | None,
    unit_aliases: tuple[str, ...],
    expected: int,
) -> None:
    parsed = parse_answer(
        response,
        "number",
        answer_unit=answer_unit,
        unit_aliases=unit_aliases,
    )

    assert parsed.value == expected


def test_number_parser_rejects_undeclared_or_mismatched_units() -> None:
    assert parse_answer("45 furlongs", "number").status == "unparseable"
    assert (
        parse_answer("45mi", "number", answer_unit="km").status == "unparseable"
    )


def test_dates_require_declared_unambiguous_formats() -> None:
    assert parse_answer("3rd March 2001", "date").status == "unparseable"
    assert parse_answer(
        "3rd March 2001", "date", date_formats=("%d %B %Y",)
    ).value == "2001-03-03"
    parsed = parse_answer("03/04/2009", "date", date_formats=("%d/%m/%Y",))
    assert parsed.value == "2009-04-03"
    ambiguous = parse_answer(
        "03/04/2009",
        "date",
        date_formats=("%d/%m/%Y", "%m/%d/%Y"),
    )
    assert ambiguous.status == "unparseable"


def test_missing_final_marker_can_use_only_the_last_line_when_contract_allows_it() -> None:
    parsed = parse_answer(
        "working\nThe answer is (B).",
        "option",
        require_final=True,
        recover_missing_final=True,
    )
    assert parsed.value == "B"
    assert parsed.protocol_violations == ["missing_final_marker"]


def test_set_parser_normalizes_members_and_rejects_duplicates() -> None:
    parsed = parse_answer("URGENT, Billing!", "set")
    assert parsed.value == ["billing", "urgent"]
    assert parse_answer("billing, Billing", "set").status == "unparseable"


@pytest.mark.parametrize("kind", ["json", "tool"])
def test_json_recovery_preserves_protocol_violation(kind: str) -> None:
    parsed = parse_answer('```json\n{"answer": 7}\n```', kind)

    assert parsed.value == {"answer": 7}
    assert parsed.status == "recovered"
    assert parsed.protocol_violations == ["markdown_fence"]


def test_json_parser_rejects_malformed_and_concatenated_values() -> None:
    assert parse_answer('{"answer": 7', "json").status == "unparseable"
    assert parse_answer('{"a": 1}{"b": 2}', "json").status == "ambiguous"


def test_json_surrounding_text_is_recoverable_only_when_allowed() -> None:
    response = 'Result: {"answer": 7}. Done.'
    assert parse_answer(response, "json").status == "recovered"
    assert parse_answer(response, "json", allow_recovery=False).status == "unparseable"


def test_confidence_parser_separates_value_from_missing_label() -> None:
    parsed = parse_answer("False\n100", "confidence")

    assert parsed.value == {"answer": "false", "confidence": 100}
    assert parsed.protocol_violations == ["missing_confidence_label"]


def test_confidence_numeric_answer_uses_the_declared_unit_contract() -> None:
    parsed = parse_answer(
        "25%\nconfidence: 100",
        "confidence",
        confidence_answer_kind="number",
        answer_unit="%",
    )

    assert parsed.value == {"answer": 25, "confidence": 100}
    assert "remove_declared_unit" in parsed.normalization_steps


def test_code_fence_recovery_is_logged() -> None:
    parsed = parse_answer("```python\ndef solve():\n    return 1\n```", "code")
    assert parsed.value.startswith("def solve")
    assert parsed.protocol_violations == ["markdown_fence"]


def test_truncated_output_is_never_recovered_from_a_gold_looking_value() -> None:
    parsed = parse_answer("reasoning... FINAL: B", "option", finish_reason="length")
    assert parsed.status == "truncated"
    assert parsed.value is None


def test_truncated_output_keeps_a_complete_explicit_final_slot() -> None:
    parsed = parse_answer(
        "reasoning...\nFINAL: 27/128",
        "text",
        require_final=True,
        finish_reason="length",
    )

    assert parsed.parsed
    assert parsed.extracted_answer == "27/128"
