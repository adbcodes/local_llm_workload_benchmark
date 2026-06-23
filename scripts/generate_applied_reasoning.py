from __future__ import annotations

import argparse
import random
from collections import Counter
from datetime import date, datetime, timedelta
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any

import yaml


GENERATOR = "applied_reasoning_v1"
DEFAULT_SEED = 20260721
DEFAULT_OUTPUT = Path(
    "data/applied_reasoning/generated.yaml"
)
FINAL_ANSWER_INSTRUCTION = (
    "You may show concise working. End with exactly one final line in this format: "
    "FINAL: <answer>"
)


def generate_items(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    def add(
        *,
        item_id: str,
        subcategory: str,
        difficulty: str,
        prompt: str,
        expected: Any,
        scoring: str = "numeric_tolerance",
        response_type: str = "number",
        response_format: str | None = None,
        tags: list[str],
        answer_format: str | None = None,
    ) -> None:
        if scoring == "numeric_tolerance":
            parameters: dict[str, Any] = {
                "absolute_tolerance": 0,
                "allow_surrounding_text": True,
            }
        elif scoring == "rational_value":
            parameters = {
                "absolute_tolerance": 1e-9,
                "allow_surrounding_text": True,
            }
        elif scoring == "date_value":
            parameters = {}
        else:
            parameters = {
                "strip": True,
                "case_sensitive": True,
                "allow_surrounding_text": answer_format is not None,
            }
            if answer_format is not None:
                parameters["answer_format"] = answer_format
        items.append(
            {
                "id": item_id,
                "subcategory": subcategory,
                "difficulty": difficulty,
                "split": "dev" if not any(
                    item["subcategory"] == subcategory for item in items
                ) else "test",
                "prompt": prompt,
                "response_contract": {
                    "type": response_type,
                    "format": response_format,
                },
                "expected": {"value": expected},
                "scoring": {"method": scoring, "parameters": parameters},
                "provenance": {
                    "kind": "synthetic",
                    "review_status": "human_checked",
                    "generator": GENERATOR,
                    "seed": seed,
                },
                "tags": ["fresh_generated", *tags],
            }
        )

    # Arithmetic and percentages: easy, medium, hard.
    percent, total = rng.choice([(15, 800), (20, 600), (25, 480)])
    add(
        item_id="reason_percentage_001",
        subcategory="arithmetic_percentages",
        difficulty="easy",
        prompt=f"What is {percent}% of {total}?",
        expected=percent * total // 100,
        tags=["percentage"],
    )
    base_price, discount, tax = rng.choice([(2000, 10, 18), (2400, 25, 10)])
    final_price = Fraction(base_price * (100 - discount) * (100 + tax), 10_000)
    add(
        item_id="reason_discount_tax_generated_001",
        subcategory="arithmetic_percentages",
        difficulty="medium",
        prompt=(
            f"An item costs ₹{base_price:,}. It is discounted by {discount}%, "
            f"then taxed at {tax}% on the discounted price. What is the final "
            "price in rupees?"
        ),
        expected=int(final_price),
        tags=["percentage", "sequential_operations"],
    )
    start, gain, loss, fee = rng.choice([(50000, 12, 15, 2), (80000, 10, 20, 5)])
    balance = Fraction(
        start * (100 + gain) * (100 - loss) * (100 - fee),
        1_000_000,
    )
    add(
        item_id="reason_sequential_percentage_generated_001",
        subcategory="arithmetic_percentages",
        difficulty="hard",
        prompt=(
            f"An account starts with ₹{start:,}, gains {gain}%, then loses "
            f"{loss}% of its new value. A fee equal to {fee}% of the remaining "
            "value is deducted. What is the final balance in rupees?"
        ),
        expected=int(balance),
        tags=["percentage", "multi_step"],
    )

    # Ratios, rates, and work: easy, medium, hard.
    left, right, combined = rng.choice([(3, 5, 64), (2, 7, 81)])
    add(
        item_id="reason_ratio_share_generated_001",
        subcategory="ratios_rates_work",
        difficulty="easy",
        prompt=(
            f"Two quantities are in the ratio {left}:{right} and total {combined}. "
            "What is the smaller quantity?"
        ),
        expected=combined * left // (left + right),
        tags=["ratio", "proportion"],
    )
    days_a, days_b = rng.choice([(12, 18), (10, 15)])
    together = Fraction(days_a * days_b, days_a + days_b)
    add(
        item_id="reason_combined_work_generated_001",
        subcategory="ratios_rates_work",
        difficulty="medium",
        prompt=(
            f"Worker A completes a job in {days_a} days and worker B in {days_b} "
            "days. At constant rates, how many days do they need working together?"
        ),
        expected=str(together),
        scoring="rational_value",
        tags=["work_rate", "combined_rate"],
    )
    volume, initial_percent, target_percent = rng.choice([(30, 20, 35), (40, 25, 40)])
    added = Fraction(
        volume * (target_percent - initial_percent),
        100 - target_percent,
    )
    add(
        item_id="reason_mixture_generated_001",
        subcategory="ratios_rates_work",
        difficulty="hard",
        prompt=(
            f"A {volume}-litre mixture is {initial_percent}% concentrate. How many "
            f"litres of pure concentrate must be added to make it {target_percent}% "
            "concentrate?"
        ),
        expected=str(added),
        scoring="rational_value",
        tags=["mixture", "ratio", "algebraic_rate"],
    )

    # Algebraic word problems: easy, medium, medium.
    x_value, multiplier, offset = rng.choice([(9, 3, 7), (8, 4, 5)])
    add(
        item_id="reason_linear_equation_generated_001",
        subcategory="algebra_word_problems",
        difficulty="easy",
        prompt=f"Solve for x: {multiplier}x + {offset} = {multiplier * x_value + offset}.",
        expected=x_value,
        tags=["linear_equation"],
    )
    adults, children, adult_price, child_price = rng.choice(
        [(8, 6, 250, 150), (7, 9, 300, 180)]
    )
    add(
        item_id="reason_ticket_system_generated_001",
        subcategory="algebra_word_problems",
        difficulty="medium",
        prompt=(
            f"A venue sold {adults + children} tickets. Adult tickets cost "
            f"₹{adult_price} and child tickets cost ₹{child_price}. Total sales "
            f"were ₹{adults * adult_price + children * child_price}. How many "
            "adult tickets were sold?"
        ),
        expected=adults,
        tags=["simultaneous_equations", "word_problem"],
    )
    width, difference = rng.choice([(11, 5), (14, 4)])
    perimeter = 2 * (width + width + difference)
    add(
        item_id="reason_rectangle_algebra_generated_001",
        subcategory="algebra_word_problems",
        difficulty="medium",
        prompt=(
            f"A rectangle has perimeter {perimeter}. Its length is {difference} "
            "units more than its width. What is its width?"
        ),
        expected=width,
        tags=["linear_equation", "geometry_context"],
    )

    # Number properties and sequences: easy, medium, medium.
    a, b = rng.choice([(6, 8), (9, 12)])
    lcm = abs(a * b) // _gcd(a, b)
    add(
        item_id="reason_lcm_generated_001",
        subcategory="number_properties_sequences",
        difficulty="easy",
        prompt=f"What is the least positive integer divisible by both {a} and {b}?",
        expected=lcm,
        tags=["divisibility", "lcm"],
    )
    limit, divisor, excluded = rng.choice([(200, 6, 9), (300, 8, 12)])
    count = limit // divisor - limit // _lcm(divisor, excluded)
    add(
        item_id="reason_divisibility_count_generated_001",
        subcategory="number_properties_sequences",
        difficulty="medium",
        prompt=(
            f"How many integers from 1 through {limit}, inclusive, are divisible "
            f"by {divisor} but not by {excluded}?"
        ),
        expected=count,
        tags=["divisibility", "inclusion_exclusion"],
    )
    start, first_difference = rng.choice([(3, 5), (4, 4)])
    sequence = [start]
    difference = first_difference
    for _ in range(4):
        sequence.append(sequence[-1] + difference)
        difference += 2
    next_value = sequence[-1] + difference
    add(
        item_id="reason_sequence_generated_001",
        subcategory="number_properties_sequences",
        difficulty="medium",
        prompt=(
            "Find the next number in the sequence: "
            + ", ".join(str(value) for value in sequence)
            + "."
        ),
        expected=next_value,
        tags=["number_sequence", "increasing_differences"],
    )

    # Calendar and time: easy, medium, hard.
    start_time, minutes = rng.choice([("09:35", 150), ("14:20", 105)])
    end_time = datetime.strptime(start_time, "%H:%M") + timedelta(minutes=minutes)
    add(
        item_id="reason_time_duration_generated_001",
        subcategory="calendar_time",
        difficulty="easy",
        prompt=(
            f"A session starts at {start_time} and lasts {minutes} minutes. "
            "What time does it end? Return HH:MM in 24-hour time."
        ),
        expected=end_time.strftime("%H:%M"),
        scoring="exact_match",
        response_type="text",
        response_format="HH:MM",
        tags=["time", "duration"],
    )
    first_date, interval_weeks, occurrence = rng.choice(
        [(date(2026, 3, 3), 2, 7), (date(2026, 1, 7), 3, 5)]
    )
    occurrence_date = first_date + timedelta(weeks=interval_weeks * (occurrence - 1))
    add(
        item_id="reason_calendar_001",
        subcategory="calendar_time",
        difficulty="medium",
        prompt=(
            f"A meeting occurs every {interval_weeks} weeks, with the first "
            f"occurrence on {first_date.isoformat()}. Counting that as occurrence "
            f"1, what is the date of occurrence {occurrence}?"
        ),
        expected=occurrence_date.isoformat(),
        scoring="date_value",
        response_type="date",
        response_format="common_unambiguous_date",
        tags=["calendar", "recurrence"],
    )
    start_day = date(2026, 8, 3)
    holidays = {date(2026, 8, 10)}
    completion = _business_day(start_day, 12, holidays)
    add(
        item_id="reason_business_days_generated_001",
        subcategory="calendar_time",
        difficulty="hard",
        prompt=(
            "A task starts on Monday, 2026-08-03, which counts as business day 1. "
            "It requires 12 business days. Saturdays and Sundays do not count, and "
            "Monday, 2026-08-10 is a holiday. On what date is it completed?"
        ),
        expected=completion.isoformat(),
        scoring="date_value",
        response_type="date",
        response_format="common_unambiguous_date",
        tags=["calendar", "business_days", "holiday"],
    )

    # Probability and counting: easy, medium, hard.
    sides, threshold = rng.choice([(6, 4), (8, 6)])
    favorable = sides - threshold
    add(
        item_id="reason_die_probability_generated_001",
        subcategory="probability_counting",
        difficulty="easy",
        prompt=(
            f"A fair {sides}-sided die numbered 1 through {sides} is rolled. "
            f"What is the probability of rolling a number greater than {threshold}?"
        ),
        expected=str(Fraction(favorable, sides)),
        scoring="rational_value",
        tags=["probability", "single_event"],
    )
    red, blue = rng.choice([(5, 3), (4, 6)])
    both_blue = Fraction(blue, red + blue) * Fraction(blue - 1, red + blue - 1)
    add(
        item_id="reason_without_replacement_generated_001",
        subcategory="probability_counting",
        difficulty="medium",
        prompt=(
            f"A bag contains {red} red and {blue} blue balls. Two balls are drawn "
            "without replacement. What is the probability both are blue?"
        ),
        expected=str(both_blue),
        scoring="rational_value",
        tags=["probability", "without_replacement"],
    )
    flips, exact_heads = 5, 2
    conditional = Fraction(_comb(flips, exact_heads), 2**flips - 1)
    add(
        item_id="reason_conditional_coins_generated_001",
        subcategory="probability_counting",
        difficulty="hard",
        prompt=(
            f"A fair coin is flipped {flips} times. Given that at least one head "
            f"occurred, what is the probability that exactly {exact_heads} heads occurred?"
        ),
        expected=str(conditional),
        scoring="rational_value",
        tags=["probability", "conditional_probability"],
    )

    # Deductive logic: medium, medium, hard.
    add(
        item_id="reason_syllogism_generated_001",
        subcategory="deductive_logic",
        difficulty="medium",
        prompt=(
            "All analysts are readers. Some readers are writers. Must some analysts "
            "be writers? Return only yes, no, or cannot_be_determined."
        ),
        expected="cannot_be_determined",
        scoring="exact_match",
        response_type="text",
        response_format="label",
        tags=["deduction", "syllogism"],
    )
    add(
        item_id="reason_implication_chain_generated_001",
        subcategory="deductive_logic",
        difficulty="medium",
        prompt=(
            "If deployment occurs, tests pass. If tests pass, approval is logged. "
            "Approval was not logged. Did deployment occur? Return only yes or no."
        ),
        expected="no",
        scoring="exact_match",
        response_type="text",
        response_format="label",
        tags=["deduction", "modus_tollens"],
    )
    truth_values = _solve_truth_tellers()
    truthful = ",".join(label for label, value in truth_values.items() if value)
    add(
        item_id="reason_truth_tellers_generated_001",
        subcategory="deductive_logic",
        difficulty="hard",
        prompt=(
            "Each of A, B, and C is either always truthful or always lying. A says, "
            "'B and C are the same type.' B says, 'A is lying.' C says, 'B is "
            "truthful.' Which people are truthful? Return their labels separated by "
            "commas."
        ),
        expected=truthful,
        scoring="exact_match",
        response_type="text",
        response_format="comma_separated_labels",
        answer_format="comma_separated_labels",
        tags=["deduction", "truth_tellers"],
    )

    # Ordering and constraint puzzles: medium, medium, hard.
    assignment = _unique_assignment()
    add(
        item_id="reason_assignment_generated_001",
        subcategory="ordering_constraint_puzzles",
        difficulty="medium",
        prompt=(
            "Ana, Ben, and Chen are assigned API, UI, and DB, one each. Ana is not "
            "assigned UI. Ben is assigned API. Chen is not assigned DB. Return "
            "Ana's assignment, Ben's assignment, and Chen's assignment in that "
            "order, separated by commas."
        ),
        expected=",".join(assignment),
        scoring="exact_match",
        response_type="text",
        response_format="comma_separated_labels",
        answer_format="comma_separated_labels",
        tags=["constraints", "assignment"],
    )
    four_order = _unique_order(
        "CABD",
        lambda p: p[0] == "C" and p.index("B") == p.index("A") + 1 and p.index("D") > p.index("B"),
    )
    add(
        item_id="reason_four_order_generated_001",
        subcategory="ordering_constraint_puzzles",
        difficulty="medium",
        prompt=(
            "Four tasks A, B, C, and D occupy positions 1 through 4. C is first. "
            "B is immediately after A. D is after B. Return the unique order as "
            "comma-separated labels."
        ),
        expected=",".join(four_order),
        scoring="exact_match",
        response_type="text",
        response_format="comma_separated_labels",
        answer_format="comma_separated_labels",
        tags=["constraints", "ordering"],
    )
    five_order = _unique_order(
        "ABCDE",
        lambda p: p.index("B") == p.index("A") + 1
        and p.index("C") == p.index("E") + 1
        and p.index("D") < p.index("A")
        and p.index("B") < p.index("E"),
    )
    add(
        item_id="reason_ordering_001",
        subcategory="ordering_constraint_puzzles",
        difficulty="hard",
        prompt=(
            "Five workshops A, B, C, D, and E occupy positions 1 through 5. B is "
            "immediately after A. C is immediately after E. D is before A. B is "
            "before E. Return the unique order as comma-separated labels."
        ),
        expected=",".join(five_order),
        scoring="exact_match",
        response_type="text",
        response_format="comma_separated_labels",
        answer_format="comma_separated_labels",
        tags=["constraints", "ordering", "multi_step"],
    )

    _validate_distribution(items)
    return items


def write_dataset(path: Path, seed: int = DEFAULT_SEED) -> None:
    document = {
        "schema_version": 1,
        "benchmark": "applied_reasoning",
        "generated_by": GENERATOR,
        "seed": seed,
        "prompt_suffix": FINAL_ANSWER_INSTRUCTION,
        "items": generate_items(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=88),
        encoding="utf-8",
    )


def _validate_distribution(items: list[dict[str, Any]]) -> None:
    difficulties = Counter(item["difficulty"] for item in items)
    subcategories = Counter(item["subcategory"] for item in items)
    assert difficulties == {"easy": 6, "medium": 12, "hard": 6}
    assert len(subcategories) == 8
    assert set(subcategories.values()) == {3}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // _gcd(a, b)


def _comb(n: int, k: int) -> int:
    numerator = 1
    denominator = 1
    for value in range(1, k + 1):
        numerator *= n - value + 1
        denominator *= value
    return numerator // denominator


def _business_day(start: date, count: int, holidays: set[date]) -> date:
    current = start
    completed = 0
    while completed < count:
        if current.weekday() < 5 and current not in holidays:
            completed += 1
        if completed < count:
            current += timedelta(days=1)
    return current


def _solve_truth_tellers() -> dict[str, bool]:
    solutions: list[dict[str, bool]] = []
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                if a == (b == c) and b == (not a) and c == b:
                    solutions.append({"A": a, "B": b, "C": c})
    assert len(solutions) == 1
    return solutions[0]


def _unique_assignment() -> tuple[str, ...]:
    solutions = [
        candidate
        for candidate in permutations(("API", "UI", "DB"))
        if candidate[0] != "UI" and candidate[1] == "API" and candidate[2] != "DB"
    ]
    assert solutions == [("DB", "API", "UI")]
    return solutions[0]


def _unique_order(
    labels: str,
    condition: Any,
) -> tuple[str, ...]:
    solutions = [candidate for candidate in permutations(labels) if condition(candidate)]
    assert len(solutions) == 1
    return solutions[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
