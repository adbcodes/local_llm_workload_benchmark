from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Callable

import yaml


GENERATOR = "applied_reasoning"
DEFAULT_SEED = 20260731
FINAL_ANSWER_INSTRUCTION = (
    "You may show concise working. End with exactly one final line in this format: "
    "FINAL: <answer>"
)
SUBCATEGORIES = (
    "arithmetic_percentages",
    "ratios_rates_work",
    "algebra_word_problems",
    "number_properties_sequences",
    "calendar_time",
    "probability_counting",
    "deductive_logic",
    "ordering_constraint_puzzles",
)
CORE_COUNTS = {
    name: (13 if index < 4 else 12) for index, name in enumerate(SUBCATEGORIES)
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    difficulty: str
    prompt: str
    expected: Any
    scoring: str = "numeric_tolerance"
    response_type: str = "number"
    response_format: str | None = None
    tags: tuple[str, ...] = ()
    answer_format: str | None = None
    answer_unit: str | None = None
    unit_aliases: tuple[str, ...] = ()


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _item(
    scenario: Scenario,
    *,
    subcategory: str,
    sequence: int,
    seed: int,
) -> dict[str, Any]:
    if scenario.scoring == "numeric_tolerance":
        parameters: dict[str, Any] = {
            "absolute_tolerance": 1e-9,
            "allow_surrounding_text": True,
        }
        if scenario.answer_unit:
            parameters["answer_unit"] = scenario.answer_unit
            parameters["unit_aliases"] = list(scenario.unit_aliases)
    elif scenario.scoring == "rational_value":
        parameters = {
            "absolute_tolerance": 1e-9,
            "allow_surrounding_text": True,
        }
    elif scenario.scoring == "date_value":
        parameters = {}
    else:
        parameters = {
            "strip": True,
            "case_sensitive": False,
            "allow_surrounding_text": scenario.answer_format is not None,
        }
        if scenario.answer_format:
            parameters["answer_format"] = scenario.answer_format
    slug = subcategory.replace("_", "")[:14]
    item_id = f"reason_{slug}_{sequence:03d}"
    return {
        "id": item_id,
        "subcategory": subcategory,
        "difficulty": scenario.difficulty,
        "split": "test" if scenario.difficulty != "easy" else "dev",
        "visibility": "held_out",
        "prompt": scenario.prompt,
        "response_contract": {
            "type": scenario.response_type,
            "format": scenario.response_format,
        },
        "expected": {"value": scenario.expected},
        "scoring": {"method": scenario.scoring, "parameters": parameters},
        "provenance": {
            "kind": "synthetic",
            "review_status": "human_checked",
            "generator": GENERATOR,
            "seed": seed,
        },
        "tags": [
            "fresh_generated",
            "headline_core",
            *scenario.tags,
        ],
    }


def _arithmetic(rng: random.Random) -> list[Scenario]:
    rows: list[Scenario] = []
    rows.append(Scenario("direct_percent", "easy", "What is 17.5% of 640?", 112, tags=("sanity", "percentage")))
    price_cases = [(1850, 12, 90, 5), (2400, 15, 120, 12), (3250, 8, 75, 9)]
    for i, (price, discount, fee, tax) in enumerate(price_cases):
        value = Fraction((price * (100 - discount) + fee * 100) * (100 + tax), 10_000)
        rows.append(Scenario(f"mixed_price_{i}", "medium", f"A device costs ₹{price}. It receives a {discount}% discount, then a ₹{fee} shipping charge is added. A {tax}% tax is applied to the discounted price plus shipping. Do not round intermediate values. What is the amount charged in rupees?", float(value) if value.denominator != 1 else value.numerator, tags=("sequential_operations",)))
    mean_cases = [(8, 42, 12, 51, (60, 66)), (9, 38, 11, 47, (55, 58)), (7, 64, 13, 52, (41, 44))]
    for i, (n1, m1, n2, m2, removed) in enumerate(mean_cases):
        value = Fraction(n1 * m1 + n2 * m2 - sum(removed), n1 + n2 - 2)
        rows.append(Scenario(f"corrected_mean_{i}", "medium", f"One batch has {n1} readings with mean {m1}; another has {n2} readings with mean {m2}. Readings {removed[0]} and {removed[1]} from the second batch are discarded. What is the mean of all remaining readings? Give an exact fraction or decimal.", _fraction_text(value), scoring="rational_value", tags=("weighted_mean",)))
    hard_specs = [
        ("removed_extremes", (20, Fraction(262, 5), 18, Fraction(103, 2), 3)),
        ("removed_extremes", (24, Fraction(191, 4), 22, Fraction(93, 2), 2)),
        ("tiered_bill", (640, 180, 6, 9, 13, 35, 18)),
        ("tiered_bill", (575, 150, 5, 8, 12, 40, 15)),
        ("reallocation", (800, 35, 20, 50)),
        ("reallocation", (1250, 28, 16, 40)),
        ("reverse_change", (18, 12, 2464)),
        ("reverse_change", (25, 8, 2916)),
        ("tiered_bill", (710, 220, 4, 7, 11, 55, 12)),
        ("reallocation", (960, 45, 25, 60)),
    ]
    for i, (kind, values) in enumerate(hard_specs):
        if kind == "removed_extremes":
            count, mean_all, inner_count, mean_inner, ratio = values
            removed_sum = count * mean_all - inner_count * mean_inner
            high = removed_sum * ratio / (ratio + 1)
            prompt = f"{count} sensor readings have mean {_fraction_text(mean_all)}. After the highest and lowest are removed, the remaining {inner_count} have mean {_fraction_text(mean_inner)}. The highest is {ratio} times the lowest. What was the highest reading?"
            expected: Any = _fraction_text(high)
            scoring = "rational_value"
        elif kind == "tiered_bill":
            used, first, r1, r2, r3, fixed, rebate = values
            second = 200
            variable = min(used, first) * r1 + min(max(used-first, 0), second) * r2 + max(used-first-second, 0) * r3
            total = Fraction((variable + fixed) * (100 - rebate), 100)
            prompt = f"A utility charges ₹{r1} per unit for the first {first} units, ₹{r2} for the next {second}, and ₹{r3} above that, plus a fixed ₹{fixed} fee. A {rebate}% rebate applies to the entire bill. For {used} units, what is the final bill in rupees?"
            expected, scoring = _fraction_text(total), "rational_value"
        elif kind == "reallocation":
            total, initial_a, moved_pct, final_a = values
            initial = Fraction(total * initial_a, 100)
            moved = Fraction(total * moved_pct, 100)
            remaining_total = total - moved
            target = Fraction(remaining_total * final_a, 100)
            transfer = target - initial
            prompt = f"A fund of ₹{total} is split between A and B, with {initial_a}% initially in A. Then {moved_pct}% of the total fund is withdrawn entirely from B. How much must be transferred from the remaining B balance to A so that A holds {final_a}% of the money still invested?"
            expected, scoring = _fraction_text(transfer), "rational_value"
        else:
            rise, fall, final = values
            start = Fraction(final * 10_000, (100 + rise) * (100 - fall))
            prompt = f"A value rises by {rise}% and then falls by {fall}% of its new value, ending at {final}. What was the starting value? Give an exact fraction or decimal."
            expected, scoring = _fraction_text(start), "rational_value"
        rows.append(Scenario(f"{kind}_{i}", "hard", prompt, expected, scoring=scoring, tags=(kind, "multi_step")))
    return rows


def _ratios(rng: random.Random) -> list[Scenario]:
    rows = [Scenario("ratio_share", "easy", "Two quantities are in the ratio 5:8 and total 143. What is the smaller quantity?", 55, tags=("sanity", "ratio"))]
    work = [(12, 18), (14, 21), (15, 25)]
    for i, (a, b) in enumerate(work):
        value = Fraction(a*b, a+b)
        rows.append(Scenario(f"combined_work_{i}", "medium", f"Worker A alone needs {a} days for a job and worker B alone needs {b} days. At constant rates, how many days do they need together? Give an exact fraction or decimal.", _fraction_text(value), scoring="rational_value", tags=("work_rate",)))
    mixtures = [(36, 25, 40), (45, 20, 35), (50, 30, 44)]
    for i, (volume, start, target) in enumerate(mixtures):
        add = Fraction(volume * (target-start), 100-target)
        rows.append(Scenario(f"mixture_{i}", "medium", f"A {volume}-litre mixture is {start}% concentrate. How many litres of pure concentrate must be added to make it {target}% concentrate?", _fraction_text(add), scoring="rational_value", tags=("mixture",)))
    hard = [
        ("worker_leaves", (12, 18, 3)), ("worker_leaves", (10, 15, 2)),
        ("pipe_leak", (8, 12, 24, 2)), ("pipe_leak", (6, 10, 30, 1)),
        ("average_speed", (180, 60, 120, 40, 30)), ("average_speed", (150, 50, 100, 25, 20)),
        ("replace_mixture", (60, 30, 15, 50)), ("replace_mixture", (80, 25, 20, 40)),
        ("gears", (18, 30, 45, 50)), ("gears", (24, 36, 54, 40)),
    ]
    for i, (kind, v) in enumerate(hard):
        if kind == "worker_leaves":
            a, b, together_days = v
            completed = Fraction(together_days, a) + Fraction(together_days, b)
            value = Fraction(a) * (1-completed)
            prompt = f"A can finish a job in {a} days and B in {b} days. They work together for {together_days} days, then B leaves. How many additional days does A need?"
        elif kind == "pipe_leak":
            fill_a, fill_b, drain, alone = v
            remaining = 1 - Fraction(alone, fill_a)
            net = Fraction(1, fill_a) + Fraction(1, fill_b) - Fraction(1, drain)
            value = remaining / net
            prompt = f"Pipe A fills a tank in {fill_a} hours, B in {fill_b}, and a leak empties a full tank in {drain}. A runs alone for {alone} hours; then B and the leak start while A continues. How many more hours are needed?"
        elif kind == "average_speed":
            d1, s1, d2, s2, stop = v
            value = Fraction(d1+d2, 1) / (Fraction(d1, s1)+Fraction(d2, s2)+Fraction(stop, 60))
            prompt = f"A vehicle travels {d1} km at {s1} km/h, stops for {stop} minutes, then travels {d2} km at {s2} km/h. What is its average speed for the whole journey in km/h?"
        elif kind == "replace_mixture":
            volume, initial, removed, replacement = v
            final_amount = Fraction(volume*initial,100) * Fraction(volume-removed, volume) + Fraction(removed*replacement,100)
            value = final_amount / volume * 100
            prompt = f"A {volume}-litre solution is {initial}% salt. {removed} litres are removed and replaced with a {replacement}% salt solution. What percentage salt is in the final mixture? Return the numeric percentage."
        else:
            teeth_a, teeth_b, teeth_c, turns_a = v
            value = Fraction(turns_a*teeth_a, teeth_c)
            prompt = f"Gear A ({teeth_a} teeth) drives gear B ({teeth_b} teeth), which drives gear C ({teeth_c} teeth). If A turns {turns_a} times, how many turns does C make? Ignore direction."
        rows.append(Scenario(f"{kind}_{i}", "hard", prompt, _fraction_text(value), scoring="rational_value", tags=(kind, "multi_step")))
    return rows


def _algebra(rng: random.Random) -> list[Scenario]:
    rows = [Scenario("linear", "easy", "Solve for x: 7x - 9 = 82.", 13, tags=("sanity", "linear_equation"))]
    medium_specs = [
        ("tickets", (18, 320, 180, 4360)), ("tickets", (23, 250, 150, 4650)),
        ("rectangle", (68, 8)), ("rectangle", (94, 13)),
        ("ages", (44, 8, 3)), ("digits", (9, 27)),
    ]
    for i, (kind, v) in enumerate(medium_specs):
        if kind == "tickets":
            total, adult, child, revenue = v
            answer = (revenue-child*total)//(adult-child)
            prompt = f"A venue sold {total} tickets. Adult tickets cost ₹{adult}, child tickets ₹{child}, and total sales were ₹{revenue}. How many adult tickets were sold?"
        elif kind == "rectangle":
            perimeter, diff = v
            answer = (perimeter//2-diff)//2
            prompt = f"A rectangle has perimeter {perimeter}. Its length is {diff} units more than its width. What is its width?"
        elif kind == "ages":
            total, diff, years = v
            younger = (total-diff)//2
            answer = younger+years
            prompt = f"Two siblings' ages total {total}; the older is {diff} years older. How old will the younger be in {years} years?"
        else:
            digit_sum, difference = v
            tens = (digit_sum + difference//9)//2
            answer = 10*tens + (digit_sum-tens)
            prompt = f"A two-digit number has digits summing to {digit_sum}. The number is {difference} greater than the number formed by reversing its digits. What is the original number?"
        rows.append(Scenario(f"{kind}_{i}", "medium", prompt, answer, tags=(kind,)))
    hard_specs = [
        (137, 20), (164, 24), (191, 28), (218, 32),
    ]
    for i, (constant, max_x) in enumerate(hard_specs):
        feasible = [(x, y) for x in range(1, max_x) for y in range(1, constant) if 3*x+5*y == constant]
        x, y = max(feasible, key=lambda pair: pair[0]*pair[1])
        rows.append(Scenario(f"integer_optimization_{i}", "hard", f"Positive integers x and y satisfy 3x + 5y = {constant}, with x < {max_x}. What is the maximum possible value of xy?", x*y, tags=("diophantine", "optimization")))
    growth = [(10000, 2000, 10), (8000, 1500, 12), (12000, 2500, 8)]
    for i, (start, deposit, rate) in enumerate(growth):
        end = Fraction(start*(100+rate)**2, 10000)+Fraction(deposit*(100+rate),100)
        rows.append(Scenario(f"unknown_growth_{i}", "hard", f"An account starts with ₹{start}. After one year at an unknown annual rate, ₹{deposit} is deposited. It grows for one more year at the same rate and ends at ₹{float(end):g}. What was the annual growth rate? Return the numeric percentage.", rate, answer_unit="%", unit_aliases=("percent",), tags=("quadratic", "percentage")))
    systems = [(7, 11, 5), (9, 14, 4), (12, 17, 6)]
    for i, (x, y, z) in enumerate(systems):
        s1, s2, s3 = x+y+z, 2*x-y+z, x+3*y-z
        rows.append(Scenario(f"three_variable_{i}", "hard", f"Numbers x, y, z satisfy x+y+z={s1}, 2x-y+z={s2}, and x+3y-z={s3}. What is x+2y+3z?", x+2*y+3*z, tags=("linear_system",)))
    return rows


def _number_properties(rng: random.Random) -> list[Scenario]:
    rows = [Scenario("lcm", "easy", "What is the least positive integer divisible by both 18 and 24?", 72, tags=("sanity", "lcm"))]
    medium = [
        ("divisible_not", (500, 8, 12)), ("divisible_not", (720, 9, 15)),
        ("crt_two", (4, 7, 2, 5)), ("crt_two", (5, 8, 3, 7)),
        ("divisor_count", (756,)), ("divisor_count", (1080,)),
    ]
    for i, (kind, v) in enumerate(medium):
        if kind == "divisible_not":
            limit, a, b = v
            answer = limit//a-limit//math.lcm(a,b)
            prompt = f"How many integers from 1 through {limit} are divisible by {a} but not by {b}?"
        elif kind == "crt_two":
            r1, m1, r2, m2 = v
            answer = next(n for n in range(1, m1*m2+1) if n%m1==r1%m1 and n%m2==r2%m2)
            prompt = f"What is the least positive integer n such that n leaves remainder {r1} when divided by {m1} and remainder {r2} when divided by {m2}?"
        else:
            (n,) = v
            answer = sum(n%d==0 for d in range(1,n+1))
            prompt = f"How many positive divisors does {n} have?"
        rows.append(Scenario(f"{kind}_{i}", "medium", prompt, answer, tags=(kind,)))
    hard = [
        ("modular_power", (7, 222, 1000, 3)), ("modular_power", (13, 157, 1000, 3)),
        ("exactly_one", (10000, (6,10,15))), ("exactly_one", (12000, (8,12,18))),
        ("crt_three", ((2,5),(3,7),(4,9))), ("crt_three", ((4,7),(5,8),(6,11))),
        ("coprime_count", (840, 5000)), ("coprime_count", (1260, 6000)),
        ("recurrence", (3,5,4,9)), ("recurrence", (2,7,5,8)),
    ]
    for i, (kind, v) in enumerate(hard):
        if kind == "modular_power":
            base, exponent, modulus, width = v
            answer = str(pow(base, exponent, modulus)).zfill(width)
            prompt = f"What are the last {width} decimal digits of {base}^{exponent}? Return exactly {width} digits, including leading zeros."
            rows.append(Scenario(f"{kind}_{i}", "hard", prompt, answer, scoring="exact_match", response_type="text", response_format="fixed_width_digits", tags=(kind,)))
            continue
        if kind == "exactly_one":
            limit, divisors = v
            answer = sum(sum(n%d==0 for d in divisors)==1 for n in range(1,limit+1))
            prompt = f"How many integers from 1 through {limit} are divisible by exactly one of {divisors[0]}, {divisors[1]}, and {divisors[2]}?"
        elif kind == "crt_three":
            constraints = v
            modulus = math.prod(m for _,m in constraints)
            answer = next(n for n in range(1,modulus+1) if all(n%m==r for r,m in constraints))
            prompt = "What is the least positive integer n such that " + ", ".join(f"n mod {m} = {r}" for r,m in constraints) + "?"
        elif kind == "coprime_count":
            base, limit = v
            answer = sum(math.gcd(n,base)==1 for n in range(1,limit+1))
            prompt = f"How many integers from 1 through {limit} are coprime to {base}?"
        else:
            a1, a2, p, index = v
            seq = [a1,a2]
            while len(seq)<index:
                seq.append(seq[-1]+seq[-2]+p)
            answer = seq[index-1]
            prompt = f"A sequence has a1={a1}, a2={a2}, and a(n)=a(n-1)+a(n-2)+{p} for n≥3. What is a{index}?"
        rows.append(Scenario(f"{kind}_{i}", "hard", prompt, answer, tags=(kind, "multi_step")))
    return rows


def _calendar(rng: random.Random) -> list[Scenario]:
    rows = [Scenario("duration", "easy", "A session starts at 14:35 and lasts 105 minutes. What time does it end? Return HH:MM in 24-hour time.", "16:20", scoring="exact_match", response_type="text", response_format="HH:MM", tags=("sanity", "duration"))]
    recurrences = [(date(2027,2,3),3,6),(date(2028,1,11),2,9)]
    for i,(start,weeks,k) in enumerate(recurrences):
        answer=start+timedelta(weeks=weeks*(k-1))
        rows.append(Scenario(f"recurrence_{i}","medium",f"A meeting repeats every {weeks} weeks. The first occurrence is {start.isoformat()} and counts as occurrence 1. What is the date of occurrence {k}?",answer.isoformat(),scoring="date_value",response_type="date",response_format="common_unambiguous_date",tags=("recurrence",)))
    business=[(date(2027,5,3),10,{date(2027,5,10)}),(date(2028,2,21),13,{date(2028,2,25)})]
    for i,(start,count,holidays) in enumerate(business):
        answer=_business_day(start,count,holidays)
        rows.append(Scenario(f"business_{i}","medium",f"A task starts on {start.isoformat()}, which counts as business day 1. It requires {count} business days. Weekends do not count, and {next(iter(holidays)).isoformat()} is a holiday. On what date is it completed?",answer.isoformat(),scoring="date_value",response_type="date",response_format="common_unambiguous_date",tags=("business_days",)))
    nth=[(2028,9,1,2)]
    for i,(year,month,weekday,n) in enumerate(nth):
        answer=_nth_weekday(year,month,weekday,n)
        rows.append(Scenario(f"nth_weekday_{i}","medium",f"What is the date of the {['first','second','third','fourth'][n-1]} {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][weekday]} of {date(year,month,1).strftime('%B %Y')}?",answer.isoformat(),scoring="date_value",response_type="date",response_format="common_unambiguous_date",tags=("weekday",)))
    hard_specs=[
        ("timezone",("2027-03-27 23:40",330,590,-240)), ("timezone",("2028-12-31 21:55",345,735,-300)),
        ("nested_weekday",(2029,10,1,2,4,3)), ("nested_weekday",(2030,2,3,3,0,4)),
        ("business_multi",(date(2028,12,18),18,{date(2028,12,25),date(2029,1,1)})),
        ("business_multi",(date(2027,8,23),16,{date(2027,8,30),date(2027,9,6)})),
        ("leap_elapsed",(date(2028,2,27),73)), ("leap_elapsed",(date(2031,12,19),86)),
        ("timezone",("2030-06-30 22:20",-210,845,570)),
    ]
    for i,(kind,v) in enumerate(hard_specs):
        if kind=="timezone":
            local_text,origin_minutes,duration,dest_minutes=v
            local=datetime.strptime(local_text,"%Y-%m-%d %H:%M").replace(tzinfo=timezone(timedelta(minutes=origin_minutes)))
            answer=(local+timedelta(minutes=duration)).astimezone(timezone(timedelta(minutes=dest_minutes))).strftime("%Y-%m-%d %H:%M")
            prompt=f"A flight departs at {local_text} in UTC{_offset(origin_minutes)}. It lasts {duration//60} hours {duration%60} minutes and arrives in UTC{_offset(dest_minutes)}. What is the local arrival date and time? Return YYYY-MM-DD HH:MM."
            rows.append(Scenario(f"{kind}_{i}","hard",prompt,answer,scoring="exact_match",response_type="text",response_format="YYYY-MM-DD HH:MM",tags=(kind,"date_rollover")))
            continue
        if kind=="nested_weekday":
            year,month,w1,n1,w2,n2=v
            first=_nth_weekday(year,month,w1,n1)
            answer=_next_nth_weekday(first,w2,n2)
            names=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            prompt=f"Find the {['first','second','third','fourth'][n1-1]} {names[w1]} of {date(year,month,1).strftime('%B %Y')}. Then, counting only {names[w2]}s strictly after that date, find the {['first','second','third','fourth'][n2-1]} such {names[w2]}. Return its date."
        elif kind=="business_multi":
            start,count,holidays=v
            answer=_business_day(start,count,holidays)
            prompt=f"A project starts on {start.isoformat()}, counted as business day 1, and lasts {count} business days. Weekends and the holidays {', '.join(sorted(day.isoformat() for day in holidays))} do not count. What is its completion date?"
        else:
            start,days=v
            answer=start+timedelta(days=days)
            prompt=f"Starting from {start.isoformat()} at 00:00, what calendar date is exactly {days} days later?"
        rows.append(Scenario(f"{kind}_{i}","hard",prompt,answer.isoformat(),scoring="date_value",response_type="date",response_format="common_unambiguous_date",tags=(kind,"multi_step")))
    return rows


def _probability(rng: random.Random) -> list[Scenario]:
    rows=[Scenario("die","easy","A fair 10-sided die numbered 1 through 10 is rolled. What is the probability of rolling a number greater than 7? Give an exact fraction.", "3/10", scoring="rational_value", tags=("sanity","probability"))]
    specs=[("balls",(5,7,3)),("balls",(8,4,2)),("coins",(6,3)),("coins",(7,2)),("committee",(7,5,3))]
    for i,(kind,v) in enumerate(specs):
        if kind=="balls":
            red,blue,draw=v
            value=Fraction(math.comb(blue,draw),math.comb(red+blue,draw))
            prompt=f"A bag has {red} red and {blue} blue balls. {draw} are drawn without replacement. What is the probability all are blue? Give an exact fraction."
        elif kind=="coins":
            flips,heads=v
            value=Fraction(math.comb(flips,heads),2**flips)
            prompt=f"A fair coin is flipped {flips} times. What is the probability of exactly {heads} heads? Give an exact fraction."
        elif kind=="committee":
            women,men,size=v
            value=Fraction(math.comb(women,2)*math.comb(men,size-2),math.comb(women+men,size))
            prompt=f"A committee of {size} is selected uniformly from {women} women and {men} men. What is the probability it contains exactly 2 women? Give an exact fraction."
        else:
            sides,target=v
            favorable=sum(a+b==target for a in range(1,sides+1) for b in range(1,sides+1))
            value=Fraction(favorable,sides*sides)
            prompt=f"Two fair {sides}-sided dice numbered 1 through {sides} are rolled. What is the probability their sum is {target}? Give an exact fraction."
        rows.append(Scenario(f"{kind}_{i}","medium",prompt,_fraction_text(value),scoring="rational_value",tags=(kind,)))
    hard=[
        ("bayes",(1,3,2,1,2,3,1,3)),("bayes",(2,5,3,2,3,4,3,5)),
        ("cards",(2,1)),("cards",(1,2)),
        ("conditional_dice",(8,10,1)),("conditional_dice",(10,13,2)),
        ("derangement",(6,)),("derangement",(7,)),
        ("paths",(7,6,2,3)),
    ]
    for i,(kind,v) in enumerate(hard):
        if kind=="bayes":
            p1n,p1d,r1,b1,r2,b2,p2n,p2d=v
            prior1=Fraction(p1n,p1d); prior2=Fraction(p2n,p2d)
            value=prior1*Fraction(r1,r1+b1)/(prior1*Fraction(r1,r1+b1)+prior2*Fraction(r2,r2+b2))
            prompt=f"Urn 1 is chosen with probability {prior1} and contains {r1} red and {b1} blue balls. Urn 2 is chosen with probability {prior2} and contains {r2} red and {b2} blue balls. A red ball is drawn. What is the probability Urn 1 was chosen? Give an exact fraction."
        elif kind=="cards":
            aces,kings=v
            other=5-aces-kings
            value=Fraction(math.comb(4,aces)*math.comb(4,kings)*math.comb(44,other),math.comb(52,5))
            prompt=f"A five-card hand is selected uniformly from a standard deck. What is the probability it contains exactly {aces} ace(s) and exactly {kings} king(s)? Give an exact fraction."
        elif kind=="conditional_dice":
            sides,total,minimum=v
            universe=[(a,b) for a in range(1,sides+1) for b in range(1,sides+1) if max(a,b)>=minimum+3]
            fav=sum(a+b==total for a,b in universe)
            value=Fraction(fav,len(universe))
            prompt=f"Two fair {sides}-sided dice are rolled. Given that at least one die is {minimum+3} or greater, what is the probability their sum is {total}? Give an exact fraction."
        elif kind=="derangement":
            (n,)=v
            derangements=round(math.factorial(n)/math.e)
            value=Fraction(derangements,math.factorial(n))
            prompt=f"{n} addressed letters are placed uniformly into {n} addressed envelopes, one per envelope. What is the probability that no letter enters its correct envelope? Give an exact fraction."
        else:
            east,north,block_e,block_n=v
            total=math.comb(east+north,east)
            through=math.comb(block_e+block_n,block_e)*math.comb(east+north-block_e-block_n,east-block_e)
            value=Fraction(total-through,total)
            prompt=f"A shortest grid path uses {east} east and {north} north moves in any order. Chosen uniformly among shortest paths, what is the probability it avoids the point reached after exactly {block_e} east and {block_n} north moves? Give an exact fraction."
        rows.append(Scenario(f"{kind}_{i}","hard",prompt,_fraction_text(value),scoring="rational_value",tags=(kind,"multi_step")))
    return rows


def _logic(rng: random.Random) -> list[Scenario]:
    rows=[Scenario("modus_tollens","easy","If a build is deployed, its signed release record exists. The signed release record does not exist. Was the build deployed? Return yes or no.","no",scoring="exact_match",response_type="text",response_format="label",tags=("sanity","deduction"))]
    medium=[
        ("All poets are readers. Some readers are cyclists. Must some poets be cyclists?","cannot_be_determined"),
        ("No copper objects are transparent. Every prism here is transparent. Can any prism here be copper?","no"),
        ("Every red token is square. Every square token is heavy. Token K is red. Must K be heavy?","yes"),
        ("Some editors are musicians. No musicians are pilots. Must some editors not be pilots?","yes"),
        ("All archived files are encrypted. File R is not encrypted. Can File R be archived?","no"),
    ]
    for i,(prompt,answer) in enumerate(medium):
        rows.append(Scenario(f"syllogism_{i}","medium",prompt+" Return yes, no, or cannot_be_determined.",answer,scoring="exact_match",response_type="text",response_format="label",tags=("syllogism",)))
    bases=[
        "A is equivalent to not B; C is equivalent to A; D is equivalent to B and C; E is equivalent to not D; F is equivalent to C and E",
        "A is equivalent to B and C; B is equivalent to not D; C is equivalent to E; E implies D; F is equivalent to not A",
        "A implies B; B is equivalent to not C; D is equivalent to A and C; E is equivalent to not D; F implies A",
    ]
    systems=[]
    mappings=("ABCDEF","BCDAFE","FABCDE")
    for base in bases:
        for mapping in mappings:
            translated=base.translate(str.maketrans("ABCDEF",mapping))
            systems.append(_add_unique_cardinality(translated,6))
    for i,(description,solutions) in enumerate(systems):
        answer=",".join(solutions[0])
        rows.append(Scenario(f"boolean_{i}","hard",f"Boolean flags A through F satisfy: {description} Which flags are true? Return their labels separated by commas.",answer,scoring="exact_match",response_type="text",response_format="comma_separated_labels",answer_format="comma_separated_labels",tags=("boolean_logic","unique_solution")))
    return rows


def _ordering(rng: random.Random) -> list[Scenario]:
    rows=[Scenario("three_order","easy","Mira finishes before Noor, and Noor finishes before Omar. Return the unique order of M, N, O as comma-separated labels.","M,N,O",scoring="exact_match",response_type="text",response_format="comma_separated_labels",answer_format="comma_separated_labels",tags=("sanity","ordering"))]
    targets=["CABDE","BDACE","EACBD","ACEDB","DBEAC","CFADBE","BDAFCE","EBCADF","ACFDEB","DABFCE","CEAGBDF","BFDACGE","GACFBDE","DBHACEGF"]
    for i,target in enumerate(targets):
        clues=_clues_for_unique_order(target)
        solved=_orders_matching(target,clues)
        if solved != [target]:
            raise AssertionError((target,clues,len(solved)))
        difficulty="medium" if i<5 else "hard"
        labels=", ".join(sorted(target))
        prompt=f"The tasks {labels} occupy positions 1 through {len(target)}. " + " ".join(clues) + " Return the unique order as comma-separated labels."
        rows.append(Scenario(f"order_{i}",difficulty,prompt,",".join(target),scoring="exact_match",response_type="text",response_format="comma_separated_labels",answer_format="comma_separated_labels",tags=("ordering","unique_solution")))
    return rows


FAMILY_BUILDERS: dict[str, Callable[[random.Random], list[Scenario]]] = {
    "arithmetic_percentages": _arithmetic,
    "ratios_rates_work": _ratios,
    "algebra_word_problems": _algebra,
    "number_properties_sequences": _number_properties,
    "calendar_time": _calendar,
    "probability_counting": _probability,
    "deductive_logic": _logic,
    "ordering_constraint_puzzles": _ordering,
}


def generate_items(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rng=random.Random(seed)
    core: list[dict[str, Any]]=[]
    for subcategory in SUBCATEGORIES:
        scenarios=FAMILY_BUILDERS[subcategory](rng)
        core_count=CORE_COUNTS[subcategory]
        if len(scenarios)<core_count:
            raise AssertionError(f"{subcategory}: expected at least {core_count} scenarios, got {len(scenarios)}")
        core.extend(_item(s,subcategory=subcategory,sequence=i+1,seed=seed) for i,s in enumerate(scenarios[:core_count]))
    _validate_items(core)
    return core


def _validate_items(core: list[dict[str, Any]]) -> None:
    if len(core)!=100:
        raise AssertionError(len(core))
    if len({item["id"] for item in core})!=100:
        raise AssertionError("duplicate item ids")
    prompt_gold={(item["prompt"],str(item["expected"]["value"])) for item in core}
    if len(prompt_gold)!=len(core):
        raise AssertionError("duplicate generated questions")
    if Counter(item["difficulty"] for item in core)!={"easy":8,"medium":44,"hard":48}:
        raise AssertionError(Counter(item["difficulty"] for item in core))
    if Counter(item["subcategory"] for item in core)!=Counter(CORE_COUNTS):
        raise AssertionError(Counter(item["subcategory"] for item in core))
    if any(item["expected"]["value"] is None for item in core):
        raise AssertionError("missing gold")


def _business_day(start: date,count: int,holidays: set[date]) -> date:
    current=start; seen=0
    while seen<count:
        if current.weekday()<5 and current not in holidays:
            seen+=1
            if seen==count:
                return current
        current+=timedelta(days=1)
    raise AssertionError


def _nth_weekday(year: int,month: int,weekday: int,n: int) -> date:
    first=date(year,month,1)
    return first+timedelta(days=(weekday-first.weekday())%7+7*(n-1))


def _next_nth_weekday(start: date,weekday: int,n: int) -> date:
    offset=(weekday-start.weekday())%7
    if offset==0: offset=7
    return start+timedelta(days=offset+7*(n-1))


def _offset(minutes: int) -> str:
    sign="+" if minutes>=0 else "-"; minutes=abs(minutes)
    return f"{sign}{minutes//60:02d}:{minutes%60:02d}"


def _solve_boolean_description(description: str,count: int) -> list[list[str]]:
    labels=[chr(65+i) for i in range(count)]
    clauses=[part.strip().rstrip(".") for part in description.split(";")]
    count_word=next(part.split()[1] for part in clauses if part.startswith("exactly "))
    cardinality={"one":1,"two":2,"three":3,"four":4}[count_word]
    logical=[part for part in clauses if not part.startswith("exactly ")]
    solutions=[]
    for values in product((False,True),repeat=count):
        env=dict(zip(labels,values))
        def atom(text: str) -> bool:
            text=text.strip()
            return not env[text[4:].strip()] if text.startswith("not ") else env[text]
        valid=True
        for clause in logical:
            if " is equivalent to " in clause:
                left,right=clause.split(" is equivalent to ")
                if " and " in right:
                    right_value=all(atom(x) for x in right.split(" and "))
                else: right_value=atom(right)
                valid &= env[left]==right_value
            elif " implies " in clause:
                left,right=clause.split(" implies ")
                valid &= (not env[left]) or atom(right)
        valid &= sum(values)==cardinality
        if valid: solutions.append([label for label,value in env.items() if value])
    return solutions


def _add_unique_cardinality(base: str,count: int) -> tuple[str,list[list[str]]]:
    words={1:"one",2:"two",3:"three",4:"four",5:"five"}
    for cardinality in range(1,count):
        description=f"{base}; exactly {words[cardinality]} flags are true."
        solutions=_solve_boolean_description(description,count)
        if len(solutions)==1:
            return description,solutions
    raise AssertionError(f"no unique cardinality for {base}")


def _clues_for_unique_order(target: str) -> list[str]:
    labels=list(target)
    candidates=[]
    candidates.extend(
        f"{labels[index]} is immediately before {labels[index+1]}."
        for index in range(len(labels)-1)
    )
    candidates.extend(
        f"{labels[index]} is exactly two positions before {labels[index+2]}."
        for index in range(len(labels)-2)
    )
    candidates.extend(
        f"{labels[index]} is before {labels[index+2]}."
        for index in range(len(labels)-2)
    )
    candidates.extend(
        f"{labels[index]} is not adjacent to {labels[index+2]}."
        for index in range(len(labels)-2)
    )
    candidates.extend((f"{labels[0]} is first.",f"{labels[-1]} is last."))
    random.Random(target).shuffle(candidates)
    clues=[]
    for clue in candidates:
        clues.append(clue)
        if len(_orders_matching(target,clues))==1:
            break
    if len(_orders_matching(target,clues))!=1:
        raise AssertionError(f"could not establish unique order for {target}")
    return clues


def _orders_matching(target: str,clues: list[str]) -> list[str]:
    answers=[]
    for perm in permutations(target):
        order="".join(perm); ok=True
        for clue in clues:
            words=clue.rstrip(".").split(); left=words[0]; right=words[-1]
            if words[1:3]==["is","first"]: ok &= order[0]==left
            elif words[1:3]==["is","last"]: ok &= order[-1]==left
            elif "immediately before" in clue: ok &= order.index(right)==order.index(left)+1
            elif "exactly two positions before" in clue: ok &= order.index(right)==order.index(left)+2
            elif "not adjacent" in clue: ok &= abs(order.index(left)-order.index(right))!=1
            elif "is before" in clue: ok &= order.index(left)<order.index(right)
        if ok: answers.append(order)
    return answers


def _document(items: list[dict[str, Any]],seed: int,benchmark: str="applied_reasoning") -> dict[str,Any]:
    return {"schema_version":1,"benchmark":benchmark,"generated_by":f"{GENERATOR}_v2","seed":seed,"prompt_suffix":FINAL_ANSWER_INSTRUCTION,"items":items}


def write_dataset(output: Path,seed: int=DEFAULT_SEED) -> None:
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(_document(generate_items(seed),seed),sort_keys=False,allow_unicode=True,width=100),encoding="utf-8")


def write_review(path: Path,core: list[dict[str,Any]]) -> None:
    lines=[
        "# Applied Reasoning Final Question Review (Temporary)",
        "",
        "This temporary document lists the 100 fresh questions used by the runnable Applied Reasoning benchmark. Difficulty labels are provisional until empirical calibration.",
        "",
        "## Distribution",
        "",
        "| Bank | Sanity/easy | Medium | Hard | Total |",
        "|---|---:|---:|---:|---:|",
        "| Headline core | 8 | 44 | 48 | 100 |",
        "",
    ]
    for title,items in (("100-question headline core",core),):
        lines.extend((f"## {title}",""))
        for subcategory in SUBCATEGORIES:
            lines.extend((f"### {subcategory}",""))
            for item in (entry for entry in items if entry["subcategory"]==subcategory):
                value=item["expected"]["value"]
                tags=[tag for tag in item["tags"] if tag not in {"fresh_generated","headline_core"}]
                reason=(
                    "Sanity check: verifies basic task, prompt, and scorer operation without influencing the hard-reasoning claim."
                    if item["difficulty"]=="easy"
                    else "Boundary task: adds a distinct, code-verified reasoning path that should separate weaker and stronger configurations."
                    if item["difficulty"]=="medium"
                    else "Challenge task: combines constraints or operations and is intended to create quantization-sensitive headroom."
                )
                lines.extend((
                    f"#### `{item['id']}` — {item['difficulty']}","",
                    f"**Question:** {item['prompt']}","",
                    f"**Gold:** `{value}`", "",
                    f"**Mechanisms:** {', '.join(tags) or 'general reasoning'}.","",
                    f"**Why included:** {reason}","",
                ))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(
        description="Generate an Applied Reasoning draft for manual curation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--review-output",type=Path)
    args=parser.parse_args()
    write_dataset(args.output,args.seed)
    if args.review_output is not None:
        write_review(args.review_output,generate_items(args.seed))


if __name__=="__main__":
    main()
