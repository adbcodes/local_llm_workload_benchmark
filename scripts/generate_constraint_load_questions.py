from __future__ import annotations

import argparse
import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OUTPUT = Path("data/constraint_load_curve/questions.yaml")
GENERATOR_ID = "instruction_following_v2"
DIFFICULTY_BY_LEVEL = {1: "easy", 2: "medium", 3: "medium", 4: "hard"}
SUBCATEGORY_BY_LEVEL = {
    1: "one_constraint",
    2: "two_constraints",
    3: "three_constraints",
    4: "four_constraints",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv(rows: list[list[Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().rstrip("\n")


EMPLOYEES = [
    {"id": "E101", "name": "Asha Iyer", "team": "Platform", "city": "Bengaluru", "status": "active", "years": 7},
    {"id": "E102", "name": "Rohan Shah", "team": "Sales", "city": "Mumbai", "status": "inactive", "years": 4},
    {"id": "E103", "name": "Neha Rao", "team": "Data", "city": "Hyderabad", "status": "active", "years": 2},
    {"id": "E104", "name": "Kabir Khan", "team": "Platform", "city": "Pune", "status": "active", "years": 5},
    {"id": "E105", "name": "Tara Bose", "team": "Support", "city": "Kolkata", "status": "active", "years": 1},
    {"id": "E106", "name": "Vikram Nair", "team": "Data", "city": "Chennai", "status": "inactive", "years": 8},
    {"id": "E107", "name": "Isha Mehta", "team": "Security", "city": "Delhi", "status": "active", "years": 6},
    {"id": "E108", "name": "Dev Patel", "team": "Support", "city": "Ahmedabad", "status": "active", "years": 3},
    {"id": "E109", "name": "Maya Sen", "team": "Sales", "city": "Mumbai", "status": "active", "years": 9},
    {"id": "E110", "name": "Arjun Das", "team": "Security", "city": "Kochi", "status": "inactive", "years": 2},
    {"id": "E111", "name": "Leena Roy", "team": "Platform", "city": "Bengaluru", "status": "active", "years": 4},
    {"id": "E112", "name": "Omar Ali", "team": "Data", "city": "Hyderabad", "status": "active", "years": 5},
]
ACTIVE_EMPLOYEES = [employee for employee in EMPLOYEES if employee["status"] == "active"]
SORTED_ACTIVE_EMPLOYEES = sorted(
    ACTIVE_EMPLOYEES, key=lambda employee: (-employee["years"], employee["id"])
)
SENIORITY_EMPLOYEES = [
    {
        **employee,
        "seniority": (
            "junior"
            if employee["years"] <= 2
            else "mid"
            if employee["years"] <= 5
            else "senior"
        ),
    }
    for employee in SORTED_ACTIVE_EMPLOYEES
]

BOOKS = [
    ["The River Code", "Mira Das", 2018],
    ["Quiet Signals", "Arun Mehta", 2022],
    ["Paper Skies", "Leena Roy", 2015],
    ["Copper Rain", "Nisha Kapoor", 2012],
    ["The Last Algorithm", "Dev Nair", 2020],
    ["Monsoon Ledger", "Tara Bose", 2018],
    ["Glass Harbor", "Omar Ali", 2016],
    ["Small Machines", "Isha Rao", 2024],
    ["Northern Byte", "Kabir Sen", 2014],
    ["Borrowed Maps", "Maya Shah", 2019],
    ["Threaded City", "Rohan Iyer", 2022],
    ["Debugging Dawn", "Asha Menon", 2017],
    ["Signals, Systems, and Sand", "Vikram Das", 2018],
    ["Quiet Compiler", "Neha Jain", 2020],
    ["Data Orchard", "Arjun Rao", 2013],
]
RECENT_BOOKS = [book for book in BOOKS if book[2] >= 2015]
SORTED_RECENT_BOOKS = sorted(
    sorted(RECENT_BOOKS, key=lambda book: book[1].casefold(), reverse=True),
    key=lambda book: -book[2],
)
TIE_SORTED_RECENT_BOOKS = sorted(
    RECENT_BOOKS, key=lambda book: (-book[2], book[1].casefold())
)
BOOK_HEADER = [["title", "author", "year"]]

TICKETS = [
    ("T01", "A customer was charged twice for one invoice.", "billing", "high"),
    ("T02", "I forgot my password and need a reset link.", "account", "normal"),
    ("T03", "The production API is down for every request.", "technical", "urgent"),
    ("T04", "Please send your current pricing brochure.", "general", "normal"),
    ("T05", "My approved refund has not reached my card.", "billing", "high"),
    ("T06", "I cannot sign in after changing my email address.", "account", "high"),
    ("T07", "Large CSV exports time out after five minutes.", "technical", "high"),
    ("T08", "We would like to discuss a partnership.", "general", "normal"),
    ("T09", "We were charged after cancelling the subscription.", "billing", "urgent"),
    ("T10", "Please transfer account ownership to our new manager.", "account", "normal"),
    ("T11", "Security alerts are missing during an active incident.", "technical", "urgent"),
    ("T12", "Where can I find the webinar recording?", "general", "normal"),
    ("T13", "The tax ID printed on our invoice is wrong.", "billing", "normal"),
    ("T14", "Remove a former employee's access immediately.", "account", "urgent"),
    ("T15", "The mobile app crashes whenever a photo is uploaded.", "technical", "high"),
    ("T16", "Can you share the public product roadmap?", "general", "normal"),
]
TICKET_L1 = [{"id": item_id, "category": category} for item_id, _, category, _ in TICKETS]
TICKET_L2 = [
    {"id": item_id, "category": category, "priority": priority}
    for item_id, _, category, priority in TICKETS
]
PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2}
TICKET_L3 = sorted(
    TICKET_L2, key=lambda ticket: (PRIORITY_ORDER[ticket["priority"]], ticket["id"])
)
TICKET_COUNTS = {
    category: sum(ticket["category"] == category for ticket in TICKET_L2)
    for category in ("account", "billing", "general", "technical")
}
TICKET_L4 = [*TICKET_L3, {"summary": TICKET_COUNTS}]

LANGUAGE_GROUPS = {
    "C": "systems", "Rust": "systems", "Go": "systems",
    "Java": "jvm", "Kotlin": "jvm", "Scala": "jvm",
    "JavaScript": "web", "TypeScript": "web", "PHP": "web",
    "Python": "data", "R": "data", "Julia": "data",
    "Swift": "mobile", "Dart": "mobile", "Objective-C": "mobile",
}

POINTS = {
    "amber": 72, "birch": 88, "cobalt": 95, "delta": 81, "ember": 72,
    "fable": 90, "grove": 81, "harbor": 67, "indigo": 95, "juniper": 88,
    "kite": 76, "linen": 67, "maple": 90, "nova": 76, "orbit": 84,
}


TASKS: list[dict[str, Any]] = [
    {
        "slug": "api_rate_limiting",
        "title": "Explain API rate limiting",
        "carrier": "prose",
        "split": "dev",
        "instruction": "Explain API rate limiting to a junior developer.",
        "source": None,
        "answers": [
            "Rate limiting controls how many requests a client sends in a period. It protects the service from overload. Clients that send too many requests must wait. This keeps access fair for everyone.",
            "Rate limiting controls request volume and protects the service from overload. A server may respond with 429 when a client sends too many requests. The client should read Retry-After before trying again. This keeps access fair for everyone.",
            "Rate limiting controls request volume and protects the service from overload. A server may respond with 429 and a Retry-After value. The client should use exponential backoff with jitter before trying again. This avoids a new burst from many clients retrying together.",
            "Rate limiting protects a service from excessive request volume. After a 429 response, read Retry-After and retry with exponential backoff plus jitter. Stop after three failed retries instead of retrying forever. Return a clear error so the caller can try later.",
        ],
        "content": [
            {"required_facts": [{"name": "purpose", "any_of": ["protects the service", "protects a service", "protect a service", "protect the service"]}]},
            {"required_facts": [
                {"name": "purpose", "any_of": ["protects the service", "protects a service", "protect a service", "protect the service"]},
                {"name": "retry_after", "any_of": ["Retry-After"]},
            ]},
            {"required_facts": [
                {"name": "purpose", "any_of": ["protects the service", "protects a service", "protect a service", "protect the service"]},
                {"name": "retry_after", "any_of": ["Retry-After"]},
                {"name": "backoff", "any_of": ["exponential backoff with jitter", "exponential backoff plus jitter"]},
            ]},
            {"required_facts": [
                {"name": "purpose", "any_of": ["protects the service", "protects a service", "protect a service", "protect the service"]},
                {"name": "retry_after", "any_of": ["Retry-After"]},
                {"name": "backoff", "any_of": ["exponential backoff with jitter", "exponential backoff plus jitter"]},
                {"name": "stop_condition", "any_of": ["Stop after three failed retries", "maximum retry count", "retry limit"]},
            ]},
        ],
        "rules": [
            ("Use exactly 4 sentences", "exact_sentences", 4),
            ('Include both "429" and "Retry-After"', "required_terms", ["429", "Retry-After"]),
            ('Explain both exponential backoff and jitter', "required_forbidden_terms", {"required": ["exponential backoff", "jitter"], "forbidden": ["retry immediately"]}),
            ("Keep the explanation under 65 words", "max_words", 64),
        ],
    },
    {
        "slug": "order_extraction",
        "title": "Extract order IDs",
        "carrier": "extraction",
        "split": "dev",
        "instruction": "Extract the order IDs from the fifteen order statements.",
        "source": (
            "Order 1042 is processing with a total of $58.40. "
            "Order 1007 shipped yesterday and totals $120.00. "
            "Order 1031 is pending payment for $42.75. "
            "Order 1015 was cancelled after a duplicate request for $89.99. "
            "Order 1024 is delivered and totals $15.50. Order 1058 is packed for dispatch at $33.20. "
            "Order 1064 was cancelled after a stock error at $74.00. Order 1071 is processing at $18.99. "
            "Order 1083 shipped this morning at $210.40. Order 1090 is pending payment at $64.35. "
            "Order 1102 was delivered at $9.80. Order 1117 was cancelled as a duplicate at $51.10. "
            "Order 1125 is ready for pickup at $27.60. Order 1139 shipped yesterday at $145.00. "
            "Order 1146 is processing at $39.25."
        ),
        "answers": [
            "1042,1007,1031,1015,1024,1058,1064,1071,1083,1090,1102,1117,1125,1139,1146",
            "1007,1015,1024,1031,1042,1058,1064,1071,1083,1090,1102,1117,1125,1139,1146",
            "1007,1024,1031,1042,1058,1071,1083,1090,1102,1125,1139,1146",
            "ORD-1007,ORD-1024,ORD-1031,ORD-1042,ORD-1058,ORD-1071,ORD-1083,ORD-1090,ORD-1102,ORD-1125,ORD-1139,ORD-1146",
        ],
        "content": [
            {
                "required_values": {
                    "values": ["1042", "1007", "1031", "1015", "1024", "1058", "1064", "1071", "1083", "1090", "1102", "1117", "1125", "1139", "1146"],
                    "separator": ",",
                    "strip_prefix": "ORD-",
                }
            },
            {
                "required_values": {
                    "values": ["1042", "1007", "1031", "1015", "1024", "1058", "1064", "1071", "1083", "1090", "1102", "1117", "1125", "1139", "1146"],
                    "separator": ",",
                    "strip_prefix": "ORD-",
                }
            },
            {
                "required_values": {
                    "values": ["1042", "1007", "1031", "1024", "1058", "1071", "1083", "1090", "1102", "1125", "1139", "1146"],
                    "separator": ",",
                    "strip_prefix": "ORD-",
                }
            },
            {
                "required_values": {
                    "values": ["1042", "1007", "1031", "1024", "1058", "1071", "1083", "1090", "1102", "1125", "1139", "1146"],
                    "separator": ",",
                    "strip_prefix": "ORD-",
                }
            },
        ],
        "rules": [
            (
                "Output only the order IDs, comma-separated, with no prose",
                "comma_separated",
                {"item_pattern": "(?:ORD-)?[0-9]{4}"},
            ),
            ("Sort the IDs in ascending numeric order", "sorted_numeric", True),
            ("Exclude orders marked cancelled", "excluded_values", ["1015", "1064", "1117"]),
            ('Prefix every ID with "ORD-"', "item_prefix", "ORD-"),
        ],
    },
    {
        "slug": "language_list",
        "title": "Generate a programming-language list",
        "carrier": "list",
        "split": "dev",
        "instruction": "Choose 10 languages for a balanced engineering curriculum.",
        "source": "Candidates by focus: systems=C|Rust|Go; JVM=Java|Kotlin|Scala; web=JavaScript|TypeScript|PHP; data=Python|R|Julia; mobile=Swift|Dart|Objective-C.",
        "answers": [
            "1. C\n2. Rust\n3. Go\n4. Java\n5. Kotlin\n6. Scala\n7. Python\n8. R\n9. Julia\n10. Swift",
            "1. C\n2. Go\n3. Java\n4. Julia\n5. Kotlin\n6. Python\n7. R\n8. Rust\n9. Scala\n10. Swift",
            "1. C — embedded systems\n2. Go — network services\n3. Java — enterprise services\n4. Julia — scientific computing\n5. Kotlin — Android services\n6. Python — data automation\n7. R — statistical analysis\n8. Rust — safe systems\n9. Scala — data platforms\n10. Swift — iOS applications",
            "1. C — embedded systems\n2. Dart — mobile applications\n3. Java — enterprise services\n4. JavaScript — browser applications\n5. Kotlin — Android services\n6. Python — data automation\n7. R — statistical analysis\n8. Rust — safe systems\n9. Swift — iOS applications\n10. TypeScript — typed web apps",
        ],
        "content": [{"none": True}] * 4,
        "rules": [
            (
                "Use a numbered list with one item per line",
                "numbered_list",
                {"count": 10},
            ),
            ("Sort the languages alphabetically", "sorted_alphabetically", True),
            ("Add a short practical use after an em dash on every line", "list_item_descriptions", True),
            (
                "Include exactly 2 languages from each supplied focus area",
                "list_group_balance",
                {"groups": LANGUAGE_GROUPS, "count_per_group": 2},
            ),
        ],
    },
    {
        "slug": "employee_json",
        "title": "Create employee JSON",
        "carrier": "structured_json",
        "split": "test",
        "instruction": "Turn the employee roster into a JSON report.",
        "source": "; ".join(", ".join(f"{key}={value}" for key, value in employee.items()) for employee in EMPLOYEES) + ".",
        "answers": [_json(EMPLOYEES), _json(ACTIVE_EMPLOYEES), _json(SORTED_ACTIVE_EMPLOYEES), _json(SENIORITY_EMPLOYEES)],
        "content": [{"exact_json": value} for value in [EMPLOYEES, ACTIVE_EMPLOYEES, SORTED_ACTIVE_EMPLOYEES, SENIORITY_EMPLOYEES]],
        "rules": [
            ("Return one valid JSON array with one object per selected employee and no prose", "json_only", "array"),
            (
                "Keep only employees whose status is active",
                "json_array_field_equals",
                {"field": "status", "equals": "active"},
            ),
            (
                "Sort employees by years descending, then ID ascending",
                "json_array_sorted_by",
                [{"field": "years", "direction": "descending"}, {"field": "id", "direction": "ascending"}],
            ),
            (
                "Add seniority: junior for 0-2 years, mid for 3-5, and senior for 6 or more",
                "json_derived_bands",
                {"source_field": "years", "target_field": "seniority", "bands": [{"maximum": 2, "value": "junior"}, {"maximum": 5, "value": "mid"}], "otherwise": "senior"},
            ),
        ],
    },
    {
        "slug": "book_csv",
        "title": "Extract book data as CSV",
        "carrier": "structured_csv",
        "split": "test",
        "instruction": "Extract the book records from the passage as CSV.",
        "source": " | ".join(f'{title} by {author} was published in {year}.' for title, author, year in BOOKS),
        "answers": [
            _csv(BOOK_HEADER + BOOKS),
            _csv(BOOK_HEADER + RECENT_BOOKS),
            _csv(BOOK_HEADER + SORTED_RECENT_BOOKS),
            _csv(BOOK_HEADER + TIE_SORTED_RECENT_BOOKS),
        ],
        "content": [
            {"csv_records": [*BOOKS]},
            {"csv_records": [*RECENT_BOOKS]},
            {"csv_records": [*SORTED_RECENT_BOOKS]},
            {"csv_records": [*TIE_SORTED_RECENT_BOOKS]},
        ],
        "rules": [
            (
                "Return valid CSV with the exact header title,author,year and at least 10 data rows",
                "csv_format",
                {"header": ["title", "author", "year"], "minimum_data_rows": 10},
            ),
            ("Keep only books published in 2015 or later", "csv_year_min", 2015),
            (
                "Sort rows by year descending",
                "csv_sorted_by",
                {"column": "year", "direction": "descending"},
            ),
            ("For books from the same year, sort authors alphabetically", "csv_tie_sort", {"primary": "year", "secondary": "author"}),
        ],
    },
    {
        "slug": "paragraph_rewrite",
        "title": "Rewrite a clunky paragraph",
        "carrier": "prose",
        "split": "test",
        "instruction": "Rewrite the paragraph clearly while preserving its meaning.",
        "source": (
            "The project team was asked to review the launch plan because the "
            "original schedule was no longer realistic, and there were "
            "concerns about whether the deadline could be met within the "
            "budget. The team was also told that stakeholders were "
            "expecting a clearer explanation of the remaining work, the staffing "
            "gaps, and the decisions that had been delayed. A revised plan was "
            "requested so everyone could understand the tradeoffs and agree on the "
            "next steps together before the review meeting."
        ),
        "answers": [
            "The schedule needs revision because staffing gaps and delayed decisions left remaining work unresolved. The team should explain tradeoffs and agree on next steps before the review meeting.",
            "The schedule needs revision because staffing gaps and delayed decisions threaten the deadline and budget. The team should clarify remaining work and tradeoffs before stakeholders agree on next steps.",
            "A revised plan should address staffing gaps and delayed decisions. These issues threaten the deadline and budget. The team should clarify remaining work and tradeoffs before stakeholders agree on next steps.",
            "A revised plan should address staffing gaps and delayed decisions that threaten the deadline and budget. The team should clarify remaining work and tradeoffs before stakeholders agree on next steps.",
        ],
        "content": [
            {
                "required_facts": [
                    {
                        "name": "revised_plan",
                        "any_of": ["schedule needs revision", "revised plan"],
                    },
                    {"name": "remaining_work", "any_of": ["remaining work"]},
                    {"name": "tradeoffs", "any_of": ["tradeoffs"]},
                ]
            }
        ]
        * 4,
        "rules": [
            ("Use fewer than 40 words", "max_words", 39),
            ('Keep the words "deadline" and "budget"', "required_terms", ["deadline", "budget"]),
            ('Do not use "was", "were", or "been"', "forbidden_terms", ["was", "were", "been"]),
            ("Use exactly 2 sentences", "exact_sentences", 2),
        ],
        "hotspot": "short_rewrite_with_banned_verbs",
    },
    {
        "slug": "message_classification",
        "title": "Classify messages as spam",
        "carrier": "classification",
        "split": "test",
        "instruction": "Classify each support ticket and build a JSON routing report.",
        "source": (
            " | ".join(f"{item_id}. {text}" for item_id, text, _, _ in TICKETS)
        ),
        "answers": [
            _json(TICKET_L1), _json(TICKET_L2), _json(TICKET_L3), _json(TICKET_L4),
        ],
        "content": [
            {
                "exact_json": value
            }
            for value in [TICKET_L1, TICKET_L2, TICKET_L3, TICKET_L4]
        ],
        "rules": [
            (
                "Return one valid JSON array with objects containing id and category",
                "json_only",
                "array",
            ),
            (
                "Add priority to every ticket using urgent, high, or normal",
                "json_array_required_keys",
                ["id", "category", "priority"],
            ),
            (
                "Sort urgent tickets first, then high, then normal; sort by ID inside each group",
                "json_array_sorted_by",
                [{"field": "priority", "order": ["urgent", "high", "normal"]}, {"field": "id", "direction": "ascending"}],
            ),
            (
                "Append one summary object with the count for each category",
                "json_summary_counts",
                {"field": "category", "summary_key": "summary"},
            ),
        ],
    },
    {
        "slug": "vendor_email",
        "title": "Decline a vendor renewal",
        "carrier": "prose",
        "split": "test",
        "instruction": "Draft an email declining Sam's vendor renewal proposal.",
        "source": "Alex's team will decline the renewal because next year's software budget is 12% lower and product usage fell 38%. The current contract ends August 31. Data export should finish by August 15, and access should remain available through August 31. The team will reassess its needs in Q3.",
        "answers": [
            "Hi Sam,\n\nThank you for the renewal proposal. We have reviewed it and will not renew the service for the next term. We will reassess our needs in Q3.\n\nRegards, Alex",
            "Hi Sam,\n\nThank you for the renewal proposal and your support this year. We have decided not to renew because next year's software budget is 12% lower and our product usage has fallen 38%. We will reassess our needs in Q3 and contact you if our requirements change.\n\nRegards, Alex",
            "Hi Sam,\n\nThank you for the renewal proposal. We will not renew because next year's software budget is 12% lower and product usage fell 38%. Please keep access available through August 31 while we complete our data export by August 15. We will reassess our needs in Q3.\n\nRegards, Alex",
            "Hi Sam,\n\nThank you for the renewal proposal. We will not renew because next year's software budget is 12% lower and product usage fell 38%.\n\nPlease keep access available through August 31 while we complete our data export by August 15. We will reassess our needs in Q3.\n\nRegards, Alex",
        ],
        "content": [
            {
                "required_facts": [
                    {
                        "name": "decline",
                        "any_of": [
                            "declining the renewal",
                            "not to continue",
                            "will not be renewing",
                            "not be renewing",
                            "will not renew",
                            "decline",
                        ],
                    },
                ]
            },
            {
                "required_facts": [
                    {"name": "decline", "any_of": ["will not renew", "not renew", "not to renew", "declining the renewal", "not to continue"]},
                    {"name": "reasons", "any_of": ["budget is 12% lower", "budget is 12 percent lower"]},
                ]
            },
            {
                "required_facts": [
                    {"name": "decline", "any_of": ["will not renew", "not renew", "declining the renewal", "not to continue"]},
                    {"name": "reasons", "any_of": ["budget is 12% lower", "budget is 12 percent lower"]},
                    {"name": "transition", "any_of": ["August 15"]},
                ]
            },
            {
                "required_facts": [
                    {"name": "decline", "any_of": ["will not renew", "not renew", "declining the renewal", "not to continue"]},
                    {"name": "reasons", "any_of": ["budget is 12% lower", "budget is 12 percent lower"]},
                    {"name": "transition", "any_of": ["August 15"]},
                ]
            },
        ],
        "rules": [
            (
                'Start with "Hi Sam," and end with "Regards, Alex"',
                "boundary",
                {"prefix": "Hi Sam,", "suffix": "Regards, Alex"},
            ),
            ("Use between 45 and 80 words", "word_range", {"min": 45, "max": 80}),
            (
                'Include "August 15", "August 31", and "Q3" but never "unfortunately"',
                "required_forbidden_terms",
                {
                    "required": ["August 15", "August 31", "Q3"],
                    "forbidden": ["unfortunately"],
                },
            ),
            (
                "Use exactly 4 paragraphs separated by blank lines",
                "exact_paragraphs",
                4,
            ),
        ],
        "hotspot": "word_range_with_paragraph_structure",
    },
    {
        "slug": "service_yaml",
        "title": "Create a service YAML config",
        "carrier": "structured_yaml",
        "split": "test",
        "instruction": "Create YAML configuration for the Atlas API deployment.",
        "source": "Service atlas-api uses image registry.example/atlas:2.4, listens on port 8080, runs 3 replicas in ap-south-1, and exposes /health every 30 seconds.",
        "answers": [
            "service: atlas-api\nimage: registry.example/atlas:2.4\nport: 8080",
            "service: atlas-api\nimage: registry.example/atlas:2.4\nport: 8080\nreplicas: 2\nregion: ap-south-1",
            "service: atlas-api\nimage: registry.example/atlas:2.4\nport: 8080\nreplicas: 3\nregion: ap-south-1",
            "service: atlas-api\nimage: registry.example/atlas:2.4\nport: 8080\nreplicas: 3\nregion: ap-south-1\nhealthcheck:\n  path: /health\n  interval_seconds: 30",
        ],
        "content": [
            {"exact_yaml": {"service": "atlas-api", "image": "registry.example/atlas:2.4", "port": 8080}},
            {"exact_yaml": {"service": "atlas-api", "image": "registry.example/atlas:2.4", "port": 8080, "replicas": 2, "region": "ap-south-1"}},
            {"exact_yaml": {"service": "atlas-api", "image": "registry.example/atlas:2.4", "port": 8080, "replicas": 3, "region": "ap-south-1"}},
            {"exact_yaml": {"service": "atlas-api", "image": "registry.example/atlas:2.4", "port": 8080, "replicas": 3, "region": "ap-south-1", "healthcheck": {"path": "/health", "interval_seconds": 30}}},
        ],
        "rules": [
            ("Return valid YAML only, with no prose", "yaml_only", True),
            (
                "Add the top-level deployment fields replicas and region",
                "required_top_level_keys",
                ["service", "image", "port", "replicas", "region"],
            ),
            (
                "Set port to 8080, replicas to 3, and region to ap-south-1",
                "yaml_field_constraints",
                {
                    "port": {"type": "integer", "equals": 8080},
                    "replicas": {"type": "integer", "equals": 3},
                    "region": {"type": "string", "equals": "ap-south-1"},
                },
            ),
            (
                "Add a healthcheck at /health with a 30-second interval",
                "yaml_healthcheck",
                {"path": "/health", "interval_seconds": 30},
            ),
        ],
    },
    {
        "slug": "point_ordering",
        "title": "Order and transform scored words",
        "carrier": "ordering",
        "split": "test",
        "instruction": "Order all fifteen project codes by score from highest to lowest.",
        "source": ", ".join(f"{name}={score}" for name, score in POINTS.items()) + ".",
        "answers": [
            "indigo,cobalt,maple,fable,juniper,birch,orbit,grove,delta,nova,kite,ember,amber,linen,harbor",
            "INDIGO,COBALT,MAPLE,FABLE,JUNIPER,BIRCH,ORBIT,GROVE,DELTA,NOVA,KITE,EMBER,AMBER,LINEN,HARBOR",
            "1:INDIGO,2:COBALT,3:MAPLE,4:FABLE,5:JUNIPER,6:BIRCH,7:ORBIT,8:GROVE,9:DELTA,10:NOVA,11:KITE,12:EMBER,13:AMBER,14:LINEN,15:HARBOR",
            "1:COBALT,2:INDIGO,3:FABLE,4:MAPLE,5:BIRCH,6:JUNIPER,7:ORBIT,8:DELTA,9:GROVE,10:KITE,11:NOVA,12:AMBER,13:EMBER,14:HARBOR,15:LINEN",
        ],
        "content": [
            {
                "required_values": {
                    "values": list(POINTS),
                    "separator": ",",
                    "strip_prefix": "",
                }
            }
        ]
        * 4,
        "rules": [
            (
                "Return all words ordered by point value, comma-separated",
                "sorted_by_points",
                POINTS,
            ),
            ("Write every word in uppercase", "uppercase_items", True),
            ('Prefix every word with its rank, such as "1:WORD"', "ranked_items", True),
            (
                "Break equal-point ties alphabetically",
                "ties_alphabetical",
                POINTS,
            ),
        ],
    },
]


def build_document() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for task in TASKS:
        base_id = f'constraint_{task["slug"]}_001'
        for level in range(1, 5):
            selected_rules = task["rules"][:level]
            constraints = " ".join(
                f"{index}. {description}."
                for index, (description, _, _) in enumerate(selected_rules, start=1)
            )
            prompt_parts = [task["instruction"]]
            if task["source"]:
                prompt_parts.append(f'Source data: {task["source"]}')
            prompt_parts.append(f"Mandatory constraints: {constraints}")
            tags = [
                "instruction_following",
                f"constraint_load_{level}",
                f'{task["carrier"]}_carrier',
                f'{task["slug"]}_base',
            ]
            if task.get("hotspot"):
                tags.append(task["hotspot"])
            item: dict[str, Any] = {
                "id": f'constraint_{task["slug"]}_{level:03d}',
                "subcategory": SUBCATEGORY_BY_LEVEL[level],
                "difficulty": DIFFICULTY_BY_LEVEL[level],
                "split": task["split"],
                "prompt": " ".join(prompt_parts),
                "response_contract": {"type": "text", "format": task["carrier"]},
                "expected": {"value": task["answers"][level - 1]},
                "scoring": {
                    "method": "constraint_rules",
                    "parameters": {
                        "content_requirements": deepcopy(task["content"][level - 1]),
                        "rules": {
                            rule_key: deepcopy(rule_value)
                            for _, rule_key, rule_value in selected_rules
                        },
                    },
                },
                "provenance": {
                    "kind": "synthetic",
                    "review_status": "human_checked",
                    "generator": GENERATOR_ID,
                    "seed": 20260722,
                },
                "tags": tags,
            }
            if level > 1:
                item["variant_of"] = base_id
            items.append(item)
    return {
        "schema_version": 1,
        "benchmark": "constraint_load_curve",
        "generated_by": GENERATOR_ID,
        "seed": 20260722,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = build_document()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
