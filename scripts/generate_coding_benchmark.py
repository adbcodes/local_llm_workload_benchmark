from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "code_debug_repair" / "generated_questions.yaml"
GENERATOR = "generate_coding_benchmark.py"
SEED = 20260629

# The 80-item final set deliberately retires narrow or redundant exercises from
# the earlier 100-item draft. Keeping the selection here makes the reduction
# reproducible without ever editing generated YAML or JSONL by hand.
RETIRED_ITEM_IDS = {
    # Repetitive standalone implementations.
    "code_active_usernames_001",
    "code_alert_streaks_001",
    "code_allocate_stock_fairly_001",
    "code_data_drift_001",
    "code_filter_compatible_versions_001",
    "code_group_status_runs_001",
    "code_inventory_totals_001",
    "code_recurring_slots_001",
    "code_rooms_required_001",
    "code_snapshot_changes_001",
    # Obvious or duplicate diagnosis cases.
    "diagnose_average_latency_001",
    "diagnose_daily_totals_001",
    "diagnose_first_valid_row_001",
    "diagnose_ranked_feed_001",
    # Narrow repairs duplicated by stronger state/boundary cases.
    "repair_clamp_percentage_001",
    "repair_compact_ranges_001",
    "repair_reconcile_orders_001",
    "repair_window_peaks_001",
    # One invalid selection and one boundary case duplicated elsewhere.
    "test_latest_duplicate_001",
    "test_touching_windows_001",
}

IMPLEMENTATION_CONTEXT_BY_TAG = (
    ("dependency_propagation", "a deployment-planning service"),
    ("graph", "an internal workflow service"),
    ("policy_resolution", "an access and rollout policy service"),
    ("reconciliation", "a back-office reconciliation job"),
    ("scheduling", "an operations scheduling service"),
    ("event_processing", "an event-consumer worker"),
    ("routing", "an incident-routing service"),
    ("parsing", "a repository configuration loader"),
    ("records", "an internal records pipeline"),
    ("aggregation", "an operational reporting job"),
    ("filtering", "an internal data-processing service"),
)


REALISM_IMPLEMENTATION_PROMPTS = {
    "code_validate_invoice_filename_001": (
        "quick helper from our accounts download job: people keep dropping renamed or backup files into the invoice folder.",
        "The file already imports re above this helper. Reply with exactly the is_invoice_filename function and don't repeat the import.",
    ),
    "code_extract_ticket_mentions_001": (
        "we paste incident notes from Slack into a triage script and need to pull the ticket references back out.",
        "re is already imported in this module. Output only the extract_ticket_mentions function definition, without another import.",
    ),
    "code_rename_phone_photos_001": (
        "one-off cleanup script: a site visit left me with a list of phone-photo filenames that need predictable names before upload.",
        "Don't change the input list. Send back only the rename_phone_photos function, with no imports or example calls.",
    ),
    "code_validate_release_tag_001": (
        "our CI job is accepting a few malformed release tags, so I need the check isolated in a tiny helper.",
        "This file already has import re above the helper. Return just the is_release_tag function, with no Markdown or explanation.",
    ),
    "code_transform_expense_csv_001": (
        "got a small expense CSV export, already parsed into rows, and I need to turn it into the shape our reimbursement upload accepts.",
        "Leave rows untouched and return only the transform_expense_rows function definition. No imports or surrounding prose.",
    ),
    "code_extract_log_timestamps_001": (
        "pulled a mixed bag of worker log lines during an incident; can you write the little parser I can paste into our cleanup script?",
        "Preserve lines exactly and provide only the extract_log_timestamps function. The answer must contain no imports or usage example.",
    ),
    "code_extract_error_codes_001": (
        "need a quick helper for an incident summary: the raw lines have repeated error codes mixed with punctuation and other text.",
        "Do not mutate lines. Reply solely with the extract_error_codes function definition and do not import anything.",
    ),
}


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    name: str
    params: str
    specification: str
    body: str
    cases: list[list[Any]]
    tags: list[str]


@dataclass(frozen=True)
class Repair:
    id: str
    difficulty: str
    name: str
    params: str
    specification: str
    body: str
    cases: list[list[Any]]
    mutations: list[tuple[str, str, str]]
    tags: list[str]


def _source(task: Task) -> str:
    body = "\n".join("    " + line if line else "" for line in task.body.splitlines())
    return f"def {task.name}({task.params}):\n{body}"


def _expected(source: str, name: str, args: list[Any]) -> Any:
    namespace: dict[str, Any] = {"re": re}
    exec(source, namespace, namespace)
    return namespace[name](*json.loads(json.dumps(args)))


def _observed(source: str, name: str, args: list[Any]) -> dict[str, Any]:
    """Return a deterministic, JSON-safe outcome for a generated bug report."""
    namespace: dict[str, Any] = {}
    try:
        exec(source, namespace, namespace)
        value = namespace[name](*json.loads(json.dumps(args)))
    except Exception as error:  # generated sources intentionally contain bugs
        return {"raised": type(error).__name__, "message": str(error)}
    return {"returned": value}


def _implementation_context(task: Task) -> str:
    for tag, context in IMPLEMENTATION_CONTEXT_BY_TAG:
        if tag in task.tags:
            return context
    return "a small internal Python service"


def _implementation_prompt(task: Task, index: int) -> str:
    realism_parts = REALISM_IMPLEMENTATION_PROMPTS.get(task.id)
    if realism_parts is not None:
        opener, output_rule = realism_parts
        return (
            f"{opener}\n\n"
            f"Function: {task.name}({task.params})\n"
            f"Behavior: {task.specification}\n\n"
            f"{output_rule}"
        )

    context = _implementation_context(task)
    introductions = (
        f"A maintenance ticket for {context} needs this missing helper.",
        f"Complete a small utility used by {context}.",
        f"A regression-safe change in {context} requires this helper.",
        f"Implement the repository helper below for {context}.",
    )
    return (
        f"{introductions[index % len(introductions)]}\n\n"
        f"Function: {task.name}({task.params})\n"
        f"Contract: {task.specification}\n\n"
        "Keep the public signature unchanged and do not mutate inputs. "
        "Return only the function definition and use no imports."
    )


def _additional_implementation_tasks() -> list[Task]:
    """Fresh practical tasks added for broader local-assistant coverage."""
    return [
        Task(
            "code_validate_invoice_filename_001", "easy", "is_invoice_filename", "name",
            "Return whether name exactly matches INV-YYYY-NNNNN.pdf, with uppercase INV, a four-digit year, five-digit sequence, and lowercase .pdf.",
            "return re.fullmatch(r'INV-[0-9]{4}-[0-9]{5}\\.pdf', name) is not None",
            [["INV-2026-00817.pdf"], ["inv-2026-00817.pdf"], ["INV-26-00817.pdf"], ["INV-2026-00817.pdf.bak"]],
            ["regex", "parsing", "filename_validation"],
        ),
        Task(
            "code_extract_ticket_mentions_001", "easy", "extract_ticket_mentions", "text",
            "Use a regex to return every standalone INC- or REQ- ticket ID with 4 to 6 digits, in occurrence order. Keep duplicates. Do not match IDs embedded inside letters or digits.",
            "return re.findall(r'(?<![A-Za-z0-9])(?:INC|REQ)-[0-9]{4,6}(?![A-Za-z0-9])', text)",
            [["retry INC-4821, then REQ-770044"], ["INC-123 REQ-1234567 xINC-9999"], [""], ["INC-4821 / INC-4821"]],
            ["regex", "parsing", "ticket_ids"],
        ),
        Task(
            "code_rename_phone_photos_001", "easy", "rename_phone_photos", "names, prefix",
            "Return new filenames in input order. Rename only IMG_YYYYMMDD_NNNN.JPG files to prefix-YYYY-MM-DD-NNNN.jpg; leave every nonmatching filename unchanged. Do not mutate names.",
            "result = []\nfor name in names:\n    if len(name) == 21 and name.startswith('IMG_') and name.endswith('.JPG'):\n        date = name[4:12]\n        sequence = name[13:17]\n        if name[12] == '_' and date.isdigit() and sequence.isdigit():\n            name = prefix + '-' + date[:4] + '-' + date[4:6] + '-' + date[6:] + '-' + sequence + '.jpg'\n    result.append(name)\nreturn result",
            [[["IMG_20260803_0042.JPG", "notes.txt"], "site-a"], [[], "trip"], [["IMG_2026083_0001.JPG"], "field"], [["img_20260803_0042.jpg"], "trip"]],
            ["one_off_script", "files", "immutable_input"],
        ),
        Task(
            "code_validate_release_tag_001", "medium", "is_release_tag", "tag",
            "Return whether tag exactly matches service-name-vMAJOR.MINOR.PATCH. service-name is lowercase alphanumeric words joined by single hyphens and starts with a letter. Version components are 0 or a nonzero digit followed by digits, so leading zeroes are invalid.",
            "pattern = r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*-v(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)'\nreturn re.fullmatch(pattern, tag) is not None",
            [["billing-api-v2.14.3"], ["billing-api-v02.14.3"], ["Billing-api-v2.14.3"], ["a-v0.0.0"], ["billing--api-v1.2.3"]],
            ["regex", "parsing", "release_automation"],
        ),
        Task(
            "code_transform_expense_csv_001", "medium", "transform_expense_rows", "rows",
            "rows is CSV data represented as lists. The first row is a header containing date, category, amount, and optional extra columns. Return a new table with header [date,category,amount_paise]. For later rows with the same width, trim date/category, accept non-negative amounts written as a whole number or with one or two decimal places, and convert them to integer paise. Skip blank, malformed, or negative amounts. Preserve valid row order and do not mutate rows.",
            "if not rows:\n    return [['date', 'category', 'amount_paise']]\nheader = [value.strip().lower() for value in rows[0]]\ntry:\n    date_i = header.index('date')\n    category_i = header.index('category')\n    amount_i = header.index('amount')\nexcept ValueError:\n    return [['date', 'category', 'amount_paise']]\nresult = [['date', 'category', 'amount_paise']]\nfor row in rows[1:]:\n    if len(row) != len(rows[0]):\n        continue\n    raw = row[amount_i].strip()\n    parts = raw.split('.')\n    if len(parts) > 2 or not parts[0].isdigit() or (len(parts) == 2 and (not parts[1].isdigit() or len(parts[1]) > 2)):\n        continue\n    paise = int(parts[0]) * 100 + (int(parts[1].ljust(2, '0')) if len(parts) == 2 else 0)\n    result.append([row[date_i].strip(), row[category_i].strip(), paise])\nreturn result",
            [[[["date", "category", "amount", "note"], [" 2026-08-01 ", " Taxi ", "418.5", "airport"], ["2026-08-02", "Meals", "-20", "refund"], ["2026-08-03", "Stay", "2400.00", ""]]], [[]], [[["amount", "date", "category"], ["7", "2026-01-01", "Tea"]]], [[["date", "category", "amount"], ["2026-01-01", "Tea", "7.123"], ["2026-01-02", "Bus", "0"]]]],
            ["one_off_script", "csv", "parsing", "immutable_input"],
        ),
        Task(
            "code_extract_log_timestamps_001", "medium", "extract_log_timestamps", "lines",
            "Return [timestamp, message] for each line that starts with a bracketed UTC timestamp in exactly [YYYY-MM-DDTHH:MM:SS.mmmZ] form followed by one space. Preserve order and the message text after that space; ignore all other lines.",
            "result = []\nfor line in lines:\n    if len(line) < 26 or line[0] != '[' or line[25:27] != '] ':\n        continue\n    stamp = line[1:25]\n    if stamp[4] != '-' or stamp[7] != '-' or stamp[10] != 'T' or stamp[13] != ':' or stamp[16] != ':' or stamp[19] != '.' or stamp[23] != 'Z':\n        continue\n    digits = stamp[:4] + stamp[5:7] + stamp[8:10] + stamp[11:13] + stamp[14:16] + stamp[17:19] + stamp[20:23]\n    if digits.isdigit():\n        result.append([stamp, line[27:]])\nreturn result",
            [[["[2026-08-03T09:14:07.118Z] worker started", "noise", "[2026-08-03T09:14:08.002Z] retry 1"]], [[]], [["[2026-8-03T09:14:07.118Z] bad"]], [["[2026-08-03T09:14:07.118Z] "]]],
            ["log_parsing", "timestamps", "raw_logs"],
        ),
        Task(
            "code_extract_error_codes_001", "medium", "extract_error_codes", "lines",
            "From raw log lines, return first occurrences of standalone error codes shaped E- followed by exactly five digits, in encounter order. A line may contain several codes. Do not match a code touching a letter or digit.",
            "result = []\nfor line in lines:\n    for index in range(max(0, len(line) - 6)):\n        token = line[index:index + 7]\n        left_ok = index == 0 or not line[index - 1].isalnum()\n        right_ok = index + 7 == len(line) or not line[index + 7].isalnum()\n        if token.startswith('E-') and token[2:].isdigit() and left_ok and right_ok and token not in result:\n            result.append(token)\nreturn result",
            [[["WARN retry after E-10422; upstream E-88201", "E-10422 repeated"]], [[]], [["xE-12345 E-1234 E-123456"]], [["codes=E-00001/E-00002", "tail E-00003."]]],
            ["log_parsing", "error_codes", "raw_logs"],
        ),
        Task(
            "code_normalize_phonebook_001", "easy", "normalize_phonebook", "records, country_code",
            "Trim names and phone numbers, remove spaces and hyphens from numbers, prefix local 10-digit numbers with country_code, ignore blank names or numbers, and let later names win. Return a name-sorted dictionary.",
            "values = {}\nfor name, phone in records:\n    key = name.strip()\n    number = phone.replace(' ', '').replace('-', '')\n    if key and number:\n        if len(number) == 10 and number.isdigit():\n            number = country_code + number\n        values[key] = number\nreturn {key: values[key] for key in sorted(values)}",
            [
                [[[' Mira ', '98765-43210'], ['Zed', '+44 20-1234'], ['Mira', '11111 22222']], '+91'],
                [[], '+1'],
                [[['', '123'], ['A', '']], '+1'],
                [[['B', '1234567890']], '+1'],
            ],
            ["normalization", "records"],
        ),
        Task(
            "code_merge_preferences_001", "easy", "merge_preferences", "defaults, user, locked",
            "Return a new preference dictionary. Start with defaults, apply user values except for locked keys, and include locked keys only when present in defaults. Preserve all inputs.",
            "result = dict(defaults)\nlocked_set = set(locked)\nfor key, value in user.items():\n    if key not in locked_set:\n        result[key] = value\nreturn {key: result[key] for key in sorted(result)}",
            [[{"theme": "light", "region": "in"}, {"theme": "dark", "region": "us"}, ["region"]], [{}, {"x": 1}, []], [{"x": 1}, {"x": 2}, ["x"]], [{"a": 1}, {}, ["missing"]]],
            ["immutable_update", "policy_resolution"],
        ),
        Task(
            "code_summarize_http_statuses_001", "easy", "summarize_http_statuses", "codes, include_zero",
            "Count 2xx as success, 4xx as client_error, 5xx as server_error, and everything else as other. Return keys in that order, omitting zero-count groups unless include_zero is true.",
            "counts = {'success': 0, 'client_error': 0, 'server_error': 0, 'other': 0}\nfor code in codes:\n    if 200 <= code < 300:\n        key = 'success'\n    elif 400 <= code < 500:\n        key = 'client_error'\n    elif 500 <= code < 600:\n        key = 'server_error'\n    else:\n        key = 'other'\n    counts[key] += 1\nreturn {key: counts[key] for key in ['success', 'client_error', 'server_error', 'other'] if include_zero or counts[key]}",
            [[[200, 204, 404, 503], False], [[], True], [[301, 418], False], [[500, 599, 600], True]],
            ["classification", "aggregation"],
        ),
        Task(
            "code_expiring_items_001", "easy", "expiring_items", "records, now",
            "Keep active records whose integer expires_at is at or before now. Return their ids ordered by expires_at then id. Do not mutate records.",
            "rows = [record for record in records if record['active'] and record['expires_at'] <= now]\nreturn [record['id'] for record in sorted(rows, key=lambda record: (record['expires_at'], record['id']))]",
            [[[{"id": "b", "expires_at": 5, "active": True}, {"id": "a", "expires_at": 5, "active": True}], 5], [[], 3], [[{"id": "x", "expires_at": 2, "active": False}], 5], [[{"id": "x", "expires_at": 9, "active": True}], 8]],
            ["filtering", "tie_breaking"],
        ),
        Task(
            "code_coalesce_notes_001", "easy", "coalesce_notes", "lines, separator",
            "Trim each note, discard blank notes, collapse consecutive duplicate notes case-insensitively, and join the retained original-cased notes with separator.",
            "result = []\nfor line in lines:\n    value = line.strip()\n    if value and (not result or result[-1].casefold() != value.casefold()):\n        result.append(value)\nreturn separator.join(result)",
            [[[" Ready ", "ready", "", "Ship"], " | "], [[], ","], [["A", "a", "A"], "-"], [[" x "], "/"]],
            ["text_processing", "deduplication"],
        ),
        Task(
            "code_invoice_totals_001", "medium", "invoice_totals", "lines, discounts",
            "Lines are [invoice_id, quantity, unit_price]. Sum each invoice, then apply its integer percentage discount once to the total using floor division. Return [invoice_id,total] rows ordered by id.",
            "totals = {}\nfor invoice, quantity, price in lines:\n    totals[invoice] = totals.get(invoice, 0) + quantity * price\nresult = []\nfor invoice in sorted(totals):\n    discount = discounts.get(invoice, 0)\n    result.append([invoice, totals[invoice] * (100 - discount) // 100])\nreturn result",
            [
                [[['b', 2, 50], ['a', 1, 99], ['b', 1, 25]], {'b': 10}],
                [[], {}],
                [[['x', 3, 10]], {'x': 100}],
                [[['x', 1, 101]], {'x': 25}],
            ],
            ["aggregation", "percentage", "records"],
        ),
        Task(
            "code_feature_rollout_001", "medium", "feature_rollout", "users, rules, overrides",
            "Rules are [region, minimum_score]. A user is enabled when any matching-region rule is met; an override by user id wins. Return enabled user ids sorted.",
            "enabled = []\nfor user in users:\n    value = any(user['region'] == region and user['score'] >= minimum for region, minimum in rules)\n    if user['id'] in overrides:\n        value = overrides[user['id']]\n    if value:\n        enabled.append(user['id'])\nreturn sorted(enabled)",
            [[[{"id": "b", "region": "in", "score": 8}, {"id": "a", "region": "us", "score": 4}], [["in", 7], ["us", 5]], {"a": True}], [[], [], {}], [[{"id": "x", "region": "in", "score": 9}], [["in", 5]], {"x": False}], [[{"id": "x", "region": "eu", "score": 1}], [], {"x": True}]],
            ["policy_resolution", "overrides"],
        ),
        Task(
            "code_first_capacity_breach_001", "medium", "first_capacity_breach", "capacity, changes",
            "Changes are [time, delta]. Combine changes at the same time, process times ascending, and return the first [time,total] whose running total exceeds capacity, or None.",
            "by_time = {}\nfor time, delta in changes:\n    by_time[time] = by_time.get(time, 0) + delta\ntotal = 0\nfor time in sorted(by_time):\n    total += by_time[time]\n    if total > capacity:\n        return [time, total]\nreturn None",
            [[3, [[1, 2], [2, 2]]], [3, [[1, 4], [1, -1]]], [0, []], [5, [[2, 6], [1, -2]]]],
            ["event_processing", "aggregation", "boundaries"],
        ),
        Task(
            "code_correlate_requests_001", "medium", "correlate_requests", "requests, responses",
            "Requests are [id,start]; responses are [id,end,status]. Use the latest response for each id. Return [id,latency,status] for matched ids, ordered by descending latency then id.",
            "starts = {request_id: start for request_id, start in requests}\nlatest = {}\nfor request_id, end, status in responses:\n    latest[request_id] = [end, status]\nrows = [[request_id, latest[request_id][0] - start, latest[request_id][1]] for request_id, start in starts.items() if request_id in latest]\nreturn sorted(rows, key=lambda row: (-row[1], row[0]))",
            [[[["a", 2], ["b", 5]], [["a", 7, 200], ["b", 8, 500]]], [[], []], [[['x', 1]], [['x', 3, 202], ['x', 5, 200]]], [[['x', 1]], [['y', 2, 200]]]],
            ["join", "latency", "tie_breaking"],
        ),
        Task(
            "code_partition_batches_001", "medium", "partition_batches", "items, limit",
            "Items are [id,size] in processing order and each size is at most limit. Greedily fill a batch until the next item would exceed limit. Return batches of ids and preserve order.",
            "batches = []\ncurrent = []\nused = 0\nfor item_id, size in items:\n    if current and used + size > limit:\n        batches.append(current)\n        current = []\n        used = 0\n    current.append(item_id)\n    used += size\nif current:\n    batches.append(current)\nreturn batches",
            [[[["a", 2], ["b", 3], ["c", 2]], 5], [[], 3], [[['x', 5]], 5], [[['a', 4], ['b', 2]], 5]],
            ["greedy", "batching", "boundaries"],
        ),
        Task(
            "code_inventory_events_001", "medium", "inventory_events", "opening, events",
            "Apply [event_id,sku,delta] events once per unique event id, keeping the first occurrence. Reject an event that would make stock negative. Return [rejected_ids, sorted_stock].",
            "stock = dict(opening)\nseen = set()\nrejected = []\nfor event_id, sku, delta in events:\n    if event_id in seen:\n        continue\n    seen.add(event_id)\n    value = stock.get(sku, 0) + delta\n    if value < 0:\n        rejected.append(event_id)\n    else:\n        stock[sku] = value\nreturn [rejected, {sku: stock[sku] for sku in sorted(stock)}]",
            [[{"a": 3}, [["e1", "a", -2], ["e2", "a", -2]]], [{}, []], [{}, [["e", "x", 2], ["e", "x", 9]]], [{"b": 1, "a": 0}, []]],
            ["reconciliation", "deduplication", "state_tracking"],
        ),
        Task(
            "code_service_health_001", "medium", "service_health", "services, checks",
            "Checks are [service,check,passed]. A service is healthy only if it has every required check and all latest check results pass. Return [healthy_ids, unhealthy_ids], each sorted.",
            "latest = {}\nfor service, check, passed in checks:\n    latest[[service, check][0] + '\\0' + check] = passed\nhealthy = []\nunhealthy = []\nfor service, required in services.items():\n    ok = all((service + '\\0' + check) in latest and latest[service + '\\0' + check] for check in required)\n    (healthy if ok else unhealthy).append(service)\nreturn [sorted(healthy), sorted(unhealthy)]",
            [[{"api": ["ping", "db"], "web": ["ping"]}, [["api", "ping", True], ["api", "db", False], ["web", "ping", True]]], [{}, []], [{"x": []}, []], [{"x": ["a"]}, [["x", "a", False], ["x", "a", True]]]],
            ["latest_state", "completeness", "classification"],
        ),
        Task(
            "code_rank_search_results_001", "medium", "rank_search_results", "results, blocked",
            "Ignore blocked ids and keep the greatest score per id, with equal scores keeping the later title. Return [id,title,score] rows ordered by score descending then id.",
            "blocked_set = set(blocked)\nchosen = {}\nfor item_id, title, score in results:\n    if item_id not in blocked_set and (item_id not in chosen or score >= chosen[item_id][1]):\n        chosen[item_id] = [title, score]\nrows = [[item_id] + chosen[item_id] for item_id in chosen]\nreturn sorted(rows, key=lambda row: (-row[2], row[0]))",
            [[[["b", "B", 8], ["a", "old", 8], ["a", "new", 8]], []], [[], []], [[['x', 'X', 3]], ['x']], [[['x', 'low', 2], ['x', 'high', 5]], []]],
            ["ranking", "deduplication", "tie_breaking"],
        ),
        Task(
            "code_recurring_slots_001", "medium", "recurring_slots", "start, count, step, blackouts",
            "Generate count candidate times start+i*step. Remove candidates inside any inclusive blackout [left,right]. Return the remaining times.",
            "result = []\nfor index in range(count):\n    time = start + index * step\n    if not any(left <= time <= right for left, right in blackouts):\n        result.append(time)\nreturn result",
            [[10, 5, 3, [[13, 16]]], [0, 0, 2, []], [5, 1, 9, [[5, 5]]], [0, 4, 2, [[1, 1]]]],
            ["scheduling", "filtering", "boundaries"],
        ),
        Task(
            "code_membership_events_001", "medium", "membership_events", "snapshot, events",
            "Apply [user,action,group] events where action is add or remove. Ignore duplicate effects. Return a user-sorted dictionary of sorted non-empty group lists.",
            "state = {user: set(groups) for user, groups in snapshot.items()}\nfor user, action, group in events:\n    groups = state.setdefault(user, set())\n    if action == 'add':\n        groups.add(group)\n    else:\n        groups.discard(group)\nreturn {user: sorted(groups) for user, groups in sorted(state.items()) if groups}",
            [[{"u": ["a"]}, [["u", "add", "b"], ["u", "remove", "a"]]], [{}, []], [{}, [["x", "remove", "a"]]], [{"b": ["z"], "a": ["y"]}, []]],
            ["event_processing", "sets", "immutable_update"],
        ),
        Task(
            "code_route_incidents_001", "medium", "route_incidents", "incidents, on_call",
            "Each incident has id, team, severity, and open. Keep open incidents with a known on-call owner. Return [owner,id,severity] ordered by severity critical/high/medium/low then id.",
            "order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}\nrows = []\nfor incident in incidents:\n    if incident['open'] and incident['team'] in on_call:\n        rows.append([on_call[incident['team']], incident['id'], incident['severity']])\nreturn sorted(rows, key=lambda row: (order[row[2]], row[1]))",
            [[[{"id": "i2", "team": "api", "severity": "low", "open": True}, {"id": "i1", "team": "db", "severity": "high", "open": True}], {"api": "A", "db": "B"}], [[], {}], [[{"id": "i", "team": "x", "severity": "critical", "open": False}], {"x": "A"}], [[{"id": "i", "team": "x", "severity": "medium", "open": True}], {}]],
            ["routing", "priority_order", "filtering"],
        ),
        Task(
            "code_release_waves_001", "hard", "release_waves", "services, dependencies, blocked",
            "Dependencies are [service,prerequisite]. Exclude blocked services and every service transitively depending on one. Return remaining services in sorted parallel release waves, or [] for a cycle.",
            "excluded = set(blocked)\nprogress = True\nwhile progress:\n    progress = False\n    for service, prerequisite in dependencies:\n        if prerequisite in excluded and service not in excluded:\n            excluded.add(service)\n            progress = True\nremaining = set(services) - excluded\nresult = []\ndone = set()\nwhile remaining:\n    ready = sorted(service for service in remaining if all(prerequisite in done or prerequisite in excluded for owner, prerequisite in dependencies if owner == service))\n    if not ready:\n        return []\n    result.append(ready)\n    done.update(ready)\n    remaining.difference_update(ready)\nreturn result",
            [[['db', 'api', 'web'], [['api', 'db'], ['web', 'api']], []], [['a', 'b', 'c'], [['b', 'a'], ['c', 'b']], ['a']], [['a', 'b'], [['a', 'b'], ['b', 'a']], []], [[], [], []]],
            ["graph", "dependency_propagation", "batching"],
        ),
        Task(
            "code_permission_closure_001", "hard", "permission_closure", "user_roles, inherits, grants, denies",
            "Roles inherit other roles through [role,parent] edges. For each user, collect transitive grants from all roles, then remove user-specific denies. Return a user-sorted dictionary of sorted permissions; cycles must terminate.",
            "parents = {}\nfor role, parent in inherits:\n    parents.setdefault(role, []).append(parent)\nresult = {}\nfor user, roles in user_roles.items():\n    pending = list(roles)\n    seen = set()\n    allowed = set()\n    while pending:\n        role = pending.pop()\n        if role in seen:\n            continue\n        seen.add(role)\n        allowed.update(grants.get(role, []))\n        pending.extend(parents.get(role, []))\n    allowed.difference_update(denies.get(user, []))\n    result[user] = sorted(allowed)\nreturn {user: result[user] for user in sorted(result)}",
            [[{"u": ["admin"]}, [["admin", "reader"]], {"admin": ["write"], "reader": ["read"]}, {"u": ["write"]}], [{}, [], {}, {}], [{"u": ["a"]}, [["a", "b"], ["b", "a"]], {"b": ["p"]}, {}], [{"b": ["x"], "a": []}, [], {"x": ["z"]}, {}]],
            ["graph", "transitive_closure", "policy_resolution"],
        ),
        Task(
            "code_allocate_stock_fairly_001", "hard", "allocate_stock_fairly", "stock, requests",
            "Requests are [id,sku,quantity,priority]. Process priority descending, then id. Fully accept only requests with enough remaining stock. Return [accepted_ids,rejected_ids,remaining_stock].",
            "remaining = dict(stock)\naccepted = []\nrejected = []\nfor request_id, sku, quantity, priority in sorted(requests, key=lambda row: (-row[3], row[0])):\n    if remaining.get(sku, 0) >= quantity:\n        remaining[sku] -= quantity\n        accepted.append(request_id)\n    else:\n        rejected.append(request_id)\nreturn [accepted, rejected, {sku: remaining[sku] for sku in sorted(remaining)}]",
            [[{"x": 5}, [["b", "x", 3, 1], ["a", "x", 4, 2]]], [{}, []], [{"x": 2}, [["a", "x", 2, 1], ["b", "x", 1, 1]]], [{"b": 1, "a": 1}, [["x", "missing", 1, 9]]]],
            ["allocation", "priority_order", "state_tracking"],
        ),
        Task(
            "code_data_drift_001", "hard", "data_drift", "baseline, current, tolerance, minimum_count",
            "For keys present in both maps with both counts at least minimum_count, flag a key when abs(current-baseline)/baseline exceeds tolerance. Ignore a zero baseline. Return [key,rounded_ratio_3dp] sorted by ratio descending then key.",
            "rows = []\nfor key in baseline.keys() & current.keys():\n    old = baseline[key]\n    new = current[key]\n    if old != 0 and old >= minimum_count and new >= minimum_count:\n        ratio = abs(new - old) / old\n        if ratio > tolerance:\n            rows.append([key, round(ratio, 3)])\nreturn sorted(rows, key=lambda row: (-row[1], row[0]))",
            [[{"a": 100, "b": 50}, {"a": 130, "b": 40}, 0.2, 10], [{}, {}, 0.1, 1], [{"x": 0}, {"x": 9}, 0, 1], [{"a": 10}, {"a": 12}, 0.2, 1]],
            ["numeric_reasoning", "filtering", "ranking"],
        ),
    ]


def _realism_implementation_replacements() -> dict[str, Task]:
    replacement_ids = (
        "code_parse_feature_flags_001",
        "code_bucket_response_times_001",
        "code_coalesce_notes_001",
        "code_has_config_cycle_001",
        "code_tiered_charge_001",
        "code_retry_times_001",
        "code_partition_batches_001",
    )
    replacement_tasks = _additional_implementation_tasks()[:7]
    if len(replacement_tasks) != len(replacement_ids):
        raise ValueError("realism replacement task count drifted")
    return dict(zip(replacement_ids, replacement_tasks, strict=True))


def _implementation_items() -> list[dict[str, Any]]:
    tasks = [
        Task("code_normalize_event_codes_001", "easy", "normalize_event_codes", "codes", "Trim each code, uppercase it, discard blank codes, and return first occurrences in input order.", "result = []\nfor code in codes:\n    value = code.strip().upper()\n    if value and value not in result:\n        result.append(value)\nreturn result", [[" auth ", "AUTH", "", "pay"], [], ["a", " b ", "A"], [" x "]], ["sanity", "normalization"]),
        Task("code_parse_feature_flags_001", "easy", "parse_feature_flags", "lines", "Ignore blank and # comment lines. Parse NAME=on/off case-insensitively, trim both sides, and let later entries win. Return a dictionary of booleans.", "result = {}\nfor line in lines:\n    text = line.strip()\n    if not text or text.startswith('#'):\n        continue\n    key, value = text.split('=', 1)\n    result[key.strip()] = value.strip().lower() == 'on'\nreturn result", [["A=on", "B=off", "A=off"], [" # note", " X = ON "], [], ["cache=off"]], ["sanity", "parsing"]),
        Task("code_apply_profile_patch_001", "easy", "apply_profile_patch", "profile, patch", "Return a new profile with patch values applied; a None patch value deletes that key. Preserve the original inputs.", "result = dict(profile)\nfor key, value in patch.items():\n    if value is None:\n        result.pop(key, None)\n    else:\n        result[key] = value\nreturn result", [[{"a": 1, "b": 2}, {"b": 3, "a": None}], [{}, {"x": 1}], [{"x": 1}, {}], [{"x": 1}, {"missing": None}]], ["sanity", "immutable_update"]),
        Task("code_select_latest_records_001", "easy", "select_latest_records", "records", "For each id retain the record with the greatest integer version; equal versions keep the later input record. Return retained records ordered by id.", "latest = {}\nfor record in records:\n    key = record['id']\n    if key not in latest or record['version'] >= latest[key]['version']:\n        latest[key] = dict(record)\nreturn [latest[key] for key in sorted(latest)]", [[[{"id": "b", "version": 1}, {"id": "a", "version": 2}, {"id": "b", "version": 3}]], [[{"id": "a", "version": 1, "v": "x"}, {"id": "a", "version": 1, "v": "y"}]], [[]], [[{"id": "x", "version": 0}]]], ["sanity", "records"]),
        Task("code_bucket_response_times_001", "easy", "bucket_response_times", "values, limits", "Return counts for len(limits)+1 buckets. Bucket i contains values greater than the previous limit and less than or equal to limits[i]; the final bucket is above the last limit. Limits are sorted.", "counts = [0] * (len(limits) + 1)\nfor value in values:\n    index = 0\n    while index < len(limits) and value > limits[index]:\n        index += 1\n    counts[index] += 1\nreturn counts", [[[1, 5, 6, 10, 11], [5, 10]], [[], [2]], [[3, 3], [3]], [[1, 9], []]], ["sanity", "boundaries"]),
        Task("code_group_status_runs_001", "easy", "group_status_runs", "statuses", "Run-length encode consecutive equal statuses as [status, count] pairs.", "result = []\nfor status in statuses:\n    if result and result[-1][0] == status:\n        result[-1][1] += 1\n    else:\n        result.append([status, 1])\nreturn result", [["ok", "ok", "fail", "ok"], [], ["x"], ["a", "a", "a"]], ["sanity", "state_tracking"]),
        Task("code_inventory_totals_001", "easy", "inventory_totals", "records", "Ignore inactive records, sum integer quantity by sku, and return [[sku,total], ...] ordered by sku. Include zero and negative adjustments.", "totals = {}\nfor record in records:\n    if record['active']:\n        sku = record['sku']\n        totals[sku] = totals.get(sku, 0) + record['quantity']\nreturn [[sku, totals[sku]] for sku in sorted(totals)]", [[[{"sku": "b", "quantity": 2, "active": True}, {"sku": "a", "quantity": 3, "active": True}, {"sku": "b", "quantity": -1, "active": True}]], [[{"sku": "x", "quantity": 9, "active": False}]], [[]], [[{"sku": "x", "quantity": 0, "active": True}]]], ["sanity", "aggregation"]),
        Task("code_mask_record_fields_001", "easy", "mask_record_fields", "records, fields", "Return new records, replacing each present named field with '***'. Preserve record order and both inputs.", "hidden = set(fields)\nreturn [{key: ('***' if key in hidden else value) for key, value in record.items()} for record in records]", [[[{"name": "Ada", "email": "a@x"}], ["email"]], [[{"a": 1}, {"b": 2}], ["x"]], [[], ["x"]], [[{"token": "z"}], []]], ["sanity", "immutable_update"]),
        Task("code_active_usernames_001", "easy", "active_usernames", "records", "Return lowercase trimmed usernames for active records, removing duplicates after normalization and sorting the result.", "names = set()\nfor record in records:\n    if record['active']:\n        value = record['username'].strip().lower()\n        if value:\n            names.add(value)\nreturn sorted(names)", [[[{"username": " Ada ", "active": True}, {"username": "ada", "active": True}, {"username": "Lin", "active": False}]], [[]], [[{"username": " ", "active": True}]], [[{"username": "B", "active": True}, {"username": "a", "active": True}]]], ["sanity", "filtering"]),
        Task("code_parse_measurements_001", "easy", "parse_measurements", "lines", "Parse name:value lines, ignore malformed lines, trim names, convert values to integers, and let later valid entries win. Return keys sorted in a dictionary.", "values = {}\nfor line in lines:\n    if ':' not in line:\n        continue\n    name, raw = line.split(':', 1)\n    name = name.strip()\n    raw = raw.strip()\n    if name and raw.lstrip('-').isdigit():\n        values[name] = int(raw)\nreturn {key: values[key] for key in sorted(values)}", [["b:2", "bad", "a: 1", "b:3"], [], ["x:no", "x:-4"], [":3", "z:0"]], ["sanity", "parsing"]),
        Task("code_reconcile_windows_001", "medium", "reconcile_windows", "windows, blocked, minimum", "Merge overlapping or touching half-open windows, subtract every blocked half-open interval, discard pieces shorter than minimum, and return pieces ordered by start.", "merged = []\nfor start, end in sorted(windows):\n    if merged and start <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], end)\n    else:\n        merged.append([start, end])\npieces = merged\nfor cut_start, cut_end in blocked:\n    updated = []\n    for start, end in pieces:\n        if cut_end <= start or cut_start >= end:\n            updated.append([start, end])\n        else:\n            if start < cut_start:\n                updated.append([start, cut_start])\n            if cut_end < end:\n                updated.append([cut_end, end])\n    pieces = updated\nreturn [piece for piece in pieces if piece[1] - piece[0] >= minimum]", [[[[1, 5], [4, 9]], [[3, 4], [7, 8]], 2], [[[5, 8], [1, 3], [3, 5]], [], 3], [[], [[1, 2]], 1], [[[0, 10]], [[-1, 20]], 1]], ["intervals", "composed_operations"]),
        Task(
            "code_session_summaries_001", "medium", "session_summaries", "events, gap",
            "Events are [user,time] pairs sorted by time. A new session starts when the gap from that user's previous event is greater than gap. Return [user,start,end,count] sessions ordered by start then user.",
            "active = {}\nresult = []\nfor user, time in events:\n    if user not in active or time - active[user][1] > gap:\n        if user in active:\n            result.append([user] + active[user])\n        active[user] = [time, time, 1]\n    else:\n        active[user][1] = time\n        active[user][2] += 1\nfor user, data in active.items():\n    result.append([user] + data)\nreturn sorted(result, key=lambda row: (row[1], row[0]))",
            [[[["a", 1], ["b", 2], ["a", 4], ["a", 10]], 3],
             [[], 2],
             [[["x", 1], ["x", 3]], 2],
             [[["b", 1], ["a", 1]], 0]],
            ["event_processing", "state_tracking"],
        ),
        Task(
            "code_dependency_batches_001", "medium", "dependency_batches", "jobs, dependencies",
            "Return deterministic execution batches for jobs. Each [job, prerequisite] edge requires the prerequisite first; each batch contains all currently available jobs sorted. Return [] if a cycle exists.",
            "remaining = {job: set() for job in jobs}\nfor job, prerequisite in dependencies:\n    remaining[job].add(prerequisite)\nresult = []\ncompleted = set()\nwhile len(completed) < len(jobs):\n    batch = sorted(job for job in jobs if job not in completed and remaining[job] <= completed)\n    if not batch:\n        return []\n    result.append(batch)\n    completed.update(batch)\nreturn result",
            [[["build", "test", "ship"], [["test", "build"], ["ship", "test"]]],
             [["a", "b", "c"], [["c", "a"]]],
             [["a", "b"], [["a", "b"], ["b", "a"]]],
             [[], []]],
            ["graph", "deterministic_order"],
        ),
        Task("code_ledger_balances_001", "medium", "ledger_balances", "opening, entries, voided", "Starting from opening balances, apply non-voided [entry_id,account,amount] entries in order. Return [[account,balance], ...] for every resulting nonzero account, ordered by account.", "balances = dict(opening)\nvoided_ids = set(voided)\nfor entry_id, account, amount in entries:\n    if entry_id not in voided_ids:\n        balances[account] = balances.get(account, 0) + amount\nreturn [[account, balances[account]] for account in sorted(balances) if balances[account] != 0]", [[{"a": 10}, [["e1", "a", -3], ["e2", "b", 4]], ["e2"]], [{}, [], []], [{"x": 1}, [["e", "x", -1]], []], [{}, [["e", "b", 2], ["f", "a", 3]], []]], ["records", "reconciliation"]),
        Task("code_effective_permissions_001", "medium", "effective_permissions", "roles, grants, denies", "roles maps each user to role names. grants maps roles to permissions. Return each user's sorted granted permissions after removing that user's denied permissions.", "result = {}\nfor user, user_roles in roles.items():\n    allowed = set()\n    for role in user_roles:\n        allowed.update(grants.get(role, []))\n    allowed.difference_update(denies.get(user, []))\n    result[user] = sorted(allowed)\nreturn {user: result[user] for user in sorted(result)}", [[{"u": ["reader", "writer"]}, {"reader": ["read"], "writer": ["read", "write"]}, {"u": ["write"]}], [{}, {}, {}], [{"b": ["x"], "a": []}, {"x": ["p"]}, {}], [{"u": ["missing"]}, {}, {"u": ["p"]}]], ["sets", "policy_resolution"]),
        Task("code_rooms_required_001", "medium", "rooms_required", "meetings", "Meetings are half-open [start,end] intervals. Return the minimum rooms required; a room is reusable when one meeting ends exactly as another starts.", "events = []\nfor start, end in meetings:\n    events.append([start, 1])\n    events.append([end, -1])\nevents.sort(key=lambda event: (event[0], event[1]))\nactive = 0\nbest = 0\nfor _, change in events:\n    active += change\n    best = max(best, active)\nreturn best", [[[[0, 10], [5, 7], [10, 12]]], [[]], [[[1, 2]]], [[[1, 4], [2, 5], [3, 6]]]], ["scheduling", "sweep_line"]),
        Task("code_alert_streaks_001", "medium", "alert_streaks", "readings, threshold, minimum", "Return inclusive index ranges of maximal consecutive readings at least threshold whose length is at least minimum.", "result = []\nstart = None\nfor index in range(len(readings) + 1):\n    qualifies = index < len(readings) and readings[index] >= threshold\n    if qualifies and start is None:\n        start = index\n    if not qualifies and start is not None:\n        if index - start >= minimum:\n            result.append([start, index - 1])\n        start = None\nreturn result", [[[1, 5, 6, 2, 7, 8, 9], 5, 2], [[], 1, 1], [[3, 3], 3, 2], [[5, 1, 5], 5, 2]], ["sequence_processing", "boundaries"]),
        Task("code_has_config_cycle_001", "medium", "has_config_cycle", "links", "links maps a configuration name to its parent name or None. Return whether any chain enters a cycle, including a self-cycle.", "done = set()\nfor start in links:\n    seen = set()\n    node = start\n    while node is not None and node in links and node not in done:\n        if node in seen:\n            return True\n        seen.add(node)\n        node = links[node]\n    done.update(seen)\nreturn False", [[{"a": "b", "b": "a"}], [{"a": "b", "b": None}], [{}], [{"x": "x"}]], ["graph", "cycle_detection"]),
        Task("code_filter_compatible_versions_001", "medium", "filter_compatible_versions", "versions, minimum, excluded", "Versions are [major,minor,patch]. Return sorted distinct versions at least minimum with no component pattern present in excluded.", "blocked = {tuple(value) for value in excluded}\nvalues = {tuple(value) for value in versions if tuple(value) >= tuple(minimum) and tuple(value) not in blocked}\nreturn [list(value) for value in sorted(values)]", [[[[1, 2, 0], [1, 1, 9], [1, 2, 0], [2, 0, 0]], [1, 2, 0], [[2, 0, 0]]], [[], [0, 0, 0], []], [[[1, 0, 0]], [2, 0, 0], []], [[[1, 0, 1]], [1, 0, 0], []]], ["versioning", "filtering"]),
        Task("code_fulfill_orders_001", "medium", "fulfill_orders", "stock, orders", "Process [order_id,sku,quantity] in order. Fulfill an order only when enough stock remains. Return [fulfilled_ids, remaining_stock] with stock keys sorted.", "remaining = dict(stock)\nfulfilled = []\nfor order_id, sku, quantity in orders:\n    if remaining.get(sku, 0) >= quantity:\n        remaining[sku] -= quantity\n        fulfilled.append(order_id)\nreturn [fulfilled, {sku: remaining[sku] for sku in sorted(remaining)}]", [[{"a": 5}, [["o1", "a", 3], ["o2", "a", 3]]], [{}, [["o", "x", 1]]], [{"b": 1, "a": 2}, [],], [{"a": 2}, [["o", "a", 2]]]], ["state_tracking", "allocation"]),
        Task("code_snapshot_changes_001", "medium", "snapshot_changes", "before, after", "Return [added, removed, changed] key lists, each sorted. A changed key exists in both dictionaries with unequal values.", "before_keys = set(before)\nafter_keys = set(after)\nadded = sorted(after_keys - before_keys)\nremoved = sorted(before_keys - after_keys)\nchanged = sorted(key for key in before_keys & after_keys if before[key] != after[key])\nreturn [added, removed, changed]", [[{"a": 1, "b": 2}, {"b": 3, "c": 4}], [{}, {}], [{"x": 1}, {"x": 1}], [{}, {"z": 0}]], ["records", "set_operations"]),
        Task("code_tiered_charge_001", "medium", "tiered_charge", "usage, tiers", "tiers contains [inclusive_limit, unit_price] rows with increasing limits; the final limit is None. Charge successive usage units by tier and return the integer total.", "remaining = usage\nprevious = 0\ntotal = 0\nfor limit, price in tiers:\n    amount = remaining if limit is None else min(remaining, limit - previous)\n    total += amount * price\n    remaining -= amount\n    if remaining == 0:\n        break\n    previous = limit\nreturn total", [[15, [[10, 2], [20, 3], [None, 5]]], [0, [[None, 2]]], [10, [[10, 2], [None, 4]]], [25, [[10, 1], [20, 2], [None, 3]]]], ["arithmetic", "piecewise_rules"]),
        Task("code_compress_log_bursts_001", "medium", "compress_log_bursts", "events, gap", "Events are sorted [timestamp,message]. Merge consecutive equal messages when their timestamp gap is at most gap. Return [start,end,message,count] rows.", "result = []\nfor timestamp, message in events:\n    if result and result[-1][2] == message and timestamp - result[-1][1] <= gap:\n        result[-1][1] = timestamp\n        result[-1][3] += 1\n    else:\n        result.append([timestamp, timestamp, message, 1])\nreturn result", [[[[1, "x"], [3, "x"], [4, "y"], [10, "y"]], 2], [[], 1], [[[1, "x"]], 0], [[[1, "x"], [2, "y"], [3, "x"]], 5]], ["event_processing", "state_tracking"]),
        Task("code_retry_times_001", "medium", "retry_times", "start, delays, maintenance", "Apply cumulative delays to start. Exclude retry times inside any half-open maintenance interval and return the remaining times.", "result = []\ntime = start\nfor delay in delays:\n    time += delay\n    blocked = any(left <= time < right for left, right in maintenance)\n    if not blocked:\n        result.append(time)\nreturn result", [[0, [1, 2, 4], [[2, 4]]], [5, [], []], [0, [2], [[2, 3]]], [10, [1, 1], []]], ["scheduling", "cumulative_state"]),
        Task("code_join_customer_orders_001", "medium", "join_customer_orders", "customers, orders", "customers maps ids to names. Sum positive order amounts for known customers and return [customer_id,name,total] rows ordered by descending total then id.", "totals = {}\nfor customer, amount in orders:\n    if customer in customers and amount > 0:\n        totals[customer] = totals.get(customer, 0) + amount\nrows = [[customer, customers[customer], totals[customer]] for customer in totals]\nreturn sorted(rows, key=lambda row: (-row[2], row[0]))", [[{"a": "Ada", "b": "Bob"}, [["a", 3], ["b", 5], ["a", 4]]], [{}, [["x", 2]]], [{"a": "A"}, [["a", -1]]], [{"b": "B", "a": "A"}, [["b", 2], ["a", 2]]]], ["records", "join_aggregate"]),
        Task("code_eligible_shortest_route_001", "hard", "eligible_shortest_route", "nodes, edges, start, end, disabled", "For an undirected graph, return the lexicographically smallest node sequence among shortest start-to-end routes that avoid disabled nodes. Return [] if unavailable.", "disabled_set = set(disabled)\nif start in disabled_set or end in disabled_set:\n    return []\ngraph = {node: [] for node in nodes}\nfor left, right in edges:\n    graph[left].append(right)\n    graph[right].append(left)\npaths = [[start]]\nseen_depth = {start: 0}\nwhile paths:\n    path = paths.pop(0)\n    node = path[-1]\n    if node == end:\n        return path\n    depth = len(path)\n    for neighbor in sorted(graph[node]):\n        if neighbor not in disabled_set and seen_depth.get(neighbor, depth) >= depth:\n            seen_depth[neighbor] = depth\n            paths.append(path + [neighbor])\nreturn []", [[["a", "b", "c", "d"], [["a", "b"], ["b", "d"], ["a", "c"], ["c", "d"]], "a", "d", []], [["a", "b"], [["a", "b"]], "a", "b", ["b"]], [["a"], [], "a", "a", []], [["a", "b", "c"], [["a", "b"]], "a", "c", []]], ["graph", "tie_breaking", "composed_operations"]),
        Task("code_deployment_impact_001", "hard", "deployment_impact", "services, dependencies, changed, protected", "dependencies are [service,dependency]. Return affected services reachable from changed dependencies, excluding protected services and anything depending on an excluded service, ordered in deterministic dependency batches.", "blocked = set(protected)\nprogress = True\nwhile progress:\n    progress = False\n    for service, dependency in dependencies:\n        if dependency in blocked and service not in blocked:\n            blocked.add(service)\n            progress = True\naffected = set(changed) - blocked\nprogress = True\nwhile progress:\n    progress = False\n    for service, dependency in dependencies:\n        if service not in blocked and dependency in affected and service not in affected:\n            affected.add(service)\n            progress = True\nremaining = set(affected)\nresult = []\nwhile remaining:\n    batch = sorted(service for service in remaining if all(dependency not in remaining for owner, dependency in dependencies if owner == service))\n    if not batch:\n        return []\n    result.append(batch)\n    remaining.difference_update(batch)\nreturn result", [[["db", "api", "web"], [["api", "db"], ["web", "api"]], ["db"], []], [["a", "b"], [["b", "a"]], ["a"], ["b"]], [["a"], [], [], []], [["a", "b", "c"], [["c", "a"], ["b", "a"]], ["a"], []], [["a", "b", "c"], [["c", "a"], ["c", "b"]], ["a"], ["b"]]], ["graph", "dependency_propagation"]),
        Task("code_reservation_capacity_001", "hard", "reservation_capacity", "capacity, reservations, cancellations", "Reservations are [id,start,end,units] half-open intervals. Ignore cancelled ids. Return the earliest timestamp where active units exceed capacity, or None. Process endings before starts at equal timestamps.", "cancelled = set(cancellations)\nevents = []\nfor reservation, start, end, units in reservations:\n    if reservation not in cancelled:\n        events.append([start, units])\n        events.append([end, -units])\nevents.sort(key=lambda event: (event[0], event[1]))\nactive = 0\nfor time, change in events:\n    active += change\n    if active > capacity:\n        return time\nreturn None", [[3, [["a", 1, 5, 2], ["b", 3, 6, 2]], []], [2, [["a", 1, 3, 2], ["b", 3, 5, 2]], []], [1, [["a", 1, 4, 2]], ["a"]], [5, [], []]], ["sweep_line", "event_ordering"]),
        Task("code_workflow_duration_001", "hard", "workflow_duration", "steps, durations, dependencies", "Return the minimum total completion time with unlimited parallelism. Each [step,prerequisite] dependency must finish first. Return -1 for a cycle.", "remaining = set(steps)\nfinish = {}\nwhile remaining:\n    ready = sorted(step for step in remaining if all(prerequisite in finish for owner, prerequisite in dependencies if owner == step))\n    if not ready:\n        return -1\n    for step in ready:\n        start = max([finish[prerequisite] for owner, prerequisite in dependencies if owner == step] or [0])\n        finish[step] = start + durations[step]\n        remaining.remove(step)\nreturn max(finish.values()) if finish else 0", [[["a", "b", "c"], {"a": 2, "b": 3, "c": 4}, [["c", "a"], ["c", "b"]]], [["a", "b"], {"a": 1, "b": 1}, [["a", "b"], ["b", "a"]]], [[], {}, []], [["a", "b"], {"a": 2, "b": 3}, []]], ["dag", "critical_path"]),
        Task(
            "code_reconcile_event_streams_001", "hard", "reconcile_event_streams", "primary, replica",
            "Events are [id,sequence,payload]. For each id choose the greatest sequence across both streams; equal sequences prefer primary. Return [id,sequence,payload,source] rows ordered by sequence then id.",
            "chosen = {}\nfor source, events in [['replica', replica], ['primary', primary]]:\n    for event_id, sequence, payload in events:\n        if event_id not in chosen or sequence > chosen[event_id][0] or (sequence == chosen[event_id][0] and source == 'primary'):\n            chosen[event_id] = [sequence, payload, source]\nrows = [[event_id] + chosen[event_id] for event_id in chosen]\nreturn sorted(rows, key=lambda row: (row[1], row[0]))",
            [[[["a", 2, "p"], ["b", 1, "x"]], [["a", 2, "r"], ["c", 3, "z"]]],
             [[], []],
             [[["a", 1, "p"]], [["a", 2, "r"]]],
             [[["b", 1, "x"], ["a", 1, "p"]], []]],
            ["records", "conflict_resolution", "tie_breaking"],
        ),
    ]
    replacements = _realism_implementation_replacements()
    tasks.extend(_additional_implementation_tasks()[7:])
    tasks = [replacements.get(task.id, task) for task in tasks]
    tasks = [task for task in tasks if task.id not in RETIRED_ITEM_IDS]
    unwrapped_single_argument_cases = {
        "code_normalize_event_codes_001",
        "code_parse_feature_flags_001",
        "code_group_status_runs_001",
        "code_parse_measurements_001",
    }
    extra_cases = {
        "code_eligible_shortest_route_001": [
            [["a", "b", "c", "d", "e"], [["a", "c"], ["c", "d"], ["a", "b"], ["b", "d"], ["d", "e"]], "a", "e", []],
        ],
        "code_deployment_impact_001": [
            [["a", "b", "c", "d"], [["b", "a"], ["c", "a"], ["d", "b"], ["d", "c"]], ["a"], []],
        ],
        "code_reservation_capacity_001": [
            [3, [["ending", 1, 4, 3], ["starting", 4, 7, 3]], []],
        ],
        "code_workflow_duration_001": [
            [["a", "b", "c", "d"], {"a": 2, "b": 4, "c": 3, "d": 1}, [["c", "a"], ["c", "b"], ["d", "c"]]],
        ],
        "code_reconcile_event_streams_001": [
            [[["x", 4, "primary-old"], ["a", 2, "p"]], [["x", 5, "replica-new"], ["a", 2, "r"]]],
        ],
        "code_release_waves_001": [
            [["a", "b", "c", "d"], [["b", "a"], ["c", "a"], ["d", "b"]], ["c"]],
        ],
        "code_permission_closure_001": [
            [{"u": ["editor"]}, [["editor", "reader"], ["reader", "base"]], {"base": ["view"], "editor": ["edit"]}, {"u": []}],
        ],
        "code_allocate_stock_fairly_001": [
            [{"x": 5}, [["b", "x", 2, 3], ["a", "x", 2, 3], ["c", "x", 2, 2]]],
        ],
        "code_data_drift_001": [
            [{"a": 100, "b": 100}, {"a": 140, "b": 150}, 0.1, 1],
        ],
    }
    items = []
    for index, task in enumerate(tasks):
        source = _source(task)
        tests = []
        for args in [*task.cases, *extra_cases.get(task.id, [])]:
            if task.id in unwrapped_single_argument_cases:
                args = [args]
            preserved = [i for i, value in enumerate(args) if isinstance(value, (list, dict))]
            test = {"args": args, "expected": _expected(source, task.name, args)}
            if preserved:
                test["preserve_args"] = preserved
            tests.append(test)
        items.append({
            "id": task.id,
            "subcategory": "function_implementation",
            "difficulty": task.difficulty,
            "split": "dev" if index % 4 == 0 else "test",
            "visibility": "public" if index % 2 == 0 else "held_out",
            "prompt": _implementation_prompt(task, index),
            "response_contract": {"type": "code", "format": "python_function"},
            "expected": {"value": {"entry_point": task.name, "tests": tests, "reference_solution": source}},
            "scoring": {"method": "executable_python", "parameters": {"timeout_seconds": 1.0, "memory_limit_mb": 128, "max_output_characters": 10000}},
            "provenance": {"kind": "synthetic", "review_status": "human_checked", "generator": GENERATOR, "seed": SEED},
            "tags": ["practical_python", "fresh_composed", "pass_at_1", *task.tags],
        })
    return items


def _diagnosis_items() -> list[dict[str, Any]]:
    labels = (
        "boundary_update, state_scope, row_aliasing, wrong_precedence, "
        "stale_cache_key, direction_error, lossy_conversion, "
        "mutation_iteration, missing_finalization, tie_break_error, "
        "wrong_default, early_return, unit_mismatch, shared_default, "
        "incomplete_key, off_by_one, shadowed_name, exception_scope, "
        "unstable_sort, missing_deduplication"
    )
    cases = [
        ("diagnose_page_cursor_001", "easy", "off_by_one", "def page_after(items, cursor):\n    index = items.index(cursor)\n    return items[index:]", "Failing check: page_after(['a','b','c'], 'b') expected ['c']; actual ['b','c'].", ["boundaries"]),
        ("diagnose_daily_totals_001", "medium", "state_scope", "def daily_totals(days):\n    total = 0\n    result = []\n    for values in days:\n        for value in values: total += value\n        result.append(total)\n    return result", "Regression case: daily_totals([[2],[3]]) expected [2,3]; actual [2,5].", ["state_scope"]),
        ("diagnose_matrix_template_001", "easy", "row_aliasing", "def matrix(rows, cols):\n    values = [[None] * cols] * rows\n    values[0][0] = 'x'\n    return values", "matrix(2,2) should change only the first row, but both rows start with 'x'.", ["aliasing"]),
        ("diagnose_access_rule_001", "medium", "wrong_precedence", "def allowed(active, admin, suspended):\n    return active or admin and not suspended", "Policy: a user must not be suspended and must be active or an admin. Input (True, False, True) expected False; actual True.", ["boolean_logic"]),
        ("diagnose_price_cache_001", "medium", "stale_cache_key", "def priced(items, tax):\n    cache = {}\n    def one(item):\n        if item['sku'] in cache: return cache[item['sku']]\n        value = item['price'] + tax\n        cache[item['sku']] = value\n        return value\n    return [one(item) for item in items]", "Input [{'sku':'x','price':10},{'sku':'x','price':14}] with tax 2 expected [12,16]; actual [12,12].", ["memoization"]),
        ("diagnose_dependency_edges_001", "hard", "direction_error", "def impacted(edges, changed):\n    result = set(changed)\n    progress = True\n    while progress:\n        progress = False\n        for service, dependency in edges:\n            if service in result and dependency not in result:\n                result.add(dependency)\n                progress = True\n    return sorted(result)", "Edges are [service, dependency]. For [['api','db'],['web','api']] and changed ['db'], expected ['api','db','web']; actual ['db'].", ["graph", "transitive_state"]),
        ("diagnose_average_latency_001", "easy", "lossy_conversion", "def average_latency(values):\n    return int(sum(values) / len(values))", "Failing check: average_latency([1,2]) expected 1.5; actual 1.", ["numeric_semantics"]),
        ("diagnose_remove_expired_001", "medium", "mutation_iteration", "def remove_expired(records):\n    for record in records:\n        if record['expired']:\n            records.remove(record)\n    return records", "Two adjacent expired records are supplied. The second one remains in the returned list.", ["collection_mutation"]),
        ("diagnose_flush_groups_001", "medium", "missing_finalization", "def groups(values):\n    result = []\n    current = []\n    for value in values:\n        if current and value != current[-1]:\n            result.append(current); current = []\n        current.append(value)\n    return result", "For ['a','a','b'], expected [['a','a'],['b']]; actual [['a','a']].", ["state_machine"]),
        ("diagnose_route_choice_001", "hard", "tie_break_error", "def choose_candidate(candidates):\n    return min(candidates, key=lambda item: (-item['score'], item['name'], item['latency']))", "Selection priority is highest score, then lowest latency, then name. Candidates [{name:'alpha',score:9,latency:8},{name:'beta',score:9,latency:3}] should select beta; actual alpha.", ["multi_key_ordering", "tie_breaking"]),
        ("diagnose_missing_setting_001", "easy", "wrong_default", "def retries(config):\n    return config.get('retries', 0)", "Product policy says a missing retries setting means 3 attempts. retries({}) expected 3; actual 0.", ["configuration", "defaults"]),
        ("diagnose_first_valid_row_001", "medium", "early_return", "def valid_rows(rows):\n    result = []\n    for row in rows:\n        if row.get('active'):\n            result.append(row)\n        return result", "With three active rows, only the first row is returned.", ["control_flow", "iteration"]),
        ("diagnose_cache_expiry_001", "medium", "unit_mismatch", "def expired(created_ms, ttl_seconds, now_ms):\n    return now_ms >= created_ms + ttl_seconds", "expired(1000, 5, 2000) expected False because five seconds have not passed; actual True.", ["units", "time"]),
        ("diagnose_group_members_001", "medium", "shared_default", "def group_members(records):\n    groups = dict.fromkeys(['admin', 'reader'], [])\n    for name, role in records:\n        groups[role].append(name)\n    return groups", "Adding Ada only as admin also makes Ada appear in the reader list.", ["aliasing", "mutable_state"]),
        ("diagnose_tenant_cache_001", "hard", "incomplete_key", "def lookup(cache, tenant, user):\n    key = user\n    return cache.get(key)", "Tenant A and tenant B both have user 42, but a cached result for A is returned while looking up B.", ["cache", "multi_tenant"]),
        ("diagnose_retry_budget_001", "easy", "off_by_one", "def attempts(max_retries):\n    return list(range(max_retries))", "The contract allows the initial attempt plus max_retries. attempts(2) expected three attempt slots; actual two.", ["boundaries", "retry"]),
        ("diagnose_discount_total_001", "medium", "shadowed_name", "def discounted(prices, rate):\n    total = 0\n    for rate in prices:\n        total += rate\n    return total * (1 - rate)", "discounted([10, 20], 0.1) expected 27; actual uses the final price as the rate.", ["variable_scope", "numeric_semantics"]),
        ("diagnose_optional_field_001", "medium", "exception_scope", "def display_name(record):\n    try:\n        return record['profile']['name'].strip()\n    except KeyError:\n        return 'unknown'", "display_name({'profile': None}) should return 'unknown' but raises TypeError.", ["exceptions", "nested_data"]),
        ("diagnose_ranked_feed_001", "hard", "unstable_sort", "def ranked(items):\n    return sorted(items, key=lambda item: -item['score'])", "The contract requires equal-score items ordered by id, but input order changes the output tie order.", ["ranking", "determinism"]),
        ("diagnose_alert_recipients_001", "medium", "missing_deduplication", "def recipients(groups):\n    result = []\n    for members in groups:\n        result.extend(members)\n    return sorted(result)", "A person in two escalation groups receives the same alert twice, although each recipient should appear once.", ["deduplication", "aggregation"]),
    ]
    introductions = (
        "Review the function and its failing check:",
        "A regression report contains this function and symptom:",
        "Trace the code against the reported behavior:",
    )
    questions = (
        "Which fault category best explains the behavior?",
        "Select the primary defect class.",
        "Identify the root-cause category.",
    )
    cases = [case for case in cases if case[0] not in RETIRED_ITEM_IDS]
    items = []
    for offset, (item_id, difficulty, label, source, observation, tags) in enumerate(cases, start=30):
        style = offset % len(introductions)
        prompt = (
            f"{introductions[style]}\n\n{source}\n\n{observation}\n\n"
            f"{questions[style]} Options: {labels}. Return only the category."
        )
        items.append({
            "id": item_id,
            "subcategory": "bug_diagnosis",
            "difficulty": difficulty,
            "split": "dev" if offset % 4 == 0 else "test",
            "visibility": "public" if offset % 2 == 0 else "held_out",
            "prompt": prompt,
            "response_contract": {"type": "text", "format": "diagnostic_label"},
            "expected": {"value": label},
            "scoring": {"method": "exact_match", "parameters": {"strip": True, "case_sensitive": False}},
            "provenance": {"kind": "synthetic", "review_status": "human_checked", "generator": GENERATOR, "seed": SEED},
            "tags": ["python", "diagnosis", "failure_trace", *tags],
        })
    return items


def _refund_repair() -> Repair:
    """Correctly apply positivity after selecting each latest duplicate."""
    return Repair(
        "repair_refund_total_001",
        "easy",
        "refund_total",
        "refunds, reversed_ids",
        "Sum positive refund amounts whose ids are not reversed. Duplicate non-reversed ids count once using their latest amount.",
        "reversed_set = set(reversed_ids)\n"
        "latest = {}\n"
        "for refund_id, amount in refunds:\n"
        "    if refund_id not in reversed_set:\n"
        "        latest[refund_id] = amount\n"
        "return sum(amount for amount in latest.values() if amount > 0)",
        [
            [[["r1", 3], ["r2", 5], ["r1", 4]], ["r2"]],
            [[], []],
            [[["r", -2]], []],
            [[["r", 2]], ["r"]],
            [[["r", 5], ["r", -2]], []],
        ],
        [
            (
                "counts_duplicates",
                "latest[refund_id] = amount",
                "latest[refund_id] = latest.get(refund_id, 0) + amount",
            ),
            (
                "includes_reversed",
                "refund_id not in reversed_set",
                "refund_id in reversed_set",
            ),
            ("includes_nonpositive", " if amount > 0", ""),
        ],
        ["aggregation", "deduplication"],
    )


def _repair_items() -> list[dict[str, Any]]:
    repairs = [
        Repair("repair_quota_adjustments_001", "easy", "quota_after_adjustments", "quota, adjustments", "Apply integer adjustments in order and clamp the result to zero after every adjustment.", "value = quota\nfor adjustment in adjustments:\n    value = max(0, value + adjustment)\nreturn value", [[5, [-3, -4, 2]], [0, [3]], [5, [5, -10]], [4, [-4, -1]]], [("clamp_only_at_end", "value = max(0, value + adjustment)", "value = value + adjustment"), ("ignore_order", "for adjustment in adjustments:", "for adjustment in sorted(adjustments):"), ("wrong_floor", "max(0, value + adjustment)", "max(1, value + adjustment)")], ["state_tracking"]),
        Repair("repair_latest_webhooks_001", "easy", "latest_webhooks", "events", "Keep the greatest sequence per webhook id; equal sequences keep the later event. Return [id,payload] rows ordered by id.", "latest = {}\nfor event_id, sequence, payload in events:\n    if event_id not in latest or sequence >= latest[event_id][0]:\n        latest[event_id] = [sequence, payload]\nreturn [[event_id, latest[event_id][1]] for event_id in sorted(latest)]", [[["b", 1, "x"], ["a", 2, "old"], ["a", 3, "new"]], [["a", 1, "x"], ["a", 1, "y"]], [], [["z", 0, "p"]]], [("keeps_smallest", "sequence >= latest[event_id][0]", "sequence <= latest[event_id][0]"), ("drops_equal_update", "sequence >= latest[event_id][0]", "sequence > latest[event_id][0]"), ("input_order_output", "for event_id in sorted(latest)", "for event_id in latest")], ["records", "versioning"]),
        _refund_repair(),
        Repair("repair_availability_windows_001", "medium", "availability_windows", "windows, minimum", "Merge overlapping or touching half-open windows and return merged windows whose length is at least minimum.", "merged = []\nfor start, end in sorted(windows):\n    if merged and start <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], end)\n    else:\n        merged.append([start, end])\nreturn [window for window in merged if window[1] - window[0] >= minimum]", [[[[1, 3], [3, 5], [8, 9]], 2], [[], 1], [[[5, 7], [1, 2]], 1], [[[1, 2]], 2]], [("does_not_merge_touching", "start <= merged[-1][1]", "start < merged[-1][1]"), ("keeps_short_windows", ">= minimum", "> 0"), ("loses_final_window", "return [window for window in merged", "return [window for window in merged[:-1]")], ["intervals", "boundaries"]),
        Repair("repair_compact_ranges_001", "medium", "compact_ranges", "values", "Sort distinct integers and compress each consecutive run into [start,end]. Return ranges in ascending order.", "numbers = sorted(set(values))\nif not numbers:\n    return []\nresult = []\nstart = numbers[0]\nprevious = numbers[0]\nfor value in numbers[1:]:\n    if value == previous + 1:\n        previous = value\n    else:\n        result.append([start, previous])\n        start = value\n        previous = value\nresult.append([start, previous])\nreturn result", [[[1, 2, 3, 7, 8, 10]], [[]], [[3, 1, 2, 2]], [[-2, -1, 1]], [[5, 7, 6, 10]]], [("merges_single_gap", "value == previous + 1", "value <= previous + 2"), ("drops_final_range", "result.append([start, previous])\n    return result", "return result"), ("keeps_duplicates", "sorted(set(values))", "sorted(values)")], ["sequence_processing", "range_compaction"]),
        Repair("repair_rolling_totals_001", "medium", "rolling_totals", "values, width", "Return the sum of every full consecutive window of the positive size width. Return [] when width exceeds the input length.", "if width > len(values):\n    return []\ncurrent = sum(values[:width])\nresult = [current]\nfor index in range(width, len(values)):\n    current += values[index] - values[index - width]\n    result.append(current)\nreturn result", [[[1, 2, 3, 4], 2], [[5], 1], [[], 1], [[3, -1, 2], 3], [[2, 0, 2, 0], 3]], [("wrong_outgoing_index", "values[index - width]", "values[index - width + 1]"), ("skips_second_window", "range(width, len(values))", "range(width + 1, len(values))"), ("drops_last_window", "return result", "return result[:-1]")], ["sliding_window", "arithmetic"]),
        Repair("repair_lookup_path_001", "medium", "lookup_path", "data, path, default", "Follow string tokens through dictionaries and non-negative integer tokens through lists. Return default when a token is missing, has the wrong type, or is out of range. A stored None is a valid result.", "current = data\nfor token in path:\n    try:\n        if token < 0:\n            return default\n    except TypeError:\n        pass\n    try:\n        current = current[token]\n    except (KeyError, IndexError, TypeError):\n        return default\nreturn current", [[{"user": {"names": ["Ada", "Lin"]}}, ["user", "names", 1], "missing"], [{"x": None}, ["x"], "missing"], [[10, 20], [-1], "missing"], [{"x": [0]}, ["x", 0], 99], [{"x": 1}, ["x", "y"], "missing"]], [("allows_negative_index", "if token < 0", "if False"), ("treats_none_as_missing", "return current", "return default if current is None else current"), ("rejects_falsy_value", "current = current[token]", "current = current[token]\n        if not current:\n            return default")], ["nested_data", "type_boundaries"]),
        Repair("repair_lru_cache_001", "hard", "lru_cache", "capacity, accesses", "Simulate a positive-capacity cache of keys. A hit moves its key to most-recent position. A miss at capacity evicts the least-recent key. Return [evicted_keys, final_keys] with final keys ordered least to most recent.", "order = []\nevicted = []\nfor key in accesses:\n    if key in order:\n        order.remove(key)\n        order.append(key)\n    else:\n        if len(order) == capacity:\n            evicted.append(order.pop(0))\n        order.append(key)\nreturn [evicted, order]", [[2, ["a", "b", "a", "c"]], [1, ["x", "y", "y", "z"]], [3, []], [3, ["a", "b", "c", "a"]], [2, ["a", "b", "c", "b", "d"]]], [("duplicates_cache_hits", "order.remove(key)", "order = order"), ("evicts_most_recent", "order.pop(0)", "order.pop()"), ("evicts_too_late", "len(order) == capacity", "len(order) > capacity")], ["cache", "state_machine", "event_ordering"]),
        Repair("repair_clean_tags_001", "easy", "clean_tags", "tags", "Trim and lowercase tags, discard blanks, and return first occurrences in input order.", "result = []\nfor tag in tags:\n    value = tag.strip().lower()\n    if value and value not in result:\n        result.append(value)\nreturn result", [[" A ", "a", "", "B"], [], ["x"], [" One", "TWO "]], [("keeps_whitespace", "tag.strip().lower()", "tag.lower()"), ("keeps_case", "tag.strip().lower()", "tag.strip()"), ("keeps_duplicates", "if value and value not in result", "if value")], ["normalization", "deduplication"]),
        Repair("repair_clamp_percentage_001", "easy", "clamp_percentage", "value", "Return value clamped to the inclusive range 0 through 100.", "return min(100, max(0, value))", [[-5], [0], [42], [100], [140]], [("missing_lower_bound", "min(100, max(0, value))", "min(100, value)"), ("missing_upper_bound", "min(100, max(0, value))", "max(0, value)"), ("exclusive_upper_bound", "min(100, max(0, value))", "min(99, max(0, value))")], ["boundaries", "numeric_semantics"]),
        Repair("repair_latest_status_001", "medium", "latest_status", "events", "Keep the event with the greatest timestamp for each id; equal timestamps keep the later event. Return [id,status] rows sorted by id.", "latest = {}\nfor item_id, timestamp, status in events:\n    if item_id not in latest or timestamp >= latest[item_id][0]:\n        latest[item_id] = [timestamp, status]\nreturn [[item_id, latest[item_id][1]] for item_id in sorted(latest)]", [[["b", 1, "old"], ["a", 2, "ok"], ["b", 3, "new"]], [["x", 1, "a"], ["x", 1, "b"]], [], [["z", 0, "ok"]]], [("keeps_oldest", "timestamp >= latest[item_id][0]", "timestamp <= latest[item_id][0]"), ("drops_equal_update", "timestamp >= latest[item_id][0]", "timestamp > latest[item_id][0]"), ("unsorted_output", "for item_id in sorted(latest)", "for item_id in latest")], ["records", "latest_state"]),
        Repair("repair_filter_audit_events_001", "medium", "filter_audit_events", "events, actor, actions, start, end", "Keep events for actor whose action is allowed and timestamp is in the inclusive start-to-end range. Return ids ordered by timestamp then id.", "allowed = set(actions)\nrows = [event for event in events if event['actor'] == actor and event['action'] in allowed and start <= event['time'] <= end]\nreturn [event['id'] for event in sorted(rows, key=lambda event: (event['time'], event['id']))]", [[[{"id": "b", "actor": "u", "action": "read", "time": 3}, {"id": "a", "actor": "u", "action": "write", "time": 1}], "u", ["read", "write"], 1, 3], [[], "u", [], 0, 1], [[{"id": "x", "actor": "v", "action": "read", "time": 2}], "u", ["read"], 0, 3], [[{"id": "x", "actor": "u", "action": "read", "time": 3}], "u", ["read"], 3, 3], [[{"id": "x", "actor": "u", "action": "delete", "time": 2}], "u", ["read"], 0, 3]], [("wrong_actor", "event['actor'] == actor", "event['actor'] != actor"), ("ignores_action", "event['action'] in allowed", "True"), ("exclusive_end", "event['time'] <= end", "event['time'] < end")], ["filtering", "boundaries", "audit"]),
        Repair("repair_merge_counters_001", "medium", "merge_counters", "left, right", "Sum matching integer counters from both dictionaries, remove zero totals, and return keys sorted in a dictionary. Preserve inputs.", "totals = dict(left)\nfor key, value in right.items():\n    totals[key] = totals.get(key, 0) + value\nreturn {key: totals[key] for key in sorted(totals) if totals[key] != 0}", [[{"a": 2, "b": 1}, {"a": -2, "c": 4}], [{}, {}], [{"x": 1}, {}], [{}, {"z": 0}]], [("overwrites_left", "totals.get(key, 0) + value", "value"), ("keeps_zero", " if totals[key] != 0", ""), ("drops_left_only", "totals = dict(left)", "totals = {}")], ["aggregation", "immutable_update"]),
        Repair("repair_next_available_slot_001", "medium", "next_available_slot", "start, step, attempts, blocked", "Check start and then step-sized later candidates, up to attempts candidates total. Return the first candidate outside every half-open blocked interval, or None.", "for index in range(attempts):\n    candidate = start + index * step\n    if not any(left <= candidate < right for left, right in blocked):\n        return candidate\nreturn None", [[10, 5, 4, [[10, 15], [20, 30]]], [7, 2, 1, []], [0, 3, 0, []], [5, 1, 2, [[0, 6]]], [20, 5, 2, [[0, 5], [20, 25]]]], [("skips_start", "start + index * step", "start + (index + 1) * step"), ("closed_interval", "left <= candidate < right", "left <= candidate <= right"), ("checks_first_block_only", "for left, right in blocked", "for left, right in blocked[:1]")], ["scheduling", "boundaries", "search"]),
        Repair("repair_sla_breaches_001", "medium", "sla_breaches", "tickets, now, limits", "A ticket is breached when now-created is strictly greater than its priority limit. Ignore closed tickets and unknown priorities. Return ids ordered by largest overdue amount then id.", "rows = []\nfor ticket in tickets:\n    if ticket['open'] and ticket['priority'] in limits:\n        overdue = now - ticket['created'] - limits[ticket['priority']]\n        if overdue > 0:\n            rows.append([ticket['id'], overdue])\nreturn [item_id for item_id, _ in sorted(rows, key=lambda row: (-row[1], row[0]))]", [[[{"id": "a", "priority": "high", "created": 2, "open": True}, {"id": "b", "priority": "low", "created": 0, "open": True}], 10, {"high": 5, "low": 20}], [[], 5, {}], [[{"id": "x", "priority": "p", "created": 0, "open": False}], 9, {"p": 1}], [[{"id": "x", "priority": "p", "created": 4, "open": True}], 9, {"p": 5}], [[{"id": "a", "priority": "p", "created": 5, "open": True}, {"id": "x", "priority": "p", "created": 0, "open": True}], 10, {"p": 1}]], [("includes_deadline", "if overdue > 0", "if overdue >= 0"), ("includes_closed", "ticket['open'] and ", ""), ("ascending_overdue", "(-row[1], row[0])", "(row[1], row[0])")], ["sla", "priority_order", "filtering"]),
        Repair("repair_mask_columns_001", "medium", "mask_columns", "records, fields", "Return new records in input order, replacing each present field named in fields with '***'. Do not add missing fields or mutate inputs.", "hidden = set(fields)\nreturn [{key: ('***' if key in hidden else value) for key, value in record.items()} for record in records]", [[[{"name": "Ada", "email": "a@x"}], ["email"]], [[], ["x"]], [[{"a": 1}], ["missing"]], [[{"x": 0}, {"x": None}], ["x"]]], [("masks_everything", "if key in hidden", "if key not in hidden"), ("ignores_fields", "hidden = set(fields)", "hidden = set()"), ("adds_missing_fields", "return [{key: ('***' if key in hidden else value) for key, value in record.items()} for record in records]", "return [{**record, **{field: '***' for field in hidden}} for record in records]")], ["privacy", "immutable_update", "records"]),
        Repair("repair_window_peaks_001", "medium", "window_peaks", "values, width", "Return the maximum of every full consecutive positive-width window. Return [] when width exceeds the input length.", "if width > len(values):\n    return []\nreturn [max(values[index:index + width]) for index in range(len(values) - width + 1)]", [[[1, 3, 2, 5], 2], [[4], 1], [[], 1], [[1, 2], 3], [[-3, -1, -2], 2]], [("drops_last_window", "range(len(values) - width + 1)", "range(len(values) - width)"), ("wrong_window_end", "index + width", "index + width - 1"), ("uses_minimum", "max(values[index:index + width])", "min(values[index:index + width])")], ["sliding_window", "boundaries"]),
        Repair("repair_paginate_records_001", "medium", "paginate_records", "records, page, size", "Pages are one-based and size is positive. Return a new list containing that page, or [] when it starts past the records.", "start = (page - 1) * size\nend = start + size\nreturn list(records[start:end])", [[[1, 2, 3, 4, 5], 2, 2], [[], 1, 3], [[1, 2], 3, 1], [[1, 2, 3], 1, 5]], [("zero_based_page", "(page - 1) * size", "page * size"), ("short_page", "start + size", "start + size - 1"), ("returns_view_source", "list(records[start:end])", "records[start:start]")], ["pagination", "boundaries"]),
        Repair("repair_reconcile_orders_001", "hard", "reconcile_orders", "events", "Events are [order_id,sequence,status,total]. Keep the greatest sequence per order; equal sequences keep the later event. Exclude cancelled orders and return [id,status,total] ordered by total descending then id.", "latest = {}\nfor order_id, sequence, status, total in events:\n    if order_id not in latest or sequence >= latest[order_id][0]:\n        latest[order_id] = [sequence, status, total]\nrows = [[order_id, value[1], value[2]] for order_id, value in latest.items() if value[1] != 'cancelled']\nreturn sorted(rows, key=lambda row: (-row[2], row[0]))", [[["a", 1, "open", 5], ["b", 1, "open", 9], ["a", 2, "cancelled", 5]], [["x", 1, "open", 2], ["x", 1, "paid", 3]], [], [["b", 1, "open", 4], ["a", 1, "open", 4]], [["x", 3, "paid", 7], ["x", 2, "open", 9]], [["a", 1, "open", 2], ["b", 1, "paid", 8]]], [("keeps_oldest", "sequence >= latest[order_id][0]", "sequence <= latest[order_id][0]"), ("includes_cancelled", " if value[1] != 'cancelled'", ""), ("sorts_ascending_total", "(-row[2], row[0])", "(row[2], row[0])")], ["records", "reconciliation", "tie_breaking"]),
        Repair("repair_token_bucket_001", "hard", "token_bucket", "capacity, refill, requests", "Requests are [time,cost] in nondecreasing time. Start full, refill refill tokens per elapsed unit capped at capacity, and accept only when enough tokens remain. Return booleans in request order.", "tokens = capacity\nlast = 0\nresult = []\nfor time, cost in requests:\n    tokens = min(capacity, tokens + (time - last) * refill)\n    last = time\n    accepted = tokens >= cost\n    if accepted:\n        tokens -= cost\n    result.append(accepted)\nreturn result", [[5, 1, [[0, 4], [1, 3], [3, 3]]], [2, 0, [[0, 1], [1, 2]]], [3, 2, []], [1, 1, [[0, 1], [0, 1]]], [3, 5, [[0, 3], [1, 3]]], [3, 5, [[1, 3], [1, 3]]]], [("no_capacity_cap", "min(capacity, tokens + (time - last) * refill)", "tokens + (time - last) * refill"), ("strict_capacity_check", "tokens >= cost", "tokens > cost"), ("charges_rejected", "if accepted:\n            tokens -= cost", "if not accepted:\n            tokens -= cost")], ["rate_limiting", "state_machine", "time"]),
    ]
    repairs = [repair for repair in repairs if repair.id not in RETIRED_ITEM_IDS]
    items = []
    for offset, repair in enumerate(repairs, start=40):
        source = _source(Task(repair.id, repair.difficulty, repair.name, repair.params, repair.specification, repair.body, repair.cases, repair.tags))
        tests = []
        for args in repair.cases:
            if repair.id in {
                "repair_latest_webhooks_001",
                "repair_clean_tags_001",
                "repair_latest_status_001",
                "repair_reconcile_orders_001",
            }:
                args = [args]
            preserved = [i for i, value in enumerate(args) if isinstance(value, (list, dict))]
            test = {"args": args, "expected": _expected(source, repair.name, args)}
            if preserved:
                test["preserve_args"] = preserved
            tests.append(test)
        mutants = []
        for mutant_id, old, new in repair.mutations:
            if old not in source:
                raise ValueError(f"mutation {mutant_id} does not match {repair.id}")
            mutants.append({"id": mutant_id, "source": source.replace(old, new, 1)})
        buggy = mutants[0]["source"]
        failing_test = next(
            (
                test
                for test in tests
                if _observed(buggy, repair.name, test["args"])
                != {"returned": test["expected"]}
            ),
            None,
        )
        if failing_test is None:
            raise ValueError(f"shown bug has no failing regression: {repair.id}")
        actual = _observed(buggy, repair.name, failing_test["args"])
        prompt = (
            "A regression report identifies the smallest helper that needs repair.\n\n"
            f"Contract: {repair.specification}\n\n"
            f"Current code:\n{buggy}\n\n"
            "Failing regression:\n"
            f"- arguments: {json.dumps(failing_test['args'], ensure_ascii=False)}\n"
            f"- expected return: {json.dumps(failing_test['expected'], ensure_ascii=False)}\n"
            f"- actual outcome: {json.dumps(actual, ensure_ascii=False)}\n\n"
            f"Make the smallest correction needed. Keep the {repair.name} signature and "
            "the no-import convention. Return only the corrected function."
        )
        items.append({
            "id": repair.id,
            "subcategory": "code_repair",
            "difficulty": repair.difficulty,
            "split": "dev" if offset % 4 == 0 else "test",
            "visibility": "public" if offset % 2 == 0 else "held_out",
            "prompt": prompt,
            "response_contract": {"type": "code", "format": "python_function"},
            "expected": {"value": {"entry_point": repair.name, "tests": tests, "reference_solution": source, "mutants": mutants}},
            "scoring": {"method": "executable_python", "parameters": {"timeout_seconds": 1.0, "memory_limit_mb": 128, "max_output_characters": 10000}},
            "provenance": {"kind": "synthetic", "review_status": "human_checked", "generator": GENERATOR, "seed": SEED},
            "tags": ["python", "repair", "generated_mutation", "failing_test_context", "pass_at_1", *repair.tags],
        })
    return items


def _regression_test_items() -> list[dict[str, Any]]:
    """Select the one regression test that exposes a stated implementation risk."""
    cases = [
        ("test_traceback_missing_region_001", "easy", "def region_name(config):\n    return config['deployment']['region'].strip().lower()\n\nTraceback (most recent call last):\n  File \"deploy_report.py\", line 41, in <module>\n    region_name({'deployment': {}})\n  File \"deploy_report.py\", line 2, in region_name\n    return config['deployment']['region'].strip().lower()\nKeyError: 'region'", "config always contains a deployment object. Return 'unknown' when deployment.region is missing; otherwise return its trimmed lowercase value.", {"missing_region": "region_name({'deployment': {}}) == 'unknown'", "present_region": "region_name({'deployment': {'region': ' West '}}) == 'west'", "whitespace_region": "region_name({'deployment': {'region': '   '}}) == ''"}, "missing_region", ["traceback", "missing_nested_key", "diagnostic_label"]),
        ("test_expiry_boundary_001", "easy", "def is_expired(expires_at, now):\n    return now > expires_at", "An item is expired when now is at or after expires_at.", {"before_boundary": "is_expired(10, 9) is False", "exact_boundary": "is_expired(10, 10) is True", "after_boundary": "is_expired(10, 11) is True"}, "exact_boundary", ["boundaries", "time"]),
        ("test_latest_duplicate_001", "easy", "def latest(rows):\n    return {key: value for key, value in rows}", "Later rows for the same key must win.", {"empty_rows": "latest([]) == {}", "distinct_keys": "latest([['a',1],['b',2]]) == {'a':1,'b':2}", "duplicate_key": "latest([['a',1],['a',2]]) == {'a':2}"}, "duplicate_key", ["records", "deduplication"]),
        ("test_lookup_negative_index_001", "medium", "def lookup(values, index, default):\n    try:\n        return values[index]\n    except IndexError:\n        return default", "Only non-negative indices are valid; invalid indices return default.", {"in_range": "lookup([4,5], 1, 0) == 5", "past_end": "lookup([4,5], 2, 0) == 0", "negative_index": "lookup([4,5], -1, 0) == 0", "empty_values": "lookup([], 0, 0) == 0"}, "negative_index", ["boundaries", "python_semantics"]),
        ("test_touching_windows_001", "medium", "def overlap(left, right):\n    return left[1] >= right[0] and right[1] >= left[0]", "Intervals are half-open, so touching endpoints do not overlap.", {"separate": "overlap([1,2],[3,4]) is False", "intersecting": "overlap([1,3],[2,4]) is True", "touching": "overlap([1,2],[2,4]) is False", "identical": "overlap([1,2],[1,2]) is True"}, "touching", ["intervals", "boundaries"]),
        ("test_rank_tie_001", "medium", "def rank(items):\n    return sorted(items, key=lambda row: -row['score'])", "Sort by score descending, then id ascending for deterministic ties.", {"different_scores": "ids for scores 9 and 5 are ordered high first", "empty_items": "rank([]) == []", "equal_score_ids": "equal scores supplied as b,a return a,b", "negative_scores": "-1 ranks above -2"}, "equal_score_ids", ["tie_breaking", "determinism"]),
        ("test_retry_state_001", "medium", "attempts = 0\ndef retry(ok):\n    global attempts\n    while attempts < 3:\n        attempts += 1\n        if ok:\n            return attempts\n    return None", "Every call receives a fresh three-attempt budget.", {"success_first_call": "retry(True) == 1", "failure_first_call": "retry(False) is None", "repeated_calls": "retry(True) followed by retry(True) returns 1 both times", "boolean_false": "retry(False) makes three attempts"}, "repeated_calls", ["state_scope", "reentrancy"]),
        ("test_unicode_username_001", "medium", "def same_user(left, right):\n    return left.lower() == right.lower()", "Compare user names using Unicode-aware case-insensitive normalization.", {"ascii_case": "same_user('Ada','ADA') is True", "different_names": "same_user('Ada','Lin') is False", "unicode_casefold": "same_user('straße','STRASSE') is True", "empty_names": "same_user('','') is True"}, "unicode_casefold", ["text_normalization", "unicode"]),
        ("test_partial_dependency_cycle_001", "hard", "def order(nodes, edges):\n    needs = {node: set() for node in nodes}\n    for job, prerequisite in edges:\n        needs[job].add(prerequisite)\n    result = []\n    while len(result) < len(nodes):\n        ready = sorted(node for node in nodes if node not in result and needs[node] <= set(result))\n        if not ready:\n            return result\n        result.extend(ready)\n    return result", "Return a complete dependency order, or [] when any cycle prevents completion.", {"independent_nodes": "order(['a','b'], []) == ['a','b']", "simple_chain": "order(['a','b'], [['b','a']]) == ['a','b']", "partial_cycle": "order(['a','b','c'], [['a','b'],['b','a']]) == []", "one_node": "order(['a'], []) == ['a']"}, "partial_cycle", ["graph", "cycle_detection", "partial_progress"]),
        ("test_input_mutation_001", "hard", "def sorted_active(records):\n    records.sort(key=lambda row: row['id'])\n    return [row for row in records if row['active']]", "Return active records sorted by id without mutating the input list or its records.", {"sorted_output": "unsorted active rows return in id order", "inactive_filter": "inactive rows are absent", "empty_records": "an empty input returns []", "preserve_input": "the input remains byte-for-byte equal after the call"}, "preserve_input", ["immutable_input", "side_effects"]),
    ]
    cases = [case for case in cases if case[0] not in RETIRED_ITEM_IDS]
    items = []
    for index, (item_id, difficulty, source, contract, options, answer, tags) in enumerate(cases):
        option_text = "\n".join(f"- {label}: {test}" for label, test in options.items())
        prompt = f"Review this function:\n\n{source}\n\nContract: {contract}\n\nWhich single regression test most directly exposes the defect?\n{option_text}\n\nReturn only the option label."
        if item_id == "test_traceback_missing_region_001":
            prompt = (
                "this crashed while I was generating a deploy report. which one test "
                "should I add for the exact bug in this traceback?\n\n"
                f"{source}\n\nExpected behavior: {contract}\n\n"
                "Pick the single regression test that fails because of this defect:\n"
                f"{option_text}\n\nReturn only the option label."
            )
        items.append({
            "id": item_id,
            "subcategory": "regression_test_selection",
            "difficulty": difficulty,
            "split": "dev" if index % 4 == 0 else "test",
            "visibility": "public" if index % 2 == 0 else "held_out",
            "prompt": prompt,
            "response_contract": {"type": "text", "format": "diagnostic_label"},
            "expected": {"value": answer},
            "scoring": {"method": "exact_match", "parameters": {"strip": True, "case_sensitive": False}},
            "provenance": {"kind": "synthetic", "review_status": "human_checked", "generator": GENERATOR, "seed": SEED},
            "tags": ["python", "testing", "regression_selection", *tags],
        })
    return items


def generate() -> str:
    items = (
        _implementation_items()
        + _diagnosis_items()
        + _repair_items()
        + _regression_test_items()
    )
    header = ["schema_version: 1", "benchmark: code_debug_repair", f"generated_by: {GENERATOR}", f"seed: {SEED}", "items:"]
    lines = header + ["  - " + json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in items]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generate()
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit(f"out of date: {OUTPUT}")
    else:
        OUTPUT.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
