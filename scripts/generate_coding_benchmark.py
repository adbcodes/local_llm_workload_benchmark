from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "code_debug_repair" / "generated_questions.yaml"
GENERATOR = "generate_coding_benchmark.py"
SEED = 20260629


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
    namespace: dict[str, Any] = {}
    exec(source, namespace, namespace)
    return namespace[name](*json.loads(json.dumps(args)))


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
        Task("code_deployment_impact_001", "hard", "deployment_impact", "services, dependencies, changed, protected", "dependencies are [service,dependency]. Return affected services reachable from changed dependencies, excluding protected services and anything depending on an excluded service, ordered in deterministic dependency batches.", "blocked = set(protected)\naffected = set(changed) - blocked\nprogress = True\nwhile progress:\n    progress = False\n    for service, dependency in dependencies:\n        if service not in blocked and dependency in affected and service not in affected:\n            affected.add(service)\n            progress = True\nremaining = set(affected)\nresult = []\nwhile remaining:\n    batch = sorted(service for service in remaining if all(dependency not in remaining for owner, dependency in dependencies if owner == service))\n    if not batch:\n        return []\n    result.append(batch)\n    remaining.difference_update(batch)\nreturn result", [[["db", "api", "web"], [["api", "db"], ["web", "api"]], ["db"], []], [["a", "b"], [["b", "a"]], ["a"], ["b"]], [["a"], [], [], []], [["a", "b", "c"], [["c", "a"], ["b", "a"]], ["a"], []]], ["graph", "dependency_propagation"]),
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
            "prompt": f"Implement {task.name}({task.params}). {task.specification} Do not mutate inputs. Return only the function definition and use no imports.",
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
        "mutation_iteration, missing_finalization, tie_break_error"
    )
    cases = [
        ("diagnose_page_cursor_001", "easy", "boundary_update", "def page_after(items, cursor):\n    index = items.index(cursor)\n    return items[index:]", "Failing check: page_after(['a','b','c'], 'b') expected ['c']; actual ['b','c'].", ["boundaries"]),
        ("diagnose_daily_totals_001", "medium", "state_scope", "def daily_totals(days):\n    total = 0\n    result = []\n    for values in days:\n        for value in values: total += value\n        result.append(total)\n    return result", "Regression case: daily_totals([[2],[3]]) expected [2,3]; actual [2,5].", ["state_scope"]),
        ("diagnose_matrix_template_001", "easy", "row_aliasing", "def matrix(rows, cols):\n    values = [[None] * cols] * rows\n    values[0][0] = 'x'\n    return values", "matrix(2,2) should change only the first row, but both rows start with 'x'.", ["aliasing"]),
        ("diagnose_access_rule_001", "medium", "wrong_precedence", "def allowed(active, admin, suspended):\n    return active or admin and not suspended", "Policy: a user must not be suspended and must be active or an admin. Input (True, False, True) expected False; actual True.", ["boolean_logic"]),
        ("diagnose_price_cache_001", "medium", "stale_cache_key", "def priced(items, tax):\n    cache = {}\n    def one(item):\n        if item['sku'] in cache: return cache[item['sku']]\n        value = item['price'] + tax\n        cache[item['sku']] = value\n        return value\n    return [one(item) for item in items]", "Input [{'sku':'x','price':10},{'sku':'x','price':14}] with tax 2 expected [12,16]; actual [12,12].", ["memoization"]),
        ("diagnose_dependency_edges_001", "hard", "direction_error", "def impacted(edges, changed):\n    result = set(changed)\n    progress = True\n    while progress:\n        progress = False\n        for service, dependency in edges:\n            if service in result and dependency not in result:\n                result.add(dependency)\n                progress = True\n    return sorted(result)", "Edges are [service, dependency]. For [['api','db'],['web','api']] and changed ['db'], expected ['api','db','web']; actual ['db'].", ["graph", "transitive_state"]),
        ("diagnose_average_latency_001", "easy", "lossy_conversion", "def average_latency(values):\n    return int(sum(values) / len(values))", "Failing check: average_latency([1,2]) expected 1.5; actual 1.", ["numeric_semantics"]),
        ("diagnose_remove_expired_001", "medium", "mutation_iteration", "def remove_expired(records):\n    for record in records:\n        if record['expired']:\n            records.remove(record)\n    return records", "Two adjacent expired records are supplied. The second one remains in the returned list.", ["collection_mutation"]),
        ("diagnose_flush_groups_001", "medium", "missing_finalization", "def groups(values):\n    result = []\n    current = []\n    for value in values:\n        if current and value != current[-1]:\n            result.append(current); current = []\n        current.append(value)\n    return result", "For ['a','a','b'], expected [['a','a'],['b']]; actual [['a','a']].", ["state_machine"]),
        ("diagnose_route_choice_001", "hard", "tie_break_error", "def choose_candidate(candidates):\n    return min(candidates, key=lambda item: (-item['score'], item['name'], item['latency']))", "Selection priority is highest score, then lowest latency, then name. Candidates [{name:'alpha',score:9,latency:8},{name:'beta',score:9,latency:3}] should select beta; actual alpha.", ["multi_key_ordering", "tie_breaking"]),
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


def _repair_items() -> list[dict[str, Any]]:
    repairs = [
        Repair("repair_quota_adjustments_001", "easy", "quota_after_adjustments", "quota, adjustments", "Apply integer adjustments in order and clamp the result to zero after every adjustment.", "value = quota\nfor adjustment in adjustments:\n    value = max(0, value + adjustment)\nreturn value", [[5, [-3, -4, 2]], [0, [3]], [5, [5, -10]], [4, [-4, -1]]], [("clamp_only_at_end", "value = max(0, value + adjustment)", "value = value + adjustment"), ("ignore_order", "for adjustment in adjustments:", "for adjustment in sorted(adjustments):"), ("wrong_floor", "max(0, value + adjustment)", "max(1, value + adjustment)")], ["state_tracking"]),
        Repair("repair_latest_webhooks_001", "easy", "latest_webhooks", "events", "Keep the greatest sequence per webhook id; equal sequences keep the later event. Return [id,payload] rows ordered by id.", "latest = {}\nfor event_id, sequence, payload in events:\n    if event_id not in latest or sequence >= latest[event_id][0]:\n        latest[event_id] = [sequence, payload]\nreturn [[event_id, latest[event_id][1]] for event_id in sorted(latest)]", [[["b", 1, "x"], ["a", 2, "old"], ["a", 3, "new"]], [["a", 1, "x"], ["a", 1, "y"]], [], [["z", 0, "p"]]], [("keeps_smallest", "sequence >= latest[event_id][0]", "sequence <= latest[event_id][0]"), ("drops_equal_update", "sequence >= latest[event_id][0]", "sequence > latest[event_id][0]"), ("input_order_output", "for event_id in sorted(latest)", "for event_id in latest")], ["records", "versioning"]),
        Repair("repair_refund_total_001", "easy", "refund_total", "refunds, reversed_ids", "Sum positive refund amounts whose ids are not reversed. Duplicate non-reversed ids count once using their latest amount.", "reversed_set = set(reversed_ids)\nlatest = {}\nfor refund_id, amount in refunds:\n    if refund_id not in reversed_set and amount > 0:\n        latest[refund_id] = amount\nreturn sum(latest.values())", [[[["r1", 3], ["r2", 5], ["r1", 4]], ["r2"]], [[], []], [[["r", -2]], []], [[["r", 2]], ["r"]]], [("counts_duplicates", "latest[refund_id] = amount", "latest[refund_id] = latest.get(refund_id, 0) + amount"), ("includes_reversed", "refund_id not in reversed_set", "refund_id in reversed_set"), ("includes_negative", "and amount > 0", "")], ["aggregation", "deduplication"]),
        Repair("repair_availability_windows_001", "medium", "availability_windows", "windows, minimum", "Merge overlapping or touching half-open windows and return merged windows whose length is at least minimum.", "merged = []\nfor start, end in sorted(windows):\n    if merged and start <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], end)\n    else:\n        merged.append([start, end])\nreturn [window for window in merged if window[1] - window[0] >= minimum]", [[[[1, 3], [3, 5], [8, 9]], 2], [[], 1], [[[5, 7], [1, 2]], 1], [[[1, 2]], 2]], [("does_not_merge_touching", "start <= merged[-1][1]", "start < merged[-1][1]"), ("keeps_short_windows", ">= minimum", "> 0"), ("loses_final_window", "return [window for window in merged", "return [window for window in merged[:-1]")], ["intervals", "boundaries"]),
        Repair("repair_compact_ranges_001", "medium", "compact_ranges", "values", "Sort distinct integers and compress each consecutive run into [start,end]. Return ranges in ascending order.", "numbers = sorted(set(values))\nif not numbers:\n    return []\nresult = []\nstart = numbers[0]\nprevious = numbers[0]\nfor value in numbers[1:]:\n    if value == previous + 1:\n        previous = value\n    else:\n        result.append([start, previous])\n        start = value\n        previous = value\nresult.append([start, previous])\nreturn result", [[[1, 2, 3, 7, 8, 10]], [[]], [[3, 1, 2, 2]], [[-2, -1, 1]], [[5, 7, 6, 10]]], [("merges_single_gap", "value == previous + 1", "value <= previous + 2"), ("drops_final_range", "result.append([start, previous])\n    return result", "return result"), ("keeps_duplicates", "sorted(set(values))", "sorted(values)")], ["sequence_processing", "range_compaction"]),
        Repair("repair_rolling_totals_001", "medium", "rolling_totals", "values, width", "Return the sum of every full consecutive window of the positive size width. Return [] when width exceeds the input length.", "if width > len(values):\n    return []\ncurrent = sum(values[:width])\nresult = [current]\nfor index in range(width, len(values)):\n    current += values[index] - values[index - width]\n    result.append(current)\nreturn result", [[[1, 2, 3, 4], 2], [[5], 1], [[], 1], [[3, -1, 2], 3], [[2, 0, 2, 0], 3]], [("wrong_outgoing_index", "values[index - width]", "values[index - width + 1]"), ("skips_second_window", "range(width, len(values))", "range(width + 1, len(values))"), ("drops_last_window", "return result", "return result[:-1]")], ["sliding_window", "arithmetic"]),
        Repair("repair_lookup_path_001", "medium", "lookup_path", "data, path, default", "Follow string tokens through dictionaries and non-negative integer tokens through lists. Return default when a token is missing, has the wrong type, or is out of range. A stored None is a valid result.", "current = data\nfor token in path:\n    try:\n        if token < 0:\n            return default\n    except TypeError:\n        pass\n    try:\n        current = current[token]\n    except (KeyError, IndexError, TypeError):\n        return default\nreturn current", [[{"user": {"names": ["Ada", "Lin"]}}, ["user", "names", 1], "missing"], [{"x": None}, ["x"], "missing"], [[10, 20], [-1], "missing"], [{"x": [0]}, ["x", 0], 99], [{"x": 1}, ["x", "y"], "missing"]], [("allows_negative_index", "if token < 0", "if False"), ("treats_none_as_missing", "return current", "return default if current is None else current"), ("rejects_falsy_value", "current = current[token]", "current = current[token]\n        if not current:\n            return default")], ["nested_data", "type_boundaries"]),
        Repair("repair_lru_cache_001", "hard", "lru_cache", "capacity, accesses", "Simulate a positive-capacity cache of keys. A hit moves its key to most-recent position. A miss at capacity evicts the least-recent key. Return [evicted_keys, final_keys] with final keys ordered least to most recent.", "order = []\nevicted = []\nfor key in accesses:\n    if key in order:\n        order.remove(key)\n        order.append(key)\n    else:\n        if len(order) == capacity:\n            evicted.append(order.pop(0))\n        order.append(key)\nreturn [evicted, order]", [[2, ["a", "b", "a", "c"]], [1, ["x", "y", "y", "z"]], [3, []], [3, ["a", "b", "c", "a"]], [2, ["a", "b", "c", "b", "d"]]], [("duplicates_cache_hits", "order.remove(key)", "order = order"), ("evicts_most_recent", "order.pop(0)", "order.pop()"), ("evicts_too_late", "len(order) == capacity", "len(order) > capacity")], ["cache", "state_machine", "event_ordering"]),
    ]
    items = []
    for offset, repair in enumerate(repairs, start=40):
        source = _source(Task(repair.id, repair.difficulty, repair.name, repair.params, repair.specification, repair.body, repair.cases, repair.tags))
        tests = []
        for args in repair.cases:
            if repair.id == "repair_latest_webhooks_001":
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
        prompt = (
            f"Repair this function so it satisfies the contract: {repair.specification}\n\n"
            f"{buggy}\n\nReturn only the corrected {repair.name} function and use no imports."
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
            "tags": ["python", "repair", "generated_mutation", "pass_at_1", *repair.tags],
        })
    return items


def generate() -> str:
    items = _implementation_items() + _diagnosis_items() + _repair_items()
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
