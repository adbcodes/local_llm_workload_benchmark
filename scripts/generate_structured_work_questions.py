from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "structured_work_v1"
SEED = 20260723


def _base_item(
    benchmark: str,
    number: int,
    subcategory: str,
    difficulty: str,
    prompt: str,
    contract_type: str,
    expected: Any,
    method: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{benchmark}_{number:03d}",
        "subcategory": subcategory,
        "difficulty": difficulty,
        "split": "dev" if number % 2 else "test",
        "visibility": "public" if number % 2 else "held_out",
        "prompt": prompt.strip(),
        "response_contract": {"type": contract_type, "format": None},
        "expected": {"value": expected},
        "scoring": {"method": method, "parameters": parameters or {}},
        "provenance": {
            "kind": "synthetic",
            "review_status": "draft",
            "generator": GENERATOR,
            "seed": SEED,
        },
        "tags": ["fresh_generated", subcategory],
    }


def _table_items() -> list[dict[str, Any]]:
    specs = [
        ("totals_and_differences", "easy", "Monthly units:\nJan | 120\nFeb | 150\nMar | 130\nWhat is the total? Return only the number.", "number", 400, "numeric_tolerance"),
        ("totals_and_differences", "easy", "Team | Tickets closed\nA | 92\nB | 75\nHow many more tickets did A close than B? Return only the number.", "number", 17, "numeric_tolerance"),
        ("decision_from_table_evidence", "easy", "Region | Orders\nEast | 44\nWest | 61\nNorth | 58\nWhich region has the most orders? Return only the region name.", "text", "West", "exact_match"),
        ("decision_from_table_evidence", "easy", "Item | Stock\nP | 12\nQ | 5\nR | 9\nWhich item has the lowest stock? Return only its item letter.", "text", "Q", "exact_match"),
        ("decision_from_table_evidence", "easy", "Agent | Score\nA | 82\nB | 77\nC | 91\nD | 80\nReturn the agents scoring at least 80 as comma-separated letters only.", "text", ["A", "C", "D"], "set_match"),
        ("small_joins", "easy", "Price table: P = 20, Q = 15.\nOrder table: P quantity = 3, Q quantity = 4.\nWhat is the total order value? Return only the number.", "number", 120, "numeric_tolerance"),
        ("trends_with_exception_rows", "easy", "Product | Week 1 | Week 2\nA | 10 | 12\nB | 8 | 7\nC | 5 | 9\nWhich product is the only one that declined? Return only its letter.", "text", "B", "exact_match"),
        ("duplicated_rows", "easy", "Row | Invoice ID\n1 | 101\n2 | 102\n3 | 101\n4 | 103\nWhich invoice ID is duplicated? Return only the ID.", "number", 101, "numeric_tolerance"),
        ("totals_and_differences", "medium", "Order | Status | Amount\nA | completed | 120\nB | cancelled | 90\nC | completed | 140\nWhat is the total for completed orders only? Return only the number.", "number", 260, "numeric_tolerance"),
        ("small_joins", "medium", "Customers: C1 = North, C2 = South, C3 = West.\nOpen orders belong to C1 and C3.\nReturn the regions with open orders as comma-separated names only.", "text", ["North", "West"], "set_match"),
        ("trends_with_exception_rows", "medium", "Product | Start | End\nA | 100 | 130\nB | 50 | 70\nC | 200 | 230\nWhich product has the largest percentage growth? Return only its letter.", "text", "B", "exact_match"),
        ("small_joins", "medium", "Product | Quantity | Unit price\nP | 5 | 120\nQ | 8 | 75\nR | 4 | 200\nWhat is total revenue? Return only the number.", "number", 2000, "numeric_tolerance"),
        ("duplicated_rows", "medium", "Invoice | Amount\nI1 | 100\nI1 | 100\nI2 | 250\nI3 | 150\nThe repeated I1 row is an accidental duplicate. What is the deduplicated total? Return only the number.", "number", 500, "numeric_tolerance"),
        ("inconsistent_rows", "medium", "Item | Qty | Unit price | Reported total\nA | 2 | 10 | 20\nB | 3 | 8 | 25\nC | 4 | 6 | 24\nWhich item has an inconsistent total? Return only its letter.", "text", "B", "exact_match"),
        ("decision_from_table_evidence", "medium", "Vendor | Price | Rating | Lead days\nV1 | 48 | 4.6 | 6\nV2 | 50 | 4.5 | 5\nV3 | 45 | 4.2 | 4\nChoose vendors with price at most 50, rating at least 4.5, and lead time at most 5 days. Return comma-separated vendor IDs only.", "text", ["V2"], "set_match"),
        ("totals_and_differences", "medium", "Run | Score\n1 | 70\n2 | missing\n3 | 90\n4 | 80\nWhat is the average of the available scores? Return only the number.", "number", 80, "numeric_tolerance"),
        ("small_joins", "medium", "Invoices: C1 overdue, C2 paid, C3 overdue.\nCustomers: C1 active, C2 active, C3 inactive.\nReturn active customer IDs with overdue invoices, comma-separated only.", "text", ["C1"], "set_match"),
        ("trends_with_exception_rows", "medium", "Product | Jan | Feb | Mar\nA | 10 | 12 | 14\nB | 20 | 18 | 22\nC | 5 | 6 | 7\nWhich product breaks the month-by-month upward trend? Return only its letter.", "text", "B", "exact_match"),
        ("decision_from_table_evidence", "medium", "Item | Stock | Reorder point | Inbound\nA | 4 | 10 | 0\nB | 8 | 8 | 0\nC | 2 | 5 | 6\nReorder only when stock is below the reorder point and no units are inbound. Return item letters, comma-separated only.", "text", ["A"], "set_match"),
        ("totals_and_differences", "medium", "Ten deliveries were checked: 8 were on time and 2 were late. What percentage were on time? Return only the number without a percent sign.", "number", 80, "numeric_tolerance"),
        ("decision_from_table_evidence", "medium", "Campaign | Cost | Conversions\nA | 1000 | 25\nB | 700 | 14\nC | 1200 | 40\nChoose the lowest cost per conversion among campaigns with at least 20 conversions. Return only the campaign letter.", "text", "C", "exact_match"),
        ("inconsistent_rows", "medium", "Account | Time | Status\nX | 10:00 | active\nX | 12:00 | suspended\nY | 11:00 | active\nUsing the latest row for each account, what is X's status? Return one word.", "text", "suspended", "exact_match"),
        ("decision_from_table_evidence", "medium", "Site | Revenue | Target | Complaints\nP | 105 | 100 | 1\nQ | 120 | 100 | 4\nR | 90 | 85 | 2\nReturn sites that beat target and have at most 2 complaints, comma-separated only.", "text", ["P", "R"], "set_match"),
        ("small_joins", "hard", "Sales: A sold 10 units; B sold 5.\nPrices: A = 50; B = 100.\nUnit costs: A = 30; B = 70.\nWhat is total profit across both products? Return only the number.", "number", 350, "numeric_tolerance"),
        ("decision_from_table_evidence", "hard", "Cohort | Starting users | Month-3 users\nA | 100 | 70\nB | 80 | 60\nC | 50 | 35\nWhich cohort has the highest month-3 retention rate? Return only its letter.", "text", "B", "exact_match"),
        ("duplicated_rows", "hard", "Event | Time | Amount | Status\ne1 | 09:00 | 30 | valid\ne1 | 10:00 | 40 | valid\ne2 | 09:30 | 25 | valid\ne3 | 08:00 | 10 | valid\ne3 | 11:00 | 10 | cancelled\nKeep only the latest row per event, then sum amounts for rows whose latest status is valid. Return only the number.", "number", 65, "numeric_tolerance"),
        ("decision_from_table_evidence", "hard", "Region | Demand | Capacity | Transfer out\nEast | 120 | 100 | 15\nWest | 75 | 90 | 0\nNorth | 60 | 65 | 0\nAfter transfers, demand is demand minus transfer out. Which regions remain over capacity? Return comma-separated names only.", "text", ["East"], "set_match"),
        ("small_joins", "hard", "Vendor | Price | Currency | Rating | Lead days\nA | 100 | USD | 4.6 | 6\nB | 90 | EUR | 4.5 | 7\nC | 8200 | INR | 4.4 | 5\nRates: 1 USD = 83 INR; 1 EUR = 92 INR. Choose the cheapest vendor with rating at least 4.4 and lead time at most 7 days. Return only its letter.", "text", "C", "exact_match"),
        ("trends_with_exception_rows", "hard", "Product | Jan | Feb | Mar\nA | 100 | 90 | 80\nB | 50 | 70 | 100\nTotals | 150 | 160 | 180\nThe total rises each month. Which product nevertheless declines every month? Return only its letter.", "text", "A", "exact_match"),
        ("inconsistent_rows", "hard", "Invoice | Billed | Paid\nI1 | 100 | 100\nI2 | 120 | 100\nI3 | 80 | 90\nReturn every invoice whose paid amount does not equal its billed amount, comma-separated only.", "text", ["I2", "I3"], "set_match"),
    ]
    items = []
    for number, (subcategory, difficulty, prompt, contract, expected, method) in enumerate(specs, 1):
        params = {"absolute_tolerance": 0, "allow_surrounding_text": True} if method == "numeric_tolerance" else ({"separator": ",", "case_sensitive": False} if method == "set_match" else {"strip": True, "case_sensitive": False})
        items.append(_base_item("tables_to_decisions", number, subcategory, difficulty, prompt, contract, expected, method, params))
    return items


ROUTING_LABELS = "billing, account_access, delivery, cancellation, product_support, security, sales, feedback, needs_information, out_of_scope, urgent"


def _inbox_items() -> list[dict[str, Any]]:
    specs = [
        ("single_label_routing", "easy", "I was charged twice for the same monthly plan.", ["billing"]),
        ("single_label_routing", "easy", "I forgot my password and cannot sign in.", ["account_access"]),
        ("single_label_routing", "easy", "My parcel was due yesterday and still has not arrived.", ["delivery"]),
        ("single_label_routing", "easy", "Please cancel my subscription before it renews.", ["cancellation"]),
        ("single_label_routing", "easy", "The mobile app crashes whenever I upload a photo.", ["product_support"]),
        ("single_label_routing", "easy", "I received a login alert from a device I do not recognize.", ["security"]),
        ("single_label_routing", "easy", "Can you share enterprise pricing for 200 seats?", ["sales"]),
        ("single_label_routing", "easy", "The new search filters are excellent. Thank you!", ["feedback"]),
        ("multi_label_routing", "medium", "Please help today: a duplicate charge has left my account overdrawn.", ["billing", "urgent"]),
        ("urgency_under_polite_tone", "medium", "When convenient, could you check the unauthorized transfer? It happened minutes ago and more money may leave.", ["security", "urgent"]),
        ("multi_label_routing", "medium", "My temperature-sensitive medicine was promised this morning. It must arrive within two hours.", ["delivery", "urgent"]),
        ("multi_label_routing", "medium", "I cancelled yesterday but was renewed and charged this morning.", ["cancellation", "billing"]),
        ("ask_for_more_information_cases", "medium", "It is broken again. Please fix it.", ["needs_information"]),
        ("out_of_scope_messages", "medium", "Can your team cater lunch for a wedding next Sunday?", ["out_of_scope"]),
        ("multi_label_routing", "medium", "Before buying the business plan, I need to know whether your API supports SAML.", ["sales", "product_support"]),
        ("multi_label_routing", "medium", "An unknown person changed my two-factor number and now I am locked out.", ["security", "account_access"]),
        ("urgency_under_polite_tone", "medium", "Sorry to bother you, but my flight is tonight and I cannot access the account holding my ticket.", ["account_access", "urgent"]),
        ("single_label_routing", "medium", "Tracking has not updated for four days. Order ID is R-8821. Where is it?", ["delivery"]),
        ("ask_for_more_information_cases", "medium", "Please sort this out as soon as you can.", ["needs_information"]),
        ("multi_label_routing", "medium", "Dark mode is a great addition, but its save button does nothing on Android.", ["feedback", "product_support"]),
        ("single_label_routing", "medium", "I enjoyed the service, but I no longer need it after this billing period. Do not renew it.", ["cancellation"]),
        ("single_label_routing", "medium", "Why is tax shown twice on invoice INV-44?", ["billing"]),
        ("out_of_scope_messages", "medium", "I would like to apply for a software engineering job at your company.", ["out_of_scope"]),
        ("multi_label_routing", "hard", "Someone replaced my recovery email, made purchases, and signed me out. Freeze access immediately.", ["security", "account_access", "urgent"]),
        ("multi_label_routing", "hard", "I paid extra for express shipping, but the package arrived after the event. Please refund the express fee.", ["delivery", "billing"]),
        ("multi_label_routing", "hard", "A renewal I cancelled was charged today, and the bank dispute deadline is in three hours.", ["cancellation", "billing", "urgent"]),
        ("ask_for_more_information_cases", "hard", "The same problem from last time is back. You already know what I mean.", ["needs_information"]),
        ("multi_label_routing", "hard", "Your desktop app shows an invalid security certificate and refuses to connect.", ["product_support", "security"]),
        ("single_label_routing", "hard", "We want to discuss a reseller partnership and volume pricing for our clients.", ["sales"]),
        ("multi_label_routing", "hard", "Our warehouse integration stopped creating shipping labels, so today's dispatch is blocked.", ["product_support", "urgent"]),
    ]
    items = []
    for number, (subcategory, difficulty, message, labels) in enumerate(specs, 1):
        prompt = f"Available labels: {ROUTING_LABELS}.\nMessage: {message}\nReturn every applicable label as a comma-separated list. Return labels only."
        items.append(_base_item("inbox_routing", number, subcategory, difficulty, prompt, "text", labels, "set_match", {"separator": ",", "case_sensitive": False}))
    return items


def _tool_items() -> list[dict[str, Any]]:
    specs: list[tuple[str, str, str, list[dict[str, Any]], list[Any], dict[str, Any]]] = [
        ("no_tool_needed_traps", "easy", "Tools available: calculator(expression). The user asks: What is 2 + 2? Do not call a tool for arithmetic you can do directly.", [], [], {"answer": 4}),
        ("argument_conversion", "easy", "Tools available: get_weather(city). Find the current weather in Pune.", [{"tool": "get_weather", "arguments": {"city": "Pune"}}], [{"temperature_c": 29, "condition": "cloudy"}], {"city": "Pune", "temperature_c": 29, "condition": "cloudy"}),
        ("no_tool_needed_traps", "easy", "Tools available: get_order(order_id). The message already says order A-17 is delivered. Report that state without calling the tool.", [], [], {"order_id": "A-17", "status": "delivered"}),
        ("argument_conversion", "easy", "Tools available: get_inventory(sku). Check stock for SKU pen-blue-10. Tool SKUs must be uppercase.", [{"tool": "get_inventory", "arguments": {"sku": "PEN-BLUE-10"}}], [{"available": 42}], {"sku": "PEN-BLUE-10", "available": 42}),
        ("unnecessary_call_avoidance", "easy", "Tools available: send_email(to, subject, body). Draft a subject line for a receipt email. Do not send anything.", [], [], {"subject": "Your receipt"}),
        ("two_call_chains", "medium", "Tools available: find_customer(email), list_open_orders(customer_id). Find open orders for maya@example.com.", [{"tool": "find_customer", "arguments": {"email": "maya@example.com"}}, {"tool": "list_open_orders", "arguments": {"customer_id": "C-42"}}], [{"customer_id": "C-42"}, {"order_ids": ["O-7", "O-9"]}], {"customer_id": "C-42", "open_orders": ["O-7", "O-9"]}),
        ("using_one_call_result_in_the_next", "medium", "Tools available: get_weather(city), create_event(title, start_time). Check Bengaluru weather, then create 'Morning walk' at 07:00 only if there is no rain.", [{"tool": "get_weather", "arguments": {"city": "Bengaluru"}}, {"tool": "create_event", "arguments": {"title": "Morning walk", "start_time": "07:00"}}], [{"rain": False}, {"event_id": "E-18"}], {"event_created": True, "event_id": "E-18"}),
        ("argument_conversion", "medium", "Tools available: schedule_timer(seconds). Set a timer for 5 minutes. Convert minutes to seconds before calling.", [{"tool": "schedule_timer", "arguments": {"seconds": 300}}], [{"timer_id": "T-5"}], {"timer_id": "T-5", "seconds": 300}),
        ("error_recovery", "medium", "Tools available: lookup_user(username), lookup_user(email). Find Priya. Try username 'priya' first; if not found, retry with email priya@example.com.", [{"tool": "lookup_user", "arguments": {"username": "priya"}}, {"tool": "lookup_user", "arguments": {"email": "priya@example.com"}}], [{"error": "not_found"}, {"user_id": "U-11"}], {"user_id": "U-11"}),
        ("two_call_chains", "medium", "Tools available: geocode(place), get_weather(latitude, longitude). Find the weather at India Gate using its coordinates.", [{"tool": "geocode", "arguments": {"place": "India Gate"}}, {"tool": "get_weather", "arguments": {"latitude": 28.6129, "longitude": 77.2295}}], [{"latitude": 28.6129, "longitude": 77.2295}, {"temperature_c": 34}], {"place": "India Gate", "temperature_c": 34}),
        ("no_tool_needed_traps", "medium", "Tools available: send_email(to, body). Write a draft apology to sam@example.com, but the user has not approved sending it. Do not call send_email.", [], [], {"draft": "Sorry for the delay."}),
        ("using_one_call_result_in_the_next", "medium", "Tools available: get_exchange_rate(from_currency, to_currency), convert_amount(amount, rate). Convert 50 USD to INR using the live rate.", [{"tool": "get_exchange_rate", "arguments": {"from_currency": "USD", "to_currency": "INR"}}, {"tool": "convert_amount", "arguments": {"amount": 50, "rate": 83.2}}], [{"rate": 83.2}, {"converted_amount": 4160}], {"amount_inr": 4160}),
        ("unnecessary_call_avoidance", "medium", "Tools available: calculate_tax(amount, rate). The invoice states subtotal 1000 and tax 180. Report the total without recalculating tax.", [], [], {"total": 1180}),
        ("error_recovery", "medium", "Tools available: open_file(path), search_file(name). Open report.csv. First try /reports/report.csv; if missing, search by name and open the returned path.", [{"tool": "open_file", "arguments": {"path": "/reports/report.csv"}}, {"tool": "search_file", "arguments": {"name": "report.csv"}}, {"tool": "open_file", "arguments": {"path": "/archive/report.csv"}}], [{"error": "file_not_found"}, {"path": "/archive/report.csv"}, {"rows": 24}], {"path": "/archive/report.csv", "rows": 24}),
        ("two_call_chains", "medium", "Tools available: get_ticket(ticket_id), close_ticket(ticket_id, resolution). Close ticket T-90 only if its status is resolved; use resolution 'verified'.", [{"tool": "get_ticket", "arguments": {"ticket_id": "T-90"}}, {"tool": "close_ticket", "arguments": {"ticket_id": "T-90", "resolution": "verified"}}], [{"status": "resolved"}, {"closed": True}], {"ticket_id": "T-90", "closed": True}),
        ("using_one_call_result_in_the_next", "hard", "Tools available: get_order(order_id), validate_address(address), update_shipping_address(order_id, address). For order O-88, validate its proposed address '14 Lake Road, Pune', then update only if valid.", [{"tool": "get_order", "arguments": {"order_id": "O-88"}}, {"tool": "validate_address", "arguments": {"address": "14 Lake Road, Pune"}}, {"tool": "update_shipping_address", "arguments": {"order_id": "O-88", "address": "14 Lake Road, Pune"}}], [{"status": "processing"}, {"valid": True}, {"updated": True}], {"order_id": "O-88", "address_updated": True}),
        ("two_call_chains", "hard", "Tools available: find_slots(attendees, date), book_meeting(attendees, start). Find slots for Ana and Dev on 2026-08-04, then book the earliest returned slot.", [{"tool": "find_slots", "arguments": {"attendees": ["Ana", "Dev"], "date": "2026-08-04"}}, {"tool": "book_meeting", "arguments": {"attendees": ["Ana", "Dev"], "start": "2026-08-04T10:30:00+05:30"}}], [{"slots": ["2026-08-04T10:30:00+05:30", "2026-08-04T15:00:00+05:30"]}, {"meeting_id": "M-4"}], {"meeting_id": "M-4", "start": "2026-08-04T10:30:00+05:30"}),
        ("using_one_call_result_in_the_next", "hard", "Tools available: find_stock(sku), transfer_stock(sku, from_store, to_store, quantity). Move 6 units of SKU K9 to Store-B from the returned store with enough stock.", [{"tool": "find_stock", "arguments": {"sku": "K9"}}, {"tool": "transfer_stock", "arguments": {"sku": "K9", "from_store": "Store-A", "to_store": "Store-B", "quantity": 6}}], [{"locations": [{"store": "Store-A", "quantity": 9}, {"store": "Store-C", "quantity": 3}]}, {"transfer_id": "X-6"}], {"transfer_id": "X-6", "quantity": 6}),
        ("error_recovery", "hard", "Tools available: get_payment(payment_id), search_payments(email), refund_payment(payment_id, amount). Refund 500 for lee@example.com. Try old payment ID P-old first; if missing, search by email and refund the returned payment.", [{"tool": "get_payment", "arguments": {"payment_id": "P-old"}}, {"tool": "search_payments", "arguments": {"email": "lee@example.com"}}, {"tool": "refund_payment", "arguments": {"payment_id": "P-55", "amount": 500}}], [{"error": "not_found"}, {"payment_id": "P-55", "amount": 500}, {"refund_id": "R-55"}], {"refund_id": "R-55", "amount": 500}),
        ("unnecessary_call_avoidance", "hard", "Tools available: delete_project(project_id), archive_project(project_id). The user asks what archiving does and has not asked to change project P-9. Explain the state without calling either tool.", [], [], {"action_taken": False, "explanation": "Archiving preserves the project without deleting it."}),
    ]
    return [
        _base_item("tool_use", number, subcategory, difficulty, prompt, "json", {"calls": calls, "observations": observations, "final_state": final_state}, "tool_trace")
        for number, (subcategory, difficulty, prompt, calls, observations, final_state) in enumerate(specs, 1)
    ]


def _template(benchmark: str, method: str, contract: str) -> dict[str, Any]:
    return {
        "id": f"{benchmark}_replace_001",
        "subcategory": "replace_with_one_declared_task_type",
        "difficulty": "easy",
        "split": "dev",
        "visibility": "public",
        "prompt": "Replace with the complete question shown to the model.",
        "response_contract": {"type": contract, "format": None},
        "expected": {"value": "replace_with_gold_answer"},
        "scoring": {"method": method, "parameters": {}},
        "provenance": {"kind": "hand_authored", "review_status": "draft"},
        "tags": ["replace_tag"],
    }


def _write(benchmark: str, items: list[dict[str, Any]], template: dict[str, Any]) -> None:
    document = {
        "schema_version": 1,
        "benchmark": benchmark,
        "generated_by": GENERATOR,
        "seed": SEED,
        "item_template": template,
        "items": items,
    }
    path = ROOT / "data" / benchmark / "questions.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")


def main() -> None:
    _write("tables_to_decisions", _table_items(), _template("tables_to_decisions", "numeric_tolerance", "number"))
    _write("inbox_routing", _inbox_items(), _template("inbox_routing", "set_match", "text"))
    tool_template = _template("tool_use", "tool_trace", "json")
    tool_template["expected"]["value"] = {"calls": [], "observations": [], "final_state": {"replace_state": True}}
    _write("tool_use", _tool_items(), tool_template)


if __name__ == "__main__":
    main()
