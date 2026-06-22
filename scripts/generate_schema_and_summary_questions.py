from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


SCHEMA_OUTPUT = Path("data/messy_text_to_schema/questions.yaml")
SUMMARY_OUTPUT = Path("data/grounded_compression/questions.yaml")


def _schema_fixture(
    item_id: str,
    subcategory: str,
    difficulty: str,
    text: str,
    expected: dict[str, Any] | list[Any],
    *,
    note: str = "",
    tags: list[str] | None = None,
    checked: bool = False,
) -> dict[str, Any]:
    if isinstance(expected, dict):
        keys = ", ".join(expected)
        instruction = f"Extract {keys} from the text below."
        direct_json_example = '{"key":"value"}'
        boundary_instruction = "The first character must be { and the last must be }."
        format_name = "nested_object" if any(
            isinstance(value, (dict, list)) for value in expected.values()
        ) else "object"
    else:
        instruction = "Extract the records from the text below as a JSON array."
        direct_json_example = '[{"key":"value"}]'
        boundary_instruction = "The first character must be [ and the last must be ]."
        format_name = "array"
    prompt = (
        f"{instruction} Return valid JSON with exactly the requested structure and "
        "nothing else. Return the raw JSON directly, for example "
        f"{direct_json_example}. Do not wrap it in ``` or ```json Markdown fences. "
        f"{boundary_instruction} Do not add an explanation before or after it. "
        "Use JSON null for a requested value that is missing. Keep numbers and "
        "booleans as JSON values, not strings. "
    )
    if note:
        prompt += note.strip() + " "
    prompt += "Text: " + text.strip()
    return {
        "id": item_id,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "split": "dev",
        "prompt": prompt,
        "response_contract": {"type": "json", "format": format_name},
        "expected": {"value": expected},
        "scoring": {
            "method": "json_exact",
            "parameters": {"allow_diagnostic_normalization": True},
        },
        "provenance": {
            "kind": "hand_authored",
            "review_status": "human_checked" if checked else "draft",
        },
        "tags": ["extraction", subcategory, *(tags or [])],
    }


SCHEMA_ITEMS = [
    _schema_fixture(
        "schema_invoice_001", "clean_invoice", "easy",
        "Invoice INV-204 from Acme Paper. Total due: INR 4,250.00.",
        {"invoice_number": "INV-204", "vendor": "Acme Paper", "currency": "INR", "total": 4250.0},
        checked=True,
    ),
    _schema_fixture(
        "schema_contact_001", "clean_contact", "easy",
        "Account contact: Meera Nair | meera.nair@example.com | +91 98765 43210 | Pune.",
        {"name": "Meera Nair", "email": "meera.nair@example.com", "phone": "+91 98765 43210", "city": "Pune"},
    ),
    _schema_fixture(
        "schema_shipment_001", "clean_shipment", "easy",
        "Shipment SH-882 leaves Jaipur on 2026-09-03 and is expected in Kochi on 2026-09-07. Carrier: BlueDart.",
        {"shipment_id": "SH-882", "carrier": "BlueDart", "origin": "Jaipur", "destination": "Kochi", "expected_date": "2026-09-07"},
    ),
    _schema_fixture(
        "schema_booking_001", "clean_booking", "easy",
        "Booking BK-310: guest Arjun Rao, 2 rooms, check-in 14 October 2026, check-out 17 October 2026.",
        {"booking_id": "BK-310", "guest": "Arjun Rao", "rooms": 2, "check_in": "2026-10-14", "check_out": "2026-10-17"},
        note="Format both dates as YYYY-MM-DD.",
    ),
    _schema_fixture(
        "schema_product_001", "clean_product", "easy",
        "SKU KB-14, Compact Keyboard, category Accessories, price USD 49.95, in stock: yes.",
        {"sku": "KB-14", "name": "Compact Keyboard", "category": "Accessories", "price": 49.95, "in_stock": True},
    ),
    _schema_fixture(
        "schema_event_002", "clean_event", "easy",
        "Cloud Basics Workshop; 12 November 2026; Chennai; starts 09:30; capacity 80.",
        {"title": "Cloud Basics Workshop", "event_date": "2026-11-12", "city": "Chennai", "start_time": "09:30", "capacity": 80},
        note="Format event_date as YYYY-MM-DD and start_time as HH:MM.",
    ),
    _schema_fixture(
        "schema_employee_001", "clean_employee", "easy",
        "Employee E-091 is Sana Kapoor from the Data team. Joined 2023-06-12. Active: true.",
        {"employee_id": "E-091", "name": "Sana Kapoor", "team": "Data", "joined_on": "2023-06-12", "active": True},
    ),
    _schema_fixture(
        "schema_support_001", "clean_support", "easy",
        "Ticket T-551 was opened by Vikram Sen. Priority high. Topic: duplicate charge. Status: open.",
        {"ticket_id": "T-551", "customer": "Vikram Sen", "priority": "high", "topic": "duplicate charge", "status": "open"},
    ),
    _schema_fixture(
        "schema_event_001", "missing_fields", "medium",
        "Drafted on 4 August 2026. The Bengaluru Data Meetup will happen in Bengaluru's Indiranagar neighbourhood on 19 August 2026. Registration details will be shared later.",
        {"title": "Bengaluru Data Meetup", "event_date": "2026-08-19", "city": "Bengaluru", "contact_email": None},
        note="Format event_date as YYYY-MM-DD.", checked=True,
        tags=["distractor_date"],
    ),
    _schema_fixture(
        "schema_invoice_002", "distracting_numbers", "medium",
        "Reminder 2 of 3, sent 07/09/2026. Bill no. B-778 from North Star Office. Terms: 30 days. Subtotal INR 18,000; GST INR 3,240; amount payable INR 21,240.",
        {"invoice_number": "B-778", "vendor": "North Star Office", "currency": "INR", "subtotal": 18000.0, "tax": 3240.0, "total": 21240.0},
    ),
    _schema_fixture(
        "schema_subscription_001", "optional_fields", "medium",
        "Plan change for org ORG-42: Pro annual, 35 seats, renewal 2027-01-15. Coupon field left blank. Auto-renew is OFF.",
        {"organization_id": "ORG-42", "plan": "Pro annual", "seats": 35, "renewal_date": "2027-01-15", "coupon": None, "auto_renew": False},
    ),
    _schema_fixture(
        "schema_candidate_001", "mixed_format", "medium",
        "CANDIDATE / C-18\nname=Leena D'Souza\nrole :: Backend Engineer\nnotice period: forty-five days\nexpected CTC INR 24.5 lakh\nremote? YES",
        {"candidate_id": "C-18", "name": "Leena D'Souza", "role": "Backend Engineer", "notice_days": 45, "expected_ctc_inr": 2450000, "remote": True},
    ),
    _schema_fixture(
        "schema_delivery_001", "ocr_noise", "medium",
        "DELlVERY D-909 | fr0m: Surat | t0: Nagpur | 6 cart0ns | wt 42.5 kg | fragile: N0 | ETA 03-12-2026",
        {"delivery_id": "D-909", "origin": "Surat", "destination": "Nagpur", "cartons": 6, "weight_kg": 42.5, "fragile": False, "eta": "2026-12-03"},
        note="The date is DD-MM-YYYY; return eta as YYYY-MM-DD.",
    ),
    _schema_fixture(
        "schema_meeting_001", "missing_fields", "medium",
        "Minutes header: Project Cedar sync, Friday 6 March 2026, 15:00 IST. Owner: Omar Ali. Room changed twice; final location is Zoom. No recording link was added.",
        {"meeting": "Project Cedar sync", "date": "2026-03-06", "time": "15:00", "timezone": "IST", "owner": "Omar Ali", "location": "Zoom", "recording_url": None},
    ),
    _schema_fixture(
        "schema_refund_001", "status_history", "medium",
        "Refund RF-650 for order O-188. Requested 2 May; approved 4 May; bank transfer failed 5 May; retried 7 May. Current state: processing. Amount ₹1,899.00.",
        {"refund_id": "RF-650", "order_id": "O-188", "amount": 1899.0, "currency": "INR", "current_status": "processing", "last_action_date": "2026-05-07"},
        note="All dates are in 2026; format last_action_date as YYYY-MM-DD.",
    ),
    _schema_fixture(
        "schema_address_001", "multiline_address", "medium",
        "Ship to: Riya Menon\nFlat 8B, Lake View Towers\n17 M.G. Road\nErnakulam, Kerala 682016\nCountry: India\nLandmark: —",
        {"recipient": "Riya Menon", "line1": "Flat 8B, Lake View Towers", "line2": "17 M.G. Road", "city": "Ernakulam", "state": "Kerala", "postal_code": "682016", "country": "India", "landmark": None},
    ),
    _schema_fixture(
        "schema_asset_001", "conflicting_values", "medium",
        "Asset register correction: laptop LT-44 was first entered as assigned to E-120. That entry is wrong. Final owner E-128, serial SN9K22, purchased 2025-11-09, warranty expires 2028-11-08.",
        {"asset_id": "LT-44", "employee_id": "E-128", "serial_number": "SN9K22", "purchase_date": "2025-11-09", "warranty_end": "2028-11-08"},
    ),
    _schema_fixture(
        "schema_payment_001", "redacted_fields", "medium",
        "Payment P-771 | payer: Bright Foods Pvt Ltd | method: card ending 4412 | amount USD 860.50 | auth code [REDACTED] | settled yes.",
        {"payment_id": "P-771", "payer": "Bright Foods Pvt Ltd", "method": "card", "last_four": "4412", "amount": 860.5, "currency": "USD", "authorization_code": None, "settled": True},
    ),
    _schema_fixture(
        "schema_release_001", "version_noise", "medium",
        "Release notes drafted with Python 3.12. Build r2026.08.17 ships app version 4.7.2 to Android only, staged at 25%, owner Mobile Platform, rollback version 4.7.1.",
        {"build_id": "r2026.08.17", "app_version": "4.7.2", "platform": "Android", "rollout_percent": 25, "owner": "Mobile Platform", "rollback_version": "4.7.1"},
    ),
    _schema_fixture(
        "schema_course_001", "list_field", "medium",
        "Course DS-210: Practical Analytics. Instructor: Nisha Rao. Runs 8 weeks. Prerequisites are Python and basic statistics. SQL is recommended, not required.",
        {"course_id": "DS-210", "title": "Practical Analytics", "instructor": "Nisha Rao", "duration_weeks": 8, "prerequisites": ["Python", "basic statistics"]},
        note="Keep prerequisites in the order stated and exclude recommendations.",
    ),
    _schema_fixture(
        "schema_incident_001", "timeline", "medium",
        "INC-73 opened 09:12 after API errors reached 14%. Mitigation began 09:24. Errors returned below 1% at 09:41. Resolved 10:05. Severity SEV-2; lead Asha Iyer; cause still unknown.",
        {"incident_id": "INC-73", "severity": "SEV-2", "lead": "Asha Iyer", "opened_at": "09:12", "mitigated_at": "09:41", "resolved_at": "10:05", "cause": None},
    ),
    _schema_fixture(
        "schema_feedback_001", "quoted_text", "medium",
        'Response FB-19 from Maya Shah gave 8/10 and said, "Fast setup, but export labels are confusing." Follow-up allowed: no.',
        {"response_id": "FB-19", "customer": "Maya Shah", "rating": 8, "comment": "Fast setup, but export labels are confusing.", "follow_up_allowed": False},
    ),
    _schema_fixture(
        "schema_inventory_001", "multiple_records", "medium",
        "Warehouse W2 count: A-10 / adapters / 18 units; C-07 / cables / 0 units; H-22 / hubs / 6 units. Counted 2026-07-31.",
        {"warehouse": "W2", "counted_on": "2026-07-31", "items": [{"sku": "A-10", "name": "adapters", "quantity": 18}, {"sku": "C-07", "name": "cables", "quantity": 0}, {"sku": "H-22", "name": "hubs", "quantity": 6}]},
        note="Each items entry must contain exactly sku, name, and quantity, in source order.",
    ),
    _schema_fixture(
        "schema_order_001", "nested_ocr_order", "hard",
        "Ord3r K-77 / curr: INR / 2 x USB-C Hub @ 1,250.00 / 3 x Cable 2m @ 300.00 / GST: 612.00 / GRAND T0TAL 4,012.00.",
        {"order_id": "K-77", "currency": "INR", "line_items": [{"description": "USB-C Hub", "quantity": 2, "unit_price": 1250.0}, {"description": "Cable 2m", "quantity": 3, "unit_price": 300.0}], "tax": 612.0, "total": 4012.0},
        note="Each line_items entry must contain exactly description, quantity, and unit_price. The OCR-like word before the identifier is only a label.", checked=True,
    ),
    _schema_fixture(
        "schema_expense_001", "nested_expenses", "hard",
        "EXP R-18 / owner Neha Jain / trip BLR→DEL / 11-13 Sep 2026. Lines: 11/09 flight INR 8,450 approved; 12/09 taxi INR 780 approved; 12/09 minibar INR 460 rejected. Claimed 9,690; approved total 9,230.",
        {"report_id": "R-18", "owner": "Neha Jain", "route": {"from": "BLR", "to": "DEL"}, "items": [{"date": "2026-09-11", "category": "flight", "amount": 8450.0, "approved": True}, {"date": "2026-09-12", "category": "taxi", "amount": 780.0, "approved": True}, {"date": "2026-09-12", "category": "minibar", "amount": 460.0, "approved": False}], "approved_total": 9230.0},
        note="Each items entry must contain exactly date, category, amount, and approved. Format dates as YYYY-MM-DD.",
    ),
    _schema_fixture(
        "schema_manifest_001", "ocr_table", "hard",
        "MANlFEST M-55; route CCU-BOM; dep 22:10 04/12/26. PKGS: P01|12.4kg|fragile Y; P02|8kg|fragile N; P03|15.75kg|fragile Y. Note: 3 pieces, total shown 36.15kg.",
        {"manifest_id": "M-55", "origin": "CCU", "destination": "BOM", "departure": "2026-12-04T22:10", "packages": [{"id": "P01", "weight_kg": 12.4, "fragile": True}, {"id": "P02", "weight_kg": 8.0, "fragile": False}, {"id": "P03", "weight_kg": 15.75, "fragile": True}], "total_weight_kg": 36.15},
        note="The date is DD/MM/YY. Each packages entry must contain exactly id, weight_kg, and fragile.",
    ),
    _schema_fixture(
        "schema_purchase_order_001", "revisions", "hard",
        "PO PO-902 rev1 (ignore): 40 chairs at 3,000. Revision 2 FINAL: vendor Cedar Works; 36 chairs at INR 2,850 each and 6 tables at INR 7,200 each. Delivery 18 Jan 2027. Freight 4,500. GST 18%. Pre-tax total including freight 150,300.",
        {"purchase_order": "PO-902", "revision": 2, "vendor": "Cedar Works", "items": [{"name": "chair", "quantity": 36, "unit_price": 2850.0}, {"name": "table", "quantity": 6, "unit_price": 7200.0}], "freight": 4500.0, "tax_rate": 0.18, "pre_tax_total": 150300.0, "delivery_date": "2027-01-18"},
        note="Use only the final revision. Each items entry must contain exactly name, quantity, and unit_price.",
    ),
    _schema_fixture(
        "schema_project_001", "nested_status", "hard",
        "Project Atlas, owner Tara Bose. Old target 30 June is obsolete; revised target 18 July 2026. Work: API=done (Dev Patel); migration=in progress, 70% (Isha Rao); docs=blocked, 40% (Omar Ali). Blocker: legal review, expected 10 July. Overall risk amber.",
        {"project": "Atlas", "owner": "Tara Bose", "target_date": "2026-07-18", "risk": "amber", "workstreams": [{"name": "API", "status": "done", "percent": 100, "owner": "Dev Patel"}, {"name": "migration", "status": "in progress", "percent": 70, "owner": "Isha Rao"}, {"name": "docs", "status": "blocked", "percent": 40, "owner": "Omar Ali"}], "blocker": {"name": "legal review", "expected_date": "2026-07-10"}},
        note="Each workstreams entry must contain exactly name, status, percent, and owner.",
    ),
    _schema_fixture(
        "schema_sensor_001", "mixed_units", "hard",
        "Station S7 log: 08:00 temp 86°F humidity 61%; 12:00 temp 32°C humidity 48%; 16:00 temp 303.15K humidity sensor ERROR. Convert all temperatures to Celsius. Device remained online. Battery 37%.",
        {"station_id": "S7", "readings": [{"time": "08:00", "temperature_c": 30.0, "humidity_percent": 61}, {"time": "12:00", "temperature_c": 32.0, "humidity_percent": 48}, {"time": "16:00", "temperature_c": 30.0, "humidity_percent": None}], "online": True, "battery_percent": 37},
        note="Each readings entry must contain exactly time, temperature_c, and humidity_percent. Convert Fahrenheit and Kelvin exactly as implied by the supplied values.",
    ),
    _schema_fixture(
        "schema_batch_001", "multiple_noisy_records", "hard",
        "Batch export: [u-1] Asha / asha@example.com / admin / enabled; [u-2] Rohan / email MISSING / viewer / disabled; duplicate old row [u-1] role viewer (ignore); [u-3] Mei / mei@example.com / editor / enabled.",
        [{"user_id": "u-1", "name": "Asha", "email": "asha@example.com", "role": "admin", "enabled": True}, {"user_id": "u-2", "name": "Rohan", "email": None, "role": "viewer", "enabled": False}, {"user_id": "u-3", "name": "Mei", "email": "mei@example.com", "role": "editor", "enabled": True}],
        note="Each array entry must contain exactly user_id, name, email, role, and enabled. Keep the first current u-1 row and ignore the labelled old duplicate.",
    ),
]


def _summary_fixture(
    item_id: str,
    subcategory: str,
    difficulty: str,
    task: str,
    source: str,
    expected: str,
    max_words: int,
    *,
    checked: bool = False,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "split": "dev",
        "prompt": f"{task.strip()}\n\nSource: {source.strip()}",
        "response_contract": {"type": "text", "format": "plain_text"},
        "expected": {"value": expected.strip()},
        "scoring": {
            "method": "llm_judge",
            "parameters": {
                "rubric": "grounded_summary_v1",
                "pass_threshold": 0.7,
                "minimum_faithfulness": 3,
                "max_words": max_words,
            },
        },
        "provenance": {
            "kind": "hand_authored",
            "review_status": "human_checked" if checked else "draft",
        },
        "tags": ["summarization", "groundedness", subcategory, "pointwise_judge"],
    }


SUMMARY_ITEMS = [
    _summary_fixture(
        "summary_project_001", "project_update", "easy",
        "Summarize this update for the project lead in no more than 45 words. Include progress, the blocker, and the next action.",
        "The website refresh has completed design and frontend work. Content migration is 80% complete. Legal approval for the new privacy text is blocking launch. Priya will send the revised wording to legal today, and the team will review approval status on Friday. The launch date has not changed yet.",
        "Design and frontend work are complete, while content migration is 80% done. Legal approval of the privacy text is blocking launch. Priya will send revised wording today, and the team will check approval on Friday. The launch date is unchanged for now.", 45,
    ),
    _summary_fixture(
        "summary_delivery_001", "customer_update", "easy",
        "Write a customer-facing summary in no more than 40 words. State the delay, new date, and what the customer needs to do.",
        "Order 771 was due on Tuesday. Heavy rain closed the regional sorting centre for one day, so the carrier now expects delivery on Thursday. Tracking will update automatically when the parcel leaves the centre. The address is correct and the customer does not need to contact support or place a new order.",
        "Order 771 is delayed from Tuesday to Thursday because heavy rain closed the sorting centre. Tracking will update automatically when it leaves the centre. Your address is correct, and you do not need to contact support or reorder.", 40,
    ),
    _summary_fixture(
        "summary_policy_001", "policy_notice", "easy",
        "Summarize the policy change for employees in no more than 45 words. Focus on what changes and when.",
        "Starting 1 October, employees may work remotely for up to three days each week instead of two. Team leads may set common office days with two weeks' notice. Existing security and working-hour rules remain unchanged. Employees do not need to submit a new remote-work agreement.",
        "From 1 October, employees may work remotely up to three days a week. Team leads can set common office days with two weeks' notice. Security and working-hour rules stay the same, and no new remote-work agreement is required.", 45,
    ),
    _summary_fixture(
        "summary_event_001", "event_recap", "easy",
        "Summarize the event outcome for sponsors in no more than 45 words. Include attendance, feedback, and follow-up.",
        "The data workshop registered 120 people and 96 attended. Eighty-two attendees completed the survey; 76 rated the workshop good or excellent. The most common request was more hands-on SQL time. Recordings and exercises will be emailed on Monday, and the team will consider a longer lab for the next workshop.",
        "The workshop drew 96 of 120 registrants. Among 82 survey respondents, 76 rated it good or excellent, with more hands-on SQL time the top request. Recordings and exercises go out Monday, and a longer future lab is under consideration.", 45,
    ),
    _summary_fixture(
        "summary_product_001", "release_note", "easy",
        "Summarize this product change for users in no more than 40 words. Include the benefit and any limitation.",
        "Version 3.4 adds scheduled CSV exports. Users can choose daily or weekly delivery to one verified email address. Exports use the account's current filters. The feature is available on Pro and Enterprise plans. Password-protected files and delivery to multiple addresses are not supported in this release.",
        "Version 3.4 lets Pro and Enterprise users schedule daily or weekly filtered CSV exports to one verified email address. This release does not support password-protected files or delivery to multiple addresses.", 40,
    ),
    _summary_fixture(
        "summary_incident_001", "operational_update", "medium",
        "Summarize the operational update for an engineering manager in no more than 60 words. Prioritize customer impact, current status, and the next decision. Clearly distinguish confirmed facts from the suspected cause.",
        "At 10:15 IST, monitoring showed elevated checkout latency shortly after the recommendation-service release that began at 10:05. Eighteen percent of checkout requests took longer than two seconds, compared with a normal rate below three percent. The service returned no additional errors, no payments were duplicated, and no customer data was lost. Engineers rolled back the release at 10:32, and latency returned to baseline by 10:41. Eleven customers contacted support about slow checkout confirmation. A cache warm-up configuration is the leading suspected cause, but this has not been confirmed. The infrastructure team is reviewing traces and will decide by 16:00 whether the release can be retried the following day. Customers do not need to take any action.",
        "Checkout latency rose after the recommendation-service release, with 18% of requests exceeding two seconds but no additional errors, duplicate payments, or data loss. Rollback restored baseline performance by 10:41. A cache warm-up configuration is suspected but unconfirmed; infrastructure will review traces and decide by 16:00 whether to retry tomorrow.", 60, checked=True,
    ),
    _summary_fixture(
        "summary_experiment_001", "experiment_result", "medium",
        "Summarize the experiment for a product manager in no more than 60 words. State the result, important limits, and recommended decision.",
        "A two-week checkout experiment showed the shorter form to 20,400 visitors and the current form to 20,180. Purchase completion was 4.8% on the shorter form and 4.5% on the current form. The 0.3 percentage-point lift was statistically significant at the team's preset 95% threshold. Mobile users showed most of the gain; desktop results were flat. Refund rate and support contacts did not change. The test excluded returning subscribers and ran only in India. The analytics team recommends rolling out to new Indian customers while running a separate test for subscribers and other countries.",
        "The shorter checkout raised completion from 4.5% to 4.8%, meeting the preset significance threshold, with gains concentrated on mobile and no change in refunds or support contacts. Because the India-only test excluded returning subscribers, roll it out to new Indian customers while testing other users separately.", 60,
    ),
    _summary_fixture(
        "summary_budget_001", "decision_brief", "medium",
        "Summarize this request for the finance approver in no more than 65 words. Include cost, expected benefit, uncertainty, and the decision deadline.",
        "The support team requests INR 9.6 lakh for a one-year knowledge-base tool covering 40 agents. The vendor estimates a 15% reduction in average handling time, but this estimate comes from its own customer study and has not been tested internally. A four-week trial with eight agents reduced handling time by 9% and did not change satisfaction scores. Integration would require about six engineering days. The discounted quote expires 30 September. Finance must decide by 25 September to leave time for security review.",
        "Support requests INR 9.6 lakh for one year. An eight-agent trial cut handling time 9% with no satisfaction change, below the vendor's unverified 15% estimate; integration needs about six engineering days. Finance should decide by 25 September, ahead of security review and the 30 September quote expiry.", 65,
    ),
    _summary_fixture(
        "summary_research_001", "customer_research", "medium",
        "Summarize the customer research for the design team in no more than 65 words. Separate common findings from minority views.",
        "Researchers interviewed 18 small-business administrators who prepare monthly reports. Fourteen said finding the right filters takes too long, and 12 save screenshots because exported files lose the visible chart labels. Ten wanted reusable report templates. Five asked for more chart colours, while three wanted AI-written commentary. Two participants said the current workflow was already fast enough. The sample included existing customers who use reports at least monthly; it did not include trial users or large enterprises.",
        "Most interviewed administrators struggled with filters (14 of 18), lost chart labels in exports (12), and wanted reusable templates (10). Smaller groups requested more colours (5) or AI commentary (3), while two found the workflow fast enough. Findings apply to frequent small-business users, not trial users or enterprises.", 65,
    ),
    _summary_fixture(
        "summary_migration_001", "migration_plan", "medium",
        "Summarize the migration status for leadership in no more than 65 words. Include progress, risk, fallback, and the next checkpoint.",
        "The team has moved 62 of 80 services to the new secrets platform. Twelve of the remaining services are scheduled this week. Six older services cannot use the standard client library and may need custom work. No production incidents have been linked to completed migrations. The old platform contract ends 31 December. If the six older services are not ready by 15 November, the fallback is a three-month contract extension estimated at USD 18,000. Architecture will review the custom approach on 2 October.",
        "Sixty-two of 80 services have migrated without linked production incidents; 12 more are scheduled this week. Six legacy services may require custom work, risking the 31 December deadline. Architecture reviews the approach on 2 October. If they are not ready by 15 November, the fallback is an estimated USD 18,000 three-month extension.", 65,
    ),
    _summary_fixture(
        "summary_hiring_001", "hiring_update", "medium",
        "Summarize the hiring update for the department head in no more than 55 words. Focus on funnel health, the bottleneck, and the requested action.",
        "For four backend openings, recruiting screened 86 applicants, advanced 24 to technical interviews, and produced seven final-round candidates. Two offers were accepted, one was declined for compensation, and four final decisions are pending. Median scheduling time for technical interviews increased from four to nine days because only three trained interviewers are available. Recruiting asks two senior engineers to complete interviewer training this month. The approved salary bands have not changed.",
        "Two of four backend roles are filled, with four final decisions pending after 86 screens and 24 technical interviews. Technical-interview scheduling has slowed from four to nine days because only three interviewers are trained. Recruiting requests two senior engineers complete training this month; salary bands remain unchanged.", 55,
    ),
    _summary_fixture(
        "summary_vendor_001", "contract_review", "medium",
        "Summarize the vendor renewal for procurement in no more than 65 words. Include the price change, service record, open issue, and negotiation position.",
        "The monitoring vendor proposes a two-year renewal at USD 132,000 per year, up 10% from the current USD 120,000. It met the 99.9% availability commitment in every measured month and resolved priority-one tickets within the contracted time. Usage grew 28%. However, the promised regional data-storage option is six months late and now expected in February. Engineering considers the tool hard to replace before the November renewal date. Procurement recommends accepting a two-year term only if year-one pricing stays at USD 120,000 or the vendor adds a service credit for the delayed feature.",
        "The vendor seeks a 10% increase to USD 132,000 yearly despite meeting availability and support commitments as usage grew 28%. Its regional storage feature is six months late. Because replacement before November is difficult, procurement should accept two years only with USD 120,000 year-one pricing or a delay-related service credit.", 65,
    ),
    _summary_fixture(
        "summary_security_001", "risk_review", "medium",
        "Summarize the security review for the launch owner in no more than 60 words. State what is cleared, what remains, and whether launch is blocked.",
        "Security completed its review of the analytics integration. Encryption, access controls, and deletion behaviour passed. The vendor has not yet supplied its latest penetration-test report; the previous report expired in June. The integration processes usage metadata but not message content or payment data. Security rates the missing report as medium risk and permits an internal beta capped at 50 employees. Public launch remains blocked until the current report is reviewed. The vendor says it will deliver the report by 8 August.",
        "Encryption, access controls, and deletion checks passed, and the integration handles usage metadata rather than message or payment data. Security allows an internal beta for up to 50 employees. Public launch remains blocked by the missing current penetration-test report, rated medium risk and promised by 8 August.", 60,
    ),
    _summary_fixture(
        "summary_capacity_001", "operations_forecast", "medium",
        "Summarize the capacity forecast for operations in no more than 60 words. Include expected demand, current headroom, and the trigger for action.",
        "Traffic is forecast to rise 35% during the festival campaign. At last week's peak, the API used 58% of available compute. Load tests suggest the current setup can handle roughly 50% more traffic before p95 latency exceeds the 800 ms target, although the test did not include the new image feature. Adding two compute nodes would cost INR 1.2 lakh for the month and take about three hours. Operations proposes monitoring daily and adding nodes if peak compute exceeds 75% for two consecutive hours.",
        "Festival traffic is expected to rise 35%. Current peak usage is 58%, and tests show about 50% headroom before latency misses target, but they excluded the new image feature. Operations will monitor daily and add two nodes for INR 1.2 lakh if peak compute stays above 75% for two hours.", 60,
    ),
    _summary_fixture(
        "summary_training_001", "program_result", "medium",
        "Summarize the training pilot for the learning team in no more than 60 words. Include completion, measured result, limits, and next step.",
        "Sixty employees enrolled in the secure-coding pilot and 51 completed all modules. Average quiz scores rose from 68% before training to 84% after training. Thirty participants submitted code samples; high-severity findings fell from 11 before training to 5 afterward. The comparison was not controlled, and participants volunteered, so the results may not represent all engineers. The learning team proposes repeating the pilot with two full product teams and measuring findings for three months.",
        "Fifty-one of 60 employees completed the pilot. Quiz scores rose from 68% to 84%, while high-severity findings in 30 submitted samples fell from 11 to 5. Because participation was voluntary and uncontrolled, the learning team proposes a three-month repeat with two complete product teams.", 60,
    ),
    _summary_fixture(
        "summary_outage_001", "uncertain_incident", "hard",
        "Write an executive incident brief in no more than 75 words. Preserve the timeline, business impact, uncertainty, and remaining action. Do not state a cause as confirmed.",
        "Between 18:06 and 18:43 UTC, customers in Europe intermittently failed to upload files larger than 20 MB. Smaller uploads and customers in other regions were unaffected. Logs show 6,240 failed attempts from 1,870 accounts; retries succeeded for about 61% of affected attempts. No files were corrupted, and stored files remained available. Engineers disabled a new traffic-routing rule at 18:31, after which failures declined and stopped by 18:43. The rule is a possible contributor, but two failures also appeared in logs from before its 17:50 deployment. A network provider reported brief packet loss in Frankfurt but has not confirmed timing. Support received 94 contacts. Engineering will compare packet traces and run a controlled replay before deciding whether to re-enable the rule.",
        "From 18:06–18:43 UTC, large-file uploads intermittently failed for European customers: 6,240 attempts across 1,870 accounts, with 61% succeeding on retry. Smaller uploads, other regions, and stored files were unaffected; no corruption occurred. Failures stopped after a routing rule was disabled, but earlier failures and possible Frankfurt packet loss leave the cause unconfirmed. Engineering will analyse traces and replay traffic before re-enabling it.", 75,
    ),
    _summary_fixture(
        "summary_proposal_001", "tradeoff_brief", "hard",
        "Summarize the proposal for the steering group in no more than 80 words. Compare both options, include the major assumptions, and identify the decision needed.",
        "The team must choose how to archive seven years of audit logs. Option A keeps the current vendor and moves older logs to its cold tier. It costs an estimated USD 74,000 annually, needs about three weeks of engineering work, and restores archived data in four to eight hours. Option B moves archives to object storage, costs an estimated USD 39,000 annually, and takes eight to twelve weeks to build. Its restore estimate is two to six hours, but this has only been tested on one month of data. Compliance requires retrieval within 12 hours and has approved either design. Option A's quote is fixed through December; Option B's estimate excludes ongoing maintenance. The team needs a decision by 15 October to complete either path before the current retention exception ends in January.",
        "Option A costs about USD 74,000 yearly, needs three weeks, and restores in four to eight hours. Option B is estimated at USD 39,000 plus unpriced maintenance, needs eight to twelve weeks, and has a two-to-six-hour restore estimate tested on only one month. Both meet the 12-hour compliance limit. The steering group must choose by 15 October to finish before January.", 80,
    ),
    _summary_fixture(
        "summary_study_001", "conflicting_evidence", "hard",
        "Summarize the evidence for a non-technical decision maker in no more than 80 words. Explain where the studies agree, where they differ, and why no firm conclusion is possible.",
        "Three internal analyses examined whether optional focus hours improve delivery speed. The first compared six teams before and after adoption and found cycle time fell 12%, but those teams also hired more senior engineers. The second compared eight adopting teams with eight non-adopting teams and found no meaningful cycle-time difference, although policy use varied widely. The third surveyed 420 employees: 71% reported fewer interruptions, while 38% said meeting availability became harder. All three found no clear change in defect rate. None randomly assigned teams, and the analyses covered only one quarter. Leadership is considering a six-month controlled pilot rather than a company-wide policy.",
        "The analyses agree that focus hours showed no clear defect-rate change. One found 12% faster cycles but was confounded by senior hiring; a matched comparison found no meaningful speed difference amid uneven policy use. Employees mostly reported fewer interruptions, though 38% found meetings harder. Non-random assignment and one-quarter duration prevent a firm conclusion, supporting a six-month controlled pilot.", 80,
    ),
    _summary_fixture(
        "summary_launch_001", "multi_stakeholder", "hard",
        "Write a launch-readiness summary in no more than 80 words for product, support, and engineering leads. Include ready areas, blockers, workarounds, and the decision point.",
        "The new billing portal passed functional and load testing, and finance validated invoice totals against 2,000 historical accounts. Translation is complete for English, Hindi, and Spanish, but German legal text is awaiting approval. Support has trained 42 of 50 agents; the remaining eight work the weekend shift and train Friday. A known issue causes saved bank details to display slowly for about 3% of users, though payments still complete; engineering has a fix in review and can disable saved details as a workaround. Marketing has scheduled email for Monday at 09:00. The launch group meets Friday at 16:00 to decide whether to launch Monday, limit launch to approved languages, or delay.",
        "Functional, load, and invoice checks passed; three languages are ready, and 42 of 50 support agents are trained. German legal approval and Friday training remain open. About 3% of users see slow saved-bank-detail display, but payments work; a fix is under review and disabling saved details is the fallback. Friday's 16:00 meeting must choose Monday launch, approved-language-only launch, or delay.", 80,
    ),
    _summary_fixture(
        "summary_audit_001", "compliance_update", "hard",
        "Summarize the audit findings for the risk committee in no more than 80 words. Separate verified failures, scope limits, immediate controls, and unresolved exposure.",
        "An access audit reviewed 14 production systems and sampled 320 active accounts. It verified that 17 accounts belonging to former contractors remained enabled between 4 and 29 days after contract end; logs show no use of those accounts after departure. Nine of the 17 had read access to customer metadata, while none could view payment details. All 17 accounts were disabled during the audit. The review did not cover two legacy systems because their account exports failed, so their exposure is unknown. Human Resources and IT records disagreed on 23 contractor end dates. Security has added a daily comparison for the reviewed systems and will complete manual checks of the legacy systems by Tuesday.",
        "The audit confirmed 17 former-contractor accounts stayed active for 4–29 days; none were used after departure. Nine could read customer metadata, but none had payment access. All are disabled, and daily HR–IT comparisons now cover reviewed systems. Exposure remains unknown for two legacy systems whose exports failed, while 23 end-date mismatches remain. Manual legacy checks are due Tuesday.", 80,
    ),
]


def _write(path: Path, benchmark: str, items: list[dict[str, Any]]) -> None:
    document = {
        "schema_version": 1,
        "benchmark": benchmark,
        "generated_by": "schema_and_summary_v1",
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-output", type=Path, default=SCHEMA_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()
    _write(args.schema_output, "messy_text_to_schema", SCHEMA_ITEMS)
    _write(args.summary_output, "grounded_compression", SUMMARY_ITEMS)


if __name__ == "__main__":
    main()
