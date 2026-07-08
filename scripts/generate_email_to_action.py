from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "email_to_action" / "questions.yaml"
GENERATOR = "email_to_action_v1"
SEED = 20260803

LABEL_ORDER = (
    "to_respond",
    "fyi_notification",
    "marketing",
    "spam",
    "suspicious",
    "job_search",
)

ENRON_POINTER = {
    "dataset": "Enron corpus (style pointer; fully rewritten)",
    "url": "https://www.cs.cmu.edu/~enron/",
    "license": "Public research release; no explicit license",
}
LEGACY_POINTER = {
    "dataset": "Legacy inbox_routing scenarios (fully rewritten as complete emails)",
    "url": "git-history:data/inbox_routing/questions.yaml",
    "license": "Repository-authored synthetic material",
}


@dataclass(frozen=True)
class Field:
    key: str
    evidence: str | None
    value: Any


@dataclass(frozen=True)
class EmailSpec:
    mix: str
    difficulty: str
    sender: str
    recipient: str
    subject: str
    date: str
    greeting: str
    body: str
    signature: str
    categories: tuple[str, ...]
    action_required: bool
    urgency: str
    fields: tuple[Field, ...]
    tags: tuple[str, ...]
    pointer: dict[str, str] | None = None


def _f(key: str, evidence: str | None, value: Any) -> Field:
    return Field(key, evidence, value)


SPECS: tuple[EmailSpec, ...] = (
    EmailSpec(
        "single_label_clear", "easy", "receipts@paysetu.in", "meera.nair@example.net",
        "Payment received for SetuCloud", "Mon, 03 Aug 2026 09:14:22 +0530", "Hello Meera,",
        "We received your UPI payment of INR 1,487.32 for the SetuCloud Plus renewal. The transaction was posted at 09:13 IST and no balance remains on this invoice. Receipt RCPT-8K41 is attached as a PDF for your records. If your bank app still shows the payment as pending, it may take a few hours to refresh; please do not pay again.",
        "Thanks,\nSetuCloud Billing\nThis is an automated receipt.", ("fyi",), False, "normal",
        (_f("amount", "INR 1,487.32", 1487.32), _f("receipt_id", "RCPT-8K41", "RCPT-8K41"), _f("paid_at", "09:13 IST", "2026-08-03T09:13:00+05:30")),
        ("payment_receipt", "automated_notification", "indian_context"),
    ),
    EmailSpec(
        "single_label_clear", "easy", "rhea.kapoor@northstar-ops.in", "data-platform@northstar-ops.in",
        "Friday handover notes", "Fri, 31 Jul 2026 18:42:07 +0530", "Hi all,",
        "Quick handover before I log off. The Kochi ingestion backfill finished at 17:55, the retry queue is empty, and the dashboard is green. I moved the two noisy vendor feeds to Monday's watch list and added screenshots to the incident folder. Nothing needs changing tonight; this is just so the weekend rota has the same context. The next scheduled load begins Monday at 06:30 IST.",
        "cheers,\nRhea\nData Platform", ("fyi",), False, "normal",
        (_f("sender_name", "Rhea", "Rhea"), _f("next_load", "Monday at 06:30 IST", "2026-08-03T06:30:00+05:30"), _f("location", "Kochi", "Kochi")),
        ("internal_update", "enron_style"), ENRON_POINTER,
    ),
    EmailSpec(
        "single_label_clear", "easy", "letters@slowweekend.co", "ananya@example.org",
        "Three monsoon reads for a quiet Sunday", "Thu, 30 Jul 2026 07:10:00 +0530", "Morning Ananya,",
        "This week's Slow Weekend letter has three essays on repairing old homes during the rains, a photo walk through Panjim, and our interview with a second-generation bookbinder. Members can also download the August reading calendar. You are receiving this because you subscribed on 12 June. Prefer fewer emails? Use the monthly option in your settings, or unsubscribe at https://slowweekend.co/u/4m2q.",
        "Warmly,\nLeena from Slow Weekend", ("marketing",), False, "low",
        (_f("newsletter_name", "Slow Weekend", "Slow Weekend"), _f("unsubscribe_url", "https://slowweekend.co/u/4m2q", "https://slowweekend.co/u/4m2q")),
        ("newsletter", "unsubscribe_footer", "html_to_text_artifact"),
    ),
    EmailSpec(
        "single_label_clear", "easy", "offers@bulkdesk-mail.biz", "undisclosed-recipients:;",
        "LOW COST TONER STOCK - AUGUST LIST", "Wed, 29 Jul 2026 02:18:41 +0000", "Dear office purchaser,",
        "We are clearing mixed printer toner cartons from our warehouse and can ship across India. Reply with the word CATALOG if you want the current list. There is no account relationship and no order has been placed for you. Minimum carton quantities apply, models may be substituted, and the sender does not guarantee compatibility with your printer. Promotional code BULK17 expires whenever remaining stock is sold.",
        "Regards,\nN. Batra\nBulkDesk Surplus Mailing", ("spam",), False, "low",
        (_f("sender_name", "N. Batra", "N. Batra"), _f("offer_code", "BULK17", "BULK17")),
        ("unwanted_bulk", "harmless_spam"),
    ),
    EmailSpec(
        "single_label_clear", "easy", "talent@riverstoneanalytics.com", "kabir.s@example.com",
        "We received your application", "Tue, 28 Jul 2026 16:26:09 +0530", "Hi Kabir,",
        "Thanks for applying for the Data Quality Analyst role at Riverstone Analytics. Your application is in our review queue under reference APP-29418. The hiring team usually completes the first review within seven business days. There is nothing else you need to send right now; if we need work samples, a recruiter will contact you from an @riverstoneanalytics.com address. You can check status in the candidate portal at any time.",
        "Riverstone Talent Team\nPlease do not reply to this automated acknowledgement.", ("job",), False, "normal",
        (_f("company", "Riverstone Analytics", "Riverstone Analytics"), _f("role", "Data Quality Analyst", "Data Quality Analyst"), _f("application_id", "APP-29418", "APP-29418")),
        ("application_confirmation", "job_search", "automated_notification"),
    ),
    EmailSpec(
        "single_label_clear", "medium", "office@lakeviewpublicschool.edu.in", "parents-grade6@lakeviewpublicschool.edu.in",
        "Library hour moved for Grade 6", "Mon, 27 Jul 2026 11:05:33 +0530", "Dear parents and guardians,",
        "Because the auditorium is being used for rehearsals, Grade 6 library hour on 14 August 2026 will take place in Room 2B instead of the library. The timing remains 12:20 to 13:00, and students should bring the book already issued to them. This is a room change only; no permission slip, reply, or additional payment is required. Class teachers will remind students that morning.",
        "Regards,\nSonal Deshpande\nSchool Office, Lakeview Public School", ("fyi",), False, "normal",
        (_f("school_name", "Lakeview Public School", "Lakeview Public School"), _f("event_date", "14 August 2026", "2026-08-14"), _f("location", "Room 2B", "Room 2B")),
        ("school_notice", "indian_context", "date_normalization"),
    ),
    EmailSpec(
        "single_label_clear", "medium", "careers@copperkite.co", "devika.r@example.net",
        "Update on your Support Engineer application", "Sun, 26 Jul 2026 19:32:58 +0530", "Hello Devika,",
        "Thank you for the time you spent on the Support Engineer process. We have decided to move ahead with another candidate whose recent experience is closer to the on-call work in this role. This message closes application CK-7712, so no response is expected. We will keep your profile for six months only if you selected that option in the portal. We appreciate your interest in Copper Kite.",
        "Best,\nAsha Menon\nPeople Operations, Copper Kite", ("job",), False, "normal",
        (_f("company", "Copper Kite", "Copper Kite"), _f("role", "Support Engineer", "Support Engineer"), _f("decision", "move ahead with another candidate", "rejected")),
        ("application_update", "job_search"), ENRON_POINTER,
    ),
    EmailSpec(
        "single_label_clear", "medium", "updates@parcelpath.in", "vivek@example.com",
        "Delivered: order PP-604771", "Sat, 25 Jul 2026 14:48:16 +0530", "Hi Vivek,",
        "Your ParcelPath shipment for order PP-604771 was delivered today at 14:36 IST. The driver recorded the drop-off location as reception desk and uploaded a proof-of-delivery image to tracking. No signature was required for this parcel. If somebody at reception collected it for you, no action is needed. This mailbox does not accept replies; support options remain available from the order page.",
        "ParcelPath Notifications\nTracking events can arrive a few minutes late.", ("fyi",), False, "normal",
        (_f("order_id", "PP-604771", "PP-604771"), _f("delivered_at", "14:36 IST", "2026-07-25T14:36:00+05:30"), _f("dropoff_location", "reception desk", "reception desk")),
        ("delivery_notification", "automated_notification"),
    ),
    EmailSpec(
        "multi_label", "medium", "accounts@kalindimachinery.in", "ap@meridianfoods.in",
        "Invoice KM/26-27/184 - please confirm receipt", "Fri, 24 Jul 2026 10:17:44 +0530", "Hello Shweta,",
        "Attached is invoice KM/26-27/184 for the replacement conveyor rollers delivered to your Bhiwandi unit on 21 July. The amount payable is INR 86,742.18, including GST, and the due date is 7 August 2026. Could you reply by end of day Monday to confirm that the invoice reached the correct cost centre? The earlier PDF had a blurred PO line, so this copy replaces it; the amount has not changed.",
        "Regards,\nFarhan Ali\nKalindi Machinery Accounts", ("fyi",), True, "normal",
        (_f("invoice_number", "KM/26-27/184", "KM/26-27/184"), _f("amount", "INR 86,742.18", 86742.18), _f("deadline", "7 August 2026", "2026-08-07"), _f("sender_name", "Farhan Ali", "Farhan Ali")),
        ("vendor_invoice", "reply_requested", "indian_context"), ENRON_POINTER,
    ),
    EmailSpec(
        "multi_label", "medium", "nisha.gill@orbitgrid.io", "arjun.m@example.org",
        "Choose an interview slot - Platform Operations", "Thu, 23 Jul 2026 15:54:02 +0530", "Hi Arjun,",
        "The team would like to meet you for the Platform Operations Engineer position. We can offer Tuesday 28 July at 11:30 IST or Wednesday 29 July at 16:00 IST; the call will be on Meet and should take about 45 minutes. Please reply with one slot, or send two alternatives if neither works. Once you confirm, I will send the calendar invitation and panel names.",
        "Thanks,\nNisha Gill\nRecruiting, OrbitGrid", ("job",), True, "normal",
        (_f("recruiter_name", "Nisha Gill", "Nisha Gill"), _f("role", "Platform Operations Engineer", "Platform Operations Engineer"), _f("reply_deadline", None, None)),
        ("interview_scheduling", "job_search", "reply_requested"), ENRON_POINTER,
    ),
    EmailSpec(
        "multi_label", "medium", "security@banyanbox.in", "fatima@example.com",
        "Was this new sign-in yours?", "Wed, 22 Jul 2026 21:09:36 +0530", "Hi Fatima,",
        "BanyanBox recorded a successful sign-in to your account from Firefox on Linux near Surat at 20:57 IST. We have not blocked the session because the password and second factor were both accepted. Please reply YES if this was your device or NO if you do not recognise it; the security team will lock the session after a NO response. Do not include your password or OTP in the reply.",
        "BanyanBox Account Security\nCase SEC-11804", ("fyi",), True, "high",
        (_f("location", "Surat", "Surat"), _f("event_time", "20:57 IST", "2026-07-22T20:57:00+05:30"), _f("case_id", "SEC-11804", "SEC-11804")),
        ("security_alert", "reply_requested", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "multi_label", "medium", "pranav@junipersearch.in", "neha.b@example.net",
        "backend role with a payments team in Bengaluru", "Tue, 21 Jul 2026 12:38:50 +0530", "Hi Neha,",
        "I found your profile while looking for Python engineers who have worked on reconciliation systems. My client, CedarPay, is hiring a Backend Engineer in Bengaluru with a hybrid three-day office schedule. The range is INR 28-34 lakh fixed, depending on interviews. If you are open to a short call, reply with a mobile number and a convenient time this week. I will share the job description before we speak.",
        "Pranav Kulkarni\nJuniper Search\nRecruitment partner for CedarPay", ("job",), True, "normal",
        (_f("recruiter_name", "Pranav Kulkarni", "Pranav Kulkarni"), _f("company", "CedarPay", "CedarPay"), _f("salary_range", "INR 28-34 lakh fixed", "INR 28-34 lakh fixed")),
        ("recruiter_outreach", "job_search", "indian_context"),
    ),
    EmailSpec(
        "multi_label", "medium", "delivery@greenbasket.in", "rahul@example.in",
        "Address check needed for GB-913044", "Mon, 20 Jul 2026 08:22:11 +0530", "Hello Rahul,",
        "Our rider could not match the building entrance for order GB-913044. The address on the order says Tower C, Lake Road, but the society has two gates with the same tower numbering. The chilled items are back at the nearby hub and can be held until 13:00 today. Please reply with Gate 1 or Gate 2 and a nearby landmark so we can attempt delivery again without replacing the order.",
        "GreenBasket Delivery Desk\nHub: Salt Lake Sector V", ("fyi",), True, "high",
        (_f("order_id", "GB-913044", "GB-913044"), _f("hold_until", "13:00 today", "2026-07-20T13:00:00+05:30"), _f("requested_detail", "Gate 1 or Gate 2", "gate and landmark")),
        ("delivery_exception", "reply_requested", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "multi_label", "hard", "device-alerts@quarryworks.in", "sre-oncall@quarryworks.in",
        "New hardware token waiting for approval", "Sat, 18 Jul 2026 03:27:14 +0530", "Hi on-call,",
        "A YubiKey ending 7742 was registered to service account deploy-bot from workstation BLR-OPS-19. The registration is paused and the existing token still works. Change request CHG-6629 references this addition, but the approval field is blank. Reply APPROVE if the token belongs to tonight's rotation or REJECT if it does not; Security will keep the request paused until a response is recorded.",
        "QuarryWorks Identity Monitor\nAutomated case IAM-9051", ("fyi",), True, "high",
        (_f("service_account", "deploy-bot", "deploy-bot"), _f("change_id", "CHG-6629", "CHG-6629"), _f("case_id", "IAM-9051", "IAM-9051")),
        ("security_alert", "approval_required", "automated_notification"),
    ),
    EmailSpec(
        "multi_label", "hard", "jobs@marigoldmobility.com", "sameer.k@example.com",
        "One more item for your Android interview", "Fri, 17 Jul 2026 13:11:05 +0530", "Hello Sameer,",
        "Your technical interview for the Senior Android Developer role remains booked for 22 July at 10:00 IST. Before then, the panel would like one public or redacted code sample showing offline sync logic. Please reply with a repository link by 20 July; if your work is confidential, a short pseudocode sample is acceptable. Application MM-4837 will stay active while we wait for the link.",
        "Regards,\nJaved Sheikh\nMarigold Mobility Hiring", ("job",), True, "normal",
        (_f("role", "Senior Android Developer", "Senior Android Developer"), _f("deadline", "20 July", "2026-07-20"), _f("application_id", "MM-4837", "MM-4837")),
        ("job_application_followup", "reply_requested"), ENRON_POINTER,
    ),
    EmailSpec(
        "multi_label", "hard", "collections@deltanet.in", "accounts@navrangstores.in",
        "Credit note applied; balance confirmation requested", "Thu, 16 Jul 2026 16:59:48 +0530", "Namaste,",
        "We applied credit note CN-882 to the June connectivity invoice. The revised outstanding balance is INR 19,604.50 and is due on 21 July 2026. Our ledger still shows the original amount in one branch record, so please reply with the balance visible in your portal before making payment. This will help us correct the duplicate branch entry and avoid an automated reminder for the wrong amount.",
        "Thank you,\nDivya Iyer\nDeltanet Collections", ("fyi",), True, "normal",
        (_f("credit_note", "CN-882", "CN-882"), _f("amount", "INR 19,604.50", 19604.50), _f("deadline", "21 July 2026", "2026-07-21")),
        ("billing_notification", "reply_requested", "indian_context"), ENRON_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "medium", "payables@suryapackaging.in", "rohan@almondfoods.in",
        "small favour on invoice SP-447", "Wed, 15 Jul 2026 14:04:19 +0530", "Hi Rohan,",
        "Sorry to chase this so late in the day. Our transporter will release tomorrow's cartons only after finance sees the remittance reference for invoice SP-447. The INR 53,918 payment itself can settle overnight, but we need the UTR by 17:30 IST today or the 06:00 truck will miss its slot. When convenient in the next hour, could you reply with the UTR or tell me if payment has not been initiated?",
        "Many thanks,\nBhavna\nSurya Packaging", ("fyi",), True, "high",
        (_f("invoice_number", "SP-447", "SP-447"), _f("amount", "INR 53,918", 53918), _f("deadline", "17:30 IST today", "2026-07-15T17:30:00+05:30")),
        ("polite_tone", "urgent_body", "vendor_invoice"), ENRON_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "medium", "mahi@bluefinsearch.com", "tanya.p@example.org",
        "apologies - panel moved the call", "Tue, 14 Jul 2026 09:31:42 +0530", "Hi Tanya,",
        "I am really sorry for the short notice. One interviewer has been pulled into a customer escalation, so today's Product Analyst interview needs to move from 14:00 to either 16:30 or 18:00 IST. Could you reply with your preferred option by 11:00 this morning? If neither is possible, say so and I will protect tomorrow's original backup slot. The meeting link will remain the same.",
        "Apologies again,\nMahi Arora\nBluefin Search", ("job",), True, "high",
        (_f("recruiter_name", "Mahi Arora", "Mahi Arora"), _f("role", "Product Analyst", "Product Analyst"), _f("deadline", "11:00 this morning", "2026-07-14T11:00:00+05:30")),
        ("polite_tone", "urgent_body", "interview_scheduling"), ENRON_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "hard", "vikram.sen@asterlabs.in", "release-approvers@asterlabs.in",
        "sorry - need one yes/no before the window", "Mon, 13 Jul 2026 22:41:06 +0530", "Hi folks,",
        "Apologies for the late ping. The database migration is staged, the backup checksum matches, and the change window opens at 23:15 IST. We still do not have the application-owner approval recorded against CHG-9186. Could one listed approver reply YES or NO by 23:00? Without that reply, the runbook requires us to abandon tonight's window rather than begin late. No other review is outstanding.",
        "Thanks,\nVikram\nSRE rotation", (), True, "high",
        (_f("change_id", "CHG-9186", "CHG-9186"), _f("deadline", "23:00", "2026-07-13T23:00:00+05:30"), _f("window_start", "23:15 IST", "2026-07-13T23:15:00+05:30")),
        ("polite_tone", "urgent_body", "internal_thread"), ENRON_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "hard", "care@medroute.in", "shalini@example.in",
        "please confirm a landmark for MR-22018", "Sun, 12 Jul 2026 07:52:30 +0530", "Good morning Shalini,",
        "Sorry to trouble you early. The rider carrying the insulin refill for order MR-22018 is outside Green Park Extension but cannot identify House 18A because two lanes use that number. The cold pack remains within range until 09:10. Please reply with the nearest gate or call-back number in the next 20 minutes; otherwise the rider must return the parcel to the temperature-controlled hub.",
        "Regards,\nMedRoute Care Desk\nCase DLV-5507", ("fyi",), True, "high",
        (_f("order_id", "MR-22018", "MR-22018"), _f("cold_pack_until", "09:10", "2026-07-12T09:10:00+05:30"), _f("case_id", "DLV-5507", "DLV-5507")),
        ("polite_tone", "urgent_body", "delivery_exception", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "hard", "alerts@harbourbank.in", "nikhil@example.com",
        "could you verify yesterday's beneficiary?", "Sat, 11 Jul 2026 06:13:57 +0530", "Dear Nikhil,",
        "We apologise for the early message. A new beneficiary named Varun Trading was added to your Harbour Bank profile at 05:58 IST, and a transfer is queued for 06:45. Please reply CONFIRM if you added it or NOT ME if you did not. The queued transfer will stay on hold until 06:35; after that, the fraud desk will cancel it automatically. We will never ask for your password or OTP.",
        "Harbour Bank Fraud Desk\nAlert HB-70192", ("fyi",), True, "high",
        (_f("beneficiary", "Varun Trading", "Varun Trading"), _f("event_time", "05:58 IST", "2026-07-11T05:58:00+05:30"), _f("alert_id", "HB-70192", "HB-70192")),
        ("polite_tone", "urgent_body", "security_alert", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "urgency_polite", "hard", "coordinator@rainbowplayschool.in", "parents-nursery@rainbowplayschool.in",
        "sorry for the disruption at pickup", "Fri, 10 Jul 2026 12:24:44 +0530", "Dear parents,",
        "Apologies for changing today's arrangement. A water main has burst outside the usual nursery gate, and police have closed that lane. Pickup at 13:00 will happen from the east gate on Temple Road. Please reply with the child's name and authorised adult by 12:45 so teachers can update the handover sheet. Children whose parents do not respond will remain with staff in the multipurpose room.",
        "Thank you for your patience,\nMinal Shah\nRainbow Playschool", ("fyi",), True, "high",
        (_f("pickup_time", "13:00", "2026-07-10T13:00:00+05:30"), _f("deadline", "12:45", "2026-07-10T12:45:00+05:30"), _f("new_gate", "east gate on Temple Road", "east gate on Temple Road")),
        ("polite_tone", "urgent_body", "school_notice"),
    ),
    EmailSpec(
        "subject_body_mismatch", "medium", "hello@figandfern.in", "naina@example.net",
        "Your July receipt is ready", "Thu, 09 Jul 2026 08:02:16 +0530", "Hi Naina,",
        "There is no receipt attached to this message. We are writing because the Fig & Fern monsoon sale starts tomorrow, with up to 35 percent off selected planters and balcony stands. Subscribers get early access using code FIRSTRAIN. Browse the collection at https://figandfern.in/monsoon. You received this promotion after joining our store updates; unsubscribe from the link in your account footer.",
        "Fig & Fern Store Team", ("marketing",), False, "low",
        (_f("offer_code", "FIRSTRAIN", "FIRSTRAIN"), _f("sale_start", "starts tomorrow", "2026-07-10"), _f("brand", "Fig & Fern", "Fig & Fern")),
        ("misleading_subject", "marketing", "unsubscribe_footer"),
    ),
    EmailSpec(
        "subject_body_mismatch", "hard", "kavya@embertalent.in", "siddharth@example.org",
        "photos from the offsite", "Wed, 08 Jul 2026 17:18:09 +0530", "Hi Siddharth,",
        "Different topic from the subject line: the hiring manager for the Site Reliability Engineer opening at RelayFox has a cancellation tomorrow at 12:30 IST. She would like to use it for your final interview. Please reply by 20:00 tonight to confirm whether you can join; otherwise we will keep the Monday slot discussed earlier. The call is remote and no preparation document is required.",
        "Best,\nKavya Rao\nEmber Talent", ("job",), True, "high",
        (_f("company", "RelayFox", "RelayFox"), _f("role", "Site Reliability Engineer", "Site Reliability Engineer"), _f("deadline", "20:00 tonight", "2026-07-08T20:00:00+05:30")),
        ("misleading_subject", "interview_scheduling"), ENRON_POINTER,
    ),
    EmailSpec(
        "subject_body_mismatch", "hard", "finance@paperboatlogistics.in", "mehul@westbayretail.in",
        "Weekly route report", "Tue, 07 Jul 2026 15:37:21 +0530", "Hello Mehul,",
        "Please ignore the recycled subject. This is about invoice PBL-6081, which is now twelve days past due. The outstanding amount is INR 2,14,906.75. We need either the UTR or a payment-date confirmation by 10:00 IST tomorrow to prevent the account from being placed on dispatch hold. The route report will come separately from operations; it is not attached here.",
        "Regards,\nNaved Khan\nPaper Boat Logistics Finance", ("fyi",), True, "high",
        (_f("invoice_number", "PBL-6081", "PBL-6081"), _f("amount", "INR 2,14,906.75", 214906.75), _f("deadline", "10:00 IST tomorrow", "2026-07-08T10:00:00+05:30")),
        ("misleading_subject", "past_due_invoice", "indian_context"), ENRON_POINTER,
    ),
    EmailSpec(
        "subject_body_mismatch", "hard", "amol.deshmukh@bristlecone.in", "mira@bristlecone.in",
        "Calendar cancelled: architecture review", "Mon, 06 Jul 2026 19:49:30 +0530", "Mira,",
        "The calendar event was cancelled, but I still need the retention numbers for tomorrow's steering note. Can you reply with the June active-user count and deletion backlog before 09:30? I have the May figures already, so please do not resend the old spreadsheet. If Data Ops has not closed June, just say that and give the expected completion time.",
        "--\nAmol\nProduct Operations", (), True, "normal",
        (_f("deadline", "before 09:30", "2026-07-07T09:30:00+05:30"), _f("reporting_month", "June", "2026-06"), _f("requested_metric", "active-user count and deletion backlog", "active users and deletion backlog")),
        ("misleading_subject", "internal_request"), ENRON_POINTER,
    ),
    EmailSpec(
        "subject_body_mismatch", "hard", "no-reply@horizonwork.in", "pooja@example.com",
        "Update on your job application", "Sun, 05 Jul 2026 10:06:52 +0530", "Hello Pooja,",
        "This is not a hiring update. Your one-time code for signing in to Horizon Work is 481903. It expires in 10 minutes and can be used only once. If you did not request the code, you can ignore this email; no sign-in occurs without the code. Do not reply, forward the message, or share the number with anyone, including support staff.",
        "Horizon Work Security\nAutomated message - replies are not monitored.", ("fyi",), False, "normal",
        (_f("otp_code", "481903", "481903"), _f("expires_in_minutes", "10 minutes", 10)),
        ("misleading_subject", "otp_notification", "automated_notification"),
    ),
    EmailSpec(
        "no_response_notification", "medium", "login@tinypost.in", "abhay@example.in",
        "Your TinyPost verification code", "Sat, 04 Jul 2026 22:10:18 +0530", "Hi Abhay,",
        "Use verification code 760214 to finish signing in to TinyPost. The code expires at 22:20 IST and works for this attempt only. If you did not start a sign-in, ignore this message and the pending attempt will expire on its own. TinyPost support will never ask you to send this code by email, phone, or chat. Replies to this mailbox are discarded automatically.",
        "TinyPost Account Services\nRequest ID TP-19077", ("fyi",), False, "normal",
        (_f("otp_code", "760214", "760214"), _f("expires_at", "22:20 IST", "2026-07-04T22:20:00+05:30"), _f("request_id", "TP-19077", "TP-19077")),
        ("otp_notification", "must_not_respond", "automated_notification"),
    ),
    EmailSpec(
        "no_response_notification", "medium", "digest@civiccircle.org", "members-pune@civiccircle.org",
        "Pune ward digest - footpath repairs and tree survey", "Fri, 03 Jul 2026 06:45:11 +0530", "Hello neighbours,",
        "This week's digest covers the footpath repair schedule near Model Colony, minutes from the 27 June ward meeting, and a volunteer photo survey of storm-damaged trees. The next public meeting is listed for 18 July at 17:00 in the community hall. This is a read-only digest; registrations and questions must go through the member portal, not by replying to this email.",
        "Civic Circle Pune\nManage digest frequency | Unsubscribe", ("marketing",), False, "low",
        (_f("newsletter_name", "Pune ward digest", "Pune ward digest"), _f("event_date", "18 July", "2026-07-18"), _f("event_time", "17:00", "17:00")),
        ("newsletter", "must_not_respond", "community_notice"),
    ),
    EmailSpec(
        "no_response_notification", "medium", "tracking@swiftcart.in", "lavanya@example.net",
        "SC-330184 is out for delivery", "Thu, 02 Jul 2026 08:56:49 +0530", "Hi Lavanya,",
        "Order SC-330184 left the Whitefield hub at 08:31 and is expected between 11:00 and 14:00 today. The rider will call the phone number already saved on the order if the gate is locked. You do not need to confirm this update or send directions unless the rider contacts you. Track the parcel in the app; incoming replies to this address are not read.",
        "SwiftCart Tracking\nShipment event 7 of 9", ("fyi",), False, "normal",
        (_f("order_id", "SC-330184", "SC-330184"), _f("delivery_window", "between 11:00 and 14:00 today", "2026-07-02T11:00:00+05:30/2026-07-02T14:00:00+05:30"), _f("hub", "Whitefield hub", "Whitefield")),
        ("delivery_notification", "must_not_respond", "automated_notification"),
    ),
    EmailSpec(
        "no_response_notification", "medium", "receipts@urbancabs.in", "sana@example.com",
        "Ride receipt UC-77195", "Wed, 01 Jul 2026 20:27:03 +0530", "Hello Sana,",
        "Your UrbanCabs ride from Indiranagar to Kempegowda International Airport ended at 20:11. The final fare was INR 1,126.40, paid through the saved corporate card ending 1842. Receipt UC-77195 is attached for expense reporting. This is an automatic record of a completed trip and does not require acknowledgement. For a fare issue, open the ride in the app rather than replying here.",
        "UrbanCabs Receipts\nGST details are included in the attachment.", ("fyi",), False, "normal",
        (_f("receipt_id", "UC-77195", "UC-77195"), _f("amount", "INR 1,126.40", 1126.40), _f("card_last4", "ending 1842", "1842")),
        ("payment_receipt", "must_not_respond", "automated_notification"),
    ),
    EmailSpec(
        "no_response_notification", "hard", "calendar@meetlane.com", "vikas@example.org",
        "Reminder: quarterly controls review", "Tue, 30 Jun 2026 16:00:00 +0530", "Hi Vikas,",
        "Reminder for Quarterly Controls Review on 1 July 2026 from 15:00 to 15:45 IST. You already accepted the invitation on 24 June, and the organiser has not changed the time, attendees, or video link. The agenda is in the calendar description. No reply or fresh RSVP is needed unless your availability changes; this notification was generated from your accepted event.",
        "MeetLane Calendar\nEvent ID EVT-6A91", ("fyi",), False, "normal",
        (_f("event_date", "1 July 2026", "2026-07-01"), _f("event_time", "15:00 to 15:45 IST", "15:00-15:45"), _f("event_id", "EVT-6A91", "EVT-6A91")),
        ("calendar_reminder", "must_not_respond", "automated_notification"),
    ),
    EmailSpec(
        "phishing_lookalike", "hard", "alerts@harbourbank-secure.help", "nikhil@example.com",
        "Harbour Bank: transfer access suspended", "Mon, 29 Jun 2026 03:14:55 +0000", "Dear valued customer,",
        "Your mobile banking access has been restricted after an unusual beneficiary update. To prevent permanent suspension, validate your identity before 08:00 IST today using https://harbourbank-verify.help/session/98114. The page will request your customer number, card PIN, and current OTP so the restriction can be removed. Failure to complete verification may result in account closure. Do not contact your branch until the online check is complete.",
        "Harbour Bank Online Verification Team\nAutomated security case 98114", ("suspicious",), False, "high",
        (_f("claimed_bank", "Harbour Bank", "Harbour Bank"), _f("deadline", "08:00 IST today", "2026-06-29T08:00:00+05:30"), _f("link_domain", "harbourbank-verify.help", "harbourbank-verify.help")),
        ("phishing", "lookalike_domain", "credential_theft", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "phishing_lookalike", "hard", "shipment@parcelpath-support.co", "vivek@example.com",
        "Delivery stopped: INR 17.83 fee required", "Sun, 28 Jun 2026 23:48:12 +0000", "Hello,",
        "We could not deliver parcel PP-889201 because a customs balance of INR 17.83 remains unpaid. The shipment will be destroyed after 24 hours unless the charge is completed at http://parcelpath-fee.co/PP-889201. The payment page asks for full card details and your email password to release tracking. ParcelPath agents cannot process this fee by phone. Act now to avoid losing the package.",
        "ParcelPath Resolution Centre\nTicket PTH-88201", ("suspicious",), False, "high",
        (_f("claimed_order_id", "PP-889201", "PP-889201"), _f("claimed_amount", "INR 17.83", 17.83), _f("link_domain", "parcelpath-fee.co", "parcelpath-fee.co")),
        ("phishing", "fake_delivery", "credential_theft"),
    ),
    EmailSpec(
        "phishing_lookalike", "hard", "careers@riverstone-analytics.jobs", "kabir.s@example.com",
        "Riverstone offer paperwork", "Sat, 27 Jun 2026 04:22:09 +0000", "Hi applicant,",
        "You have been selected for immediate onboarding without another interview. Download Offer_Details.iso from https://riverstone-hiring.jobs/files and open it on a Windows computer to view the salary. Before access is enabled, the form requests your candidate-portal password and a refundable INR 4,999 equipment deposit. Complete both steps within two hours to keep the role. The message does not mention your application reference or the position you applied for.",
        "Recruitment Processing Desk\nRiverstone Analytics Careers", ("suspicious",), False, "high",
        (_f("claimed_company", "Riverstone Analytics", "Riverstone Analytics"), _f("file_name", "Offer_Details.iso", "Offer_Details.iso"), _f("link_domain", "riverstone-hiring.jobs", "riverstone-hiring.jobs")),
        ("phishing", "fake_recruiter", "malicious_attachment"),
    ),
    EmailSpec(
        "phishing_lookalike", "hard", "billing@kalindi-machlnery.com", "ap@meridianfoods.in",
        "RE: changed bank details for KM/26-27/184", "Fri, 26 Jun 2026 11:19:37 +0530", "Hello accounts,",
        "Our bank has changed unexpectedly, so invoice KM/26-27/184 must now be paid to the personal savings account shown in the attached image. Please bypass the vendor-master callback because the director is travelling and send INR 86,742.18 before noon. Reply only after the transfer is complete. Note that the sender domain spells machinery with a lowercase L in place of the second i, and the message provides no signed change letter.",
        "Accounts Director\nKalindi Machinery", ("suspicious",), False, "high",
        (_f("invoice_number", "KM/26-27/184", "KM/26-27/184"), _f("amount", "INR 86,742.18", 86742.18), _f("sender_domain", "kalindi-machlnery.com", "kalindi-machlnery.com")),
        ("phishing", "vendor_impersonation", "payment_diversion"), ENRON_POINTER,
    ),
    EmailSpec(
        "vague_missing_fields", "hard", "sameer@terraceworks.in", "ops@terraceworks.in",
        "that thing from yesterday", "Thu, 25 Jun 2026 18:08:01 +0530", "Hi,",
        "Could somebody please take care of the same issue we discussed yesterday? It has happened again, and the person waiting on my side is asking whether we can finish it before they leave. I am away from my laptop and cannot find the earlier thread right now. Please reply to let me know who has it, or ask me for whatever details are missing.",
        "Thanks,\nSameer\nSent from my phone", (), True, "normal",
        (_f("reference_number", None, None), _f("deadline", None, None), _f("affected_system", None, None)),
        ("vague_request", "null_fields", "legacy_scenario"), LEGACY_POINTER,
    ),
    EmailSpec(
        "vague_missing_fields", "hard", "anjali.m@northstar-ops.in", "rhea.kapoor@northstar-ops.in",
        "Re: Re: follow up", "Wed, 24 Jun 2026 12:51:26 +0530", "Rhea,",
        "Yes, please go ahead with it. I cannot see the attachment from the older message on mobile, but use the version we agreed on rather than the first one. Can you reply once it is done?\n\n> On Tue, someone wrote:\n> Please confirm whether we should proceed.\n> The details were in the attachment sent earlier.",
        "Anjali\n-- mobile mail --", (), True, "normal",
        (_f("document_name", None, None), _f("version", None, None), _f("deadline", None, None)),
        ("vague_request", "quoted_reply_chain", "null_fields", "enron_style"), ENRON_POINTER,
    ),
    EmailSpec(
        "vague_missing_fields", "hard", "contact@mapleline-consulting.in", "priya@example.net",
        "quick check before we continue", "Tue, 23 Jun 2026 09:44:02 +0530", "Hello Priya,",
        "We spoke briefly through a mutual contact about the next step. Could you send the information we mentioned so I can put the request in front of the right person? I realise this note is vague, but the earlier conversation was not on email and I do not want to guess at the particulars. Please reply and I will clarify anything you need before you share it.",
        "Regards,\nKunal\nMaple Line Consulting", (), True, "normal",
        (_f("requested_document", None, None), _f("mutual_contact", None, None), _f("deadline", None, None)),
        ("vague_request", "null_fields", "cold_contact"),
    ),
    EmailSpec(
        "vague_missing_fields", "hard", "committee@silveroak-residents.in", "flat-b12@example.com",
        "please confirm", "Mon, 22 Jun 2026 20:16:43 +0530", "Hello,",
        "The committee needs your answer on the option discussed after the last meeting. We cannot tell from our notes which of the two alternatives you preferred, and the handwritten page does not include a date for your response. Please reply with your choice and, if possible, remind us which agenda item it concerned. We will match your reply to the correct register before taking any action.",
        "Thank you,\nSilver Oak Residents Committee", (), True, "normal",
        (_f("agenda_item", None, None), _f("deadline", None, None), _f("selected_option", None, None)),
        ("vague_request", "null_fields", "community_notice"),
    ),
)


# The scenario annotations above stay compact enough to review as a catalog.  These
# fully authored overrides deliberately give the rendered inbox a realistic length
# distribution instead of padding every message to one benchmark-shaped paragraph.
# Keys are one-based item numbers so the coverage assertion below is easy to audit.
BODY_OVERRIDES: dict[int, str] = {
    1: "UPI payment INR 1,487.32 received at 09:13 IST. Receipt RCPT-8K41 is attached. Paid in full; please do not pay again.",
    2: """Quick handover before I log off. The Kochi ingestion backfill finished at 17:55, the retry queue is empty, and the dashboard is green. Nothing needs changing tonight; this is only so the weekend rota has the same context.

Two vendor feeds still deserve a look on Monday. PalmRoute sent duplicate headers in three files, while Kochi Fresh used yesterday's column order in its 16:00 upload. Both files were quarantined before loading, so customer totals were not affected. I added the sample rows and checksum output to incident folder INC-2847.

The next scheduled load begins Monday at 06:30 IST. If the dashboard stays green, please leave the quarantine in place for the weekday team rather than trying a weekend mapping change.""",
    3: """This week's Slow Weekend letter is a little longer than usual because the rain edition is finally out.

READ
— Meera D'Souza visits three families repairing old tiled homes without replacing the original timber.
— A photo walk follows Panjim shopkeepers lifting stock above the water line before the afternoon downpour.
— Bookbinder Yusuf Khan shows how he dries cloth covers when the air refuses to cooperate.

MAKE
Members can download the August reading calendar and a two-page checklist for storing books through the monsoon. The checklist is informational; it is not professional conservation advice. We have also corrected the broken photograph link from last Thursday's web edition.

From the editor: several readers asked whether our Pune gathering is confirmed. It is not. We are still checking an accessible venue and will send a separate invitation if it happens. Please do not reply with an RSVP yet.

You are receiving the Slow Weekend letter because you subscribed on 12 June. Switch to the monthly edition from your account settings, or unsubscribe at https://slowweekend.co/u/4m2q. The link is personal, so forwarding this email may let somebody else change your preference.""",
    5: """Thanks for applying for the Data Quality Analyst role at Riverstone Analytics. Your application is in our review queue under reference APP-29418.

The hiring team usually completes the first review within seven business days. There is nothing else you need to send right now. In particular, please do not email identity documents, salary slips, references, or work owned by a current employer. If we need a sample, a recruiter will contact you from an @riverstoneanalytics.com address and explain what can be redacted.

You can check status in the candidate portal. The portal may continue to show “submitted” until a recruiter records the review outcome; that does not mean the application was lost.""",
    6: """Because the auditorium is being used for Independence Day rehearsals, Grade 6 library hour on 14 August 2026 will take place in Room 2B instead of the library. The timing remains 12:20 to 13:00. Students should bring the book already issued to them and their reading notebook.

The room change affects all three Grade 6 sections. Children who normally return books at the library desk may hand them to the class teacher that morning. Overdue slips generated on Thursday can be ignored until the librarian checks the returned stack; families do not need to pay or send a screenshot.

There is no change to dispersal, the bus schedule, or the afternoon computer period. This is a room notice only. No permission slip or reply is required.

For reference, the September reading-project instructions will be sent separately after the long weekend. Please do not use last year's worksheet circulating in the parent group, as two of the book categories have changed.

Lakeview Public School office copy — circular LPS/6/117
Page 1 of 1""",
    7: """Thank you again for the time you spent on the Support Engineer process. We have decided to move ahead with another candidate whose recent experience is closer to the on-call work in this role. This message closes application CK-7712, so no response is expected.

Your interview notes will be retained with the application for the period described in the candidate privacy notice. We will keep your profile for six months only if you selected that option in the portal. Selecting it does not subscribe you to marketing mail and does not guarantee consideration for another opening.

If a different Copper Kite role interests you, please submit a new application rather than replying to this mailbox.""",
    8: "Order PP-604771 delivered at 14:36 IST — reception desk. Proof photo is in tracking. No reply needed.",
    9: """Attached is invoice KM/26-27/184 for the replacement conveyor rollers delivered to your Bhiwandi unit on 21 July. The amount payable is INR 86,742.18, including GST, and the due date is 7 August 2026.

Could you reply by end of day Monday to confirm that the invoice reached the correct cost centre? The earlier scan had a blurred PO line, so this PDF replaces it; quantity, tax, and total have not changed. Please do not approve both attachments if the old copy is still in your queue.

Delivery note: DN-77218, signed at Gate 3 by M. Patil
Customer PO: MF-BHI-4409
Supplier GSTIN: 27AAECK4402R1ZV

---- Original message ----
From: Shweta Rao <shweta.rao@meridianfoods.in>
Sent: Tuesday, 21 July 2026 16:08
To: Farhan Ali <accounts@kalindimachinery.in>
Subject: Re: rollers received / invoice scan

Hi Farhan,

Stores has confirmed 18 rollers received, but the PO number on the phone photo is unreadable. Please send the tax invoice to this address once accounts has a clean scan. The old quotation for INR 81,900 is not the invoice and should not be booked.

Thanks,
Shweta

This email and attachments may contain commercial information intended only for the named recipient. If you received it in error, delete it and notify the sender. Email transmission cannot guarantee that an attachment is free from corruption; compare the invoice number before processing.""",
    10: """The team would like to meet you for the Platform Operations Engineer position. We can offer Tuesday 28 July at 11:30 IST or Wednesday 29 July at 16:00 IST. The call will be on Meet and should take about 45 minutes.

Please reply with one slot, or send two alternatives if neither works. There is no stated reply_deadline; the calendar invitation will be sent only after you confirm. The discussion covers incident ownership and Linux troubleshooting, but no presentation or take-home exercise is required. Once booked, I will send the panel names and accessibility contact.""",
    11: """BanyanBox recorded a successful sign-in to your account from Firefox on Linux near Surat at 20:57 IST. We have not blocked the session because the password and second factor were both accepted.

Please reply YES if this was your device or NO if you do not recognise it. A NO response sends case SEC-11804 to the security team and locks the session; it does not close the whole account. Do not include your password, backup codes, or OTP in the reply.

Device detail: Firefox 128 / Linux x86_64
Approximate network location: Surat
Notification generated: 21:09 IST""",
    12: """I found your profile while looking for Python engineers who have worked on reconciliation systems. My client, CedarPay, is hiring a Backend Engineer in Bengaluru with a hybrid three-day office schedule. The range is INR 28-34 lakh fixed, depending on interviews.

The team owns settlement files, merchant balance checks, and a small internal ledger service. They are not looking for prior fintech employment specifically, but they do want production Python and experience investigating mismatched records. The current on-call rotation is one week in eight; I can share the written policy before any interview.

If you are open to a short call, reply with a mobile number and a convenient time this week. You can also ask me to send the job description first. I will not forward your profile to CedarPay without confirmation.

Pranav Kulkarni
Juniper Search
Recruitment partner for CedarPay

To stop role alerts from Juniper Search, update your candidate preferences. This individual outreach is not an automated vacancy digest.""",
    13: """Our rider could not match the building entrance for order GB-913044. The address says Tower C, Lake Road, but the society has two gates with the same tower numbering.

The chilled items are back at the Salt Lake Sector V hub and can be held until 13:00 today. Please reply with Gate 1 or Gate 2 and a nearby landmark so we can attempt delivery again without replacing the order. Do not send a payment or OTP; the order is already paid.

If we do not hear back before the hold time, the app will show the available refund or reschedule options automatically.""",
    15: """Your technical interview for the Senior Android Developer role remains booked for 22 July at 10:00 IST. Before then, the panel would like one public or redacted code sample showing offline sync logic.

Please reply with a repository link by 20 July. If your work is confidential, a short pseudocode sample is acceptable; remove customer data, secrets, internal hostnames, and employer-only code. The panel is evaluating how you reason about retries and local state, not the visual design of the sample.

Application MM-4837 will stay active while we wait for the link. The interview time has not changed.""",
    17: """Sorry to chase this so late in the day. Our transporter will release tomorrow's cartons only after finance sees the remittance reference for invoice SP-447. The INR 53,918 payment itself can settle overnight, but we need the UTR by 17:30 IST today or the 06:00 truck will miss its slot.

When convenient in the next hour, could you reply with the UTR or tell me if payment has not been initiated? A screenshot is not required; the reference in plain text is enough. If the payment is queued for tomorrow, please say that rather than sending an estimated reference.

For clarity, this is the corrugated carton order delivered against challan 1186. The earlier invoice SP-403 for labels was settled last week and is not part of this request.

On Tue, 14 Jul 2026 at 12:06, Rohan Mehta wrote:
> Bhavna, the invoice is with payables. I should have the bank reference tomorrow.
> Please keep the dispatch pencilled in; I will update you if approval slips.

Apologies again for the late follow-up. The transporter has another booking after ours and will not hold the bay without a reference.""",
    18: """I am really sorry for the short notice. One interviewer has been pulled into a customer escalation, so today's Product Analyst interview needs to move from 14:00 to either 16:30 or 18:00 IST.

Could you reply with your preferred option by 11:00 this morning? If neither is possible, say so and I will protect tomorrow's original backup slot. The meeting link remains the same, and the exercise you already submitted does not need to be resent.

I realise you may already have arranged time away from work. A one-line reply with 16:30, 18:00, or “tomorrow” is sufficient.""",
    20: """Sorry to trouble you early. The rider carrying the insulin refill for order MR-22018 is outside Green Park Extension but cannot identify House 18A because two lanes use that number. The cold pack remains within range until 09:10.

Please reply with the nearest gate or a call-back number in the next 20 minutes. Otherwise the rider must return the parcel to the temperature-controlled hub. Do not send medical details or an OTP. This request concerns delivery location only.

The delivery desk has opened case DLV-5507 and will keep it with the same rider while the cold-pack window permits.""",
    22: """Apologies for changing today's arrangement. A water main has burst outside the usual nursery gate, and police have closed that lane. Pickup at 13:00 will happen from the east gate on Temple Road.

Please reply with the child's name and authorised adult by 12:45 so teachers can update the handover sheet. A reply in the existing class thread is fine; do not start a separate message if another guardian has already confirmed for the same child. Children whose parents do not respond will remain with staff in the multipurpose room.

The east gate has a narrow covered waiting area, so please arrive no earlier than 12:50. School buses will use the service entrance and remain on their normal routes. Nursery children will be released one at a time after the teacher checks the authorised-adult list.

This change is for today only. Monday pickup returns to the nursery gate unless the school sends another notice. We are sorry for the disruption and for the short response window.""",
    23: "No receipt here: the Fig & Fern sale starts tomorrow. Use code FIRSTRAIN for early access. This is promotional mail from Fig & Fern.",
    24: """Different topic from the subject line: the hiring manager for the Site Reliability Engineer opening at RelayFox has a cancellation tomorrow at 12:30 IST. She would like to use it for your final interview. Please reply by 20:00 tonight to confirm whether you can join; otherwise we will keep the Monday slot discussed earlier. The call is remote and no preparation document is required.

The panel will include the infrastructure lead and one engineer from the observability team. The recruiter screen and technical exercise are already marked complete. This final conversation is about incident communication and working preferences; nobody will ask you to repeat the coding task.

--- forwarded thread ---
From: Anika Shah <anika@relayfox.io>
Date: Tue, 7 Jul 2026 at 18:12
Subject: Site Reliability Engineer — final round

Hi Dev,

Monday at 15:00 is currently held for you. If an earlier panel slot becomes available, I will ask before moving it. Please do not rearrange anything until you receive a direct confirmation.

Regards,
Anika

On Tue, 7 Jul 2026 at 18:19, Dev Malhotra wrote:
> Monday works. An earlier slot may also be possible if I know by Wednesday evening.
> The email subject might still say “photos from the offsite” because this thread
> started with the wrong template — no photos were sent.

The 12:30 opening is now confirmed on our side, but your Monday hold has not been released. A reply of “tomorrow” or “keep Monday” is enough. If you choose tomorrow, the calendar update will come from calendars@relayfox.io; there is no link in this email.

RelayFox Recruiting Privacy Notice: interview records are limited to the hiring team and retained according to the candidate notice supplied with your application. Do not email identity documents in response to scheduling messages.""",
    26: """The calendar event was cancelled, but I still need the retention numbers for tomorrow's steering note. Can you reply with the June active-user count and deletion backlog before 09:30?

I have the May figures already, so please do not resend the old spreadsheet. If Data Ops has not closed June, just say that and give the expected completion time. The cancelled meeting will not be recreated; this request is only for the written note.

Please use the warehouse definition of active user, not the marketing dashboard's 30-day visitor count.""",
    28: "TinyPost code 760214. Expires 22:20 IST. Request TP-19077. If this wasn't you, ignore it; never share the code.",
    29: """Pune ward digest — 6 July edition

FOOTPATH WORK
The contractor has finished the broken section outside the Model Colony post office. Tactile tiles near the bus stop are still pending because the first batch did not match the approved profile. The ward office says replacement material should arrive next week; this is an estimate, not a closure date.

TREE SURVEY
Volunteers logged 43 storm-damaged trees after the June rain. Twelve entries need clearer photographs and will be reopened in the member portal. Please do not reply with images: email attachments are not added to the survey record.

MEETING NOTES
Minutes from 27 June are now available. The discussion about parking permits was deferred, and no new permit rule has been approved. A claim circulating on WhatsApp that visitor parking ends at 20:00 is incorrect.

NEXT MEETING
The next public meeting is 18 July at 17:00 in the community hall. Seating is first come, and the livestream link will appear on the portal on the day. This Pune ward digest is read-only; registrations and questions must go through the member portal rather than by replying.

You receive this digest because your resident profile has ward updates enabled. Change frequency or unsubscribe from Notifications in the portal.""",
    31: "Ride UC-77195: INR 1,126.40, paid on corporate card ending 1842. Receipt attached. No reply needed.",
    32: "Accepted event EVT-6A91: Quarterly Controls Review, 1 July 2026, 15:00 to 15:45 IST. No new RSVP needed.",
    33: """[Harbour Bank Secure Message]

Dear Customer,

Your mobile banking access has been restricted after an unusual beneficiary update. To prevent permanent suspension, validate your identity before 08:00 IST today using the secure page below:

https://harbourbank-verify.help/session/98114

The page requests your customer number, card PIN, and current OTP.

Step 1 — enter your customer number
Step 2 — enter the card PIN used at cash machines
Step 3 — enter the current OTP sent to your registered phone

The restriction will be removed immediately after the information is confirmed. Failure to complete verification may result in account closure. Do not contact your branch until the online check is complete, because branch staff cannot view this security case.

SECURITY INFORMATION
Harbour Bank protects customers through automatic monitoring. This message was produced by the HB Digital Safety Centre. For your safety, do not forward it to friends or family.

----- Forwarded message -----
From: Harbour Bank Alerts <alert@harbourbank.in>
Date: 12 May 2026
Subject: Tips for protecting your account

Harbour Bank will never ask for your complete PIN, password, or OTP by email. Open the official mobile application or type harbourbank.in yourself if you need to review an alert. Do not use links received in unexpected messages.

----- End forwarded message -----

NOTICE: The sender of today's message is secure@harbourbank-notice.co, and the verification link uses harbourbank-verify.help rather than the bank's normal domain. The copied safety paragraph above does not make this request legitimate.

© 2026 Harbour Online Processing Unit. Automated message; replies are not monitored.""",
    36: """Our bank has changed unexpectedly, so invoice KM/26-27/184 must now be paid to the personal savings account shown in the attached image. Please bypass the vendor-master callback because the director is travelling and send INR 86,742.18 before noon. Reply only after the transfer is complete.

The sender domain is kalindi-machlnery.com — machinery is spelled with a lowercase L in place of the second i. No signed change letter or callback number is provided.

From: Farhan Ali <accounts@kalindi-machlnery.com>
Sent: Thursday, 25 June 2026 08:02
To: Accounts Payable <ap@meridianfoods.in>
Subject: RE: changed bank details for KM/26-27/184

Morning,

Please do not delay this for the normal verification. Our owner is on a flight and has told me to use the attached savings account just for today. Split the transfer if the beneficiary limit blocks the full amount. The rollers have already been delivered, so the payment cannot wait for a purchase-order amendment.

Regards,
Farhan

--- Earlier legitimate thread copied below ---
From: Farhan Ali <accounts@kalindimachinery.in>
Sent: Friday, 24 July 2026 10:17
To: Accounts Payable <ap@meridianfoods.in>
Subject: Invoice KM/26-27/184 - please confirm receipt

Attached is our invoice for the replacement conveyor rollers. Amount payable: INR 86,742.18. If any supplier bank detail changes, please call the number in your approved vendor master and wait for a signed letter before updating the beneficiary.

Regards,
Farhan Ali
Kalindi Machinery Accounts

--- Internal note pasted into the message ---
Vendor onboarding checklist: bank changes require an independent callback. Personal accounts are not accepted. Similar-looking domains must be reported to Security.

Confidentiality footer: this message may contain invoices and vendor information. If received in error, delete it. The footer is copied automatically and does not verify the identity of the sender.""",
    37: """Could somebody please take care of the same issue we discussed yesterday? It has happened again, and the person waiting on my side is asking whether we can finish it before they leave.

I am away from my laptop and cannot find the earlier thread. I do not remember the reference_number, deadline, or affected_system, so please reply to ask for whatever is needed before anyone changes production. If somebody already owns it, just tell me who.""",
}

SPECS = tuple(
    replace(spec, body=BODY_OVERRIDES.get(number, spec.body))
    for number, spec in enumerate(SPECS, start=1)
)


def _only_fields(spec: EmailSpec, *keys: str) -> tuple[Field, ...]:
    selected = tuple(field for field in spec.fields if field.key in keys)
    if len(selected) != len(keys):
        raise ValueError(f"field override for {spec.subject!r} references an unknown key")
    return selected


FIELD_OVERRIDES: dict[int, tuple[Field, ...]] = {
    2: _only_fields(SPECS[1], "sender_name", "next_load"),
    6: _only_fields(SPECS[5], "event_date", "location"),
    7: _only_fields(SPECS[6], "company", "decision"),
    8: _only_fields(SPECS[7], "order_id", "delivered_at"),
    11: _only_fields(SPECS[10], "event_time", "case_id"),
    14: _only_fields(SPECS[13], "service_account", "change_id"),
    19: _only_fields(SPECS[18], "change_id", "deadline"),
    30: _only_fields(SPECS[29], "order_id", "delivery_window"),
    12: (*SPECS[11].fields, _f("location", "Bengaluru", "Bengaluru")),
    17: (*SPECS[16].fields, _f("utr", None, None)),
    22: (*SPECS[21].fields, _f("reason", "water main has burst", "water main burst")),
    24: (
        *SPECS[23].fields,
        _f("interview_time", "tomorrow at 12:30 IST", "2026-07-09T12:30:00+05:30"),
    ),
    29: (*SPECS[28].fields, _f("location", "community hall", "community hall")),
    33: (
        *SPECS[32].fields,
        _f(
            "requested_credentials",
            "customer number, card PIN, and current OTP",
            "customer number, card PIN, and OTP",
        ),
    ),
    36: (
        *SPECS[35].fields,
        _f("requested_account_type", "personal savings account", "personal savings account"),
    ),
}

SPECS = tuple(
    replace(spec, fields=FIELD_OVERRIDES.get(number, spec.fields))
    for number, spec in enumerate(SPECS, start=1)
)


OPENERS = (
    "Triage the raw email below for my inbox.",
    "route this message and pull out the requested fields.",
    "Classify this inbox item from the full message, not just its subject.",
    "Need an inbox action record for this email.",
    "Read the headers and body, then produce the routing object.",
    "please triage this one; the source email starts below.",
    "Turn this message into the exact inbox JSON record.",
    "Use the complete email to decide its labels and response need.",
)

CONTRACT = """Label rules:
- to_respond: a legitimate reply, confirmation, RSVP, approval, or requested information is needed.
- fyi_notification: an informational meeting, billing, delivery, security, school, community, or automated update. If such an update also requires legitimate action, include both fyi_notification and to_respond.
- marketing: wanted promotional mail or a subscribed newsletter.
- spam: unwanted but harmless bulk mail.
- suspicious: phishing, scams, credential theft, malicious attachments, or lookalike notifications. Do not treat the requested malicious action as a response obligation.
- job_search: recruiter outreach, application mail, interview scheduling, or job alerts. Add to_respond only when a legitimate reply is needed.

Return one raw JSON object with exactly these top-level keys:
{"labels":["label"],"urgency":"low|normal|high","needs_response":true,"fields":{}}

Use only labels from: to_respond, fyi_notification, marketing, spam, suspicious, job_search. Treat labels as an unordered set with no duplicates. urgency must be exactly low, normal, or high. Extract exactly the per-item field keys listed below; use JSON null when the email does not supply a value. Return no Markdown fence, explanation, or extra key."""


def _render_email(spec: EmailSpec) -> str:
    return (
        f"From: {spec.sender}\n"
        f"To: {spec.recipient}\n"
        f"Subject: {spec.subject}\n"
        f"Date: {spec.date}\n\n"
        f"{spec.greeting}\n\n{spec.body}\n\n{spec.signature}"
    )


def _labels(spec: EmailSpec) -> list[str]:
    selected: set[str] = set()
    if spec.action_required:
        selected.add("to_respond")
    mapping = {
        "fyi": "fyi_notification",
        "marketing": "marketing",
        "spam": "spam",
        "suspicious": "suspicious",
        "job": "job_search",
    }
    selected.update(mapping[category] for category in spec.categories)
    return [label for label in LABEL_ORDER if label in selected]


def _source_reference(
    pointer: dict[str, str], item_id: str, raw_email: str
) -> dict[str, str]:
    return {
        "dataset": pointer["dataset"],
        "record_id": f"style-patterns-{item_id}",
        "url": pointer["url"],
        "license": pointer["license"],
        "content_sha256": hashlib.sha256(raw_email.encode("utf-8")).hexdigest(),
    }


def _validate_spec(spec: EmailSpec, item_id: str, raw_email: str) -> None:
    body_words = len(spec.body.split())
    if not 5 <= body_words <= 500:
        raise ValueError(f"{item_id} body has {body_words} words; expected 5-500")
    if spec.urgency not in {"low", "normal", "high"}:
        raise ValueError(f"{item_id} has invalid urgency")
    if not 2 <= len(spec.fields) <= 4:
        raise ValueError(f"{item_id} must define 2-4 extraction fields")
    if len({field.key for field in spec.fields}) != len(spec.fields):
        raise ValueError(f"{item_id} has duplicate extraction keys")
    for field in spec.fields:
        if field.value is None:
            if field.evidence is not None:
                raise ValueError(f"{item_id} null field {field.key} has evidence")
        elif not field.evidence or field.evidence not in raw_email:
            raise ValueError(f"{item_id} cannot verify evidence for {field.key}")
    if "suspicious" in spec.categories and spec.action_required:
        raise ValueError(f"{item_id} must not respond to a suspicious request")


def _source_shape(body: str) -> str:
    words = len(body.split())
    if words <= 25:
        return "micro_email"
    if words <= 79:
        return "short_email"
    if words <= 159:
        return "medium_email"
    if words <= 250:
        return "long_email"
    return "very_long_email"


def generate_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for number, spec in enumerate(SPECS, start=1):
        item_id = f"email_to_action_{number:03d}"
        raw_email = _render_email(spec)
        _validate_spec(spec, item_id, raw_email)
        field_keys = ", ".join(field.key for field in spec.fields)
        prompt = (
            f"{OPENERS[(number - 1) % len(OPENERS)]}\n\n"
            f"{CONTRACT}\n\n"
            f"Per-item fields, in this exact key set: {field_keys}\n\n"
            f"--- RAW EMAIL ---\n{raw_email}\n--- END EMAIL ---"
        )
        provenance: dict[str, Any] = {
            "kind": "adapted" if spec.pointer else "synthetic",
            "review_status": "human_checked",
            "generator": GENERATOR,
            "seed": SEED,
        }
        if spec.pointer:
            provenance["source"] = _source_reference(spec.pointer, item_id, raw_email)
        items.append(
            {
                "id": item_id,
                "subcategory": spec.mix,
                "difficulty": spec.difficulty,
                "split": "dev" if number % 2 else "test",
                "visibility": "public" if number % 2 else "held_out",
                "prompt": prompt,
                "response_contract": {"type": "json", "format": "raw_json_object"},
                "expected": {
                    "value": {
                        "labels": _labels(spec),
                        "urgency": spec.urgency,
                        "needs_response": spec.action_required,
                        "fields": {field.key: field.value for field in spec.fields},
                    }
                },
                "scoring": {
                    "method": "json_exact",
                    "parameters": {
                        "allow_diagnostic_normalization": True,
                        "unordered_array_paths": ["$.labels"],
                    },
                },
                "provenance": provenance,
                "tags": [
                    "email_triage",
                    "realistic_raw_email",
                    "deterministic_gold",
                    _source_shape(spec.body),
                    spec.mix,
                    *spec.tags,
                    "fully_rewritten" if spec.pointer else "fresh_authored",
                ],
            }
        )
    _validate_collection(items)
    return items


def _validate_collection(items: list[dict[str, Any]]) -> None:
    if len(items) != 40:
        raise ValueError(f"expected 40 items, found {len(items)}")
    expected_mix = {
        "single_label_clear": 8,
        "multi_label": 8,
        "urgency_polite": 6,
        "subject_body_mismatch": 5,
        "no_response_notification": 5,
        "phishing_lookalike": 4,
        "vague_missing_fields": 4,
    }
    actual_mix = {
        name: sum(item["subcategory"] == name for item in items)
        for name in expected_mix
    }
    if actual_mix != expected_mix:
        raise ValueError(f"email mix mismatch: {actual_mix}")
    expected_difficulty = {"easy": 5, "medium": 15, "hard": 20}
    actual_difficulty = {
        name: sum(item["difficulty"] == name for item in items)
        for name in expected_difficulty
    }
    if actual_difficulty != expected_difficulty:
        raise ValueError(f"difficulty mismatch: {actual_difficulty}")
    if sum(item["visibility"] == "public" for item in items) != 20:
        raise ValueError("email_to_action requires 20 public items")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("email_to_action IDs must be unique")
    shape_counts = {
        shape: sum(shape in item["tags"] for item in items)
        for shape in (
            "micro_email",
            "short_email",
            "medium_email",
            "long_email",
            "very_long_email",
        )
    }
    minimums = {
        "micro_email": 5,
        "short_email": 10,
        "medium_email": 8,
        "long_email": 4,
        "very_long_email": 2,
    }
    if any(shape_counts[name] < minimum for name, minimum in minimums.items()):
        raise ValueError(f"email source-shape coverage is too narrow: {shape_counts}")
    field_counts = {
        count: sum(len(item["expected"]["value"]["fields"]) == count for item in items)
        for count in (2, 3, 4)
    }
    if field_counts != {2: 11, 3: 21, 4: 8}:
        raise ValueError(f"email field-count coverage drifted: {field_counts}")


def write_questions(path: Path) -> None:
    document = {
        "schema_version": 1,
        "benchmark": "email_to_action",
        "generated_by": GENERATOR,
        "seed": SEED,
        "items": generate_items(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic email triage items")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_questions(args.output)


if __name__ == "__main__":
    main()
