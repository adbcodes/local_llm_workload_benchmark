from __future__ import annotations

import json
from typing import Any


def _render_handover(title: str, entries: list[tuple[str, str, str, str, str]]) -> str:
    paragraphs = [
        f"sorry this is long — forwarding the working handover for {title}. This is the "
        "plain-text export from the shared channel, with the dated notes kept in order. A "
        "few replies were copied from email and some were dictated after calls, so the style "
        "and punctuation jump around. Old figures and tentative dates are still visible where "
        "someone replied to them; the line marked Follow-up is the current action for that topic. "
        "Nothing below is a new request by itself. I am pasting it because the attachments are "
        "spread across personal drives and the next person on duty needs enough context to find them."
    ]
    thread_notes = (
        "Status in the channel is still open; the last reaction was an acknowledgement, not approval.",
        "Evidence mentioned in replies: one phone photo and the vendor's earlier PDF, both retained for comparison.",
        "A later call clarified the wording but did not change the owner or the follow-up recorded here.",
        "The amount and date in the quoted email are provisional until the named owner posts a confirmation.",
        "Two people joined this thread late, so the decision was repeated in writing rather than left in call notes.",
        "Attachment names were preserved in the export; their filenames should not be treated as current instructions.",
        "One reply says 'looks fine', but the team left the task open because the dependency was not actually cleared.",
        "The tracker and chat used slightly different labels for this topic; the owner below is the current one.",
    )
    for index, (entry_date, owner, area, update, action) in enumerate(entries):
        paragraphs.append(
            f"[{entry_date}] #{area}\n"
            f"{owner}: {update}\n"
            f"Follow-up: {action}\n"
            f"Thread note: {thread_notes[index % len(thread_notes)]}"
        )
    paragraphs.append(
        "End of handover. Loose files mentioned elsewhere in the export include a scan "
        "with handwritten corrections, two spreadsheets whose totals do not agree, and a "
        "calendar screenshot taken before the latest reschedule. They are useful audit clues, "
        "not authority to send mail, create events, place orders, or change a live record. If a "
        "later message asks for one of those actions, use that message's exact names, dates, and "
        "limits instead of borrowing similar details from this handover. The group also asked that "
        "phone numbers, access codes, account references, and personal addresses stay inside the "
        "original restricted thread. I have removed the actual secrets from this paste but left the "
        "surrounding sentences so the sequence still makes sense. Please keep the dated order when "
        "replying; several earlier summaries caused confusion by grouping related topics together."
    )
    paragraphs.append(
        "For completeness, the export also contained routine acknowledgements such as 'seen', "
        "'will check after lunch', and 'please resend the photo'. I have not converted those into "
        "actions. There were several automatic calendar notices, mail-delivery footers, and missed-call "
        "alerts too; they explain gaps in the timestamps but do not settle any of the open points. One "
        "spreadsheet was labelled FINAL and then replaced the next morning, which is why the team now "
        "uses dates and owners instead of filenames when referring to a decision. If something below "
        "appears to contradict the final request after this paste, treat the final request as the new "
        "instruction and this material only as older conversation."
    )
    paragraphs.append(
        "No one expects a reply to the handover itself. It is included only because the chat export "
        "will be archived after the current work closes, and searching it later by a half-remembered "
        "name has already wasted time twice. The short request after this archive is the message that "
        "needs an answer now."
    )
    return "\n\n".join(paragraphs)


def _history(
    title: str,
    entries: list[tuple[str, str, str, str, str]],
    exchanges: list[tuple[str, dict[str, Any], dict[str, Any], str]],
    appendix: str = "",
) -> list[dict[str, str]]:
    if not 6 <= len(exchanges) <= 9:
        raise ValueError(f"{title} must contain six to nine prior tool calls")
    history: list[dict[str, str]] = []
    handover = _render_handover(title, entries)
    for index, (request, call, response, acknowledgement) in enumerate(exchanges):
        if index == 0:
            user_content = "\n\n".join(
                part for part in (handover, appendix.strip(), request) if part
            )
        else:
            user_content = request
        history.extend(
            [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": json.dumps(call, separators=(",", ":")),
                },
                {
                    "role": "user",
                    "content": "Tool response: "
                    + json.dumps(response, separators=(",", ":")),
                },
                {"role": "assistant", "content": acknowledgement},
            ]
        )
    word_count = sum(len(message["content"].split()) for message in history)
    if not 1500 <= word_count <= 3000:
        raise ValueError(
            f"{title} prior conversation has {word_count} words; expected 1500-3000"
        )
    return history


OFFICE_MOVE_ENTRIES = [
    ("03 Jul", "Nandita", "lease handover", "The landlord accepted 30 September as the handover date, but the basement storeroom stays excluded from our lease.", "legal must circulate the marked floor plan before anyone orders shelving."),
    ("04 Jul", "Kabir", "seat count", "The confirmed headcount is 86 regular seats plus eight hot desks, not the 110 shown in the broker deck.", "workplace should retain six accessible desks near the east lift and show them separately."),
    ("05 Jul", "Megha", "network room", "The smaller room beside reception overheats after noon and cannot take the primary rack without extra ventilation.", "facilities must obtain an electrical-load reading before the rack vendor revises its quote."),
    ("07 Jul", "Tenzin", "internet links", "The fibre provider can deliver one link by 12 September, while the diverse backup path needs another two weeks.", "IT should confirm whether temporary wireless failover is acceptable during the overlap."),
    ("08 Jul", "Raghav", "meeting rooms", "Two six-person rooms can be combined, but the movable partition has poor acoustic isolation in the demo video.", "procurement needs a live sample test before approving the partition fabric."),
    ("09 Jul", "Sonal", "access control", "Existing staff cards use a format the new controller reads, though visitor badges require a separate printer profile.", "security should test ten real cards and record failures rather than relying on the vendor spreadsheet."),
    ("11 Jul", "Irfan", "furniture reuse", "Forty-two chairs passed inspection, eleven need new gas lifts, and the orange lounge chairs will remain at the old office.", "the mover must label reusable chairs by floor instead of counting every item as new stock."),
    ("13 Jul", "Deepa", "pantry services", "The coffee contractor serves the new building but cannot install drainage at the originally proposed counter.", "design should shift the machine beside the sink and recheck the queue clearance."),
    ("15 Jul", "Nandita", "budget control", "Finance approved the base move budget but removed the speculative plants and second display wall from committed spend.", "owners must tag optional lines before the next purchase-order review."),
    ("17 Jul", "Kabir", "employee survey", "Most respondents prefer quieter focus areas over another large collaboration table, with support strongest among hybrid staff.", "workplace should publish the anonymised summary and avoid presenting the survey as a binding vote."),
    ("19 Jul", "Megha", "power backup", "The generator supports lighting and core network loads but not every desk socket or the pantry induction units.", "IT and facilities need one common list of circuits that remain live during an outage."),
    ("21 Jul", "Tenzin", "move waves", "Customer support asked to move on the second weekend because the first weekend overlaps its quarterly release.", "the sequence must keep at least one support pod operational at either site throughout Monday."),
    ("23 Jul", "Raghav", "signage", "The building directory allows only the registered company name, whereas internal wayfinding can use the shorter product brand.", "design should send two artwork files and stop mixing the names in vendor emails."),
    ("25 Jul", "Sonal", "visitor process", "Reception will issue temporary QR passes, but government-ID images must not be retained after the visit closes.", "security and privacy should agree on the deletion check before the pilot."),
    ("27 Jul", "Irfan", "packing labels", "The first label sample used team nicknames that the mover could not map to the approved seating plan.", "each crate label should include floor, zone, owner, and a sequence number."),
    ("29 Jul", "Deepa", "cleaning scope", "Deep cleaning covers floors and glass after fit-out, while monitor arms and personal equipment remain the employee's responsibility.", "workplace needs a plain-language checklist before the packing briefing."),
    ("31 Jul", "Nandita", "insurance", "The mover's policy excludes devices left loose in desk drawers and caps declared value per sealed crate.", "finance should decide which high-value test devices travel separately with staff."),
    ("02 Aug", "Kabir", "opening week", "The proposed welcome breakfast conflicts with the fire-drill window and the caterer cannot arrive before security opens loading access.", "the people team should move breakfast to Tuesday and confirm dietary counts on Friday."),
]

OFFICE_MOVE_EXCHANGES = [
    ("first can you find the latest floor-plan pdf called cedar-floorplan, only files changed after 2026-07-20", {"tool_call": "search_files", "arguments": {"name": "cedar-floorplan", "extension": "pdf", "modified_after": "2026-07-20"}}, {"files": [{"name": "cedar-floorplan", "extension": "pdf", "modified_at": "2026-07-31"}]}, "Found the revised floor plan dated 31 July."),
    ("thanks. check Bengaluru weather in celsius for the loading walkthrough tomorrow", {"tool_call": "get_weather", "arguments": {"location": "Bengaluru", "unit": "celsius"}}, {"temperature": 27, "condition": "light rain"}, "The forecast is 27°C with light rain, so the loading route may need cover."),
    ("set a 20 minute timer labelled call rack vendor", {"tool_call": "schedule_timer", "arguments": {"seconds": 1200, "label": "call rack vendor"}}, {"scheduled": True}, "The 20-minute rack-vendor timer is set."),
    ("please create Cabling walkthrough at 2026-08-04T11:30 Asia/Kolkata for megha@example.com and t.zin@example.com", {"tool_call": "create_calendar_event", "arguments": {"title": "Cabling walkthrough", "starts_at": "2026-08-04T11:30", "timezone": "Asia/Kolkata", "attendees": ["megha@example.com", "t.zin@example.com"]}}, {"event_id": "EV-440", "created": True}, "The cabling walkthrough was created for both attendees."),
    ("email nandita@example.com, subject Floor plan located, body I found the 31 July floor plan and the walkthrough is booked.", {"tool_call": "send_email", "arguments": {"to": "nandita@example.com", "subject": "Floor plan located", "body": "I found the 31 July floor plan and the walkthrough is booked."}}, {"message_id": "M-188", "sent": True}, "The update was sent to Nandita."),
    ("search active Workplace team users and return id and email", {"tool_call": "search_users", "arguments": {"filters": {"status": "active", "team": "Workplace"}, "fields": ["id", "email"]}}, {"users": [{"id": "U-18", "email": "kabir@example.com"}, {"id": "U-31", "email": "deepa@example.com"}]}, "Two active Workplace users matched."),
    ("one last check, query facilities-api warning logs 2026-08-02T09:00Z to 2026-08-02T10:00Z limit 20", {"tool_call": "query_logs", "arguments": {"service": "facilities-api", "start_at": "2026-08-02T09:00Z", "end_at": "2026-08-02T10:00Z", "severity": "warning", "limit": 20}}, {"count": 2, "codes": ["badge_sync_delay", "sensor_timeout"]}, "The window contains two warnings: a badge sync delay and a sensor timeout."),
]

WEDDING_ENTRIES = [
    ("12 Jun", "Masi Renu", "guest list", "The working list contains 164 adults and 23 children, but nine Bengaluru cousins have not confirmed travel.", "the family group should not give the caterer a final count until Sunday evening."),
    ("14 Jun", "Aman", "rail arrivals", "The Jaipur train reaches Ajmer at 06:40 and the Delhi group arrives by two separate trains around noon.", "transport should keep the morning minibus separate from the airport vehicles."),
    ("16 Jun", "Zoya", "room allocation", "Grandparents need lift access, the college friends are fine with triple sharing, and room 208 has a plumbing complaint.", "the hotel must send a new floor map before names are assigned."),
    ("18 Jun", "Masi Renu", "mehendi timing", "The artist needs four hours for the bridal party and asked for a bright room with two low tables.", "the schedule should start before the photographer's outdoor slot."),
    ("20 Jun", "Aman", "catering tasting", "The dal was approved, the paneer needs less salt, and the vendor's first dessert estimate assumed a larger portion size.", "the revised menu and price must be written into the banquet order."),
    ("22 Jun", "Zoya", "allergy notes", "Three guests reported nut allergies and one child needs a completely egg-free breakfast, not merely an eggless dessert.", "catering should label separate utensils and name the kitchen supervisor responsible."),
    ("24 Jun", "Masi Renu", "invitations", "Printed cards use the correct venue but the English insert gives the old baraat assembly time.", "only the corrected digital insert should go to outstation guests."),
    ("26 Jun", "Aman", "airport pickups", "The Mumbai flight may land after midnight and the traveller carrying costumes has oversized baggage.", "the late vehicle should be an SUV and the driver must receive the terminal number."),
    ("28 Jun", "Zoya", "decor", "Marigold strings are available locally, while the imported fabric in the mood board would arrive too close to setup.", "decor should price the local option and keep the stage colours unchanged."),
    ("30 Jun", "Masi Renu", "payments", "The photographer received a twenty-percent advance and the makeup artist received a flat booking fee.", "finance should store both receipts without treating the percentages as comparable."),
    ("02 Jul", "Aman", "music", "The venue sound limit applies after 10 pm and the dhol team needs a separate entry pass for its van.", "the event manager should move the last dance set earlier and register the vehicle."),
    ("04 Jul", "Zoya", "clothing delivery", "Two blouses need alteration and the courier tracking page still shows only label created.", "someone in Delhi must collect directly if the parcel does not move by Wednesday."),
    ("06 Jul", "Masi Renu", "ceremony supplies", "The priest confirmed the common items but asked the family to bring a specific brass vessel from home.", "the packing list should distinguish venue-provided supplies from family items."),
    ("08 Jul", "Aman", "childcare", "A supervised play corner is possible in the lounge, although the vendor cannot watch children under three.", "parents of toddlers need that limitation in the welcome note."),
    ("10 Jul", "Zoya", "weather plan", "The lawn drains slowly after heavy rain and the covered hall needs two hours to reset from dinner to ceremony seating.", "the venue should define a decision time for moving indoors."),
    ("12 Jul", "Masi Renu", "gift desk", "The family wants numbered receipt slips for envelopes but no public announcement of amounts or names.", "two trusted relatives should reconcile the slips privately after the reception."),
    ("14 Jul", "Aman", "departure transport", "Most guests leave after breakfast, while six elders need a later checkout before the evening train.", "hotel and transport lists must use the same departure version."),
    ("16 Jul", "Zoya", "emergency contacts", "The printed sheet contains an old driver number and one doctor's surname is misspelled.", "the corrected list should be shared as a phone-friendly image after both contacts verify it."),
]

WEDDING_EXCHANGES = [
    ("can you find current weather for Ajmer in celsius, we're checking whether the lawn backup is realistic", {"tool_call": "get_weather", "arguments": {"location": "Ajmer", "unit": "celsius"}}, {"temperature": 33, "condition": "scattered thunderstorms"}, "Ajmer is showing 33°C with scattered thunderstorms."),
    ("find coordinates for Pushkar Bus Stand so I can send the driver the right pin", {"tool_call": "geocode", "arguments": {"place": "Pushkar Bus Stand"}}, {"latitude": 26.4898, "longitude": 74.5511}, "I found the bus-stand coordinates."),
    ("book an Ajmer hotel search for 3 guests 2026-08-18 through 2026-08-20 max nightly 7500", {"tool_call": "book_hotel", "arguments": {"city": "Ajmer", "check_in": "2026-08-18", "check_out": "2026-08-20", "guests": 3, "max_nightly_rate": 7500}}, {"hotels": [{"name": "Lake Courtyard", "nightly_rate": 6900}]}, "One matching hotel was returned within the nightly cap."),
    ("convert 240 USD to INR at the current rate for the musician's equipment deposit", {"tool_call": "convert_currency", "arguments": {"amount": 240, "from_currency": "USD", "to_currency": "INR"}}, {"amount": 19920, "rate": 83.0}, "At the returned rate, USD 240 converts to INR 19,920."),
    ("create Transport list freeze at 2026-08-12T20:30 Asia/Kolkata for aman@example.com and zoya@example.com", {"tool_call": "create_calendar_event", "arguments": {"title": "Transport list freeze", "starts_at": "2026-08-12T20:30", "timezone": "Asia/Kolkata", "attendees": ["aman@example.com", "zoya@example.com"]}}, {"event_id": "EV-771", "created": True}, "The transport-list review is on both calendars."),
    ("send zoya@example.com subject Driver pin and body Pushkar Bus Stand pin is 26.4898, 74.5511.", {"tool_call": "send_email", "arguments": {"to": "zoya@example.com", "subject": "Driver pin", "body": "Pushkar Bus Stand pin is 26.4898, 74.5511."}}, {"message_id": "M-902", "sent": True}, "The driver pin was emailed to Zoya."),
    ("set a 45 minute timer called check alteration courier", {"tool_call": "schedule_timer", "arguments": {"seconds": 2700, "label": "check alteration courier"}}, {"scheduled": True}, "The 45-minute courier reminder is set."),
]

SOCIETY_ENTRIES = [
    ("05 May", "Ritu", "lift maintenance", "Lift B stopped twice between floors and the service engineer found dust on a door sensor rather than a motor fault.", "the committee should request the engineer's signed visit sheet before approving replacement parts."),
    ("07 May", "Harish", "water pressure", "Upper-floor pressure drops during the morning peak, while the basement gauge remains within the pump vendor's stated range.", "plumbing should log roof-tank levels at three fixed times for a full week."),
    ("09 May", "Farida", "parking labels", "Several old tenant stickers remain on cars after flats changed occupants, and visitor guards cannot read two faded labels.", "security should issue new stickers only after owners confirm vehicle numbers."),
    ("11 May", "Ritu", "waste pickup", "The wet-waste vehicle arrives before housekeeping finishes collection on two weekdays, causing bags to wait near the gate.", "the vendor and supervisor need one revised pickup window in writing."),
    ("13 May", "Harish", "terrace seepage", "Damp patches reappeared below the north parapet after rain, but the contractor's photograph shows only the repaired south edge.", "the next inspection must mark locations on a roof plan rather than use vague directions."),
    ("15 May", "Farida", "intercom", "Calls from tower C reach reception but not the side gate, and the installer says one switch profile was not copied.", "security should test a flat on every floor after the profile update."),
    ("17 May", "Ritu", "garden irrigation", "Two sprinkler heads water the driveway and a cracked hose near the children's area leaks after the timer stops.", "gardening should isolate that zone until replacement fittings arrive."),
    ("19 May", "Harish", "diesel account", "The generator log and fuel invoice differ by fourteen litres because one emergency refill was written in a guard notebook.", "treasury should reconcile the notebook entry before paying the invoice."),
    ("21 May", "Farida", "fire drill", "The first proposed date overlaps school exams and several elderly residents asked for advance notice of the alarm test.", "the safety team should publish a new date in English and Hindi."),
    ("23 May", "Ritu", "clubhouse booking", "A birthday request and yoga class overlap because one was written only in the paper diary.", "the office should transfer all confirmed diary bookings into the shared calendar."),
    ("25 May", "Harish", "facade work", "The contractor wants to suspend ropes from the east terrace, where residents currently keep movable planters.", "affected flats need a notice before the access inspection."),
    ("27 May", "Farida", "pet policy", "Complaints concern dogs off leash near the basement ramp, not pets using the main lift as an older summary claimed.", "the circular should address the actual location and avoid adding a new lift rule."),
    ("29 May", "Ritu", "visitor records", "The gate register contains full phone numbers visible to other visitors, which is unnecessary for package delivery.", "security should mask old pages and trial a privacy screen."),
    ("31 May", "Harish", "solar quote", "One vendor included battery storage and another priced only panels, so headline totals are not comparable.", "the energy group should request matching scopes before ranking quotes."),
    ("02 Jun", "Farida", "noise complaint", "Renovation drilling continued past the approved afternoon window, though the flat owner says workers misunderstood the Saturday timing.", "management should send the exact hours to both owner and contractor."),
    ("04 Jun", "Ritu", "rainwater pit", "Silt reduced the inlet flow and the cleaning vendor's estimate omits disposal of removed material.", "the final work order must state disposal and before-after photographs."),
    ("06 Jun", "Harish", "accounting archive", "The auditor asked for April invoices and approvals together, but files are split between scans and email attachments.", "treasury should create one indexed folder without renaming signed originals."),
    ("08 Jun", "Farida", "resident meeting", "Sunday morning suits most committee members, while the ground-floor hall is unavailable until eleven after a class.", "the agenda should start at 11:30 and reserve time for unresolved water issues."),
]

SOCIETY_EXCHANGES = [
    ("search for files named april-maintenance with pdf extension modified after 2026-05-01", {"tool_call": "search_files", "arguments": {"name": "april-maintenance", "extension": "pdf", "modified_after": "2026-05-01"}}, {"files": [{"name": "april-maintenance", "extension": "pdf", "modified_at": "2026-06-06"}]}, "The matching maintenance PDF was modified on 6 June."),
    ("look up wireless door sensors under 3200 and only items in stock", {"tool_call": "search_products", "arguments": {"query": "wireless door sensors", "max_price": 3200, "in_stock_only": True}}, {"products": [{"name": "EntrySense Mini", "price": 2899, "in_stock": True}]}, "One in-stock sensor fits the stated cap."),
    ("create North terrace inspection on 2026-08-09T08:30 Asia/Kolkata for ritu@example.com and harish@example.com", {"tool_call": "create_calendar_event", "arguments": {"title": "North terrace inspection", "starts_at": "2026-08-09T08:30", "timezone": "Asia/Kolkata", "attendees": ["ritu@example.com", "harish@example.com"]}}, {"event_id": "EV-319", "created": True}, "The north-terrace inspection has been added."),
    ("send farida@example.com subject Fire drill translation and body Please review the Hindi fire-drill notice before circulation.", {"tool_call": "send_email", "arguments": {"to": "farida@example.com", "subject": "Fire drill translation", "body": "Please review the Hindi fire-drill notice before circulation."}}, {"message_id": "M-601", "sent": True}, "Farida received the fire-drill review request."),
    ("check Pune weather in celsius for the terrace inspection day", {"tool_call": "get_weather", "arguments": {"location": "Pune", "unit": "celsius"}}, {"temperature": 25, "condition": "rain"}, "The returned Pune forecast is 25°C with rain."),
    ("set a 30 minute timer labelled call lift service", {"tool_call": "schedule_timer", "arguments": {"seconds": 1800, "label": "call lift service"}}, {"scheduled": True}, "The lift-service reminder is set for 30 minutes."),
    ("search active Security team users, fields name then email", {"tool_call": "search_users", "arguments": {"filters": {"status": "active", "team": "Security"}, "fields": ["name", "email"]}}, {"users": [{"name": "Mahesh Rao", "email": "mahesh@example.com"}]}, "One active Security user matched."),
    ("query society-portal warning logs from 2026-08-02T06:00Z to 2026-08-02T07:00Z, limit 15", {"tool_call": "query_logs", "arguments": {"service": "society-portal", "start_at": "2026-08-02T06:00Z", "end_at": "2026-08-02T07:00Z", "severity": "warning", "limit": 15}}, {"count": 1, "codes": ["visitor_sync_lag"]}, "That hour contains one visitor-sync warning."),
]

COMMUNITY_ENTRIES = [
    ("01 Jul", "Lalita", "meal count", "Weekday lunch demand averages 118 portions but rises sharply on pension-distribution days at the ward office.", "the kitchen should keep the base plan and label surge stock separately."),
    ("03 Jul", "Sameer", "rice supplier", "The cooperative can deliver broken rice at the old rate, while the preferred sona masoori quote increased after transport charges.", "procurement should compare cooked yield rather than bag price alone."),
    ("05 Jul", "Jose", "volunteer rota", "College volunteers are available Saturdays but not during their internal exams in the second week of August.", "the rota needs named backup cooks for that week."),
    ("07 Jul", "Lalita", "gas safety", "One burner has a stiff valve and the technician asked staff not to force it before the scheduled visit.", "the morning team should mark the burner out of service."),
    ("09 Jul", "Sameer", "delivery route", "Roadwork closes the usual lane after 10 am and the smaller van can enter through the market side.", "dispatch should move the first drop earlier and give the driver the alternate pin."),
    ("11 Jul", "Jose", "nutrition note", "The clinic requested lower-salt meals twice a week, but did not ask for separate menus every day.", "the dietitian should confirm portions and identify the correct recipients privately."),
    ("13 Jul", "Lalita", "container returns", "Thirty-one steel carriers remain with partner centres and some return labels have washed off.", "coordinators should count by embossed number rather than marker ink."),
    ("15 Jul", "Sameer", "donation receipts", "Two bank transfers arrived without donor phone numbers, though the reference text identifies the campaign.", "accounts should acknowledge by email where available and avoid guessing contacts."),
    ("17 Jul", "Jose", "monsoon storage", "The rear storeroom wall feels damp and cardboard cartons on the floor absorbed moisture.", "dry goods should move to raised pallets before the next delivery."),
    ("19 Jul", "Lalita", "school partnership", "The nearby school can lend tables after classes but needs them returned before Monday assembly.", "the event plan must include Sunday-night transport rather than assume Monday pickup."),
    ("21 Jul", "Sameer", "water filter", "The outlet test passed, while the service indicator still shows an old cartridge date.", "the vendor should reset the indicator and leave the test slip."),
    ("23 Jul", "Jose", "festival demand", "Last year's meal count included a separate evening distribution that will not run this year.", "forecasting should use the lunch count only and keep a modest walk-in buffer."),
    ("25 Jul", "Lalita", "packaging", "Compostable bowls soften when hot dal sits longer than twenty minutes, according to the trial.", "packing should fill dal last and avoid pre-packing the entire batch."),
    ("27 Jul", "Sameer", "cash purchases", "The vegetable market issues handwritten slips with inconsistent names but the wholesaler stamp is readable.", "buyers should photograph slips immediately and write the kitchen date on the back."),
    ("29 Jul", "Jose", "first-aid kit", "The burn dressing is near expiry and the list lacks oral rehydration sachets for outdoor distribution.", "the safety volunteer should replace both before the Sunday event."),
    ("31 Jul", "Lalita", "community notice", "The Hindi poster has the right venue but the Marathi line gives an earlier serving time.", "the language volunteers must review one final image before printing."),
    ("02 Aug", "Sameer", "cold storage", "The borrowed refrigerator holds dairy safely but cannot fit the prepared vegetable crates as originally assumed.", "the prep schedule should keep cut vegetables in the kitchen cold room."),
    ("03 Aug", "Jose", "closing checklist", "The last team left one regulator open and forgot to upload the remaining-portion count.", "shift leads should sign the gas and inventory checks separately."),
]

COMMUNITY_EXCHANGES = [
    ("check current Nagpur weather in celsius for Sunday's outdoor serving", {"tool_call": "get_weather", "arguments": {"location": "Nagpur", "unit": "celsius"}}, {"temperature": 29, "condition": "overcast"}, "Nagpur is showing 29°C and overcast conditions."),
    ("find coordinates for Cotton Market Nagpur, that's where the small van should enter", {"tool_call": "geocode", "arguments": {"place": "Cotton Market Nagpur"}}, {"latitude": 21.1452, "longitude": 79.0882}, "I found coordinates for Cotton Market in Nagpur."),
    ("search for insulated food carriers under 9000, in stock only", {"tool_call": "search_products", "arguments": {"query": "insulated food carriers", "max_price": 9000, "in_stock_only": True}}, {"products": [{"name": "ServeSafe 35L", "price": 8240, "in_stock": True}]}, "One in-stock insulated carrier is within budget."),
    ("create Container count at 2026-08-08T17:00 Asia/Kolkata for lalita@example.com and jose@example.com", {"tool_call": "create_calendar_event", "arguments": {"title": "Container count", "starts_at": "2026-08-08T17:00", "timezone": "Asia/Kolkata", "attendees": ["lalita@example.com", "jose@example.com"]}}, {"event_id": "EV-845", "created": True}, "The container count was added for Lalita and Jose."),
    ("email sameer@example.com subject Market entry pin and body Use Cotton Market pin 21.1452, 79.0882 for the small van.", {"tool_call": "send_email", "arguments": {"to": "sameer@example.com", "subject": "Market entry pin", "body": "Use Cotton Market pin 21.1452, 79.0882 for the small van."}}, {"message_id": "M-314", "sent": True}, "Sameer received the alternate market-entry pin."),
    ("convert 75 USD to INR with the current rate, donor needs an approximate local value", {"tool_call": "convert_currency", "arguments": {"amount": 75, "from_currency": "USD", "to_currency": "INR"}}, {"amount": 6225, "rate": 83.0}, "The returned conversion is INR 6,225."),
    ("set a fifteen minute timer labelled check gas knobs", {"tool_call": "schedule_timer", "arguments": {"seconds": 900, "label": "check gas knobs"}}, {"scheduled": True}, "The gas-knob check timer is set for fifteen minutes."),
    ("find active Kitchen Operations users and return name and email", {"tool_call": "search_users", "arguments": {"filters": {"status": "active", "team": "Kitchen Operations"}, "fields": ["name", "email"]}}, {"users": [{"name": "Lalita Rao", "email": "lalita@example.com"}, {"name": "Jose Mathew", "email": "jose@example.com"}]}, "Two active Kitchen Operations users matched."),
    ("query meal-dispatch error logs 2026-08-03T05:00Z through 2026-08-03T06:00Z limit 10", {"tool_call": "query_logs", "arguments": {"service": "meal-dispatch", "start_at": "2026-08-03T05:00Z", "end_at": "2026-08-03T06:00Z", "severity": "error", "limit": 10}}, {"count": 1, "codes": ["route_manifest_retry"]}, "One route-manifest retry error was returned."),
]


WEDDING_ARCHIVE_APPENDIX = """Additional notes copied from the family spreadsheet discussion:

The hotel initially circulated a room list with abbreviated names, then replaced it after two guests with the same first name were assigned one room. The corrected version uses phone suffixes only in the private coordinator copy. A decorative “final room list” image in the public chat is older and should not be used for transport counts. The banquet manager also confirmed that vendor meals are counted separately from guest plates; an earlier total combined them and made the catering estimate look eight people higher.

One aunt suggested moving the mehendi outdoors after seeing a sunny forecast screenshot, but the venue did not approve that change and the artist still requires the bright indoor room. Similarly, a voice note proposed collecting every guest's identity document in advance. The hotel later clarified that adults can show identification at check-in, so no family volunteer should gather those documents. These details remain in the archive to explain the later corrections; they are not open requests."""

SOCIETY_ARCHIVE_APPENDIX = """The exported committee thread also contains a long reconciliation of maintenance complaints. It begins with three photographs labelled “Lift A”, although the service engineer later identified the doors as Lift B. The filenames were never corrected. The signed visit sheet, when received, should control over those labels. Residents discussed replacing a motor, a controller, and a sensor in the same thread, but only sensor cleaning has actually occurred; no part order is approved.

Water-pressure notes are similarly mixed. A plumber's morning reading was copied beside an evening roof-tank level, creating a graph that looked like a single test. Harish marked the graph unsuitable and asked for seven days of fixed-time readings. Nobody has authorised a pump change. A quotation with a large total includes both pump work and unrelated terrace waterproofing, so the headline amount is not comparable with the plumbing-only estimate.

The fire-drill translation went through two drafts. The Hindi draft has the correct assembly area but an old alarm time; the English draft has the new time. Farida is combining them before circulation. An automatic mail footer says “approved by management,” but that refers to the translation platform account, not the notice. The actual committee approval remains in the minutes.

Finally, the visitor-record discussion produced a temporary paper screen at the gate. Guards can still see full numbers when turning pages, so the privacy action is not closed. The planned digital trial is a proposal only and has no purchase approval. These archived distinctions matter for the committee record but do not ask the assistant to contact vendors or modify live bookings."""

COMMUNITY_ARCHIVE_APPENDIX = """Older kitchen-channel messages retained for audit context:

The meal-count worksheet has three tabs. “Daily base” is the working lunch forecast; “festival 2025” contains last year's evening distribution and must not be added to this year's requirement; “walk-ins” is a manual tally with two missing dates. Lalita asked volunteers to retain the blanks rather than invent zeroes. A screenshot in the chat shows 176 portions, but that was a stress-test quantity used to size pots, not an approved production count.

Rice comparisons also need context. One cooperative quote is per 25 kg bag and another is per 50 kg bag with delivery included. Sameer calculated cooked yield from a single trial and clearly labelled it provisional. No supplier was selected. An enthusiastic reply saying “book this” referred to a tasting visit, not a purchase order. Accounts will compare written scopes after the cooperative confirms transport charges.

During the gas-safety discussion, a volunteer suggested using pliers on the stiff burner valve. Jose immediately marked the burner out of service and the technician repeated that nobody should force it. The unsafe suggestion remains visible only because exports preserve deleted-message placeholders. The maintenance appointment, not the suggestion, is the current path.

The nutrition note includes sensitive recipient names in the restricted clinic attachment. They have been removed from this paste. The operational requirement is simply that the dietitian privately confirm the correct lower-salt portions twice weekly. Public labels should describe the meal, not identify a health condition or recipient.

Container counts changed as centres returned stock. A photo caption says thirty-four missing, a later embossed-number count says thirty-one, and three carriers then arrived without readable marker labels. The embossed numbers are authoritative. The current follow-up is still a fresh count before the Sunday event, not a replacement purchase.

For the monsoon storage issue, dry goods moved onto borrowed pallets on 30 July. The wall inspection is pending, so the move is containment rather than proof the leak is repaired. Two cartons were discarded after packaging became soft; their value appears in an expense draft that Finance has not approved.

The school offered tables subject to Sunday-night return. A volunteer calendar entry accidentally used Monday morning, then was cancelled. The transport owner has not posted a vehicle number. Nothing in that calendar history confirms logistics are complete.

Language volunteers corrected the Marathi serving time in the community poster. The chat preview still caches the old image on some phones, while the print-ready file has a dated filename and reviewer initials. Printing waits for one final review. Routine reactions and forwarded posters in this archive are background, not a request to send or publish anything now."""


CONTEXT_PRESSURE_HISTORIES = {
    "tool_use_011": _history(
        "the Cedar office move",
        OFFICE_MOVE_ENTRIES,
        OFFICE_MOVE_EXCHANGES[:6],
    ),
    "tool_use_012": _history(
        "the Ajmer wedding logistics thread",
        WEDDING_ENTRIES,
        WEDDING_EXCHANGES,
        WEDDING_ARCHIVE_APPENDIX,
    ),
    "tool_use_015": _history(
        "Lakeview Residency committee work",
        SOCIETY_ENTRIES,
        SOCIETY_EXCHANGES,
        SOCIETY_ARCHIVE_APPENDIX,
    ),
    "tool_use_016": _history(
        "the Nagpur community kitchen rota",
        COMMUNITY_ENTRIES,
        COMMUNITY_EXCHANGES,
        COMMUNITY_ARCHIVE_APPENDIX,
    ),
}


CONTEXT_PRESSURE_FINAL_REQUESTS = {
    "tool_use_011": (
        "anyway, the actual thing I need now: add a Design review for 7 Aug 2026 "
        "at 2:30 pm IST, with ana@example.com and dev@example.com"
    ),
    "tool_use_012": (
        "sorry, unrelated to all that — email sam@example.com. subject: Invoice "
        "copy. body: Attached is the requested invoice."
    ),
    "tool_use_015": (
        "need the quarterly-summary PDFs modified after 1 June 2026. filename is "
        "quarterly-summary, extension pdf"
    ),
    "tool_use_016": (
        "ok final ask: find a Bengaluru hotel for 2 people, 10 to 13 Sep 2026, "
        "max ₹6000 a night"
    ),
}
