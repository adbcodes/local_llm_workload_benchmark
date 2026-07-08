"""Fresh long-form sources for the grounded-compression generator.

The dialogue and meeting corpora named in style_pointer informed only cadence,
turn-taking, and artifact shape. Every source, entity, fact, and reference summary
below was written for this benchmark.
"""

NEW_SUMMARY_SPECS = [
    {
        "item_id": "summary_chat_handoff_001",
        "subcategory": "chat_history_handoff",
        "difficulty": "easy",
        "task": "Compress this conversation for the next support assistant in no more than 95 words. The audience is another model taking over the case. Capture settled facts, decisions, completed steps, and open questions; do not repeat superseded details.",
        "source_kind": "chat_history",
        "style_pointer": "dialogsum",
        "source": """User: hi, I need help with a duplicate train booking. The app froze after I paid, so I tried again. Now I have two tickets for the same Bengaluru to Chennai train next Friday.

Assistant: I can help organise the facts and the refund steps. Please share the two booking references, passenger names, travel date, and whether either ticket has been cancelled. Do not send card details or OTPs.

User: First one is IR-482771 and second is IR-482809. Both are for Nandita Rao and Kiran Rao, 7 August, train 12658. Neither is cancelled. The first email arrived late, which is why I booked again.

Assistant: Understood: same two passengers, train, and date, with separate confirmed references. Which ticket do you want to keep, and do the coach or seat assignments differ?

User: Keep IR-482771. It has lower berths in B2. The other one is B5 and cost INR 3,460 including the booking fee. The first was INR 3,412. Please don't cancel the good one by mistake.

Assistant: Decision recorded: keep IR-482771 and seek cancellation or duplicate-charge review only for IR-482809. Before cancellation, check whether the operator labels the second booking as fully refundable and whether a departure-time cutoff applies.

User: The app says cancellation fee INR 240 per passenger if I cancel now. But support chat earlier said they might waive it for duplicate bookings caused by a payment-status delay. I didn't save that chat.

Assistant: Then it is safer not to cancel yet. We should ask support to treat IR-482809 as a duplicate and request a fee waiver, while explicitly stating that IR-482771 must remain active. Useful evidence is the payment timestamps and delayed confirmation email.

User: Bank timestamps are 18:42 and 18:49 yesterday. Confirmation for the first ticket came at 18:51, after the second payment. I have screenshots of both transactions and both emails.

Assistant: That timeline supports the request: first payment 18:42, second payment 18:49, first confirmation 18:51. Attach redacted screenshots showing times and references, but hide account balances and unrelated transactions.

User: Can you draft what I should send? Also, the travel date is Friday 7 August 2026, not this coming Friday. I said next Friday loosely.

Assistant: Thanks for correcting the date. Suggested message: “Please review booking IR-482809 as a duplicate of IR-482771 for Nandita Rao and Kiran Rao on train 12658, 7 August 2026. The first payment completed at 18:42, but its confirmation arrived only at 18:51, after I made a second payment at 18:49. Please keep IR-482771 active, cancel only IR-482809, and waive the duplicate-booking cancellation fees.”

User: Good. Where do I send it? The app has “cancel ticket” and “raise issue,” and the email receipt says replies aren't monitored.

Assistant: Use “raise issue,” not the automatic cancellation button, because you are requesting a discretionary fee waiver. Choose the closest payment or duplicate-booking category. Include both references and the exact keep/cancel instruction in the first paragraph.

User: I raised issue CASE-19308 and uploaded three screenshots. It says response within 48 hours. I did not cancel anything. Should I also call?

Assistant: The case is now the primary route. Calling is optional unless the 48-hour window would pass close to the operator's cancellation cutoff. Do you know the last time for cancelling with any refund?

User: Not yet. The ticket only says chart preparation rules apply. Departure is 22:40 on 7 August. I can check the operator policy tomorrow. Please leave that as an open question.

Assistant: Noted. Current state: case CASE-19308 filed with evidence; no ticket cancelled; IR-482771 must be protected. Open question: determine the refund cutoff and call only if support timing threatens it.

User: one more detail, support form auto-filled my old phone number ending 1180. My current number ends 7741. I added the current one in the description, but there was no edit button after submission.

Assistant: Record the contact mismatch. If support replies by email, no action is needed. If the case status shows an attempted phone contact or no response, add a follow-up noting that 7741 is current and 1180 is obsolete. Do not open a second case unless the first cannot be updated.""",
        "expected": "Duplicate bookings cover Nandita and Kiran Rao on train 12658, 7 August 2026. Keep IR-482771; request fee-waived cancellation only for IR-482809. Payments were 18:42 and 18:49; the first confirmation arrived 18:51. CASE-19308 was filed with three redacted screenshots; nothing was cancelled. Support estimates 48 hours. Open: check the refund cutoff and call if delay risks it. Case auto-filled obsolete phone ending 1180; current number ends 7741, already noted in the description.",
        "max_words": 95,
    },
    {
        "item_id": "summary_chat_handoff_002",
        "subcategory": "chat_history_handoff",
        "difficulty": "easy",
        "task": "Create a handoff note for another assistant in no more than 100 words. The audience is a model continuing this home-network troubleshooting chat. Preserve observations, ruled-out causes, changes already made, the agreed next test, and unresolved questions.",
        "source_kind": "chat_history",
        "style_pointer": "dialogsum",
        "source": """User: hey, can we continue fixing the wifi drops? Every few hours video calls freeze, but my phone still shows full bars. I restarted everything yesterday and it came back.

Assistant: Yes. Full Wi-Fi bars only show the link to the access point, not internet health. Is the problem on one device or several, and is your connection fibre, cable, or mobile broadband?

User: Fibre from MetroWave. It happens on my work MacBook and my wife's Windows laptop at the same time. Phones seem okay, though we aren't usually calling on them. Router is theirs, model MW-RX6.

Assistant: Simultaneous failure on two laptops makes a single-device issue less likely. During the next drop, we should compare a local router ping with an external IP ping. Are the laptops on the same Wi-Fi band, and is any Ethernet device available?

User: Both were on the network called Home-5G. The TV is wired and Netflix kept playing last time, or at least it didn't visibly stop. I have a USB Ethernet adapter for the Mac.

Assistant: That points tentatively toward the 5 GHz wireless path, but buffered video means the TV is not decisive. Please avoid changing several settings at once. First, connect the Mac by Ethernet for one work session while leaving the Windows laptop on Home-5G.

User: Did that this morning. At 11:17 the Windows call froze for about 40 seconds. The wired Mac call did not freeze. Router log has “radar detected, channel switch 100 -> 44” at 11:16:52.

Assistant: That is useful evidence. Channel 100 is a DFS channel and radar detection can force a switch, disrupting 5 GHz clients. The wired connection staying stable supports a Wi-Fi-specific cause. It does not yet prove every historical drop had the same cause.

User: Can I just set channel 44? I found the wireless page. It currently says automatic, width 160 MHz, country India. There are separate settings for 2.4 and 5.

Assistant: A controlled change is reasonable: set only the 5 GHz channel to 44 and reduce width to 80 MHz if the router permits. Leave 2.4 GHz unchanged. Record the old settings so they can be restored. Do not change transmit power, DNS, or security mode in the same test.

User: I set channel 44 but forgot and also changed width to 80, which you did say was okay. Saved at 12:03. Router rebooted itself. Old settings were auto and 160. Everything reconnected.

Assistant: Those two changes belong to the same planned 5 GHz test, so that is fine. We need at least a day of normal use. Note any drop time and check whether the log shows a channel switch or WAN reconnect.

User: At 16:20 the internet disappeared on all devices, including wired TV. Log says WAN DHCP renew failed, then connected 28 seconds later. So that's a different problem, right?

Assistant: Correct. The 16:20 event affected Ethernet and shows a WAN renewal failure, distinct from the 11:17 wireless channel switch. We may have two failure modes. The fixed channel cannot prevent an ISP-side or router-WAN interruption.

User: MetroWave support says optical levels look normal and asked me to factory reset. I don't want to lose all our device names. They gave ticket MW-552901.

Assistant: Do not factory reset yet. Save or photograph the current configuration first, and collect timestamps for another WAN event. Ask MetroWave whether MW-RX6 firmware 3.8.12 has known DHCP renewal issues and whether they can see lease errors for ticket MW-552901.

User: Firmware page does say 3.8.12, last updated automatically 18 June. There is an export config button. I exported a file but haven't opened it.

Assistant: Keep the export securely; it may include sensitive Wi-Fi settings, so do not upload it here. Agreed next test: keep channel 44/80 MHz, watch for 24 hours, and classify each event as Wi-Fi-only or whole-network using Ethernet plus logs.

User: Should I split the network names? Both bands currently share “Home,” but earlier I called it Home-5G because the laptop details showed 5 GHz.

Assistant: Thanks for correcting that. The SSID is actually shared. Splitting names could help later, but it would add another variable now. Leave band steering and SSID names unchanged during this observation period.

User: okay. Open questions are whether firmware 3.8.12 has the DHCP bug and whether channel 44 stops the call freezes. I will update the MetroWave ticket, not reset yet.

Assistant: Exactly. Also retain the two known timestamps: 11:17 Wi-Fi-only after DFS radar detection, and 16:20 whole-network after WAN DHCP renewal failure. Those distinguish the issues for the next troubleshooting step.""",
        "expected": "MetroWave fibre uses an MW-RX6 on firmware 3.8.12 with one shared SSID. At 11:17, both 5 GHz laptops had a DFS channel switch; Ethernet stayed stable. The 5 GHz setting changed from auto/160 MHz to channel 44/80 MHz. At 16:20, a separate WAN DHCP renewal failure affected wired and wireless devices for 28 seconds. Do not factory-reset yet; config was exported. Observe 24 hours, classify drops with Ethernet and logs, and update ticket MW-552901. Open: whether channel 44 stops freezes and whether firmware has a DHCP bug.",
        "max_words": 100,
    },
    {
        "item_id": "summary_chat_handoff_003",
        "subcategory": "chat_history_handoff",
        "difficulty": "medium",
        "task": "Compress this planning chat for the next assistant in no more than 110 words. The audience is another model helping the user finish a community workshop plan. Separate confirmed decisions from tentative ideas and list remaining questions.",
        "source_kind": "chat_history",
        "style_pointer": "dialogsum",
        "source": """User: I volunteered to organise a basic smartphone workshop for our residents' association. Mostly older neighbours. I have too many notes and need a workable plan.

Assistant: Let us turn the notes into decisions, constraints, and open questions. What date, venue, attendance, and topics are already confirmed?

User: Sunday 16 August, community room at Silver Oak Apartments, 10:30 to noon. The room seats 32 but I think 18 to 25 people will come. Topics requested were WhatsApp photos, UPI safety, and booking doctor appointments.

Assistant: Confirmed: 90 minutes, capacity 32, expected 18–25, and three requested topics. Do participants use Android, iPhone, or both? That affects demonstrations.

User: Mostly Android. Registration form has 14 Android, 3 iPhone, and 4 people skipped the question. Also two people wrote that they use feature phones, which probably means this isn't for them.

Assistant: Do not exclude them yet; they may bring a family member's phone or want scam-awareness guidance. We can state that hands-on steps require a smartphone and ask what device they will bring.

User: Good. I asked my niece Tara to help and she said yes. Ramesh uncle might help with Hindi explanations but hasn't confirmed. I can teach in English and some Marathi, not Hindi.

Assistant: Tara is confirmed; Ramesh is tentative. With up to 25 people, two helpers plus you would be comfortable. Language coverage remains a risk if he cannot attend.

User: The association has a projector, but the HDMI cable sometimes cuts out. Wi-Fi is weak in the room. I was going to use live apps, but maybe screenshots are safer.

Assistant: Plan offline-first: prepare large screenshots or short recordings, bring a tested spare HDMI cable, and use mobile data only for one controlled demo. Avoid asking participants to perform real UPI transfers or expose medical details.

User: Yes, definitely no real payment. Could we use a fake payment screenshot? For the doctor booking, our local clinic uses an app but I don't want to advertise one company.

Assistant: Use a fictional payment screen and demonstrate verification cues rather than a transaction. For appointments, teach the generic sequence—find the official provider, check date and patient name, review before confirming—and mention that interfaces vary.

User: I need a schedule. I thought 15 minutes introduction, 25 WhatsApp, 25 UPI, 20 appointments, 5 questions. That totals 90, but five minutes for questions is too little.

Assistant: One option is 10 introduction and accessibility setup, 20 WhatsApp, 25 UPI safety, 20 appointments, and 15 questions. Keep a five-minute buffer inside each demo by limiting it to two key tasks.

User: Let's use that. WhatsApp should cover sending one photo and stopping automatic media download. UPI should cover checking the receiver name, collect requests, and never sharing PIN or OTP. Appointments should cover official links and cancellation details.

Assistant: Those learning objectives are now decisions. We should print a one-page takeaway in large type. Does the association have a printing budget?

User: Treasurer said up to INR 1,500 for printing and tea, but I don't know how much tea costs. Print shop quoted INR 6 per double-sided colour sheet or INR 2 black and white. Need 30 copies plus a few extras.

Assistant: Thirty-five black-and-white copies cost INR 70; colour would cost INR 210. Both fit, but tea cost is unknown. Use colour only if colour cues are essential; otherwise high-contrast black and white is more accessible.

User: High-contrast black and white. We also need consent if Tara records a two-minute recap. Actually, let's not record participants at all. She can photograph the empty setup.

Assistant: Decision: no participant recording; an empty-room setup photo is acceptable if the association permits it. That avoids a consent workflow.

User: Registration currently asks name and flat number. Should I ask phone model and language preference too? Deadline can be 12 August so I can print.

Assistant: Add device type and preferred explanation language, but make both optional. Set registration deadline to 12 August. Avoid collecting phone numbers unless there is a clear need.

User: We need a reminder message. Say bring a charged phone, but don't ask them to install apps. Also maybe tell feature-phone users the safety section is still useful.

Assistant: Agreed. Reminder should include date, time, room, charged phone, no advance installs, and that scam-safety guidance is useful even without a smartphone.

User: Remaining things: confirm Ramesh, test projector and spare cable, price tea, get permission for the empty room photo, and write the handout. Anything else?

Assistant: Also verify accessibility: enough chairs with clear aisles, readable projection from the back, and a microphone only if the room needs one. Confirm who brings extension boards, because participants may arrive with low batteries despite the reminder.

User: Add extension boards as open. Tara has one, I may borrow another from the office. Final attendance won't be known until 12 August.

Assistant: Noted. The plan is stable enough for a handoff, with staffing, tea, equipment, photo permission, accessibility, extension boards, and final attendance still open.""",
        "expected": "Confirmed: smartphone workshop, Silver Oak community room, 16 August, 10:30–12:00, capacity 32; 21 registered so far. Tara assists; Ramesh’s Hindi help is tentative. Agenda: 10-minute setup, 20 WhatsApp, 25 UPI safety, 20 appointment booking, 15 Q&A. Use offline fictional demos, no real payments or participant recording, and 35 high-contrast handouts. Registration closes 12 August and adds optional device/language fields. Open: Ramesh, tea price, projector/cable test, empty-room photo permission, accessibility, extension boards, handout, and final attendance.",
        "max_words": 110,
    },
    {
        "item_id": "summary_chat_handoff_004",
        "subcategory": "chat_history_handoff",
        "difficulty": "hard",
        "task": "Write a compact case handoff for another assistant in no more than 115 words. The audience is a model continuing this freelance invoice dispute. Preserve chronology, amounts, evidence, decisions, and unresolved risks; distinguish the client's claim from verified facts.",
        "source_kind": "chat_history",
        "style_pointer": "dialogsum",
        "source": """User: I have a messy freelance payment dispute and need the next steps without making it worse. Client says my final invoice is outside scope, but they already use the work.

Assistant: I can organise the record, not provide legal advice. Please give the contract terms, invoice amount, work delivered, approvals, and what the client specifically disputes.

User: Contract was INR 2,40,000 for a dashboard redesign: 40% advance, 30% after prototype approval, 30% on final handoff. They paid first two instalments. Final invoice is INR 72,000, number NR-026. Due 30 June.

Assistant: So INR 1,68,000 has been paid and INR 72,000 is invoiced. What evidence shows prototype approval and final handoff, and was extra work requested?

User: Prototype approved by email 8 May. Final Figma and assets sent 12 June. Product lead replied “looks good, we’ll ship this.” It went live 18 June. Then finance said the mobile empty states were extra and asked for a revised invoice excluding them.

Assistant: The live deployment and written response support delivery, but we should not interpret them as a legal conclusion. Were mobile empty states in the signed scope or added later?

User: Scope says “responsive dashboard screens for desktop and mobile, including loading, error and empty states.” There were 14 screens in appendix A. I delivered 14. Later they asked for two tablet variants in Slack. I did those too and never charged extra.

Assistant: That wording appears relevant: mobile empty states are expressly listed, while the tablet variants were later additions not billed. Preserve the signed appendix, the 8 May approval, 12 June delivery, and 18 June live evidence.

User: Finance email on 2 July says they will pay INR 48,000 as “undisputed portion” if I issue a credit note for INR 24,000. They want the credit note by tomorrow. I haven't replied.

Assistant: Do not issue a credit note just to acknowledge receipt unless you intend to reduce the invoice. A neutral response can ask them to identify the contractual basis for INR 24,000 and state that NR-026 remains unchanged while you review.

User: Product lead called me and said finance is doing this to every contractor because budgets froze. That's verbal, no recording. He asked me not to quote him.

Assistant: Treat that as an unverified private claim, not evidence to include in a demand. Record it privately as context and do not attribute it without permission.

User: I also found that the purchase order says INR 2,16,000, not 2,40,000. They created it after signing and I did not notice. Payments so far match 40% and 30% of 2,40,000, though: 96,000 and 72,000.

Assistant: That discrepancy is important. The signed contract total and payment pattern support INR 2,40,000, but the lower PO may create an accounts-payable block. Ask whether the PO needs correction rather than assuming the contract changed.

User: Can I accept the 48,000 and still pursue 24,000? Their email says payment requires “full and final settlement” in the credit-note description.

Assistant: That language creates a meaningful risk. Do not tell them it is acceptable without professional advice. You can decline to issue the requested credit note for now and ask whether they can pay the undisputed amount without a full-and-final condition.

User: I need cash, but I don't want to waive the rest. I can wait three days to speak to an accountant or lawyer. Is there a non-angry reply?

Assistant: Yes: acknowledge the offer, state that you cannot issue a INR 24,000 credit note or accept a full-and-final condition at this time, ask for payment of INR 48,000 without prejudice to the remaining balance, and request the calculation plus PO correction.

User: Drafted: “Thanks, I acknowledge the proposed 48k. Invoice NR-026 remains 72k based on the signed scope. I can't issue the requested credit note or mark this full and final. Please release the undisputed 48k without that condition and share the 24k calculation. Also please correct PO 7731 to the signed total.” Too aggressive?

Assistant: It is factual and restrained. Add the contract and delivery references, avoid saying they acted in bad faith, and give a reasonable reply date. What date do you want?

User: 6 July by 5 pm. Today is 3 July. Contract has a dispute clause: project leads meet within seven days, then mediation in Bengaluru before court. I don't want to threaten mediation yet.

Assistant: Agreed. Ask for a project-lead discussion under the ordinary resolution process without invoking later stages. Keep mediation as a future option, not part of the first reply.

User: I sent it at 16:12 and attached contract, appendix A, prototype approval, delivery email, and a screenshot of the live dashboard. I did not attach Slack or mention the budget-freeze call.

Assistant: Good evidence set. Save the sent message and attachments together. Do not alter invoice NR-026 while the amount is disputed.

User: Finance replied automatically: Sneha is out until 6 July, contact ap@clientdomain for urgent issues. No human reply yet. PO 7731 is the number. What remains open?

Assistant: Open items: human response by 6 July 17:00, whether they release INR 48,000 without settlement language, their INR 24,000 calculation, correction of PO 7731, and whether you need professional advice before accepting any partial-payment terms. The budget-freeze claim remains unverified and private.""",
        "expected": "Signed contract is INR 2,40,000; INR 1,68,000 was paid. Final invoice NR-026 is INR 72,000, due 30 June. Scope includes mobile empty states; 14 scoped screens were delivered 12 June, praised in writing, and went live 18 June. Finance offered INR 48,000 only with a INR 24,000 credit note and full settlement. User declined that condition, requested unconditional partial payment, the calculation, PO 7731 correction, and a project-lead discussion by 6 July 17:00, attaching core evidence. Open: human response, payment terms, PO discrepancy, and professional advice. Budget-freeze claim is unverified/private.",
        "max_words": 115,
    },
    {
        "item_id": "summary_meeting_launch_001",
        "subcategory": "meeting_transcript",
        "difficulty": "medium",
        "task": "Summarize this meeting for product and support leads who missed it in no more than 105 words. Separate decisions from discussion, name owners and deadlines, and do not turn proposals into commitments.",
        "source_kind": "meeting_transcript",
        "style_pointer": "meetingbank",
        "source": """MAYA (Product): Okay, we're two minutes over, so let's start. This is the go/no-go for the family-plan beta, not public launch. I want us to leave with a decision, owners, and, um, no invented certainty.

ARUN (Engineering): Build 6.4.19 is in staging. The invite flow, seat removal, and payment proration tests passed. We still have the Safari 17 issue where the confirmation spinner can sit for about twelve seconds.

LEENA (Support): Does it fail, or is it just slow? Because the macro is different.

ARUN: Slow. We have fifty reproductions, all completed. No duplicate charges in the logs. The UI looks stuck, though, and people may click twice.

MAYA: Can we disable the button after the first click?

ARUN: That patch exists, but it missed this build. Earliest verified build is tomorrow afternoon. We could also put a line above the button saying it may take fifteen seconds on Safari.

DEV (Design): A warning before every confirmation makes the normal path feel broken. I would rather change the button text after click to “Adding member…” but, yeah, that is also code.

LEENA: For a hundred invited accounts, support can live with a temporary note. I need the exact affected browser and confirmation that Android, Chrome desktop, and iOS app are clean.

ARUN: Confirmed from the test matrix: observed only Safari 17 on macOS. We have not tested Safari 16 this week. The iOS app uses a different flow.

RITU (Analytics): On measurement, the beta list currently has 132 account owners. If we cap at one hundred, we should randomise rather than take the first hundred; the first group came from the research panel and is unusually engaged.

MAYA: Agreed in principle. Can you produce a random hundred and keep thirty-two as holdback?

RITU: Yes, by 18:00 today. One caveat: two records lack marketing consent, although this is a product invite, not a campaign. Legal should decide whether they stay in the pool.

FARAH (Legal): Remove those two from outreach. Product messages still need the contact basis documented, and we do not have it in this sheet. Randomise from the remaining 130.

MAYA: Decision. Ninety-eight invitees, then? Or replace them from somewhere else?

RITU: If the target is exactly one hundred, I can draw a hundred from the 130 eligible records. The two are simply excluded.

LEENA: What about the help centre? The draft says members can leave and rejoin without a new invitation, but staging asks for a new invite after removal.

ARUN: The product behaviour is correct; the article is wrong. Removal revokes the seat token. Rejoining needs a fresh invitation.

DEV: I can correct the article wording and the in-product tooltip. Tooltip requires a release, so not for this beta. Article can change today.

MAYA: Article today. Tooltip goes into the next build, not a beta blocker. Leena, can you update the macro with Safari timing and re-invite behaviour?

LEENA: By noon tomorrow. I also need an escalation tag. Could we use family_beta rather than billing_bug so reports stay clean?

RITU: family_beta is fine. I will add it to the dashboard filter.

ARUN: One more issue. Proration emails show tax as a separate line in the UI, but the email combines tax into the adjustment. Totals match. Finance signed off last week.

MAYA: Is that confusing enough to block?

LEENA: Not for a beta if the total matches. Give support a note. We should not call it a billing error.

FARAH: And do not promise the breakdown will remain that way. Just say the presentation differs and the total is validated.

MAYA: Okay. Proposed decision: launch the beta Thursday at 14:00 IST to one hundred randomly selected eligible accounts, with the Safari note, corrected article, and support macro ready first.

ARUN: Thursday works if tomorrow's smoke test passes. I want a stop condition: any duplicate charge, invite acceptance failure above five percent, or more than ten Safari contacts in the first day.

RITU: I can alert on acceptance failure. Duplicate charge comes from finance monitoring, not my dashboard.

LEENA: I can count tagged contacts, but ten contacts is not ten unique customers. Is that okay?

MAYA: Use ten unique accounts. Arun, write the stop conditions in the runbook by tomorrow 11:00. Ritu owns acceptance alerting; Finance owns duplicate-charge monitoring; Leena owns unique tagged accounts.

FARAH: The invite copy still says “share with anyone in your household.” Make it “people who live at your address,” because that is the beta eligibility definition.

DEV: I will update invite copy and the article by 17:00 today.

MAYA: Great. Final decision is conditional go for Thursday 14:00, pending smoke test and readiness items. This is not approval for public launch. Anything else? No? I'll send notes.""",
        "expected": "Decision: conditional beta go Thursday at 14:00 IST for 100 randomly selected eligible accounts, excluding two without documented contact basis; this is not public-launch approval. It depends on smoke testing and readiness work. Dev updates invite copy and help article by today 17:00. Ritu selects invitees by 18:00 and alerts on acceptance failures. Arun writes stop conditions by tomorrow 11:00. Leena updates the macro by tomorrow noon and tracks unique family_beta contacts. Stop for any duplicate charge, acceptance failures above 5%, or over 10 unique Safari-contact accounts. Safari 17 delay is known; totals remain validated.",
        "max_words": 105,
    },
    {
        "item_id": "summary_meeting_incident_001",
        "subcategory": "meeting_transcript",
        "difficulty": "medium",
        "task": "Write an incident-review summary for engineering leadership in no more than 110 words. Separate confirmed facts, suspected causes, decisions, and follow-up owners. The audience did not attend the meeting.",
        "source_kind": "meeting_transcript",
        "style_pointer": "qmsum",
        "source": """PRIYA (Incident lead): Thanks, everyone. This is the review for yesterday's notification delay. We are not assigning blame, and, just to be explicit, the cause is not confirmed yet.

OMAR (SRE): Customer impact started 09:12 IST and ended 10:04. Push notifications were delayed between six and forty-one minutes. Email notifications were normal. We counted 184,220 delayed pushes across about 61,000 devices.

NEEL (Mobile): Did we lose any, or were they all eventually delivered?

OMAR: We have delivery receipts for 183,907. Three hundred thirteen expired because their time-to-live was fifteen minutes. Those were promotional price alerts, not security or transaction messages.

KAVITA (Support): We had 286 contacts, mostly “sale alert arrived after item sold.” No reports of missed OTPs because OTP uses a different provider.

PRIYA: Good, that's a confirmed scope distinction. What changed before 09:12?

OMAR: Queue worker release 2.18 started at 08:50 and reached all workers at 09:07. Cloud provider metrics show an availability-zone network event beginning 09:18, six minutes after our first queue alarm.

SAMEER (Backend): Release 2.18 changed batch size from 500 to 2,000 and enabled payload compression. In load test it reduced API calls. It also made each retry batch larger.

NEEL: So the release caused it?

SAMEER: Maybe contributed. But the first delayed batch was on a worker still reporting 2.17. Clock skew could make that version label stale; we have not checked.

OMAR: Also, queue age rose first in zone b. When we shifted workers away from zone b at 09:39, age started falling. We rolled back 2.18 at 09:46, and age kept falling. Since both actions overlap, production timing cannot isolate them.

PRIYA: What is definitely fixed now?

OMAR: Queue age returned below thirty seconds at 10:04. We are on 2.17, zone b capacity remains at twenty percent instead of forty, and alerts have stayed green for twenty-six hours.

SAMEER: There is a hidden limit I found this morning. The provider API allows 1,200 compressed messages per request for our account, not 2,000. Requests above that are accepted, then partially retried. Documentation says limits vary by contract.

PRIYA: Is that confirmed from support?

SAMEER: Confirmed from our response headers and a provider ticket reply, yes. But it does not explain the 2.17 worker unless the label was stale or zone networking caused independent retries.

KAVITA: Are we sending customers a notice?

PRIYA: Status page already says delayed promotional pushes, resolved, no transaction or security notifications affected. Do we need direct mail?

KAVITA: I don't recommend it. The affected alerts are non-critical, and direct mail could cause more confusion. We should give agents a credit rule for customers who bought after a late price alert.

MEERA (Product): Agree. Credit only when the order timestamp is after the offer ended and the alert receipt is late. Cap at INR 500?

KAVITA: INR 500 account credit, one per affected account, with the alert ID attached. I can publish the macro today.

PRIYA: Decision. Now action items.

OMAR: I will add zone-level queue-age alerts and preserve worker-version labels from the deployment service, due Friday. We currently scrape the process label, which may lag.

SAMEER: I will build a replay test that separates batch size, compression, and zone latency. Results by Monday. Until then, batch size stays 500 and compression stays off.

NEEL: Mobile can expose received-at versus sent-at in the internal diagnostics screen. That is useful, but it is not needed for root cause.

PRIYA: Put that in the backlog, no deadline. Provider follow-up?

OMAR: Ticket CP-88214 is open. We asked for zone-b packet-loss timing and written confirmation of the 1,200 limit.

MEERA: One concern: marketing wants the weekend campaign. Are pushes safe?

OMAR: Current metrics are normal at expected weekend volume. I recommend proceeding with a staffed watch, not claiming risk is zero.

PRIYA: Decision: campaign proceeds, SRE staffed for the first two hours. Rollback remains in place. Root cause stays “under investigation”; leading contributors are oversized compressed batches and the zone-b event, but neither alone is proven.

KAVITA: Macro and credit rule by 16:00 today.

PRIYA: Great. I will circulate the review and schedule the Monday evidence check.""",
        "expected": "Confirmed: 184,220 pushes to about 61,000 devices were delayed 6–41 minutes from 09:12–10:04 IST; 313 promotional alerts expired. Email, OTP, security, and transaction messages were unaffected. Recovery followed zone-b capacity reduction and release rollback, so causality is unresolved. Oversized compressed batches and a zone-b network event are leading contributors, not proven root causes. Campaign proceeds with two hours of staffed SRE watch; batch size stays 500 and compression off. Omar owns alerts/version evidence and provider ticket CP-88214; Sameer runs replay tests by Monday; Kavita publishes the INR 500 credit rule by 16:00 today.",
        "max_words": 110,
    },
    {
        "item_id": "summary_meeting_budget_001",
        "subcategory": "meeting_transcript",
        "difficulty": "hard",
        "task": "Summarize this procurement meeting for the CFO in no more than 115 words. Separate decisions from discussion, preserve assumptions and unresolved exposure, and name the next owners and dates.",
        "source_kind": "meeting_transcript",
        "style_pointer": "meetingbank",
        "source": """ANITA (Finance): The purpose today is to decide whether we renew FieldNote now, negotiate, or prepare an exit. The current contract ends 30 September. Please distinguish actual quotes from our estimates.

RAHUL (Operations): We have 318 active field users. FieldNote quoted INR 42.6 lakh for one year, up from 36 lakh. They say the increase reflects offline maps and priority support.

ZARA (Procurement): The 42.6 is a written quote valid until 31 August. They also offered 78 lakh for two years, paid annually. There is no price cap after year two.

VIKAS (IT): Usage numbers need context. We provision 350 licences, 318 logged in last month, and 271 used it on at least ten days. The quote is still for 350 seats.

ANITA: Can we reduce seats?

RAHUL: Seasonal surveyors take us to 342 between November and January. Reducing below 350 means buying monthly seats later, and their list price is much higher.

ZARA: Vendor has not formally quoted seasonal seats. “Much higher” is from last year's rate card, so mark that as an assumption.

MEENA (Security): Renewal also depends on the penetration report. Their current report covers version 7.2; production is 7.5. They promised an updated report by 20 August. We have no evidence of a vulnerability, but the control document is stale.

VIKAS: And single sign-on is still on the roadmap. We use password plus device certificate. It meets our present exception, which expires 31 December.

ANITA: What is the alternative?

VIKAS: TrackRural gave an indicative estimate of INR 29 lakh for 350 users. Not a binding quote. Migration would take, I think, twelve to sixteen weeks and roughly four engineers half-time.

RAHUL: That estimate ignores offline survey templates. We have 87 templates and around 1.8 million cached records on devices. The pilot imported three simple templates, not the complex inspection forms.

ZARA: Also TrackRural charges separately for map tiles. They have not answered our data-residency questionnaire.

ANITA: So 29 lakh is not comparable yet.

VIKAS: Correct. My internal migration labour estimate is INR 18–24 lakh equivalent, plus disruption. Again, estimate, not spend approved.

RAHUL: Field teams had two severity-one incidents this year. One lasted 47 minutes, one 81 minutes. Both were within the four-hour SLA. That sounds okay, but the second happened during a survey deadline and we had to extend submissions.

ZARA: Service credits totalled INR 1.2 lakh. The new quote removes automatic credits and requires us to claim within ten days.

ANITA: That is a regression. What can we negotiate?

ZARA: My proposal: one-year renewal at no more than 38 lakh, retain automatic credits, require the current penetration report, and add an SSO delivery milestone by 30 November with termination rights. I do not think they accept all of that.

MEENA: The penetration report is a pre-signing condition, not merely a covenant. For SSO, I can extend the exception to 31 March only if a signed delivery plan exists.

RAHUL: Termination rights mid-season are not operationally useful. If they fail SSO in December, we cannot migrate during the survey peak anyway.

VIKAS: We could make failure trigger a price credit and an exit-assistance package rather than immediate termination.

ANITA: That seems more usable. Do we have budget room at 38?

ANITA (continuing): Approved operating plan has 39 lakh. Anything above that needs my variance note. The two-year 78 lakh averages 39, but it reduces leverage before SSO is delivered.

ZARA: And it is not a discount against the budget; it is exactly 39 per year. I recommend one year.

RAHUL: I agree, provided we do not signal an exit to field teams yet. Rumours will cause people to export data manually.

MEENA: We should still start the security questionnaire with TrackRural. That is due diligence, not a migration announcement.

ANITA: Decision: negotiate a one-year renewal, target 38 lakh, absolute authority ceiling 39 lakh. No two-year term. Current 7.5 penetration report is required before signature. SSO plan by 30 November, with credits and exit assistance if missed. Keep automatic service credits.

ZARA: I can send redlines tomorrow, 13 August, and request a response by 19 August. Quote expires 31 August, so that leaves one round.

VIKAS: I will produce a comparable TrackRural total-cost range by 21 August, including map tiles, template conversion, labour, and seasonal seats. It may still have wide uncertainty.

MEENA: I will finish the TrackRural residency questionnaire review within two days of receiving their answer. For FieldNote, no security approval until the updated penetration report arrives.

RAHUL: I will validate the count of 87 templates and identify the ten hardest conversions by Friday.

ANITA: If FieldNote stays above 39 or misses the report, we reconvene 24 August to decide on a short extension versus accelerated exit. No commitment to either today.

ZARA: One correction: a short extension has not been quoted. We should not assume it is available.

ANITA: Good correction. Record it as an unpriced, unconfirmed fallback, not a plan.""",
        "expected": "Decision: negotiate a one-year FieldNote renewal; target INR 38 lakh, hard ceiling INR 39 lakh, no two-year term. Signature requires a version 7.5 penetration report. Redlines retain automatic credits and require an SSO plan by 30 November, with credits and exit help if missed. Zara sends redlines 13 August and seeks response by 19 August. Vikas produces a comparable TrackRural cost range by 21 August; Rahul validates 87 templates and the ten hardest conversions by Friday. If price or security conditions fail, reconvene 24 August. TrackRural’s INR 29 lakh and migration cost are estimates; a short extension is unquoted and unconfirmed.",
        "max_words": 115,
    },
    {
        "item_id": "summary_long_incident_001",
        "subcategory": "long_incident_update",
        "difficulty": "medium",
        "task": "Write an executive handoff for the incoming incident commander in no more than 110 words. Prioritize impact, timeline, current controls, and next actions. Clearly separate confirmed facts from suspected causes.",
        "source_kind": "long_update",
        "style_pointer": None,
        "source": """INCIDENT UPDATE — PAYOUT FILE REJECTIONS
Last edited: 3 August 2026, 18:40 IST
Owner: Payments Operations

At 12:18 IST, the reconciliation dashboard alerted on an increase in rejected payout files sent to Samudra Bank. The first affected file had been submitted at 11:55. Between 11:55 and 13:07, Samudra rejected 46 files containing 8,912 merchant payouts worth INR 7.84 crore. No money left the settlement account for rejected rows. Files sent to the other three banking partners were accepted normally, and card authorisations, customer refunds, and merchant balance calculations were not affected.

The bank’s rejection message was “header date invalid,” even though the file header contained 20260803, which matches the format in the current integration guide. Operations paused automatic retries at 12:31 to avoid producing duplicate submissions. The payout records remained in ready state with their original idempotency keys. Support received 37 merchant contacts after estimated arrival times passed; 11 of those merchants said the delayed funds could affect same-day supplier payments. No merchant was told that a payout had completed.

At 12:44, the team compared an accepted file from 31 July with a rejected file from today. Besides expected dates and record values, the accepted file used a six-digit sequence in its filename while today’s file used seven digits after the counter rolled past 999999. The file contents were otherwise structurally identical in that comparison. This filename difference is a suspected contributor, not a confirmed cause: the bank error names the header, and its published naming rule does not specify a maximum sequence length.

Samudra’s integration desk acknowledged ticket SB-440187 at 12:52. At 13:03 it reported that a validation change had been deployed at 10:30, but the bank has not supplied release notes or confirmed whether the change examined filenames. At 13:07, Payments Engineering generated one test file with a manually padded six-digit sequence and four internal test payouts. Samudra accepted it at 13:16. A second test at 13:24 used the seven-digit sequence but changed the header date to 03-08-2026; it was rejected with the same message. These tests strengthen the filename hypothesis but do not prove it, because they were not a fully controlled pair and the bank-side validator may have changed between attempts.

At 13:38, the incident commander approved a temporary filename rule that wraps the external sequence to six digits while preserving the full internal sequence and idempotency key in the ledger. Legal and Finance confirmed that this does not alter payout order or reporting identifiers. The first production retry batch contained 500 payouts and was accepted at 13:52. Two further batches were accepted by 14:21. Operations then resumed processing at a capped rate of one file every five minutes, compared with the normal two files per minute.

By 17:30, 7,980 delayed payouts had been accepted by the bank. The remaining 932 payouts are in nine files scheduled before 20:00. Merchant arrival depends on Samudra’s downstream clearing window; the bank estimates that accepted payouts before 18:00 will arrive today, while later ones may arrive on 4 August. This estimate is not guaranteed. No duplicate payout has been detected. Monitoring compares idempotency keys, bank acknowledgements, and ledger state after each file.

Current controls are the six-digit external filename, capped submission rate, manual review after each bank acknowledgement, and a freeze on unrelated payout-file changes. The incident remains open until all 8,912 payouts are acknowledged and duplicate checks complete. An engineer and an operations analyst are staffed through 21:00. Support is using macro PAYOUT-DELAY-03, which states that payouts are delayed, not lost, and avoids promising a same-day arrival.

Open actions: Operations owns the remaining nine files and a merchant-impact list by 20:30. Engineering owns a controlled test matrix for filename length and header format by 11:00 on 4 August. Samudra owes validator release notes, a confirmed root cause, and written assurance about supported sequence length under SB-440187. Finance will decide tomorrow whether merchants with documented late fees qualify for credits. Do not remove the workaround until the bank confirms its validator behaviour and the controlled replay passes.""",
        "expected": "Confirmed: Samudra rejected 46 files covering 8,912 payouts worth INR 7.84 crore; no funds left for rejected rows, other partners were normal, and no duplicates are detected. A six-digit filename workaround restored acceptance; 7,980 payouts were accepted by 17:30, with 932 due before 20:00 under capped, manually reviewed processing. Filename length is suspected, not confirmed; bank validator changes remain unclear. Keep the workaround and freeze. Operations owns completion and impact list; Engineering runs controlled tests by 4 August 11:00; Samudra owes evidence under SB-440187. Arrival timing and merchant credits remain open.",
        "max_words": 110,
    },
    {
        "item_id": "summary_long_budget_001",
        "subcategory": "long_budget_update",
        "difficulty": "medium",
        "task": "Summarize this budget update for the board finance committee in no more than 115 words. State the forecast, confirmed drivers, assumptions, decisions already taken, and choices still required.",
        "source_kind": "long_update",
        "style_pointer": None,
        "source": """Q2 OPERATING FORECAST — CUSTOMER OPERATIONS
Prepared for the 6 August 2026 finance review

The approved annual budget for Customer Operations is INR 18.40 crore. Actual spend through June is INR 9.62 crore, compared with a straight-line half-year budget of INR 9.20 crore. The current full-year forecast is INR 19.18 crore, or INR 78 lakh above plan. This forecast includes actions already approved in July but excludes any new programme discussed below. It is not a cash-flow forecast; vendor payment timing may move between months.

The largest confirmed variance is temporary staffing. The original plan assumed an average of 42 temporary agents from April through September. Actual staffing averaged 58 after two product migrations generated more contacts than forecast. The extra staffing cost through June is INR 31 lakh. Contact volume has now fallen, and Operations ended 12 contracts on 31 July. The forecast assumes eight more contracts end on 31 August and the remaining temporary pool returns to 42 agents in September. If contact volume stays above 118,000 cases per month, those eight contracts may need to continue; each additional month would add approximately INR 6.8 lakh.

Translation spend is INR 17 lakh above plan. Product teams launched four help-centre collections that were not included in the operating plan, covering Hindi, Tamil, Bengali, and Marathi. The invoices and completed word counts have been verified. A further INR 9 lakh is forecast for planned releases through December. That figure assumes product teams freeze source copy ten business days before launch. Late copy changes are billed again and could add INR 2–5 lakh; no contingency for repeated translation is included.

Telephony is INR 11 lakh below plan because more customers used in-app calling. This saving is confirmed through June. The forecast assumes the lower run rate continues, although a telecom price increase of up to 6% is under consultation and has not been announced. The model uses the current tariff, not the possible higher rate.

The knowledge-base platform is another decision point. The existing contract costs INR 54 lakh annually and renews 1 October. The vendor offered a new AI-search module for an additional INR 18 lakh. A six-week internal trial with 22 agents reduced median lookup time from 96 seconds to 71 seconds, but resolution time did not change measurably, and the trial did not include new hires. Operations recommends renewing the base platform but not buying the module this year. The forecast includes the base renewal at the current price; procurement has not yet received a binding renewal quote, so inflation risk remains.

Training is INR 8 lakh under plan because the June induction cohort moved to August. This is timing, not a saving: the expense is included in the second half. Travel is INR 6 lakh under plan and is treated as a permanent saving after three regional meetings moved online. Facilities spend is on plan. No severance, restructuring, or office expansion is assumed.

Management has already frozen non-essential travel, stopped the 12 temporary contracts, and required finance approval for translation changes after source freeze. Together, these actions reduce the forecast by INR 29 lakh compared with the unmitigated July estimate. These are implemented decisions, not proposals.

The committee still needs to decide three items. First, whether to hold a INR 20 lakh central contingency for temporary staffing and translation rework; adding it would increase the forecast overrun from INR 78 lakh to INR 98 lakh but avoid unfunded requests later. Second, whether to authorise procurement to renew the knowledge-base base plan up to a ceiling of INR 58 lakh; the current forecast carries INR 54 lakh. Third, whether product teams, rather than Customer Operations, should fund translation caused by post-freeze changes from September onward.

Finance recommends the INR 20 lakh contingency, base renewal authority up to INR 58 lakh, and chargeback for avoidable translation rework. Operations supports the contingency and renewal but wants a one-month warning before chargeback begins. The forecast should be refreshed on 10 September after August contact volume, temporary staffing decisions, and the binding platform quote are known.""",
        "expected": "Customer Operations forecasts INR 19.18 crore, INR 78 lakh over its INR 18.40 crore plan. Confirmed drivers are excess temporary staffing and unplanned translations, partly offset by telephony and travel savings. The forecast assumes eight contracts end in August, lower telephony rates continue, and the base knowledge platform renews at INR 54 lakh; staffing, translation rework, tariff, and renewal-price risks remain. Implemented actions include travel controls, 12 contract endings, and translation approval. Committee decisions: INR 20 lakh contingency, base-renewal authority up to INR 58 lakh, and translation chargeback timing. Refresh is due 10 September.",
        "max_words": 115,
    },
    {
        "item_id": "summary_long_audit_001",
        "subcategory": "long_audit_update",
        "difficulty": "hard",
        "task": "Prepare a risk-committee brief in no more than 120 words. The audience needs verified findings, scope limitations, immediate containment, suspected exposure, and dated follow-ups kept distinct.",
        "source_kind": "long_update",
        "style_pointer": None,
        "source": """ACCESS REVIEW — REGIONAL PARTNER PORTAL
Draft 4, 2 August 2026
Distribution: Risk Committee, Security, Partner Operations

The review examined access to the Regional Partner Portal after an internal control reported that partner departures were not always reaching the identity system. The review period covered 1 January through 30 June 2026. Auditors obtained complete account exports for the India, Singapore, and United Arab Emirates instances. The South Africa instance could not produce a historical role export because its audit plug-in stopped writing role-change events on 11 March. Current South Africa accounts were reviewed, but historical access there is not verified.

Across the three complete instances, auditors matched 4,812 portal accounts against partner rosters and contract end dates. They verified 63 accounts that remained enabled after the associated person’s recorded end date. The excess period ranged from one to 74 days, with a median of nine days. Forty-eight accounts had read-only access to shipment status and partner contact details. Fifteen could also download monthly invoice PDFs for their own partner organisation. None had administrator rights, could change bank details, or could view another partner’s records under the permissions exported for this review.

Authentication logs were available for 58 of the 63 accounts. Five accounts belong to an acquired partner whose logs were retained in a system that has not yet been restored; their post-departure use is unknown. Of the 58 with logs, four were used after the recorded end date. Three uses came from the same office IP addresses used before departure and downloaded shipment-status CSV files. One account logged in from a new residential network and downloaded two invoice PDFs. The review confirms the logins and downloads but does not establish who controlled the credentials. Partner Operations has contacted the four organisations. Two confirmed that shared team leads continued using the accounts; two have not answered.

No evidence shows bank-detail changes, cross-partner access, or administrator activity. That statement is limited to available portal logs and exported permissions. The investigation has not reviewed files after download, endpoint devices, forwarded documents, or activity in the unavailable acquired-partner logs. It is therefore inaccurate to say there was no data exposure. Verified downloaded data was limited to shipment-status rows and two invoices, while any onward use remains unknown.

All 63 accounts were disabled by 29 July. Security invalidated active sessions and reset API tokens linked to 11 of the accounts. A temporary daily comparison now matches enabled accounts to the central partner roster. Exceptions require approval from both the regional operations manager and Security and expire after seven days. This control covers the India, Singapore, and UAE instances. South Africa is running a manual twice-weekly review until its audit plug-in is repaired.

The process failure has two confirmed elements. First, 27 partner organisations submitted departure updates through free-text email rather than the required roster template; the operations mailbox had no service-level target for processing them. Second, a June workflow change stopped copying regional identity teams on contract-closure tickets. It is not yet known how many of the 63 delayed closures were caused by each issue. Staff turnover and training gaps have been mentioned in interviews but are not supported by sufficient evidence and should remain hypotheses.

The South Africa limitation is material. Current review found six accounts whose users were absent from the latest roster; all six were disabled on 30 July. Because historical role and event data after 11 March are incomplete, auditors cannot determine how long they were active, whether permissions changed, or whether they were used after departure. Database login logs may partially reconstruct activity, but they do not include downloaded file names.

Follow-up ownership is recorded. Partner Operations will obtain written responses from the two organisations with unexplained post-departure use by 5 August and confirm whether the downloaded files were retained. Identity Engineering will restore the acquired-partner logs by 8 August and repair South Africa’s audit plug-in by 12 August. Internal Audit will sample 100 closures from July after the new daily control has operated for a month, reporting on 7 September. Legal will assess notification obligations only after the four unexplained or partially explained account cases and missing logs are resolved.

The committee is asked to approve keeping the portal open under temporary controls rather than suspending all partner access. Security supports continued operation because high-risk permissions were absent and containment is complete for known accounts, but rates residual risk medium due to incomplete logs and uncertain credential control. Internal Audit agrees provided the South Africa repair date and September effectiveness test remain formal tracked actions.""",
        "expected": "Audit verified 63 late-disabled accounts across three complete regions; four were used after departure, including shipment CSV and two invoice downloads. No available evidence shows admin, bank-change, or cross-partner access, but credential control, onward file use, five acquired-partner logs, and South Africa history remain unresolved. Six additional South Africa accounts were disabled. All known accounts, sessions, and linked tokens are contained; daily matching and temporary manual reviews operate. Confirmed process failures involve free-text departures and a June ticket-routing change; training gaps are only suspected. Operations follows up by 5 August, logs by 8 August, South Africa repair by 12 August, and control testing by 7 September. Residual risk is medium.",
        "max_words": 120,
    },
]


# These are fixed, fully authored sources rather than fragments sampled or shuffled at
# generation time.  The overrides and appendices make source shape an explicit coverage
# dimension: compact, ordinary, long, and very long histories all remain coherent cases.
COMPACT_SOURCE_OVERRIDES = {
    "summary_chat_handoff_001": """User: hi, duplicate train booking problem. the app froze after payment and I booked again before the first confirmation arrived.

Assistant: Send the two references, passengers, exact travel date, and which booking you want protected. Do not include card details or OTPs.

User: IR-482771 and IR-482809, both for Nandita Rao and Kiran Rao on train 12658, Friday 7 August 2026. Keep IR-482771 because it has the lower berths. The second cost INR 3,460.

Assistant: Understood: protect IR-482771 and ask for fee-waived cancellation only of IR-482809. What were the payment and confirmation times?

User: payments 18:42 and 18:49 yesterday. confirmation for the first came 18:51. I have screenshots. The app shows INR 240 cancellation fee per passenger.

Assistant: Do not use automatic cancellation yet. Raise a duplicate-booking issue with the three timestamps, both references, and redacted screenshots. Ask support to waive the fees.

User: done, case CASE-19308, three screenshots uploaded. It says reply within 48 hours. Nothing cancelled. Departure is 22:40 on 7 August; I still need to check the refund cutoff.

Assistant: Good. Call only if the 48-hour support window could run past that cutoff. Keep the cutoff as an open question.

User: small complication: the form auto-filled old phone ending 1180. Current is 7741; I put 7741 in the description but can't edit the submitted form.

Assistant: Handoff state: CASE-19308 is open, IR-482771 must stay active, and IR-482809 is the only cancellation target. If support attempts phone contact, clarify that 7741 is current and 1180 obsolete; do not open another case unless the first cannot be updated.""",
    "summary_meeting_launch_001": """Ritu: Okay, we have twenty minutes. This is a beta decision, not a public-launch meeting. Can we expose the new family payment flow on Thursday?

Dev: Checkout totals match the old flow in our test accounts. Safari 17 still waits several seconds after the bank redirect, but the payment completes and refresh shows the receipt. The duplicate-charge tests passed.

Leena: Support can cover a small cohort. The help article still says the receipt appears immediately, so that line must change before invitations go out.

Arun: I want explicit stop conditions. Any duplicate charge stops the beta. I also propose stopping if payment-acceptance failures exceed five percent or more than ten distinct Safari accounts contact support.

Ritu: Cohort proposal is 100 randomly selected eligible accounts. Two accounts in the draft list have no documented contact basis, so exclude them. This does not approve public release.

Dev: I can update the invite copy and help article by 17:00 today, then run the production smoke test tomorrow morning.

Leena: I will update the support macro by noon tomorrow and tag contacts family_beta. We need unique accounts, not number of tickets.

Arun: I will write the stop-condition query and owner rotation by 11:00 tomorrow. Acceptance failure means a bank-authorised payment that our service does not record successfully; user cancellations do not count.

Ritu: Decision is a conditional go for Thursday at 14:00 IST, provided smoke testing and those readiness tasks pass. I will select the 100 accounts by 18:00 today and watch invitation failures.

Dev: Fine. Safari delay remains a known issue, not a blocker for this limited beta. Public launch comes back to this group with beta evidence.""",
    "summary_long_incident_001": """PAYOUT PROCESSING UPDATE — 3 AUGUST 2026, 18:40 IST

Samudra Bank rejected 46 payout files submitted from 11:55 onward. The files contained 8,912 merchant payouts worth INR 7.84 crore. Rejected rows never left the settlement account; files sent to three other banks were accepted normally. No duplicate payout is detected.

The bank returned “header date invalid,” although the header matched its current guide. Comparison with an older accepted file found that today's filename used a seven-digit sequence after the counter rolled over, while the older file used six digits. A manually padded six-digit test was accepted. This supports a filename-length hypothesis but does not prove it: Samudra also deployed an undocumented validator change, and the test was not a fully controlled pair.

Incident command approved a temporary six-digit external filename while retaining the complete internal sequence and idempotency key. The first retry succeeded. Processing resumed at a capped rate with manual review after every acknowledgement and a freeze on unrelated payout changes.

By 17:30, Samudra had accepted 7,980 payouts. The remaining 932 are in nine files scheduled before 20:00. Arrival today is not guaranteed even after acceptance. Operations owns completion and the merchant-impact list by 20:30. Engineering will run a controlled filename/header matrix by 11:00 on 4 August. Samudra ticket SB-440187 must provide release notes, a confirmed cause, and supported filename limits. Finance has not decided whether documented late fees qualify for credits. Keep the workaround until bank evidence and replay results agree.""",
}

SOURCE_APPENDICES = {
    "summary_chat_handoff_003": """

User: I found the old association checklist. It says the community room has only six usable sockets, and extension leads cannot cross the centre aisle. That affects the charging plan.

Assistant: Add that to equipment constraints. Place a charging table against the side wall, tape any cable along the wall, and avoid promising charging for every participant. The reminder should still ask for charged phones.

User: Tara can bring one four-socket board. Mine is three sockets but the cable is short. I asked the office manager about another board; no answer yet.

Assistant: Tara's board is confirmed. The second board and a safe layout remain open. Do not daisy-chain the boards, and let the association's facility contact approve placement.

User: Tea quote came back: INR 28 per person including paper cups, minimum 20. If 25 attend that's INR 700. With 35 black-and-white sheets at INR 2, total so far INR 770, right?

Assistant: Yes, INR 700 for 25 teas plus INR 70 printing is INR 770, within the INR 1,500 cap. Attendance could change the tea amount, so order after registration closes. This resolves the price question but not the final quantity.

User: Ramesh confirmed, but only from 10:15 to 11:30 because he has a doctor appointment. He can help with the opening and UPI section, then Tara and I handle appointments and questions.

Assistant: Staffing update: Ramesh is confirmed through 11:30, covering the parts where Hindi support matters most. Mention to him that no real PIN, OTP, account balance, or medical record should appear during demonstrations.

User: Projector test was mixed. My spare HDMI cable works, but text at the back is small below 28 point. Microphone isn't needed. Association approved an empty-room setup photo as long as no flat numbers are visible.

Assistant: Resolved: use the spare cable, design slides at 28 point or larger, no microphone, and allow only an empty-room photograph without flat numbers. Back-row readability should be checked again after chairs are arranged.

User: I drafted the handout with screenshots, but grayscale makes the green verified tick and grey pending clock almost identical. Maybe one colour page is worth it?

Assistant: Do not rely on colour alone. Add text labels and distinct shapes. You can still print colour if useful, but accessibility should survive black and white. That also keeps the budget flexible.

User: Let's keep black and white with labels. Final decisions: 35 copies, high contrast, large type. Tea order after 12 August. Ramesh confirmed until 11:30. Second extension board and final chair layout are still open.

Assistant: Recorded. The content, agenda, demonstration safety, handout format, first extension board, cable, photo permission, and partial Hindi staffing are settled. Remaining questions are final attendance and tea quantity, the second board, safe cable placement, chair layout, and the completed handout proof.""",
    "summary_chat_handoff_004": """

User: update: project lead replied, but only in Slack. She wrote “mobile empty states were always in scope; finance is mixing them up with the two tablet variants.” She also said she cannot approve payments herself.

Assistant: Preserve that message with its timestamp and channel. It supports the scope distinction but does not itself authorise payment. Keep the signed scope and email approvals as primary evidence.

User: Timestamp is 6 July 11:18 in #dashboard-redesign. I exported that thread as PDF. She tagged finance partner Kunal, who replied with a thumbs-up but no words.

Assistant: Record the project lead's statement; do not treat Kunal's reaction as agreement or approval. Has Finance answered the request for its INR 48,000 calculation or the PO correction?

User: Kunal emailed at 14:06. Says the INR 48,000 was “commercial discretion” and not a screen-by-screen calculation. He will ask procurement whether PO 7731 can be raised from INR 2,16,000 to INR 2,40,000. No timeline.

Assistant: That clarifies that the reduction has no supplied scope calculation. PO correction is being investigated, not promised. Save the email and ask for an owner and date without accusing anyone of changing the contract.

User: I replied asking for procurement owner and update by 8 July noon. I also repeated that unconditional part-payment can be made without accepting full settlement. Was that too much?

Assistant: It is a factual follow-up. Avoid sending additional messages until that deadline unless they ask a question. Did you attach bank or tax details again?

User: no. They already have those in the vendor portal. I attached nothing this time.

Assistant: Good. Keep financial details in the approved portal and do not move the conversation to personal messaging.

User: Their product lead called me. I didn't record it. She said budget freeze is real but applies to new change requests, and she thinks final milestone should still be paid. Can I quote the call?

Assistant: You can write a neutral contemporaneous note and send a short confirmation email: date, attendees, and your understanding, inviting correction. Do not present an unrecorded call as an agreed written commitment.

User: I sent: “Thanks for speaking at 15:20. My understanding is the freeze concerns new change requests and you believe the contracted final milestone remains payable; please correct me if I misunderstood.” She replied “yes, that's accurate, finance still owns timing.”

Assistant: That is now written support for the distinction, while payment timing remains unresolved. Add both messages to the chronology.

User: Chronology is getting long: contract 3 March, prototype approval 8 May, final handoff 12 June, live 18 June, invoice due 30 June, finance dispute 2 July, my response 3 July, lead/finance messages today 6 July.

Assistant: Keep that chronology. Separate contract evidence, delivery evidence, and later dispute correspondence rather than merging screenshots into one unreadable file.

User: Procurement replied early on 8 July. PO 7731 was created at the wrong value because someone entered the post-advance balance as the contract total. They corrected it to INR 2,40,000. Revised PO PDF is in the portal.

Assistant: That resolves the PO discrepancy and removes one accounts-payable blocker. It does not confirm the payment date or withdraw Finance's settlement proposal.

User: Finance then paid INR 48,000 on 9 July with remittance note “part payment, rights reserved by both parties.” I never issued the INR 24,000 credit note. Bank received it this morning.

Assistant: Update the amounts: INR 2,16,000 total has now been paid, leaving INR 24,000 of invoice NR-026 outstanding. The wording and absence of a credit note indicate this was not the full-settlement condition previously proposed, but keep the remittance record and avoid a legal conclusion.

User: They are asking me to join a call Friday about the remaining INR 24,000. I can do 11:00 or 16:00. Should I send all evidence again?

Assistant: Reply with availability and a concise issue list. Link the existing evidence folder rather than attaching duplicates. Ask who has authority to agree payment timing and whether procurement or Finance will attend.

User: I chose Friday 11:00. Attendees now show product lead, Kunal from Finance, and procurement manager. My accountant says the invoice stays open for INR 24,000. I have not asked a lawyer yet.

Assistant: Current state for handoff: corrected PO is INR 2,40,000; INR 48,000 unconditional part-payment arrived 9 July; NR-026 remains open for INR 24,000; Friday 11:00 discussion is booked with product, Finance, and procurement. Open risks are payment timing, whether the client maintains any scope objection after the lead's written clarification, and whether professional legal advice becomes necessary.""",
    "summary_meeting_budget_001": """

Meera: Before we close, I want the migration estimate in the record. TrackRural's sales deck says eight weeks, but Vikas, your team said that excluded attachment history.

Vikas: Correct. Eight weeks covers current forms and active users. Exporting seven years of attachments could add three to five weeks. We have not tested FieldNote's bulk-export rate, so neither number is a commitment.

Rahul: Ten of our 87 templates contain offline photo validation. Those are the hard conversions. The other 77 look mechanical, but “look” is doing work there; we have validated only five end to end.

Zara: That is why the TrackRural number remains a comparison range, not a ready fallback. It gives us leverage in negotiation, but it does not justify rejecting a one-year renewal today.

Meera: What about the short extension people mentioned in email?

Zara: FieldNote has not quoted it. Their standard order form renews for twelve months. I can request three months, but we should not put an unpriced option in the decision as though it exists.

Vikas: Security also asked whether the penetration report covers version 7.5 or only the hosted platform generally. The vendor's old report names 7.3. We need the current report to identify the tested version explicitly.

Rahul: If it arrives and has only low findings, can we sign immediately?

Zara: Only after Security confirms acceptability. The commercial ceiling and redlines are authority to negotiate, not permission to waive the report condition.

Meera: Agreed. Record the migration assumptions, keep the short extension unconfirmed, and do not describe the old penetration report as current evidence.

Vikas: I will include attachment export and internal labour in the comparable cost range due 21 August. Cloud fees alone would understate TrackRural.

Rahul: I will validate all 87 template names and run the ten hardest conversions by Friday. Any template we cannot reproduce gets a named owner and manual fallback estimate.

Meera: Final check: one year, target 38, ceiling 39, no two-year term; current security report required; redlines and dates as already stated. If price or security fails, reconvene 24 August.

Zara: Yes. Nothing in this discussion changes those decisions.""",
    "summary_long_audit_001": """

APPENDIX A — SAMPLE AND DATA LIMITATIONS

The 4,812-account match used partner rosters received by 15 July. Eleven partners supplied spreadsheets with locally formatted dates; Audit normalized them and retained the originals. A second auditor reviewed every matched departure behind the 63 confirmed accounts. Records with only an unverified nickname match were excluded rather than counted as findings. This reduces false positives but means an account under a substantially different name may remain outside the result.

The permissions conclusion comes from role exports, not screen-by-screen attempts. Audit tested representative read-only and invoice roles in the India instance and observed the expected restrictions. It did not impersonate all 63 users in production. The absence of administrator, bank-change, and cross-partner permissions is verified against available role mappings and logs, but a defect outside those sources would not have been detected by this review.

The two invoice PDFs downloaded by the residential-network login contain the partner's own billing address, tax registration, service description, and monthly amount. They do not contain card or bank-account details. Legal has not decided whether the downloads require notification because credential ownership and onward sharing are still unknown.

South Africa's audit plug-in repair has a tested rollback plan but has not yet been deployed. The 12 August date includes a two-day observation period. Until the repair produces complete events, the twice-weekly manual review remains a compensating control rather than proof that historical exposure is absent.

The committee secretary asked whether “containment complete” can appear in external language. Security advised using “known affected accounts have been disabled and active sessions invalidated.” That wording stays within verified scope and does not imply that missing logs or downstream copies have been resolved.""",
}

TASK_OVERRIDES = {
    "summary_chat_handoff_001": (
        "Compress this short support chat for the next assistant in no more than 75 words. "
        "Preserve the decision, completed action, corrected contact detail, and open question."
    ),
    "summary_chat_handoff_003": (
        "Compress this extended planning chat for another assistant in no more than 140 words. "
        "Separate settled workshop decisions from remaining logistics and preserve the latest revisions."
    ),
    "summary_chat_handoff_004": (
        "Write a case handoff for another assistant in no more than 165 words. Preserve chronology, "
        "amounts, written evidence, superseded disputes, current decisions, and unresolved risks."
    ),
    "summary_meeting_launch_001": (
        "Summarize this short launch meeting for product, support, and risk leads in no more than 80 words. "
        "Separate the conditional decision from readiness work and stop conditions."
    ),
    "summary_meeting_budget_001": (
        "Summarize this extended renewal meeting for the procurement steering group in no more than 145 words. "
        "Separate decisions, conditions, estimates, owners, dates, and unconfirmed fallback options."
    ),
    "summary_long_incident_001": (
        "Compress this incident update for the payments incident handoff in no more than 85 words. "
        "Distinguish confirmed impact and recovery from the suspected cause, then retain controls and open owners."
    ),
    "summary_long_audit_001": (
        "Prepare a risk-committee brief in no more than 150 words. Separate verified findings, evidence limits, "
        "containment, suspected exposure, and dated follow-ups."
    ),
}

MAX_WORD_OVERRIDES = {
    "summary_chat_handoff_001": 75,
    "summary_chat_handoff_003": 140,
    "summary_chat_handoff_004": 165,
    "summary_meeting_launch_001": 80,
    "summary_meeting_budget_001": 145,
    "summary_long_incident_001": 85,
    "summary_long_audit_001": 150,
}

EXPECTED_OVERRIDES = {
    "summary_meeting_launch_001": "Conditional beta go: Thursday 14:00 IST for 100 randomly selected eligible accounts, excluding two without documented contact basis; public launch is not approved. Dev updates invite/help copy by 17:00 and runs smoke tests. Ritu selects accounts by 18:00 and watches invitation failures. Arun defines stop queries by tomorrow 11:00; Leena updates the macro by noon and counts unique family_beta contacts. Stop for any duplicate charge, acceptance failures above 5%, or more than 10 distinct Safari-contact accounts. Safari delay is known.",
    "summary_chat_handoff_003": "Confirmed: Silver Oak workshop is 16 August, 10:30–12:00; 21 registered, capacity 32. Agenda is 10 setup, 20 WhatsApp, 25 UPI safety, 20 appointments, 15 Q&A. Use offline fictional demonstrations, no participant recording, and 35 labelled high-contrast black-and-white handouts. Tara assists; Ramesh is confirmed only until 11:30. Tea is INR 28 each and should be ordered after registration closes 12 August. Spare HDMI works; slides need 28-point text; empty-room photo is approved without flat numbers. Open: final attendance/tea quantity, second extension board, safe cable and chair layout, and final handout proof.",
    "summary_chat_handoff_004": "Contract total is INR 2,40,000; mobile empty states were in signed scope, 14 screens were delivered 12 June and launched 18 June. Prototype and delivery approvals plus the project lead's written clarification support scope; the budget-freeze claim was later confirmed to concern new requests, not the milestone. PO 7731 was corrected from INR 2,16,000 to INR 2,40,000. Finance's original INR 48,000-for-credit proposal lacked a scope calculation and was not accepted. An unconditional INR 48,000 part-payment arrived 9 July without a credit note, bringing total paid to INR 2,16,000 and leaving invoice NR-026 open for INR 24,000. Friday 11:00 meeting includes product, Finance, and procurement. Open: remaining-payment timing, Finance's current scope position, and whether professional advice is needed.",
    "summary_meeting_budget_001": "Decision: negotiate a one-year FieldNote renewal; target INR 38 lakh, ceiling INR 39 lakh, no two-year term. Signature requires Security's acceptance of a version 7.5 penetration report. Redlines retain automatic credits and require an SSO plan by 30 November, with credits and exit help if missed. Zara sends redlines 13 August and seeks response by 19 August. Vikas produces a TrackRural comparison by 21 August including attachment export and labour; its INR 29 lakh, eight-week migration, and a short extension remain unconfirmed estimates. Rahul validates all 87 templates and the ten hardest conversions by Friday. If price or security fails, reconvene 24 August.",
    "summary_long_audit_001": "Audit verified 63 late-disabled accounts across three complete regions; four were used after departure, including shipment CSV and two invoice downloads. Available roles and logs show no admin, bank-change, or cross-partner access, but they cannot resolve credential control, onward use, five acquired-partner logs, unmatched identities, or South Africa history. Six additional South Africa accounts were disabled. All known accounts, sessions, and linked tokens are contained; daily matching and manual reviews operate. Confirmed failures involve free-text departures and a June ticket-routing change; training gaps are only suspected. Operations follows up by 5 August, logs by 8 August, South Africa repair by 12 August, and control testing by 7 September. Residual risk is medium; external wording must say known accounts are contained, not imply all exposure is resolved.",
    "summary_long_incident_001": "Samudra rejected 46 files covering 8,912 payouts worth INR 7.84 crore; rejected funds did not leave, other banks were normal, and no duplicates are detected. A six-digit filename workaround restored acceptance: 7,980 payouts were accepted by 17:30, with 932 scheduled before 20:00 under capped manual review. Filename length is suspected, not confirmed amid an undocumented validator change. Keep the workaround and freeze. Operations owns completion; Engineering tests by 4 August 11:00; Samudra owes evidence under SB-440187. Arrival and credits remain open.",
}

for _spec in NEW_SUMMARY_SPECS:
    _item_id = _spec["item_id"]
    if _item_id in COMPACT_SOURCE_OVERRIDES:
        _spec["source"] = COMPACT_SOURCE_OVERRIDES[_item_id]
    if _item_id in SOURCE_APPENDICES:
        _spec["source"] = _spec["source"].rstrip() + SOURCE_APPENDICES[_item_id]
    if _item_id in TASK_OVERRIDES:
        _spec["task"] = TASK_OVERRIDES[_item_id]
    if _item_id in MAX_WORD_OVERRIDES:
        _spec["max_words"] = MAX_WORD_OVERRIDES[_item_id]
    if _item_id in EXPECTED_OVERRIDES:
        _spec["expected"] = EXPECTED_OVERRIDES[_item_id]
