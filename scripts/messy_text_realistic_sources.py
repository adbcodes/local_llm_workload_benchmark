from __future__ import annotations

from datetime import date
from decimal import Decimal
from textwrap import dedent
from typing import Any


SOURCE_POINTERS = {
    "sroie": {
        "dataset": "SROIE (artifact-pattern pointer; fully rewritten)",
        "url": "https://github.com/zzzDavid/ICDAR-2019-SROIE",
        "license": "Research dataset; see source repository terms",
    },
    "funsd": {
        "dataset": "FUNSD (artifact-pattern pointer; fully rewritten)",
        "url": "https://github.com/crcresearch/FUNSD",
        "license": "Research dataset; see source repository terms",
    },
    "cord": {
        "dataset": "CORD (artifact-pattern pointer; fully rewritten)",
        "url": "https://github.com/clovaai/cord",
        "license": "CC BY 4.0",
    },
}


def _gold(
    correspondent: str,
    document_type: str,
    document_date: str,
    reference_number: str,
    amount: str,
    currency: str = "INR",
) -> dict[str, Any]:
    """Compute and validate the common document-extraction gold."""

    date.fromisoformat(document_date)
    decimal_amount = Decimal(amount)
    if decimal_amount < 0:
        raise ValueError("document amount must be non-negative")
    return {
        "correspondent": correspondent,
        "document_type": document_type,
        "document_date": document_date,
        "reference_number": reference_number,
        "amount": float(decimal_amount),
        "currency": currency,
    }


def _spec(
    item_id: str,
    source_kind: str,
    source_pointer: str,
    source: str,
    *,
    correspondent: str,
    document_type: str,
    document_date: str,
    reference_number: str,
    amount: str,
    currency: str = "INR",
) -> dict[str, Any]:
    normalized_source = dedent(source).strip()
    word_count = len(normalized_source.split())
    if source_kind == "scanned_document" and not 150 <= word_count <= 400:
        raise ValueError(
            f"{item_id} scanned source has {word_count} words; expected 150-400"
        )
    return {
        "source_kind": source_kind,
        "source": normalized_source,
        "expected": _gold(
            correspondent,
            document_type,
            document_date,
            reference_number,
            amount,
            currency,
        ),
        "source_pointer": SOURCE_POINTERS[source_pointer],
        "note": (
            "Normalize document_date to YYYY-MM-DD. Use the final payable or stated "
            "document amount, not an earlier estimate, subtotal, deposit, or footer number."
        ),
    }


REALISTIC_SCHEMA_REPLACEMENTS = {
    "schema_invoice_001": _spec(
        "schema_invoice_001",
        "scanned_document",
        "sroie",
        """
        KAVERl  OFFICE  SYSTEMS  PVT. LTD.                 Page 1 / 2
        44, Residency Rd., Bengaluru 560 025
        GSTlN 29AAHCK7319Q1Z6       Helpdesk: 080 4127 0938

                         TAX  INV0ICE
        lnvoice No : KOS/INV/26-27/1847       Date : 14-08-2026
        Customer copy / scanned at 200 dpi

        Bill To
        Saffron Analytics LLP, 2nd Floor, C.V. Raman Nagar, Bengaluru
        Buyer GSTIN 29AAXFS4082R1ZP     PO ref SA-PO-918 (not our invoice ref)

        Descriptlon                         Qty      Rate        Amount
        Ergonomic keyboard, wired            7     1,485.00     10,395.00
        USB-C docking stat-                   3     2,150.00      6,450.00
        ion with 90 W adapter
        Freight / local deliv-                1       380.00        380.00
        ery charge

        Taxable value                                            17,225.00
        CGST 4.5%                                                    775.13
        SGST 4.5%                                                    775.12
        round off                                                     -0.05
        Less: advance received 01/08/26                              31.60

        AMOUNT PAYABLE                                      Rs. 18,743.60
        Amount in words: Eighteen thousand seven hundred forty-three rupees
        and sixty paise only.

        Please remit to account ending 2081. Earlier pro-forma PF-611 showed
        Rs 19,120.00; that was only an estimate and is cancelled. Goods once
        opened are subject to the warranty terms on page 2. Interest at 18%
        p.a. may apply after 30 days. This is a computer-generated invoice;
        the faint blue stamp near the margin is not a payment receipt.

        Rece1ved by: ____________        for Kaveri Office Systems Pvt Ltd
        Scan batch 08-26 / image 0041 / 2                 1
        """,
        correspondent="Kaveri Office Systems Pvt Ltd",
        document_type="invoice",
        document_date="2026-08-14",
        reference_number="KOS/INV/26-27/1847",
        amount="18743.60",
    ),
    "schema_event_002": _spec(
        "schema_event_002",
        "scanned_document",
        "funsd",
        """
        SAHYADRl LEARNING F0UNDATION                 FORM WS / rev. 4
        Community Skills Centre, Baner, Pune - 411045

                   W0RKSHOP REGISTRATlON & FEE ACKNOWLEDGEMENT

        Ref. number: SLF/WS/2026/093          form date 19 AUG 2026
        (office use only) batch 7B             page  1 of 1

        Participant name: Kavya Krishnan
        Programme selected: Practical Excel for Small Shops
        Session: Sunday, 23/08/2026, 10.00 am to 4.30 pm
        Venue: Training Hall 2, not the old Aundh classroom printed below.
        Contact written by applicant: 98 220 41 908

        FEE DETAILS
        Standard programme fee                                 Rs 1,500.00
        neighbourhood association concession                  - Rs 250.00
        AM0UNT T0 BE PAID / received                           Rs 1,250.00
        payment mode: UPI      txn suffix: 66319     status: received

        The Rs 300 shown on the tear-off lunch coupon is a refundable crockery
        deposit and is not part of this form amount. Bring one identity document.
        Materials will be shared digitally; printed handouts are optional.

        Applicant note, faint pencil:
        "pls send map on whatsapp, first time coming from Pimpri"

        Old venue (STRUCK OUT): 18 ITI Road, Aundh.
        Current venue confirmed by coordinator: Baner centre, Hall 2.

        Cancellation requests received after 21 August may be moved to a later
        batch but are not automatically refunded. The founda- tion does not ask
        for card PINs or OTPs. For timetable changes call the number on our site.

        Checked by  M. Jadhav              cashier initials: mj
        receipt printer counter 000093 / scan 1mage 018       -- 4 --
        """,
        correspondent="Sahyadri Learning Foundation",
        document_type="workshop registration form",
        document_date="2026-08-19",
        reference_number="SLF/WS/2026/093",
        amount="1250.00",
    ),
    "schema_contact_001": _spec(
        "schema_contact_001",
        "scanned_document",
        "funsd",
        """
        VlDARBHA CO-OPERATIVE BANK LTD.              Nagpur Civil Lines Branch
        (scheduled bank)                              scan copy - customer file

        Date: 28 JUL 2026                    Our Ref: VCB/KYC/NGP/7719

        To,
        Ms Farah Qureshi
        17, New Colony, Sadar, Nagpur 440001

        Subject: account contact and KYC verification acknowledgement

        Dear Madam,

        We refer to the contact update form handed over at counter 4 on 25 July.
        Your mobile number ending 7318 and email address beginning farah.q have
        been recorded against savings account ending 0062. The address proof was
        readable although the reverse side of the scan was faint. No fresh nominee
        instruction was included; the nomination already on file remains unchanged.

        A service charge of Rs. 100.00 plus GST Rs. 18.00 was debited for the
        attested paper statement requested with this update. TOTAL CHARGE POSTED:
        INR 118.00. The balance printed by the passbook kiosk, INR 62,911.45, is
        informational and is not the amount of this letter.

        Please verify the masked details above. If either contact is incorrect,
        visit the branch with original identification within seven working days.
        We will never ask for an 0TP, card PIN, or internet-banking password by
        telephone. Calls from 0712-410-xxxx may be recorded for quality review.

        This acknowledgement does not certify account balance, tax residence, or
        credit eligibility. A stamped statement, if requested, will be sent separ-
        ately. The lower corner contains a scanner routing code only.

        Yours faithfully,
        Branch Operations Manager
        Vidarbha Co-operative Bank Ltd.
        Doc queue KYC-14     0007719     Page l
        """,
        correspondent="Vidarbha Co-operative Bank Ltd",
        document_type="KYC verification acknowledgement",
        document_date="2026-07-28",
        reference_number="VCB/KYC/NGP/7719",
        amount="118.00",
    ),
    "schema_product_001": _spec(
        "schema_product_001",
        "scanned_document",
        "cord",
        """
        DECCAN APPLIANCE CARE                           SERVICE FORM
        Authorised workshop: Camp Road, Pune 411001
        Tel 020-6712 8840      GSTlN 27AARFD1198L1ZH

        WARRAN- TY CLAlM / PAID REPAIR ESTIMATE
        Document date 06/08/2026             Ref DAC/SVC/88431

        Customer: Nikhil Batra              mobile ... 5042
        Appliance: countertop mixer         model MX-730
        Serial read from label: MX73O-PN-4418   (letter O may look like zero)
        Purchased 17/02/2024. Warranty expired; customer informed at 12:16.

        Complaint as written: "motor smells after 2 mins, jar is ok, pls check
        before replacing anything. machine used mostly for dosa batter"

        TECHNICIAN FINDlNG
        carbon brush worn; coupler cracked. Armature tested within range. Old job
        card 87002 mentioned a noisy bearing, but no bearing fault was found today.

        Parts: brush pair                         Rs   620.00
        drive coupler                             Rs   480.00
        labour / cleaning                         Rs 1,100.00
        pickup charge                             Rs   160.00
        FINAL ESTIMATED AM0UNT                    Rs 2,360.00

        The crossed-out figure Rs 2,840.00 included a replacement jar that the
        customer declined. Approval received by SMS at 14:07 for the revised amount.
        Payment is due after testing and before delivery; this form is not a tax
        invoice or proof of payment. Replaced parts will be returned if requested.

        Expected completion: 09 Aug after 5 pm, subject to parts arriving. Storage
        fee of Rs 75/day starts only seven days after the completion call and is
        not included above. Keep the claim slip; workshop staff cannot retrieve
        goods using a photo of the barcode alone.

        tech: RP       counter: 03       scanner page 01 / 01       88431
        """,
        correspondent="Deccan Appliance Care",
        document_type="paid repair estimate",
        document_date="2026-08-06",
        reference_number="DAC/SVC/88431",
        amount="2360.00",
    ),
    "schema_shipment_001": _spec(
        "schema_shipment_001",
        "scanned_document",
        "sroie",
        """
        K0NKAN PARCEL NETWORK                         DELIVERY ADVICE
        Hub 04, Turbhe MIDC, Navi Mumbai             copy: consignee
        GSTIN 27AAGFK5021M1Z8

        Advice no. KPN/DA/2608/5192             Date 11-08-2026
        Waybill KPN88419031                      route BOM - GOI

        CONSIGNOR
        Meraki Retail Fixtures, Bhiwandi
        CONSIGNEE
        Caju Corner Foods Pvt Ltd, Verna Industrial Estate, Goa 403722

        6 packages / actual wt. 184.5 kg / charged wt. 192 kg
        Contents declared: powder-coated display shelves, knocked down
        Vehicle MH 04 LT 7319      seal 09148      dock slot D-17

        Freight                                     INR 5,760.00
        fuel adjustment                                460.80
        handling / floor delivery                      320.00
        CGST                                            149.60
        SGST                                            149.60
        FINAL AMOUNT DUE ON DELlVERY                 INR 6,840.00

        An older routing print attached behind this page shows INR 6,512.00 and
        destination Panaji depot. Ignore it: floor delivery to Verna was added by
        amendment at 18:42 on 10 August. Do not collect the COD value INR 41,300;
        that is the consignee's goods value for insurance, not our charge.

        Delivery window requested 12 Aug, 14:00-17:00. Call site contact only after
        reaching the security gate. Forklift not available; tail-lift vehicle was
        confirmed. Receiver should write package count before signing. Shortage or
        visible damage must be noted on both copies; remarks sent later by email may
        not be accepted by the carrier.

        POD upload token 5192-A (not the advice reference). Terms continue on the
        reverse. Generated 11/08/26 07:03; scan desk timestamp 07:19.

        authorised by: logistics desk       pg l of 2       image 0006
        """,
        correspondent="Konkan Parcel Network",
        document_type="delivery advice",
        document_date="2026-08-11",
        reference_number="KPN/DA/2608/5192",
        amount="6840.00",
    ),
    "schema_booking_001": _spec(
        "schema_booking_001",
        "scanned_document",
        "cord",
        """
        NlLGIRI TRAILS RESIDENCY                    RESERVATlON CONFIRMATION
        Coonoor Road, Ooty 643001      reception +91 423 244 8016

        Confirmation printed: 09/08/2026       Ref NTR/CONF/66281
        Source: telephone booking / desk agent 12

        Guest: Dr. Ananya Sethi
        Arrival 14 AUG 2026 after 13:00        Departure 17 AUG before 11:00
        1 family room, 3 nights, 2 adults + one child age 7
        Meal plan: breakfast. Airport transfer: not requested.

        Room tariff 3 x INR 4,200.00                         12,600.00
        extra child breakfast 3 x 280.00                       840.00
        heritage levy                                           372.00
        taxes                                                    950.00
        FINAL BOOKlNG AMOUNT                               INR 14,762.00
        Advance received by UPI on 09/08                    INR 5,000.00
        BALANCE AT CHECK-IN                                 INR 9,762.00

        For company reimbursement, the hotel will certify the full booking amount;
        the balance line only shows what remains to be paid at check-in. A tentative
        garden-view upgrade at INR 1,800 per night is waitlisted and has not been
        added. The handwritten number 204 near the fold is the requested room
        location, not a charge or confirmation number.

        Guest note: late lunch may be needed; travelling from Coimbatore by road.
        Property note: the upper driveway is closed for repairs, so cars should use
        the east gate. Government photo identification is required for every adult.
        Outside food can be stored at reception but the room refrigerators are not
        guaranteed. Wi-Fi coverage is weaker in the old wing.

        Cancellation without charge until 12 Aug 18:00 IST. One night's tariff may
        be charged after that time. This paper was folded before scanning; letters
        at the centre line may be incomplete.

        desk sign: LK       terminal 02       page 1/1       006628l
        """,
        correspondent="Nilgiri Trails Residency",
        document_type="reservation confirmation",
        document_date="2026-08-09",
        reference_number="NTR/CONF/66281",
        amount="14762.00",
    ),
    "schema_ci_run_001": _spec(
        "schema_ci_run_001",
        "email_body",
        "sroie",
        """
        From: billing@asterlinelabs.in
        To: accounts-payable@mirrormesh.io
        Cc: lab-ops@mirrormesh.io
        Date: Sun, 2 Aug 2026 18:47:22 +0530
        Message-ID: <ALS-60731-20260802@mailer.asterlinelabs.in>
        Subject: invoice ALS-INV-60731 | July cold-chain supplies

        hi team,

        attaching the final July invoice requested by Rukmini. The signed delivery
        sheets are in the same PDF after page 2; the spreadsheet is only a packing
        reconciliation and should not be booked separately.

        ASTERLINE LAB SUPPLIES PVT LTD
        Tax invoice ALS-INV-60731
        Invoice date 31 July 2026                  INR 32,864.25 due

        This includes the corrected insulated-container quantity. The draft shared
        on Friday showed INR 34,102.25 because two returned gel-pack crates had not
        yet been credited. PO MM-LAB-2098 and delivery note DN-44812 are quoted for
        matching, but neither is the invoice reference.

        Could you queue it in the 10 August payment run? No need to reply-all unless
        something is missing. Our bank details have not changed; please call the
        number on your vendor master if any message asks you to use another account.

        thanks,
        Charu
        Receivables | Asterline Lab Supplies
        +91 80 4019 2286

        -- This email and attachments may contain confidential commercial material.
        If you received it in error, delete it and inform the sender. Virus scanning
        is the recipient's responsibility. Ticket footer: mailgw-18 / route 77204.

        > On Fri, 31 Jul at 4:12 pm, MirrorMesh AP wrote:
        > Please send one final invoice after the returns credit. The figure in the
        > earlier mail will stay on hold and must not be processed.
        """,
        correspondent="Asterline Lab Supplies Pvt Ltd",
        document_type="tax invoice",
        document_date="2026-07-31",
        reference_number="ALS-INV-60731",
        amount="32864.25",
    ),
    "schema_access_request_001": _spec(
        "schema_access_request_001",
        "email_body",
        "cord",
        """
        Return-Path: <reservations@mangotreestays.in>
        Received: from mx7.mangotreestays.in by mail.kiteworks.test; 6 Aug 2026 09:04 IST
        From: Mango Tree Stays <reservations@mangotreestays.in>
        To: ishita.rao@example.net
        Date: Thu, 06 Aug 2026 09:03:41 +0530
        Subject: your Kochi booking is confirmed — MTS-KOC-88419

        Hello Ishita,

        all set for your Fort Kochi stay. Please keep this email or the attached PDF
        handy at check-in.

        MANGO TREE STAYS — BOOKING CONFIRMATION
        MTS-KOC-88419                              confirmed 06 August 2026
        Fort Kochi courtyard room, 3 nights       total INR 11,946.00

        Stay: 21 Aug to 24 Aug 2026, three nights, one courtyard room for two guests.
        The INR 3,500 card authorisation taken today is a part-payment, not the total.
        Remaining INR 8,446 is payable at the property. A boat pickup was discussed
        on chat but is not confirmed and has not been charged.

        Check-in starts at 2 pm. If the train is late, just message reception before
        9 pm so the night guard knows. Breakfast is included; lunch isn't. The lane
        is narrow and app cabs usually stop near the pharmacy about 80 metres away.

        We previously sent quote Q-7713 for INR 12,540. That quote expired when the
        courtyard-room promotion was applied, so please don't use it for expenses.

        Warmly,
        Firoz, reservations desk
        Mango Tree Stays, Kochi

        Manage preferences | privacy policy | booking help
        This transactional message was sent because a reservation was completed.
        Email tracking id 444901; footer year 2026; do not treat either as a booking ref.

        ----- Forwarded chat excerpt -----
        guest: can you hold a room till evening?
        desk: yes, once payment link succeeds we'll send a proper confirmation.
        """,
        correspondent="Mango Tree Stays",
        document_type="booking confirmation",
        document_date="2026-08-06",
        reference_number="MTS-KOC-88419",
        amount="11946.00",
    ),
    "schema_subscription_001": _spec(
        "schema_subscription_001",
        "mixed_language",
        "funsd",
        """
        NARMADA BROADBAND SERVICES
        From: collections@narmadabroadband.in
        To: rohit.kulkarni@example.com
        Subject: account NB-77104 — August payment reminder
        Sent: 16/08/2026 08:12 IST

        namaste Rohit ji,

        ye sirf payment reminder hai; aapka connection abhi active hai. July ka
        receipt already post ho gaya tha, so please usko dobara pay mat kijiye.
        Current reminder / भुगतान विवरण
        Narmada Broadband Services
        Bill date / दिनांक: 14/08/2026
        Bill ref: NBS/BLR/2608/77104              abhi due: INR 2,187.40

        Breakdown: home fibre plan INR 1,899, router protection INR 149, tax and
        rounding INR 139.40. The app may also show wallet balance INR 312.60; that
        is credit available for a later cycle and has not been deducted from this
        reminder. A previous SMS mentioned INR 2,336.40 before the loyalty adjustment.

        payment ho chuka hai toh ignore kar dena, update ko 24 ghante lag sakte hain.
        Otherwise use the customer app or the UPI handle printed on your signed bill.
        We never ask for OTP or screen sharing. Service suspension review starts only
        after 22 August, not on the date of this email.

        Regards,
        Kavita | customer accounts
        Narmada Broadband Services, Indore

        This message includes automated footer numbers: campaign 208, node 17,
        template v3.4. They are not account or reference identifiers. Please do not
        print unless required; email delivery status: accepted.
        """,
        correspondent="Narmada Broadband Services",
        document_type="payment reminder",
        document_date="2026-08-14",
        reference_number="NBS/BLR/2608/77104",
        amount="2187.40",
    ),
    "schema_address_001": _spec(
        "schema_address_001",
        "mixed_language",
        "cord",
        """
        ARAVALI PUBLIC SCHOOL, Jaipur
        Parent portal copy / कक्षा शुल्क सूचना
        generated 05/08/2026 17:42        page 1

        Dear Mrs Meenal Soni,

        This is the fee notice for Aarav Soni, admission APS-19-448, Class VII-B.
        कृपया नीचे दी गई अंतिम राशि 12 अगस्त तक जमा करें। Cash is accepted only at
        the school counter between 9:30 and 12:30; online payment remains available
        through the parent portal.

        TERM 2 FEE NOTICE / शुल्क सूचना
        Aravali Public School
        Notice date: 03/08/2026       Ref. APS/FEE/T2/26448
        Amount due by 12 August: INR 15,480.00

        Tuition for term 2 is INR 12,900, activity fee INR 1,350, and transport for
        August INR 1,230. Library deposit INR 2,000 appears in the student's profile
        but was paid at admission and is not due again. The uniform shop balance of
        INR 740 belongs to a separate vendor and is also excluded.

        Agar payment already kar diya hai, receipt number portal par check karke is
        notice ko ignore kar sakte hain. Please use the notice reference above for
        any bank transfer; admission number is only the student identifier. Late fee
        is not included yet and would apply after the due date.

        The first PDF issued on 02 August had the wrong bus stop and an amount of
        INR 16,110. It was withdrawn. This corrected notice uses Vaishali Nagar stop
        4 and the final amount shown above.

        Accounts Office
        Aravali Public School
        phone queue 6 | print batch 112 | portal session expired after 10 minutes
        """,
        correspondent="Aravali Public School",
        document_type="term fee notice",
        document_date="2026-08-03",
        reference_number="APS/FEE/T2/26448",
        amount="15480.00",
    ),
}
