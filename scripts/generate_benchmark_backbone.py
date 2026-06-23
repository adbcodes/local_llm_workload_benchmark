from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("data")


PLANNED: list[dict[str, Any]] = [
    {
        "id": "knowledge_abstention", "suite": "A", "title": "Knowledge and Abstention",
        "target": 30, "distribution": {"easy": 10, "medium": 15, "hard": 5},
        "types": ["answerable facts with a stated cutoff", "fabricated APIs and events", "unanswerable questions", "ambiguous questions requiring clarification"],
        "methods": ["exact_match", "behavior_rules"],
        "metrics": ["accuracy", "hallucination_rate", "abstention_accuracy", "clarification_accuracy"],
        "score_formula": "accuracy_minus_hallucination",
    },
    {
        "id": "tables_to_decisions", "suite": "B", "title": "Tables to Decisions",
        "target": 30, "distribution": {"easy": 8, "medium": 15, "hard": 7},
        "types": ["totals and differences", "small joins", "trends with exception rows", "duplicated rows", "inconsistent rows", "decision from table evidence"],
        "methods": ["numeric_tolerance", "exact_match", "set_match"],
        "metrics": ["answer_accuracy", "row_selection_accuracy", "set_accuracy"],
    },
    {
        "id": "inbox_routing", "suite": "B", "title": "Inbox Routing",
        "target": 30, "distribution": {"easy": 8, "medium": 15, "hard": 7},
        "types": ["single-label routing", "multi-label routing", "urgency under polite tone", "ask-for-more-information cases", "out-of-scope messages"],
        "methods": ["set_match", "behavior_rules"],
        "metrics": ["label_accuracy", "multilabel_jaccard", "urgency_accuracy", "clarification_accuracy"],
    },
    {
        "id": "tool_use", "suite": "B", "title": "Tool Use",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "execution_mode": "tool_scenario",
        "types": ["no-tool-needed traps", "two-call chains", "using one call result in the next", "error recovery", "argument conversion", "unnecessary-call avoidance"],
        "methods": ["tool_trace"],
        "metrics": ["tool_selection_accuracy", "argument_accuracy", "call_order_accuracy", "unnecessary_call_rate", "final_state_accuracy"],
    },
    {
        "id": "negative_instructions", "suite": "C", "title": "Negative Instructions",
        "target": 15, "distribution": {"easy": 4, "medium": 8, "hard": 3},
        "types": ["prohibited content", "prohibited words", "prohibited formats", "no questions", "no code examples"],
        "methods": ["constraint_rules"],
        "metrics": ["strict_all_pass_rate", "prohibition_compliance"],
    },
    {
        "id": "instruction_hierarchy", "suite": "C", "title": "Instruction Hierarchy",
        "target": 15, "distribution": {"easy": 4, "medium": 8, "hard": 3},
        "types": ["system versus user conflict", "injected instructions inside source text", "quoted instructions", "authority-preserving extraction"],
        "methods": ["exact_match", "behavior_rules", "json_exact"],
        "metrics": ["authority_accuracy", "injection_resistance"],
    },
    {
        "id": "raw_output_discipline", "suite": "C", "title": "Raw Output Discipline",
        "target": 10, "distribution": {"easy": 3, "medium": 5, "hard": 2},
        "types": ["raw JSON", "raw CSV", "single labels", "no preamble", "no code fences", "no trailing explanation"],
        "methods": ["json_exact", "exact_match", "constraint_rules"],
        "metrics": ["parseability", "fence_wrapping_rate", "extra_chatter_rate", "value_accuracy"],
    },
    {
        "id": "india_focused_tasks", "suite": "D", "title": "India-Focused Tasks",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "types": ["lakh and crore arithmetic", "GST", "Indian dates", "Indian addresses", "Hinglish instructions"],
        "methods": ["numeric_tolerance", "date_value", "json_exact", "llm_judge"],
        "metrics": ["accuracy", "format_accuracy", "judge_score"],
    },
    {
        "id": "long_text_retrieval", "suite": "E", "title": "Long-Text Retrieval",
        "target": 27, "distribution": {"easy": 9, "medium": 9, "hard": 9},
        "execution_mode": "paired_variants",
        "types": ["fact at start", "fact in middle", "fact at end", "1K context", "4K context", "8K context"],
        "methods": ["exact_match", "numeric_tolerance", "date_value"],
        "metrics": ["accuracy", "retained_score", "accuracy_by_position", "accuracy_by_context_length"],
    },
    {
        "id": "conversation_memory", "suite": "E", "title": "Conversation Memory",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "execution_mode": "multi_turn",
        "types": ["fact retention", "format retention", "preference retention", "correction retention", "turn-5 probes", "turn-10 probes"],
        "methods": ["exact_match", "json_exact", "constraint_rules"],
        "metrics": ["turn_5_accuracy", "turn_10_accuracy", "retained_score"],
    },
    {
        "id": "clean_vs_noisy", "suite": "E", "title": "Clean vs Noisy",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "execution_mode": "paired_variants",
        "types": ["typos", "OCR noise", "paraphrase", "irrelevant text", "distracting numbers"],
        "methods": ["exact_match", "numeric_tolerance", "json_exact", "set_match"],
        "metrics": ["clean_score", "noisy_score", "retained_score"],
    },
    {
        "id": "false_missing_information", "suite": "E", "title": "False or Missing Information",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "types": ["false premise", "fabricated entity", "conflicting facts", "unanswerable question"],
        "methods": ["behavior_rules"],
        "metrics": ["behavior_accuracy", "hallucination_rate"],
    },
    {
        "id": "answer_stability", "suite": "E", "title": "Answer Stability",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "execution_mode": "multi_turn",
        "types": ["are-you-sure challenge", "confident wrong suggestion", "correct-to-wrong flip", "wrong-to-correct self-correction", "three repeated runs"],
        "methods": ["exact_match", "numeric_tolerance", "date_value"],
        "metrics": ["stood_by_correct_rate", "sycophancy_rate", "self_correction_rate", "run_to_run_flip_rate"],
    },
    {
        "id": "confidence_correctness", "suite": "E", "title": "Confidence vs Correctness",
        "target": 48, "distribution": {"easy": 12, "medium": 24, "hard": 12},
        "execution_mode": "paired_variants",
        "types": ["reasoning confidence", "coding confidence", "confidence under ambiguity", "confidence after challenge"],
        "methods": ["confidence_value"],
        "metrics": ["accuracy", "brier_score", "expected_calibration_error"],
    },
    {
        "id": "shuffled_choices", "suite": "E", "title": "Shuffled Choices",
        "target": 20, "distribution": {"easy": 5, "medium": 10, "hard": 5},
        "execution_mode": "paired_variants",
        "types": ["gold answer in each position", "light choice paraphrase", "mapped answer labels"],
        "methods": ["exact_match"],
        "metrics": ["mapped_accuracy", "position_bias_delta"],
    },
    {
        "id": "prompt_format_sensitivity", "suite": "E", "title": "Prompt-Format Sensitivity",
        "target": 18, "distribution": {"easy": 6, "medium": 6, "hard": 6},
        "execution_mode": "paired_variants",
        "types": ["plain prose", "Markdown", "XML tags", "system-turn instruction", "user-turn instruction"],
        "methods": ["exact_match", "json_exact", "constraint_rules"],
        "metrics": ["format_variant_accuracy", "retained_score"],
    },
    {
        "id": "over_refusal", "suite": "E", "title": "Over-Refusal",
        "target": 10, "distribution": {"easy": 3, "medium": 5, "hard": 2},
        "types": ["kill a process", "strip a wire", "inject a dependency", "benign security terminology", "benign medical terminology"],
        "methods": ["behavior_rules"],
        "metrics": ["benign_completion_rate", "over_refusal_rate"],
    },
]


SUITES = {
    "suite_a_core.yaml": ["applied_reasoning", "code_debug_repair", "knowledge_abstention"],
    "suite_b_structured.yaml": ["messy_text_to_schema", "tables_to_decisions", "inbox_routing", "tool_use"],
    "suite_c_instruction.yaml": ["constraint_load_curve", "negative_instructions", "instruction_hierarchy", "raw_output_discipline"],
    "suite_d_communication.yaml": ["grounded_compression", "india_focused_tasks"],
    "suite_e_reliability.yaml": ["long_text_retrieval", "conversation_memory", "clean_vs_noisy", "false_missing_information", "answer_stability", "confidence_correctness", "shuffled_choices", "prompt_format_sensitivity", "over_refusal"],
}


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def _definition(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "title": spec["title"],
        "description": f"Planned {spec['title'].casefold()} question set from the final benchmark plan.",
        "suite": spec["suite"],
        "status": "planned",
        "execution_mode": spec.get("execution_mode", "single_turn"),
        "task_types": spec["types"],
        "metrics": spec["metrics"],
        "score_formula": spec.get(
            "score_formula",
            "clean_score_retained" if spec["suite"] == "E" else "mean_score",
        ),
        "items_path": "items.jsonl",
        "authoring_paths": ["questions.yaml"],
        "current_question_count": 0,
        "target_question_count": spec["target"],
        "current_difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
        "difficulty_distribution": spec["distribution"],
        "target_visibility_distribution": {
            "public": (spec["target"] + 1) // 2,
            "held_out": spec["target"] // 2,
        },
        "order_rule": "easy_to_hard",
        "scoring_methods": spec["methods"],
    }


def _item_template(spec: dict[str, Any]) -> dict[str, Any]:
    method = spec["methods"][0]
    response_type = "text"
    expected: Any = "replace_with_gold_answer"
    parameters: dict[str, Any] = {}
    if method == "numeric_tolerance":
        response_type = "number"
        expected = 0
        parameters = {"absolute_tolerance": 0}
    elif method == "tool_trace":
        response_type = "json"
        expected = {
            "calls": [{"tool": "replace_tool", "arguments": {"replace_argument": "value"}}],
            "observations": [{"replace_result": "value_returned_by_fake_tool"}],
            "final_state": {"replace_state": True},
        }
    elif method == "behavior_rules":
        expected = {
            "label": "replace_behavior_label",
            "required_any": ["replace accepted phrase"],
            "forbidden": ["replace forbidden phrase"],
        }
    elif method == "confidence_value":
        expected = {"answer": "replace_with_gold_answer"}
        parameters = {"answer_type": "exact", "case_sensitive": True}
    elif method == "constraint_rules":
        parameters = {
            "content_requirements": {"none": True},
            "rules": {"forbidden_terms": ["replace_forbidden_term"]},
        }
    conversation = (
        [
            {"role": "user", "content": "Replace with an earlier turn."},
            {"role": "assistant", "content": "Replace with the fixed earlier reply."},
            {"role": "user", "content": "Replace with the scored probe turn."},
        ]
        if spec.get("execution_mode") == "multi_turn"
        else None
    )
    template = {
        "id": f"{spec['id']}_replace_001",
        "subcategory": "replace_with_one_declared_task_type",
        "difficulty": "easy",
        "split": "dev",
        "visibility": "public",
        "prompt": "Replace with the complete question shown to the model.",
        "response_contract": {"type": response_type, "format": None},
        "expected": {"value": expected},
        "scoring": {"method": method, "parameters": parameters},
        "provenance": {"kind": "hand_authored", "review_status": "draft"},
        "tags": ["replace_tag"],
    }
    if conversation is not None:
        template["conversation"] = conversation
    if spec.get("execution_mode") == "paired_variants":
        template["source_item"] = "replace_with_clean_item_id"
    return template


def generate(root: Path = ROOT) -> None:
    for spec in PLANNED:
        directory = root / spec["id"]
        _write_yaml(directory / "benchmark.yaml", _definition(spec))
        _write_yaml(
            directory / "questions.yaml",
            {
                "schema_version": 1,
                "benchmark": spec["id"],
                "item_template": _item_template(spec),
                "items": [],
            },
        )
        (directory / "items.jsonl").write_text("", encoding="utf-8")

    for filename, benchmark_ids in SUITES.items():
        _write_yaml(
            root / "suites" / filename,
            {
                "schema_version": 1,
                "name": filename.removesuffix(".yaml").replace("_", "-"),
                "version": 1,
                "status": "pilot",
                "benchmark_files": [f"../{benchmark_id}/benchmark.yaml" for benchmark_id in benchmark_ids],
            },
        )

    all_ids = [benchmark_id for ids in SUITES.values() for benchmark_id in ids]
    _write_yaml(
        root / "suites" / "all.yaml",
        {
            "schema_version": 1,
            "name": "local-llm-all-benchmarks",
            "version": 1,
            "status": "pilot",
            "benchmark_files": [f"../{benchmark_id}/benchmark.yaml" for benchmark_id in all_ids],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    generate(parser.parse_args().root)
