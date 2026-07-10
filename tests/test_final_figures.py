from llm_workload_benchmark.final_figures import merge_adjudication_labels


def test_adjudication_labels_override_raw_failure_without_erasing_format_tax() -> None:
    items = [
        {
            "variant_id": "qwen3-8b-q4",
            "item_id": "item-1",
            "repetition": "1",
            "passed": "False",
            "semantic_outcome": "incorrect",
            "evaluation_details": "{}",
        },
        {
            "variant_id": "qwen3-8b-q4",
            "item_id": "item-2",
            "repetition": "1",
            "passed": "True",
            "semantic_outcome": "correct",
            "evaluation_details": "{}",
        },
    ]
    adjudications = [
        {
            "model_id": "qwen3-8b-q4",
            "item_id": "item-1",
            "repetition": 1,
            "status": "completed",
            "derived": {
                "strict_pass": False,
                "semantic_correct": True,
                "format_tax": True,
            },
        }
    ]

    merged = merge_adjudication_labels(items, adjudications)

    assert merged[0]["strict_pass"] is False
    assert merged[0]["semantic_pass"] is True
    assert merged[0]["format_tax"] is True
    assert merged[0]["label_source"] == "adjudication"
    assert merged[1]["strict_pass"] is True
    assert merged[1]["semantic_pass"] is True
    assert merged[1]["format_tax"] is False
