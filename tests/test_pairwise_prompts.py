from __future__ import annotations

import json

import pytest

from littraceqa.answer_derivation import validate_answer_semantics
from littraceqa.di_pipeline.contracts import Query
from littraceqa.pairwise_prompts import (
    ANSWER_EXAMPLES,
    ANSWER_PROMPT_VERSION,
    JUDGMENT_EXAMPLES,
    JUDGMENT_PROMPT_VERSION,
    example_manifest,
    render_answer_prompt,
    render_judgment_prompt,
    selected_answer_examples,
    selected_judgment_examples,
)


_COMPLETE_RESPONSE_MARKER = "Complete response object:\n"
_COMPLETE_RESPONSE_CASES = (
    (
        "A1_reported_over_recomputed",
        "freeform+multiple_choice",
        {"answer.freeform.text", "answer.multiple_choice"},
    ),
    (
        "A2_yes_no_polarity",
        "multiple_choice",
        {"answer.multiple_choice"},
    ),
    (
        "A8_multi_paper_owner_completeness",
        "table",
        {"answer.table.rows[0]", "answer.table.rows[1]"},
    ),
    (
        "A11_variable_option_labels",
        "multiple_choice",
        {"answer.multiple_choice"},
    ),
    (
        "A14_combined_freeform_table",
        "freeform+table",
        {
            "answer.freeform.text",
            "answer.table.rows[0]",
            "answer.table.rows[1]",
        },
    ),
)


def _complete_response(example_id: str) -> dict:
    example = next(
        item for item in ANSWER_EXAMPLES if item.example_id == example_id
    )
    assert example.body.count(_COMPLETE_RESPONSE_MARKER) == 1
    encoded = example.body.split(_COMPLETE_RESPONSE_MARKER, maxsplit=1)[1].strip()
    parsed = json.loads(encoded)
    assert isinstance(parsed, dict)
    return parsed


def _figure_query() -> Query:
    return Query(
        query_id="test_figure",
        benchmark="LitTraceQA",
        question="How many subfigures are in Figure 4 of WavePipe?",
        answer_types=["freeform", "multiple_choice"],
        options={"A": "2", "B": "4", "C": "8", "E": "10"},
    )


def _table_query() -> Query:
    return Query(
        query_id="test_table",
        benchmark="LitTraceQA",
        question="Across 2025 papers, what score and pass status does each method report?",
        answer_types=["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
            {"name": "Passed", "type": "boolean", "is_row_key": False},
        ],
    )


def test_few_shot_selection_is_stable_bounded_and_query_aware():
    query = _figure_query()

    judgment_first = selected_judgment_examples(query)
    judgment_second = selected_judgment_examples(query)
    answer = selected_answer_examples(query)

    judgment_ids = [item.example_id for item in judgment_first]
    answer_ids = [item.example_id for item in answer]
    assert judgment_first == judgment_second
    assert 3 <= len(judgment_ids) <= 9
    assert 4 <= len(answer_ids) <= 12
    assert len(judgment_ids) == len(set(judgment_ids))
    assert len(answer_ids) == len(set(answer_ids))
    assert "J1_wrong_owner_same_figure_number" in judgment_ids
    assert "J5_visual_required_but_missing" in judgment_ids
    assert "J6_visual_panel_count" in judgment_ids
    assert "A3_count_consistency" in answer_ids
    assert "A11_variable_option_labels" in answer_ids
    assert "A12_missing_image" in answer_ids


def test_table_query_selects_native_type_and_multi_paper_examples():
    ids = [item.example_id for item in selected_answer_examples(_table_query())]

    assert "A9_native_table_types" in ids
    assert "A8_multi_paper_owner_completeness" in ids
    assert "A13_wrong_setting_omitted" not in ids

    constrained_query = Query(
        "constrained_table",
        "Across datasets, return each model row only for the requested split.",
        ["table"],
        table_schema=_table_query().table_schema,
    )
    constrained_ids = [
        item.example_id for item in selected_answer_examples(constrained_query)
    ]
    assert "A13_wrong_setting_omitted" in constrained_ids


def test_specialised_examples_require_all_tags_and_avoid_substring_false_positives():
    images_query = Query(
        "images_model",
        "Which Images model reports the requested value?",
        ["freeform"],
    )
    judgment_ids = {
        item.example_id for item in selected_judgment_examples(images_query)
    }
    assert "J1_wrong_owner_same_figure_number" not in judgment_ids
    assert "J6_visual_panel_count" not in judgment_ids

    reference_free_query = Query(
        "reference_free",
        "Across reference-free methods, list each method.",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True}
        ],
    )
    judgment_ids = {
        item.example_id
        for item in selected_judgment_examples(reference_free_query)
    }
    answer_ids = {
        item.example_id for item in selected_answer_examples(reference_free_query)
    }
    assert "J7_reference_identity" not in judgment_ids
    assert "A6_distinct_citations" not in answer_ids


def test_paper_title_ending_in_images_is_not_treated_as_a_visual_request():
    query = Query(
        "q_title_images",
        (
            "How many parentheses are in Equation 6 of Continuous Latent "
            "Dynamical Models from Images?"
        ),
        ["freeform", "multiple_choice"],
        options={"A": "2", "B": "4", "C": "6", "D": "8"},
    )

    judgment_ids = {
        item.example_id for item in selected_judgment_examples(query)
    }
    answer_ids = {item.example_id for item in selected_answer_examples(query)}

    assert "J6_visual_panel_count" not in judgment_ids
    assert "A3_count_consistency" not in answer_ids
    assert "A2_yes_no_polarity" not in answer_ids
    assert "A7_literal_parenthesis_pairs" in answer_ids


@pytest.mark.parametrize(
    "question",
    [
        "How many parameters does the image generation model use?",
        "How many layers does the graph neural network contain?",
        "How many stages does the figure generation model use?",
        "What is the score in the image-missing condition?",
        "Which encoder builds the KNN graph in the graph-score method?",
        "What decrease does the refined FLUX image show relative to its origin?",
        "How many frames are used in the chart-attribution annotation pipeline?",
    ],
)
def test_visual_topic_terms_do_not_select_visual_count_examples(question):
    query = Query(
        "q_visual_topic",
        question,
        ["freeform", "multiple_choice"],
        options={"A": "2", "B": "4"},
    )

    judgment_ids = {
        item.example_id for item in selected_judgment_examples(query)
    }
    answer_ids = {item.example_id for item in selected_answer_examples(query)}

    assert "J6_visual_panel_count" not in judgment_ids
    assert "A3_count_consistency" not in answer_ids


def test_explicit_image_inspection_selects_visual_count_examples():
    query = Query(
        "q_explicit_image",
        "How many bars are visible in the provided image?",
        ["freeform", "multiple_choice"],
        options={"A": "2", "B": "4"},
    )

    judgment_ids = {
        item.example_id for item in selected_judgment_examples(query)
    }
    answer_ids = {item.example_id for item in selected_answer_examples(query)}

    assert "J6_visual_panel_count" in judgment_ids
    assert "A3_count_consistency" in answer_ids


def test_primary_framework_figure_selects_visual_examples():
    query = Query(
        "q_primary_figure",
        "Which papers mention search in their primary method/framework figure?",
        ["freeform", "table"],
        table_schema=[
            {"name": "Paper Title", "type": "string", "is_row_key": True}
        ],
    )

    judgment_ids = {
        item.example_id for item in selected_judgment_examples(query)
    }
    answer_ids = {item.example_id for item in selected_answer_examples(query)}

    assert "J1_wrong_owner_same_figure_number" in judgment_ids
    assert "J4_multi_paper_one_complete_row" in judgment_ids
    assert "A14_combined_freeform_table" in answer_ids


def test_combined_argmax_query_selects_header_alignment_example():
    query = Query(
        "q_argmax",
        (
            "In the Agent paper, which backbone model achieves the highest "
            "performance on the API benchmark when integrated with Agent?"
        ),
        ["freeform", "multiple_choice"],
        options={
            "A": "Backbone-A",
            "B": "Backbone-B",
            "C": "Backbone-C",
        },
    )

    answer_ids = {item.example_id for item in selected_answer_examples(query)}

    assert "A5_argmax_header_alignment" in answer_ids


def test_visual_count_examples_teach_units_without_validation_total():
    judgment = next(
        item for item in JUDGMENT_EXAMPLES if item.example_id == "J6_visual_panel_count"
    )
    answer = next(
        item for item in ANSWER_EXAMPLES if item.example_id == "A3_count_consistency"
    )

    for body in (judgment.body, answer.body):
        assert "independent coordinate-axes" in body
        assert "eight" not in body.casefold()
        assert '"result":8' not in body
    assert "five independent plot frames" in judgment.body
    assert '"result":5' in answer.body
    assert "answer.freeform.text" in answer.body
    assert "answer.multiple_choice" in answer.body


def test_argmax_example_uses_fully_synthetic_labels_and_binds_both_answers():
    example = next(
        item
        for item in ANSWER_EXAMPLES
        if item.example_id == "A5_argmax_header_alignment"
    )

    assert "Cedar=17, Flint=24, Quartz=19" in example.body
    assert "Backbone-A" not in example.body
    assert "answer.freeform.text" in example.body
    assert "answer.multiple_choice" in example.body


@pytest.mark.parametrize(
    "example_id,answer_key,expected_answer_paths",
    _COMPLETE_RESPONSE_CASES,
)
def test_complete_stage_two_examples_are_parseable_and_internally_linked(
    example_id: str,
    answer_key: str,
    expected_answer_paths: set[str],
):
    payload = _complete_response(example_id)

    assert set(payload) == {
        "status",
        "paper_relevance",
        "papers",
        "derivation",
        "answer",
        "support",
        "completeness",
    }
    assert payload["status"] == "ready"
    expected_answer_keys = set(answer_key.split("+"))
    assert set(payload["answer"]) == expected_answer_keys
    assert payload["completeness"]["answered_parts"]
    assert payload["completeness"]["missing"] == []

    relevance_ids = {
        item["paper_id"] for item in payload["paper_relevance"]
    }
    paper_chunks = {
        item["paper_id"]: set(item["evidence_chunk_ids"])
        for item in payload["papers"]
    }
    assert relevance_ids == set(paper_chunks)
    assert all(paper_id.startswith("syn_") for paper_id in relevance_ids)
    for paper_id, chunk_ids in paper_chunks.items():
        assert chunk_ids
        assert all(chunk_id.startswith(f"{paper_id}#") for chunk_id in chunk_ids)

    derivation = payload["derivation"]
    assert derivation["facts"]
    assert isinstance(derivation["operations"], list)
    assert derivation["answer_bindings"]
    assert derivation["final_semantic_answer"]
    for fact in derivation["facts"]:
        assert fact["id"]
        assert fact["value_kind"] in {"reported", "visual", "text"}
        assert fact["paper_id"] in paper_chunks
        assert set(fact["chunk_ids"]).issubset(paper_chunks[fact["paper_id"]])

    assert {item["answer_path"] for item in payload["support"]} == (
        expected_answer_paths
    )
    for item in payload["support"]:
        assert item["paper_id"] in paper_chunks
        assert item["chunk_ids"]
        assert set(item["chunk_ids"]).issubset(paper_chunks[item["paper_id"]])


def test_complete_stage_two_examples_cover_requested_reasoning_patterns():
    reported = _complete_response("A1_reported_over_recomputed")
    comparison = _complete_response("A2_yes_no_polarity")
    multi_table = _complete_response("A8_multi_paper_owner_completeness")
    five_option = _complete_response("A11_variable_option_labels")

    assert reported["derivation"]["operations"] == []
    assert reported["answer"]["multiple_choice"] == {
        "label": "B",
        "selected_option_text": "7.42",
    }
    assert comparison["derivation"]["operations"] == [
        {
            "id": "op_compare",
            "kind": "compare",
            "fact_ids": ["f_category_l", "f_category_r"],
            "left": 44,
            "operator": "<",
            "right": 51,
            "result": True,
            "answer_binding": {
                "answer_path": "answer.multiple_choice.selected_option_text",
                "expected": True,
                "answer_fragment": "Yes",
            },
        }
    ]
    assert multi_table["answer"]["table"]["rows"] == [
        {"System": "CedarNet", "Vocabulary Size": 48000},
        {"System": "FlintNet", "Vocabulary Size": 65536},
    ]
    assert five_option["answer"]["multiple_choice"] == {
        "label": "E",
        "selected_option_text": "Epsilon",
    }


def test_missing_image_example_is_a_complete_non_ready_contract():
    example = next(
        item for item in ANSWER_EXAMPLES if item.example_id == "A12_missing_image"
    )
    marker = "Complete non-ready response object:\n"
    payload = json.loads(example.body.split(marker, maxsplit=1)[1])

    assert payload["status"] == "needs_image"
    assert payload["paper_relevance"][0]["role"] == "target_owner"
    assert payload["papers"] == []
    assert payload["derivation"]["facts"] == []
    assert payload["answer"] == {}
    assert payload["support"] == []
    assert payload["completeness"]["missing"]


@pytest.mark.parametrize(
    ("example_id", "query"),
    [
        (
            "A1_reported_over_recomputed",
            Query(
                "syn_a1_q",
                "Which improvement is reported?",
                ["freeform", "multiple_choice"],
                options={"A": "7.43", "B": "7.42", "C": "6.42"},
            ),
        ),
        (
            "A2_yes_no_polarity",
            Query(
                "syn_a2_q",
                "Does Category L have fewer entries than Category R?",
                ["multiple_choice"],
                options={"A": "Yes", "B": "No"},
            ),
        ),
        (
            "A8_multi_paper_owner_completeness",
            Query(
                "syn_a8_q",
                "Return tokenizer vocabulary sizes.",
                ["table"],
                table_schema=[
                    {"name": "System", "type": "string", "is_row_key": True},
                    {
                        "name": "Vocabulary Size",
                        "type": "number",
                        "is_row_key": False,
                    },
                ],
            ),
        ),
        (
            "A11_variable_option_labels",
            Query(
                "syn_a11_q",
                "Which variant is final?",
                ["multiple_choice"],
                options={
                    "A": "Alpha",
                    "B": "Beta",
                    "C": "Gamma",
                    "D": "Delta",
                    "E": "Epsilon",
                },
            ),
        ),
        (
            "A14_combined_freeform_table",
            Query(
                "syn_a14_q",
                "Return a sentence and table.",
                ["freeform", "table"],
                table_schema=[
                    {"name": "Method", "type": "string", "is_row_key": True},
                    {"name": "Base Model", "type": "string", "is_row_key": False},
                ],
            ),
        ),
    ],
)
def test_complete_ready_few_shots_pass_the_runtime_semantic_validator(
    example_id: str,
    query: Query,
):
    payload = _complete_response(example_id)

    validated = validate_answer_semantics(
        query,
        derivation=payload["derivation"],
        answer=payload["answer"],
    )

    assert validated["final_semantic_answer"] == payload["derivation"][
        "final_semantic_answer"
    ]


def test_judgment_render_places_examples_before_live_data_and_marks_no_images():
    query = _figure_query()
    prompt = render_judgment_prompt(
        query=query,
        query_payload=query.to_dict(),
        candidate_payload={
            "paper_id": "paper_live",
            "rank": 3,
            "title": "WavePipe",
            "venue": "TEST",
            "year": 2025,
        },
        paper_text="[chunk live#fig4]\nLIVE_PAPER_SENTINEL",
        image_legend="",
    )

    assert JUDGMENT_PROMPT_VERSION == (
        "pairwise-paper-judge-v11-single-selected-context"
    )
    assert prompt.index("SYNTHETIC FEW-SHOT EXAMPLES") < prompt.index("LIVE TASK")
    assert prompt.index("LIVE TASK") < prompt.index("LIVE_PAPER_SENTINEL")
    assert '"multiple_choice_options"' in prompt
    assert '"label":"E","text":"10"' in prompt
    assert '"paper_id":"paper_live"' in prompt
    assert "Selected paper context" in prompt
    assert "Content not shown here is unknown" in prompt
    assert "do not infer or cite it" in prompt
    assert "Paper batch:" not in prompt
    assert "Actually attached image mapping: NONE" in prompt
    assert "Do not claim visual inspection" in prompt


def test_answer_render_includes_answer_shape_limits_and_actual_image_mapping():
    query = _table_query()
    answer_shape = {
        "table": {
            "rows": [{"Method": "<source string>", "Score": 0, "Passed": False}]
        }
    }
    prompt = render_answer_prompt(
        query=query,
        query_payload=query.to_dict(),
        accepted_summary=[{"paper_id": "p1", "label": "partial_answer"}],
        evidence_text="LIVE_EVIDENCE_SENTINEL",
        image_legend="Image 1: chunk_ids=p1#tab1 file=table.jpg",
        answer_shape=answer_shape,
        max_evidence=12,
        max_evidence_per_paper=2,
    )

    assert ANSWER_PROMPT_VERSION == (
        "accepted-evidence-answer-v19-observable-query-tags"
    )
    assert prompt.index("SYNTHETIC FEW-SHOT EXAMPLES") < prompt.index("LIVE TASK")
    assert '"Passed":false' in prompt
    assert '"Score":0' in prompt
    assert "LIVE_EVIDENCE_SENTINEL" in prompt
    assert "Actually attached image mapping" in prompt
    assert "Image 1: chunk_ids=p1#tab1 file=table.jpg" in prompt
    assert "at most 12 distinct chunk_ids total" in prompt
    assert "at most 2 per paper" in prompt


def test_example_manifest_contains_only_stable_unique_ids():
    manifest = example_manifest(_figure_query())

    assert set(manifest) == {"judgment", "answer"}
    assert manifest["judgment"] == list(dict.fromkeys(manifest["judgment"]))
    assert manifest["answer"] == list(dict.fromkeys(manifest["answer"]))
    assert all(example_id.startswith("J") for example_id in manifest["judgment"])
    assert all(example_id.startswith("A") for example_id in manifest["answer"])
