from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Query
from littraceqa.query_requirements import (
    QUERY_REQUIREMENTS_VERSION,
    explicit_table_row_items,
    missing_explicit_table_items,
    table_output_contract,
    unaccounted_explicit_table_items,
)


def _query(question: str, *, table: bool = True) -> Query:
    return Query(
        query_id="q_test",
        question=question,
        answer_types=["table"] if table else ["freeform"],
        table_schema=(
            [
                {"name": "Method", "type": "string", "is_row_key": True},
                {"name": "Value", "type": "string", "is_row_key": False},
            ]
            if table
            else []
        ),
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What are the scores for TCM, sCT, ECM-XL (100k iterations), and IMM?",
            ("TCM", "sCT", "ECM-XL (100k iterations)", "IMM"),
        ),
        (
            "What percentage is reported in SUN RGB-D, ARKitScenes, KITTI, and nuScenes?",
            ("SUN RGB-D", "ARKitScenes", "KITTI", "nuScenes"),
        ),
        (
            "What are the accuracies of NCFM, AP-BPTT, ATT, and DEDA given IPC=10?",
            ("NCFM", "AP-BPTT", "ATT", "DEDA"),
        ),
        (
            "What is the performance with ground-truth prompts and Cube R-CNN 2D detections?",
            ("ground-truth prompts", "Cube R-CNN 2D detections"),
        ),
    ],
)
def test_extracts_confident_explicit_inventory(
    question: str, expected: tuple[str, ...]
) -> None:
    assert explicit_table_row_items(_query(question)) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Which papers cite UniAD in their main comparison table?",
        "What training objective does each proposed method optimize?",
        "Who is the first author?",
        (
            "Among papers that propose an objective for LLM alignment without a "
            "reference model, what does each method optimize?"
        ),
        (
            "Among methods for text-to-image generation evaluated on GenEval, "
            "what base model does each method build on?"
        ),
        (
            "What recurrence does each method define for producing the next "
            "state from the previous state and the current input?"
        ),
    ],
)
def test_open_ended_or_non_list_question_has_no_inventory(question: str) -> None:
    assert explicit_table_row_items(_query(question)) == ()


def test_non_table_question_has_no_inventory() -> None:
    assert explicit_table_row_items(
        _query("What are the values for A, B, and C?", table=False)
    ) == ()


@pytest.mark.parametrize(
    "question",
    [
        "What are the scores for Cedar and Flint and Larch?",
        "What are the scores for Cedar, Flint and Larch?",
    ],
)
def test_three_item_non_oxford_inventory_keeps_each_row_independent(
    question: str,
) -> None:
    query = _query(question)

    assert explicit_table_row_items(query) == ("Cedar", "Flint", "Larch")
    assert missing_explicit_table_items(
        query,
        [
            {"Method": "Cedar", "Value": "1"},
            {"Method": "Flint", "Value": "2"},
        ],
    ) == ("Larch",)


def test_table_output_contract_is_schema_and_question_derived() -> None:
    query = _query("What are the scores for Cedar, Flint, and Larch?")

    contract = table_output_contract(query)

    assert QUERY_REQUIREMENTS_VERSION == "gold-free-table-output-contract-v3"
    assert contract == {
        "derived_from": ["question", "table_schema"],
        "row_key_policy": {
            "paper_title": "metadata_title_exact",
            "other": "query_facing_shortest_explicit_label",
        },
        "non_row_key_string_policy": "source_exact",
        "schema_columns": [
            {
                "name": "Method",
                "type": "string",
                "is_row_key": True,
                "output_policy": "query_facing_shortest_explicit_label",
            },
            {
                "name": "Value",
                "type": "string",
                "is_row_key": False,
                "output_policy": "source_exact",
            },
        ],
        "explicit_row_inventory": ["Cedar", "Flint", "Larch"],
    }
    assert "query_id" not in contract


def test_table_output_contract_assigns_metadata_and_native_type_policies() -> None:
    query = Query(
        query_id="synthetic",
        question="Which papers satisfy the stated condition?",
        answer_types=["table"],
        table_schema=[
            {"name": "Paper Title", "type": "string", "is_row_key": True},
            {"name": "Description", "type": "string", "is_row_key": False},
            {"name": "Score", "type": "number", "is_row_key": False},
            {"name": "Passed", "type": "boolean", "is_row_key": False},
        ],
    )

    contract = table_output_contract(query)

    assert contract is not None
    assert [
        column["output_policy"] for column in contract["schema_columns"]
    ] == [
        "metadata_title_exact",
        "source_exact",
        "native_json_number",
        "native_json_boolean",
    ]
    assert contract["explicit_row_inventory"] == []


def test_non_table_question_has_no_table_output_contract() -> None:
    assert (
        table_output_contract(_query("What is the value?", table=False)) is None
    )


def test_missing_items_accepts_safe_surface_aliases() -> None:
    query = _query(
        "What are the accuracies of NCFM, AP-BPTT, ATT, and DEDA given IPC=10?"
    )
    rows = [
        {"Method": "NCFM (Ours)", "Value": "27.4±0.6"},
        {"Method": "AT-BPTT", "Value": "32.7±0.5"},
        {"Method": "ATT [22]", "Value": "25.8±0.4"},
        {"Method": "RN-18 DEDA", "Value": "44.5±0.6"},
    ]
    assert missing_explicit_table_items(query, rows) == ()


def test_missing_items_detects_dropped_rows() -> None:
    query = _query(
        "What are the scores for TCM, ECM-XL (with 102.4M training budget), "
        "iCT-deep, and SiD as reported in their respective papers?"
    )
    rows = [
        {"Method": "TCM", "Value": "2.46"},
        {"Method": "SiD", "Value": "1.923"},
    ]
    assert missing_explicit_table_items(query, rows) == (
        "ECM-XL (with 102.4M training budget)",
        "iCT-deep",
    )


def test_condition_rows_match_descriptive_source_labels() -> None:
    query = _query(
        "What is the performance with ground-truth prompts and Cube R-CNN 2D detections?"
    )
    rows = [
        {"Method": "DetAny3D (ours) w/ Ground Truth", "Value": "38.68"},
        {"Method": "DetAny3D (ours) w/ Cube RCNN", "Value": "31.61"},
    ]
    assert missing_explicit_table_items(query, rows) == ()


@pytest.mark.parametrize(
    "actual_label",
    [
        "DetAny3D (ours) w/ Cube RCNN 3D detections",
        "DetAny3D (ours) w/ Cube RCNN 3D",
    ],
)
def test_detection_dimension_mismatch_does_not_satisfy_requested_row(
    actual_label: str,
) -> None:
    query = _query(
        "What is the performance with ground-truth prompts and Cube R-CNN 2D detections?"
    )
    rows = [
        {"Method": "DetAny3D (ours) w/ Ground Truth", "Value": "38.68"},
        {
            "Method": actual_label,
            "Value": "31.61",
        },
    ]

    assert missing_explicit_table_items(query, rows) == (
        "Cube R-CNN 2D detections",
    )


def test_explicitly_missing_item_satisfies_completeness_contract() -> None:
    query = _query("What are the values for A, B, C, and D?")
    rows = [
        {"Method": "A", "Value": "1"},
        {"Method": "B", "Value": "2"},
        {"Method": "D", "Value": "4"},
    ]
    assert unaccounted_explicit_table_items(query, rows, ["C: not reported"]) == ()
    assert unaccounted_explicit_table_items(query, rows, []) == ("C",)
