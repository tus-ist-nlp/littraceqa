from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import (
    Answer,
    Evidence,
    EvidenceLocator,
    Prediction,
    Query,
)
from littraceqa.submission import deterministic_mc_letter, prediction_to_submission


def test_mc_fallback_is_deterministic_and_valid():
    assert deterministic_mc_letter("q_001") == deterministic_mc_letter("q_001")
    assert deterministic_mc_letter("q_001") in "ABCD"


def test_serializer_drops_analysis_and_unused_answer_types():
    query = Query("q1", "question", ["freeform"], None)
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence("p1", "text_span", EvidenceLocator(page=1), "secret excerpt")
        ],
        answer=Answer(
            freeform={"text": "answer"},
            multiple_choice={"gold": "B"},
            table={"rows": [{"x": 1}]},
        ),
        trace=[{"secret": "trace"}],
        candidate_papers=["p1", "p2"],
    )

    serialized = prediction_to_submission(query, prediction)

    assert set(serialized) == {"query_id", "gold_papers", "evidence", "answer"}
    assert serialized["answer"] == {"freeform": {"text": "answer"}}
    assert "evidence_text_or_value" not in serialized["evidence"][0]


def test_numeric_table_cells_keep_information():
    query = Query(
        "q1",
        "question",
        ["table"],
        [{"name": "Score", "type": "number", "is_row_key": True}],
    )
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(table={"rows": [{"Score": "91.2%"}]}),
    )

    rows = prediction_to_submission(query, prediction)["answer"]["table"]["rows"]

    assert rows == [{"Score": 91.2}]


def test_invalid_numeric_table_cell_fails_closed():
    query = Query(
        "q1",
        "question",
        ["table"],
        [{"name": "Score", "type": "number", "is_row_key": True}],
    )
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(table={"rows": [{"Score": "44.5±0.6"}]}),
    )

    with pytest.raises(ValueError, match="invalid numeric"):
        prediction_to_submission(query, prediction)


def test_row_keys_control_table_deduplication():
    query = Query(
        "q1",
        "question",
        ["table"],
        [
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
        ],
    )
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(
            table={
                "rows": [
                    {"Method": "X", "Score": 1},
                    {"Method": "X", "Score": 2},
                ]
            }
        ),
    )

    rows = prediction_to_submission(query, prediction)["answer"]["table"]["rows"]

    assert rows == [{"Method": "X", "Score": 1}]


def test_invalid_mc_letter_is_replaced_with_a_to_d_fallback():
    query = Query("q1", "question", ["multiple_choice"], None)
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(multiple_choice={"gold": "Z"}),
    )

    letter = prediction_to_submission(query, prediction)["answer"][
        "multiple_choice"
    ]["gold"]

    assert letter in "ABCD"


def test_evidence_locator_is_coarse_and_page_is_validated():
    query = Query("q1", "question", ["freeform"], None)
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence(
                "p1",
                "table",
                EvidenceLocator(page="6", table_id="Table 4", row="r", column="c"),
            )
        ],
        answer=Answer(freeform={"text": "answer"}),
    )

    locator = prediction_to_submission(query, prediction)["evidence"][0]["locator"]

    assert locator == {"page": 6, "table_id": "Table 4"}
