from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import (
    Answer,
    Evidence,
    EvidenceLocator,
    Prediction,
    Query,
)
from littraceqa.submission import (
    deterministic_mc_letter,
    normalize_visible_id,
    prediction_to_submission,
)


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


def test_numeric_table_cells_must_already_use_native_json_numbers():
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

    with pytest.raises(TypeError, match="must be a JSON number"):
        prediction_to_submission(query, prediction)


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

    with pytest.raises(TypeError, match="must be a JSON number"):
        prediction_to_submission(query, prediction)


def test_duplicate_row_keys_fail_closed_after_official_normalization():
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

    with pytest.raises(ValueError, match="duplicate table row key"):
        prediction_to_submission(query, prediction)


def test_serializer_uses_first_schema_column_as_implicit_row_key():
    query = Query(
        "q1",
        "question",
        ["table"],
        [
            {"name": "Method", "type": "string", "is_row_key": False},
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
                    {"Method": " x ", "Score": 2},
                ]
            }
        ),
    )

    with pytest.raises(ValueError, match="duplicate table row key"):
        prediction_to_submission(query, prediction)


@pytest.mark.parametrize(
    "row",
    [
        {"Method": "X"},
        {"Method": "X", "Score": 1, "Unexpected": "value"},
    ],
)
def test_table_rows_require_exact_schema_columns(row):
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
        answer=Answer(table={"rows": [row]}),
    )

    with pytest.raises(ValueError, match="contain exactly"):
        prediction_to_submission(query, prediction)


def test_invalid_mc_letter_uses_legacy_fallback_only_without_options():
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


def test_serializer_accepts_variable_official_mc_labels_including_e():
    query = Query(
        "q1",
        "question",
        ["multiple_choice"],
        options={"A": "First", "B": "Second", "E": "Fifth"},
    )
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(multiple_choice={"gold": "E"}),
    )

    answer = prediction_to_submission(query, prediction)["answer"]

    assert answer["multiple_choice"] == {"gold": "E"}


def test_serializer_rejects_label_not_released_for_query():
    query = Query(
        "q1",
        "question",
        ["multiple_choice"],
        options={"A": "First", "E": "Fifth"},
    )
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", "text_span", EvidenceLocator(page=1))],
        answer=Answer(multiple_choice={"gold": "B"}),
    )

    with pytest.raises(ValueError, match="is not one of"):
        prediction_to_submission(query, prediction)


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


def test_algorithm_locator_preserves_official_algorithm_id():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence(
                "p1",
                "equation_algorithm",
                EvidenceLocator(page=4, algorithm_id="Algorithm 2"),
            )
        ],
        answer=Answer(freeform={"text": "answer"}),
    )

    locator = prediction_to_submission(query, prediction)["evidence"][0]["locator"]

    assert locator == {"page": 4, "algorithm_id": "Algorithm 2"}


@pytest.mark.parametrize(
    "source_type,raw_locator,expected",
    [
        (
            "text_span",
            EvidenceLocator(section="3 Results"),
            {"section": "3 Results"},
        ),
        (
            "table",
            EvidenceLocator(section="Results", table_id="Table 2"),
            {"section": "Results", "table_id": "Table 2"},
        ),
        (
            "figure",
            EvidenceLocator(section="Results", figure_id="Figure 4"),
            {"section": "Results", "figure_id": "Figure 4"},
        ),
        (
            "citation_context",
            EvidenceLocator(citation_id="24"),
            {"citation_id": "24"},
        ),
        (
            "equation_algorithm",
            EvidenceLocator(page=5, equation_id="Equation 6"),
            {"page": 5, "equation_id": "Equation 6"},
        ),
        (
            "equation_algorithm",
            EvidenceLocator(algorithm_id="Algorithm 2"),
            {"algorithm_id": "Algorithm 2"},
        ),
        (
            "citation_context",
            EvidenceLocator(section="References"),
            {"section": "References"},
        ),
    ],
)
def test_serializer_preserves_current_official_locator_keys(
    source_type, raw_locator, expected
):
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[Evidence("p1", source_type, raw_locator)],
        answer=Answer(freeform={"text": "answer"}),
    )

    locator = prediction_to_submission(query, prediction)["evidence"][0]["locator"]

    assert locator == expected


def test_test_extra_serializer_can_omit_unscored_evidence():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[],
        answer=Answer(freeform={"text": "answer"}),
    )

    serialized = prediction_to_submission(
        query,
        prediction,
        require_evidence=False,
    )

    assert set(serialized) == {"query_id", "gold_papers", "answer"}


def test_scored_test_serializer_still_requires_evidence():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[],
        answer=Answer(freeform={"text": "answer"}),
    )

    with pytest.raises(ValueError, match="no valid evidence"):
        prediction_to_submission(query, prediction)


@pytest.mark.parametrize(
    "value,prefix,expected",
    [
        ("Table2", "table", "table 2"),
        ("2", "table", "table 2"),
        ('"TABLE   2A"', "table", "table 2a"),
        ("Figure 4", "figure", "figure 4"),
        ("24", "citation", "citation 24"),
        ("Algorithm 2", "equation", "algorithm 2"),
    ],
)
def test_visible_id_normalization_matches_official_evaluator(
    value, prefix, expected
):
    assert normalize_visible_id(value, prefix) == expected


def test_table_evidence_dedupes_by_official_visible_id():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence(
                "p1", "table", EvidenceLocator(page=3, table_id="Table 2")
            ),
            Evidence("p1", "table", EvidenceLocator(page=3, table_id="2")),
            Evidence(
                "p1", "table", EvidenceLocator(page=3, table_id="TABLE2")
            ),
        ],
        answer=Answer(freeform={"text": "answer"}),
    )

    evidence = prediction_to_submission(query, prediction)["evidence"]

    assert evidence == [
        {
            "paper_id": "p1",
            "source_type": "table",
            "locator": {"page": 3, "table_id": "Table 2"},
        }
    ]


def test_equation_and_algorithm_dedupe_uses_evaluator_precedence_and_prefix():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence(
                "p1",
                "equation_algorithm",
                EvidenceLocator(page=5, equation_id="Equation 2"),
            ),
            # The official evaluator normalizes a bare algorithm_id with the
            # "equation" prefix, so this is the same key as Equation 2.
            Evidence(
                "p1",
                "equation_algorithm",
                EvidenceLocator(page=5, algorithm_id="2"),
            ),
            # A visibly prefixed Algorithm 2 remains "algorithm 2" and is a
            # distinct official key.
            Evidence(
                "p1",
                "equation_algorithm",
                EvidenceLocator(page=5, algorithm_id="Algorithm 2"),
            ),
        ],
        answer=Answer(freeform={"text": "answer"}),
    )

    evidence = prediction_to_submission(query, prediction)["evidence"]

    assert [item["locator"] for item in evidence] == [
        {"page": 5, "equation_id": "Equation 2"},
        {"page": 5, "algorithm_id": "Algorithm 2"},
    ]


def test_citation_and_section_fallback_dedupe_like_official_evaluator():
    query = Query("q1", "question", ["freeform"])
    prediction = Prediction(
        query_id="q1",
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            Evidence(
                "p1",
                "citation_context",
                EvidenceLocator(section=" References ", citation_id="24"),
            ),
            Evidence(
                "p1",
                "citation_context",
                EvidenceLocator(section="References", citation_id="Citation24"),
            ),
        ],
        answer=Answer(freeform={"text": "answer"}),
    )

    evidence = prediction_to_submission(query, prediction)["evidence"]

    assert evidence == [
        {
            "paper_id": "p1",
            "source_type": "citation_context",
            "locator": {"section": "References", "citation_id": "24"},
        }
    ]
