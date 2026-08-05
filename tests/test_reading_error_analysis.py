from __future__ import annotations

import json

from littraceqa.reading_error_analysis import (
    analyze_reading_run,
    render_query_markdown,
    write_analysis_outputs,
)


def _candidate(query_id: str, *paper_ids: str) -> dict:
    return {
        "query_id": query_id,
        "candidate_papers": [
            {"rank": rank, "paper_id": paper_id}
            for rank, paper_id in enumerate(paper_ids, start=1)
        ],
    }


def _freeform_gold(query_id: str, paper_id: str = "p1") -> dict:
    return {
        "query_id": query_id,
        "question": "What is the reported value?",
        "answer_types": ["freeform"],
        "task_family": "hidden_source_single_paper",
        "primary_evidence_type": "table",
        "gold_papers": [{"paper_id": paper_id}],
        "evidence": [
            {
                "paper_id": paper_id,
                "source_type": "table",
                "locator": {"page": 3, "table_id": "Table 2"},
            }
        ],
        "answer": {"freeform": {"text": "42"}},
    }


def test_analyzer_separates_candidate_relevance_evidence_and_answer_errors():
    gold = [_freeform_gold("q_001"), _freeform_gold("q_002", "p_missing")]
    candidates = [_candidate("q_001", "p1", "noise"), _candidate("q_002", "noise")]
    traces = [
        {
            "query_id": "q_001",
            "candidate_papers": ["p1", "noise"],
            "relevance_judgments": [
                {
                    "paper_id": "p1",
                    "rank": 1,
                    "relevant": True,
                    "answerable": True,
                    "evidence_chunk_ids": ["p1:c1"],
                },
                {
                    "paper_id": "noise",
                    "rank": 2,
                    "relevant": True,
                    "answerable": False,
                    "evidence_chunk_ids": [],
                },
            ],
            "submission": {
                "gold_papers": [{"paper_id": "p1"}],
                "evidence": [
                    {
                        "paper_id": "p1",
                        "source_type": "table",
                        "locator": {"page": 4, "table_id": "Table 9"},
                    }
                ],
                "answer": {"freeform": {"text": "41"}},
            },
        },
        {
            "query_id": "q_002",
            "candidate_papers": ["noise"],
            "relevance_judgments": [
                {"paper_id": "noise", "rank": 1, "relevant": False}
            ],
            "predicted_answer": "unknown",
        },
    ]

    analysis = analyze_reading_run(gold, candidates, traces)
    q1, q2 = analysis["queries"]

    assert q1["relevance_analysis"]["false_positive_paper_ids"] == ["noise"]
    assert "relevance_filter_overselection" in q1["error_categories"]
    assert "evidence_chunk_selection_error" in q1["error_categories"]
    assert "modality_read_error" in q1["error_categories"]
    assert q1["metrics"]["answer_exact"] is False
    assert q2["candidate_analysis"]["missing_required_gold_paper_ids"] == [
        "p_missing"
    ]
    assert q2["error_categories"][0] == "candidate_missing"
    assert analysis["summary"]["category_counts"]["candidate_missing"] == 1


def test_semantic_mc_answer_exposes_protocol_blocker_without_oracle_inference():
    gold = [
        {
            "query_id": "q_mc",
            "question": "Is it larger?",
            "answer_types": ["multiple_choice"],
            "gold_papers": [{"paper_id": "p1"}],
            "evidence": [
                {
                    "paper_id": "p1",
                    "source_type": "text_span",
                    "locator": {"page": 1},
                }
            ],
            "answer": {
                "multiple_choice": {
                    "options": {"A": "Yes", "B": "No"},
                    "gold": "A",
                }
            },
        }
    ]
    candidates = [_candidate("q_mc", "p1")]
    traces = [
        {
            "query_id": "q_mc",
            "relevance_judgments": [
                {"paper_id": "p1", "relevant": True, "answerable": True}
            ],
            "semantic_multiple_choice": {"text": "Yes"},
            "submission": {
                "gold_papers": [{"paper_id": "p1"}],
                "evidence": [
                    {
                        "paper_id": "p1",
                        "source_type": "text_span",
                        "locator": {"page": 1},
                    }
                ],
                "answer": {"multiple_choice": {"gold": "B"}},
            },
        }
    ]

    detail = analyze_reading_run(gold, candidates, traces)["queries"][0]

    assert detail["metrics"]["semantic_multiple_choice_exact"] is True
    assert detail["metrics"]["multiple_choice_exact"] is False
    assert detail["metrics"]["multiple_choice_protocol_blocked"] is True
    assert detail["metrics"]["reading_answer_exact"] is True
    assert "multiple_choice_protocol_blocker" in detail["error_categories"]
    assert "answer_extraction_or_reasoning_error" not in detail["error_categories"]


def test_multi_paper_answer_failure_is_classified_after_correct_evidence():
    evidence = [
        {
            "paper_id": paper_id,
            "source_type": "table",
            "locator": {"page": index, "table_id": "Table 1"},
        }
        for index, paper_id in enumerate(("p1", "p2"), start=1)
    ]
    gold = [
        {
            "query_id": "q_multi",
            "question": "Combine the values.",
            "answer_types": ["freeform"],
            "task_family": "multi_paper",
            "primary_evidence_type": "table",
            "gold_papers": [{"paper_id": "p1"}, {"paper_id": "p2"}],
            "evidence": evidence,
            "answer": {"freeform": {"text": "combined"}},
        }
    ]
    trace = {
        "query_id": "q_multi",
        "relevance_judgments": [
            {"paper_id": paper_id, "relevant": True, "answerable": True}
            for paper_id in ("p1", "p2")
        ],
        "submission": {
            "gold_papers": [{"paper_id": "p1"}, {"paper_id": "p2"}],
            "evidence": evidence,
            "answer": {"freeform": {"text": "wrong"}},
        },
    }

    detail = analyze_reading_run(
        gold, [_candidate("q_multi", "p1", "p2")], [trace]
    )["queries"][0]

    assert detail["metrics"]["evidence_recall"] == 1.0
    assert "multi_paper_integration_error" in detail["error_categories"]
    assert "answer_extraction_or_reasoning_error" in detail["error_categories"]


def test_known_dataset_issues_and_per_query_outputs(tmp_path):
    q054 = _freeform_gold("q_054")
    q054["question"] = "What is $AP^{kit}_{3D}$?"
    q054["answer"] = {
        "table": {
            "schema": [
                {"name": "Method", "type": "string", "is_row_key": True},
                {"name": "$AP^{nus}_{3D}$", "type": "string"},
            ],
            "rows": [{"Method": "X", "$AP^{nus}_{3D}$": "1"}],
        }
    }
    q054["answer_types"] = ["table"]
    analysis = analyze_reading_run(
        [q054],
        [_candidate("q_054", "p1")],
        [
            {
                "query_id": "q_054",
                "relevance_judgments": [{"paper_id": "p1", "relevant": True}],
                "submission": {
                    "gold_papers": [{"paper_id": "p1"}],
                    "evidence": q054["evidence"],
                    "answer": q054["answer"],
                },
            }
        ],
    )

    detail = analysis["queries"][0]
    assert "dataset_inconsistency" in detail["error_categories"]
    assert any("AP^kit" in issue for issue in detail["dataset_issues"])
    assert "Dataset issues" in render_query_markdown(detail)

    write_analysis_outputs(analysis, tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    per_query = json.loads((tmp_path / "queries" / "q_054.json").read_text())
    assert per_query["query_id"] == "q_054"
    assert "q_054" in (tmp_path / "queries" / "q_054.md").read_text()


def test_manual_mc_audit_does_not_treat_distractor_evidence_as_required():
    target = "iclr2025_03031"
    distractor = "icml2025_01371"
    evidence = [
        {
            "paper_id": paper_id,
            "source_type": "text_span",
            "locator": {"page": 1},
        }
        for paper_id in (target, distractor)
    ]
    gold = {
        "query_id": "q_035",
        "question": "Which statement about the target method is correct?",
        "answer_types": ["multiple_choice"],
        "gold_papers": [{"paper_id": target}, {"paper_id": distractor}],
        "evidence": evidence,
        "answer": {
            "multiple_choice": {
                "options": {"A": "correct meaning", "B": "distractor"},
                "gold": "A",
            }
        },
    }
    trace = {
        "query_id": "q_035",
        "relevance_judgments": [
            {"paper_id": target, "relevant": True, "answerable": True}
        ],
        "semantic_multiple_choice": {"text": "correct meaning"},
        "submission": {
            "gold_papers": [{"paper_id": target}],
            "evidence": [evidence[0]],
            "answer": {"multiple_choice": {"gold": "B"}},
        },
    }

    detail = analyze_reading_run(
        [gold], [_candidate("q_035", target)], [trace]
    )["queries"][0]

    assert detail["candidate_analysis"]["required_gold_paper_ids"] == [target]
    assert detail["candidate_analysis"]["required_papers_source"] == "manual_validation_audit"
    assert "candidate_missing" not in detail["error_categories"]
    assert detail["metrics"]["evidence_recall"] == 1.0
    assert detail["metrics"]["official_evidence_recall"] == 0.5
