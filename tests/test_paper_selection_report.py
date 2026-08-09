from __future__ import annotations

import json

from littraceqa.di_pipeline.evaluation.paper_selection_report import (
    build_report,
    collect_review_cases,
    index_retrieval_entries,
)
from littraceqa.di_pipeline.evaluation.selection_input import load_paper_metadata
from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
)


def _record(query_id: str, gold_ids: list[str]) -> dict:
    return {
        "query_id": query_id,
        "question": "Which paper is required?",
        "answer_types": ["table"],
        "gold_papers": [{"paper_id": paper_id} for paper_id in gold_ids],
        "evidence": [
            {"evidence_id": "gold", "paper_id": gold_ids[-1], "value": "42"},
            {"evidence_id": "other", "paper_id": "not-gold", "value": "noise"},
        ],
        "answer": {"table": {"rows": []}},
    }


def test_report_classifies_cardinality_misses_and_filters_evidence():
    records = [_record("q1", ["gold-1", "gold-2"])]
    payload = {
        "queries": [
            {
                "query_id": "q1",
                "ranked_papers": ["gold-1", "wrong", "gold-2"],
                "pre_rerank_papers": ["gold-2", "gold-1", "wrong"],
                "ranking_details": [
                    {"paper_id": "gold-2", "qwen3_score": 1.25}
                ],
            }
        ]
    }
    cases, wanted = collect_review_cases(
        records,
        index_retrieval_entries(payload),
        CardinalityPaperSelector(default_count=1),
        top_candidates=2,
    )

    report = build_report(
        cases,
        {
            "gold-1": {"title": "First"},
            "gold-2": {"title": "Second"},
            "wrong": {"title": "Wrong"},
        },
        analysis_cutoff=20,
        top_candidates=2,
        sources={"gold": "fixture.jsonl"},
    )

    assert wanted == {"gold-1", "gold-2", "wrong"}
    assert report["summary"]["candidate_statuses"] == {
        "within_analysis_cutoff": 1
    }
    assert report["summary"]["selection_failures"] == {
        "selector_cardinality": 1
    }
    assert report["summary"]["cardinality_misses_by_selection_reason"] == {
        "default": 1
    }
    query = report["queries"][0]
    missed = next(p for p in query["gold_papers"] if p["paper_id"] == "gold-2")
    assert missed["final_rank"] == 3
    assert missed["pre_rerank_rank"] == 1
    assert missed["provenance"]["qwen3_score"] == 1.25
    assert [item["evidence_id"] for item in missed["evidence"]] == ["gold"]
    assert query["missed_gold_paper_ids"] == ["gold-2"]


def test_report_distinguishes_analysis_cutoff_and_saved_pool():
    records = [_record("q1", ["tail"]), _record("q2", ["outside"])]
    entries = index_retrieval_entries(
        {
            "queries": [
                {"query_id": "q1", "ranked_papers": ["wrong", "tail"]},
                {"query_id": "q2", "ranked_papers": ["wrong"]},
            ]
        }
    )
    cases, _ = collect_review_cases(
        records,
        entries,
        CardinalityPaperSelector(),
        top_candidates=1,
    )

    report = build_report(
        cases,
        {},
        analysis_cutoff=1,
        top_candidates=1,
        sources={},
    )

    assert report["summary"]["candidate_statuses"] == {
        "below_analysis_cutoff": 1,
        "outside_saved_pool": 1,
    }
    assert report["summary"]["selection_failures"] == {
        "candidate_generation": 1,
        "selector_ranking": 1,
    }


def test_review_cases_apply_an_optional_post_selection_refiner():
    class PickGold:
        def refine(self, query, candidates, selection):
            assert query.query_id == "q1"
            assert tuple(candidates) == ("wrong", "gold")
            return PaperSelection(("gold",), 1, "fixture_refiner")

    records = [_record("q1", ["gold"])]
    entries = index_retrieval_entries(
        {"queries": [{"query_id": "q1", "ranked_papers": ["wrong", "gold"]}]}
    )

    cases, wanted = collect_review_cases(
        records,
        entries,
        CardinalityPaperSelector(),
        top_candidates=2,
        selection_refiner=PickGold(),
    )

    assert cases == []
    assert wanted == set()


def test_report_separates_ranking_from_count_when_both_fail():
    records = [_record("q1", ["gold-1", "gold-2", "gold-3"])]
    entries = index_retrieval_entries(
        {
            "queries": [
                {
                    "query_id": "q1",
                    "ranked_papers": ["wrong", "gold-1", "gold-2", "gold-3"],
                }
            ]
        }
    )
    cases, _ = collect_review_cases(
        records,
        entries,
        CardinalityPaperSelector(default_count=2),
        top_candidates=4,
    )

    report = build_report(
        cases,
        {},
        analysis_cutoff=20,
        top_candidates=4,
        sources={},
    )

    assert report["summary"]["selection_failures"] == {
        "selector_cardinality": 1,
        "selector_ranking": 1,
    }
    failures = {
        paper["paper_id"]: paper.get("selection_failure")
        for paper in report["queries"][0]["gold_papers"]
    }
    assert failures["gold-2"] == "selector_ranking"
    assert failures["gold-3"] == "selector_cardinality"


def test_report_groups_cardinality_misses_by_selection_reason():
    open_set = _record("q1", ["a", "b", "c"])
    open_set["question"] = "Which CVPR 2025 papers cite UniAD?"
    stated = _record("q2", ["d", "e", "f"])
    stated["question"] = "Compare the two papers."
    default = _record("q3", ["g", "h"])
    records = [open_set, stated, default]
    entries = index_retrieval_entries(
        {
            "queries": [
                {"query_id": "q1", "ranked_papers": ["a", "b", "c"]},
                {"query_id": "q2", "ranked_papers": ["d", "e", "f"]},
                {"query_id": "q3", "ranked_papers": ["g", "h"]},
            ]
        }
    )
    cases, _ = collect_review_cases(
        records,
        entries,
        CardinalityPaperSelector(open_set_count=1),
        top_candidates=3,
    )

    report = build_report(
        cases,
        {},
        analysis_cutoff=20,
        top_candidates=3,
        sources={},
    )

    assert report["summary"]["cardinality_misses_by_selection_reason"] == {
        "default": 1,
        "open_set_enumeration": 2,
        "stated_in_question": 1,
    }


def test_report_includes_false_positive_only_queries():
    records = [_record("q1", ["gold"])]
    entries = index_retrieval_entries(
        {
            "queries": [
                {"query_id": "q1", "ranked_papers": ["gold", "wrong"]}
            ]
        }
    )
    cases, _ = collect_review_cases(
        records,
        entries,
        CardinalityPaperSelector(default_count=2),
        top_candidates=2,
    )

    report = build_report(
        cases,
        {},
        analysis_cutoff=20,
        top_candidates=2,
        sources={},
    )

    assert report["summary"] == {
        "queries_with_selection_errors": 1,
        "queries_with_missed_gold": 0,
        "queries_with_false_positives": 1,
        "missed_gold_papers": 0,
        "false_positive_papers": 1,
        "false_positive_failures": {
            "selector_precision_or_cardinality_over": 1
        },
        "candidate_statuses": {},
        "selection_failures": {},
        "cardinality_misses_by_selection_reason": {},
        "analysis_cutoff": 20,
        "top_candidates_per_query": 2,
    }
    assert report["queries"][0]["missed_gold_paper_ids"] == []
    assert [
        paper["paper_id"]
        for paper in report["queries"][0]["false_positive_papers"]
    ] == ["wrong"]
    assert report["queries"][0]["false_positive_papers"][0][
        "selection_failure"
    ] == "selector_precision_or_cardinality_over"


def test_metadata_loader_keeps_only_requested_papers(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text(
        json.dumps(
            {
                "paper_id": "wanted",
                "title": "A title",
                "authors": ["A. Author"],
                "venue": "ACL",
                "year": 2025,
                "abstract": "A long abstract with useful words.",
            }
        )
        + "\nthis trailing line is never read\n",
        encoding="utf-8",
    )

    papers = load_paper_metadata(path, {"wanted"}, abstract_chars=12)

    assert set(papers) == {"wanted"}
    assert papers["wanted"]["authors"] == ["A. Author"]
    assert papers["wanted"]["abstract"] == "A long abst..."


def test_metadata_loader_does_not_open_the_file_when_nothing_is_requested(
    tmp_path,
):
    assert load_paper_metadata(
        tmp_path / "missing.jsonl", set(), abstract_chars=800
    ) == {}
