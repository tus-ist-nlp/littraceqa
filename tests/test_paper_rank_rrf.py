"""Tests for paper-rank RRF and chunk-derived output budgeting."""

from __future__ import annotations

import pytest

from litqa.contracts import RetrievalResult
from litqa.retrieve.paper_rank_rrf import PaperRankRRFFuser


def _result(chunk_id: str, paper_id: str, source: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=paper_id,
        score=1.0,
        text=chunk_id,
        chunk_type="paper" if source == "paper_bm25" else "text_span",
        metadata={},
        source=source,
    )


def test_fuser_uses_unique_chunk_papers_as_budget_and_fuses_paper_ranks():
    chunk_run = [
        _result("a1", "a", "bm25s"),
        _result("a2", "a", "bm25s"),
        _result("b1", "b", "bm25s"),
        _result("c1", "c", "bm25s"),
    ]
    paper_run = [
        _result("c-paper", "c", "paper_bm25"),
        _result("b-paper", "b", "paper_bm25"),
        _result("d-paper", "d", "paper_bm25"),
    ]
    fuser = PaperRankRRFFuser(k=60, budget_source="bm25s")

    results = fuser.fuse([chunk_run, paper_run], top_k=3)

    assert len(results) == 2
    assert [result.paper_id for result in results] == ["c", "b"]
    assert all(result.source == "paper_rank_rrf" for result in results)
    assert all(result.metadata["paper_rank_budget"] == 2 for result in results)


def test_fuser_can_fill_requested_budget_from_all_paper_rankings():
    chunk_run = [
        _result("a1", "a", "bm25s"),
        _result("a2", "a", "bm25s"),
        _result("b1", "b", "bm25s"),
        _result("c1", "c", "bm25s"),
    ]
    paper_run = [
        _result("c-paper", "c", "paper_bm25"),
        _result("b-paper", "b", "paper_bm25"),
        _result("d-paper", "d", "paper_bm25"),
    ]
    fuser = PaperRankRRFFuser(
        k=60,
        budget_source="bm25s",
        fill_to_top_k=True,
    )

    results = fuser.fuse([chunk_run, paper_run], top_k=4)

    assert [result.paper_id for result in results] == ["c", "b", "a", "d"]
    assert len({result.paper_id for result in results}) == 4
    assert results[-1].chunk_id == "d-paper"
    assert results[-1].metadata["paper_rank_representative_source"] == (
        "paper_bm25"
    )
    assert all(result.metadata["paper_rank_budget"] == 3 for result in results)


def test_fuser_fill_returns_all_available_distinct_papers_below_top_k():
    runs = [
        [_result("a1", "a", "bm25s")],
        [
            _result("b-paper", "b", "paper_bm25"),
            _result("a-paper", "a", "paper_bm25"),
        ],
    ]
    fuser = PaperRankRRFFuser(fill_to_top_k=True)

    results = fuser.fuse(runs, top_k=20)

    assert [result.paper_id for result in results] == ["a", "b"]


def test_fuser_fill_can_fall_back_when_budget_source_has_no_results():
    fuser = PaperRankRRFFuser(
        budget_source="bm25s",
        fill_to_top_k=True,
    )

    results = fuser.fuse(
        [[_result("paper", "p1", "paper_bm25")]],
        top_k=20,
    )

    assert [result.paper_id for result in results] == ["p1"]
    assert results[0].metadata["paper_rank_budget"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_fuser_rejects_non_boolean_fill_option(value):
    with pytest.raises(TypeError, match="fill_to_top_k"):
        PaperRankRRFFuser(fill_to_top_k=value)


def test_fuser_is_deterministic():
    runs = [
        [_result("a1", "a", "bm25s"), _result("b1", "b", "bm25s")],
        [_result("b-paper", "b", "paper_bm25"), _result("a-paper", "a", "paper_bm25")],
    ]
    fuser = PaperRankRRFFuser(k=0)

    first = fuser.fuse(runs, top_k=2)
    second = fuser.fuse(runs, top_k=2)

    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


def test_fuser_rejects_missing_budget_source():
    fuser = PaperRankRRFFuser(budget_source="bm25s")

    with pytest.raises(ValueError, match="budget_source"):
        fuser.fuse([[_result("p", "p", "paper_bm25")]], top_k=5)
