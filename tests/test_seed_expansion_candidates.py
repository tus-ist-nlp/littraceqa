"""Candidate generation: paper-level RRF fusion, dedup, ties and cutoffs."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.candidates import (
    align_scores_with_output_order,
)
from seed_expansion_doubles import (
    _FakeReranker,
    _FakeRetriever,
    _result,
)


def test_fuses_unique_papers_with_equal_weight_rrf():
    initial = [_result("p1"), _result("p2"), _result("p3")]
    expanded = [_result("p2"), _result("p3"), _result("p4")]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p2", "p3", "p1", "p4"]
    assert results[0].chunk_id == "p2#c0000"
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 63 + 1 / 62)
    assert results[2].score == pytest.approx(1 / 61)
    assert results[3].score == pytest.approx(1 / 63)
    assert results[0].source == "seed_expansion_rrf"
    assert results[0].metadata["seed_expansion_original_rank"] == 2
    assert results[0].metadata["seed_expansion_expanded_rank"] == 1


def test_duplicate_chunks_do_not_consume_paper_ranks():
    initial = [
        _result("p1", chunk_id="p1#c0000"),
        _result("p1", chunk_id="p1#c0001"),
        _result("p2"),
    ]
    expanded = [_result("p2"), _result("p3")]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p2", "p1", "p3"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)


def test_ties_are_deterministic_and_prefer_first_seen_run():
    initial = [_result("original")]
    expanded = [_result("expanded")]

    first = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded])
    ).retrieve("question", 10)
    second = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded])
    ).retrieve("question", 10)

    assert [result.paper_id for result in first] == ["original", "expanded"]
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


def test_empty_initial_results_return_without_second_retrieval():
    inner = _FakeRetriever([[]])
    retriever = SeedExpansionRetriever(inner)

    assert retriever.retrieve("question", 10) == []
    assert len(inner.calls) == 1


def test_empty_expansion_falls_back_to_initial_ranking():
    initial = [_result("p1"), _result("p2")]
    inner = _FakeRetriever([initial, []])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("question", 10)

    assert results == initial
    assert len(inner.calls) == 2


def test_final_reranker_also_applies_when_expanded_search_is_empty():
    initial = [_result("p1"), _result("p2"), _result("p3")]
    inner = _FakeRetriever([initial, []])
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        inner,
        reranker=reranker,
        rerank_pool_k=2,
    )

    results = retriever.retrieve("question", 2)

    assert len(inner.calls) == 2
    assert len(reranker.calls) == 1
    assert [candidate.paper_id for candidate in reranker.calls[0][1]] == [
        "p1",
        "p2",
    ]
    assert [result.paper_id for result in results] == ["p2", "p1"]


def test_default_max_results_caps_output_at_ten():
    initial = [_result(f"p{i:02d}") for i in range(15)]
    expanded = [_result(f"p{i:02d}") for i in range(15)]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 50)

    assert len(results) == 10


def test_requested_top_k_can_reduce_max_results():
    initial = [_result(f"p{i}") for i in range(5)]
    expanded = [_result(f"p{i}") for i in range(5)]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    assert len(retriever.retrieve("question", 3)) == 3


def test_nonpositive_top_k_returns_without_retrieval():
    inner = _FakeRetriever([[]])
    retriever = SeedExpansionRetriever(inner)

    assert retriever.retrieve("question", 0) == []
    assert inner.calls == []


def test_final_output_order_is_preserved_by_score_based_paper_aggregation():
    results = [
        _result("first", score=0.2),
        _result("second", score=0.9),
    ]

    aligned = align_scores_with_output_order(results)

    assert [result.paper_id for result in aligned] == ["first", "second"]
    assert [result.score for result in aligned] == [0.9, 0.2]
    assert to_gold_papers(aligned) == ["first", "second"]
    assert aligned[0].metadata["pre_output_order_score"] == 0.2
    assert aligned[1].metadata["pre_output_order_score"] == 0.9
    assert [result.metadata["output_order_rank"] for result in aligned] == [1, 2]


def test_aligned_scores_leave_already_ranked_results_unchanged():
    results = [
        _result("first", score=0.9),
        _result("second", score=0.2),
    ]

    aligned = align_scores_with_output_order(results)

    assert aligned is results
    assert aligned[0] is results[0]
    assert aligned[1] is results[1]
