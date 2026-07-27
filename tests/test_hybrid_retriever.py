"""Tests for HybridRetriever candidate-pool sizing."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever


def _results(count: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"paper-{index}#c0000",
            paper_id=f"paper-{index}",
            score=float(count - index),
            text=f"text {index}",
            chunk_type="text_span",
            metadata={},
            source="fake",
        )
        for index in range(count)
    ]


class FakeIndexer:
    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return _results(top_k)


class RecordingFuser:
    def __init__(self) -> None:
        self.requested: list[int] = []

    def fuse(self, runs, top_k: int) -> list[RetrievalResult]:
        self.requested.append(top_k)
        return runs[0][:top_k]


class RecordingReranker:
    def __init__(self) -> None:
        self.pool_sizes: list[int] = []

    def rerank(self, query, candidates, top_k):
        self.pool_sizes.append(len(candidates))
        return candidates[:top_k]


@pytest.mark.parametrize("pool_k", [20, 50])
def test_explicit_rerank_pool_size(pool_k: int):
    fuser = RecordingFuser()
    reranker = RecordingReranker()
    retriever = HybridRetriever(
        [FakeIndexer()],
        fuser,
        reranker=reranker,
        per_index_k=100,
        pool_k=pool_k,
    )

    results = retriever.retrieve("question", top_k=5)

    assert len(results) == 5
    assert fuser.requested == [pool_k]
    assert reranker.pool_sizes == [pool_k]


def test_default_rerank_pool_remains_three_times_top_k():
    fuser = RecordingFuser()
    reranker = RecordingReranker()
    retriever = HybridRetriever(
        [FakeIndexer()], fuser, reranker=reranker, per_index_k=100
    )

    retriever.retrieve("question", top_k=7)

    assert fuser.requested == [21]
    assert reranker.pool_sizes == [21]


def test_pool_is_not_smaller_than_requested_output():
    fuser = RecordingFuser()
    reranker = RecordingReranker()
    retriever = HybridRetriever(
        [FakeIndexer()], fuser, reranker=reranker, per_index_k=100, pool_k=3
    )

    retriever.retrieve("question", top_k=5)

    assert fuser.requested == [5]
    assert reranker.pool_sizes == [5]


def test_non_positive_pool_is_rejected():
    with pytest.raises(ValueError, match="pool_k"):
        HybridRetriever(
            [FakeIndexer()], RecordingFuser(), reranker=RecordingReranker(), pool_k=0
        )


def test_attribute_hints_softly_rerank_a_broader_candidate_pool():
    class AttributeIndexer:
        def search(self, query: str, top_k: int) -> list[RetrievalResult]:
            results = _results(3)
            results[0].metadata["venue"] = "EMNLP"
            results[1].metadata["venue"] = "ACL"
            return results

    fuser = RecordingFuser()
    retriever = HybridRetriever(
        [AttributeIndexer()],
        fuser,
        per_index_k=3,
        pool_k=3,
        attribute_weight=2.0,
    )

    results = retriever.retrieve(
        "question",
        top_k=1,
        hints=SearchHints(venues=("ACL",)),
    )

    assert fuser.requested == [3]
    assert [result.paper_id for result in results] == ["paper-1"]


def test_omitting_attribute_hints_preserves_existing_behavior():
    fuser = RecordingFuser()
    retriever = HybridRetriever([FakeIndexer()], fuser, per_index_k=10, pool_k=10)

    results = retriever.retrieve("question", top_k=2)

    assert fuser.requested == [2]
    assert [result.paper_id for result in results] == ["paper-0", "paper-1"]
