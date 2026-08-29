"""`rerank_blend`: mixing the reranker's order into the pre-rerank one with RRF.

Two things matter above the rest:

* **Omitting it behaves exactly as before** — the reranker replaces the order.
* **The blended rank is written into `score`.** Everything downstream re-sorts by
  score (the accumulation in agent.py, _candidate_papers, to_gold_papers),
  so a ranking carried only by list order is thrown away every single time.
"""

from __future__ import annotations

from littraceqa.search.contracts import RetrievalResult
from littraceqa.search.retrieve import (
    HybridRetriever,
    RerankBlend,
    RetrievalConfig,
)


def _result(name: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{name}#c0",
        paper_id=name,
        score=score,
        text=f"body {name}",
        chunk_type="text_span",
        metadata={},
    )


class _StubIndexer:
    """An index that returns a fixed ranking."""

    def __init__(self, order: list[str]):
        self.order = order

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [_result(name, 1.0 - i / 100) for i, name in enumerate(self.order)][:top_k]


class _ReverseReranker:
    """A reranker that reverses the fused order outright, so replace and blend differ visibly."""

    def rerank(self, query, candidates, top_k):
        # As the real one does, overwrite score with the yes probability first.
        ordered = list(reversed(candidates))
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                score=1.0 - i / 100,
                text=r.text,
                chunk_type=r.chunk_type,
                metadata=r.metadata,
            )
            for i, r in enumerate(ordered)
        ][:top_k]


def _retriever(**kwargs) -> HybridRetriever:
    return HybridRetriever(
        indexers=[_StubIndexer(["a", "b", "c", "d", "e"])],
        reranker=_ReverseReranker(),
        config=RetrievalConfig(per_index_k=10, pool_k=5, **kwargs),
    )


def test_without_blend_the_reranker_replaces_the_ranking():
    """The default (no rerank_blend) is the reranker's order, exactly as before."""
    got = _retriever().retrieve("q", top_k=5)
    assert [r.paper_id for r in got] == ["e", "d", "c", "b", "a"]


def test_blend_keeps_the_original_ranking_partly_alive():
    """Weighted towards the original order, the reranker cannot reverse it outright."""
    got = _retriever(
        rerank_blend=RerankBlend(original_weight=1.0, rerank_weight=0.0, rrf_k=60)
    ).retrieve("q", top_k=5)
    assert [r.paper_id for r in got] == ["a", "b", "c", "d", "e"]


def test_blend_score_matches_the_returned_order():
    """**The blended rank is in `score`**, so a downstream re-sort does not break it."""
    got = _retriever(
        rerank_blend=RerankBlend(original_weight=0.6, rerank_weight=0.4, rrf_k=60)
    ).retrieve("q", top_k=5)
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)
    # Re-sorting by score gives the same order; without this neither the blend nor
    # the protection survives
    assert [r.paper_id for r in sorted(got, key=lambda r: -r.score)] == [
        r.paper_id for r in got
    ]


def test_protect_top_keeps_the_original_head_set():
    """The pre-fusion top set cannot be pushed out of the head by the reranker."""
    got = _retriever(
        rerank_blend=RerankBlend(original_weight=0.0,  # even looking at the reranker alone...
            rerank_weight=1.0,
            rrf_k=60,
            protect_top=2)
    ).retrieve("q", top_k=5)
    # ...the pre-fusion top two (a, b) stay at the front, ordered by the blend
    assert set(r.paper_id for r in got[:2]) == {"a", "b"}
    # The protection is in score too, so a re-sort does not undo it
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)


def test_no_reranker_is_untouched():
    """With no reranker, writing rerank_blend does nothing."""
    retriever = HybridRetriever(
        indexers=[_StubIndexer(["a", "b", "c"])],
        reranker=None,
        config=RetrievalConfig(
            per_index_k=10,
            rerank_blend=RerankBlend(original_weight=0.6, rerank_weight=0.4),
        ),
    )
    assert [r.paper_id for r in retriever.retrieve("q", top_k=3)] == ["a", "b", "c"]
