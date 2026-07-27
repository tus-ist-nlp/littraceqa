"""複数の Indexer と Fuser（任意で Reranker）を束ねる Retriever 本体。

各 Indexer で検索した結果を Fuser で1つのランキングに統合し、
必要に応じて Reranker で再ランクして返す。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.attributes import rerank_by_attributes
from littraceqa.di_pipeline.retrieve.base import Fuser, Reranker


class HybridRetriever:
    def __init__(
        self,
        indexers: list,
        fuser: Fuser,
        reranker: Reranker | None = None,
        per_index_k: int = 100,
        pool_k: int | None = None,
        attribute_weight: float = 0.25,
    ):
        if pool_k is not None and pool_k <= 0:
            raise ValueError("pool_k must be positive when specified")
        self.indexers = indexers
        self.fuser = fuser
        self.reranker = reranker
        self.per_index_k = per_index_k
        self.pool_k = pool_k
        self.attribute_weight = attribute_weight

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        if not self.indexers:
            return []

        runs = [indexer.search(query, self.per_index_k) for indexer in self.indexers]
        use_attributes = hints is not None and not hints.is_empty
        if self.reranker is not None or use_attributes:
            # Keep the historical three-times expansion unless an experiment
            # explicitly fixes the candidate pool size (for example, 20 or 50).
            fuse_k = self.pool_k if self.pool_k is not None else top_k * 3
            # A reranker cannot return more results than it receives.
            fuse_k = max(top_k, fuse_k)
        else:
            fuse_k = top_k
        fused = self.fuser.fuse(runs, top_k=fuse_k)

        if use_attributes:
            fused = rerank_by_attributes(
                fused,
                hints,
                attribute_weight=self.attribute_weight,
            )

        if self.reranker is not None:
            return self.reranker.rerank(query, fused, top_k)
        return fused[:top_k]


def to_gold_papers(
    results: list[RetrievalResult],
    max_papers: int | None = None,
    agg: str = "max",
) -> list[str]:
    scores: dict[str, float] = {}
    for result in results:
        if agg == "max":
            scores[result.paper_id] = max(scores.get(result.paper_id, result.score), result.score)
        elif agg == "sum":
            scores[result.paper_id] = scores.get(result.paper_id, 0.0) + result.score
        else:
            raise ValueError(f"unknown agg: {agg!r}")

    papers = sorted(scores, key=lambda paper_id: scores[paper_id], reverse=True)
    if max_papers is not None:
        papers = papers[:max_papers]
    return papers
