"""複数の Indexer と Fuser（任意で Reranker）を束ねる Retriever 本体。

各 Indexer で検索した結果を Fuser で1つのランキングに統合し、
必要に応じて Reranker で再ランクして返す。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.base import Fuser, Reranker


class HybridRetriever:
    def __init__(
        self,
        indexers: list,
        fuser: Fuser,
        reranker: Reranker | None = None,
        per_index_k: int = 100,
    ):
        self.indexers = indexers
        self.fuser = fuser
        self.reranker = reranker
        self.per_index_k = per_index_k

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.indexers:
            return []

        runs = [indexer.search(query, self.per_index_k) for indexer in self.indexers]
        if self.reranker is not None:
            fuse_k = top_k * 3
        else:
            fuse_k = top_k
        fused = self.fuser.fuse(runs, top_k=fuse_k)

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
