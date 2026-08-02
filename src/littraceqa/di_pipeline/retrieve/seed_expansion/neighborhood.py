"""Paper-neighbourhood reranking.

Thin seed-expansion adapter over :class:`PaperNeighborhoodReranker`, which
scores candidates by the citation links between the papers already retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.paper_neighborhood import (
    PaperNeighborhoodReranker,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)


@dataclass(frozen=True)
class PaperNeighborhoodExpansion:
    """Uses explicit cross-paper mentions to rerank a bounded candidate pool."""

    rrf_k: int
    candidate_k: int
    relation_weight: float
    two_hop_weight: float
    max_hub_degree: int

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        indexers,
    ) -> list[RetrievalResult]:
        """Rerank the head of the pool and leave the tail untouched."""

        if (
            self.relation_weight <= 0 and self.two_hop_weight <= 0
        ) or len(candidates) < 2:
            return candidates

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return candidates

        pool_size = min(self.candidate_k, len(candidates))
        selector = PaperNeighborhoodReranker(
            get_document=paper_index.get_document,
            rrf_k=self.rrf_k,
            relation_weight=self.relation_weight,
            two_hop_weight=self.two_hop_weight,
            max_hub_degree=self.max_hub_degree,
        )
        reranked = selector.rerank(query, candidates[:pool_size], pool_size)
        return [*reranked, *candidates[pool_size:]]
