"""Final reranking.

Reorders a completed candidate set without changing its membership.  Every
validation failure falls back to the incoming order, so a misbehaving reranker
can never drop or invent a paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.retrieve.base import Reranker
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)


@dataclass(frozen=True)
class FinalCandidateReranker:
    """Reranks the fixed final paper set using full paper-level documents."""

    reranker: Reranker | None
    document_chars: int

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        indexers,
    ) -> list[RetrievalResult]:
        """Rerank a completed candidate set without changing its membership."""

        if not candidates:
            return candidates

        try:
            preserved = self._apply(query, candidates, indexers)
        except Exception as exc:
            metadata = dict(candidates[0].metadata)
            metadata.update(
                {
                    "final_rerank_status": "fallback",
                    "final_rerank_candidate_set_preserved": True,
                    "final_rerank_error_type": type(exc).__name__,
                }
            )
            return [
                replace(candidates[0], metadata=metadata),
                *candidates[1:],
            ]

        metadata = dict(preserved[0].metadata)
        metadata.update(
            {
                "final_rerank_status": "applied",
                "final_rerank_candidate_set_preserved": True,
                "pre_rerank_candidate_papers": [
                    candidate.paper_id for candidate in candidates
                ],
            }
        )
        metadata.pop("final_rerank_error_type", None)
        preserved[0] = replace(preserved[0], metadata=metadata)
        return preserved

    def _apply(
        self,
        query: str,
        candidates: list[RetrievalResult],
        indexers,
    ) -> list[RetrievalResult]:
        """Run the reranker and reject any result that alters the candidate set."""

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            raise RuntimeError(
                "final candidate reranking requires paper_bm25 documents"
            )
        if self.reranker is None:
            raise RuntimeError(
                "final candidate reranking requires an enabled reranker"
            )

        original_ids = [candidate.paper_id for candidate in candidates]
        if len(set(original_ids)) != len(original_ids):
            raise ValueError("final rerank input contains duplicate paper IDs")

        proxies = self._build_proxies(paper_index, candidates)
        reranked = list(self.reranker.rerank(query, proxies, len(proxies)))
        self._validate(reranked, candidates, original_ids)

        original_by_id = {
            candidate.paper_id: candidate for candidate in candidates
        }
        preserved: list[RetrievalResult] = []
        for result in reranked:
            original = original_by_id[result.paper_id]
            metadata = dict(original.metadata)
            if not isinstance(result.metadata, dict):
                raise TypeError("final reranker returned invalid metadata")
            for key, value in result.metadata.items():
                if key not in original.metadata or key.startswith(
                    ("pre_rerank_", "qwen3_", "rank_fusion_")
                ):
                    metadata[key] = value
            preserved.append(
                replace(
                    original,
                    score=float(result.score),
                    metadata=metadata,
                )
            )
        return preserved

    def _build_proxies(
        self,
        paper_index,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Swap each candidate's text for its full paper-level document."""

        proxies: list[RetrievalResult] = []
        for candidate in candidates:
            document = paper_index.get_document(candidate.paper_id)
            if (
                not isinstance(document, Chunk)
                or document.paper_id != candidate.paper_id
                or not isinstance(document.text, str)
                or not document.text.strip()
            ):
                raise ValueError(
                    "paper_bm25 returned an invalid final rerank document "
                    f"for {candidate.paper_id}"
                )
            proxies.append(
                replace(candidate, text=document.text[: self.document_chars])
            )
        return proxies

    @staticmethod
    def _validate(
        reranked: list,
        candidates: list[RetrievalResult],
        original_ids: list[str],
    ) -> None:
        """Reject rerankers that change the set, the count, or the ordering."""

        if len(reranked) != len(candidates):
            raise ValueError("final reranker changed the candidate count")
        if not all(isinstance(result, RetrievalResult) for result in reranked):
            raise TypeError("final reranker returned a non-RetrievalResult value")

        reranked_ids = [result.paper_id for result in reranked]
        if len(set(reranked_ids)) != len(reranked_ids) or set(
            reranked_ids
        ) != set(original_ids):
            raise ValueError("final reranker changed the candidate paper ID set")

        previous_score = math.inf
        for result in reranked:
            score = result.score
            if (
                isinstance(score, bool)
                or not isinstance(score, Real)
                or not math.isfinite(score)
                or float(score) > previous_score
            ):
                raise ValueError("final reranker returned invalid ranking scores")
            previous_score = float(score)
