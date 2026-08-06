"""Paper-level fusion for a base retrieval lane and an expanded lane."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.retrieve.base import Reranker
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)


_RERANK_METADATA_PREFIXES = (
    "pre_rerank_",
    "qwen3_",
    "rank_fusion_",
)


def _unique_papers(
    candidates: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Keep the first result for each paper without changing lane order."""

    unique: list[RetrievalResult] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.paper_id in seen:
            continue
        seen.add(candidate.paper_id)
        unique.append(candidate)
    return unique


@dataclass(frozen=True)
class PaperTwoLaneReranker:
    """Rerank two paper lanes and combine their ranks with weighted RRF."""

    reranker: Reranker
    document_chars: int
    rrf_k: float
    base_weight: float
    expansion_weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.document_chars, bool)
            or not isinstance(self.document_chars, int)
            or self.document_chars <= 0
        ):
            raise ValueError("document_chars must be a positive integer")
        self._validate_number("rrf_k", self.rrf_k, allow_zero=True)
        self._validate_number("base_weight", self.base_weight, allow_zero=True)
        self._validate_number(
            "expansion_weight",
            self.expansion_weight,
            allow_zero=True,
        )
        if self.base_weight == 0 and self.expansion_weight == 0:
            raise ValueError("at least one lane weight must be positive")

        rerank = getattr(self.reranker, "rerank", None)
        shared_api = self._has_shared_score_api()
        if not callable(rerank) and not shared_api:
            raise TypeError("reranker must provide a supported reranking API")

    @staticmethod
    def _validate_number(name: str, value: object, *, allow_zero: bool) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a number")
        number = float(value)
        lower_bound_ok = number >= 0 if allow_zero else number > 0
        if not math.isfinite(number) or not lower_bound_ok:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {qualifier} number")

    def fuse(
        self,
        query: str,
        base: list[RetrievalResult],
        expansion: list[RetrievalResult],
        indexers,
        max_candidates: int,
    ) -> list[RetrievalResult]:
        """Rerank a bounded lane union once when possible, then fuse ranks."""

        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")

        base_unique = _unique_papers(base)
        expansion_unique = _unique_papers(expansion)
        if not base_unique and not expansion_unique:
            return []

        unscored = self._weighted_fuse(
            base_unique,
            expansion_unique,
            max_candidates=max_candidates,
        )
        allowed_ids = {candidate.paper_id for candidate in unscored}
        base_pool = [
            candidate
            for candidate in base_unique
            if candidate.paper_id in allowed_ids
        ]
        expansion_pool = [
            candidate
            for candidate in expansion_unique
            if candidate.paper_id in allowed_ids
        ]

        try:
            paper_index = find_paper_index(indexers, "get_document")
            if paper_index is None:
                raise RuntimeError(
                    "two-lane reranking requires paper_bm25 documents"
                )
            proxies = self._build_proxies(paper_index, unscored)
            proxy_by_id = {proxy.paper_id: proxy for proxy in proxies}
            if self._has_shared_score_api():
                reranked_base, reranked_expansion = self._rerank_with_shared_scores(
                    query,
                    base_pool,
                    expansion_pool,
                    proxies,
                    proxy_by_id,
                )
                status = "applied_shared"
            else:
                reranked_base = self._rerank_lane(
                    query,
                    base_pool,
                    proxy_by_id,
                )
                reranked_expansion = self._rerank_lane(
                    query,
                    expansion_pool,
                    proxy_by_id,
                )
                status = "applied_independent"
            fused = self._weighted_fuse(
                reranked_base,
                reranked_expansion,
                max_candidates=max_candidates,
            )
        except Exception as exc:
            return self._mark_status(
                unscored,
                status="fallback",
                error_type=type(exc).__name__,
            )

        return self._mark_status(fused, status=status)

    def _has_shared_score_api(self) -> bool:
        return callable(getattr(self.reranker, "score_candidates", None)) and callable(
            getattr(self.reranker, "rerank_scored", None)
        )

    def _build_proxies(
        self,
        paper_index,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
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
                    "paper_bm25 returned an invalid two-lane document for "
                    f"{candidate.paper_id}"
                )
            proxies.append(
                replace(candidate, text=document.text[: self.document_chars])
            )
        return proxies

    def _rerank_with_shared_scores(
        self,
        query: str,
        base: list[RetrievalResult],
        expansion: list[RetrievalResult],
        union_proxies: list[RetrievalResult],
        proxy_by_id: dict[str, RetrievalResult],
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        scores = list(self.reranker.score_candidates(query, union_proxies))
        if len(scores) != len(union_proxies):
            raise ValueError("reranker returned the wrong number of shared scores")

        score_by_id: dict[str, float] = {}
        for proxy, raw_score in zip(union_proxies, scores, strict=True):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("reranker returned a non-finite shared score")
            score_by_id[proxy.paper_id] = score

        return (
            self._rerank_scored_lane(base, proxy_by_id, score_by_id),
            self._rerank_scored_lane(expansion, proxy_by_id, score_by_id),
        )

    def _rerank_scored_lane(
        self,
        lane: list[RetrievalResult],
        proxy_by_id: dict[str, RetrievalResult],
        score_by_id: dict[str, float],
    ) -> list[RetrievalResult]:
        if not lane:
            return []
        proxies = [proxy_by_id[candidate.paper_id] for candidate in lane]
        scores = [score_by_id[candidate.paper_id] for candidate in lane]
        reranked = list(
            self.reranker.rerank_scored(proxies, scores, len(proxies))
        )
        return self._restore_lane(lane, reranked)

    def _rerank_lane(
        self,
        query: str,
        lane: list[RetrievalResult],
        proxy_by_id: dict[str, RetrievalResult],
    ) -> list[RetrievalResult]:
        if not lane:
            return []
        proxies = [proxy_by_id[candidate.paper_id] for candidate in lane]
        reranked = list(self.reranker.rerank(query, proxies, len(proxies)))
        return self._restore_lane(lane, reranked)

    @staticmethod
    def _restore_lane(
        originals: list[RetrievalResult],
        reranked: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        original_by_id = {candidate.paper_id: candidate for candidate in originals}
        reranked_ids = [
            result.paper_id
            for result in reranked
            if isinstance(result, RetrievalResult)
        ]
        expected_ids = [candidate.paper_id for candidate in originals]
        if (
            len(reranked_ids) != len(reranked)
            or len(reranked) != len(originals)
            or len(set(reranked_ids)) != len(reranked_ids)
            or set(reranked_ids) != set(expected_ids)
        ):
            raise ValueError("reranker changed a two-lane candidate set")

        restored: list[RetrievalResult] = []
        for result in reranked:
            if not math.isfinite(float(result.score)):
                raise ValueError("reranker returned a non-finite lane score")
            if not isinstance(result.metadata, dict):
                raise TypeError("reranker returned invalid metadata")
            original = original_by_id[result.paper_id]
            metadata = dict(original.metadata)
            for key, value in result.metadata.items():
                if key not in original.metadata or key.startswith(
                    _RERANK_METADATA_PREFIXES
                ):
                    metadata[key] = value
            restored.append(
                replace(
                    original,
                    score=float(result.score),
                    metadata=metadata,
                )
            )
        return restored

    def _weighted_fuse(
        self,
        base: list[RetrievalResult],
        expansion: list[RetrievalResult],
        *,
        max_candidates: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {
            "base": {},
            "expansion": {},
        }
        representatives: dict[str, RetrievalResult] = {}
        first_seen: dict[str, int] = {}

        for source, lane, weight in (
            ("base", base, float(self.base_weight)),
            ("expansion", expansion, float(self.expansion_weight)),
        ):
            for rank, candidate in enumerate(lane, start=1):
                paper_id = candidate.paper_id
                if paper_id in ranks[source]:
                    continue
                ranks[source][paper_id] = rank
                scores[paper_id] = scores.get(paper_id, 0.0) + weight / (
                    float(self.rrf_k) + rank
                )
                first_seen.setdefault(paper_id, len(first_seen))
                if source == "base" or paper_id not in representatives:
                    representatives[paper_id] = candidate

        ranked_ids = sorted(
            scores,
            key=lambda paper_id: (
                -scores[paper_id],
                first_seen[paper_id],
                paper_id,
            ),
        )[:max_candidates]

        fused: list[RetrievalResult] = []
        for paper_id in ranked_ids:
            representative = representatives[paper_id]
            metadata = dict(representative.metadata)
            base_rank = ranks["base"].get(paper_id)
            expansion_rank = ranks["expansion"].get(paper_id)
            metadata.update(
                {
                    "two_lane_base_rank": base_rank,
                    "two_lane_expansion_rank": expansion_rank,
                    "two_lane_rrf_score": scores[paper_id],
                    "two_lane_sources": [
                        source
                        for source, rank in (
                            ("base", base_rank),
                            ("expansion", expansion_rank),
                        )
                        if rank is not None
                    ],
                }
            )
            fused.append(
                replace(
                    representative,
                    score=scores[paper_id],
                    metadata=metadata,
                    source="paper_two_lane_rrf",
                )
            )
        return fused

    @staticmethod
    def _mark_status(
        candidates: list[RetrievalResult],
        *,
        status: str,
        error_type: str | None = None,
    ) -> list[RetrievalResult]:
        if not candidates:
            return candidates
        metadata = dict(candidates[0].metadata)
        metadata["two_lane_rerank_status"] = status
        if error_type is None:
            metadata.pop("two_lane_rerank_error_type", None)
        else:
            metadata["two_lane_rerank_error_type"] = error_type
        return [replace(candidates[0], metadata=metadata), *candidates[1:]]
