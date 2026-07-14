"""Fuse chunk and paper rankings after converting each run to paper ranks."""

from __future__ import annotations

import dataclasses

from litqa.contracts import RetrievalResult
from litqa.registry import register


@register("fuser", "paper_rank_rrf")
class PaperRankRRFFuser:
    """Apply RRF by paper ID with an optional paper-ranking backfill."""

    def __init__(
        self,
        k: int = 60,
        weights: dict[str, float] | None = None,
        budget_source: str = "bm25s",
        fill_to_top_k: bool = False,
    ):
        if k < 0:
            raise ValueError("k must be non-negative")
        if not budget_source:
            raise ValueError("budget_source must not be empty")
        if not isinstance(fill_to_top_k, bool):
            raise TypeError("fill_to_top_k must be a boolean")
        self.k = k
        self.weights = weights or {}
        self.budget_source = budget_source
        self.fill_to_top_k = fill_to_top_k

    @staticmethod
    def _source(run: list[RetrievalResult]) -> str | None:
        return run[0].source if run else None

    @staticmethod
    def _unique_papers(run: list[RetrievalResult]) -> list[RetrievalResult]:
        seen: set[str] = set()
        unique = []
        for result in run:
            if result.paper_id not in seen:
                seen.add(result.paper_id)
                unique.append(result)
        return unique

    def fuse(
        self,
        runs: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        budget_run = next(
            (run for run in runs if self._source(run) == self.budget_source),
            None,
        )
        if budget_run is None:
            if not self.fill_to_top_k:
                raise ValueError(
                    f"budget_source {self.budget_source!r} is missing from "
                    "retrieval runs"
                )
            paper_budget = 0
        else:
            paper_budget = len(
                {result.paper_id for result in budget_run[:top_k]}
            )
        if paper_budget == 0 and not self.fill_to_top_k:
            return []

        scores: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        representatives: dict[str, RetrievalResult] = {}
        supporting_sources: dict[str, list[str]] = {}
        seen_counter = 0

        for run in runs:
            source = self._source(run)
            if source is None:
                continue
            weight = self.weights.get(source, 1.0)
            for rank, result in enumerate(self._unique_papers(run), start=1):
                paper_id = result.paper_id
                scores[paper_id] = scores.get(paper_id, 0.0) + weight / (
                    self.k + rank
                )
                if paper_id not in first_seen:
                    first_seen[paper_id] = seen_counter
                    seen_counter += 1
                if source == self.budget_source or paper_id not in representatives:
                    representatives[paper_id] = result
                sources = supporting_sources.setdefault(paper_id, [])
                if source not in sources:
                    sources.append(source)

        paper_ids = sorted(
            scores,
            key=lambda paper_id: (
                -scores[paper_id],
                first_seen[paper_id],
                paper_id,
            ),
        )
        results = []
        output_budget = top_k if self.fill_to_top_k else min(top_k, paper_budget)
        for paper_id in paper_ids[:output_budget]:
            representative = representatives[paper_id]
            metadata = dict(representative.metadata)
            metadata.update(
                {
                    "paper_rank_fusion": "rrf",
                    "paper_rank_budget": paper_budget,
                    "paper_rank_score": scores[paper_id],
                    "paper_rank_sources": supporting_sources[paper_id],
                    "paper_rank_representative_source": representative.source,
                }
            )
            results.append(
                dataclasses.replace(
                    representative,
                    score=scores[paper_id],
                    metadata=metadata,
                    source="paper_rank_rrf",
                )
            )
        return results
