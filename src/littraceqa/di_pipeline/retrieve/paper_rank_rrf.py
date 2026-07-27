"""Fuse chunk and paper rankings after converting each run to paper ranks."""

from __future__ import annotations

import dataclasses

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register


@register("fuser", "paper_rank_rrf")
class PaperRankRRFFuser:
    """Apply RRF by paper ID with an optional paper-ranking backfill."""

    def __init__(
        self,
        k: int = 60,
        weights: dict[str, float] | None = None,
        budget_source: str = "bm25s",
        fill_to_top_k: bool = False,
        expansion_source: str | None = None,
    ):
        if k < 0:
            raise ValueError("k must be non-negative")
        if not budget_source:
            raise ValueError("budget_source must not be empty")
        if not isinstance(fill_to_top_k, bool):
            raise TypeError("fill_to_top_k must be a boolean")
        if expansion_source is not None and (
            not isinstance(expansion_source, str) or not expansion_source.strip()
        ):
            raise ValueError("expansion_source must be a non-empty string or None")
        self.k = k
        self.weights = weights or {}
        self.budget_source = budget_source
        self.fill_to_top_k = fill_to_top_k
        self.expansion_source = expansion_source

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
            paper_budget = len({result.paper_id for result in budget_run[:top_k]})
        if paper_budget == 0 and not self.fill_to_top_k:
            return []

        scores: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        representatives: dict[str, RetrievalResult] = {}
        expansion_results: dict[str, RetrievalResult] = {}
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
                if source == self.expansion_source:
                    expansion_results.setdefault(paper_id, result)
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
            expansion_result = expansion_results.get(paper_id)
            if expansion_result is not None:
                metadata.update(
                    {
                        "paper_rank_expansion_text": expansion_result.text,
                        "paper_rank_expansion_source": expansion_result.source,
                    }
                )
                expansion_metadata = expansion_result.metadata
                if isinstance(expansion_metadata, dict):
                    method_names = expansion_metadata.get("method_names")
                    if isinstance(method_names, (list, tuple)) and all(
                        isinstance(name, str) for name in method_names
                    ):
                        metadata["method_names"] = list(method_names)
            results.append(
                dataclasses.replace(
                    representative,
                    score=scores[paper_id],
                    metadata=metadata,
                    source="paper_rank_rrf",
                )
            )
        return results
