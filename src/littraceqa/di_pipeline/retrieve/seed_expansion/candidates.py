"""Initial candidate generation.

Runs the wrapped retriever for each lane and fuses the runs into one
paper-level ranking, plus the helpers that keep the emitted scores consistent
with the order the papers are finally returned in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Retriever

MAX_OPEN_SET_SEEDS = 8


def unique_papers(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep the first result for each paper while preserving rank order."""

    unique: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in results:
        if result.paper_id in seen:
            continue
        seen.add(result.paper_id)
        unique.append(result)
    return unique


def align_scores_with_output_order(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Keep downstream score aggregation consistent with the final order."""

    current_scores = [float(result.score) for result in results]
    descending_scores = sorted(current_scores, reverse=True)
    if current_scores == descending_scores:
        return results
    aligned = []
    for index, result in enumerate(results):
        metadata = dict(result.metadata)
        metadata.setdefault("pre_output_order_score", result.score)
        metadata["output_order_rank"] = index + 1
        aligned.append(
            replace(
                result,
                score=descending_scores[index],
                metadata=metadata,
            )
        )
    return aligned


def append_result_tail(
    candidates: list[RetrievalResult],
    selected_prefix: list[RetrievalResult],
    output_k: int,
) -> list[RetrievalResult]:
    """Append a lower-scored tail without changing the selected prefix."""

    selected = align_scores_with_output_order(list(selected_prefix[:output_k]))
    prefix_length = len(selected)
    selected_ids = {candidate.paper_id for candidate in selected}
    for candidate in candidates:
        if len(selected) >= output_k:
            break
        if candidate.paper_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.paper_id)

    if prefix_length == 0:
        return align_scores_with_output_order(selected)

    previous_score = float(selected[prefix_length - 1].score)
    for index in range(prefix_length, len(selected)):
        result = selected[index]
        original_score = float(result.score)
        if original_score < previous_score:
            previous_score = original_score
            continue
        aligned_score = math.nextafter(previous_score, -math.inf)
        metadata = dict(result.metadata)
        metadata.setdefault("pre_output_order_score", result.score)
        metadata["output_order_rank"] = index + 1
        selected[index] = replace(
            result,
            score=aligned_score,
            metadata=metadata,
        )
        previous_score = aligned_score
    return selected


@dataclass(frozen=True)
class CandidateGeneration:
    """Bounded retrieval for each lane plus the paper-level RRF that fuses them."""

    retriever: Retriever
    candidate_k: int
    rrf_k: int

    def search(
        self,
        query: str,
        hints: SearchHints | None,
    ) -> list[RetrievalResult]:
        """Run one lane against the wrapped retriever, bounded by ``candidate_k``."""

        return self.retriever.retrieve(
            query,
            self.candidate_k,
            hints=hints,
        )[: self.candidate_k]

    def fuse_by_paper(
        self,
        initial: list[RetrievalResult],
        expanded: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Fuse the lanes with reciprocal rank fusion, one entry per paper."""

        scores: dict[str, float] = {}
        representatives: dict[str, RetrievalResult] = {}
        first_seen: dict[str, int] = {}
        ranks_by_run: list[dict[str, int]] = []

        for run in (initial, expanded):
            run_ranks: dict[str, int] = {}
            for result in run:
                if result.paper_id in run_ranks:
                    continue
                rank = len(run_ranks) + 1
                run_ranks[result.paper_id] = rank
                scores[result.paper_id] = scores.get(result.paper_id, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                if result.paper_id not in representatives:
                    representatives[result.paper_id] = result
                    first_seen[result.paper_id] = len(first_seen)
            ranks_by_run.append(run_ranks)

        original_ranks, expanded_ranks = ranks_by_run
        paper_ids = sorted(
            scores,
            key=lambda paper_id: (
                -scores[paper_id],
                first_seen[paper_id],
                paper_id,
            ),
        )

        fused: list[RetrievalResult] = []
        for paper_id in paper_ids:
            representative = representatives[paper_id]
            metadata = dict(representative.metadata)
            metadata["seed_expansion_original_rank"] = original_ranks.get(paper_id)
            metadata["seed_expansion_expanded_rank"] = expanded_ranks.get(paper_id)
            fused.append(
                replace(
                    representative,
                    score=scores[paper_id],
                    metadata=metadata,
                    source="seed_expansion_rrf",
                )
            )
        return fused


@dataclass(frozen=True)
class OpenSetExploration:
    """Insert one candidate supported by several independent seed searches."""

    min_support: int
    max_seed_rank: int
    slot_k: int

    def insert(
        self,
        results: list[RetrievalResult],
        runs: Sequence[tuple[str, list[RetrievalResult]]],
    ) -> list[RetrievalResult]:
        """Preserve the stable head and fill one exploration slot."""

        if not runs or len(results) < self.slot_k:
            return results

        selected = self._select(
            runs,
            excluded_paper_ids={
                result.paper_id for result in results[: self.slot_k]
            },
        )
        attempted = self._mark_attempt(results, len(runs), selected)
        if selected is None:
            return attempted

        original_rank = next(
            (
                rank
                for rank, result in enumerate(attempted, start=1)
                if result.paper_id == selected.paper_id
            ),
            None,
        )
        metadata = dict(selected.metadata)
        metadata.update(
            {
                "open_set_expansion_original_rank": original_rank,
                "open_set_expansion_selected": True,
                "open_set_expansion_slot_k": self.slot_k,
            }
        )
        selected = replace(
            selected,
            metadata=metadata,
            source="open_set_seed_consensus",
        )

        without_selected = [
            result
            for result in attempted
            if result.paper_id != selected.paper_id
        ]
        inserted = [
            *without_selected[: self.slot_k - 1],
            selected,
            *without_selected[self.slot_k - 1 :],
        ][: len(results)]
        return align_scores_with_output_order(inserted)

    def _select(
        self,
        runs: Sequence[tuple[str, list[RetrievalResult]]],
        *,
        excluded_paper_ids: set[str],
    ) -> RetrievalResult | None:
        """Choose the strongest consensus outside the protected head."""

        representatives: dict[str, RetrievalResult] = {}
        support: dict[str, int] = {}
        best_rank: dict[str, int] = {}
        via_papers: dict[str, list[str]] = {}
        first_seen: dict[str, tuple[int, int]] = {}

        for run_index, (seed_paper_id, run) in enumerate(runs):
            for rank, result in enumerate(unique_papers(run), start=1):
                paper_id = result.paper_id
                support[paper_id] = support.get(paper_id, 0) + 1
                previous_best_rank = best_rank.get(paper_id)
                if previous_best_rank is None or rank < previous_best_rank:
                    best_rank[paper_id] = rank
                    representatives[paper_id] = result
                via_papers.setdefault(paper_id, []).append(seed_paper_id)
                first_seen.setdefault(paper_id, (run_index, rank))

        eligible = [
            paper_id
            for paper_id in representatives
            if paper_id not in excluded_paper_ids
            and support[paper_id] >= self.min_support
            and best_rank[paper_id] <= self.max_seed_rank
        ]
        if not eligible:
            return None

        paper_id = min(
            eligible,
            key=lambda candidate_id: (
                best_rank[candidate_id],
                -support[candidate_id],
                first_seen[candidate_id],
                candidate_id,
            ),
        )
        representative = representatives[paper_id]
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "open_set_expansion_best_rank": best_rank[paper_id],
                "open_set_expansion_run_count": len(runs),
                "open_set_expansion_support": support[paper_id],
                "open_set_expansion_via_papers": via_papers[paper_id],
            }
        )
        return replace(representative, metadata=metadata)

    @staticmethod
    def _mark_attempt(
        results: list[RetrievalResult],
        run_count: int,
        selected: RetrievalResult | None,
    ) -> list[RetrievalResult]:
        """Record that the guarded exploration ran, even without a selection."""

        metadata = dict(results[0].metadata)
        metadata.update(
            {
                "open_set_expansion_attempted": True,
                "open_set_expansion_run_count": run_count,
                "open_set_expansion_selected_paper_id": (
                    selected.paper_id if selected is not None else None
                ),
            }
        )
        return [replace(results[0], metadata=metadata), *results[1:]]
