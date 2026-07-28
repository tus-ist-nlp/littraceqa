"""Initial candidate generation.

Runs the wrapped retriever for each lane and fuses the runs into one
paper-level ranking, plus the helpers that keep the emitted scores consistent
with the order the papers are finally returned in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Retriever


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
    local_expansion_weight: float

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
        local_expanded: list[RetrievalResult] | None = None,
    ) -> list[RetrievalResult]:
        """Fuse the lanes with reciprocal rank fusion, one entry per paper."""

        scores: dict[str, float] = {}
        representatives: dict[str, RetrievalResult] = {}
        first_seen: dict[str, int] = {}
        ranks_by_run: list[dict[str, int]] = []

        weighted_runs: list[tuple[list[RetrievalResult], float]] = [
            (initial, 1.0),
            (expanded, 1.0),
        ]
        if local_expanded is not None:
            weighted_runs.append((local_expanded, self.local_expansion_weight))

        for run, weight in weighted_runs:
            run_ranks: dict[str, int] = {}
            for result in run:
                if result.paper_id in run_ranks:
                    continue
                rank = len(run_ranks) + 1
                run_ranks[result.paper_id] = rank
                scores[result.paper_id] = scores.get(result.paper_id, 0.0) + (
                    weight / (self.rrf_k + rank)
                )
                if result.paper_id not in representatives:
                    representatives[result.paper_id] = result
                    first_seen[result.paper_id] = len(first_seen)
            ranks_by_run.append(run_ranks)

        original_ranks, expanded_ranks, *optional_ranks = ranks_by_run
        local_ranks = optional_ranks[0] if optional_ranks else None
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
            if local_ranks is not None:
                metadata["seed_expansion_local_rank"] = local_ranks.get(paper_id)
            fused.append(
                replace(
                    representative,
                    score=scores[paper_id],
                    metadata=metadata,
                    source="seed_expansion_rrf",
                )
            )
        return fused
