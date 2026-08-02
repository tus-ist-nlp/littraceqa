"""Consensus dense exploration.

Spends exactly one final slot on the paper that several leading seeds agree
on. Runs only after the reciprocal and bridge lanes decline the slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.seed_expansion.dense_neighbors import (
    finite_similarity,
    leading_seed_ids,
    usable_document,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)


@dataclass(frozen=True)
class DenseConsensusExploration:
    """Uses multi-seed dense agreement to explore exactly one final slot."""

    rrf_k: int
    seed_text_chars: int
    seed_k: int
    max_results: int
    min_support: int

    def replace_last(
        self,
        results: list[RetrievalResult],
        indexers,
        get_embedding_store,
        *,
        has_embedding_index: bool,
    ) -> list[RetrievalResult]:
        """Swap the final slot for an agreed-upon neighbour, or return as is."""

        if (
            self.seed_k <= 0
            or not has_embedding_index
            or len(results) < self.min_support
        ):
            return results

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return results

        embedding_store = get_embedding_store()
        if embedding_store is None:
            return results

        result_ids = {result.paper_id for result in results}
        seed_ids = leading_seed_ids(results, self.seed_k)
        if len(seed_ids) < self.min_support:
            return results

        try:
            consensus_by_id = self._collect_consensus(
                embedding_store,
                seed_ids,
                result_ids,
            )
            ranked = sorted(
                (
                    state
                    for state in consensus_by_id.values()
                    if len(state["via_papers"]) >= self.min_support
                ),
                key=lambda state: (
                    -state["rrf_score"],
                    -len(state["via_papers"]),
                    state["best_neighbor_rank"],
                    -state["best_similarity"],
                    state["paper_id"],
                ),
            )
            selected_state = None
            selected_document = None
            for state in ranked:
                document = usable_document(paper_index, state["paper_id"])
                if document is None:
                    continue
                selected_state = state
                selected_document = document
                break
        except Exception:
            return results

        if selected_state is None or selected_document is None:
            return results

        metadata = (
            dict(selected_document.metadata)
            if isinstance(selected_document.metadata, dict)
            else {}
        )
        metadata.update(
            {
                "paper_dense_consensus_support": len(
                    selected_state["via_papers"]
                ),
                "paper_dense_consensus_via_papers": sorted(
                    selected_state["via_papers"]
                ),
                "paper_dense_consensus_best_neighbor_rank": selected_state[
                    "best_neighbor_rank"
                ],
                "paper_dense_consensus_best_similarity": selected_state[
                    "best_similarity"
                ],
                "paper_dense_consensus_rrf_score": selected_state["rrf_score"],
                "paper_dense_consensus_replaced_paper_id": results[-1].paper_id,
                "paper_dense_consensus_is_new": True,
            }
        )
        replacement = RetrievalResult(
            chunk_id=selected_document.chunk_id,
            paper_id=selected_state["paper_id"],
            score=results[-1].score,
            text=selected_document.text[: self.seed_text_chars],
            chunk_type=selected_document.chunk_type,
            metadata=metadata,
            source="paper_dense_consensus_exploration",
        )
        return [*results[:-1], replacement]

    def _collect_consensus(
        self,
        embedding_store,
        seed_ids: list[str],
        result_ids: set[str],
    ) -> dict[str, dict]:
        """Count how many seeds independently reach each new paper."""

        consensus_by_id: dict[str, dict] = {}
        for seed_id in seed_ids:
            neighbors = tuple(
                embedding_store.search_by_paper_id(seed_id, self.max_results)
            )
            seen_for_seed: set[str] = set()
            for neighbor_rank, neighbor in enumerate(neighbors, start=1):
                if not isinstance(neighbor, RetrievalResult):
                    continue
                paper_id = neighbor.paper_id
                if (
                    not isinstance(paper_id, str)
                    or not paper_id
                    or paper_id in result_ids
                    or paper_id in seen_for_seed
                ):
                    continue
                seen_for_seed.add(paper_id)
                similarity = finite_similarity(neighbor)
                if similarity is None:
                    continue
                state = consensus_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "via_papers": set(),
                        "best_neighbor_rank": neighbor_rank,
                        "best_similarity": similarity,
                        "rrf_score": 0.0,
                    },
                )
                state["via_papers"].add(seed_id)
                state["best_neighbor_rank"] = min(
                    state["best_neighbor_rank"],
                    neighbor_rank,
                )
                state["best_similarity"] = max(
                    state["best_similarity"],
                    similarity,
                )
                state["rrf_score"] += 1.0 / (self.rrf_k + neighbor_rank)
        return consensus_by_id
