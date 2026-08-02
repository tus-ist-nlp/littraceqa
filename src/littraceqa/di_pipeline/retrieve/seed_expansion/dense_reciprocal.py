"""Reciprocal dense exploration.

Spends exactly one final slot, and only on a paper that points back to enough
of the leading seeds. The reciprocal requirement is what keeps a single strong
but unrelated neighbour from taking the slot.
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

MAX_DENSE_RECIPROCAL_CANDIDATES = 128


@dataclass(frozen=True)
class DenseReciprocalExploration:
    """Explores one slot only when a new paper points back to many seeds."""

    rrf_k: int
    seed_text_chars: int
    seed_k: int
    forward_k: int
    reverse_k: int
    min_support: int
    max_candidates: int

    def replace_last(
        self,
        results: list[RetrievalResult],
        indexers,
        get_embedding_store,
        *,
        has_embedding_index: bool,
    ) -> list[RetrievalResult]:
        """Swap the final slot for a reciprocal neighbour, or return as is."""

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

        seed_id_set = set(seed_ids)
        try:
            candidates_by_id = self._collect_forward(
                embedding_store,
                seed_ids,
                result_ids,
            )
            forward_ranked = sorted(
                candidates_by_id.values(),
                key=lambda state: (
                    -len(state["forward_via_papers"]),
                    -state["forward_rrf_score"],
                    state["best_forward_rank"],
                    -state["best_similarity"],
                    state["paper_id"],
                ),
            )
            examined = forward_ranked[: self.max_candidates]
            eligible = self._examine_reverse(
                embedding_store,
                examined,
                seed_id_set,
            )
            eligible.sort(
                key=lambda state: (
                    -len(state["reverse_seed_ranks"]),
                    -len(state["forward_via_papers"]),
                    -(
                        state["forward_rrf_score"]
                        + state["reverse_rrf_score"]
                    ),
                    state["best_forward_rank"],
                    -state["best_similarity"],
                    state["paper_id"],
                )
            )
            selected_state = None
            selected_document = None
            for state in eligible:
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

        reverse_seed_ranks = selected_state["reverse_seed_ranks"]
        metadata = (
            dict(selected_document.metadata)
            if isinstance(selected_document.metadata, dict)
            else {}
        )
        metadata.update(
            {
                "paper_dense_reciprocal_seed_count": len(seed_ids),
                "paper_dense_reciprocal_discovered_candidates": len(
                    candidates_by_id
                ),
                "paper_dense_reciprocal_examined_candidates": len(examined),
                "paper_dense_reciprocal_support": len(reverse_seed_ranks),
                "paper_dense_reciprocal_forward_support": len(
                    selected_state["forward_via_papers"]
                ),
                "paper_dense_reciprocal_best_forward_rank": selected_state[
                    "best_forward_rank"
                ],
                "paper_dense_reciprocal_best_reverse_rank": min(
                    reverse_seed_ranks.values()
                ),
                "paper_dense_reciprocal_best_similarity": selected_state[
                    "best_similarity"
                ],
                "paper_dense_reciprocal_forward_rrf_score": selected_state[
                    "forward_rrf_score"
                ],
                "paper_dense_reciprocal_reverse_rrf_score": selected_state[
                    "reverse_rrf_score"
                ],
                "paper_dense_reciprocal_forward_via_papers": sorted(
                    selected_state["forward_via_papers"]
                ),
                "paper_dense_reciprocal_reverse_via_papers": sorted(
                    reverse_seed_ranks
                ),
                "paper_dense_reciprocal_replaced_paper_id": results[-1].paper_id,
                "paper_dense_reciprocal_is_new": True,
            }
        )
        replacement = RetrievalResult(
            chunk_id=selected_document.chunk_id,
            paper_id=selected_state["paper_id"],
            score=results[-1].score,
            text=selected_document.text[: self.seed_text_chars],
            chunk_type=selected_document.chunk_type,
            metadata=metadata,
            source="paper_dense_reciprocal_exploration",
        )
        return [*results[:-1], replacement]

    def _collect_forward(
        self,
        embedding_store,
        seed_ids: list[str],
        result_ids: set[str],
    ) -> dict[str, dict]:
        """Gather papers the seeds point at, with RRF-weighted forward support."""

        candidates_by_id: dict[str, dict] = {}
        for seed_id in seed_ids:
            neighbors = tuple(
                embedding_store.search_by_paper_id(seed_id, self.forward_k)
            )
            seen_for_seed: set[str] = set()
            for forward_rank, neighbor in enumerate(neighbors, start=1):
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
                similarity = finite_similarity(neighbor)
                if similarity is None:
                    continue
                seen_for_seed.add(paper_id)
                state = candidates_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "forward_via_papers": set(),
                        "best_forward_rank": forward_rank,
                        "best_similarity": similarity,
                        "forward_rrf_score": 0.0,
                    },
                )
                state["forward_via_papers"].add(seed_id)
                state["best_forward_rank"] = min(
                    state["best_forward_rank"],
                    forward_rank,
                )
                state["best_similarity"] = max(
                    state["best_similarity"],
                    similarity,
                )
                state["forward_rrf_score"] += 1.0 / (self.rrf_k + forward_rank)
        return candidates_by_id

    def _examine_reverse(
        self,
        embedding_store,
        examined: list[dict],
        seed_id_set: set[str],
    ) -> list[dict]:
        """Keep only candidates whose own neighbours point back at enough seeds."""

        eligible: list[dict] = []
        for state in examined:
            reverse_neighbors = tuple(
                embedding_store.search_by_paper_id(
                    state["paper_id"],
                    self.reverse_k,
                )
            )
            reverse_seed_ranks: dict[str, int] = {}
            for reverse_rank, neighbor in enumerate(reverse_neighbors, start=1):
                if not isinstance(neighbor, RetrievalResult):
                    continue
                paper_id = neighbor.paper_id
                if (
                    not isinstance(paper_id, str)
                    or paper_id not in seed_id_set
                    or paper_id in reverse_seed_ranks
                ):
                    continue
                reverse_seed_ranks[paper_id] = reverse_rank

            if len(reverse_seed_ranks) < self.min_support:
                continue
            state["reverse_seed_ranks"] = reverse_seed_ranks
            state["reverse_rrf_score"] = sum(
                1.0 / (self.rrf_k + rank)
                for rank in reverse_seed_ranks.values()
            )
            eligible.append(state)
        return eligible
