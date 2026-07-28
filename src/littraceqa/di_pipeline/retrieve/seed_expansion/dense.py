"""Dense-neighbor expansion.

Reaches papers that share an embedding neighbourhood with papers already
selected.  The tail lane fuses neighbours into the positions after the stable
prefix; the reciprocal and consensus lanes each spend exactly one final slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.relations import (
    find_method_owner_records,
)


MAX_DENSE_TAIL_NEW_PAPERS = 10
MAX_DENSE_RECIPROCAL_CANDIDATES = 128

_TAIL_LANE_PREFIXES = (
    ("method", "method_dense_tail"),
    ("paper", "paper_dense_tail"),
)


def _finite_similarity(result) -> float | None:
    """Return the neighbour score as a float, or ``None`` when unusable."""

    similarity = getattr(result, "score", None)
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, Real)
        or not math.isfinite(similarity)
    ):
        return None
    return float(similarity)


def _usable_document(paper_index, paper_id: str) -> Chunk | None:
    """Return a non-empty paper document, or ``None`` when unusable."""

    document = paper_index.get_document(paper_id)
    if (
        not isinstance(document, Chunk)
        or document.paper_id != paper_id
        or not isinstance(document.text, str)
        or not document.text.strip()
    ):
        return None
    return document


def _leading_seed_ids(results: list[RetrievalResult], seed_k: int) -> list[str]:
    """Collect the first distinct paper IDs to use as dense seeds."""

    seed_ids: list[str] = []
    for result in results[:seed_k]:
        paper_id = result.paper_id
        if not isinstance(paper_id, str) or not paper_id or paper_id in seed_ids:
            continue
        seed_ids.append(paper_id)
    return seed_ids


@dataclass(frozen=True)
class DenseTailFusion:
    """Fuses bounded paper neighbours only after the stable selected prefix."""

    rrf_k: int
    seed_text_chars: int
    method_weight: float
    method_seed_k: int
    method_max_results: int
    method_max_new_papers: int
    paper_weight: float
    paper_seed_k: int
    paper_max_results: int

    def fuse(
        self,
        candidates: list[RetrievalResult],
        selected_prefix: list[RetrievalResult],
        indexers,
        get_embedding_store,
        *,
        hints: SearchHints | None,
        has_embedding_index: bool,
    ) -> list[RetrievalResult]:
        """Rebuild the tail from dense neighbours, or return ``candidates``."""

        use_method_seeds = (
            self.method_weight > 0
            and hints is not None
            and bool(hints.methods)
        )
        use_prefix_seeds = self.paper_weight > 0
        if not has_embedding_index or not (use_method_seeds or use_prefix_seeds):
            return candidates

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return candidates

        prefix_ids = {result.paper_id for result in selected_prefix}
        if not prefix_ids:
            return candidates

        seed_specs = self._collect_seed_specs(
            paper_index,
            selected_prefix,
            prefix_ids,
            hints=hints,
            use_method_seeds=use_method_seeds,
            use_prefix_seeds=use_prefix_seeds,
        )
        if not seed_specs:
            return candidates

        embedding_store = get_embedding_store()
        if embedding_store is None:
            return candidates

        baseline_tail: list[RetrievalResult] = []
        baseline_by_id: dict[str, RetrievalResult] = {}
        for candidate in candidates:
            if (
                candidate.paper_id in prefix_ids
                or candidate.paper_id in baseline_by_id
            ):
                continue
            baseline_by_id[candidate.paper_id] = candidate
            baseline_tail.append(candidate)
        baseline_rank_by_id = {
            candidate.paper_id: rank
            for rank, candidate in enumerate(baseline_tail, start=1)
        }

        valid_documents: dict[str, Chunk] = {}
        dense_by_id = self._collect_neighbors(
            paper_index,
            embedding_store,
            seed_specs,
            prefix_ids,
            valid_documents,
        )
        if not dense_by_id:
            return candidates

        ranked_dense = sorted(
            dense_by_id.values(),
            key=lambda state: (
                -state["rrf_score"],
                state["best_result_rank"],
                state["best_seed_rank"],
                state["paper_id"],
            ),
        )
        allowed_ids = self._select_allowed(ranked_dense, baseline_by_id)
        if not allowed_ids:
            return candidates

        dense_rank_by_id = {
            state["paper_id"]: rank
            for rank, state in enumerate(ranked_dense, start=1)
        }
        return self._rebuild_tail(
            baseline_by_id,
            baseline_tail,
            baseline_rank_by_id,
            dense_by_id,
            dense_rank_by_id,
            ranked_dense,
            allowed_ids,
            valid_documents,
        )

    def _collect_seed_specs(
        self,
        paper_index,
        selected_prefix: list[RetrievalResult],
        prefix_ids: set[str],
        *,
        hints: SearchHints | None,
        use_method_seeds: bool,
        use_prefix_seeds: bool,
    ) -> dict[str, dict[str, tuple[float, int]]]:
        """Pick the prefix papers to expand from, per lane."""

        seed_specs: dict[str, dict[str, tuple[float, int]]] = {}
        if use_method_seeds and callable(
            getattr(paper_index, "find_method_owners", None)
        ):
            try:
                owner_records = tuple(
                    find_method_owner_records(
                        paper_index,
                        hints.methods,
                        selected_prefix,
                        limit=max(self.method_seed_k, len(selected_prefix)),
                    )
                )
            except Exception:
                owner_records = ()

            method_seed_count = 0
            for record in owner_records:
                if not isinstance(record, dict):
                    continue
                paper_id = record.get("paper_id")
                if (
                    not isinstance(paper_id, str)
                    or paper_id not in prefix_ids
                    or paper_id in seed_specs
                ):
                    continue
                try:
                    owner_document = paper_index.get_document(paper_id)
                except Exception:
                    continue
                if (
                    not isinstance(owner_document, Chunk)
                    or owner_document.paper_id != paper_id
                ):
                    continue
                seed_specs[paper_id] = {
                    "method": (self.method_weight, self.method_max_results)
                }
                method_seed_count += 1
                if method_seed_count >= self.method_seed_k:
                    break

        if use_prefix_seeds:
            for result in selected_prefix[: self.paper_seed_k]:
                paper_id = result.paper_id
                if not isinstance(paper_id, str) or not paper_id:
                    continue
                try:
                    document = paper_index.get_document(paper_id)
                except Exception:
                    continue
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != paper_id
                ):
                    continue
                seed_specs.setdefault(paper_id, {})["paper"] = (
                    self.paper_weight,
                    self.paper_max_results,
                )

        return seed_specs

    def _collect_neighbors(
        self,
        paper_index,
        embedding_store,
        seed_specs: dict[str, dict[str, tuple[float, int]]],
        prefix_ids: set[str],
        valid_documents: dict[str, Chunk],
    ) -> dict[str, dict]:
        """Accumulate per-paper RRF evidence across every seed and lane."""

        dense_by_id: dict[str, dict] = {}
        seed_ids = tuple(seed_specs)

        for seed_rank, (seed_id, lanes) in enumerate(seed_specs.items(), start=1):
            search_limit = max(limit for _, limit in lanes.values())
            try:
                dense_results = tuple(
                    embedding_store.search_by_paper_id(seed_id, search_limit)
                )
            except Exception:
                continue
            seen_for_seed: set[str] = set()
            for result_rank, result in enumerate(dense_results, start=1):
                if not isinstance(result, RetrievalResult):
                    continue
                paper_id = result.paper_id
                if (
                    not isinstance(paper_id, str)
                    or paper_id in prefix_ids
                    or paper_id in seed_ids
                    or paper_id in seen_for_seed
                ):
                    continue
                eligible_lanes = {
                    lane: weight
                    for lane, (weight, limit) in lanes.items()
                    if result_rank <= limit
                }
                if not eligible_lanes:
                    continue
                similarity = _finite_similarity(result)
                if similarity is None:
                    continue
                seen_for_seed.add(paper_id)
                if paper_id not in valid_documents:
                    try:
                        document = paper_index.get_document(paper_id)
                    except Exception:
                        continue
                    if (
                        not isinstance(document, Chunk)
                        or document.paper_id != paper_id
                        or not isinstance(document.text, str)
                        or not document.text.strip()
                    ):
                        continue
                    valid_documents[paper_id] = document

                state = dense_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "best_result_rank": result_rank,
                        "best_seed_rank": seed_rank,
                        "best_similarity": similarity,
                        "rrf_score": 0.0,
                        "via_papers": set(),
                        "via_by_lane": {"method": set(), "paper": set()},
                        "best_result_rank_by_lane": {},
                        "best_similarity_by_lane": {},
                    },
                )
                state["best_result_rank"] = min(
                    state["best_result_rank"],
                    result_rank,
                )
                state["best_seed_rank"] = min(state["best_seed_rank"], seed_rank)
                state["best_similarity"] = max(
                    state["best_similarity"],
                    similarity,
                )
                state["rrf_score"] += max(eligible_lanes.values()) / (
                    self.rrf_k + result_rank
                )
                state["via_papers"].add(seed_id)
                for lane in eligible_lanes:
                    state["via_by_lane"][lane].add(seed_id)
                    previous_rank = state["best_result_rank_by_lane"].get(lane)
                    if previous_rank is None:
                        state["best_result_rank_by_lane"][lane] = result_rank
                    else:
                        state["best_result_rank_by_lane"][lane] = min(
                            previous_rank,
                            result_rank,
                        )
                    previous_similarity = state["best_similarity_by_lane"].get(
                        lane
                    )
                    if previous_similarity is None:
                        state["best_similarity_by_lane"][lane] = similarity
                    else:
                        state["best_similarity_by_lane"][lane] = max(
                            previous_similarity,
                            similarity,
                        )
        return dense_by_id

    def _select_allowed(
        self,
        ranked_dense: list[dict],
        baseline_by_id: dict[str, RetrievalResult],
    ) -> set[str]:
        """Admit every reranked baseline paper plus a bounded number of new ones."""

        allowed_ids = {
            state["paper_id"]
            for state in ranked_dense
            if state["paper_id"] in baseline_by_id
        }
        added_count = 0
        for state in ranked_dense:
            paper_id = state["paper_id"]
            if paper_id in allowed_ids:
                continue
            if added_count >= self.method_max_new_papers:
                break
            allowed_ids.add(paper_id)
            added_count += 1
        return allowed_ids

    def _rebuild_tail(
        self,
        baseline_by_id: dict[str, RetrievalResult],
        baseline_tail: list[RetrievalResult],
        baseline_rank_by_id: dict[str, int],
        dense_by_id: dict[str, dict],
        dense_rank_by_id: dict[str, int],
        ranked_dense: list[dict],
        allowed_ids: set[str],
        valid_documents: dict[str, Chunk],
    ) -> list[RetrievalResult]:
        """Fuse baseline tail ranks with dense ranks and re-sort the tail."""

        representatives = dict(baseline_by_id)
        for paper_id in allowed_ids:
            if paper_id in representatives:
                continue
            document = valid_documents[paper_id]
            representatives[paper_id] = RetrievalResult(
                chunk_id=document.chunk_id,
                paper_id=paper_id,
                score=0.0,
                text=document.text[: self.seed_text_chars],
                chunk_type=document.chunk_type,
                metadata=(
                    dict(document.metadata)
                    if isinstance(document.metadata, dict)
                    else {}
                ),
                source="method_dense_tail",
            )

        scored_tail: list[tuple[float, int, int, str, RetrievalResult]] = []
        for paper_id, representative in representatives.items():
            baseline_rank = baseline_rank_by_id.get(paper_id)
            state = dense_by_id.get(paper_id)
            fused_score = (
                1.0 / (self.rrf_k + baseline_rank)
                if baseline_rank is not None
                else 0.0
            )
            dense_rank = len(ranked_dense) + 1
            result = representative
            if state is not None and paper_id in allowed_ids:
                fused_score += state["rrf_score"]
                dense_rank = dense_rank_by_id[paper_id]
                metadata = dict(representative.metadata)
                for lane, prefix in _TAIL_LANE_PREFIXES:
                    via_papers = state["via_by_lane"][lane]
                    if not via_papers:
                        continue
                    metadata.update(
                        {
                            f"{prefix}_baseline_rank": baseline_rank,
                            f"{prefix}_rank": dense_rank,
                            f"{prefix}_best_neighbor_rank": state[
                                "best_result_rank_by_lane"
                            ][lane],
                            f"{prefix}_best_similarity": state[
                                "best_similarity_by_lane"
                            ][lane],
                            f"{prefix}_via_papers": sorted(via_papers),
                            f"{prefix}_rrf_score": fused_score,
                            f"{prefix}_is_new": baseline_rank is None,
                        }
                    )
                has_method_lane = bool(state["via_by_lane"]["method"])
                has_paper_lane = bool(state["via_by_lane"]["paper"])
                if has_method_lane and has_paper_lane:
                    result_source = "paper_method_dense_tail_rrf"
                elif has_paper_lane:
                    result_source = "paper_dense_tail_rrf"
                else:
                    result_source = "method_dense_tail_rrf"
                result = replace(
                    representative,
                    metadata=metadata,
                    source=result_source,
                )
            scored_tail.append(
                (
                    fused_score,
                    baseline_rank
                    if baseline_rank is not None
                    else len(baseline_tail) + 1,
                    dense_rank,
                    paper_id,
                    result,
                )
            )

        scored_tail.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        return [item[-1] for item in scored_tail]


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
        seed_ids = _leading_seed_ids(results, self.seed_k)
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
                document = _usable_document(paper_index, state["paper_id"])
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
                similarity = _finite_similarity(neighbor)
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
        seed_ids = _leading_seed_ids(results, self.seed_k)
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
                document = _usable_document(paper_index, state["paper_id"])
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
                similarity = _finite_similarity(neighbor)
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
