"""Dense tail fusion.

Rebuilds the positions after the stable selected prefix by fusing the baseline
tail rank with the paper-embedding neighbours of the prefix papers that own a
method the question names. Admits every reranked baseline paper but only a
bounded number of genuinely new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.seed_expansion.dense_neighbors import (
    finite_similarity,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.method_owners import (
    find_method_owner_records,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)

MAX_DENSE_TAIL_NEW_PAPERS = 10


@dataclass(frozen=True)
class DenseTailFusion:
    """Fuses bounded paper neighbours only after the stable selected prefix."""

    rrf_k: int
    seed_text_chars: int
    method_weight: float
    method_seed_k: int
    method_max_results: int
    method_max_new_papers: int

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
        if not has_embedding_index or not use_method_seeds:
            return candidates

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return candidates

        prefix_ids = {result.paper_id for result in selected_prefix}
        if not prefix_ids:
            return candidates

        seed_ids = self._collect_seed_ids(
            paper_index,
            selected_prefix,
            prefix_ids,
            hints=hints,
        )
        if not seed_ids:
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
            seed_ids,
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

    def _collect_seed_ids(
        self,
        paper_index,
        selected_prefix: list[RetrievalResult],
        prefix_ids: set[str],
        *,
        hints: SearchHints | None,
    ) -> tuple[str, ...]:
        """Pick the prefix papers that own a method named in the question."""

        if not callable(getattr(paper_index, "find_method_owners", None)):
            return ()
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
            return ()

        seed_ids: list[str] = []
        for record in owner_records:
            if not isinstance(record, dict):
                continue
            paper_id = record.get("paper_id")
            if (
                not isinstance(paper_id, str)
                or paper_id not in prefix_ids
                or paper_id in seed_ids
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
            seed_ids.append(paper_id)
            if len(seed_ids) >= self.method_seed_k:
                break
        return tuple(seed_ids)

    def _collect_neighbors(
        self,
        paper_index,
        embedding_store,
        seed_ids: tuple[str, ...],
        prefix_ids: set[str],
        valid_documents: dict[str, Chunk],
    ) -> dict[str, dict]:
        """Accumulate per-paper RRF evidence across every seed."""

        dense_by_id: dict[str, dict] = {}

        for seed_rank, seed_id in enumerate(seed_ids, start=1):
            try:
                dense_results = tuple(
                    embedding_store.search_by_paper_id(
                        seed_id,
                        self.method_max_results,
                    )
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
                if result_rank > self.method_max_results:
                    continue
                similarity = finite_similarity(result)
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
                state["rrf_score"] += self.method_weight / (
                    self.rrf_k + result_rank
                )
                state["via_papers"].add(seed_id)
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
                metadata.update(
                    {
                        "method_dense_tail_baseline_rank": baseline_rank,
                        "method_dense_tail_rank": dense_rank,
                        "method_dense_tail_best_neighbor_rank": state[
                            "best_result_rank"
                        ],
                        "method_dense_tail_best_similarity": state[
                            "best_similarity"
                        ],
                        "method_dense_tail_via_papers": sorted(
                            state["via_papers"]
                        ),
                        "method_dense_tail_rrf_score": fused_score,
                        "method_dense_tail_is_new": baseline_rank is None,
                    }
                )
                result = replace(
                    representative,
                    metadata=metadata,
                    source="method_dense_tail_rrf",
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
