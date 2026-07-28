"""Opt-in retrieval expansion using the highest-ranked paper as a seed.

This module owns parameter validation and the order the stages run in.  The
stages themselves live in sibling modules: :mod:`query`, :mod:`candidates`,
:mod:`relations`, :mod:`dense`, :mod:`protection` and :mod:`final_rerank`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Real

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Reranker, Retriever
from littraceqa.di_pipeline.retrieve.seed_expansion.candidates import (
    CandidateGeneration,
    align_scores_with_output_order,
    append_result_tail,
    unique_papers,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.dense import (
    MAX_DENSE_RECIPROCAL_CANDIDATES,
    MAX_DENSE_TAIL_NEW_PAPERS,
    DenseConsensusExploration,
    DenseReciprocalExploration,
    DenseTailFusion,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.final_rerank import (
    FinalCandidateReranker,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.protection import (
    MAX_PROTECTED_TITLES,
    ExplicitTitleGuard,
    restore_method_protected_candidates,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    QueryPreparation,
    paper_context,
    without_method_hints,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.relations import (
    MethodBridgeExploration,
    MethodRelationExpansion,
    PaperNeighborhoodExpansion,
)


@registry.register("retriever_wrapper", "seed_expansion")
class SeedExpansionRetriever:
    """Fuse an initial search with paper- and optionally chunk-level expansion."""

    def __init__(
        self,
        retriever: Retriever,
        candidate_k: int = 50,
        seed_text_chars: int = 512,
        rrf_k: int = 60,
        max_results: int = 10,
        stable_prefix_k: int | None = None,
        reranker: Reranker | None = None,
        rerank_pool_k: int = 50,
        rerank_final_candidates: bool = False,
        final_rerank_document_chars: int = 2000,
        protect_explicit_title_matches: bool = False,
        max_protected_titles: int = 4,
        local_expansion_weight: float = 0.0,
        literal_attribute_hints: bool = False,
        literal_method_hints: bool = False,
        paper_neighborhood_weight: float = 0.0,
        paper_neighborhood_two_hop_weight: float = 0.0,
        paper_neighborhood_max_hub_degree: int = 4,
        method_owner_weight: float = 0.0,
        method_relation_weight: float = 0.0,
        method_relation_seed_k: int = 3,
        method_relation_max_results: int = 10,
        method_relation_max_new_papers: int = 2,
        method_relation_protected_top_k: int = 8,
        method_topic_weight: float = 0.0,
        method_topic_seed_chars: int = 2000,
        method_topic_seed_k: int = 1,
        method_topic_max_results: int = 50,
        method_bridge_topic_max_rank: int = 0,
        paper_embedding_index_dir: str | None = None,
        method_dense_tail_weight: float = 0.0,
        method_dense_tail_seed_k: int = 1,
        method_dense_tail_max_results: int = 20,
        method_dense_tail_max_new_papers: int = 2,
        paper_dense_tail_weight: float = 0.0,
        paper_dense_tail_seed_k: int = 1,
        paper_dense_tail_max_results: int = 7,
        paper_dense_consensus_seed_k: int = 0,
        paper_dense_consensus_max_results: int = 7,
        paper_dense_consensus_min_support: int = 2,
        paper_dense_reciprocal_seed_k: int = 0,
        paper_dense_reciprocal_forward_k: int = 20,
        paper_dense_reciprocal_reverse_k: int = 10,
        paper_dense_reciprocal_min_support: int = 6,
        paper_dense_reciprocal_max_candidates: int = 32,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if seed_text_chars <= 0:
            raise ValueError("seed_text_chars must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if stable_prefix_k is not None:
            if isinstance(stable_prefix_k, bool) or not isinstance(
                stable_prefix_k, int
            ):
                raise TypeError("stable_prefix_k must be an integer or None")
            if stable_prefix_k <= 0:
                raise ValueError("stable_prefix_k must be positive")
        if rerank_pool_k <= 0:
            raise ValueError("rerank_pool_k must be positive")
        if not isinstance(rerank_final_candidates, bool):
            raise TypeError("rerank_final_candidates must be a boolean")
        if isinstance(final_rerank_document_chars, bool) or not isinstance(
            final_rerank_document_chars,
            int,
        ):
            raise TypeError("final_rerank_document_chars must be an integer")
        if final_rerank_document_chars <= 0:
            raise ValueError("final_rerank_document_chars must be positive")
        if not isinstance(protect_explicit_title_matches, bool):
            raise TypeError("protect_explicit_title_matches must be a boolean")
        if isinstance(max_protected_titles, bool) or not isinstance(
            max_protected_titles, int
        ):
            raise TypeError("max_protected_titles must be an integer")
        if not 1 <= max_protected_titles <= MAX_PROTECTED_TITLES:
            raise ValueError(
                f"max_protected_titles must be between 1 and "
                f"{MAX_PROTECTED_TITLES}"
            )
        if isinstance(local_expansion_weight, bool) or not isinstance(
            local_expansion_weight, Real
        ):
            raise TypeError("local_expansion_weight must be a number")
        if (
            not math.isfinite(local_expansion_weight)
            or local_expansion_weight < 0
        ):
            raise ValueError(
                "local_expansion_weight must be a finite non-negative number"
            )
        if not isinstance(literal_attribute_hints, bool):
            raise TypeError("literal_attribute_hints must be a boolean")
        if not isinstance(literal_method_hints, bool):
            raise TypeError("literal_method_hints must be a boolean")
        if isinstance(paper_neighborhood_weight, bool) or not isinstance(
            paper_neighborhood_weight, Real
        ):
            raise TypeError("paper_neighborhood_weight must be a number")
        if (
            not math.isfinite(paper_neighborhood_weight)
            or paper_neighborhood_weight < 0
        ):
            raise ValueError(
                "paper_neighborhood_weight must be a finite non-negative number"
            )
        if isinstance(paper_neighborhood_two_hop_weight, bool) or not isinstance(
            paper_neighborhood_two_hop_weight, Real
        ):
            raise TypeError(
                "paper_neighborhood_two_hop_weight must be a number"
            )
        if (
            not math.isfinite(paper_neighborhood_two_hop_weight)
            or paper_neighborhood_two_hop_weight < 0
        ):
            raise ValueError(
                "paper_neighborhood_two_hop_weight must be a finite "
                "non-negative number"
            )
        if isinstance(paper_neighborhood_max_hub_degree, bool) or not isinstance(
            paper_neighborhood_max_hub_degree, int
        ):
            raise TypeError(
                "paper_neighborhood_max_hub_degree must be an integer"
            )
        if paper_neighborhood_max_hub_degree <= 0:
            raise ValueError(
                "paper_neighborhood_max_hub_degree must be positive"
            )
        for name, value in (
            ("method_owner_weight", method_owner_weight),
            ("method_relation_weight", method_relation_weight),
            ("method_topic_weight", method_topic_weight),
            ("method_dense_tail_weight", method_dense_tail_weight),
            ("paper_dense_tail_weight", paper_dense_tail_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be a finite non-negative number"
                )
        for name, value in (
            ("method_topic_seed_chars", method_topic_seed_chars),
            ("method_topic_seed_k", method_topic_seed_k),
            ("method_topic_max_results", method_topic_max_results),
            ("method_relation_seed_k", method_relation_seed_k),
            ("method_relation_max_results", method_relation_max_results),
            ("method_dense_tail_seed_k", method_dense_tail_seed_k),
            ("method_dense_tail_max_results", method_dense_tail_max_results),
            ("paper_dense_tail_seed_k", paper_dense_tail_seed_k),
            ("paper_dense_tail_max_results", paper_dense_tail_max_results),
            (
                "paper_dense_consensus_max_results",
                paper_dense_consensus_max_results,
            ),
            (
                "paper_dense_consensus_min_support",
                paper_dense_consensus_min_support,
            ),
            (
                "paper_dense_reciprocal_forward_k",
                paper_dense_reciprocal_forward_k,
            ),
            (
                "paper_dense_reciprocal_reverse_k",
                paper_dense_reciprocal_reverse_k,
            ),
            (
                "paper_dense_reciprocal_min_support",
                paper_dense_reciprocal_min_support,
            ),
            (
                "paper_dense_reciprocal_max_candidates",
                paper_dense_reciprocal_max_candidates,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(paper_dense_consensus_seed_k, bool) or not isinstance(
            paper_dense_consensus_seed_k,
            int,
        ):
            raise TypeError("paper_dense_consensus_seed_k must be an integer")
        if paper_dense_consensus_seed_k < 0:
            raise ValueError(
                "paper_dense_consensus_seed_k must be non-negative"
            )
        if isinstance(paper_dense_reciprocal_seed_k, bool) or not isinstance(
            paper_dense_reciprocal_seed_k,
            int,
        ):
            raise TypeError(
                "paper_dense_reciprocal_seed_k must be an integer"
            )
        if paper_dense_reciprocal_seed_k < 0:
            raise ValueError(
                "paper_dense_reciprocal_seed_k must be non-negative"
            )
        if isinstance(method_bridge_topic_max_rank, bool) or not isinstance(
            method_bridge_topic_max_rank,
            int,
        ):
            raise TypeError(
                "method_bridge_topic_max_rank must be an integer"
            )
        if method_bridge_topic_max_rank < 0:
            raise ValueError(
                "method_bridge_topic_max_rank must be non-negative"
            )
        if method_bridge_topic_max_rank > method_topic_max_results:
            raise ValueError(
                "method_bridge_topic_max_rank must not exceed "
                "method_topic_max_results"
            )
        if (
            paper_dense_consensus_seed_k > 0
            and paper_dense_consensus_min_support
            > paper_dense_consensus_seed_k
        ):
            raise ValueError(
                "paper_dense_consensus_min_support must not exceed "
                "paper_dense_consensus_seed_k when consensus is enabled"
            )
        if (
            paper_dense_reciprocal_seed_k > 0
            and paper_dense_reciprocal_min_support
            > min(
                paper_dense_reciprocal_seed_k,
                paper_dense_reciprocal_reverse_k,
                max_results,
            )
        ):
            raise ValueError(
                "paper_dense_reciprocal_min_support must not exceed "
                "the seed count, reverse depth, or maximum result count "
                "when reciprocal expansion is enabled"
            )
        if (
            paper_dense_reciprocal_max_candidates
            > MAX_DENSE_RECIPROCAL_CANDIDATES
        ):
            raise ValueError(
                "paper_dense_reciprocal_max_candidates must not exceed "
                f"{MAX_DENSE_RECIPROCAL_CANDIDATES}"
            )
        for name, value in (
            ("method_relation_max_new_papers", method_relation_max_new_papers),
            (
                "method_relation_protected_top_k",
                method_relation_protected_top_k,
            ),
            (
                "method_dense_tail_max_new_papers",
                method_dense_tail_max_new_papers,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            method_dense_tail_max_new_papers
            > MAX_DENSE_TAIL_NEW_PAPERS
        ):
            raise ValueError(
                "method_dense_tail_max_new_papers must not exceed "
                f"{MAX_DENSE_TAIL_NEW_PAPERS}"
            )
        if paper_embedding_index_dir is not None:
            if not isinstance(paper_embedding_index_dir, str):
                raise TypeError(
                    "paper_embedding_index_dir must be a string or None"
                )
            if not paper_embedding_index_dir.strip():
                raise ValueError(
                    "paper_embedding_index_dir must not be empty"
                )
        if getattr(retriever, "reranker", None) is not None:
            raise ValueError(
                "seed expansion cannot wrap a retriever with a reranker because "
                "that would run the reranker twice"
            )
        if rerank_final_candidates and reranker is None:
            raise ValueError(
                "rerank_final_candidates requires an enabled reranker"
            )

        self.retriever = retriever
        self.candidate_k = candidate_k
        self.seed_text_chars = seed_text_chars
        self.rrf_k = rrf_k
        self.max_results = max_results
        self.stable_prefix_k = stable_prefix_k
        self._reranker = reranker
        self.rerank_pool_k = rerank_pool_k
        self.rerank_final_candidates = rerank_final_candidates
        self.final_rerank_document_chars = final_rerank_document_chars
        self.protect_explicit_title_matches = protect_explicit_title_matches
        self.max_protected_titles = max_protected_titles
        self.local_expansion_weight = float(local_expansion_weight)
        self.literal_attribute_hints = literal_attribute_hints
        self.literal_method_hints = literal_method_hints
        self.paper_neighborhood_weight = float(paper_neighborhood_weight)
        self.paper_neighborhood_two_hop_weight = float(
            paper_neighborhood_two_hop_weight
        )
        self.paper_neighborhood_max_hub_degree = (
            paper_neighborhood_max_hub_degree
        )
        self.method_owner_weight = float(method_owner_weight)
        self.method_relation_weight = float(method_relation_weight)
        self.method_topic_weight = float(method_topic_weight)
        self.method_topic_seed_chars = method_topic_seed_chars
        self.method_topic_seed_k = method_topic_seed_k
        self.method_topic_max_results = method_topic_max_results
        self.method_bridge_topic_max_rank = (
            method_bridge_topic_max_rank
        )
        self.paper_embedding_index_dir = paper_embedding_index_dir
        self.method_dense_tail_weight = float(method_dense_tail_weight)
        self.method_dense_tail_seed_k = method_dense_tail_seed_k
        self.method_dense_tail_max_results = method_dense_tail_max_results
        self.method_dense_tail_max_new_papers = (
            method_dense_tail_max_new_papers
        )
        self.paper_dense_tail_weight = float(paper_dense_tail_weight)
        self.paper_dense_tail_seed_k = paper_dense_tail_seed_k
        self.paper_dense_tail_max_results = paper_dense_tail_max_results
        self.paper_dense_consensus_seed_k = paper_dense_consensus_seed_k
        self.paper_dense_consensus_max_results = (
            paper_dense_consensus_max_results
        )
        self.paper_dense_consensus_min_support = (
            paper_dense_consensus_min_support
        )
        self.paper_dense_reciprocal_seed_k = (
            paper_dense_reciprocal_seed_k
        )
        self.paper_dense_reciprocal_forward_k = (
            paper_dense_reciprocal_forward_k
        )
        self.paper_dense_reciprocal_reverse_k = (
            paper_dense_reciprocal_reverse_k
        )
        self.paper_dense_reciprocal_min_support = (
            paper_dense_reciprocal_min_support
        )
        self.paper_dense_reciprocal_max_candidates = (
            paper_dense_reciprocal_max_candidates
        )
        self._paper_embedding_store: object | None = None
        self._paper_embedding_store_unavailable = False
        self.method_relation_seed_k = method_relation_seed_k
        self.method_relation_max_results = method_relation_max_results
        self.method_relation_max_new_papers = (
            method_relation_max_new_papers
        )
        self.method_relation_protected_top_k = (
            method_relation_protected_top_k
        )

        self._build_stages()

    def _build_stages(self) -> None:
        """Wire the processing stages from the validated parameters."""

        self._query = QueryPreparation(
            seed_text_chars=self.seed_text_chars,
            literal_attribute_hints=self.literal_attribute_hints,
            literal_method_hints=self.literal_method_hints,
        )
        self._candidates = CandidateGeneration(
            retriever=self.retriever,
            candidate_k=self.candidate_k,
            rrf_k=self.rrf_k,
            local_expansion_weight=self.local_expansion_weight,
        )
        self._neighborhood = PaperNeighborhoodExpansion(
            rrf_k=self.rrf_k,
            candidate_k=self.candidate_k,
            relation_weight=self.paper_neighborhood_weight,
            two_hop_weight=self.paper_neighborhood_two_hop_weight,
            max_hub_degree=self.paper_neighborhood_max_hub_degree,
        )
        self._method_relations = MethodRelationExpansion(
            retriever=self.retriever,
            rrf_k=self.rrf_k,
            candidate_k=self.candidate_k,
            seed_text_chars=self.seed_text_chars,
            owner_weight=self.method_owner_weight,
            relation_weight=self.method_relation_weight,
            topic_weight=self.method_topic_weight,
            topic_seed_chars=self.method_topic_seed_chars,
            topic_seed_k=self.method_topic_seed_k,
            topic_max_results=self.method_topic_max_results,
            relation_seed_k=self.method_relation_seed_k,
            relation_max_results=self.method_relation_max_results,
            relation_max_new_papers=self.method_relation_max_new_papers,
            relation_protected_top_k=self.method_relation_protected_top_k,
        )
        self._method_bridge = MethodBridgeExploration(
            retriever=self.retriever,
            seed_text_chars=self.seed_text_chars,
            stable_prefix_k=self.stable_prefix_k,
            topic_seed_chars=self.method_topic_seed_chars,
            topic_seed_k=self.method_topic_seed_k,
            topic_max_results=self.method_topic_max_results,
            bridge_topic_max_rank=self.method_bridge_topic_max_rank,
            relation_max_results=self.method_relation_max_results,
        )
        self._dense_tail = DenseTailFusion(
            rrf_k=self.rrf_k,
            seed_text_chars=self.seed_text_chars,
            method_weight=self.method_dense_tail_weight,
            method_seed_k=self.method_dense_tail_seed_k,
            method_max_results=self.method_dense_tail_max_results,
            method_max_new_papers=self.method_dense_tail_max_new_papers,
            paper_weight=self.paper_dense_tail_weight,
            paper_seed_k=self.paper_dense_tail_seed_k,
            paper_max_results=self.paper_dense_tail_max_results,
        )
        self._dense_reciprocal = DenseReciprocalExploration(
            rrf_k=self.rrf_k,
            seed_text_chars=self.seed_text_chars,
            seed_k=self.paper_dense_reciprocal_seed_k,
            forward_k=self.paper_dense_reciprocal_forward_k,
            reverse_k=self.paper_dense_reciprocal_reverse_k,
            min_support=self.paper_dense_reciprocal_min_support,
            max_candidates=self.paper_dense_reciprocal_max_candidates,
        )
        self._dense_consensus = DenseConsensusExploration(
            rrf_k=self.rrf_k,
            seed_text_chars=self.seed_text_chars,
            seed_k=self.paper_dense_consensus_seed_k,
            max_results=self.paper_dense_consensus_max_results,
            min_support=self.paper_dense_consensus_min_support,
        )
        self._title_guard = ExplicitTitleGuard(
            enabled=self.protect_explicit_title_matches,
            max_protected_titles=self.max_protected_titles,
        )
        self._final_rerank = FinalCandidateReranker(
            reranker=self._reranker,
            document_chars=self.final_rerank_document_chars,
        )

    @property
    def indexers(self):
        """Expose the wrapped retriever's indexers to existing scripts."""

        return self.retriever.indexers

    @property
    def reranker(self):
        """Expose the optional final reranker to existing scripts."""

        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve, expand from the top paper, and return a paper-level ranking."""

        if top_k <= 0:
            return []

        effective_hints = self._query.effective_hints(query, hints)
        retrieval_hints = without_method_hints(effective_hints)

        initial = self._candidates.search(query, retrieval_hints)
        if not initial:
            return []

        expanded_query = self._query.expanded_query(query, initial[0])
        if expanded_query is None:
            return self._finalize(
                query,
                unique_papers(initial),
                top_k,
                effective_hints,
            )

        expanded = self._candidates.search(expanded_query, retrieval_hints)
        local_expanded: list[RetrievalResult] | None = None
        if (
            self.local_expansion_weight > 0
            and paper_context(initial[0]) is not None
        ):
            local_query = self._query.legacy_expanded_query(query, initial[0])
            if local_query is not None:
                local_expanded = self._candidates.search(
                    local_query,
                    retrieval_hints,
                )

        if not expanded and local_expanded is None:
            return self._finalize(
                query,
                unique_papers(initial),
                top_k,
                effective_hints,
            )

        fused = self._candidates.fuse_by_paper(initial, expanded, local_expanded)
        return self._finalize(query, fused, top_k, effective_hints)

    def _finalize(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        """Apply the optional final reranker to a bounded paper candidate pool."""

        indexers = self.indexers
        candidates = self._neighborhood.rerank(query, candidates, indexers)
        output_k = min(top_k, self.max_results)
        selection_k = min(
            output_k,
            self.stable_prefix_k
            if self.stable_prefix_k is not None
            else output_k,
        )
        candidates, method_protected = self._method_relations.expand(
            candidates,
            hints,
            indexers,
            output_k=selection_k,
        )
        reserved_paper_ids = {
            candidate.paper_id for candidate in method_protected
        }
        if self.reranker is None or self.rerank_final_candidates:
            selected = restore_method_protected_candidates(
                candidates,
                candidates[:selection_k],
                method_protected,
                selection_k,
            )
            selected = self._title_guard.restore(
                query,
                candidates,
                selected,
                selection_k,
                reserved_paper_ids=reserved_paper_ids,
            )
            finalized = self._assemble_output(
                candidates,
                selected,
                hints,
                indexers,
                output_k=output_k,
                selection_k=selection_k,
            )
            if self.rerank_final_candidates:
                return self._final_rerank.rerank(query, finalized, indexers)
            return finalized

        candidate_pool = candidates[: self.rerank_pool_k]
        selected = self.reranker.rerank(
            query,
            candidate_pool,
            selection_k,
        )[:selection_k]
        selected = restore_method_protected_candidates(
            candidates,
            selected,
            method_protected,
            selection_k,
        )
        selected = self._title_guard.restore(
            query,
            candidate_pool,
            selected,
            selection_k,
            reserved_paper_ids=reserved_paper_ids,
        )
        if not selected:
            return []

        metadata = dict(selected[0].metadata)
        metadata["pre_rerank_candidate_papers"] = [
            candidate.paper_id for candidate in candidate_pool
        ]
        selected[0] = replace(selected[0], metadata=metadata)
        return self._assemble_output(
            candidates,
            selected,
            hints,
            indexers,
            output_k=output_k,
            selection_k=selection_k,
        )

    def _assemble_output(
        self,
        candidates: list[RetrievalResult],
        selected: list[RetrievalResult],
        hints: SearchHints | None,
        indexers,
        *,
        output_k: int,
        selection_k: int,
    ) -> list[RetrievalResult]:
        """Extend the stable prefix with a dense tail, then explore one slot."""

        if selection_k < output_k:
            tail_candidates = self._dense_tail.fuse(
                candidates,
                selected,
                indexers,
                self._get_paper_embedding_store,
                hints=hints,
                has_embedding_index=self.paper_embedding_index_dir is not None,
            )
            results = append_result_tail(tail_candidates, selected, output_k)
        else:
            results = align_scores_with_output_order(selected)
        return self._replace_last_with_related_expansion(
            results,
            hints,
            indexers,
        )

    def _replace_last_with_related_expansion(
        self,
        results: list[RetrievalResult],
        hints: SearchHints | None,
        indexers,
    ) -> list[RetrievalResult]:
        """Apply one bounded relation signal without changing the stable prefix."""

        has_embedding_index = self.paper_embedding_index_dir is not None
        reciprocal = self._dense_reciprocal.replace_last(
            results,
            indexers,
            self._get_paper_embedding_store,
            has_embedding_index=has_embedding_index,
        )
        if reciprocal is not results:
            return reciprocal
        bridged = self._method_bridge.replace_last(results, hints, indexers)
        if bridged is not results:
            return bridged
        return self._dense_consensus.replace_last(
            results,
            indexers,
            self._get_paper_embedding_store,
            has_embedding_index=has_embedding_index,
        )

    def _get_paper_embedding_store(self):
        """Load the model-free paper embedding store only when the lane is used."""

        if (
            self._paper_embedding_store_unavailable
            or self.paper_embedding_index_dir is None
        ):
            return None
        if self._paper_embedding_store is not None:
            return self._paper_embedding_store
        try:
            from littraceqa.di_pipeline.index.paper_embedding import (
                PaperEmbeddingStore,
            )

            store = PaperEmbeddingStore(self.paper_embedding_index_dir)
            store.load()
        except Exception:
            self._paper_embedding_store_unavailable = True
            return None
        self._paper_embedding_store = store
        return store

    @staticmethod
    def _align_scores_with_output_order(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Keep downstream score aggregation consistent with the final order."""

        return align_scores_with_output_order(results)
