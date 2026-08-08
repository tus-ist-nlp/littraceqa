"""Opt-in retrieval expansion using the highest-ranked paper as a seed.

This module owns parameter validation and the order the stages run in.  Each
stage lives in its own sibling module: :mod:`query`, :mod:`candidates`,
:mod:`exact_match`, :mod:`structured_filter`, :mod:`dense_tail`,
:mod:`protection` and :mod:`final_rerank`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Reranker, Retriever
from littraceqa.di_pipeline.retrieve.seed_expansion.candidates import (
    CandidateGeneration,
    OpenSetExploration,
    align_scores_with_output_order,
    append_result_tail,
    unique_papers,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.dense_tail import (
    DenseTailFusion,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.structured_filter import (
    StructuredFilterSearch,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.exact_match import (
    ExactMethodMatch,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.final_rerank import (
    FinalCandidateReranker,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.protection import (
    ExplicitTitleGuard,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    QueryPreparation,
    is_open_set_enumeration,
    without_method_hints,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.settings import (
    CandidateSettings,
    DenseSettings,
    OpenSetSettings,
    OutputSettings,
    SeedExpansionSettings,
    SupplementSettings,
    validate_settings,
)


def _with_appended(
    candidates: list[RetrievalResult],
    appended: list[RetrievalResult],
    pool_k: int,
) -> list[RetrievalResult]:
    """Place additions at the bottom of the window the reranker scores.

    Everything the ordinary search ranked above them keeps its position, while
    the additions still reach the model that decides the final order.
    """

    if not appended:
        return candidates
    boundary = max(0, pool_k - len(appended))
    return [*candidates[:boundary], *appended, *candidates[boundary:]]


@dataclass(frozen=True)
class _FinalizeContext:
    """Everything one finalization pass needs besides the candidate list.

    ``requested_output_k`` is what the caller asked for, ``output_k`` is what
    the pipeline assembles (wider when a final reranker inspects a fixed pool)
    and ``selection_k`` is the prefix the selection stages may reorder.
    """

    query: str
    hints: SearchHints | None
    indexers: object
    requested_output_k: int
    output_k: int
    selection_k: int
    open_set_runs: tuple[tuple[str, list[RetrievalResult]], ...]

    def restore_titles(
        self,
        title_guard: ExplicitTitleGuard,
        pool: list[RetrievalResult],
        selected: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-admit explicit title matches that ranking dropped from the output."""

        return title_guard.restore(self.query, pool, selected, self.selection_k)


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
        final_rerank_protected_top_k: int = 0,
        protect_explicit_title_matches: bool = False,
        max_protected_titles: int = 4,
        literal_attribute_hints: bool = False,
        literal_method_hints: bool = False,
        open_set_seed_k: int = 1,
        open_set_min_support: int = 2,
        open_set_max_seed_rank: int = 2,
        open_set_slot_k: int = 20,
        paper_embedding_index_dir: str | None = None,
        method_dense_tail_weight: float = 0.0,
        method_dense_tail_seed_k: int = 1,
        method_dense_tail_max_results: int = 20,
        method_dense_tail_max_new_papers: int = 2,
        exact_method_search: bool = False,
        exact_match_max_papers: int = 5,
        structured_filter: bool = False,
        structured_filter_max_papers: int = 20,
        structured_filter_search_depth: int = 5000,
        structured_filter_protected_prefix_k: int = 5,
        paper_metadata_path: str | None = None,
    ) -> None:
        candidates = CandidateSettings(
            candidate_k=candidate_k,
            seed_text_chars=seed_text_chars,
            rrf_k=rrf_k,
            literal_attribute_hints=literal_attribute_hints,
            literal_method_hints=literal_method_hints,
        )
        output = OutputSettings(
            max_results=max_results,
            stable_prefix_k=stable_prefix_k,
            rerank_pool_k=rerank_pool_k,
            rerank_final_candidates=rerank_final_candidates,
            final_rerank_document_chars=final_rerank_document_chars,
            final_rerank_protected_top_k=final_rerank_protected_top_k,
            protect_explicit_title_matches=protect_explicit_title_matches,
            max_protected_titles=max_protected_titles,
        )
        open_set = OpenSetSettings(
            open_set_seed_k=open_set_seed_k,
            open_set_min_support=open_set_min_support,
            open_set_max_seed_rank=open_set_max_seed_rank,
            open_set_slot_k=open_set_slot_k,
        )
        dense = DenseSettings(
            paper_embedding_index_dir=paper_embedding_index_dir,
            method_dense_tail_weight=method_dense_tail_weight,
            method_dense_tail_seed_k=method_dense_tail_seed_k,
            method_dense_tail_max_results=method_dense_tail_max_results,
            method_dense_tail_max_new_papers=method_dense_tail_max_new_papers,
        )
        supplement = SupplementSettings(
            exact_method_search=exact_method_search,
            exact_match_max_papers=exact_match_max_papers,
            structured_filter=structured_filter,
            structured_filter_max_papers=structured_filter_max_papers,
            structured_filter_search_depth=structured_filter_search_depth,
            structured_filter_protected_prefix_k=structured_filter_protected_prefix_k,
            paper_metadata_path=paper_metadata_path,
        )
        settings = SeedExpansionSettings(
            candidates=candidates,
            supplement=supplement,
            output=output,
            open_set=open_set,
            dense=dense,
        )
        validate_settings(settings, retriever=retriever, reranker=reranker)

        self.retriever = retriever
        self._reranker = reranker
        self._settings = settings.with_float_weights()
        self._paper_embedding_store: object | None = None
        self._paper_embedding_store_unavailable = False
        self._expose_flat_parameters()
        self._build_stages()

    def _expose_flat_parameters(self) -> None:
        """Mirror the grouped settings back onto the historical flat names.

        Scripts and configuration tests read parameters straight off the
        retriever (``retriever.rerank_pool_k`` and friends).  Every settings
        field is named after the YAML key it comes from, so the flat surface is
        exactly the union of the grouped fields.
        """

        for group in (
            self._settings.candidates,
            self._settings.supplement,
            self._settings.output,
            self._settings.open_set,
            self._settings.dense,
        ):
            for parameter in fields(group):
                setattr(self, parameter.name, getattr(group, parameter.name))

    def _build_stages(self) -> None:
        """Wire the processing stages from the validated settings groups."""

        candidates = self._settings.candidates
        output = self._settings.output
        open_set = self._settings.open_set
        dense = self._settings.dense

        self._query = QueryPreparation(
            seed_text_chars=candidates.seed_text_chars,
            literal_attribute_hints=candidates.literal_attribute_hints,
            literal_method_hints=candidates.literal_method_hints,
        )
        self._candidates = CandidateGeneration(
            retriever=self.retriever,
            candidate_k=candidates.candidate_k,
            rrf_k=candidates.rrf_k,
        )
        self._open_set = OpenSetExploration(
            min_support=open_set.open_set_min_support,
            max_seed_rank=open_set.open_set_max_seed_rank,
            slot_k=open_set.open_set_slot_k,
        )
        self._dense_tail = DenseTailFusion(
            rrf_k=candidates.rrf_k,
            seed_text_chars=candidates.seed_text_chars,
            method_weight=dense.method_dense_tail_weight,
            method_seed_k=dense.method_dense_tail_seed_k,
            method_max_results=dense.method_dense_tail_max_results,
            method_max_new_papers=dense.method_dense_tail_max_new_papers,
        )
        supplement = self._settings.supplement
        self._exact_match = ExactMethodMatch(
            enabled=supplement.exact_method_search,
            metadata_path=supplement.paper_metadata_path,
            max_papers=supplement.exact_match_max_papers,
            seed_text_chars=candidates.seed_text_chars,
        )
        self._structured_filter = StructuredFilterSearch(
            enabled=supplement.structured_filter,
            metadata_path=supplement.paper_metadata_path,
            max_papers=supplement.structured_filter_max_papers,
            search_depth=supplement.structured_filter_search_depth,
            seed_text_chars=candidates.seed_text_chars,
        )
        self._title_guard = ExplicitTitleGuard(
            enabled=output.protect_explicit_title_matches,
            max_protected_titles=output.max_protected_titles,
        )
        self._final_rerank = FinalCandidateReranker(
            reranker=self._reranker,
            document_chars=output.final_rerank_document_chars,
            protected_top_k=output.final_rerank_protected_top_k,
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
            open_set_runs = self._open_set_runs(
                query,
                initial,
                retrieval_hints,
                excluded_queries=(),
            )
            return self._finalize(
                query,
                self._supplement_candidates(
                    query,
                    unique_papers(initial),
                    retrieval_hints,
                ),
                top_k,
                effective_hints,
                open_set_runs=open_set_runs,
            )

        expanded = self._candidates.search(expanded_query, retrieval_hints)

        open_set_runs = self._open_set_runs(
            query,
            initial,
            retrieval_hints,
            excluded_queries=(expanded_query,),
        )
        if not expanded:
            return self._finalize(
                query,
                self._supplement_candidates(
                    query,
                    unique_papers(initial),
                    retrieval_hints,
                ),
                top_k,
                effective_hints,
                open_set_runs=open_set_runs,
            )

        return self._finalize(
            query,
            self._supplement_candidates(
                query,
                self._candidates.fuse_by_paper(initial, expanded),
                retrieval_hints,
            ),
            top_k,
            effective_hints,
            open_set_runs=open_set_runs,
        )

    def _supplement_candidates(
        self,
        query: str,
        fused: list[RetrievalResult],
        hints: SearchHints | None,
    ) -> list[RetrievalResult]:
        """Rebuild the middle of the ranking when the question states a scope.

        A question naming a venue, a year and a modality restricts its answer
        to a shortlist that term overlap alone cannot reproduce, so for those
        questions the shortlist order is better than the general one. The head
        of the ordinary ranking is still correct and is left untouched; only
        the band below it is rebuilt.
        """

        supplement = self._settings.supplement
        if not (supplement.structured_filter or supplement.exact_method_search):
            return fused

        # A paper the question names outright but the search never retrieved is
        # simply missing; it is appended rather than promoted, so nothing the
        # ordinary ranking found is displaced.
        appended = self._exact_match.candidates(
            query,
            self.indexers,
            exclude_paper_ids={candidate.paper_id for candidate in fused},
        )

        promoted = self._structured_filter.candidates(
            query,
            self.indexers,
            exclude_paper_ids=set(),
        )
        if not promoted:
            return _with_appended(fused, appended, self.rerank_pool_k)

        prefix_k = supplement.structured_filter_protected_prefix_k
        prefix = fused[:prefix_k]
        protected_ids = {candidate.paper_id for candidate in prefix}
        promoted = [c for c in promoted if c.paper_id not in protected_ids]
        promoted_ids = {candidate.paper_id for candidate in promoted}
        return _with_appended(
            [
                *prefix,
                *promoted,
                *(c for c in fused[prefix_k:] if c.paper_id not in promoted_ids),
            ],
            appended,
            self.rerank_pool_k,
        )

    def _open_set_runs(
        self,
        query: str,
        initial: list[RetrievalResult],
        hints: SearchHints | None,
        *,
        excluded_queries: tuple[str | None, ...],
    ) -> list[tuple[str, list[RetrievalResult]]]:
        """Run bounded, fail-soft searches from additional paper seeds."""

        if (
            self.open_set_seed_k <= 1
            or not is_open_set_enumeration(query)
        ):
            return []

        seen_queries = {
            " ".join(candidate_query.split()).casefold()
            for candidate_query in excluded_queries
            if isinstance(candidate_query, str) and candidate_query.strip()
        }
        runs: list[tuple[str, list[RetrievalResult]]] = []
        seeds = unique_papers(initial)[: self.open_set_seed_k]
        for seed in seeds[1:]:
            expanded_query = self._query.expanded_query(query, seed)
            if expanded_query is None:
                continue
            normalized_query = " ".join(expanded_query.split()).casefold()
            if normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            try:
                run = self._candidates.search(expanded_query, hints)
            except Exception:
                continue
            runs.append((seed.paper_id, run))
        return runs

    def _finalize(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
        hints: SearchHints | None = None,
        *,
        open_set_runs: list[tuple[str, list[RetrievalResult]]] | None = None,
    ) -> list[RetrievalResult]:
        """Rerank, protect and assemble the paper-level ranking to return."""

        indexers = self.indexers
        context = self._finalize_context(
            query,
            top_k,
            hints,
            indexers,
            open_set_runs,
        )
        if self.reranker is None or self.rerank_final_candidates:
            return self._finalize_in_candidate_order(context, candidates)
        return self._finalize_by_pool_rerank(context, candidates)

    def _finalize_context(
        self,
        query: str,
        top_k: int,
        hints: SearchHints | None,
        indexers,
        open_set_runs: list[tuple[str, list[RetrievalResult]]] | None,
    ) -> _FinalizeContext:
        """Resolve how many papers this call selects, assembles and returns."""

        requested_output_k = min(top_k, self.max_results)
        output_k = requested_output_k
        if self.rerank_final_candidates:
            # The final model may inspect a wider fixed pool than the caller
            # consumes. This lets a reader request 20 papers while the
            # reranker scores up to ``rerank_pool_k`` candidates.
            output_k = max(requested_output_k, self.rerank_pool_k)
        selection_k = min(
            output_k,
            self.stable_prefix_k
            if self.stable_prefix_k is not None
            else output_k,
        )
        return _FinalizeContext(
            query=query,
            hints=hints,
            indexers=indexers,
            requested_output_k=requested_output_k,
            output_k=output_k,
            selection_k=selection_k,
            open_set_runs=tuple(open_set_runs or ()),
        )

    def _finalize_in_candidate_order(
        self,
        context: _FinalizeContext,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Keep the fused candidate order, optionally reranking the fixed pool."""

        selection_k = context.selection_k
        selected = context.restore_titles(
            self._title_guard,
            candidates,
            candidates[:selection_k],
        )
        finalized = self._assemble_output(
            candidates,
            selected,
            context.hints,
            context.indexers,
            output_k=context.output_k,
            selection_k=selection_k,
        )
        if self.rerank_final_candidates:
            reranked = self._final_rerank.rerank(
                context.query,
                finalized,
                context.indexers,
            )
            reranked = self._open_set.insert(reranked, context.open_set_runs)
            return reranked[: context.requested_output_k]
        return self._open_set.insert(finalized, context.open_set_runs)

    def _finalize_by_pool_rerank(
        self,
        context: _FinalizeContext,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Let the wrapped reranker choose the prefix from a bounded pool."""

        selection_k = context.selection_k
        candidate_pool = candidates[: self.rerank_pool_k]
        selected = self.reranker.rerank(
            context.query,
            candidate_pool,
            selection_k,
        )[:selection_k]
        selected = context.restore_titles(
            self._title_guard,
            candidate_pool,
            selected,
        )
        if not selected:
            return []

        metadata = dict(selected[0].metadata)
        metadata["pre_rerank_candidate_papers"] = [
            candidate.paper_id for candidate in candidate_pool
        ]
        selected[0] = replace(selected[0], metadata=metadata)
        finalized = self._assemble_output(
            candidates,
            selected,
            context.hints,
            context.indexers,
            output_k=context.output_k,
            selection_k=selection_k,
        )
        return self._open_set.insert(finalized, context.open_set_runs)

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
        return results

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
