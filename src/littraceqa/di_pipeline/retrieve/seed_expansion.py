"""Opt-in retrieval expansion using the highest-ranked paper as a seed."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import replace
from numbers import Real

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.attributes import extract_literal_search_hints
from littraceqa.di_pipeline.retrieve.base import Reranker, Retriever
from littraceqa.di_pipeline.retrieve.paper_neighborhood import (
    PaperNeighborhoodReranker,
)


_ALIAS_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC_TITLE_ALIASES = frozenset(
    {
        "ACL",
        "AI",
        "API",
        "AUC",
        "BERT",
        "BLEU",
        "CNN",
        "COCO",
        "CPU",
        "CV",
        "CVPR",
        "DNN",
        "DPO",
        "ECCV",
        "EMNLP",
        "FID",
        "GAN",
        "GPT",
        "GPU",
        "HTML",
        "ICCV",
        "ICLR",
        "ICML",
        "IOU",
        "JSON",
        "LLM",
        "LORA",
        "LSTM",
        "MAE",
        "ML",
        "MLP",
        "MSE",
        "NAACL",
        "NEURIPS",
        "NIPS",
        "NLP",
        "OCR",
        "PDF",
        "QA",
        "RAG",
        "RAM",
        "RL",
        "SOTA",
        "VAE",
        "VIT",
        "VLM",
        "VQA",
    }
)
_MAX_PROTECTED_TITLES = 4
_MAX_DENSE_TAIL_NEW_PAPERS = 10
_MAX_DENSE_RECIPROCAL_CANDIDATES = 128


def _title_alias(result: RetrievalResult) -> str | None:
    """Return a conservative identifier-like prefix from a candidate title."""

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    title = metadata.get("title")
    if not isinstance(title, str):
        return None

    normalized_title = unicodedata.normalize("NFKC", title)
    prefix, separator, _ = normalized_title.partition(":")
    if not separator:
        return None
    alias = " ".join(prefix.split())
    words = _ALIAS_WORD_RE.findall(alias)
    if not (3 <= len(alias) <= 40) or not (1 <= len(words) <= 4):
        return None

    generic_key = re.sub(r"[^A-Z0-9]+", "", alias.upper())
    if generic_key in _GENERIC_TITLE_ALIASES:
        return None

    letters = [character for character in alias if character.isalpha()]
    is_camel_case = bool(re.search(r"[a-z0-9][A-Z]", alias))
    is_all_caps = bool(letters) and all(character.isupper() for character in letters)
    is_identifier_like = (
        is_camel_case
        or is_all_caps
        or any(character.isdigit() for character in alias)
        or "-" in alias
    )
    return alias if is_identifier_like else None


def _standalone_alias_position(query: str, alias: str) -> int | None:
    """Return the first standalone alias position after NFKC normalization."""

    normalized_query = unicodedata.normalize("NFKC", query)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        + re.escape(alias)
        + r"(?![A-Za-z0-9_])"
    )
    match = pattern.search(normalized_query)
    return match.start() if match is not None else None


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
        if not 1 <= max_protected_titles <= _MAX_PROTECTED_TITLES:
            raise ValueError(
                f"max_protected_titles must be between 1 and "
                f"{_MAX_PROTECTED_TITLES}"
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
            > _MAX_DENSE_RECIPROCAL_CANDIDATES
        ):
            raise ValueError(
                "paper_dense_reciprocal_max_candidates must not exceed "
                f"{_MAX_DENSE_RECIPROCAL_CANDIDATES}"
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
            > _MAX_DENSE_TAIL_NEW_PAPERS
        ):
            raise ValueError(
                "method_dense_tail_max_new_papers must not exceed "
                f"{_MAX_DENSE_TAIL_NEW_PAPERS}"
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

        effective_hints = hints
        if effective_hints is None and self.literal_attribute_hints:
            effective_hints = extract_literal_search_hints(
                query,
                include_methods=self.literal_method_hints,
            )
        retrieval_hints = self._without_method_hints(effective_hints)

        initial = self.retriever.retrieve(
            query,
            self.candidate_k,
            hints=retrieval_hints,
        )[: self.candidate_k]
        if not initial:
            return []

        expanded_query = self._expanded_query(query, initial[0])
        if expanded_query is None:
            return self._finalize(
                query,
                self._unique_papers(initial),
                top_k,
                effective_hints,
            )

        expanded = self.retriever.retrieve(
            expanded_query,
            self.candidate_k,
            hints=retrieval_hints,
        )[: self.candidate_k]
        local_expanded: list[RetrievalResult] | None = None
        if (
            self.local_expansion_weight > 0
            and self._paper_context(initial[0]) is not None
        ):
            local_query = self._legacy_expanded_query(query, initial[0])
            if local_query is not None:
                local_expanded = self.retriever.retrieve(
                    local_query,
                    self.candidate_k,
                    hints=retrieval_hints,
                )[: self.candidate_k]

        if not expanded and local_expanded is None:
            return self._finalize(
                query,
                self._unique_papers(initial),
                top_k,
                effective_hints,
            )

        fused = self._fuse_by_paper(initial, expanded, local_expanded)
        return self._finalize(query, fused, top_k, effective_hints)

    def _finalize(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        """Apply the optional final reranker to a bounded paper candidate pool."""

        candidates = self._rerank_paper_neighborhood(query, candidates)
        output_k = min(top_k, self.max_results)
        selection_k = min(
            output_k,
            self.stable_prefix_k
            if self.stable_prefix_k is not None
            else output_k,
        )
        candidates, method_protected = self._rerank_method_relations(
            candidates,
            hints,
            output_k=selection_k,
        )
        if self.reranker is None or self.rerank_final_candidates:
            selected = self._restore_method_protected_candidates(
                candidates,
                candidates[:selection_k],
                method_protected,
                selection_k,
            )
            selected = self._restore_explicit_title_matches(
                query,
                candidates,
                selected,
                selection_k,
                reserved_paper_ids={
                    candidate.paper_id for candidate in method_protected
                },
            )
            if selection_k < output_k:
                tail_candidates = self._fuse_method_dense_tail(
                    candidates,
                    selected,
                    effective_hints=hints,
                )
                finalized = self._replace_last_with_related_expansion(
                    self._append_result_tail(
                        tail_candidates,
                        selected,
                        output_k,
                    ),
                    hints,
                )
            else:
                finalized = self._replace_last_with_related_expansion(
                    self._align_scores_with_output_order(selected),
                    hints,
                )
            if self.rerank_final_candidates:
                return self._rerank_final_candidate_set(query, finalized)
            return finalized

        candidate_pool = candidates[: self.rerank_pool_k]
        selected = self.reranker.rerank(
            query,
            candidate_pool,
            selection_k,
        )[:selection_k]
        selected = self._restore_method_protected_candidates(
            candidates,
            selected,
            method_protected,
            selection_k,
        )
        selected = self._restore_explicit_title_matches(
            query,
            candidate_pool,
            selected,
            selection_k,
            reserved_paper_ids={
                candidate.paper_id for candidate in method_protected
            },
        )
        if not selected:
            return []

        metadata = dict(selected[0].metadata)
        metadata["pre_rerank_candidate_papers"] = [
            candidate.paper_id for candidate in candidate_pool
        ]
        selected[0] = replace(selected[0], metadata=metadata)
        if selection_k < output_k:
            tail_candidates = self._fuse_method_dense_tail(
                candidates,
                selected,
                effective_hints=hints,
            )
            return self._replace_last_with_related_expansion(
                self._append_result_tail(
                    tail_candidates,
                    selected,
                    output_k,
                ),
                hints,
            )
        return self._replace_last_with_related_expansion(
            self._align_scores_with_output_order(selected),
            hints,
        )

    def _rerank_final_candidate_set(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank a completed candidate set without changing its membership."""

        if not candidates:
            return candidates

        try:
            paper_index = next(
                (
                    indexer
                    for indexer in self.indexers
                    if getattr(indexer, "name", None) == "paper_bm25"
                    and callable(getattr(indexer, "get_document", None))
                ),
                None,
            )
            if paper_index is None:
                raise RuntimeError(
                    "final candidate reranking requires paper_bm25 documents"
                )
            if self.reranker is None:
                raise RuntimeError(
                    "final candidate reranking requires an enabled reranker"
                )

            original_ids = [candidate.paper_id for candidate in candidates]
            if len(set(original_ids)) != len(original_ids):
                raise ValueError(
                    "final rerank input contains duplicate paper IDs"
                )

            proxies: list[RetrievalResult] = []
            for candidate in candidates:
                document = paper_index.get_document(candidate.paper_id)
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != candidate.paper_id
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
                    raise ValueError(
                        "paper_bm25 returned an invalid final rerank document "
                        f"for {candidate.paper_id}"
                    )
                proxies.append(
                    replace(
                        candidate,
                        text=document.text[
                            : self.final_rerank_document_chars
                        ],
                    )
                )

            reranked = list(
                self.reranker.rerank(
                    query,
                    proxies,
                    len(proxies),
                )
            )
            if len(reranked) != len(candidates):
                raise ValueError(
                    "final reranker changed the candidate count"
                )
            if not all(
                isinstance(result, RetrievalResult) for result in reranked
            ):
                raise TypeError(
                    "final reranker returned a non-RetrievalResult value"
                )

            reranked_ids = [result.paper_id for result in reranked]
            if (
                len(set(reranked_ids)) != len(reranked_ids)
                or set(reranked_ids) != set(original_ids)
            ):
                raise ValueError(
                    "final reranker changed the candidate paper ID set"
                )

            previous_score = math.inf
            for result in reranked:
                score = result.score
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not math.isfinite(score)
                    or float(score) > previous_score
                ):
                    raise ValueError(
                        "final reranker returned invalid ranking scores"
                    )
                previous_score = float(score)

            original_by_id = {
                candidate.paper_id: candidate for candidate in candidates
            }
            preserved: list[RetrievalResult] = []
            for result in reranked:
                original = original_by_id[result.paper_id]
                metadata = dict(original.metadata)
                if not isinstance(result.metadata, dict):
                    raise TypeError(
                        "final reranker returned invalid metadata"
                    )
                for key, value in result.metadata.items():
                    if key not in original.metadata or key.startswith(
                        ("pre_rerank_", "qwen3_", "rank_fusion_")
                    ):
                        metadata[key] = value
                preserved.append(
                    replace(
                        original,
                        score=float(result.score),
                        metadata=metadata,
                    )
                )
        except Exception as exc:
            metadata = dict(candidates[0].metadata)
            metadata.update(
                {
                    "final_rerank_status": "fallback",
                    "final_rerank_candidate_set_preserved": True,
                    "final_rerank_error_type": type(exc).__name__,
                }
            )
            return [
                replace(candidates[0], metadata=metadata),
                *candidates[1:],
            ]

        metadata = dict(preserved[0].metadata)
        metadata.update(
            {
                "final_rerank_status": "applied",
                "final_rerank_candidate_set_preserved": True,
                "pre_rerank_candidate_papers": original_ids,
            }
        )
        metadata.pop("final_rerank_error_type", None)
        preserved[0] = replace(preserved[0], metadata=metadata)
        return preserved

    def _fuse_method_dense_tail(
        self,
        candidates: list[RetrievalResult],
        selected_prefix: list[RetrievalResult],
        *,
        effective_hints: SearchHints | None,
    ) -> list[RetrievalResult]:
        """Fuse bounded paper neighbors only after the stable selected prefix."""

        use_method_seeds = (
            self.method_dense_tail_weight > 0
            and effective_hints is not None
            and bool(effective_hints.methods)
        )
        use_prefix_seeds = self.paper_dense_tail_weight > 0
        if self.paper_embedding_index_dir is None or not (
            use_method_seeds or use_prefix_seeds
        ):
            return candidates

        paper_index = next(
            (
                indexer
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(getattr(indexer, "get_document", None))
            ),
            None,
        )
        if paper_index is None:
            return candidates

        prefix_ids = {result.paper_id for result in selected_prefix}
        if not prefix_ids:
            return candidates

        seed_specs: dict[str, dict[str, tuple[float, int]]] = {}
        if use_method_seeds and callable(
            getattr(paper_index, "find_method_owners", None)
        ):
            try:
                owner_records = tuple(
                    self._find_method_owner_records(
                        paper_index,
                        effective_hints.methods,
                        selected_prefix,
                        limit=max(
                            self.method_dense_tail_seed_k,
                            len(selected_prefix),
                        ),
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
                    "method": (
                        self.method_dense_tail_weight,
                        self.method_dense_tail_max_results,
                    )
                }
                method_seed_count += 1
                if method_seed_count >= self.method_dense_tail_seed_k:
                    break

        if use_prefix_seeds:
            for result in selected_prefix[: self.paper_dense_tail_seed_k]:
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
                    self.paper_dense_tail_weight,
                    self.paper_dense_tail_max_results,
                )

        if not seed_specs:
            return candidates

        embedding_store = self._get_paper_embedding_store()
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

        dense_by_id: dict[str, dict] = {}
        valid_documents: dict[str, Chunk] = {}
        seed_ids = tuple(seed_specs)

        for seed_rank, (seed_id, lanes) in enumerate(
            seed_specs.items(),
            start=1,
        ):
            search_limit = max(limit for _, limit in lanes.values())
            try:
                dense_results = tuple(
                    embedding_store.search_by_paper_id(
                        seed_id,
                        search_limit,
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
                eligible_lanes = {
                    lane: weight
                    for lane, (weight, limit) in lanes.items()
                    if result_rank <= limit
                }
                if not eligible_lanes:
                    continue
                similarity = getattr(result, "score", None)
                if (
                    isinstance(similarity, bool)
                    or not isinstance(similarity, Real)
                    or not math.isfinite(similarity)
                ):
                    continue
                similarity = float(similarity)
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
                        "via_by_lane": {
                            "method": set(),
                            "paper": set(),
                        },
                        "best_result_rank_by_lane": {},
                        "best_similarity_by_lane": {},
                    },
                )
                state["best_result_rank"] = min(
                    state["best_result_rank"],
                    result_rank,
                )
                state["best_seed_rank"] = min(
                    state["best_seed_rank"],
                    seed_rank,
                )
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
                    previous_similarity = state[
                        "best_similarity_by_lane"
                    ].get(lane)
                    if previous_similarity is None:
                        state["best_similarity_by_lane"][lane] = similarity
                    else:
                        state["best_similarity_by_lane"][lane] = max(
                            previous_similarity,
                            similarity,
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
        dense_rank_by_id = {
            state["paper_id"]: rank
            for rank, state in enumerate(ranked_dense, start=1)
        }
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
            if added_count >= self.method_dense_tail_max_new_papers:
                break
            allowed_ids.add(paper_id)
            added_count += 1
        if not allowed_ids:
            return candidates

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
                for lane, prefix in (
                    ("method", "method_dense_tail"),
                    ("paper", "paper_dense_tail"),
                ):
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

        scored_tail.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
                item[3],
            )
        )
        return [item[-1] for item in scored_tail]

    def _replace_last_with_related_expansion(
        self,
        results: list[RetrievalResult],
        hints: SearchHints | None,
    ) -> list[RetrievalResult]:
        """Apply one bounded relation signal without changing the stable prefix."""

        reciprocal = self._replace_last_with_dense_reciprocal(results)
        if reciprocal is not results:
            return reciprocal
        bridged = self._replace_last_with_method_bridge(results, hints)
        if bridged is not results:
            return bridged
        return self._replace_last_with_dense_consensus(results)

    def _replace_last_with_dense_reciprocal(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Explore one slot only when a new paper points back to many seeds."""

        if (
            self.paper_dense_reciprocal_seed_k <= 0
            or self.paper_embedding_index_dir is None
            or len(results) < self.paper_dense_reciprocal_min_support
        ):
            return results

        paper_index = next(
            (
                indexer
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(getattr(indexer, "get_document", None))
            ),
            None,
        )
        if paper_index is None:
            return results

        embedding_store = self._get_paper_embedding_store()
        if embedding_store is None:
            return results

        result_ids = {result.paper_id for result in results}
        seed_ids: list[str] = []
        for result in results[: self.paper_dense_reciprocal_seed_k]:
            paper_id = result.paper_id
            if (
                not isinstance(paper_id, str)
                or not paper_id
                or paper_id in seed_ids
            ):
                continue
            seed_ids.append(paper_id)
        if len(seed_ids) < self.paper_dense_reciprocal_min_support:
            return results

        seed_id_set = set(seed_ids)
        candidates_by_id: dict[str, dict] = {}
        try:
            for seed_id in seed_ids:
                neighbors = tuple(
                    embedding_store.search_by_paper_id(
                        seed_id,
                        self.paper_dense_reciprocal_forward_k,
                    )
                )
                seen_for_seed: set[str] = set()
                for forward_rank, neighbor in enumerate(
                    neighbors,
                    start=1,
                ):
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
                    similarity = neighbor.score
                    if (
                        isinstance(similarity, bool)
                        or not isinstance(similarity, Real)
                        or not math.isfinite(similarity)
                    ):
                        continue
                    similarity = float(similarity)
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
                    state["forward_rrf_score"] += 1.0 / (
                        self.rrf_k + forward_rank
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
            examined = forward_ranked[
                : self.paper_dense_reciprocal_max_candidates
            ]
            eligible: list[dict] = []
            for state in examined:
                reverse_neighbors = tuple(
                    embedding_store.search_by_paper_id(
                        state["paper_id"],
                        self.paper_dense_reciprocal_reverse_k,
                    )
                )
                reverse_seed_ranks: dict[str, int] = {}
                for reverse_rank, neighbor in enumerate(
                    reverse_neighbors,
                    start=1,
                ):
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

                if (
                    len(reverse_seed_ranks)
                    < self.paper_dense_reciprocal_min_support
                ):
                    continue
                state["reverse_seed_ranks"] = reverse_seed_ranks
                state["reverse_rrf_score"] = sum(
                    1.0 / (self.rrf_k + rank)
                    for rank in reverse_seed_ranks.values()
                )
                eligible.append(state)

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
                document = paper_index.get_document(state["paper_id"])
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != state["paper_id"]
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
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
                "paper_dense_reciprocal_replaced_paper_id": results[
                    -1
                ].paper_id,
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

    def _replace_last_with_method_bridge(
        self,
        results: list[RetrievalResult],
        hints: SearchHints | None,
    ) -> list[RetrievalResult]:
        """Follow a topic-supported method edge from an existing paper."""

        if (
            self.method_bridge_topic_max_rank <= 0
            or hints is None
            or not hints.methods
            or len(results) < 2
        ):
            return results

        paper_index = next(
            (
                indexer
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(getattr(indexer, "get_document", None))
                and callable(
                    getattr(indexer, "get_method_neighbors", None)
                )
                and callable(getattr(indexer, "find_method_owners", None))
            ),
            None,
        )
        if paper_index is None:
            return results

        owner_top_k = min(
            len(results),
            self.stable_prefix_k
            if self.stable_prefix_k is not None
            else 10,
        )
        owner_candidates = results[:owner_top_k]
        owner_candidate_ids = {
            result.paper_id for result in owner_candidates
        }
        try:
            owner_records = self._find_method_owner_records(
                paper_index,
                hints.methods,
                owner_candidates,
                limit=max(
                    self.method_relation_max_results,
                    self.method_topic_seed_k,
                ),
            )
            owner_records = tuple(
                record
                for record in owner_records
                if record["paper_id"] in owner_candidate_ids
            )
            if not owner_records:
                return results

            retrieval_hints = self._without_method_hints(hints)
            topic_by_id: dict[str, dict] = {}
            for owner_rank, record in enumerate(
                owner_records[: self.method_topic_seed_k],
                start=1,
            ):
                owner_id = record["paper_id"]
                document = paper_index.get_document(owner_id)
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != owner_id
                    or not isinstance(document.text, str)
                ):
                    continue
                topic_query = document.text[
                    : self.method_topic_seed_chars
                ].strip()
                if not topic_query:
                    continue
                topic_results = self.retriever.retrieve(
                    topic_query,
                    self.method_topic_max_results,
                    hints=retrieval_hints,
                )[: self.method_topic_max_results]
                seen_topic_papers: set[str] = set()
                for topic_rank, result in enumerate(topic_results, start=1):
                    paper_id = getattr(result, "paper_id", None)
                    if (
                        not isinstance(paper_id, str)
                        or not paper_id
                        or paper_id in seen_topic_papers
                    ):
                        continue
                    seen_topic_papers.add(paper_id)
                    state = topic_by_id.setdefault(
                        paper_id,
                        {
                            "best_topic_rank": topic_rank,
                            "best_owner_rank": owner_rank,
                            "owner_papers": set(),
                        },
                    )
                    state["best_topic_rank"] = min(
                        state["best_topic_rank"],
                        topic_rank,
                    )
                    state["best_owner_rank"] = min(
                        state["best_owner_rank"],
                        owner_rank,
                    )
                    state["owner_papers"].add(owner_id)

            if not topic_by_id:
                return results

            result_ids = {result.paper_id for result in results}
            bridge_by_id: dict[str, dict] = {}
            for bridge_rank, bridge in enumerate(results[:-1], start=1):
                records = paper_index.get_method_neighbors(
                    bridge.paper_id,
                    limit=self.method_relation_max_results,
                )
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    paper_id = record.get("paper_id")
                    topic_state = topic_by_id.get(paper_id)
                    if (
                        not isinstance(paper_id, str)
                        or paper_id in result_ids
                        or topic_state is None
                        or topic_state["best_topic_rank"]
                        > self.method_bridge_topic_max_rank
                    ):
                        continue
                    state = bridge_by_id.setdefault(
                        paper_id,
                        {
                            "paper_id": paper_id,
                            "best_topic_rank": topic_state[
                                "best_topic_rank"
                            ],
                            "best_owner_rank": topic_state[
                                "best_owner_rank"
                            ],
                            "best_bridge_rank": bridge_rank,
                            "strength": 0,
                            "owner_papers": set(
                                topic_state["owner_papers"]
                            ),
                            "bridge_papers": set(),
                            "aliases": set(),
                        },
                    )
                    state["best_bridge_rank"] = min(
                        state["best_bridge_rank"],
                        bridge_rank,
                    )
                    strength = record.get("strength")
                    if isinstance(strength, int) and not isinstance(
                        strength,
                        bool,
                    ):
                        state["strength"] += max(strength, 0)
                    state["bridge_papers"].add(bridge.paper_id)
                    aliases = record.get("aliases")
                    if isinstance(aliases, (list, tuple)):
                        state["aliases"].update(
                            alias
                            for alias in aliases
                            if isinstance(alias, str)
                        )

            ranked = sorted(
                bridge_by_id.values(),
                key=lambda state: (
                    state["best_topic_rank"],
                    state["best_bridge_rank"],
                    -state["strength"],
                    state["best_owner_rank"],
                    state["paper_id"],
                ),
            )
            selected_state = None
            selected_document = None
            for state in ranked:
                document = paper_index.get_document(state["paper_id"])
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != state["paper_id"]
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
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
                "method_bridge_topic_rank": selected_state[
                    "best_topic_rank"
                ],
                "method_bridge_owner_papers": sorted(
                    selected_state["owner_papers"]
                ),
                "method_bridge_via_papers": sorted(
                    selected_state["bridge_papers"]
                ),
                "method_bridge_aliases": sorted(
                    selected_state["aliases"]
                ),
                "method_bridge_strength": selected_state["strength"],
                "method_bridge_replaced_paper_id": results[-1].paper_id,
                "method_bridge_is_new": True,
            }
        )
        replacement = RetrievalResult(
            chunk_id=selected_document.chunk_id,
            paper_id=selected_state["paper_id"],
            score=results[-1].score,
            text=selected_document.text[: self.seed_text_chars],
            chunk_type=selected_document.chunk_type,
            metadata=metadata,
            source="method_bridge_exploration",
        )
        return [*results[:-1], replacement]

    def _replace_last_with_dense_consensus(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Use multi-seed dense agreement to explore exactly one final slot."""

        if (
            self.paper_dense_consensus_seed_k <= 0
            or self.paper_embedding_index_dir is None
            or len(results) < self.paper_dense_consensus_min_support
        ):
            return results

        paper_index = next(
            (
                indexer
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(getattr(indexer, "get_document", None))
            ),
            None,
        )
        if paper_index is None:
            return results

        embedding_store = self._get_paper_embedding_store()
        if embedding_store is None:
            return results

        result_ids = {result.paper_id for result in results}
        seed_ids: list[str] = []
        for result in results[: self.paper_dense_consensus_seed_k]:
            paper_id = result.paper_id
            if (
                not isinstance(paper_id, str)
                or not paper_id
                or paper_id in seed_ids
            ):
                continue
            seed_ids.append(paper_id)
        if len(seed_ids) < self.paper_dense_consensus_min_support:
            return results

        consensus_by_id: dict[str, dict] = {}
        try:
            for seed_id in seed_ids:
                neighbors = tuple(
                    embedding_store.search_by_paper_id(
                        seed_id,
                        self.paper_dense_consensus_max_results,
                    )
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
                    similarity = neighbor.score
                    if (
                        isinstance(similarity, bool)
                        or not isinstance(similarity, Real)
                        or not math.isfinite(similarity)
                    ):
                        continue
                    similarity = float(similarity)
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
                    state["rrf_score"] += 1.0 / (
                        self.rrf_k + neighbor_rank
                    )

            ranked = sorted(
                (
                    state
                    for state in consensus_by_id.values()
                    if len(state["via_papers"])
                    >= self.paper_dense_consensus_min_support
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
                document = paper_index.get_document(state["paper_id"])
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != state["paper_id"]
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
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
                "paper_dense_consensus_rrf_score": selected_state[
                    "rrf_score"
                ],
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

    @classmethod
    def _append_result_tail(
        cls,
        candidates: list[RetrievalResult],
        selected_prefix: list[RetrievalResult],
        output_k: int,
    ) -> list[RetrievalResult]:
        """Append a lower-scored tail without changing the selected prefix."""

        selected = cls._align_scores_with_output_order(
            list(selected_prefix[:output_k])
        )
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
            return cls._align_scores_with_output_order(selected)

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

    @staticmethod
    def _align_scores_with_output_order(
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

    @staticmethod
    def _without_method_hints(
        hints: SearchHints | None,
    ) -> SearchHints | None:
        """Keep stable paper attributes while deferring methods to final ranking."""

        if hints is None:
            return None
        return SearchHints(
            venues=hints.venues,
            years=hints.years,
        )

    @staticmethod
    def _find_method_owner_records(
        provider,
        methods: tuple[str, ...],
        candidates: list[RetrievalResult],
        *,
        limit: int,
    ) -> tuple[dict, ...]:
        """Merge a prebuilt owner index with bounded live extraction."""

        try:
            indexed_records = tuple(
                provider.find_method_owners(methods, limit=limit)
            )
        except Exception:
            indexed_records = ()

        live_records: tuple = ()
        live_finder = getattr(
            provider,
            "find_method_owners_in_papers",
            None,
        )
        if not indexed_records and callable(live_finder):
            try:
                live_records = tuple(
                    live_finder(
                        methods,
                        (
                            candidate.paper_id
                            for candidate in candidates[:10]
                        ),
                        limit=limit,
                    )
                )
            except Exception:
                live_records = ()

        merged: dict[str, dict] = {}
        for record in (*indexed_records, *live_records):
            if not isinstance(record, dict):
                continue
            paper_id = record.get("paper_id")
            if not isinstance(paper_id, str) or not paper_id:
                continue
            state = merged.setdefault(
                paper_id,
                {
                    "paper_id": paper_id,
                    "aliases": set(),
                    "strength": 0,
                },
            )
            aliases = record.get("aliases")
            if isinstance(aliases, (list, tuple)):
                state["aliases"].update(
                    alias for alias in aliases if isinstance(alias, str)
                )
            strength = record.get("strength")
            if isinstance(strength, int) and not isinstance(strength, bool):
                state["strength"] = max(
                    state["strength"],
                    max(strength, 0),
                )

        records = [
            {
                "paper_id": state["paper_id"],
                "aliases": sorted(state["aliases"]),
                "strength": state["strength"],
            }
            for state in merged.values()
        ]
        records.sort(
            key=lambda record: (
                -record["strength"],
                record["paper_id"],
            )
        )
        return tuple(records[:limit])

    def _rerank_method_relations(
        self,
        candidates: list[RetrievalResult],
        hints: SearchHints | None,
        *,
        output_k: int,
    ) -> tuple[list[RetrievalResult], tuple[RetrievalResult, ...]]:
        """Rank method-linked papers and return normal results to preserve."""

        has_method_hints = hints is not None and bool(hints.methods)
        use_owners = self.method_owner_weight > 0 and has_method_hints
        use_relations = (
            self.method_relation_weight > 0 and has_method_hints
        )
        use_topics = self.method_topic_weight > 0 and has_method_hints
        if not candidates or not (use_owners or use_relations or use_topics):
            return candidates, ()

        provider = next(
            (
                indexer
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(getattr(indexer, "get_document", None))
                and callable(getattr(indexer, "get_method_neighbors", None))
                and callable(getattr(indexer, "find_method_owners", None))
            ),
            None,
        )
        if provider is None:
            return candidates, ()

        try:
            owner_records = (
                self._find_method_owner_records(
                    provider,
                    hints.methods,
                    candidates,
                    limit=max(
                        self.method_relation_max_results,
                        self.method_topic_seed_k if use_topics else 0,
                    ),
                )
                if use_owners or use_relations or use_topics
                else ()
            )
            original_by_id = {
                candidate.paper_id: candidate for candidate in candidates
            }
            normal_candidate_ids = {
                candidate.paper_id
                for candidate in candidates[: self.candidate_k]
            }
            protected_count = min(
                self.method_relation_protected_top_k,
                output_k,
                len(candidates),
            )
            protected_ids = {
                candidate.paper_id
                for candidate in candidates[:protected_count]
            }

            valid_owners: list[dict] = []
            for record in owner_records:
                if not isinstance(record, dict):
                    continue
                paper_id = record.get("paper_id")
                if (
                    isinstance(paper_id, str)
                    and paper_id in normal_candidate_ids
                ):
                    valid_owners.append(record)

            seed_ids: list[str] = []
            for record in valid_owners[: self.method_relation_seed_k]:
                paper_id = record["paper_id"]
                if paper_id not in seed_ids:
                    seed_ids.append(paper_id)

            owner_ids = {
                record["paper_id"] for record in valid_owners
            }
            topic_by_id: dict[str, dict] = {}
            topic_search_succeeded = False
            if use_topics:
                try:
                    topic_seed_ids: list[str] = []
                    for record in valid_owners[: self.method_topic_seed_k]:
                        paper_id = record["paper_id"]
                        if paper_id not in topic_seed_ids:
                            topic_seed_ids.append(paper_id)

                    retrieval_hints = self._without_method_hints(hints)

                    for seed_rank, seed_id in enumerate(
                        topic_seed_ids,
                        start=1,
                    ):
                        document = provider.get_document(seed_id)
                        if (
                            not hasattr(document, "paper_id")
                            or document.paper_id != seed_id
                            or not isinstance(document.text, str)
                        ):
                            continue
                        topic_query = document.text[
                            : self.method_topic_seed_chars
                        ].strip()
                        if not topic_query:
                            continue

                        topic_results = self.retriever.retrieve(
                            topic_query,
                            self.method_topic_max_results,
                            hints=retrieval_hints,
                        )[: self.method_topic_max_results]
                        seen_topic_papers: set[str] = set()
                        for result_rank, result in enumerate(
                            topic_results,
                            start=1,
                        ):
                            paper_id = getattr(result, "paper_id", None)
                            if (
                                not isinstance(paper_id, str)
                                or paper_id in seen_topic_papers
                            ):
                                continue
                            seen_topic_papers.add(paper_id)
                            if (
                                paper_id not in normal_candidate_ids
                                or paper_id in owner_ids
                                or paper_id in protected_ids
                            ):
                                continue
                            state = topic_by_id.setdefault(
                                paper_id,
                                {
                                    "paper_id": paper_id,
                                    "best_result_rank": result_rank,
                                    "best_seed_rank": seed_rank,
                                    "via_papers": set(),
                                },
                            )
                            state["best_result_rank"] = min(
                                state["best_result_rank"],
                                result_rank,
                            )
                            state["best_seed_rank"] = min(
                                state["best_seed_rank"],
                                seed_rank,
                            )
                            state["via_papers"].add(seed_id)
                    topic_search_succeeded = True
                except Exception:
                    # Topic expansion is optional. Keep exact owner and
                    # relation evidence when the additional search fails.
                    topic_by_id.clear()

            relation_by_id: dict[str, dict] = {}
            if use_relations and seed_ids:
                for seed_rank, seed_id in enumerate(seed_ids, start=1):
                    records = provider.get_method_neighbors(
                        seed_id,
                        limit=self.method_relation_max_results,
                    )
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        paper_id = record.get("paper_id")
                        if (
                            not isinstance(paper_id, str)
                            or paper_id == seed_id
                        ):
                            continue
                        state = relation_by_id.setdefault(
                            paper_id,
                            {
                                "paper_id": paper_id,
                                "aliases": set(),
                                "via_papers": set(),
                                "strength": 0,
                                "best_seed_rank": seed_rank,
                            },
                        )
                        aliases = record.get("aliases")
                        if isinstance(aliases, (list, tuple)):
                            state["aliases"].update(
                                alias
                                for alias in aliases
                                if isinstance(alias, str)
                            )
                        strength = record.get("strength")
                        if isinstance(strength, int) and not isinstance(
                            strength,
                            bool,
                        ):
                            state["strength"] += max(strength, 0)
                        state["via_papers"].add(seed_id)
                        state["best_seed_rank"] = min(
                            state["best_seed_rank"],
                            seed_rank,
                        )

            ranked_relation_records = sorted(
                relation_by_id.values(),
                key=lambda record: (
                    -record["strength"],
                    record["best_seed_rank"],
                    record["paper_id"],
                ),
            )[: self.method_relation_max_results]
            added_by_id: dict[str, RetrievalResult] = {}
            relation_records: list[dict] = []
            for record in ranked_relation_records:
                paper_id = record["paper_id"]
                if paper_id not in original_by_id:
                    if (
                        len(added_by_id)
                        >= self.method_relation_max_new_papers
                    ):
                        continue
                    document = provider.get_document(paper_id)
                    if (
                        not hasattr(document, "paper_id")
                        or document.paper_id != paper_id
                        or not isinstance(document.text, str)
                        or not document.text.strip()
                    ):
                        continue
                    metadata = (
                        dict(document.metadata)
                        if isinstance(document.metadata, dict)
                        else {}
                    )
                    added_by_id[paper_id] = RetrievalResult(
                        chunk_id=document.chunk_id,
                        paper_id=paper_id,
                        score=0.0,
                        text=document.text[: self.seed_text_chars],
                        chunk_type=document.chunk_type,
                        metadata=metadata,
                        source="method_relation",
                    )
                relation_records.append(record)
        except Exception:
            return candidates, ()

        topic_records = sorted(
            topic_by_id.values(),
            key=lambda record: (
                record["best_result_rank"],
                record["best_seed_rank"],
                record["paper_id"],
            ),
        )
        if not (
            (use_owners and valid_owners)
            or (use_relations and relation_records)
            or topic_records
        ):
            return candidates, ()

        owner_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(valid_owners, start=1)
        }
        owner_by_id = {
            record["paper_id"]: record for record in valid_owners
        }
        relation_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(relation_records, start=1)
        }
        relation_by_id = {
            record["paper_id"]: record for record in relation_records
        }
        topic_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(topic_records, start=1)
        }
        topic_record_by_id = {
            record["paper_id"]: record for record in topic_records
        }
        combined = [*candidates, *added_by_id.values()]
        scored: list[tuple[float, int, RetrievalResult]] = []
        for baseline_rank, candidate in enumerate(combined, start=1):
            owner_rank = owner_rank_by_id.get(candidate.paper_id)
            relation_rank = relation_rank_by_id.get(candidate.paper_id)
            topic_rank = topic_rank_by_id.get(candidate.paper_id)
            score = 1.0 / (self.rrf_k + baseline_rank)
            if owner_rank is not None:
                score += self.method_owner_weight / (
                    self.rrf_k + owner_rank
                )
            if relation_rank is not None:
                score += self.method_relation_weight / (
                    self.rrf_k + relation_rank
                )
            if topic_rank is not None:
                score += self.method_topic_weight / (
                    self.rrf_k + topic_rank
                )

            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "method_relation_baseline_rank": baseline_rank,
                    "method_owner_rank": owner_rank,
                    "method_relation_rank": relation_rank,
                }
            )
            if use_topics and topic_search_succeeded:
                metadata["method_topic_rank"] = topic_rank
            owner_record = owner_by_id.get(candidate.paper_id)
            if owner_record is not None:
                metadata["method_owner_aliases"] = list(
                    owner_record.get("aliases") or ()
                )
            relation_record = relation_by_id.get(candidate.paper_id)
            if relation_record is not None:
                metadata.update(
                    {
                        "method_relation_aliases": sorted(
                            relation_record["aliases"]
                        ),
                        "method_relation_via_papers": sorted(
                            relation_record["via_papers"]
                        ),
                        "method_relation_strength": relation_record[
                            "strength"
                        ],
                    }
                )
            topic_record = topic_record_by_id.get(candidate.paper_id)
            if topic_record is not None:
                metadata.update(
                    {
                        "method_topic_via_papers": sorted(
                            topic_record["via_papers"]
                        ),
                        "method_topic_search_rank": topic_record[
                            "best_result_rank"
                        ],
                    }
                )
            scored.append(
                (
                    score,
                    baseline_rank,
                    replace(
                        candidate,
                        score=score,
                        metadata=metadata,
                        source="method_relation_rrf",
                    ),
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2].paper_id,
            )
        )
        ranked = [result for _, _, result in scored]
        protected_candidates = tuple(candidates[:protected_count])
        if not protected_ids or output_k <= 0:
            return ranked, protected_candidates

        selected = ranked[:output_k]
        selected_ids = {candidate.paper_id for candidate in selected}
        ranked_by_id = {
            candidate.paper_id: candidate for candidate in ranked
        }
        owner_ids = set(owner_rank_by_id)
        missing = [
            ranked_by_id[candidate.paper_id]
            for candidate in candidates[:protected_count]
            if candidate.paper_id not in selected_ids
        ]
        for candidate in missing:
            replacement_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index].paper_id not in protected_ids
                    and selected[index].paper_id not in owner_ids
                ),
                None,
            )
            if replacement_index is None:
                replacement_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index].paper_id not in protected_ids
                    ),
                    None,
                )
            if replacement_index is None:
                break
            del selected[replacement_index]
            selected.append(candidate)

        selected_ids = {candidate.paper_id for candidate in selected}
        restored_ranking = [
            *selected,
            *(
                candidate
                for candidate in ranked
                if candidate.paper_id not in selected_ids
            ),
        ]
        return restored_ranking, protected_candidates

    def _rerank_paper_neighborhood(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Use explicit cross-paper mentions to rerank a bounded candidate pool."""

        if (
            self.paper_neighborhood_weight <= 0
            and self.paper_neighborhood_two_hop_weight <= 0
        ) or len(candidates) < 2:
            return candidates

        get_document = next(
            (
                getter
                for indexer in self.indexers
                if getattr(indexer, "name", None) == "paper_bm25"
                and callable(
                    getter := getattr(indexer, "get_document", None)
                )
            ),
            None,
        )
        if get_document is None:
            return candidates

        pool_size = min(self.candidate_k, len(candidates))
        selector = PaperNeighborhoodReranker(
            get_document=get_document,
            rrf_k=self.rrf_k,
            relation_weight=self.paper_neighborhood_weight,
            two_hop_weight=self.paper_neighborhood_two_hop_weight,
            max_hub_degree=self.paper_neighborhood_max_hub_degree,
        )
        reranked = selector.rerank(
            query,
            candidates[:pool_size],
            pool_size,
        )
        return [*reranked, *candidates[pool_size:]]

    def _restore_explicit_title_matches(
        self,
        query: str,
        candidate_pool: list[RetrievalResult],
        selected: list[RetrievalResult],
        output_k: int,
        *,
        reserved_paper_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Reserve bounded final slots for unambiguous titles named in the query."""

        if not self.protect_explicit_title_matches or output_k <= 0:
            return selected

        aliases: dict[str, list[tuple[int, RetrievalResult, str]]] = {}
        for rank, candidate in enumerate(candidate_pool, start=1):
            alias = _title_alias(candidate)
            if alias is None:
                continue
            aliases.setdefault(alias.casefold(), []).append(
                (rank, candidate, alias)
            )

        protected: list[tuple[int, int, RetrievalResult, str]] = []
        for matches in aliases.values():
            if len({candidate.paper_id for _, candidate, _ in matches}) != 1:
                continue
            rank, candidate, alias = matches[0]
            position = _standalone_alias_position(query, alias)
            if position is not None:
                protected.append((position, rank, candidate, alias))
        protected.sort(
            key=lambda item: (item[0], item[1], item[2].paper_id)
        )
        protected = protected[: self.max_protected_titles]
        if not protected:
            return selected

        protected_ids = {candidate.paper_id for _, _, candidate, _ in protected}
        protected_ids.update(reserved_paper_ids or ())
        selected_ids = {candidate.paper_id for candidate in selected}
        missing = [
            item for item in protected if item[2].paper_id not in selected_ids
        ]
        if not missing:
            return selected

        restored = list(selected)
        for _, rank, candidate, alias in missing:
            if len(restored) >= output_k:
                replacement_index = next(
                    (
                        index
                        for index in range(len(restored) - 1, -1, -1)
                        if restored[index].paper_id not in protected_ids
                    ),
                    None,
                )
                if replacement_index is None:
                    break
                del restored[replacement_index]

            metadata = dict(candidate.metadata)
            metadata["explicit_title_guard_alias"] = alias
            metadata["pre_title_guard_rank"] = rank
            restored.append(replace(candidate, metadata=metadata))

        return restored[:output_k]

    def _restore_method_protected_candidates(
        self,
        ranked_candidates: list[RetrievalResult],
        selected: list[RetrievalResult],
        protected_candidates: tuple[RetrievalResult, ...],
        output_k: int,
    ) -> list[RetrievalResult]:
        """Keep normal-search papers protected by method relation reranking."""

        if not protected_candidates or output_k <= 0:
            return selected[:output_k]

        ranked_by_id = {
            candidate.paper_id: candidate for candidate in ranked_candidates
        }
        protected_ids = {
            candidate.paper_id for candidate in protected_candidates
        }
        restored = list(selected[:output_k])
        selected_ids = {candidate.paper_id for candidate in restored}
        missing = [
            ranked_by_id.get(candidate.paper_id, candidate)
            for candidate in protected_candidates
            if candidate.paper_id not in selected_ids
        ]
        for candidate in missing:
            if len(restored) >= output_k:
                replacement_index = next(
                    (
                        index
                        for index in range(len(restored) - 1, -1, -1)
                        if restored[index].paper_id not in protected_ids
                    ),
                    None,
                )
                if replacement_index is None:
                    break
                removed = restored.pop(replacement_index)
                selected_ids.discard(removed.paper_id)
            restored.append(candidate)
            selected_ids.add(candidate.paper_id)

        return restored[:output_k]

    def _expanded_query(
        self,
        query: str,
        seed: RetrievalResult,
    ) -> str | None:
        metadata = seed.metadata if isinstance(seed.metadata, dict) else {}
        title = metadata.get("title")
        normalized_title = (
            " ".join(title.split())
            if isinstance(title, str) and title.strip()
            else ""
        )
        paper_context = self._paper_context(seed)
        if paper_context is not None:
            representative_text = " ".join(paper_context.split())[
                : self.seed_text_chars
            ].strip()
            include_title = bool(
                normalized_title
                and not self._starts_with_title(
                    representative_text,
                    normalized_title,
                )
            )
        else:
            return self._legacy_expanded_query(query, seed)

        return " ".join(
            part
            for part in (
                " ".join(query.split()),
                normalized_title if include_title else "",
                representative_text,
            )
            if part
        )

    def _legacy_expanded_query(
        self,
        query: str,
        seed: RetrievalResult,
    ) -> str | None:
        """Build the original title-plus-query-matched-chunk expansion."""

        metadata = seed.metadata if isinstance(seed.metadata, dict) else {}
        title = metadata.get("title")
        normalized_title = (
            " ".join(title.split())
            if isinstance(title, str) and title.strip()
            else ""
        )
        if not normalized_title or not isinstance(seed.text, str):
            return None

        representative_text = " ".join(seed.text.split())[
            : self.seed_text_chars
        ].strip()
        if not representative_text:
            return None
        return " ".join(
            (
                " ".join(query.split()),
                normalized_title,
                representative_text,
            )
        )

    @staticmethod
    def _paper_context(seed: RetrievalResult) -> str | None:
        """Return non-empty paper-level expansion text when available."""

        metadata = seed.metadata if isinstance(seed.metadata, dict) else {}
        paper_context = metadata.get("paper_rank_expansion_text")
        if isinstance(paper_context, str) and paper_context.strip():
            return paper_context
        return None

    @staticmethod
    def _starts_with_title(context: str, title: str) -> bool:
        """Return whether normalized paper context begins with its title."""

        folded_context = context.casefold()
        folded_title = title.casefold()
        return folded_context == folded_title or folded_context.startswith(
            f"{folded_title} "
        )

    @staticmethod
    def _unique_papers(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Keep the first result for each paper while preserving rank order."""

        unique: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in results:
            if result.paper_id in seen:
                continue
            seen.add(result.paper_id)
            unique.append(result)
        return unique

    def _fuse_by_paper(
        self,
        initial: list[RetrievalResult],
        expanded: list[RetrievalResult],
        local_expanded: list[RetrievalResult] | None = None,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        representatives: dict[str, RetrievalResult] = {}
        first_seen: dict[str, int] = {}
        ranks_by_run: list[dict[str, int]] = []

        weighted_runs: list[tuple[list[RetrievalResult], float]] = [
            (initial, 1.0),
            (expanded, 1.0),
        ]
        if local_expanded is not None:
            weighted_runs.append(
                (local_expanded, self.local_expansion_weight)
            )

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
