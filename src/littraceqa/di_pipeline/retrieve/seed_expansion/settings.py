"""Validated, responsibility-scoped settings for the seed expansion stages.

The public constructor keeps its flat keyword arguments so existing YAML files
keep working unchanged.  This module is the single boundary where those flat
values are grouped and checked, and it is the only place that knows which
parameter belongs to which stage.

``validate_settings`` runs the checks in the order the constructor used before
the settings objects existed.  That order is observable: when several
parameters are invalid at once it decides which error the caller sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

from littraceqa.di_pipeline.retrieve.seed_expansion.candidates import (
    MAX_OPEN_SET_SEEDS,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.exact_match import (
    MAX_EXACT_MATCH_PAPERS,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.structured_filter import (
    MAX_STRUCTURED_FILTER_PAPERS,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.dense_tail import (
    MAX_DENSE_TAIL_NEW_PAPERS,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.protection import (
    MAX_PROTECTED_TITLES,
)


@dataclass(frozen=True)
class CandidateSettings:
    """How wide the initial search is and how hints are turned into queries."""

    candidate_k: int
    seed_text_chars: int
    rrf_k: int
    literal_attribute_hints: bool
    literal_method_hints: bool


@dataclass(frozen=True)
class SupplementSettings:
    """The extra candidate source merged in before the reranker sees a pool.

    Disabled by default, so a configuration that omits it produces exactly the
    ranking it produced before the lane existed.
    """

    exact_method_search: bool
    exact_match_max_papers: int
    structured_filter: bool
    structured_filter_max_papers: int
    structured_filter_search_depth: int
    structured_filter_protected_prefix_k: int
    paper_metadata_path: str | None


@dataclass(frozen=True)
class OutputSettings:
    """How many papers leave the retriever and which of them stay fixed."""

    max_results: int
    stable_prefix_k: int | None
    rerank_pool_k: int
    rerank_final_candidates: bool
    final_rerank_document_chars: int
    final_rerank_protected_top_k: int
    protect_explicit_title_matches: bool
    max_protected_titles: int


@dataclass(frozen=True)
class OpenSetSettings:
    """The guarded exploration lane for enumeration questions."""

    open_set_seed_k: int
    open_set_min_support: int
    open_set_max_seed_rank: int
    open_set_slot_k: int


@dataclass(frozen=True)
class DenseSettings:
    """The paper-embedding tail lane seeded from the papers that own a method."""

    paper_embedding_index_dir: str | None
    method_dense_tail_weight: float
    method_dense_tail_seed_k: int
    method_dense_tail_max_results: int
    method_dense_tail_max_new_papers: int

    @property
    def has_embedding_index(self) -> bool:
        return self.paper_embedding_index_dir is not None


@dataclass(frozen=True)
class SeedExpansionSettings:
    """Every stage parameter, grouped by the stage that consumes it."""

    candidates: CandidateSettings
    supplement: SupplementSettings
    output: OutputSettings
    open_set: OpenSetSettings
    dense: DenseSettings

    def with_float_weights(self) -> SeedExpansionSettings:
        """Coerce every weight to ``float`` once validation has accepted it.

        Coercion has to happen after validation: ``float(True)`` is a valid
        number, so converting first would hide a boolean passed as a weight.
        """

        return replace(
            self,
            dense=replace(
                self.dense,
                method_dense_tail_weight=float(
                    self.dense.method_dense_tail_weight
                ),
            ),
        )


def _require_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _require_positive_integer(name: str, value: object) -> None:
    _require_integer(name, value)
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be positive")


def _require_non_negative_integer(name: str, value: object) -> None:
    _require_integer(name, value)
    if value < 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be non-negative")


def _require_boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _require_non_negative_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_candidate_pool(candidates: CandidateSettings) -> None:
    if candidates.candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if candidates.seed_text_chars <= 0:
        raise ValueError("seed_text_chars must be positive")
    if candidates.rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")


def _validate_output_shape(output: OutputSettings) -> None:
    if output.max_results <= 0:
        raise ValueError("max_results must be positive")
    if output.stable_prefix_k is not None:
        if isinstance(output.stable_prefix_k, bool) or not isinstance(
            output.stable_prefix_k, int
        ):
            raise TypeError("stable_prefix_k must be an integer or None")
        if output.stable_prefix_k <= 0:
            raise ValueError("stable_prefix_k must be positive")
    if output.rerank_pool_k <= 0:
        raise ValueError("rerank_pool_k must be positive")
    _require_boolean("rerank_final_candidates", output.rerank_final_candidates)
    _require_integer(
        "final_rerank_document_chars",
        output.final_rerank_document_chars,
    )
    if output.final_rerank_document_chars <= 0:
        raise ValueError("final_rerank_document_chars must be positive")
    _require_non_negative_integer(
        "final_rerank_protected_top_k",
        output.final_rerank_protected_top_k,
    )
    _require_boolean(
        "protect_explicit_title_matches",
        output.protect_explicit_title_matches,
    )
    _require_integer("max_protected_titles", output.max_protected_titles)
    if not 1 <= output.max_protected_titles <= MAX_PROTECTED_TITLES:
        raise ValueError(
            f"max_protected_titles must be between 1 and "
            f"{MAX_PROTECTED_TITLES}"
        )


def _validate_candidate_signals(candidates: CandidateSettings) -> None:
    _require_boolean(
        "literal_attribute_hints",
        candidates.literal_attribute_hints,
    )
    _require_boolean("literal_method_hints", candidates.literal_method_hints)


def _validate_supplement(supplement: SupplementSettings) -> None:
    _require_boolean("exact_method_search", supplement.exact_method_search)
    _require_non_negative_integer(
        "exact_match_max_papers",
        supplement.exact_match_max_papers,
    )
    if supplement.exact_match_max_papers > MAX_EXACT_MATCH_PAPERS:
        raise ValueError(
            f"exact_match_max_papers must not exceed {MAX_EXACT_MATCH_PAPERS}"
        )
    _require_boolean("structured_filter", supplement.structured_filter)
    _require_non_negative_integer(
        "structured_filter_max_papers",
        supplement.structured_filter_max_papers,
    )
    if supplement.structured_filter_max_papers > MAX_STRUCTURED_FILTER_PAPERS:
        raise ValueError(
            "structured_filter_max_papers must not exceed "
            f"{MAX_STRUCTURED_FILTER_PAPERS}"
        )
    _require_positive_integer(
        "structured_filter_search_depth",
        supplement.structured_filter_search_depth,
    )
    _require_non_negative_integer(
        "structured_filter_protected_prefix_k",
        supplement.structured_filter_protected_prefix_k,
    )
    if supplement.paper_metadata_path is not None:
        if not isinstance(supplement.paper_metadata_path, str):
            raise TypeError("paper_metadata_path must be a string or None")
        if not supplement.paper_metadata_path.strip():
            raise ValueError("paper_metadata_path must not be empty")
    if supplement.structured_filter and not supplement.paper_metadata_path:
        raise ValueError("structured_filter requires paper_metadata_path")
    if supplement.exact_method_search and not supplement.paper_metadata_path:
        raise ValueError("exact_method_search requires paper_metadata_path")


def _validate_open_set(
    open_set: OpenSetSettings,
    candidates: CandidateSettings,
    output: OutputSettings,
) -> None:
    for name, value in (
        ("open_set_seed_k", open_set.open_set_seed_k),
        ("open_set_min_support", open_set.open_set_min_support),
        ("open_set_max_seed_rank", open_set.open_set_max_seed_rank),
        ("open_set_slot_k", open_set.open_set_slot_k),
    ):
        _require_positive_integer(name, value)
    if open_set.open_set_seed_k > MAX_OPEN_SET_SEEDS:
        raise ValueError(f"open_set_seed_k must not exceed {MAX_OPEN_SET_SEEDS}")
    if open_set.open_set_min_support < 2:
        raise ValueError("open_set_min_support must be at least 2")
    if (
        open_set.open_set_seed_k > 1
        and open_set.open_set_min_support > open_set.open_set_seed_k - 1
    ):
        raise ValueError(
            "open_set_min_support must not exceed the number of "
            "additional seed searches"
        )
    if open_set.open_set_max_seed_rank > candidates.candidate_k:
        raise ValueError("open_set_max_seed_rank must not exceed candidate_k")
    if (
        open_set.open_set_seed_k > 1
        and open_set.open_set_slot_k > output.max_results
    ):
        raise ValueError("open_set_slot_k must not exceed max_results")


def _validate_dense_tail(dense: DenseSettings) -> None:
    _require_non_negative_number(
        "method_dense_tail_weight",
        dense.method_dense_tail_weight,
    )
    for name, value in (
        ("method_dense_tail_seed_k", dense.method_dense_tail_seed_k),
        (
            "method_dense_tail_max_results",
            dense.method_dense_tail_max_results,
        ),
    ):
        _require_positive_integer(name, value)
    _require_non_negative_integer(
        "method_dense_tail_max_new_papers",
        dense.method_dense_tail_max_new_papers,
    )
    if dense.method_dense_tail_max_new_papers > MAX_DENSE_TAIL_NEW_PAPERS:
        raise ValueError(
            "method_dense_tail_max_new_papers must not exceed "
            f"{MAX_DENSE_TAIL_NEW_PAPERS}"
        )


def _validate_paper_embedding_index(dense: DenseSettings) -> None:
    if dense.paper_embedding_index_dir is None:
        return
    if not isinstance(dense.paper_embedding_index_dir, str):
        raise TypeError("paper_embedding_index_dir must be a string or None")
    if not dense.paper_embedding_index_dir.strip():
        raise ValueError("paper_embedding_index_dir must not be empty")


def _validate_wrapping(
    retriever: object,
    reranker: object | None,
    output: OutputSettings,
) -> None:
    if getattr(retriever, "reranker", None) is not None:
        raise ValueError(
            "seed expansion cannot wrap a retriever with a reranker because "
            "that would run the reranker twice"
        )
    if output.rerank_final_candidates and reranker is None:
        raise ValueError("rerank_final_candidates requires an enabled reranker")


def validate_settings(
    settings: SeedExpansionSettings,
    *,
    retriever: object,
    reranker: object | None,
) -> None:
    """Check every parameter in the order the flat constructor used to."""

    _validate_candidate_pool(settings.candidates)
    _validate_output_shape(settings.output)
    _validate_candidate_signals(settings.candidates)
    _validate_supplement(settings.supplement)
    _validate_open_set(settings.open_set, settings.candidates, settings.output)
    _validate_dense_tail(settings.dense)
    _validate_paper_embedding_index(settings.dense)
    _validate_wrapping(retriever, reranker, settings.output)
