"""Per-query provenance: which stage put each paper where, and why.

Metadata keys are copied through a fixed allow-list so a new experimental key
cannot silently widen the output schema.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from littraceqa.di_pipeline.evaluation.gold import gold_paper_ids
from littraceqa.di_pipeline.evaluation.metrics import _scenario_for_gold


_METHOD_RANKING_INT_FIELDS = (
    "qwen3_rank",
    "final_rerank_pre_protection_rank",
    "final_rerank_protected_top_k",
    "method_dense_tail_baseline_rank",
    "method_dense_tail_rank",
    "method_dense_tail_best_neighbor_rank",
    "open_set_expansion_best_rank",
    "open_set_expansion_original_rank",
    "open_set_expansion_run_count",
    "open_set_expansion_slot_k",
    "open_set_expansion_support",
    "output_order_rank",
)
_METHOD_RANKING_LIST_FIELDS = (
    "attribute_matches",
    "method_dense_tail_via_papers",
    "open_set_expansion_via_papers",
)
_METHOD_RANKING_FLOAT_FIELDS = (
    "qwen3_score",
    "rank_fusion_base_weight",
    "rank_fusion_k",
    "final_rerank_pre_protection_score",
    "method_dense_tail_best_similarity",
    "method_dense_tail_rrf_score",
    "pre_output_order_score",
)
_METHOD_RANKING_BOOL_FIELDS = (
    "final_rerank_candidate_set_preserved",
    "final_rerank_prefix_protected",
    "method_dense_tail_is_new",
    "open_set_expansion_attempted",
    "open_set_expansion_selected",
)
_METHOD_RANKING_STR_FIELDS = (
    "final_rerank_status",
    "final_rerank_error_type",
    "open_set_expansion_selected_paper_id",
)


def _method_ranking_metadata(
    metadata: Any,
) -> dict[str, bool | float | int | str | list[str] | None]:
    """Copy only typed, JSON-safe method-ranking provenance fields."""
    if not isinstance(metadata, dict):
        return {}

    copied: dict[str, bool | float | int | str | list[str] | None] = {}
    for key in _METHOD_RANKING_INT_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            value
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )

    for key in _METHOD_RANKING_LIST_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            list(value)
            if isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            else []
        )
    for key in _METHOD_RANKING_FLOAT_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )
    for key in _METHOD_RANKING_BOOL_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = value if isinstance(value, bool) else None
    for key in _METHOD_RANKING_STR_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = value if isinstance(value, str) and value else None
    return copied


def paper_ranking_details(
    results: Sequence[Any],
    max_papers: int | None = None,
) -> list[dict]:
    """Return JSON-safe paper scores and reranker provenance without document text."""
    best_by_paper: dict[str, tuple[float, int, Any]] = {}
    for result_index, result in enumerate(results):
        paper_id = str(result.paper_id)
        score = float(result.score)
        previous = best_by_paper.get(paper_id)
        if previous is None or score > previous[0]:
            best_by_paper[paper_id] = (score, result_index, result)

    ranked = sorted(
        best_by_paper.values(),
        key=lambda item: (-item[0], item[1]),
    )
    if max_papers is not None:
        ranked = ranked[:max_papers]

    details = []
    for score, _, result in ranked:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        pre_rerank_rank = metadata.get("pre_rerank_rank")
        pre_rerank_score = metadata.get("pre_rerank_score")
        detail = {
            "paper_id": str(result.paper_id),
            "score": score,
            "source": str(result.source),
            "representative_chunk_id": str(result.chunk_id),
            "chunk_type": str(result.chunk_type),
            "pre_rerank_rank": (
                pre_rerank_rank
                if isinstance(pre_rerank_rank, int)
                and not isinstance(pre_rerank_rank, bool)
                else None
            ),
            "pre_rerank_score": (
                float(pre_rerank_score)
                if isinstance(pre_rerank_score, (int, float))
                and not isinstance(pre_rerank_score, bool)
                else None
            ),
        }
        detail.update(_method_ranking_metadata(metadata))
        details.append(detail)
    return details


def pre_rerank_papers(results: Sequence[Any]) -> list[str] | None:
    """Recover the full paper order recorded by a provenance-aware reranker."""
    if not results:
        return None

    for result in results:
        recorded = (result.metadata or {}).get("pre_rerank_candidate_papers")
        if recorded is None:
            continue
        if not isinstance(recorded, list) or not recorded:
            return None
        normalized = [str(paper_id).strip() for paper_id in recorded]
        if any(not paper_id for paper_id in normalized):
            return None
        if len(set(normalized)) != len(normalized):
            return None
        return normalized

    first_rank_by_paper: dict[str, int] = {}
    for result in results:
        rank = (result.metadata or {}).get("pre_rerank_rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank <= 0
        ):
            return None
        paper_id = str(result.paper_id)
        first_rank_by_paper[paper_id] = min(
            rank,
            first_rank_by_paper.get(paper_id, rank),
        )
    return sorted(first_rank_by_paper, key=first_rank_by_paper.__getitem__)


def query_diagnostic(
    record: dict,
    ranked_papers: list[str],
    ks: Sequence[int],
    *,
    pre_rerank_ranked_papers: list[str] | None = None,
    ranking_details: list[dict] | None = None,
    elapsed_seconds: float | None = None,
) -> dict:
    """Describe exact gold ranks for one query without using task labels."""
    gold = gold_paper_ids(record)
    first_ranks: dict[str, int] = {}
    for rank, paper_id in enumerate(ranked_papers, start=1):
        first_ranks.setdefault(paper_id, rank)
    diagnostic = {
        "query_id": str(record.get("query_id") or ""),
        "question": record.get("question"),
        "gold_count": len(gold),
        "scenario": _scenario_for_gold(gold),
        "gold_papers": sorted(gold),
        "gold_ranks": {
            paper_id: first_ranks.get(paper_id) for paper_id in sorted(gold)
        },
        "ranked_papers": ranked_papers,
        "all_gold_at_k": {
            str(k): gold.issubset(set(ranked_papers[:k])) for k in ks
        },
    }
    if pre_rerank_ranked_papers is not None:
        pre_rerank_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(pre_rerank_ranked_papers, start=1)
        }
        diagnostic.update(
            {
                "pre_rerank_papers": pre_rerank_ranked_papers,
                "pre_rerank_gold_ranks": {
                    paper_id: pre_rerank_ranks.get(paper_id)
                    for paper_id in sorted(gold)
                },
                "pre_rerank_all_gold_at_k": {
                    str(k): gold.issubset(set(pre_rerank_ranked_papers[:k]))
                    for k in ks
                },
            }
        )
    if ranking_details is not None:
        diagnostic["ranking_details"] = ranking_details
    if elapsed_seconds is not None:
        diagnostic["elapsed_seconds"] = elapsed_seconds
    return diagnostic
