"""Prepare paper-selection errors for offline human or model review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.select.selector import (
    PaperSelection,
    ordered_paper_ids,
)


REVIEW_INSTRUCTIONS = (
    "Decide whether each missed gold paper is required to answer the question, "
    "using only the supplied question, answer, and evidence.",
    "If it is required, distinguish a pool/ranking failure from a paper-selection "
    "failure using final_rank, pre_rerank_rank, candidate_status, "
    "selection_failure, and provenance.",
    "Look for reusable causes such as terminology mismatch, method aliases, "
    "metadata constraints, reranker demotion, or an incorrect paper count.",
    "For count errors, separate open or implicit enumeration from explicitly "
    "named targets, and distinguish a method's owner paper from a different "
    "paper that reports the requested value.",
    "Also decide whether each false-positive paper is useful support or an "
    "unnecessary submission that lowers paper precision.",
    "Propose general fixes. Do not add query IDs, gold paper IDs, or validation-only "
    "fields to production rules.",
)

FIELD_GUIDE = {
    "analysis_cutoff": (
        "A diagnostic rank boundary only; it does not truncate selector input."
    ),
    "candidate_status": (
        "Whether a missed gold paper is inside the saved ranking and the "
        "diagnostic cutoff."
    ),
    "selection_failure": (
        "The primary actionable stage: candidate generation, selected-paper "
        "count, ordering within the saved ranking, or unnecessary submission."
    ),
    "cardinality_misses_by_selection_reason": (
        "Missed papers attributed to selected-paper count, grouped by the "
        "selector rule that chose that count."
    ),
}

_FALSE_POSITIVE_FAILURE = "selector_precision_or_cardinality_over"


@dataclass(frozen=True)
class ReviewCase:
    record: dict[str, Any]
    retrieval: dict[str, Any]
    ranked: tuple[str, ...]
    gold_ids: tuple[str, ...]
    selection: PaperSelection
    missed_ids: tuple[str, ...]
    false_positive_ids: tuple[str, ...]


def _ordered_gold_paper_ids(record: Mapping[str, Any]) -> list[str]:
    """Return gold paper IDs in their source order."""

    return ordered_paper_ids(
        item.get("paper_id") if isinstance(item, dict) else item
        for item in (record.get("gold_papers") or [])
    )


def index_retrieval_entries(payload: object) -> dict[str, dict[str, Any]]:
    """Index query diagnostics from an evaluation output."""

    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("retrieval output must contain a queries list")
    entries: dict[str, dict[str, Any]] = {}
    for item in payload["queries"]:
        if not isinstance(item, dict) or not item.get("query_id"):
            raise ValueError("retrieval output contains an invalid query")
        entries[str(item["query_id"])] = item
    return entries


def collect_review_cases(
    gold_records: Sequence[dict[str, Any]],
    retrieval_entries: Mapping[str, dict[str, Any]],
    selector: Any,
    *,
    top_candidates: int,
    selection_refiner: Any | None = None,
) -> tuple[list[ReviewCase], set[str]]:
    """Apply a selector and retain queries whose submitted set is incorrect."""

    cases: list[ReviewCase] = []
    wanted_ids: set[str] = set()
    for record in gold_records:
        query_id = str(record.get("query_id") or "")
        entry = retrieval_entries.get(query_id)
        if entry is None:
            raise ValueError(f"no retrieval result for query {query_id!r}")
        ranked = tuple(ordered_paper_ids(entry.get("ranked_papers") or []))
        gold_ids = tuple(_ordered_gold_paper_ids(record))
        selection = selector.select(record.get("question", ""), ranked)
        if selection_refiner is not None:
            selection = selection_refiner.refine(
                Query.from_dict(record), ranked, selection
            )
        missed = tuple(
            paper_id for paper_id in gold_ids if paper_id not in selection.paper_ids
        )
        false_positives = tuple(
            paper_id for paper_id in selection.paper_ids if paper_id not in gold_ids
        )
        if not missed and not false_positives:
            continue
        cases.append(
            ReviewCase(
                record,
                entry,
                ranked,
                gold_ids,
                selection,
                missed,
                false_positives,
            )
        )
        wanted_ids.update(gold_ids)
        wanted_ids.update(selection.paper_ids)
        wanted_ids.update(ranked[:top_candidates])
    return cases, wanted_ids


def _rank_map(paper_ids: Sequence[str]) -> dict[str, int]:
    return {paper_id: rank for rank, paper_id in enumerate(paper_ids, start=1)}


def _paper_summary(
    paper_id: str,
    metadata: Mapping[str, dict[str, Any]],
    final_ranks: Mapping[str, int],
    pre_ranks: Mapping[str, int],
    details: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        **metadata.get(paper_id, {"metadata_missing": True}),
        "final_rank": final_ranks.get(paper_id),
        "pre_rerank_rank": pre_ranks.get(paper_id),
        "provenance": details.get(paper_id),
    }


def _candidate_status(
    paper_id: str,
    final_ranks: Mapping[str, int],
    *,
    analysis_cutoff: int,
) -> str:
    rank = final_ranks.get(paper_id)
    if rank is None:
        return "outside_saved_pool"
    if rank > analysis_cutoff:
        return "below_analysis_cutoff"
    return "within_analysis_cutoff"


def _selection_failures(
    case: ReviewCase,
    final_ranks: Mapping[str, int],
) -> dict[str, str]:
    """Assign each miss to the earliest stage that could have prevented it."""

    failures = {
        paper_id: "candidate_generation"
        for paper_id in case.missed_ids
        if paper_id not in final_ranks
    }
    ranked_misses = sorted(
        (paper_id for paper_id in case.missed_ids if paper_id in final_ranks),
        key=final_ranks.__getitem__,
    )
    selected_gold = len(set(case.selection.paper_ids) & set(case.gold_ids))
    recoverable_gold = min(
        case.selection.expected_count,
        selected_gold + len(ranked_misses),
    )
    ranking_misses = max(0, recoverable_gold - selected_gold)
    for paper_id in ranked_misses[:ranking_misses]:
        failures[paper_id] = "selector_ranking"
    for paper_id in ranked_misses[ranking_misses:]:
        failures[paper_id] = "selector_cardinality"
    return failures


def _evidence_by_paper(case: ReviewCase) -> dict[str, list[dict[str, Any]]]:
    evidence_by_paper: dict[str, list[dict[str, Any]]] = {}
    gold = set(case.gold_ids)
    for evidence in case.record.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        paper_id = str(evidence.get("paper_id") or "")
        if paper_id in gold:
            evidence_by_paper.setdefault(paper_id, []).append(evidence)
    return evidence_by_paper


def _render_gold_papers(
    case: ReviewCase,
    metadata: Mapping[str, dict[str, Any]],
    final_ranks: Mapping[str, int],
    pre_ranks: Mapping[str, int],
    details: Mapping[str, dict[str, Any]],
    *,
    analysis_cutoff: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    selected = set(case.selection.paper_ids)
    evidence = _evidence_by_paper(case)
    selection_failures = _selection_failures(case, final_ranks)
    candidate_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    papers = []

    for paper_id in case.gold_ids:
        paper = _paper_summary(paper_id, metadata, final_ranks, pre_ranks, details)
        paper["selected"] = paper_id in selected
        paper["evidence"] = evidence.get(paper_id, [])
        if paper_id not in selected:
            candidate_status = _candidate_status(
                paper_id,
                final_ranks,
                analysis_cutoff=analysis_cutoff,
            )
            selection_failure = selection_failures[paper_id]
            paper["candidate_status"] = candidate_status
            paper["selection_failure"] = selection_failure
            candidate_counts[candidate_status] += 1
            failure_counts[selection_failure] += 1
        papers.append(paper)

    return papers, candidate_counts, failure_counts


def _render_false_positive_papers(
    case: ReviewCase,
    metadata: Mapping[str, dict[str, Any]],
    final_ranks: Mapping[str, int],
    pre_ranks: Mapping[str, int],
    details: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    papers = []
    for paper_id in case.false_positive_ids:
        paper = _paper_summary(paper_id, metadata, final_ranks, pre_ranks, details)
        paper["selection_failure"] = _FALSE_POSITIVE_FAILURE
        papers.append(paper)
    return papers


def _render_top_candidates(
    case: ReviewCase,
    metadata: Mapping[str, dict[str, Any]],
    final_ranks: Mapping[str, int],
    pre_ranks: Mapping[str, int],
    details: Mapping[str, dict[str, Any]],
    *,
    top_candidates: int,
) -> list[dict[str, Any]]:
    selected = set(case.selection.paper_ids)
    gold = set(case.gold_ids)
    papers = []
    for paper_id in case.ranked[:top_candidates]:
        paper = _paper_summary(paper_id, metadata, final_ranks, pre_ranks, details)
        paper["is_gold"] = paper_id in gold
        paper["is_selected"] = paper_id in selected
        papers.append(paper)
    return papers


def _render_case(
    case: ReviewCase,
    metadata: Mapping[str, dict[str, Any]],
    *,
    analysis_cutoff: int,
    top_candidates: int,
) -> tuple[dict[str, Any], Counter[str], Counter[str], int]:
    final_ranks = _rank_map(case.ranked)
    pre_ranks = _rank_map(
        ordered_paper_ids(case.retrieval.get("pre_rerank_papers") or [])
    )
    details = {
        str(item["paper_id"]): item
        for item in (case.retrieval.get("ranking_details") or [])
        if isinstance(item, dict) and item.get("paper_id")
    }
    gold_papers, candidate_counts, failure_counts = _render_gold_papers(
        case,
        metadata,
        final_ranks,
        pre_ranks,
        details,
        analysis_cutoff=analysis_cutoff,
    )
    false_positives = _render_false_positive_papers(
        case,
        metadata,
        final_ranks,
        pre_ranks,
        details,
    )
    top = _render_top_candidates(
        case,
        metadata,
        final_ranks,
        pre_ranks,
        details,
        top_candidates=top_candidates,
    )

    rendered = {
        "query_id": str(case.record.get("query_id") or ""),
        "question": case.record.get("question"),
        "answer_types": case.record.get("answer_types") or [],
        "table_schema": case.record.get("table_schema"),
        "multiple_choice_options": case.record.get("multiple_choice_options"),
        "reference_answer": case.record.get("answer"),
        "selection": {
            "paper_ids": list(case.selection.paper_ids),
            "selected_count": len(case.selection.paper_ids),
            "gold_count": len(case.gold_ids),
            "expected_count": case.selection.expected_count,
            "reason": case.selection.reason,
        },
        "gold_papers": gold_papers,
        "missed_gold_paper_ids": list(case.missed_ids),
        "false_positive_papers": false_positives,
        "top_candidates": top,
    }
    return rendered, candidate_counts, failure_counts, len(false_positives)


def build_report(
    cases: Sequence[ReviewCase],
    metadata: Mapping[str, dict[str, Any]],
    *,
    analysis_cutoff: int,
    top_candidates: int,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Render review cases and aggregate selection-error categories."""

    queries = []
    candidate_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    cardinality_misses_by_reason: Counter[str] = Counter()
    false_positive_count = 0
    queries_with_missed_gold = 0
    queries_with_false_positives = 0
    for case in cases:
        rendered, case_candidates, case_failures, false_positives = _render_case(
            case,
            metadata,
            analysis_cutoff=analysis_cutoff,
            top_candidates=top_candidates,
        )
        queries.append(rendered)
        candidate_counts.update(case_candidates)
        failure_counts.update(case_failures)
        cardinality_misses = case_failures.get("selector_cardinality", 0)
        if cardinality_misses:
            cardinality_misses_by_reason[case.selection.reason] += cardinality_misses
        false_positive_count += false_positives
        queries_with_missed_gold += bool(case.missed_ids)
        queries_with_false_positives += bool(case.false_positive_ids)
    return {
        "schema_version": 1,
        "review_instructions": list(REVIEW_INSTRUCTIONS),
        "field_guide": dict(FIELD_GUIDE),
        "sources": dict(sources),
        "summary": {
            "queries_with_selection_errors": len(queries),
            "queries_with_missed_gold": queries_with_missed_gold,
            "queries_with_false_positives": queries_with_false_positives,
            "missed_gold_papers": sum(failure_counts.values()),
            "false_positive_papers": false_positive_count,
            "false_positive_failures": (
                {_FALSE_POSITIVE_FAILURE: false_positive_count}
                if false_positive_count
                else {}
            ),
            "candidate_statuses": dict(sorted(candidate_counts.items())),
            "selection_failures": dict(sorted(failure_counts.items())),
            "cardinality_misses_by_selection_reason": dict(
                sorted(cardinality_misses_by_reason.items())
            ),
            "analysis_cutoff": analysis_cutoff,
            "top_candidates_per_query": top_candidates,
        },
        "queries": queries,
    }
