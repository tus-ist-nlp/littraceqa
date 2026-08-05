"""Post-inference error analysis for the LitTraceQA reading stage.

This module is deliberately separate from inference.  It is the only reading
workflow component that accepts validation gold data, so gold answers and
evidence cannot accidentally enter an Azure OpenAI prompt.  The analyzer joins
three already-produced artifacts by ``query_id``:

* the validation gold JSONL;
* the sanitized, ranked candidate-paper sidecar; and
* the two-stage reader trace JSONL.

The output is intended for inspecting all 55 validation questions one by one,
not for reproducing the organizer's official score exactly.  In particular,
``required_gold_paper_ids`` is derived from gold evidence and is therefore more
useful for diagnosing reading than a multi-paper ``gold_papers`` list that can
also contain comparison-set or multiple-choice distractor papers.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from littraceqa.reading_error_annotations import (
    ANSWER_REQUIRED_PAPER_IDS,
    KNOWN_DATASET_ISSUES,
)

ERROR_CATEGORIES = (
    "candidate_missing",
    "relevance_filter_false_negative",
    "relevance_filter_overselection",
    "evidence_chunk_selection_error",
    "modality_read_error",
    "multi_paper_integration_error",
    "answer_extraction_or_reasoning_error",
    "answer_format_error",
    "multiple_choice_protocol_blocker",
    "dataset_inconsistency",
)

_NON_TEXT_MODALITIES = frozenset(
    {"table", "figure", "citation_context", "equation_algorithm"}
)
_MC_LETTER = re.compile(r"[A-D]")
_SAFE_QUERY_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and fail with a line-specific error."""

    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def analyze_reading_run(
    gold_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    trace_records: Sequence[dict[str, Any]],
    *,
    known_issues: Mapping[str, Sequence[str]] | None = None,
    required_paper_overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Analyze a completed reader run and return deterministic JSON data.

    Gold is accepted only here, after inference has completed.  Candidate and
    trace records are joined by ID instead of position so split or resumed runs
    remain safe to inspect.
    """

    gold_by_id = _unique_by_query_id(gold_records, "gold")
    candidates_by_id = _unique_by_query_id(candidate_records, "candidate")
    traces_by_id = _unique_by_query_id(trace_records, "trace")
    merged_issues: dict[str, tuple[str, ...]] = dict(KNOWN_DATASET_ISSUES)
    if known_issues:
        for query_id, issues in known_issues.items():
            merged_issues[str(query_id)] = tuple(str(issue) for issue in issues)
    merged_required_papers: dict[str, tuple[str, ...]] = dict(
        ANSWER_REQUIRED_PAPER_IDS
    )
    if required_paper_overrides:
        for query_id, paper_ids in required_paper_overrides.items():
            merged_required_papers[str(query_id)] = tuple(
                str(paper_id) for paper_id in paper_ids
            )

    query_details: list[dict[str, Any]] = []
    for query_id in sorted(gold_by_id):
        query_details.append(
            analyze_query(
                gold_by_id[query_id],
                candidates_by_id.get(query_id),
                traces_by_id.get(query_id),
                known_issues=merged_issues,
                required_paper_overrides=merged_required_papers,
            )
        )

    category_counts = Counter(
        category
        for detail in query_details
        for category in detail["error_categories"]
    )
    metrics = {
        "official_candidate_recall_macro": _mean(
            detail["metrics"]["official_candidate_recall"]
            for detail in query_details
        ),
        "required_candidate_recall_macro": _mean(
            detail["metrics"]["required_candidate_recall"]
            for detail in query_details
        ),
        "selected_paper_precision_macro": _mean(
            detail["metrics"]["selected_paper_precision"]
            for detail in query_details
        ),
        "selected_paper_recall_macro": _mean(
            detail["metrics"]["selected_paper_recall"]
            for detail in query_details
        ),
        "selected_paper_f1_macro": _mean(
            detail["metrics"]["selected_paper_f1"]
            for detail in query_details
        ),
        "evidence_precision_macro": _mean(
            detail["metrics"]["evidence_precision"] for detail in query_details
        ),
        "evidence_recall_macro": _mean(
            detail["metrics"]["evidence_recall"] for detail in query_details
        ),
        "evidence_f1_macro": _mean(
            detail["metrics"]["evidence_f1"] for detail in query_details
        ),
        "official_evidence_f1_macro": _mean(
            detail["metrics"]["official_evidence_f1"]
            for detail in query_details
        ),
        "answer_exact_macro": _mean(
            float(detail["metrics"]["answer_exact"])
            for detail in query_details
        ),
        "reading_answer_exact_macro": _mean(
            float(detail["metrics"]["reading_answer_exact"])
            for detail in query_details
        ),
    }
    missing_candidates = sorted(set(gold_by_id) - set(candidates_by_id))
    missing_traces = sorted(set(gold_by_id) - set(traces_by_id))
    extra_candidates = sorted(set(candidates_by_id) - set(gold_by_id))
    extra_traces = sorted(set(traces_by_id) - set(gold_by_id))
    return {
        "schema_version": 1,
        "summary": {
            "total_queries": len(query_details),
            "queries_with_errors": sum(
                bool(detail["error_categories"]) for detail in query_details
            ),
            "missing_candidate_record_count": len(missing_candidates),
            "missing_trace_count": len(missing_traces),
            "extra_candidate_record_count": len(extra_candidates),
            "extra_trace_count": len(extra_traces),
            "missing_candidate_records": missing_candidates,
            "missing_traces": missing_traces,
            "extra_candidate_records": extra_candidates,
            "extra_traces": extra_traces,
            "category_counts": {
                category: category_counts.get(category, 0)
                for category in ERROR_CATEGORIES
            },
            "metrics": metrics,
        },
        "queries": query_details,
    }


def analyze_query(
    gold: dict[str, Any],
    candidates: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    *,
    known_issues: Mapping[str, Sequence[str]] | None = None,
    required_paper_overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Diagnose one query, retaining enough evidence for manual review."""

    query_id = _normalize_id(gold.get("query_id"))
    if not query_id:
        raise ValueError("gold record has no query_id")
    trace = trace or {}
    candidate_items = (
        candidates.get("candidate_papers", []) if isinstance(candidates, dict) else []
    )
    candidate_ids, candidate_rank = _ranked_paper_ids(candidate_items)
    trace_candidate_ids, _ = _ranked_paper_ids(trace.get("candidate_papers", []))
    official_gold_ids = _paper_ids(gold.get("gold_papers"))
    gold_evidence = _evidence_items(gold)
    evidence_gold_ids = _paper_ids(gold_evidence)
    overrides = (
        ANSWER_REQUIRED_PAPER_IDS
        if required_paper_overrides is None
        else required_paper_overrides
    )
    curated_required_ids = set(overrides.get(query_id, ()))
    required_gold_ids = curated_required_ids or evidence_gold_ids or official_gold_ids

    judgments = trace.get("relevance_judgments")
    if not isinstance(judgments, list):
        judgments = []
    judged_ids: set[str] = set()
    relevant_ids: set[str] = set()
    answerable_ids: set[str] = set()
    relevant_chunk_ids: dict[str, list[str]] = {}
    malformed_judgment_count = 0
    for judgment in judgments:
        if not isinstance(judgment, dict):
            malformed_judgment_count += 1
            continue
        paper_id = _normalize_id(judgment.get("paper_id"))
        if not paper_id:
            malformed_judgment_count += 1
            continue
        judged_ids.add(paper_id)
        if _as_bool(judgment.get("relevant")):
            relevant_ids.add(paper_id)
        if _as_bool(
            judgment.get(
                "answerable", judgment.get("answerable_from_this_paper")
            )
        ):
            answerable_ids.add(paper_id)
        chunk_ids = judgment.get("evidence_chunk_ids")
        if isinstance(chunk_ids, list):
            existing = relevant_chunk_ids.setdefault(paper_id, [])
            for chunk_id in chunk_ids:
                normalized = _normalize_id(chunk_id)
                if normalized and normalized not in existing:
                    existing.append(normalized)

    prediction = _prediction_record(trace)
    predicted_answer = _prediction_answer(prediction, trace)
    predicted_paper_ids = _predicted_paper_ids(prediction, trace, relevant_ids)
    predicted_evidence = _predicted_evidence(prediction, trace)

    official_candidate_missing = sorted(official_gold_ids - set(candidate_ids))
    required_candidate_missing = sorted(required_gold_ids - set(candidate_ids))
    relevance_false_negatives = sorted(
        (required_gold_ids & set(candidate_ids)) - relevant_ids
    )
    relevance_false_positives = sorted(relevant_ids - required_gold_ids)
    selected_false_positives = sorted(predicted_paper_ids - required_gold_ids)

    selected_precision, selected_recall, selected_f1 = _prf(
        required_gold_ids, predicted_paper_ids
    )
    official_selected_precision, official_selected_recall, official_selected_f1 = (
        _prf(official_gold_ids, predicted_paper_ids)
    )
    official_gold_evidence_keys = _evidence_keys(gold_evidence)
    required_gold_evidence = [
        item
        for item in gold_evidence
        if _normalize_id(item.get("paper_id")) in required_gold_ids
    ]
    required_gold_evidence_keys = _evidence_keys(required_gold_evidence)
    predicted_evidence_keys = _evidence_keys(predicted_evidence)
    evidence_precision, evidence_recall, evidence_f1 = _prf(
        required_gold_evidence_keys, predicted_evidence_keys
    )
    official_evidence_precision, official_evidence_recall, official_evidence_f1 = _prf(
        official_gold_evidence_keys, predicted_evidence_keys
    )
    answer_metrics = _answer_metrics(gold, predicted_answer, trace)
    format_issues = _answer_format_issues(gold, predicted_answer)
    answer_exact = bool(answer_metrics["answer_exact"])
    reading_answer_exact = bool(
        answer_metrics.get("reading_answer_exact", answer_exact)
    )

    dataset_issues = list((known_issues or KNOWN_DATASET_ISSUES).get(query_id, ()))
    for issue in _generic_dataset_issues(gold):
        if issue not in dataset_issues:
            dataset_issues.append(issue)

    categories: set[str] = set()
    if required_candidate_missing:
        categories.add("candidate_missing")
    if relevance_false_negatives:
        categories.add("relevance_filter_false_negative")
    if relevance_false_positives:
        categories.add("relevance_filter_overselection")

    upstream_selection_ok = not required_candidate_missing and not relevance_false_negatives
    if (
        upstream_selection_ok
        and required_gold_evidence_keys != predicted_evidence_keys
    ):
        categories.add("evidence_chunk_selection_error")
    evidence_modalities = {
        _normalize_id(item.get("source_type")) for item in required_gold_evidence
    }
    has_non_text_modality = bool(evidence_modalities & _NON_TEXT_MODALITIES)
    if (
        upstream_selection_ok
        and not reading_answer_exact
        and has_non_text_modality
    ):
        categories.add("modality_read_error")
    if (
        upstream_selection_ok
        and len(required_gold_ids) > 1
        and evidence_recall == 1.0
        and not reading_answer_exact
    ):
        categories.add("multi_paper_integration_error")
    if (
        upstream_selection_ok
        and evidence_recall == 1.0
        and not reading_answer_exact
    ):
        categories.add("answer_extraction_or_reasoning_error")
    if format_issues:
        categories.add("answer_format_error")
    if answer_metrics.get("multiple_choice_protocol_blocked"):
        categories.add("multiple_choice_protocol_blocker")
    if dataset_issues:
        categories.add("dataset_inconsistency")

    ordered_categories = [
        category for category in ERROR_CATEGORIES if category in categories
    ]
    candidate_set = set(candidate_ids)
    return {
        "query_id": query_id,
        "question": str(gold.get("question") or ""),
        "answer_types": list(gold.get("answer_types") or []),
        "task_family": gold.get("task_family"),
        "primary_evidence_type": gold.get("primary_evidence_type"),
        "trace_present": bool(trace),
        "candidate_analysis": {
            "candidate_count": len(candidate_ids),
            "candidate_paper_ids": candidate_ids,
            "trace_candidate_paper_ids": trace_candidate_ids,
            "official_gold_paper_ids": sorted(official_gold_ids),
            "gold_evidence_paper_ids": sorted(evidence_gold_ids),
            "required_gold_paper_ids": sorted(required_gold_ids),
            "required_papers_source": (
                "manual_validation_audit"
                if curated_required_ids
                else "gold_evidence"
                if evidence_gold_ids
                else "official_gold_papers"
            ),
            "official_gold_candidate_ranks": {
                paper_id: candidate_rank.get(paper_id)
                for paper_id in sorted(official_gold_ids)
            },
            "required_gold_candidate_ranks": {
                paper_id: candidate_rank.get(paper_id)
                for paper_id in sorted(required_gold_ids)
            },
            "missing_official_gold_paper_ids": official_candidate_missing,
            "missing_required_gold_paper_ids": required_candidate_missing,
        },
        "relevance_analysis": {
            "judgment_count": len(judgments),
            "malformed_judgment_count": malformed_judgment_count,
            "judged_paper_ids": sorted(judged_ids),
            "relevant_paper_ids": sorted(relevant_ids),
            "answerable_paper_ids": sorted(answerable_ids),
            "required_false_negative_paper_ids": relevance_false_negatives,
            "false_positive_paper_ids": relevance_false_positives,
            "evidence_chunk_ids_by_paper": {
                paper_id: relevant_chunk_ids[paper_id]
                for paper_id in sorted(relevant_chunk_ids)
            },
        },
        "prediction_analysis": {
            "selected_paper_ids": sorted(predicted_paper_ids),
            "selected_false_positive_paper_ids": selected_false_positives,
            "predicted_evidence_count": len(predicted_evidence_keys),
            "gold_evidence_count": len(official_gold_evidence_keys),
            "required_gold_evidence_count": len(required_gold_evidence_keys),
            "gold_evidence_modalities": sorted(evidence_modalities),
            "format_issues": format_issues,
        },
        "metrics": {
            "official_candidate_recall": _recall(official_gold_ids, candidate_set),
            "required_candidate_recall": _recall(required_gold_ids, candidate_set),
            "official_selected_paper_precision": official_selected_precision,
            "official_selected_paper_recall": official_selected_recall,
            "official_selected_paper_f1": official_selected_f1,
            "selected_paper_precision": selected_precision,
            "selected_paper_recall": selected_recall,
            "selected_paper_f1": selected_f1,
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "evidence_f1": evidence_f1,
            "official_evidence_precision": official_evidence_precision,
            "official_evidence_recall": official_evidence_recall,
            "official_evidence_f1": official_evidence_f1,
            **answer_metrics,
        },
        "answer_comparison": {
            "gold": gold.get("answer") if isinstance(gold.get("answer"), dict) else {},
            "predicted": predicted_answer,
            "semantic_multiple_choice": trace.get("semantic_multiple_choice"),
        },
        "dataset_issues": dataset_issues,
        "error_categories": ordered_categories,
    }


def write_analysis_outputs(analysis: dict[str, Any], output_dir: str | Path) -> None:
    """Write aggregate and per-query JSON/Markdown reports."""

    output_dir = Path(output_dir)
    query_dir = output_dir / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", analysis)
    (output_dir / "summary.md").write_text(
        render_summary_markdown(analysis), encoding="utf-8"
    )
    for detail in analysis.get("queries", []):
        query_id = _normalize_id(detail.get("query_id"))
        if not query_id or not _SAFE_QUERY_ID.fullmatch(query_id):
            raise ValueError(f"unsafe query_id for output filename: {query_id!r}")
        _write_json(query_dir / f"{query_id}.json", detail)
        (query_dir / f"{query_id}.md").write_text(
            render_query_markdown(detail), encoding="utf-8"
        )


def render_summary_markdown(analysis: dict[str, Any]) -> str:
    """Render a compact run summary with links to every per-query report."""

    summary = analysis.get("summary", {})
    metrics = summary.get("metrics", {})
    lines = [
        "# LitTraceQA reading error analysis",
        "",
        f"- Queries: {summary.get('total_queries', 0)}",
        f"- Queries with categorized issues: {summary.get('queries_with_errors', 0)}",
        f"- Missing traces: {summary.get('missing_trace_count', 0)}",
        f"- Required-paper candidate recall (macro): {_fmt(metrics.get('required_candidate_recall_macro'))}",
        f"- Selected-paper F1 (macro): {_fmt(metrics.get('selected_paper_f1_macro'))}",
        f"- Evidence F1 (macro): {_fmt(metrics.get('evidence_f1_macro'))}",
        f"- Exact answer rate: {_fmt(metrics.get('answer_exact_macro'))}",
        f"- Reading exact rate (semantic MC): {_fmt(metrics.get('reading_answer_exact_macro'))}",
        "",
        "## Category counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    counts = summary.get("category_counts", {})
    for category in ERROR_CATEGORIES:
        lines.append(f"| `{category}` | {counts.get(category, 0)} |")
    lines.extend(
        [
            "",
            "## Per-query reports",
            "",
            "| Query | Answer exact | Required candidate recall | Categories |",
            "|---|---:|---:|---|",
        ]
    )
    for detail in analysis.get("queries", []):
        query_id = detail["query_id"]
        categories = ", ".join(
            f"`{category}`" for category in detail["error_categories"]
        ) or "none"
        lines.append(
            f"| [{query_id}](queries/{query_id}.md) | "
            f"{int(bool(detail['metrics']['answer_exact']))} | "
            f"{_fmt(detail['metrics']['required_candidate_recall'])} | {categories} |"
        )
    return "\n".join(lines) + "\n"


def render_query_markdown(detail: dict[str, Any]) -> str:
    """Render one query's evidence-backed diagnosis."""

    candidate = detail["candidate_analysis"]
    relevance = detail["relevance_analysis"]
    prediction = detail["prediction_analysis"]
    metrics = detail["metrics"]
    categories = detail["error_categories"]
    lines = [
        f"# {detail['query_id']}",
        "",
        detail.get("question", ""),
        "",
        "## Diagnosis",
        "",
        f"- Categories: {', '.join(f'`{item}`' for item in categories) or 'none'}",
        f"- Task family: `{detail.get('task_family')}`",
        f"- Primary evidence type: `{detail.get('primary_evidence_type')}`",
        f"- Trace present: `{detail.get('trace_present')}`",
        "",
        "## Candidate and relevance stages",
        "",
        f"- Required gold papers: {_inline(candidate['required_gold_paper_ids'])}",
        f"- Required paper ranks: `{json.dumps(candidate['required_gold_candidate_ranks'], ensure_ascii=False, sort_keys=True)}`",
        f"- Missing required papers: {_inline(candidate['missing_required_gold_paper_ids'])}",
        f"- AOAI-relevant papers: {_inline(relevance['relevant_paper_ids'])}",
        f"- Relevance false negatives: {_inline(relevance['required_false_negative_paper_ids'])}",
        f"- Relevance false positives: {_inline(relevance['false_positive_paper_ids'])}",
        f"- Selected papers: {_inline(prediction['selected_paper_ids'])}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Required candidate recall | {_fmt(metrics['required_candidate_recall'])} |",
        f"| Selected paper F1 | {_fmt(metrics['selected_paper_f1'])} |",
        f"| Evidence F1 | {_fmt(metrics['evidence_f1'])} |",
        f"| Freeform exact | {_fmt_optional_bool(metrics.get('freeform_exact'))} |",
        f"| Multiple-choice letter exact | {_fmt_optional_bool(metrics.get('multiple_choice_exact'))} |",
        f"| Semantic multiple-choice exact | {_fmt_optional_bool(metrics.get('semantic_multiple_choice_exact'))} |",
        f"| Table row F1 | {_fmt(metrics.get('table_row_f1'))} |",
        f"| Table cell accuracy | {_fmt(metrics.get('table_cell_accuracy'))} |",
        f"| Overall answer exact | {int(bool(metrics['answer_exact']))} |",
        "",
        "## Answer comparison",
        "",
        "```json",
        json.dumps(detail["answer_comparison"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    if prediction["format_issues"]:
        lines.extend(
            [
                "",
                "## Format issues",
                "",
                *[f"- {issue}" for issue in prediction["format_issues"]],
            ]
        )
    if detail["dataset_issues"]:
        lines.extend(
            [
                "",
                "## Dataset issues",
                "",
                *[f"- {issue}" for issue in detail["dataset_issues"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _unique_by_query_id(
    records: Sequence[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        query_id = _normalize_id(record.get("query_id"))
        if not query_id:
            raise ValueError(f"{label} record {position} has no query_id")
        if query_id in output:
            raise ValueError(f"duplicate query_id in {label} records: {query_id}")
        output[query_id] = record
    return output


def _ranked_paper_ids(raw_papers: Any) -> tuple[list[str], dict[str, int]]:
    if not isinstance(raw_papers, list):
        return [], {}
    indexed: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for position, item in enumerate(raw_papers, start=1):
        if isinstance(item, str):
            paper_id = _normalize_id(item)
            rank = position
        elif isinstance(item, dict):
            paper_id = _normalize_id(item.get("paper_id"))
            try:
                rank = int(item.get("rank", position))
            except (TypeError, ValueError):
                rank = position
        else:
            continue
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        indexed.append((rank, position, paper_id))
    indexed.sort(key=lambda item: (item[0], item[1], item[2]))
    ids = [item[2] for item in indexed]
    return ids, {paper_id: rank for rank, _, paper_id in indexed}


def _paper_ids(raw_papers: Any) -> set[str]:
    if not isinstance(raw_papers, list):
        return set()
    output: set[str] = set()
    for item in raw_papers:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get("paper_id", "")
        else:
            continue
        normalized = _normalize_id(paper_id)
        if normalized:
            output.add(normalized)
    return output


def _evidence_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, dict)]


def _evidence_keys(items: Sequence[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for item in items:
        paper_id = _normalize_id(item.get("paper_id"))
        source_type = _normalize_id(item.get("source_type"))
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        page = _normalize_id(locator.get("page"))
        object_id = ""
        if source_type == "table":
            object_id = _normalize_visible_id(locator.get("table_id"), "table")
        elif source_type == "figure":
            object_id = _normalize_visible_id(locator.get("figure_id"), "figure")
        if paper_id and source_type and page:
            keys.add((paper_id, source_type, page, object_id))
    return keys


def _prediction_record(trace: dict[str, Any]) -> dict[str, Any]:
    for key in ("submission", "prediction", "predicted"):
        nested = trace.get(key)
        if isinstance(nested, dict):
            return nested
    return trace


def _prediction_answer(
    prediction: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    answer = prediction.get("answer")
    if isinstance(answer, dict):
        return answer
    predicted_answer = prediction.get("predicted_answer")
    if not isinstance(predicted_answer, (dict, str)):
        predicted_answer = trace.get("predicted_answer")
    if isinstance(predicted_answer, dict):
        if set(predicted_answer) & {"freeform", "multiple_choice", "table"}:
            return predicted_answer
        return {"freeform": predicted_answer}
    if isinstance(predicted_answer, str):
        return {"freeform": {"text": predicted_answer}}
    answer_text = prediction.get("answer_text") or trace.get("answer_text")
    if answer_text is not None:
        return {"freeform": {"text": str(answer_text)}}
    return {}


def _predicted_paper_ids(
    prediction: dict[str, Any], trace: dict[str, Any], relevant_ids: set[str]
) -> set[str]:
    for container in (prediction, trace):
        for key in ("gold_papers", "papers", "selected_papers"):
            if key in container:
                return _paper_ids(container.get(key))
    return set(relevant_ids)


def _predicted_evidence(
    prediction: dict[str, Any], trace: dict[str, Any]
) -> list[dict[str, Any]]:
    for container in (prediction, trace):
        for key in ("evidence", "selected_evidence"):
            value = container.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _answer_metrics(
    gold: dict[str, Any], predicted: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    requested = set(gold.get("answer_types") or [])
    metrics: dict[str, Any] = {
        "freeform_exact": None,
        "multiple_choice_exact": None,
        "semantic_multiple_choice_exact": None,
        "multiple_choice_protocol_blocked": False,
        "table_row_precision": None,
        "table_row_recall": None,
        "table_row_f1": None,
        "table_cell_accuracy": None,
    }
    exact_values: list[bool] = []
    reading_exact_values: list[bool] = []
    gold_answer = gold.get("answer") if isinstance(gold.get("answer"), dict) else {}

    if "freeform" in requested:
        gold_text = _nested_text(gold_answer.get("freeform"))
        pred_text = _nested_text(predicted.get("freeform"))
        value = bool(gold_text) and _normalize_text(gold_text) == _normalize_text(pred_text)
        metrics["freeform_exact"] = value
        exact_values.append(value)
        reading_exact_values.append(value)

    if "multiple_choice" in requested:
        gold_mc = gold_answer.get("multiple_choice")
        pred_mc = predicted.get("multiple_choice")
        gold_letter = _mc_letter(gold_mc)
        pred_letter = _mc_letter(pred_mc)
        letter_exact = bool(gold_letter) and pred_letter == gold_letter
        metrics["multiple_choice_exact"] = letter_exact
        exact_values.append(letter_exact)

        semantic = trace.get("semantic_multiple_choice")
        semantic_text = _nested_text(semantic)
        gold_option_text = _gold_mc_option_text(gold_mc, gold_letter)
        semantic_exact: bool | None = None
        if semantic_text and gold_option_text:
            semantic_exact = _semantic_choice_equal(semantic_text, gold_option_text)
        metrics["semantic_multiple_choice_exact"] = semantic_exact
        reading_exact_values.append(
            semantic_exact if semantic_exact is not None else letter_exact
        )
        explicit_block = _as_bool(trace.get("multiple_choice_protocol_blocker"))
        metrics["multiple_choice_protocol_blocked"] = bool(
            explicit_block
            or (semantic_exact is True and not letter_exact)
            or (semantic_text and not pred_letter)
        )

    if "table" in requested:
        table = _table_metrics(gold_answer.get("table"), predicted.get("table"))
        metrics.update(table)
        table_exact = bool(
            table["table_row_f1"] == 1.0
            and table["table_cell_accuracy"] == 1.0
        )
        exact_values.append(table_exact)
        reading_exact_values.append(table_exact)

    metrics["answer_exact"] = bool(exact_values) and all(exact_values)
    # Official A-D accuracy and reading quality are distinct when the organizer
    # withholds option text.  Use the semantic option match for root-cause
    # classification, while retaining answer_exact for the official shape.
    metrics["reading_answer_exact"] = bool(reading_exact_values) and all(
        reading_exact_values
    )
    return metrics


def _answer_format_issues(
    gold: dict[str, Any], predicted: dict[str, Any]
) -> list[str]:
    requested = list(dict.fromkeys(gold.get("answer_types") or []))
    issues: list[str] = []
    for answer_type in requested:
        if answer_type not in predicted:
            issues.append(f"missing requested answer type: {answer_type}")
    extra = sorted(set(predicted) - set(requested))
    for answer_type in extra:
        issues.append(f"unexpected answer type: {answer_type}")

    if (
        "freeform" in requested
        and "freeform" in predicted
        and not _nested_text(predicted.get("freeform"))
    ):
        issues.append("freeform.text is empty or missing")
    if (
        "multiple_choice" in requested
        and "multiple_choice" in predicted
        and not _mc_letter(predicted.get("multiple_choice"))
    ):
        issues.append("multiple_choice answer is not a letter A-D")
    if "table" in requested and "table" in predicted:
        table = predicted.get("table")
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(rows, list) or not rows:
            issues.append("table.rows is empty or missing")
        else:
            schema = gold.get("table_schema")
            if not isinstance(schema, list):
                gold_table = gold.get("answer", {}).get("table", {})
                schema = gold_table.get("schema") if isinstance(gold_table, dict) else []
            columns = [
                str(column.get("name"))
                for column in (schema or [])
                if isinstance(column, dict) and column.get("name")
            ]
            for row_number, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    issues.append(f"table row {row_number} is not an object")
                    continue
                missing = [column for column in columns if column not in row]
                if missing:
                    issues.append(
                        f"table row {row_number} is missing columns: {', '.join(missing)}"
                    )
    return issues


def _table_metrics(gold_table: Any, predicted_table: Any) -> dict[str, float]:
    gold_table = gold_table if isinstance(gold_table, dict) else {}
    predicted_table = predicted_table if isinstance(predicted_table, dict) else {}
    schema = gold_table.get("schema") if isinstance(gold_table.get("schema"), list) else []
    gold_rows = gold_table.get("rows") if isinstance(gold_table.get("rows"), list) else []
    pred_rows = (
        predicted_table.get("rows")
        if isinstance(predicted_table.get("rows"), list)
        else []
    )
    columns = [
        str(column.get("name"))
        for column in schema
        if isinstance(column, dict) and column.get("name")
    ]
    column_types = {
        str(column.get("name")): str(column.get("type") or "string")
        for column in schema
        if isinstance(column, dict) and column.get("name")
    }
    row_keys = [
        str(column.get("name"))
        for column in schema
        if isinstance(column, dict)
        and column.get("name")
        and column.get("is_row_key")
    ]
    if not row_keys and columns:
        row_keys = columns[:1]

    gold_by_key = _rows_by_key(gold_rows, row_keys)
    pred_by_key = _rows_by_key(pred_rows, row_keys)
    row_precision, row_recall, row_f1 = _prf(set(gold_by_key), set(pred_by_key))
    value_columns = [column for column in columns if column not in row_keys]
    correct = 0
    total = 0
    for key, gold_row in gold_by_key.items():
        pred_row = pred_by_key.get(key)
        for column in value_columns:
            total += 1
            if isinstance(pred_row, dict) and _cell_equal(
                gold_row.get(column), pred_row.get(column), column_types.get(column, "string")
            ):
                correct += 1
    cell_accuracy = correct / total if total else row_f1
    return {
        "table_row_precision": row_precision,
        "table_row_recall": row_recall,
        "table_row_f1": row_f1,
        "table_cell_accuracy": cell_accuracy,
    }


def _rows_by_key(rows: list[Any], row_keys: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    output: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = tuple(_normalize_text(row.get(column)) for column in row_keys)
        if key and all(key):
            output[key] = row
    return output


def _cell_equal(gold: Any, predicted: Any, column_type: str) -> bool:
    if gold is None or predicted is None:
        return gold is None and predicted is None
    if column_type == "number":
        gold_number = _number(gold)
        pred_number = _number(predicted)
        return bool(
            gold_number is not None
            and pred_number is not None
            and math.isclose(gold_number, pred_number, rel_tol=1e-6, abs_tol=1e-6)
        )
    if column_type == "boolean":
        return _normalize_text(gold) == _normalize_text(predicted)
    return _normalize_text(gold) == _normalize_text(predicted)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).strip().replace(",", "").replace("%", ""))
    except ValueError:
        return None


def _ap_target_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for match in re.findall(r"AP\s*\^\s*(?:\{([^}]+)\}|([A-Za-z]+))", text)
        for token in match
        if token
    }


def _generic_dataset_issues(gold: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    primary = _normalize_id(gold.get("primary_evidence_type"))
    evidence_modalities = {
        _normalize_id(item.get("source_type"))
        for item in _evidence_items(gold)
        if _normalize_id(item.get("source_type"))
    }
    if primary and evidence_modalities and primary not in evidence_modalities:
        issues.append(
            f"primary_evidence_type={primary!r} disagrees with gold evidence "
            f"source types {sorted(evidence_modalities)}."
        )
    question = str(gold.get("question") or "")
    target_tokens = _ap_target_tokens(question)
    if not target_tokens:
        return issues
    answer = gold.get("answer") if isinstance(gold.get("answer"), dict) else {}
    schema_text = json.dumps(
        {
            "table_schema": gold.get("table_schema"),
            "answer": answer,
            "evidence": gold.get("evidence"),
        },
        ensure_ascii=False,
    )
    answer_tokens = _ap_target_tokens(schema_text)
    if answer_tokens and target_tokens.isdisjoint(answer_tokens):
        issues.append(
            "Question AP target token(s) "
            f"{sorted(target_tokens)} disagree with gold/schema token(s) "
            f"{sorted(answer_tokens)}."
        )
    return issues


def _normalize_id(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    return re.sub(r"\s+", " ", text)


def _normalize_visible_id(value: Any, prefix: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    match = re.fullmatch(rf"{prefix}\s*(\d+[a-z]?)", text)
    if match:
        return f"{prefix} {match.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{prefix} {text}"
    return text


def _nested_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "answer", "value", "semantic_answer"):
            if value.get(key) is not None:
                return str(value[key]).strip()
    return ""


def _mc_letter(value: Any) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
    elif isinstance(value, dict):
        candidate = str(
            value.get("gold")
            or value.get("answer")
            or value.get("predicted_answer_id")
            or ""
        ).strip().upper()
    else:
        candidate = ""
    return candidate if _MC_LETTER.fullmatch(candidate) else ""


def _gold_mc_option_text(value: Any, letter: str) -> str:
    if not isinstance(value, dict) or not letter:
        return ""
    options = value.get("options")
    if not isinstance(options, dict):
        return ""
    return str(options.get(letter) or "").strip()


def _semantic_choice_equal(predicted: str, gold: str) -> bool:
    pred = _normalize_text(predicted)
    target = _normalize_text(gold)
    if pred == target:
        return True
    stripped = re.sub(r"^(the answer is|answer)\s*[:=-]?\s*", "", pred)
    return stripped == target


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value) in {"true", "yes", "1"}


def _prf(gold: set[Any], predicted: set[Any]) -> tuple[float, float, float]:
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _recall(gold: set[Any], predicted: set[Any]) -> float:
    return len(gold & predicted) / len(gold) if gold else 1.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_optional_bool(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(bool(value)))


def _inline(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) or "none"
