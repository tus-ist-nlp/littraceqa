"""Assemble and atomically write the evaluation result document."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from littraceqa.di_pipeline.evaluation.metrics import aggregate_rankings


def _final_rerank_status_counts(
    diagnostics: Iterable[dict],
) -> dict[str, int]:
    """Count at most one typed final-rerank status per query diagnostic."""

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        details = diagnostic.get("ranking_details")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            status = detail.get("final_rerank_status")
            if not isinstance(status, str) or not status:
                continue
            counts[status] = counts.get(status, 0) + 1
            break
    return dict(sorted(counts.items()))


def build_output_payload(
    records: Sequence[dict],
    diagnostics: dict[str, dict],
    failures: dict[str, dict],
    checkpoint: dict,
    ks: Sequence[int],
) -> dict:
    """Recompute metrics from successful diagnostics and preserve input order."""
    ordered_diagnostics = [
        diagnostics[str(record["query_id"])]
        for record in records
        if str(record["query_id"]) in diagnostics
    ]
    rankings = [
        (
            {str(paper_id) for paper_id in diagnostic["gold_papers"]},
            [str(paper_id) for paper_id in diagnostic["ranked_papers"]],
        )
        for diagnostic in ordered_diagnostics
    ]
    ordered_failures = [
        failures[str(record["query_id"])]
        for record in records
        if str(record["query_id"]) in failures
    ]
    pending_query_count = (
        len(records) - len(ordered_diagnostics) - len(ordered_failures)
    )
    summary = {
        "requested_query_count": len(records),
        "successful_query_count": len(ordered_diagnostics),
        "failed_query_count": len(ordered_failures),
        "pending_query_count": pending_query_count,
        "completed": len(ordered_diagnostics) == len(records),
        "metrics_include_successful_queries_only": (
            len(ordered_diagnostics) != len(records)
        ),
    }
    final_rerank_status_counts = _final_rerank_status_counts(
        ordered_diagnostics
    )
    if final_rerank_status_counts:
        summary["final_rerank_status_counts"] = final_rerank_status_counts
    return {
        "metrics": aggregate_rankings(rankings, ks),
        "queries": ordered_diagnostics,
        "failures": ordered_failures,
        "summary": summary,
        "_checkpoint": checkpoint,
    }


def write_output_atomic(output: Path, payload: dict) -> None:
    """Atomically replace an evaluation checkpoint in its target directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output_path(output: Path, read_only_root: Path) -> Path:
    """Resolve an evaluation output path and reject shared input locations."""
    resolved = output.expanduser().resolve()
    shared = read_only_root.expanduser().resolve()
    try:
        resolved.relative_to(shared)
    except ValueError:
        return resolved
    raise ValueError(f"refusing to write evaluation output under {shared}")
