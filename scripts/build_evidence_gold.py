#!/usr/bin/env python3
"""Rebuild the retrieval gold set from evidence-backed gold papers only.

``data/validation.jsonl`` lists a paper in ``gold_papers`` even when no
``evidence`` item points at it.  For the paper-retrieval task those entries are
unverifiable: nothing in the record says which table, figure, or sentence makes
the paper necessary, so a retriever is scored against a target the annotation
does not justify.

The dataset builders already applied this rule at the query level -- the release
report records ``q_016`` being dropped with "at least one evidence item is
required".  This script applies the same rule one level down, at the paper
level, and keeps a full audit trail of what it removed.

A gold paper is kept when its ``paper_id`` appears as ``evidence[].paper_id``.
``locator.cited_paper`` is deliberately not consulted: it holds a free-text
title, not a corpus ``paper_id``.

Example:
    uv run python scripts/build_evidence_gold.py \\
      --gold data/validation.jsonl \\
      --output data/validation_evidence_gold.jsonl \\
      --report data/validation_evidence_gold_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per non-empty line."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def evidence_paper_ids(record: dict[str, Any]) -> set[str]:
    """Return the paper IDs that at least one evidence item points at."""

    paper_ids: set[str] = set()
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return paper_ids
    for item in evidence:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def evidence_without_text(record: dict[str, Any]) -> list[str]:
    """Return evidence IDs whose text/value is empty.

    These still carry a locator, so the paper stays in the gold set, but the
    release report flags them and they are worth knowing about.
    """

    empty: list[str] = []
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return empty
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        value = item.get("evidence_text_or_value")
        if value is None or (isinstance(value, str) and not value.strip()):
            empty.append(str(item.get("evidence_id") or f"evidence[{index}]"))
    return empty


def filter_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop gold papers with no evidence, returning the record and an audit row."""

    gold_papers = record.get("gold_papers")
    if not isinstance(gold_papers, list):
        raise ValueError(
            f"{record.get('query_id')}: gold_papers must be a list"
        )

    supported = evidence_paper_ids(record)
    kept: list[Any] = []
    dropped: list[str] = []
    for paper in gold_papers:
        if not isinstance(paper, dict) or not paper.get("paper_id"):
            raise ValueError(
                f"{record.get('query_id')}: gold_papers entries need a paper_id"
            )
        if paper["paper_id"] in supported:
            kept.append(paper)
        else:
            dropped.append(str(paper["paper_id"]))

    filtered = dict(record)
    filtered["gold_papers"] = kept
    audit = {
        "query_id": record.get("query_id"),
        "task_family": record.get("task_family"),
        "primary_evidence_type": record.get("primary_evidence_type"),
        "gold_before": len(gold_papers),
        "gold_after": len(kept),
        "dropped_paper_ids": dropped,
        "evidence_items": len(record.get("evidence") or []),
        "evidence_without_text": evidence_without_text(record),
    }
    return filtered, audit


def build(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter every record and summarize what changed."""

    filtered: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for record in records:
        kept_record, audit = filter_record(record)
        filtered.append(kept_record)
        audits.append(audit)

    emptied = [a["query_id"] for a in audits if a["gold_after"] == 0]
    by_family: dict[str, dict[str, int]] = {}
    for audit in audits:
        family = str(audit["task_family"])
        stats = by_family.setdefault(
            family,
            {"queries": 0, "gold_before": 0, "gold_after": 0},
        )
        stats["queries"] += 1
        stats["gold_before"] += audit["gold_before"]
        stats["gold_after"] += audit["gold_after"]

    report = {
        "rule": (
            "keep gold_papers whose paper_id appears as evidence[].paper_id"
        ),
        "queries": len(records),
        "gold_before": sum(a["gold_before"] for a in audits),
        "gold_after": sum(a["gold_after"] for a in audits),
        "queries_with_drops": sum(1 for a in audits if a["dropped_paper_ids"]),
        "queries_emptied": emptied,
        "by_task_family": by_family,
        "per_query": audits,
    }
    return filtered, report


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one compact JSON object per line."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep only evidence-backed gold papers for retrieval scoring"
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/validation.jsonl"),
        help="Source gold JSONL (default: data/validation.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Filtered gold JSONL to write",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON audit trail of every dropped paper",
    )
    parser.add_argument(
        "--drop-emptied-queries",
        action="store_true",
        help=(
            "Also drop queries left with zero gold papers. Off by default so "
            "the query set stays comparable with the original file."
        ),
    )
    args = parser.parse_args()

    if args.output.resolve() == args.gold.resolve():
        parser.error("--output must not overwrite --gold")

    records = load_jsonl(args.gold)
    filtered, report = build(records)

    if args.drop_emptied_queries:
        emptied = set(report["queries_emptied"])
        filtered = [r for r in filtered if r.get("query_id") not in emptied]
        report["dropped_queries"] = sorted(emptied)
    report["written_queries"] = len(filtered)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, filtered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"queries        : {report['queries']} -> {report['written_queries']}")
    print(
        f"gold papers    : {report['gold_before']} -> {report['gold_after']} "
        f"({report['gold_before'] - report['gold_after']} dropped)"
    )
    print(f"queries w/drops: {report['queries_with_drops']}")
    for family, stats in sorted(report["by_task_family"].items()):
        print(
            f"  {family:32s} {stats['queries']:3d} queries  "
            f"gold {stats['gold_before']:3d} -> {stats['gold_after']:3d}"
        )
    if report["queries_emptied"]:
        print(
            f"WARNING: {len(report['queries_emptied'])} queries have no "
            f"evidence-backed gold paper: {report['queries_emptied']}",
            file=sys.stderr,
        )
    print(f"wrote {args.output}")
    if args.report is not None:
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
