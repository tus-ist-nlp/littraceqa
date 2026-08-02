"""Load the answer-bearing gold file and the queries scored against it."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path


def load_gold(path: Path) -> list[dict]:
    """Load JSONL records containing questions and gold papers."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_records(records: Iterable[dict], query_ids: Sequence[str]) -> list[dict]:
    """Select explicit query IDs in request order for bounded evaluations."""
    requested = tuple(dict.fromkeys(query_ids))
    available: dict[str, dict] = {}
    for record in records:
        query_id = str(record.get("query_id") or "")
        if not query_id:
            raise ValueError("evaluation input contains an empty query_id")
        if query_id in available:
            raise ValueError(f"duplicate query_id in evaluation input: {query_id}")
        available[query_id] = record

    if not requested:
        return list(available.values())

    missing = [query_id for query_id in requested if query_id not in available]
    if missing:
        raise ValueError(f"query_id not found in evaluation input: {', '.join(missing)}")
    return [available[query_id] for query_id in requested]


def gold_paper_ids(record: dict) -> set[str]:
    """Return the non-empty paper IDs in a validation record."""
    return {
        str(paper["paper_id"])
        for paper in record.get("gold_papers", [])
        if isinstance(paper, dict) and paper.get("paper_id")
    }


def parse_ks(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive, unique retrieval cutoffs."""
    try:
        ks = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must contain integers") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks
