"""Production-safe loader for search results handed to the reading agent.

The held-out LitTraceQA input contains only ``query_id``, ``question``,
``answer_types`` and ``table_schema``.  Development handoff files may contain
``_gold`` and other analysis fields, so callers must never pass an input record
through wholesale.  This module projects the input and the search result into
small typed objects before any prompt is built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from littraceqa.di_pipeline.contracts import Query


PRODUCTION_INPUT_FIELDS = (
    "query_id",
    "question",
    "answer_types",
    "table_schema",
)
OFFICIAL_ANSWER_TYPES = frozenset({"freeform", "multiple_choice", "table"})

FORBIDDEN_CANDIDATE_FIELDS = frozenset(
    {
        "_gold",
        "gold_papers",
        "evidence",
        "answer",
        "options",
        "multiple_choice_options",
        "task_family",
        "primary_evidence_type",
    }
)
_CANDIDATE_RECORD_FIELDS = frozenset({"query_id", "candidate_papers", "_meta"})


@dataclass(frozen=True)
class CandidatePaper:
    """One paper returned by the retrieval stage."""

    paper_id: str
    rank: int
    title: str = ""
    venue: str = ""
    year: int | None = None


@dataclass(frozen=True)
class CandidateHandoff:
    """The only object accepted by the corpus reading agent."""

    query: Query
    candidate_papers: tuple[CandidatePaper, ...]


def production_query_from_record(record: dict[str, Any]) -> Query:
    """Project a possibly gold-shaped record onto the four official fields."""

    missing = [name for name in ("query_id", "question") if not record.get(name)]
    if missing:
        raise ValueError(f"production input is missing required fields: {missing}")
    answer_types = record.get("answer_types") or []
    if not isinstance(answer_types, list):
        raise TypeError("answer_types must be a list")
    unknown_answer_types = sorted(set(answer_types) - OFFICIAL_ANSWER_TYPES)
    if unknown_answer_types:
        raise ValueError(f"unknown answer types: {unknown_answer_types}")
    table_schema = record.get("table_schema")
    if table_schema is not None and not isinstance(table_schema, list):
        raise TypeError("table_schema must be a list or null")
    if "table" in answer_types and not table_schema:
        raise ValueError("table answer type requires a non-empty table_schema")
    return Query(
        query_id=str(record["query_id"]),
        question=str(record["question"]),
        answer_types=[str(item) for item in answer_types],
        table_schema=table_schema,
        options=None,
        task_family=None,
        primary_evidence_type=None,
    )


def require_production_query(query: Query) -> None:
    """Fail closed if development-only information reaches the agent."""

    forbidden = {
        "options": query.options,
        "task_family": query.task_family,
        "primary_evidence_type": query.primary_evidence_type,
    }
    present = sorted(name for name, value in forbidden.items() if value is not None)
    if present:
        raise ValueError(
            "corpus QA accepts only the four official input fields; "
            f"forbidden values are present: {', '.join(present)}"
        )


def candidate_papers_from_record(record: dict[str, Any]) -> tuple[CandidatePaper, ...]:
    """Parse and validate a ranked candidate list without reading ``_gold``."""

    unexpected = sorted(set(record) - _CANDIDATE_RECORD_FIELDS)
    forbidden = sorted(FORBIDDEN_CANDIDATE_FIELDS.intersection(record))
    if forbidden:
        raise ValueError(
            "candidate handoff contains oracle/development fields; sanitize it first: "
            + ", ".join(forbidden)
        )
    if unexpected:
        raise ValueError(
            "candidate handoff must be a separate sidecar; unexpected fields: "
            + ", ".join(unexpected)
        )

    raw_papers = record.get("candidate_papers")
    if not isinstance(raw_papers, list):
        raise ValueError(
            f"query {record.get('query_id')!r} has no candidate_papers list"
        )

    papers: list[CandidatePaper] = []
    seen: set[str] = set()
    for position, item in enumerate(raw_papers, start=1):
        if isinstance(item, str):
            paper_id = item
            rank = position
            title = venue = ""
            year = None
        elif isinstance(item, dict):
            paper_id = str(item.get("paper_id") or "")
            try:
                rank = int(item.get("rank", position))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid candidate rank for {paper_id!r}") from exc
            title = str(item.get("title") or "")
            venue = str(item.get("venue") or "")
            raw_year = item.get("year")
            try:
                year = int(raw_year) if raw_year is not None else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid candidate year for {paper_id!r}") from exc
        else:
            raise ValueError(
                f"candidate item at position {position} must be a paper_id or object"
            )
        if not paper_id:
            raise ValueError(f"candidate item at position {position} has no paper_id")
        if paper_id in seen:
            raise ValueError(f"duplicate candidate paper_id: {paper_id}")
        seen.add(paper_id)
        papers.append(
            CandidatePaper(
                paper_id=paper_id,
                rank=rank,
                title=title,
                venue=venue,
                year=year,
            )
        )

    papers.sort(key=lambda paper: (paper.rank, paper.paper_id))
    if not papers:
        raise ValueError(f"query {record.get('query_id')!r} has no usable candidates")
    ranks = [paper.rank for paper in papers]
    if ranks != list(range(1, len(papers) + 1)):
        raise ValueError(
            f"query {record.get('query_id')!r} candidate ranks must be consecutive from 1"
        )
    return tuple(papers)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def load_candidate_handoffs(
    queries_path: str | Path,
    candidates_path: str | Path | None = None,
    paper_metadata_path: str | Path | None = None,
) -> list[CandidateHandoff]:
    """Join official inputs with retrieval output by ``query_id``.

    ``queries_path`` remains authoritative for the four official fields.
    ``candidates_path`` contributes only ``query_id`` and ``candidate_papers``;
    its question, ``_gold`` and metadata are deliberately ignored.
    """

    query_records = read_jsonl(queries_path)
    candidate_records = read_jsonl(candidates_path or queries_path)
    candidates_by_id = _unique_by_query_id(candidate_records, "candidate records")

    paper_metadata = (
        _paper_metadata_by_id(read_jsonl(paper_metadata_path))
        if paper_metadata_path is not None
        else None
    )
    handoffs: list[CandidateHandoff] = []
    seen_queries: set[str] = set()
    for record in query_records:
        query = production_query_from_record(record)
        if query.query_id in seen_queries:
            raise ValueError(f"duplicate query_id in inputs: {query.query_id}")
        seen_queries.add(query.query_id)
        candidate_record = candidates_by_id.get(query.query_id)
        if candidate_record is None:
            raise ValueError(f"missing candidates for query_id {query.query_id}")
        papers = candidate_papers_from_record(candidate_record)
        if paper_metadata is not None:
            papers = _canonicalize_candidate_metadata(
                query.query_id, papers, paper_metadata
            )
        handoffs.append(
            CandidateHandoff(
                query=query,
                candidate_papers=papers,
            )
        )
    extra_candidates = sorted(set(candidates_by_id) - seen_queries)
    if extra_candidates:
        raise ValueError(f"candidate sidecar has extra query_ids: {extra_candidates}")
    return handoffs


def _unique_by_query_id(
    records: Iterable[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "")
        if not query_id:
            raise ValueError(f"{label} contains a record without query_id")
        if query_id in indexed:
            raise ValueError(f"duplicate query_id in {label}: {query_id}")
        indexed[query_id] = record
    return indexed


def _paper_metadata_by_id(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        if not paper_id or paper_id in metadata:
            raise ValueError(f"invalid/duplicate paper_id in paper metadata: {paper_id!r}")
        metadata[paper_id] = record
    return metadata


def _canonicalize_candidate_metadata(
    query_id: str,
    papers: tuple[CandidatePaper, ...],
    metadata: dict[str, dict[str, Any]],
) -> tuple[CandidatePaper, ...]:
    canonicalized: list[CandidatePaper] = []
    for paper in papers:
        canonical = metadata.get(paper.paper_id)
        if canonical is None:
            raise ValueError(
                f"{query_id}: candidate paper is absent from metadata: {paper.paper_id}"
            )
        expected = {
            "title": str(canonical.get("title") or ""),
            "venue": str(canonical.get("venue") or ""),
            "year": canonical.get("year"),
        }
        observed = {
            "title": paper.title,
            "venue": paper.venue,
            "year": paper.year,
        }
        for key in ("title", "venue", "year"):
            if observed[key] not in ("", None) and observed[key] != expected[key]:
                raise ValueError(
                    f"{query_id}: candidate {paper.paper_id} has mismatched {key}"
                )
        canonicalized.append(
            CandidatePaper(
                paper_id=paper.paper_id,
                rank=paper.rank,
                title=paper.title or expected["title"],
                venue=paper.venue or expected["venue"],
                year=paper.year if paper.year is not None else expected["year"],
            )
        )
    return tuple(canonicalized)
