"""Durable checkpoints and aggregate artifacts for pairwise reading runs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from littraceqa.aoai_pairwise_reader import PairwiseAOAIReader
from littraceqa.candidate_handoff import (
    CandidateHandoff,
    CandidatePaper,
    read_jsonl,
)
from littraceqa.di_pipeline.contracts import (
    Answer,
    Evidence,
    EvidenceLocator,
    Prediction,
    Query,
)
from littraceqa.submission import TOP_LEVEL_KEYS, prediction_to_submission


@dataclass(frozen=True)
class QueryRunPaths:
    """All per-query checkpoint paths derived from one safe query ID."""

    directory: Path
    judgments: Path
    answer: Path
    submission: Path
    errors: Path

    @classmethod
    def under(cls, run_dir: Path, query_id: str) -> QueryRunPaths:
        directory = run_dir / query_id
        return cls(
            directory=directory,
            judgments=directory / "candidate_judgments.jsonl",
            answer=directory / "answer.json",
            submission=directory / "submission.json",
            errors=directory / "errors.jsonl",
        )


@dataclass(frozen=True)
class JudgmentCheckpointStatus:
    """Completeness and cache validity of one query's paper judgments."""

    missing_paper_ids: tuple[str, ...]
    stale_paper_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_paper_ids

    @property
    def current(self) -> bool:
        return self.complete and not self.stale_paper_ids


def ensure_manifest(path: Path, manifest: dict[str, Any], resume: bool) -> None:
    """Create a run manifest or prove that a resumed run is identical."""
    if path.exists():
        if not resume:
            raise ValueError(
                f"run already exists: {path.parent}; pass --resume or choose a new --run-dir"
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "resume manifest differs from current data/code/config/model; "
                "use a new --run-dir"
            )
        return
    atomic_write_json(path, manifest)


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace one human-readable JSON checkpoint."""
    with _atomic_text_writer(path) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically replace one JSONL artifact without buffering its text."""
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def invalidate_aggregate_query(run_dir: Path, query_id: str) -> None:
    """Remove a mutated query from aggregate artifacts immediately.

    Per-query checkpoints remain durable. This prevents an interrupted force
    rejudgment from leaving an old answer paired with new judgments in the root
    trace and submission files.
    """
    for filename in ("reading_traces.jsonl", "submission.jsonl"):
        path = run_dir / filename
        if not path.exists():
            continue
        records = [
            record
            for record in read_jsonl(path)
            if str(record.get("query_id") or "") != query_id
        ]
        atomic_write_jsonl(path, records)


def load_judgments(path: Path, query_id: str) -> dict[str, dict[str, Any]]:
    """Load and validate a query's completed judgment checkpoints."""
    if not path.exists():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        paper_id = str(record.get("paper_id") or "")
        if (
            record.get("query_id") != query_id
            or not paper_id
            or paper_id in output
            or record.get("status") != "complete"
        ):
            raise ValueError(f"invalid judgment checkpoint at {path}:{line_number}")
        output[paper_id] = record
    return output


def write_judgments(
    path: Path,
    judgments: dict[str, dict[str, Any]],
    candidates: tuple[CandidatePaper, ...],
) -> None:
    """Persist judgments in the immutable candidate-ranking order."""
    candidate_order = {paper.paper_id: index for index, paper in enumerate(candidates)}
    extra = sorted(set(judgments) - set(candidate_order))
    if extra:
        raise ValueError(f"judgment checkpoint contains non-candidates: {extra}")
    ordered = sorted(
        judgments.values(), key=lambda item: candidate_order[str(item["paper_id"])]
    )
    atomic_write_jsonl(path, ordered)


def record_error(
    path: Path,
    *,
    stage: str,
    query_id: str,
    error: Exception,
    paper_id: str | None = None,
) -> None:
    """Append one durable error record without corrupting the existing log."""
    records = read_jsonl(path) if path.exists() else []
    records.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage,
            "query_id": query_id,
            "paper_id": paper_id,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    atomic_write_jsonl(path, records)


def validate_judgment_checkpoint(
    handoff: CandidateHandoff,
    judgments: dict[str, dict[str, Any]],
    reader: PairwiseAOAIReader,
) -> JudgmentCheckpointStatus:
    """Validate candidate coverage and every content-sensitive cache key."""
    query_id = handoff.query.query_id
    expected_ids = {candidate.paper_id for candidate in handoff.candidate_papers}
    unexpected_ids = sorted(set(judgments) - expected_ids)
    if unexpected_ids:
        raise ValueError(
            f"{query_id}: judgment checkpoint contains non-candidates: {unexpected_ids}"
        )
    missing_ids = tuple(sorted(expected_ids - set(judgments)))
    stale_ids: list[str] = []
    for candidate in handoff.candidate_papers:
        judgment = judgments.get(candidate.paper_id)
        if judgment is None:
            continue
        records = reader.chunk_store.load_paper(candidate.paper_id)
        expected_key = reader.judgment_cache_key(handoff.query, candidate, records)
        if judgment.get("cache_key") != expected_key:
            stale_ids.append(candidate.paper_id)
    return JudgmentCheckpointStatus(
        missing_paper_ids=missing_ids,
        stale_paper_ids=tuple(stale_ids),
    )


def _candidate_trace_payload(candidate: CandidatePaper) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "paper_id": candidate.paper_id,
        "title": candidate.title,
        "venue": candidate.venue,
        "year": candidate.year,
    }


def _prediction_from_checkpoint(raw: Any, *, query_id: str) -> Prediction:
    """Rehydrate the analysis prediction saved inside ``answer.json``."""
    if not isinstance(raw, dict):
        raise TypeError(f"{query_id}: answer checkpoint has no prediction object")
    if str(raw.get("query_id") or "") != query_id:
        raise ValueError(f"{query_id}: answer checkpoint prediction query_id mismatch")

    raw_papers = raw.get("gold_papers")
    raw_evidence = raw.get("evidence")
    raw_answer = raw.get("answer")
    if not isinstance(raw_papers, list):
        raise TypeError(f"{query_id}: checkpoint prediction has invalid gold_papers")
    if not isinstance(raw_evidence, list):
        raise TypeError(f"{query_id}: checkpoint prediction has invalid evidence")
    if not isinstance(raw_answer, dict):
        raise TypeError(f"{query_id}: checkpoint prediction has invalid answer")

    locator_fields = set(EvidenceLocator.__dataclass_fields__)
    evidence: list[Evidence] = []
    for position, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            raise TypeError(
                f"{query_id}: checkpoint evidence {position} is not an object"
            )
        raw_locator = item.get("locator")
        if not isinstance(raw_locator, dict) or set(raw_locator) - locator_fields:
            raise ValueError(
                f"{query_id}: checkpoint evidence {position} has an invalid locator"
            )
        evidence.append(
            Evidence(
                paper_id=str(item.get("paper_id") or ""),
                source_type=str(item.get("source_type") or ""),
                locator=EvidenceLocator(**raw_locator),
                evidence_text_or_value=item.get("evidence_text_or_value"),
            )
        )

    trace = raw.get("trace") or []
    candidate_papers = raw.get("candidate_papers") or []
    if not isinstance(trace, list) or not isinstance(candidate_papers, list):
        raise TypeError(f"{query_id}: checkpoint prediction analysis fields are invalid")
    return Prediction(
        query_id=query_id,
        gold_papers=raw_papers,
        evidence=evidence,
        answer=Answer(
            freeform=raw_answer.get("freeform"),
            multiple_choice=raw_answer.get("multiple_choice"),
            table=raw_answer.get("table"),
        ),
        trace=trace,
        candidate_papers=[str(item) for item in candidate_papers],
    )


def _submission_from_answer_checkpoint(
    query: Query, answer_record: dict[str, Any]
) -> dict[str, Any]:
    prediction = _prediction_from_checkpoint(
        answer_record.get("prediction"), query_id=query.query_id
    )
    return prediction_to_submission(query, prediction)


def materialize_run_outputs(
    run_dir: Path,
    handoffs: list[CandidateHandoff],
    reader: PairwiseAOAIReader,
) -> tuple[int, int]:
    """Rebuild aggregate traces/submissions only from valid checkpoints."""
    traces: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    for handoff in handoffs:
        query_id = handoff.query.query_id
        paths = QueryRunPaths.under(run_dir, query_id)
        if not paths.judgments.exists():
            if paths.answer.exists() or paths.submission.exists():
                raise ValueError(
                    f"{query_id}: answer/submission exists without candidate "
                    "judgment checkpoint"
                )
            continue

        judgments_by_id = load_judgments(paths.judgments, query_id)
        judgments = list(judgments_by_id.values())
        checkpoint = validate_judgment_checkpoint(handoff, judgments_by_id, reader)
        trace: dict[str, Any] = {
            "query_id": query_id,
            "candidate_papers": [
                _candidate_trace_payload(candidate)
                for candidate in handoff.candidate_papers
            ],
            "relevance_judgments": judgments,
            "judgment_checkpoints_complete": checkpoint.complete,
            "missing_judgment_paper_ids": list(checkpoint.missing_paper_ids),
            "judgment_checkpoints_current": checkpoint.current,
        }
        if (paths.answer.exists() or paths.submission.exists()) and not checkpoint.complete:
            raise ValueError(
                f"{query_id}: answer/submission exists with incomplete candidate "
                f"judgments; missing {len(checkpoint.missing_paper_ids)}: "
                f"{list(checkpoint.missing_paper_ids[:5])}"
            )
        if paths.submission.exists() and not paths.answer.exists():
            raise ValueError(
                f"{query_id}: per-query submission exists without answer checkpoint"
            )

        answer_is_current = False
        regenerated_submission: dict[str, Any] | None = None
        if paths.answer.exists():
            answer_record = json.loads(paths.answer.read_text(encoding="utf-8"))
            if not isinstance(answer_record, dict):
                raise ValueError(f"invalid answer checkpoint: {paths.answer}")
            expected_answer_key = reader.answer_cache_key(handoff.query, judgments)
            answer_is_current = bool(
                checkpoint.current and answer_record.get("cache_key") == expected_answer_key
            )
            trace["answer_checkpoint_current"] = answer_is_current
            if answer_is_current:
                trace["semantic_multiple_choice"] = answer_record.get(
                    "semantic_multiple_choice"
                )
                trace["completeness"] = answer_record.get("completeness")
                trace["prediction"] = answer_record.get("prediction")
                regenerated_submission = _submission_from_answer_checkpoint(
                    handoff.query, answer_record
                )

        if paths.submission.exists() and answer_is_current:
            submission = json.loads(paths.submission.read_text(encoding="utf-8"))
            if (
                not isinstance(submission, dict)
                or set(submission) != TOP_LEVEL_KEYS
                or submission.get("query_id") != query_id
            ):
                raise ValueError(f"invalid per-query submission: {paths.submission}")
            if submission != regenerated_submission:
                raise ValueError(
                    "stale/invalid per-query submission does not match answer "
                    f"checkpoint: {paths.submission}"
                )
            trace["submission"] = submission
            submissions.append(submission)
        traces.append(trace)

    atomic_write_jsonl(run_dir / "reading_traces.jsonl", traces)
    atomic_write_jsonl(run_dir / "submission.jsonl", submissions)
    return len(traces), len(submissions)
