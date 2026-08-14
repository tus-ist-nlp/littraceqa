"""Durable checkpoints and aggregate artifacts for pairwise reading runs."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from littraceqa.aoai_pairwise_reader import (
    FIXED_SELECTED_CHECKPOINT_KIND,
    FIXED_SELECTED_PAPER_POLICY,
    PairwiseAOAIReader,
    resolve_named_owner,
)
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
from littraceqa.submission import (
    TOP_LEVEL_KEYS,
    TOP_LEVEL_KEYS_WITHOUT_EVIDENCE,
    prediction_to_submission,
)

PROVIDER_ATTEMPT_LEDGER_VERSION = "provider-attempt-ledger-v1"


@dataclass(frozen=True)
class QueryRunPaths:
    """All per-query checkpoint paths derived from one safe query ID."""

    directory: Path
    judgments: Path
    answer: Path
    submission: Path
    answer_attempts: Path
    provider_attempts: Path
    errors: Path

    @classmethod
    def under(cls, run_dir: Path, query_id: str) -> QueryRunPaths:
        directory = run_dir / query_id
        return cls(
            directory=directory,
            judgments=directory / "candidate_judgments.jsonl",
            answer=directory / "answer.json",
            submission=directory / "submission.json",
            answer_attempts=directory / "answer_attempts.jsonl",
            provider_attempts=directory / "provider_attempts.jsonl",
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
def run_directory_lock(run_dir: Path) -> Iterator[None]:
    """Prevent two paid runner processes from mutating one run directory.

    The lock file intentionally remains on disk: ``flock`` state belongs to the
    open file descriptor and is released by the OS even after a crash.  Keeping
    the inode avoids an unlink/recreate race between competing processes.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner metadata unavailable"
            raise RuntimeError(
                f"run directory is already active: {run_dir} ({owner})"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                }
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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


def invalidate_aggregate_queries(
    run_dir: Path,
    query_ids: Iterable[str],
) -> None:
    """Remove pending queries from each root aggregate in one atomic rewrite.

    Per-query checkpoints remain durable. This prevents an interrupted force
    rejudgment or paid call from leaving old answers paired with new checkpoints
    in the root trace and submission files. Every aggregate is parsed at most
    once, regardless of the number of pending queries.
    """
    invalidated = {str(query_id) for query_id in query_ids if str(query_id)}
    if not invalidated:
        return
    # Remove the uploadable artifact first. If the process dies between the two
    # independent atomic rewrites, a stale diagnostic trace is safer than a stale
    # submission that still looks ready to upload.
    for filename in ("submission.jsonl", "reading_traces.jsonl"):
        path = run_dir / filename
        if not path.exists():
            continue
        records = read_jsonl(path)
        retained = [
            record
            for record in records
            if str(record.get("query_id") or "") not in invalidated
        ]
        if len(retained) != len(records):
            atomic_write_jsonl(path, retained)


def invalidate_aggregate_query(run_dir: Path, query_id: str) -> None:
    """Backward-compatible one-query aggregate invalidation wrapper."""

    invalidate_aggregate_queries(run_dir, (query_id,))


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
    details: dict[str, Any] | None = None,
) -> None:
    """Append one durable error record without corrupting the existing log."""
    records = read_jsonl(path) if path.exists() else []
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "query_id": query_id,
        "paper_id": paper_id,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if details is not None:
        record["details"] = details
    records.append(record)
    atomic_write_jsonl(path, records)


def record_answer_attempt(
    path: Path,
    *,
    query_id: str,
    attempt: dict[str, Any],
) -> None:
    """Durably append paid Stage-2 response metadata before final validation."""

    records = read_jsonl(path) if path.exists() else []
    records.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "query_id": query_id,
            "attempt_index": len(records) + 1,
            **attempt,
        }
    )
    atomic_write_jsonl(path, records)


def record_provider_attempt_event(
    path: Path,
    *,
    event: dict[str, Any],
) -> None:
    """Idempotently persist one PREPARE or FINALIZE provider event.

    Only the coordinator calls this function.  A PREPARE row is fsynced before
    the worker may enter the provider adapter; a FINALIZE row records either the
    received response metadata or a structured exception outcome.  Replaying
    the same immutable event is a no-op, while conflicting reuse of an event ID
    is rejected.
    """

    attempt_id = str(event.get("attempt_id") or "")
    event_kind = str(event.get("event_kind") or "")
    if not attempt_id or event_kind not in {"prepare", "finalize"}:
        raise ValueError("provider attempt event requires attempt_id and event_kind")
    event_id = f"{attempt_id}:{event_kind}"
    records = read_jsonl(path) if path.exists() else []
    existing = next(
        (record for record in records if record.get("event_id") == event_id),
        None,
    )
    payload = {
        "ledger_version": PROVIDER_ATTEMPT_LEDGER_VERSION,
        "event_id": event_id,
        **event,
    }
    if existing is not None:
        comparable = {
            key: value for key, value in existing.items() if key != "timestamp"
        }
        if comparable != payload:
            raise ValueError(f"conflicting provider attempt event: {event_id}")
        return
    records.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
    )
    atomic_write_jsonl(path, records)


def provider_attempt_summary(
    path: Path,
    *,
    stage: str | None = None,
    paper_id: str | None = None,
) -> dict[str, Any]:
    """Summarize unique provider attempts, including crash-uncertain prepares."""

    records = read_jsonl(path) if path.exists() else []
    filtered = [
        record
        for record in records
        if (stage is None or record.get("stage") == stage)
        and (paper_id is None or record.get("paper_id") == paper_id)
    ]
    by_attempt: dict[str, dict[str, dict[str, Any]]] = {}
    for record in filtered:
        attempt_id = str(record.get("attempt_id") or "")
        event_kind = str(record.get("event_kind") or "")
        if not attempt_id or event_kind not in {"prepare", "finalize"}:
            raise ValueError(f"invalid provider attempt ledger row: {path}")
        events = by_attempt.setdefault(attempt_id, {})
        if event_kind in events:
            raise ValueError(
                f"duplicate provider attempt {attempt_id}:{event_kind} in {path}"
            )
        events[event_kind] = record

    request_ids: list[str] = []
    usage: dict[str, int] = {}
    response_count = 0
    provider_error_count = 0
    uncertain_count = 0
    for attempt_id, events in by_attempt.items():
        final = events.get("finalize")
        if final is None:
            uncertain_count += 1
            continue
        outcome = final.get("outcome")
        if outcome == "response":
            response_count += 1
        elif outcome == "provider_error":
            provider_error_count += 1
        else:
            raise ValueError(
                f"invalid provider attempt outcome for {attempt_id}: {outcome!r}"
            )
        request_id = str(final.get("request_id") or "")
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)
        raw_usage = final.get("usage")
        if isinstance(raw_usage, dict):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            ):
                value = raw_usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value

    return {
        "ledger_version": PROVIDER_ATTEMPT_LEDGER_VERSION,
        # PREPARE is deliberately counted: after a crash it is impossible to
        # know whether the provider accepted the request. This upper-bound
        # accounting never silently under-reports a potentially billable call.
        "provider_invocation_count": len(by_attempt),
        "finalized_provider_invocation_count": len(by_attempt) - uncertain_count,
        "uncertain_provider_invocation_count": uncertain_count,
        "response_count": response_count,
        "provider_error_count": provider_error_count,
        "attempt_ids": list(by_attempt),
        "request_ids": request_ids,
        "usage": usage,
    }


def _aggregate_provider_summaries(
    summaries: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Combine per-query summaries into one materialized run accounting row."""

    by_query: dict[str, dict[str, Any]] = {}
    totals = {
        "provider_invocation_count": 0,
        "finalized_provider_invocation_count": 0,
        "uncertain_provider_invocation_count": 0,
        "response_count": 0,
        "provider_error_count": 0,
    }
    usage: dict[str, int] = {}
    request_ids: list[str] = []
    for query_id, summary in summaries:
        by_query[query_id] = summary
        for key in totals:
            totals[key] += int(summary.get(key) or 0)
        for key, value in (summary.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
        for request_id in summary.get("request_ids") or []:
            if request_id not in request_ids:
                request_ids.append(request_id)
    return {
        "ledger_version": PROVIDER_ATTEMPT_LEDGER_VERSION,
        **totals,
        "request_ids": request_ids,
        "usage": usage,
        "queries": by_query,
    }


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
    owner_resolution = resolve_named_owner(
        handoff.query,
        handoff.candidate_papers,
    )
    for candidate in handoff.candidate_papers:
        judgment = judgments.get(candidate.paper_id)
        if judgment is None:
            continue
        records = reader.chunk_store.load_paper(candidate.paper_id)
        if getattr(reader, "supports_named_owner_resolution", False):
            expected_key = reader.judgment_cache_key(
                handoff.query,
                candidate,
                records,
                owner_resolution=owner_resolution,
            )
        else:
            expected_key = reader.judgment_cache_key(
                handoff.query, candidate, records
            )
        if judgment.get("cache_key") != expected_key:
            stale_ids.append(candidate.paper_id)
            continue
        if not _current_judgment_invariants_hold(
            candidate=candidate,
            records=records,
            judgment=judgment,
            expected_checkpoint_kind=(
                FIXED_SELECTED_CHECKPOINT_KIND
                if getattr(reader, "paper_set_policy", None)
                == FIXED_SELECTED_PAPER_POLICY
                else None
            ),
        ):
            stale_ids.append(candidate.paper_id)
    return JudgmentCheckpointStatus(
        missing_paper_ids=missing_ids,
        stale_paper_ids=tuple(stale_ids),
    )


def _current_judgment_invariants_hold(
    *,
    candidate: CandidatePaper,
    records: list[dict[str, Any]],
    judgment: dict[str, Any],
    expected_checkpoint_kind: str | None = None,
) -> bool:
    """Fail closed on edited or internally inconsistent current checkpoints."""

    if judgment.get("paper_id") != candidate.paper_id:
        return False
    relevant = judgment.get("is_relevant_to_answer")
    usable = judgment.get("has_usable_answer_evidence")
    send = judgment.get("send_to_answer_agent")
    if not all(isinstance(value, bool) for value in (relevant, usable, send)):
        return False
    chunk_ids = judgment.get("evidence_chunk_ids")
    if (
        not isinstance(chunk_ids, list)
        or any(not isinstance(value, str) or not value for value in chunk_ids)
        or len(chunk_ids) != len(set(chunk_ids))
    ):
        return False
    if send is not bool(relevant and usable and chunk_ids):
        return False
    if usable is not send:
        return False
    checkpoint_kind = judgment.get("checkpoint_kind")
    if checkpoint_kind != expected_checkpoint_kind:
        return False
    fixed_selected = checkpoint_kind == FIXED_SELECTED_CHECKPOINT_KIND
    if fixed_selected and relevant is not True:
        return False

    record_ids = {
        str(record.get("chunk_id") or "")
        for record in records
        if str(record.get("chunk_id") or "")
    }
    context_ids = judgment.get("context_chunk_ids")
    if not isinstance(context_ids, list) or any(
        not isinstance(value, str) for value in context_ids
    ):
        return False
    if not set(chunk_ids).issubset(record_ids.intersection(context_ids)):
        return False

    evidence = judgment.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, dict) for item in evidence
    ):
        return False
    expanded_ids = [str(item.get("chunk_id") or "") for item in evidence]
    if expanded_ids != chunk_ids:
        return False
    if fixed_selected:
        extracted_facts = judgment.get("extracted_facts")
        if not isinstance(extracted_facts, list) or any(
            not isinstance(item, dict)
            or set(item) != {"chunk_id", "purpose", "fact", "source_excerpt"}
            or not str(item.get("chunk_id") or "")
            or not str(item.get("fact") or "")
            for item in extracted_facts
        ):
            return False
        extracted_ids = {
            str(item["chunk_id"]) for item in extracted_facts
        }
        if extracted_ids != set(chunk_ids):
            return False

    attached_ids = judgment.get("attached_image_chunk_ids")
    attached_count = judgment.get("attached_image_count")
    if (
        not isinstance(attached_ids, list)
        or any(not isinstance(value, str) for value in attached_ids)
        or isinstance(attached_count, bool)
        or not isinstance(attached_count, int)
        or attached_count < 0
        or not set(attached_ids).issubset(set(context_ids))
    ):
        return False
    return True


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
    query: Query,
    answer_record: dict[str, Any],
    *,
    require_evidence: bool = True,
    authoritative_paper_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    prediction = _prediction_from_checkpoint(
        answer_record.get("prediction"), query_id=query.query_id
    )
    submission_ids = answer_record.get("submission_paper_ids")
    stage1_relevant_ids = answer_record.get("stage1_relevant_paper_ids")
    fixed_selected = answer_record.get("paper_set_policy") == "fixed_selected"
    authoritative_ids = submission_ids if submission_ids is not None else stage1_relevant_ids
    if (
        not isinstance(authoritative_ids, list)
        or any(
            not isinstance(paper_id, str) or not paper_id
            for paper_id in authoritative_ids
        )
        or len(authoritative_ids) != len(set(authoritative_ids))
    ):
        raise ValueError(
            f"{query.query_id}: answer checkpoint has invalid authoritative "
            "submission paper IDs"
        )
    if fixed_selected and submission_ids is None:
        raise ValueError(
            f"{query.query_id}: fixed-selected answer checkpoint has no "
            "submission_paper_ids"
        )
    if fixed_selected:
        if authoritative_paper_ids is None:
            raise ValueError(
                f"{query.query_id}: fixed-selected checkpoint validation requires "
                "the externally selected candidate papers"
            )
        selected_ids = list(authoritative_paper_ids)
        if authoritative_ids != selected_ids:
            raise ValueError(
                f"{query.query_id}: answer checkpoint papers do not match the "
                "externally selected candidate papers"
            )
    prediction_relevant_ids = [
        str(item.get("paper_id") or "") for item in prediction.gold_papers
    ]
    if prediction_relevant_ids != authoritative_ids:
        raise ValueError(
            f"{query.query_id}: answer checkpoint authoritative papers do not "
            "match prediction.gold_papers"
        )
    accepted_ids = answer_record.get("accepted_paper_ids")
    if (
        not isinstance(accepted_ids, list)
        or any(not isinstance(paper_id, str) or not paper_id for paper_id in accepted_ids)
        or (
            fixed_selected
            and not set(accepted_ids).issubset(set(authoritative_ids))
        )
        or (
            not fixed_selected
            and not set(authoritative_ids).issubset(set(accepted_ids))
        )
    ):
        raise ValueError(
            f"{query.query_id}: answer checkpoint has invalid Stage-2 handoff papers"
        )
    return prediction_to_submission(
        query, prediction, require_evidence=require_evidence
    )


def ensure_submission_from_answer_checkpoint(
    query: Query,
    answer_record: dict[str, Any],
    submission_path: Path,
    *,
    require_evidence: bool = True,
    authoritative_paper_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate or atomically restore a submission from a current answer.

    Callers must first prove that ``answer_record`` has the expected cache key.
    Keeping the restoration here gives both aggregate materialization and the
    paid-call runner the same fail-closed validation path.
    """

    if answer_record.get("query_id") != query.query_id:
        raise ValueError(
            f"{query.query_id}: answer checkpoint query_id mismatch: "
            f"{submission_path.parent / 'answer.json'}"
        )
    if answer_record.get("status") != "complete":
        raise ValueError(
            f"{query.query_id}: answer checkpoint status is not complete: "
            f"{submission_path.parent / 'answer.json'}"
        )

    regenerated = _submission_from_answer_checkpoint(
        query,
        answer_record,
        require_evidence=require_evidence,
        authoritative_paper_ids=authoritative_paper_ids,
    )
    if submission_path.exists():
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
    else:
        # ``answer.json`` is written before ``submission.json``. An interruption
        # between those two atomic writes is recoverable without another LLM call.
        submission = regenerated
        atomic_write_json(submission_path, submission)

    allowed_top_level_keys = {TOP_LEVEL_KEYS}
    if not require_evidence:
        allowed_top_level_keys.add(TOP_LEVEL_KEYS_WITHOUT_EVIDENCE)
    if (
        not isinstance(submission, dict)
        or frozenset(submission) not in allowed_top_level_keys
        or submission.get("query_id") != query.query_id
    ):
        raise ValueError(f"invalid per-query submission: {submission_path}")
    if submission != regenerated:
        raise ValueError(
            "stale/invalid per-query submission does not match answer "
            f"checkpoint: {submission_path}"
        )
    return submission


def materialize_run_outputs(
    run_dir: Path,
    handoffs: list[CandidateHandoff],
    reader: PairwiseAOAIReader,
    *,
    require_evidence: bool = True,
) -> tuple[int, int]:
    """Rebuild aggregate traces/submissions only from valid checkpoints."""
    traces: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    provider_summaries: list[tuple[str, dict[str, Any]]] = []
    provider_stage_summaries: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "judge": [],
        "answer": [],
    }
    for handoff in handoffs:
        query_id = handoff.query.query_id
        paths = QueryRunPaths.under(run_dir, query_id)
        provider_summary = provider_attempt_summary(paths.provider_attempts)
        provider_summaries.append((query_id, provider_summary))
        for provider_stage, stage_summaries in provider_stage_summaries.items():
            stage_summaries.append(
                (
                    query_id,
                    provider_attempt_summary(
                        paths.provider_attempts, stage=provider_stage
                    ),
                )
            )
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
        trace["provider_attempts"] = {
            "all": provider_summary,
            "judge": provider_attempt_summary(
                paths.provider_attempts, stage="judge"
            ),
            "answer": provider_attempt_summary(
                paths.provider_attempts, stage="answer"
            ),
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
        answer_record: dict[str, Any] | None = None
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

        if answer_is_current:
            if answer_record is None:
                raise AssertionError(
                    f"{query_id}: current answer checkpoint was not loaded"
                )
            submission = ensure_submission_from_answer_checkpoint(
                handoff.query,
                answer_record,
                paths.submission,
                require_evidence=require_evidence,
                authoritative_paper_ids=(
                    candidate.paper_id for candidate in handoff.candidate_papers
                ),
            )
            trace["submission"] = submission
            submissions.append(submission)
        traces.append(trace)

    atomic_write_jsonl(run_dir / "reading_traces.jsonl", traces)
    atomic_write_jsonl(run_dir / "submission.jsonl", submissions)
    run_provider_summary = _aggregate_provider_summaries(provider_summaries)
    run_provider_summary["stages"] = {
        stage: _aggregate_provider_summaries(stage_summaries)
        for stage, stage_summaries in provider_stage_summaries.items()
    }
    atomic_write_json(run_dir / "provider_usage_summary.json", run_provider_summary)
    return len(traces), len(submissions)
