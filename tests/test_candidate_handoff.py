from __future__ import annotations

import json
from pathlib import Path

import pytest

from littraceqa.candidate_handoff import (
    FORBIDDEN_CANDIDATE_FIELDS,
    candidate_papers_from_record,
    load_candidate_handoffs,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _candidate_record(**extra) -> dict:
    record = {
        "query_id": "q1",
        "candidate_papers": [
            {
                "rank": 1,
                "paper_id": "p1",
                "title": "Paper One",
                "venue": "ACL",
                "year": 2025,
            }
        ],
    }
    record.update(extra)
    return record


def test_loader_projects_query_to_exact_production_fields(tmp_path):
    sentinel = "GOLD_SENTINEL_MUST_NOT_LEAK"
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "What value is reported?",
                "answer_types": ["freeform"],
                "table_schema": None,
                "task_family": "multi_paper",
                "primary_evidence_type": "table",
                "options": {"A": sentinel},
                "_gold": {"answer": sentinel},
            }
        ],
    )
    _write_jsonl(candidates, [_candidate_record()])

    handoff = load_candidate_handoffs(queries, candidates)[0]

    assert handoff.query.query_id == "q1"
    assert handoff.query.options is None
    assert handoff.query.task_family is None
    assert handoff.query.primary_evidence_type is None
    assert sentinel not in json.dumps(handoff.query.to_dict())


@pytest.mark.parametrize(
    "oracle_field",
    ["_gold", "answer", "evidence", "gold_papers", "task_family", "options"],
)
def test_candidate_sidecar_rejects_oracle_fields(oracle_field):
    with pytest.raises(ValueError, match="oracle/development"):
        candidate_papers_from_record(_candidate_record(**{oracle_field: {}}))


def test_candidate_sidecar_rejects_query_payload():
    with pytest.raises(ValueError, match="separate sidecar"):
        candidate_papers_from_record(_candidate_record(question="must not be here"))


def test_candidate_ranks_must_be_consecutive():
    record = _candidate_record()
    record["candidate_papers"][0]["rank"] = 2
    with pytest.raises(ValueError, match="consecutive"):
        candidate_papers_from_record(record)


def test_candidate_ids_must_be_unique():
    record = _candidate_record()
    record["candidate_papers"].append(dict(record["candidate_papers"][0], rank=2))
    with pytest.raises(ValueError, match="duplicate candidate"):
        candidate_papers_from_record(record)


def test_candidate_metadata_is_checked(tmp_path):
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "Question",
                "answer_types": ["freeform"],
                "table_schema": None,
            }
        ],
    )
    _write_jsonl(candidates, [_candidate_record()])
    _write_jsonl(
        metadata,
        [{"paper_id": "p1", "title": "Different", "venue": "ACL", "year": 2025}],
    )
    with pytest.raises(ValueError, match="mismatched title"):
        load_candidate_handoffs(queries, candidates, metadata)


def test_checked_in_validation_candidate_handoff_is_complete_and_canonical():
    records = read_jsonl(ROOT / "data/validation_candidates.jsonl")

    assert len(records) == 55
    query_ids = [str(record.get("query_id") or "") for record in records]
    assert all(query_ids)
    assert len(set(query_ids)) == len(query_ids)

    candidate_ids = {
        str(candidate.get("paper_id") or "")
        for record in records
        for candidate in record.get("candidate_papers", [])
        if isinstance(candidate, dict)
    }
    canonical_metadata = {}
    with (ROOT / "data/paper_metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metadata = json.loads(line)
            paper_id = str(metadata.get("paper_id") or "")
            if paper_id in candidate_ids:
                assert paper_id not in canonical_metadata
                canonical_metadata[paper_id] = metadata
    assert set(canonical_metadata) == candidate_ids

    candidate_schema = {"rank", "paper_id", "title", "venue", "year"}
    for record in records:
        assert set(record) == {"query_id", "candidate_papers"}
        assert not FORBIDDEN_CANDIDATE_FIELDS.intersection(record)
        raw_candidates = record["candidate_papers"]
        assert 3 <= len(raw_candidates) <= 50
        assert all(
            isinstance(candidate, dict) and set(candidate) == candidate_schema
            for candidate in raw_candidates
        )

        parsed_candidates = candidate_papers_from_record(record)
        assert [candidate.rank for candidate in parsed_candidates] == list(
            range(1, len(parsed_candidates) + 1)
        )
        for candidate in parsed_candidates:
            canonical = canonical_metadata[candidate.paper_id]
            assert candidate.title == str(canonical.get("title") or "")
            assert candidate.venue == str(canonical.get("venue") or "")
            assert candidate.year == canonical.get("year")
