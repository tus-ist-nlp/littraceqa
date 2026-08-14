from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from littraceqa.candidate_handoff import read_jsonl

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_validation_oracle_papers.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_validation_oracle_papers", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_write_new_jsonl = MODULE._write_new_jsonl
build_oracle_paper_records = MODULE.build_oracle_paper_records

OFFICIAL_VALIDATION_RELEASE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845"
)
OFFICIAL_VALIDATION_FILES = tuple(
    OFFICIAL_VALIDATION_RELEASE / "data" / name
    for name in (
        "validation.jsonl",
        "validation_inputs.jsonl",
        "paper_metadata.jsonl",
    )
)


def _queries():
    return [
        {
            "query_id": "q_001",
            "benchmark": "LitTraceQA",
            "question": "What value is reported?",
            "answer_types": ["freeform"],
        }
    ]


def _metadata():
    return [
        {"paper_id": "p1", "title": "One", "venue": "ACL", "year": 2025},
        {"paper_id": "p2", "title": "Two", "venue": "ACL", "year": 2025},
    ]


def test_oracle_builder_projects_only_sorted_paper_ids():
    gold = [
        {
            "query_id": "q_001",
            "gold_papers": [{"paper_id": "p2"}, {"paper_id": "p1"}],
            "answer": {"freeform": {"text": "SECRET_ANSWER"}},
            "evidence": [{"evidence_text_or_value": "SECRET_EVIDENCE"}],
            "task_family": "SECRET_TASK",
            "primary_evidence_type": "SECRET_TYPE",
        }
    ]

    output = build_oracle_paper_records(
        gold_records=gold,
        query_records=_queries(),
        metadata_records=_metadata(),
    )

    assert output == [
        {"query_id": "q_001", "candidate_papers": ["p1", "p2"]}
    ]
    serialized = json.dumps(output)
    for secret in (
        "SECRET_ANSWER",
        "SECRET_EVIDENCE",
        "SECRET_TASK",
        "SECRET_TYPE",
    ):
        assert secret not in serialized


def test_oracle_builder_rejects_non_official_query_fields():
    queries = _queries()
    queries[0]["task_family"] = "forbidden"

    with pytest.raises(ValueError, match="forbidden/unexpected"):
        build_oracle_paper_records(
            gold_records=[
                {"query_id": "q_001", "gold_papers": [{"paper_id": "p1"}]}
            ],
            query_records=queries,
            metadata_records=_metadata(),
        )


def test_oracle_builder_rejects_unknown_paper_and_query_mismatch():
    with pytest.raises(ValueError, match="unknown paper IDs"):
        build_oracle_paper_records(
            gold_records=[
                {"query_id": "q_001", "gold_papers": [{"paper_id": "missing"}]}
            ],
            query_records=_queries(),
            metadata_records=_metadata(),
        )

    with pytest.raises(ValueError, match="coverage mismatch"):
        build_oracle_paper_records(
            gold_records=[
                {"query_id": "q_extra", "gold_papers": [{"paper_id": "p1"}]}
            ],
            query_records=_queries(),
            metadata_records=_metadata(),
        )


def test_oracle_writer_never_overwrites_existing_file(tmp_path):
    output = tmp_path / "oracle.jsonl"
    output.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write_new_jsonl(
            output,
            [{"query_id": "q_001", "candidate_papers": ["p1"]}],
        )

    assert output.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.skipif(
    not all(path.is_file() for path in OFFICIAL_VALIDATION_FILES),
    reason="requires the external official-release validation artifacts",
)
def test_pinned_validation_oracle_projection_has_expected_safe_shape():
    validation, validation_inputs, paper_metadata = OFFICIAL_VALIDATION_FILES
    output = build_oracle_paper_records(
        gold_records=read_jsonl(validation),
        query_records=read_jsonl(validation_inputs),
        metadata_records=read_jsonl(paper_metadata),
    )

    assert len(output) == 55
    assert sum(len(item["candidate_papers"]) for item in output) == 146
    assert len(
        {
            paper_id
            for item in output
            for paper_id in item["candidate_papers"]
        }
    ) == 70
    assert all(set(item) == {"query_id", "candidate_papers"} for item in output)
    assert all(
        all(isinstance(paper_id, str) for paper_id in item["candidate_papers"])
        for item in output
    )
