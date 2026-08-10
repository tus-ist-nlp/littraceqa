from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.evaluation.evidence_coverage_input import (
    MissingPaperMetadataError,
    prepare_evidence_coverage,
)


def _query(question: str) -> Query:
    return Query(
        query_id="q",
        question=question,
        answer_types=["table"],
        table_schema=[
            {"name": "Paper Title", "type": "string", "is_row_key": True}
        ],
    )


def _citation_query() -> Query:
    return _query(
        "Which ACL 2024 papers cite BaseNet (Base Networks, ICML 2020) and "
        "use it as a baseline in their main comparison table?"
    )


def _write_metadata(path, *paper_ids: str) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "title": f"Title {paper_id}",
                    "authors": [],
                    "venue": "ACL",
                    "year": 2024,
                    "abstract": "A longer abstract for testing.",
                }
            )
            + "\n"
            for paper_id in paper_ids
        ),
        encoding="utf-8",
    )


def test_skips_metadata_io_when_no_question_needs_it(tmp_path):
    setup = prepare_evidence_coverage(
        tmp_path / "mineru",
        tmp_path / "missing.jsonl",
        {"q": _query("What does this paper report?")},
        {"q": ["p1"]},
    )

    assert setup.paper_metadata == {}


def test_loads_only_metadata_needed_by_supported_questions(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    _write_metadata(metadata, "p1", "p2", "unused")

    setup = prepare_evidence_coverage(
        tmp_path / "mineru",
        metadata,
        {"q": _citation_query()},
        {"q": ["p1", "p2"]},
        abstract_chars=8,
    )

    assert set(setup.paper_metadata) == {"p1", "p2"}
    assert setup.paper_metadata["p1"]["abstract"] == "A longe..."


def test_reports_the_first_missing_paper_id_deterministically(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    _write_metadata(metadata, "p1")

    with pytest.raises(MissingPaperMetadataError, match="missing p2"):
        prepare_evidence_coverage(
            tmp_path / "mineru",
            metadata,
            {"q": _citation_query()},
            {"q": ["p3", "p2", "p1"]},
        )
