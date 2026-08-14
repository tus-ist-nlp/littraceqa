from __future__ import annotations

import pytest

from littraceqa.aoai_pairwise_reader import (
    PairwiseAOAIReader,
    ReadingResponseError,
    _validate_explicit_row_identifier_surfaces,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query() -> Query:
    return Query(
        "synthetic_typo",
        "What are the scores of NCFM, AP-BPTT, ATT, and DEDA given IPC=10?",
        ["table"],
        table_schema=[
            {"name": "Methods", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "string", "is_row_key": False},
        ],
    )


def _records() -> dict[str, dict]:
    return {
        "p1#text": {
            "paper_id": "p1",
            "chunk_id": "p1#text",
            "chunk_type": "text_span",
            "text": (
                "[Venue 2025] Synthetic title\nWe compare our AT-BPTT "
                "with NCFM, ATT, and DEDA."
            ),
            "metadata": {"page": 4},
        }
    }


def _answer(method: str) -> dict:
    return {
        "table": {
            "rows": [
                {"Methods": "NCFM", "Score": "1"},
                {"Methods": method, "Score": "2"},
                {"Methods": "ATT", "Score": "3"},
                {"Methods": "DEDA", "Score": "4"},
            ]
        }
    }


def test_explicit_row_typo_is_rejected_in_favor_of_unique_source_identifier():
    with pytest.raises(
        ReadingResponseError,
        match="expected='AT-BPTT'.*actual='AP-BPTT'",
    ):
        _validate_explicit_row_identifier_surfaces(
            query=_query(),
            answer=_answer("AP-BPTT"),
            context_records=_records(),
        )


def test_source_canonical_explicit_row_identifier_is_accepted():
    _validate_explicit_row_identifier_surfaces(
        query=_query(),
        answer=_answer("AT-BPTT"),
        context_records=_records(),
    )


def test_punctuation_only_source_variant_preserves_query_row_surface():
    query = Query(
        "synthetic_punctuation",
        "What is the score of RISurConv?",
        ["table"],
        table_schema=[
            {"name": "Methods", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "string", "is_row_key": False},
        ],
    )
    records = {
        "p1#text": {
            "paper_id": "p1",
            "chunk_id": "p1#text",
            "chunk_type": "text_span",
            "text": "[Venue 2025] A title\nRISur-Conv scores 96.0.",
            "metadata": {"page": 4},
        }
    }
    _validate_explicit_row_identifier_surfaces(
        query=query,
        answer={"table": {"rows": [{"Methods": "RISurConv", "Score": "96.0"}]}},
        context_records=records,
    )


def test_explicit_row_identifier_repair_updates_only_identity(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("", encoding="utf-8")
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        paper_set_policy="fixed_selected",
    )
    prompt = reader._answer_locator_repair_prompt(
        query=_query(),
        original_prompt="ORIGINAL",
        rejected_response="{}",
        error=ReadingResponseError(
            "synthetic_typo: canonical explicit row identifier mismatch; "
            "expected='AT-BPTT', actual='AP-BPTT'"
        ),
        context_records={},
        repair_attempt=1,
    )
    assert "source-canonical row-identifier error" in prompt
    assert "Keep the requested value and evidence unchanged" in prompt
