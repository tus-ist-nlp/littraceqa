from __future__ import annotations

import pytest

from littraceqa.aoai_pairwise_reader import (
    PairwiseAOAIReader,
    ReadingResponseError,
    _canonical_scaling_base_model,
    _canonical_scaling_method,
    _source_identifier_surface,
    _source_specific_base_model,
    _validate_scaling_identifier_rows,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query() -> Query:
    return Query(
        "synthetic_scaling",
        "Which inference-time scaling methods use which base models?",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Base Model", "type": "string", "is_row_key": True},
        ],
    )


def _derivation(method: str, base: str) -> dict:
    return {
        "facts": [
            {
                "id": "f_method",
                "value": method,
                "paper_id": "p1",
                "chunk_ids": ["p1#text"],
            },
            {
                "id": "f_base",
                "value": base,
                "paper_id": "p1",
                "chunk_ids": ["p1#text"],
            },
        ],
        "operations": [],
        "answer_bindings": [
            {
                "answer_path": "answer.table.rows[0].Method",
                "source_type": "fact",
                "source_id": "f_method",
            },
            {
                "answer_path": "answer.table.rows[0].Base Model",
                "source_type": "fact",
                "source_id": "f_base",
            },
        ],
    }


def test_scaling_base_model_canonicalization_is_narrow_and_deterministic():
    assert _canonical_scaling_base_model("Infinity2B") == "Infinity-2B"
    assert (
        _canonical_scaling_base_model("Infinity-2B and Infinity-8B")
        == "Infinity-2B/8B"
    )
    assert (
        _canonical_scaling_base_model("SANA-1.5 4.8B v2 model")
        == "SANA-1.5 4.8B v2"
    )
    assert _canonical_scaling_base_model("World Model") == "World Model"
    assert (
        _canonical_scaling_method("SANA-1.5 + Inference Scaling")
        == "SANA-1.5"
    )
    assert _canonical_scaling_method("Reflect-DiT") == "Reflect-DiT"


def test_method_surface_comes_from_cited_body_not_synthetic_title_prefix():
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": "[Venue 2025] SANA 1.5: A title\nWe run SANA-1.5 on Canvas-3B.",
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    assert _source_identifier_surface("SANA 1.5", [record]) == "SANA-1.5"


def test_generic_base_family_recovers_one_unambiguous_source_size():
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": "[Venue 2025] A title\nExperiments use Infinity2B on GenEval.",
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    assert _source_specific_base_model("Infinity", [record]) == "Infinity-2B"


def test_versioned_family_preserves_source_space_size_and_version():
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": (
            "[Venue 2025] A title\nInference scaling uses the "
            "SANA-1.5 4.8B v2 model."
        ),
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    assert (
        _source_specific_base_model("SANA-1.5", [record])
        == "SANA-1.5 4.8B v2"
    )


def test_family_only_recovers_cointroduced_sizes_as_one_slash_cell():
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": (
            "[Venue 2025] A title\nWe evaluated the method on "
            "Infinity-2B and Infinity-8B."
        ),
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    assert (
        _source_specific_base_model("Infinity", [record])
        == "Infinity-2B/8B"
    )


def test_scaling_row_validator_reports_all_canonical_surface_repairs():
    answer = {
        "table": {
            "rows": [
                {"Method": "SANA 1.5", "Base Model": "Infinity2B"},
            ]
        }
    }
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": "[Venue 2025] SANA 1.5: A title\nSANA-1.5 uses Infinity2B.",
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    with pytest.raises(ReadingResponseError) as caught:
        _validate_scaling_identifier_rows(
            query=_query(),
            answer=answer,
            derivation=_derivation("SANA 1.5", "Infinity2B"),
            context_records={"p1#text": record},
        )
    message = str(caught.value)
    assert "canonical scaling identifier mismatch" in message
    assert "expected='SANA-1.5'" in message
    assert "expected='Infinity-2B'" in message


def test_scaling_row_validator_rejects_family_only_when_source_has_one_size():
    answer = {"table": {"rows": [{"Method": "TTS-VAR", "Base Model": "Infinity"}]}}
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#text",
        "text": "[Venue 2025] TTS-VAR: A title\nTTS-VAR uses Infinity2B on GenEval.",
        "chunk_type": "text_span",
        "metadata": {"page": 1},
    }
    with pytest.raises(ReadingResponseError, match="expected='Infinity-2B'"):
        _validate_scaling_identifier_rows(
            query=_query(),
            answer=answer,
            derivation=_derivation("TTS-VAR", "Infinity"),
            context_records={"p1#text": record},
        )


def test_scaling_identifier_repair_prompt_updates_facts_and_all_outputs(tmp_path):
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
            "synthetic_scaling: canonical scaling identifier mismatch; "
            "expected='Infinity-2B', actual='Infinity2B'"
        ),
        context_records={},
        repair_attempt=1,
    )
    assert "deterministic scaling-identifier surface error" in prompt
    assert "fact.value, table cell, freeform substring" in prompt
