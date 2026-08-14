from __future__ import annotations

import json

import pytest

from littraceqa.aoai_pairwise_reader import (
    PairwiseAOAIReader,
    ReadingResponseError,
    _required_singleton_eligibility_chunk_ids,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _reader_and_records(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#constraint",
            "chunk_type": "text_span",
            "text": "Flint is trained only on BaseSet.",
            "metadata": {"page": 5, "section": "Experiments"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#generic",
            "chunk_type": "text_span",
            "text": "Flint is a vision-language system.",
            "metadata": {"page": 2, "section": "Introduction"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": "Flint | 74",
            "metadata": {"page": 6, "table_id": "Table 1"},
        },
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        paper_set_policy="fixed_selected",
        answer_neighbor_chunks=0,
    )
    return reader, {record["chunk_id"]: record for record in records}


def _query() -> Query:
    return Query(
        "filtered_maximum",
        "Which system trained only on BaseSet has the highest score?",
        ["freeform", "multiple_choice"],
        options={"A": "Cedar", "B": "Flint"},
    )


def _payload(*, include_eligibility: bool) -> dict[str, object]:
    evidence_ids = ["p1#table"]
    facts: list[dict[str, object]] = [
        {
            "id": "f_score",
            "name": "eligible score candidate",
            "value": {"label": "Flint", "value": 74},
            "value_kind": "reported",
            "paper_id": "p1",
            "chunk_ids": ["p1#table"],
        }
    ]
    if include_eligibility:
        evidence_ids.insert(0, "p1#constraint")
        facts.append(
            {
                "id": "f_eligibility",
                "name": "explicit only-filter condition",
                "value": "trained only on BaseSet",
                "value_kind": "text",
                "paper_id": "p1",
                "chunk_ids": ["p1#constraint"],
            }
        )
    operation = {
        "id": "op_best",
        "kind": "argmax",
        "fact_ids": ["f_score"],
        "candidates": [{"label": "Flint", "value": 74}],
        "result": "Flint",
        "answer_binding": {
            "answer_path": "answer.multiple_choice",
            "expected": "Flint",
            "answer_fragment": "Flint",
        },
    }
    return {
        "status": "ready",
        "paper_relevance": [
            {"paper_id": "p1", "role": "answer_source", "reason": "owner"}
        ],
        "papers": [{"paper_id": "p1", "evidence_chunk_ids": evidence_ids}],
        "derivation": {
            "facts": facts,
            "operations": [operation],
            "answer_bindings": [
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "operation",
                    "source_id": "op_best",
                    "answer_fragment": "Flint",
                },
                {
                    "answer_path": "answer.multiple_choice",
                    "source_type": "operation",
                    "source_id": "op_best",
                    "answer_fragment": "Flint",
                },
            ],
            "final_semantic_answer": "Flint",
        },
        "answer": {
            "freeform": {"text": "Flint"},
            "multiple_choice": {"label": "B", "selected_option_text": "Flint"},
        },
        "support": [
            {
                "answer_path": "answer.freeform.text",
                "paper_id": "p1",
                "chunk_ids": evidence_ids,
            },
            {
                "answer_path": "answer.multiple_choice",
                "paper_id": "p1",
                "chunk_ids": evidence_ids,
            },
        ],
        "completeness": {"answered_parts": ["filtered maximum"], "missing": []},
    }


def test_filtered_singleton_requires_explicit_eligibility_evidence(tmp_path):
    reader, records = _reader_and_records(tmp_path)

    with pytest.raises(
        ReadingResponseError,
        match="filtered-singleton eligibility evidence",
    ):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(_payload(include_eligibility=False)),
            relevant_paper_ids={"p1"},
            context_records=records,
            required_singleton_eligibility_chunk_ids={"p1#constraint"},
        )


def test_filtered_singleton_keeps_constraint_fact_separate_from_argmax(tmp_path):
    reader, records = _reader_and_records(tmp_path)

    parsed = reader._parse_answer(
        query=_query(),
        payload_text=json.dumps(_payload(include_eligibility=True)),
        relevant_paper_ids={"p1"},
        context_records=records,
        required_singleton_eligibility_chunk_ids={"p1#constraint"},
    )

    assert parsed["answer"]["multiple_choice"] == {
        "label": "B",
        "selected_option_text": "Flint",
    }
    assert {fact["id"] for fact in parsed["derivation"]["facts"]} == {
        "f_score",
        "f_eligibility",
    }


def test_filtered_singleton_repair_says_not_to_delete_constraint_fact(tmp_path):
    reader, records = _reader_and_records(tmp_path)
    error = ReadingResponseError(
        "filtered-singleton eligibility evidence is missing from derivation "
        "facts and submitted evidence; preserve at least one explicit "
        "hard-condition fact from chunk_ids=['p1#constraint']"
    )

    prompt = reader._answer_locator_repair_prompt(
        query=_query(),
        original_prompt="ORIGINAL",
        rejected_response=json.dumps(_payload(include_eligibility=False)),
        error=error,
        context_records=records,
        repair_attempt=1,
    )

    assert "explicit eligibility-filter evidence error" in prompt
    assert "Do not delete the eligibility fact" in prompt
    assert "do not put its text value into operation.candidates" in prompt


def test_filtered_singleton_does_not_accept_a_different_generic_condition(tmp_path):
    reader, records = _reader_and_records(tmp_path)
    payload = _payload(include_eligibility=False)
    payload["papers"][0]["evidence_chunk_ids"].append("p1#generic")
    payload["derivation"]["facts"].append(
        {
            "id": "f_generic",
            "name": "broad method type",
            "value": "vision-language system",
            "value_kind": "text",
            "paper_id": "p1",
            "chunk_ids": ["p1#generic"],
        }
    )
    for support in payload["support"]:
        support["chunk_ids"].append("p1#generic")

    with pytest.raises(
        ReadingResponseError,
        match="filtered-singleton eligibility evidence",
    ):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(payload),
            relevant_paper_ids={"p1"},
            context_records=records,
            required_singleton_eligibility_chunk_ids={"p1#constraint"},
        )


def test_filtered_singleton_condition_must_be_separate_from_extremum(tmp_path):
    reader, records = _reader_and_records(tmp_path)
    payload = _payload(include_eligibility=True)
    eligibility = payload["derivation"]["facts"][-1]
    eligibility["value"] = {"label": "BaseSet filter", "value": 1}
    eligibility["value_kind"] = "reported"
    payload["derivation"]["operations"][0]["fact_ids"].append("f_eligibility")
    payload["derivation"]["operations"][0]["candidates"].append(
        {"label": "BaseSet filter", "value": 1}
    )

    with pytest.raises(
        ReadingResponseError,
        match="filtered-singleton eligibility evidence",
    ):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(payload),
            relevant_paper_ids={"p1"},
            context_records=records,
            required_singleton_eligibility_chunk_ids={"p1#constraint"},
        )


def test_filtered_singleton_rejects_unused_background_evidence(tmp_path):
    reader, records = _reader_and_records(tmp_path)
    payload = _payload(include_eligibility=True)
    payload["papers"][0]["evidence_chunk_ids"].append("p1#generic")
    payload["derivation"]["facts"].append(
        {
            "id": "f_unused_identity",
            "name": "unused broad identity",
            "value": "vision-language system",
            "value_kind": "text",
            "paper_id": "p1",
            "chunk_ids": ["p1#generic"],
        }
    )
    for support in payload["support"]:
        support["chunk_ids"].append("p1#generic")

    with pytest.raises(
        ReadingResponseError,
        match="unused identity/background facts",
    ):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(payload),
            relevant_paper_ids={"p1"},
            context_records=records,
            required_singleton_eligibility_chunk_ids={"p1#constraint"},
        )


def test_filtered_singleton_selector_ignores_free_form_fact_claims():
    judgments = [
        {
            "extracted_facts": [
                {
                    "chunk_id": "p1#constraint",
                    "purpose": "eligibility_condition",
                    "fact": "generic paraphrase",
                    "source_excerpt": "Flint is trained only on BaseSet.",
                },
                {
                    "chunk_id": "p1#generic",
                    "purpose": "eligibility_condition",
                    "fact": "Flint is trained only on BaseSet.",
                    "source_excerpt": "Flint is a vision-language system.",
                },
            ]
        }
    ]
    assert _required_singleton_eligibility_chunk_ids(_query(), judgments) == {
        "p1#constraint"
    }


def test_filtered_singleton_tied_required_chunks_are_all_mandatory(tmp_path):
    reader, records = _reader_and_records(tmp_path)
    with pytest.raises(
        ReadingResponseError,
        match="missing_fact_chunks=.*p1#generic",
    ):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(_payload(include_eligibility=True)),
            relevant_paper_ids={"p1"},
            context_records=records,
            required_singleton_eligibility_chunk_ids={
                "p1#constraint",
                "p1#generic",
            },
        )
