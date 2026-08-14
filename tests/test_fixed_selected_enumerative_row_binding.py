from __future__ import annotations

import json

import pytest

from littraceqa.aoai_pairwise_reader import PairwiseAOAIReader, ReadingResponseError
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query(*, explicit_rows: bool = False) -> Query:
    question = (
        "Report a row for Paper One and Paper Two?"
        if explicit_rows
        else "Which selected papers satisfy the requested property?"
    )
    return Query.from_dict(
        {
            "query_id": "q_table",
            "benchmark": "LitTraceQA",
            "question": question,
            "answer_types": ["table"],
            "table_schema": [
                {"name": "Paper Title", "type": "string", "is_row_key": True}
            ],
        }
    )


def _reader(tmp_path) -> PairwiseAOAIReader:
    corpus = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": paper_id,
            "chunk_id": f"{paper_id}#text",
            "chunk_type": "text_span",
            "text": text,
            "metadata": {"page": index},
        }
        for index, (paper_id, text) in enumerate(
            (("p1", "Paper One qualifies."), ("p2", "Paper Two does not qualify.")),
            start=1,
        )
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        paper_set_policy="fixed_selected",
        answer_neighbor_chunks=0,
    )


def _fact(fact_id: str, paper_id: str, value: str) -> dict[str, object]:
    return {
        "id": fact_id,
        "name": f"{paper_id} result",
        "value": value,
        "value_kind": "text",
        "paper_id": paper_id,
        "chunk_ids": [f"{paper_id}#text"],
    }


def _payload(*, bind_second_paper: bool) -> dict[str, object]:
    rows = [{"Paper Title": "Paper One"}]
    bindings = [
        {
            "answer_path": "answer.table.rows[0].Paper Title",
            "source_type": "fact",
            "source_id": "f1",
            "answer_fragment": "Paper One",
        }
    ]
    if bind_second_paper:
        rows.append({"Paper Title": "Paper Two"})
        bindings.append(
            {
                "answer_path": "answer.table.rows[1].Paper Title",
                "source_type": "fact",
                "source_id": "f2",
                "answer_fragment": "Paper Two",
            }
        )
        second_value = "Paper Two"
    else:
        # This is the loophole: p2 has source-grounded negative information and
        # a syntactically valid support path, but contributes no emitted row.
        second_value = "Paper Two does not qualify"

    return {
        "status": "ready",
        "paper_relevance": [
            {"paper_id": "p1", "role": "answer_source", "reason": "row"},
            {"paper_id": "p2", "role": "constraint_source", "reason": "checked"},
        ],
        "papers": [
            {"paper_id": "p1", "evidence_chunk_ids": ["p1#text"]},
            {"paper_id": "p2", "evidence_chunk_ids": ["p2#text"]},
        ],
        "derivation": {
            "facts": [
                _fact("f1", "p1", "Paper One"),
                _fact("f2", "p2", second_value),
            ],
            "operations": [],
            "answer_bindings": bindings,
            "final_semantic_answer": ", ".join(
                row["Paper Title"] for row in rows
            ),
        },
        "answer": {"table": {"rows": rows}},
        "support": [
            {
                "answer_path": "answer.table.rows[0]",
                "paper_id": "p1",
                "chunk_ids": ["p1#text"],
            },
            {
                # Existing support validation accepts this emitted path for p2,
                # but it cannot prove that p2 contributed the row's value.
                "answer_path": (
                    "answer.table.rows[1]"
                    if bind_second_paper
                    else "answer.table.rows[0]"
                ),
                "paper_id": "p2",
                "chunk_ids": ["p2#text"],
            },
        ],
        "completeness": {"answered_parts": ["qualifying papers"], "missing": []},
    }


def _parse(
    reader: PairwiseAOAIReader,
    query: Query,
    payload: dict[str, object],
    *,
    canonical_paper_titles: dict[str, str] | None = None,
):
    return reader._parse_answer(
        query=query,
        payload_text=json.dumps(payload),
        relevant_paper_ids={"p1", "p2"},
        context_records={
            record["chunk_id"]: record
            for paper_id in ("p1", "p2")
            for record in reader.chunk_store.load_paper(paper_id)
        },
        canonical_paper_titles=canonical_paper_titles,
    )


def test_open_ended_multi_paper_table_rejects_unbound_negative_paper_fact(tmp_path):
    reader = _reader(tmp_path)

    with pytest.raises(ReadingResponseError, match="missing_row_bindings=\\['p2'\\]"):
        _parse(reader, _query(), _payload(bind_second_paper=False))


def test_open_ended_multi_paper_table_accepts_bound_row_from_every_paper(tmp_path):
    reader = _reader(tmp_path)

    parsed = _parse(reader, _query(), _payload(bind_second_paper=True))

    assert parsed["answer"]["table"]["rows"] == [
        {"Paper Title": "Paper One"},
        {"Paper Title": "Paper Two"},
    ]


def test_explicit_named_rows_are_exempt_from_per_paper_direct_row_binding(tmp_path):
    reader = _reader(tmp_path)
    payload = _payload(bind_second_paper=False)
    payload["completeness"] = {
        "answered_parts": ["Paper One"],
        "missing": ["Paper Two"],
    }

    parsed = _parse(reader, _query(explicit_rows=True), payload)

    assert parsed["completeness"]["missing"] == ["Paper Two"]


def test_each_method_table_rejects_duplicate_method_variant_rows(tmp_path):
    reader = _reader(tmp_path)
    query = Query.from_dict(
        {
            "query_id": "q_methods",
            "benchmark": "LitTraceQA",
            "question": "What base model does each method use?",
            "answer_types": ["table"],
            "table_schema": [
                {"name": "Method", "type": "string", "is_row_key": True},
                {"name": "Base Model", "type": "string", "is_row_key": True},
            ],
        }
    )
    payload = _payload(bind_second_paper=True)
    rows = [
        {"Method": "Shared", "Base Model": "Base-A"},
        {"Method": "Shared", "Base Model": "Base-B"},
    ]
    payload["answer"] = {"table": {"rows": rows}}
    payload["derivation"]["facts"] = [
        {
            **_fact("f1", "p1", "unused"),
            "value": rows[0],
        },
        {
            **_fact("f2", "p2", "unused"),
            "value": rows[1],
        },
    ]
    payload["derivation"]["answer_bindings"] = [
        {
            "answer_path": "answer.table.rows[0]",
            "source_type": "fact",
            "source_id": "f1",
        },
        {
            "answer_path": "answer.table.rows[1]",
            "source_type": "fact",
            "source_id": "f2",
        },
    ]

    with pytest.raises(ReadingResponseError, match="one-row-per-method table"):
        _parse(reader, query, payload)


def test_each_method_table_rejects_model_decorated_duplicate_method_rows(tmp_path):
    reader = _reader(tmp_path)
    query = Query.from_dict(
        {
            "query_id": "q_methods",
            "benchmark": "LitTraceQA",
            "question": "What base model does each method use?",
            "answer_types": ["table"],
            "table_schema": [
                {"name": "Method", "type": "string", "is_row_key": True},
                {"name": "Base Model", "type": "string", "is_row_key": True},
            ],
        }
    )
    payload = _payload(bind_second_paper=True)
    rows = [
        {"Method": "Shared (Base-A)", "Base Model": "Base-A"},
        {"Method": "Shared (Base-B)", "Base Model": "Base-B"},
    ]
    payload["answer"] = {"table": {"rows": rows}}
    payload["derivation"]["facts"] = [
        {**_fact("f1", "p1", "unused"), "value": rows[0]},
        {**_fact("f2", "p2", "unused"), "value": rows[1]},
    ]
    payload["derivation"]["answer_bindings"] = [
        {"answer_path": "answer.table.rows[0]", "source_type": "fact", "source_id": "f1"},
        {"answer_path": "answer.table.rows[1]", "source_type": "fact", "source_id": "f2"},
    ]

    with pytest.raises(ReadingResponseError, match="one-row-per-method table"):
        _parse(reader, query, payload)


def test_paper_title_table_requires_exact_canonical_metadata_title(tmp_path):
    reader = _reader(tmp_path)
    payload = _payload(bind_second_paper=True)
    canonical_titles = {
        "p1": "Paper &#x27;One&#x27;",
        "p2": "Paper Two",
    }

    with pytest.raises(ReadingResponseError, match="canonical Paper Title mismatch"):
        _parse(
            reader,
            _query(),
            payload,
            canonical_paper_titles=canonical_titles,
        )

    payload["answer"]["table"]["rows"][0]["Paper Title"] = canonical_titles["p1"]
    payload["derivation"]["facts"][0]["value"] = canonical_titles["p1"]
    payload["derivation"]["answer_bindings"][0]["answer_fragment"] = canonical_titles[
        "p1"
    ]
    payload["derivation"]["final_semantic_answer"] = (
        f"{canonical_titles['p1']}, {canonical_titles['p2']}"
    )

    parsed = _parse(
        reader,
        _query(),
        payload,
        canonical_paper_titles=canonical_titles,
    )
    assert parsed["answer"]["table"]["rows"][0]["Paper Title"] == (
        "Paper &#x27;One&#x27;"
    )


def test_paper_title_table_rejects_ambiguous_multi_paper_row_owner(tmp_path):
    reader = _reader(tmp_path)
    payload = _payload(bind_second_paper=True)
    payload["derivation"]["answer_bindings"][1]["answer_path"] = (
        "answer.table.rows[0].Paper Title"
    )
    payload["derivation"]["answer_bindings"][1]["answer_fragment"] = "Paper One"
    payload["derivation"]["facts"][1]["value"] = "Paper One"
    payload["answer"]["table"]["rows"] = [{"Paper Title": "Paper One"}]
    payload["support"][1]["answer_path"] = "answer.table.rows[0]"
    payload["derivation"]["final_semantic_answer"] = "Paper One"

    with pytest.raises(ReadingResponseError, match="exactly one selected source paper"):
        _parse(
            reader,
            _query(explicit_rows=True),
            payload,
            canonical_paper_titles={"p1": "Paper One", "p2": "Paper Two"},
        )
