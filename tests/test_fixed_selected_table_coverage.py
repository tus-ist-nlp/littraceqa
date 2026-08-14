from __future__ import annotations

import json

import pytest

from littraceqa.aoai_pairwise_reader import (
    FIXED_SELECTED_CHECKPOINT_KIND,
    PairwiseAOAIReader,
    ReadingResponseError,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _table_query() -> Query:
    return Query.from_dict(
        {
            "query_id": "q_table",
            "benchmark": "LitTraceQA",
            "question": (
                "Which selected papers cite UniAD and use it as a baseline "
                "in their main comparison table?"
            ),
            "answer_types": ["table"],
            "table_schema": [
                {"name": "Paper Title", "type": "string", "is_row_key": True}
            ],
        }
    )


def _write_corpus(path) -> None:
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#citation",
            "chunk_type": "citation_context",
            "text": "Reference [23] is UniAD, Planning-oriented autonomous driving.",
            "metadata": {"page": 8, "section": "References"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#tab1",
            "chunk_type": "table",
            "text": "Table 1. Ablation study without the requested baseline.",
            "metadata": {"page": 5, "table_id": "Table 1"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#tab2",
            "chunk_type": "table",
            "text": (
                "Paper One. Table 2. Main comparison table. "
                "UniAD [23] is a baseline."
            ),
            "metadata": {"page": 6, "table_id": "Table 2"},
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#tab1",
            "chunk_type": "table",
            "text": (
                "Paper Two. Table 1. Main comparison table. "
                "UniAD [19] is a baseline."
            ),
            "metadata": {"page": 4, "table_id": "Table 1"},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _judgments() -> list[dict[str, object]]:
    return [
        {
            "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
            "paper_id": "p1",
            "rank": 1,
            "label": "partial_answer",
            "evidence": [
                {
                    "chunk_id": "p1#citation",
                    "source_type": "citation_context",
                    "purpose": "answer",
                    "quote_or_value": "Reference [23] is UniAD",
                }
            ],
            "extracted_facts": [
                {
                    "chunk_id": "p1#citation",
                    "purpose": "citation_fact",
                    "fact": "Paper One cites UniAD.",
                    "source_excerpt": "Reference [23] is UniAD",
                }
            ],
        },
        {
            "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
            "paper_id": "p2",
            "rank": 2,
            "label": "partial_answer",
            "evidence": [
                {
                    "chunk_id": "p2#tab1",
                    "source_type": "table",
                    "purpose": "answer",
                    "quote_or_value": "UniAD [19] is a baseline",
                }
            ],
            "extracted_facts": [
                {
                    "chunk_id": "p2#tab1",
                    "purpose": "table_row",
                    "fact": "Paper Two uses UniAD as a baseline.",
                    "source_excerpt": "UniAD [19] is a baseline",
                }
            ],
        },
    ]


def _reader(tmp_path) -> PairwiseAOAIReader:
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    return PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        paper_set_policy="fixed_selected",
        answer_neighbor_chunks=0,
    )


def _table_payload(*, include_p2: bool) -> dict[str, object]:
    paper_ids = ["p1", *(["p2"] if include_p2 else [])]
    chunks = {"p1": "p1#tab2", "p2": "p2#tab1"}
    titles = {"p1": "Paper One", "p2": "Paper Two"}
    facts = []
    bindings = []
    rows = []
    support = []
    for index, paper_id in enumerate(paper_ids):
        fact_id = f"f_{paper_id}"
        title = titles[paper_id]
        facts.append(
            {
                "id": fact_id,
                "name": f"{paper_id} qualifying paper title",
                "value": title,
                "value_kind": "text",
                "paper_id": paper_id,
                "chunk_ids": [chunks[paper_id]],
            }
        )
        bindings.append(
            {
                "answer_path": f"answer.table.rows[{index}].Paper Title",
                "source_type": "fact",
                "source_id": fact_id,
                "answer_fragment": title,
            }
        )
        rows.append({"Paper Title": title})
        support.append(
            {
                "answer_path": f"answer.table.rows[{index}]",
                "paper_id": paper_id,
                "chunk_ids": [chunks[paper_id]],
            }
        )
    return {
        "status": "ready",
        "paper_relevance": [
            {"paper_id": paper_id, "role": "answer_source", "reason": "row source"}
            for paper_id in paper_ids
        ],
        "papers": [
            {"paper_id": paper_id, "evidence_chunk_ids": [chunks[paper_id]]}
            for paper_id in paper_ids
        ],
        "derivation": {
            "facts": facts,
            "operations": [],
            "answer_bindings": bindings,
            "final_semantic_answer": ", ".join(titles[item] for item in paper_ids),
        },
        "answer": {"table": {"rows": rows}},
        "support": support,
        "completeness": {"answered_parts": ["qualifying papers"], "missing": []},
    }


def test_fixed_selected_multi_paper_table_supplies_ranked_table_per_paper(
    tmp_path,
):
    reader = _reader(tmp_path)

    context = reader._answer_context(_table_query(), _judgments())

    # p1's extractor found only the citation. Python adds its highest-ranked
    # eligible tables; p2 already handed off its answer table and needs no copy.
    assert context["fixed_selected_table_supplement_chunk_ids"][:2] == [
        "p1#tab2",
        "p1#tab1",
    ]
    assert "p1#tab2" in context["records_by_id"]
    assert "p1#tab2" in context["python_supplemental_chunk_ids"]
    assert "p2#tab1" not in context["fixed_selected_table_supplement_chunk_ids"]


def test_fixed_selected_multi_paper_table_rejects_silent_paper_omission(
    tmp_path,
):
    reader = _reader(tmp_path)
    context = reader._answer_context(_table_query(), _judgments())

    with pytest.raises(
        ReadingResponseError,
        match="fixed-selected multi-paper table must use every authoritative",
    ):
        reader._parse_answer(
            query=_table_query(),
            payload_text=json.dumps(_table_payload(include_p2=False)),
            relevant_paper_ids={"p1", "p2"},
            context_records=context["records_by_id"],
        )


def test_fixed_selected_multi_paper_table_accepts_grounding_from_every_paper(
    tmp_path,
):
    reader = _reader(tmp_path)
    context = reader._answer_context(_table_query(), _judgments())

    parsed = reader._parse_answer(
        query=_table_query(),
        payload_text=json.dumps(_table_payload(include_p2=True)),
        relevant_paper_ids={"p1", "p2"},
        context_records=context["records_by_id"],
    )

    assert {item["paper_id"] for item in parsed["paper_relevance"]} == {
        "p1",
        "p2",
    }
