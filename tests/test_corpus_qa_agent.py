from __future__ import annotations

import json

import pytest

from littraceqa.candidate_handoff import CandidatePaper, load_candidate_handoffs
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.agent.corpus_qa import CorpusQAAgent, _focused_excerpt
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.submission import prediction_to_submission


_PLAN = json.dumps(
    {
        "targets": [{"name": "Method X accuracy", "search_terms": ["91.2"]}],
        "venues": [],
        "years": [],
        "modalities": ["table", "text_span"],
        "requires_multiple_papers": False,
    }
)


def _write_corpus(path) -> None:
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": "Table 1 reports Accuracy 91.2 for Method X.",
            "metadata": {"page": 4, "table_id": "Table 1", "title": "Paper One"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#text",
            "chunk_type": "text_span",
            "text": "Method X obtains an accuracy of 91.2.",
            "metadata": {"page": 4, "section": "Results", "title": "Paper One"},
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#text",
            "chunk_type": "text_span",
            "text": "An unrelated baseline.",
            "metadata": {"page": 2, "title": "Paper Two"},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _query(**changes) -> Query:
    values = {
        "query_id": "q1",
        "question": "According to Table 1, what accuracy does Method X obtain?",
        "answer_types": ["freeform", "multiple_choice", "table"],
        "table_schema": [
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Accuracy", "type": "number", "is_row_key": False},
        ],
    }
    values.update(changes)
    return Query(**values)


def _candidates() -> tuple[CandidatePaper, ...]:
    return (
        CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
        CandidatePaper("p2", 2, "Paper Two", "ACL", 2025),
    )


def test_agent_answers_all_types_and_emits_exact_submission_shape(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(
        responses=[
            _PLAN,
            json.dumps(
                {"paper_ids": ["p1"], "unresolved_targets": [], "reason": "match"}
            ),
            json.dumps(
                {
                    "papers": [
                        {"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]}
                    ],
                    "answer": {
                        "freeform": {"text": "91.2"},
                        "table": {"rows": [{"Method": "Method X", "Accuracy": 91.2}]},
                    },
                    "semantic_multiple_choice": {"text": "91.2"},
                }
            ),
        ]
    )
    query = _query()
    prediction = CorpusQAAgent(ChunkStore(corpus), llm).run(query, _candidates())
    submission = prediction_to_submission(query, prediction)

    assert set(submission) == {"query_id", "gold_papers", "evidence", "answer"}
    assert set(submission["answer"]) == set(query.answer_types)
    assert submission["answer"]["freeform"]["text"] == "91.2"
    assert submission["answer"]["table"]["rows"] == [
        {"Method": "Method X", "Accuracy": 91.2}
    ]
    assert len(submission["answer"]["multiple_choice"]["gold"]) == 1
    assert submission["evidence"][0]["locator"]["table_id"] == "Table 1"
    assert "trace" not in submission and "candidate_papers" not in submission


def test_development_metadata_is_rejected_before_prompt(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM()
    agent = CorpusQAAgent(ChunkStore(corpus), llm)

    with pytest.raises(ValueError, match="four official input fields"):
        agent.run(_query(task_family="multi_paper"), _candidates())
    assert llm.calls == []


def test_llm_cannot_invent_paper_or_chunk_ids(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(
        responses=[
            _PLAN,
            '{"paper_ids":["invented"],"unresolved_targets":[]}',
            json.dumps(
                {
                    "papers": [
                        {
                            "paper_id": "p1",
                            "evidence_chunk_ids": ["invented#chunk"],
                        }
                    ],
                    "answer": {
                        "freeform": {"text": "91.2"},
                        "table": {"rows": [{"Method": "X", "Accuracy": 91.2}]},
                    },
                    "semantic_multiple_choice": {"text": "91.2"},
                }
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="no valid evidence"):
        CorpusQAAgent(ChunkStore(corpus), llm).run(_query(), _candidates())


def test_gold_sentinel_never_reaches_prompt(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    sentinel = "GOLD_SENTINEL_MUST_NOT_LEAK"
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    query_record = _query().to_dict()
    query_record.update(
        {
            "task_family": "multi_paper",
            "primary_evidence_type": "table",
            "options": {"A": sentinel},
            "_gold": {"answer": sentinel},
        }
    )
    queries.write_text(json.dumps(query_record) + "\n", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "candidate_papers": [
                    {"rank": 1, "paper_id": "p1"},
                    {"rank": 2, "paper_id": "p2"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = load_candidate_handoffs(queries, candidates)[0]
    llm = FakeLLM(
        responses=[
            _PLAN,
            '{"paper_ids":["p1"],"unresolved_targets":[]}',
            '{"papers":[{"paper_id":"p1","evidence_chunk_ids":["p1#text"]}],'
            '"answer":{"freeform":{"text":"91.2"},'
            '"table":{"rows":[{"Method":"X","Accuracy":91.2}]}},'
            '"semantic_multiple_choice":{"text":"91.2"}}',
        ]
    )
    agent = CorpusQAAgent(ChunkStore(corpus), llm)
    agent.run(handoff.query, handoff.candidate_papers)

    assert sentinel not in "\n".join(llm.calls)


def test_paper_set_is_not_shrunk_to_evidence_papers(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(
        responses=[
            _PLAN,
            '{"paper_ids":["p1","p2"],"unresolved_targets":[]}',
            '{"papers":[{"paper_id":"p1","evidence_chunk_ids":["p1#text"]}],'
            '"answer":{"freeform":{"text":"91.2"},'
            '"table":{"rows":[{"Method":"X","Accuracy":91.2}]}},'
            '"semantic_multiple_choice":{"text":"91.2"}}',
        ]
    )

    prediction = CorpusQAAgent(ChunkStore(corpus), llm).run(_query(), _candidates())

    assert prediction.gold_papers == [{"paper_id": "p1"}, {"paper_id": "p2"}]
    assert {item.paper_id for item in prediction.evidence} == {"p1"}


def test_planner_invalid_json_fails_before_fallback_is_checkpointable(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(responses=["not-json"])

    with pytest.raises(RuntimeError, match="LLM call failed"):
        CorpusQAAgent(ChunkStore(corpus), llm).run(_query(), _candidates())


def test_numbered_reference_marker_is_ranked_first(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    agent = CorpusQAAgent(ChunkStore(corpus), FakeLLM())
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#ref1",
            "chunk_type": "text_span",
            "text": "[1] First Author. Unrelated work.",
            "metadata": {"page": 10, "section": "References"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#ref24",
            "chunk_type": "text_span",
            "text": "[24] Freda Shi. Language models are multilingual.",
            "metadata": {"page": 12, "section": "References"},
        },
    ]
    query = _query(
        question="Who is the first author of the 24th reference?",
        answer_types=["freeform"],
        table_schema=None,
    )

    ranked = agent._rank_records(
        query,
        records,
        {"targets": [], "modalities": ["citation_context"]},
    )

    assert ranked[0][1]["chunk_id"] == "p1#ref24"


def test_focused_excerpt_skips_repeated_mineru_title_prefix():
    text = (
        "[ACL 2025] NaturalQ Compression Paper\n"
        + "irrelevant " * 700
        + "NaturalQ requested row reports the decisive value 14.70."
    )

    excerpt = _focused_excerpt(
        text,
        "NaturalQ Compression Paper decisive value",
        max_chars=500,
    )

    assert "decisive value 14.70" in excerpt
