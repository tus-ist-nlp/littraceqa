from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import ClassVar

import pytest

from littraceqa.aoai_pairwise_reader import (
    PairwiseAOAIReader,
    ReadingResponseError,
)
from littraceqa.candidate_handoff import CandidatePaper
from littraceqa.chunk_store import IMAGE_PATH_ERROR_KEY, ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.mineru_record import readable_image_path
from littraceqa.submission import prediction_to_submission


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
VALID_PNG_ALTERNATE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _image_root(tmp_path):
    return tmp_path / "trusted-images"


def _write_trusted_image(
    tmp_path, paper_id: str, filename: str, payload: bytes = VALID_PNG
):
    image_path = _image_root(tmp_path) / paper_id / "auto" / "images" / filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(payload)
    return image_path


def _write_corpus(path, *, long_text: str | None = None) -> None:
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": long_text or "Table 1: Method X reports 42 on Dataset Y.",
            "metadata": {"page": 3, "table_id": "Table 1"},
        },
        {
            "paper_id": "p1",
            "chunk_id": "p1#text",
            "chunk_type": "text_span",
            "text": "The result is discussed in the evaluation section.",
            "metadata": {"page": 3, "section": "Evaluation"},
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#text",
            "chunk_type": "text_span",
            "text": "This paper studies an unrelated topic.",
            "metadata": {"page": 1},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_image_corpus(path, tmp_path, *, image_count: int) -> list[str]:
    records = []
    image_paths = []
    for index in range(1, image_count + 1):
        image_path = _write_trusted_image(
            tmp_path, "p1", f"figure-{index}.png"
        )
        image_paths.append(str(image_path))
        records.append(
            {
                "paper_id": "p1",
                "chunk_id": f"p1#fig{index}",
                "chunk_type": "figure",
                "text": f"Figure {index}: Method X result {index}.",
                "metadata": {
                    "page": index,
                    "figure_id": f"Figure {index}",
                    "image_path": str(image_path),
                },
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return image_paths


class _RecordingMultimodalLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self._index = 0

    def complete_with_metadata(self, prompt, image_paths=None):
        self.calls.append(
            {"prompt": prompt, "image_paths": list(image_paths or [])}
        )
        response = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return {"text": response, "usage": None}


def _query() -> Query:
    return Query(
        query_id="q1",
        question="What value does Method X report on Dataset Y?",
        answer_types=["freeform"],
        table_schema=None,
    )


def _judgment(
    label: str,
    chunk_id: str | None = None,
    *,
    answer_meaning: str = "42",
) -> str:
    evidence = (
        [{"chunk_id": chunk_id, "quote_or_value": "42"}] if chunk_id else []
    )
    return json.dumps(
        {
            "paper_role": "target_owner" if chunk_id else "topic_only",
            "label": label,
            "answerable_from_this_paper": label == "direct_answer",
            "satisfied_constraints": ["Method X value"] if chunk_id else [],
            "missing_constraints": [] if chunk_id else ["Method X value"],
            "blocking_mismatches": [],
            "visual": {"required": False, "status": "not_needed"},
            "evidence": evidence,
            "candidate_answer": {"meaning": answer_meaning} if chunk_id else {},
            "confidence": 0.98,
            "reason": "the table contains the value" if chunk_id else "no match",
        }
    )


def _answer() -> str:
    return json.dumps(_structured_answer_payload({"p1": ["p1#table"]}))


def _structured_answer_payload(
    paper_chunks: dict[str, list[str]],
) -> dict[str, object]:
    first_paper = next(iter(paper_chunks))
    first_chunk = paper_chunks[first_paper][0]
    return {
        "status": "ready",
        "paper_relevance": [
            {"paper_id": paper_id, "role": "target_owner", "reason": "owner"}
            for paper_id in paper_chunks
        ],
        "papers": [
            {"paper_id": paper_id, "evidence_chunk_ids": chunk_ids}
            for paper_id, chunk_ids in paper_chunks.items()
        ],
        "derivation": {
            "facts": [
                {
                    "id": "f_reported_value",
                    "name": "reported value",
                    "value": "42",
                    "value_kind": "reported",
                    "paper_id": first_paper,
                    "chunk_ids": [first_chunk],
                }
            ],
            "operations": [],
            "answer_bindings": [
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "f_reported_value",
                    "answer_fragment": "42",
                }
            ],
            "final_semantic_answer": "42",
        },
        "answer": {"freeform": {"text": "42"}},
        "support": [
            {
                "answer_path": "answer.freeform.text",
                "paper_id": paper_id,
                "chunk_ids": chunk_ids,
            }
            for paper_id, chunk_ids in paper_chunks.items()
        ],
        "completeness": {"answered_parts": ["value"], "missing": []},
    }


def test_each_candidate_is_judged_independently_then_answered(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(
        responses=[
            _judgment("direct_answer", "p1#table"),
            _judgment("irrelevant"),
            _answer(),
        ]
    )
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)
    candidates = (
        CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
        CandidatePaper("p2", 2, "Paper Two", "ACL", 2025),
    )

    judgments = [reader.judge_candidate(_query(), item) for item in candidates]
    prediction, answer_record = reader.answer_from_judgments(
        _query(), candidates, judgments
    )
    submission = prediction_to_submission(_query(), prediction)

    assert len(llm.calls) == 3
    assert "p1#table" in llm.calls[0] and "p2#text" not in llm.calls[0]
    assert "p2#text" in llm.calls[1] and "p1#table" not in llm.calls[1]
    assert judgments[0]["relevant"] is True
    assert judgments[1]["relevant"] is False
    assert answer_record["accepted_paper_ids"] == ["p1"]
    assert submission["answer"] == {"freeform": {"text": "42"}}
    assert submission["gold_papers"] == [{"paper_id": "p1"}]


def test_paper_relevance_is_separate_from_minimal_answer_support(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = _query()
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    raw["paper_relevance"] = [
        {"paper_id": "p1", "role": "target_owner", "reason": "direct owner"},
        {
            "paper_id": "p2",
            "role": "comparison_source",
            "reason": "named comparison paper",
        },
    ]
    context_records = {
        record["chunk_id"]: record
        for paper_id in ("p1", "p2")
        for record in reader.chunk_store.load_paper(paper_id)
    }
    payload = reader._parse_answer(
        query=query,
        payload_text=json.dumps(raw),
        relevant_paper_ids={"p1", "p2"},
        context_records=context_records,
    )
    prediction = reader._build_prediction(
        query=query,
        payload=payload,
        context_records=context_records,
        candidate_ids=["p1", "p2"],
        relevant=[{"paper_id": "p1"}, {"paper_id": "p2"}],
        image_count=0,
    )

    assert prediction.gold_papers == [{"paper_id": "p1"}, {"paper_id": "p2"}]
    assert {item.paper_id for item in prediction.evidence} == {"p1"}


def test_stage_two_rejects_submitted_chunk_unused_by_derivation(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table", "p1#text"]})
    context_records = {
        record["chunk_id"]: record for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="unrelated_evidence"):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1"},
            context_records=context_records,
        )


def test_stage_two_rejects_fact_from_different_submitted_paper_chunk(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    raw["paper_relevance"].append(
        {"paper_id": "p2", "role": "comparison_source", "reason": "comparison"}
    )
    raw["derivation"]["facts"][0].update(
        {"paper_id": "p2", "chunk_ids": ["p2#text"]}
    )
    context_records = {
        record["chunk_id"]: record
        for paper_id in ("p1", "p2")
        for record in reader.chunk_store.load_paper(paper_id)
    }

    with pytest.raises(ReadingResponseError, match="unsupported_facts"):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1", "p2"},
            context_records=context_records,
        )


def test_one_chunk_can_support_multiple_table_cells(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        "ltqa_table",
        "Copy the supported row.",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
        ],
    )
    raw = {
        "status": "ready",
        "paper_relevance": [
            {"paper_id": "p1", "role": "target_owner", "reason": "owner"}
        ],
        "papers": [{"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]}],
        "derivation": {
            "facts": [
                {
                    "id": "f_row",
                    "name": "row",
                    "value": {"Method": "X", "Score": 42},
                    "value_kind": "reported",
                    "paper_id": "p1",
                    "chunk_ids": ["p1#table"],
                }
            ],
            "operations": [],
            "answer_bindings": [
                {
                    "answer_path": "answer.table.rows[0]",
                    "source_type": "fact",
                    "source_id": "f_row",
                }
            ],
            "final_semantic_answer": "X | 42",
        },
        "answer": {"table": {"rows": [{"Method": "X", "Score": 42}]}},
        "support": [
            {
                "answer_path": "answer.table.rows[0].Method",
                "paper_id": "p1",
                "chunk_ids": ["p1#table"],
            },
            {
                "answer_path": "answer.table.rows[0].Score",
                "paper_id": "p1",
                "chunk_ids": ["p1#table"],
            },
        ],
        "completeness": {"answered_parts": ["X row"], "missing": []},
    }
    context_records = {
        record["chunk_id"]: record
        for record in reader.chunk_store.load_paper("p1")
    }

    payload = reader._parse_answer(
        query=query,
        payload_text=json.dumps(raw),
        relevant_paper_ids={"p1"},
        context_records=context_records,
    )

    assert payload["support"] == raw["support"]


def test_official_multiple_choice_label_is_used_without_placeholder(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        "ltqa_mc",
        "Which option is supported?",
        ["multiple_choice"],
        options={"A": "Alpha", "E": "Epsilon"},
    )
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    raw["derivation"]["facts"][0]["value"] = "Epsilon"
    raw["derivation"]["final_semantic_answer"] = "Epsilon"
    raw["derivation"]["answer_bindings"] = [
        {
            "answer_path": "answer.multiple_choice",
            "source_type": "fact",
            "source_id": "f_reported_value",
            "answer_fragment": "Epsilon",
        }
    ]
    raw["answer"] = {
        "multiple_choice": {
            "label": "E",
            "selected_option_text": "Epsilon",
        }
    }
    raw["support"][0]["answer_path"] = "answer.multiple_choice"
    context_records = {
        record["chunk_id"]: record for record in reader.chunk_store.load_paper("p1")
    }
    payload = reader._parse_answer(
        query=query,
        payload_text=json.dumps(raw),
        relevant_paper_ids={"p1"},
        context_records=context_records,
    )
    prediction = reader._build_prediction(
        query=query,
        payload=payload,
        context_records=context_records,
        candidate_ids=["p1"],
        relevant=[{"paper_id": "p1"}],
        image_count=0,
    )

    assert prediction.answer.multiple_choice == {"gold": "E"}
    assert prediction_to_submission(query, prediction)["answer"] == {
        "multiple_choice": {"gold": "E"}
    }


def test_stage_two_visual_fact_requires_its_actual_attached_image(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=1)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), FakeLLM()
    )
    query = Query(
        "ltqa_visual",
        "How many panels are visible in Figure 1?",
        ["freeform"],
    )
    raw = _structured_answer_payload({"p1": ["p1#fig1"]})
    raw["derivation"]["facts"][0]["value_kind"] = "visual"
    context_records = {
        record["chunk_id"]: record
        for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="no actually attached source image"):
        reader._parse_answer(
            query=query,
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1"},
            context_records=context_records,
            attached_image_paths=[],
        )

    payload = reader._parse_answer(
        query=query,
        payload_text=json.dumps(raw),
        relevant_paper_ids={"p1"},
        context_records=context_records,
        attached_image_paths=image_paths,
    )
    assert payload["derivation"]["facts"][0]["value_kind"] == "visual"


def test_stage_one_visual_false_positive_omitted_by_stage_two_is_not_required(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    context_records = {
        record["chunk_id"]: record
        for paper_id in ("p1", "p2")
        for record in reader.chunk_store.load_paper(paper_id)
    }

    payload = reader._parse_answer(
        query=_query(),
        payload_text=json.dumps(raw),
        relevant_paper_ids={"p1", "p2"},
        context_records=context_records,
        required_visual_paper_ids={"p2"},
    )

    assert payload["papers"] == [
        {"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]}
    ]
    assert {fact["paper_id"] for fact in payload["derivation"]["facts"]} == {"p1"}


def test_stage_one_visual_requirement_for_used_paper_forces_visual_fact(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    context_records = {
        record["chunk_id"]: record for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="stage-1 marked visual evidence"):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1"},
            context_records=context_records,
            required_visual_paper_ids={"p1"},
        )


def test_explicit_visual_query_still_requires_visual_fact_without_stage_one_flag(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    context_records = {
        record["chunk_id"]: record for record in reader.chunk_store.load_paper("p1")
    }
    query = Query(
        "ltqa_explicit_visual",
        "How many panels are visible in Figure 1?",
        ["freeform"],
    )

    with pytest.raises(ReadingResponseError, match="explicitly requires visual reading"):
        reader._parse_answer(
            query=query,
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1"},
            context_records=context_records,
            required_visual_paper_ids=set(),
        )


def test_untrusted_corpus_image_is_never_attached_without_image_root(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=1)
    llm = _RecordingMultimodalLLM([_judgment("irrelevant")])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)

    reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert [call["image_paths"] for call in llm.calls] == [[]]
    metadata = reader.chunk_store.load_paper("p1")[0]["metadata"]
    assert metadata["image_path"] == ""
    assert "image_root is required" in metadata[IMAGE_PATH_ERROR_KEY]


def test_stage_two_rejects_malformed_completeness(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    raw = _structured_answer_payload({"p1": ["p1#table"]})
    raw["completeness"] = {}
    context_records = {
        record["chunk_id"]: record
        for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="exactly answered_parts and missing"):
        reader._parse_answer(
            query=_query(),
            payload_text=json.dumps(raw),
            relevant_paper_ids={"p1"},
            context_records=context_records,
        )


def test_invented_stage_one_chunk_id_is_a_hard_error(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(responses=[_judgment("direct_answer", "invented")])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)

    with pytest.raises(ReadingResponseError, match="invented or cross-cited"):
        reader.judge_candidate(
            _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
        )
    assert len(llm.calls) == 2


def test_stage_one_repairs_invented_chunk_id_once_with_same_validator(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=1)
    invented_response = _judgment("direct_answer", "p1#fig0010")
    repaired_response = _judgment("direct_answer", "p1#fig1")
    llm = _RecordingMultimodalLLM([invented_response, repaired_response])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), llm
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 26, "Paper One", "ACL", 2025)
    )

    assert judgment["evidence_chunk_ids"] == ["p1#fig1"]
    assert len(llm.calls) == 2
    assert [call["image_paths"] for call in llm.calls] == [
        image_paths,
        image_paths,
    ]
    repair_prompt = str(llm.calls[1]["prompt"])
    assert "Validation error:" in repair_prompt
    assert "p1#fig0010" in repair_prompt
    assert 'Allowed selected-context chunk_ids: ["p1#fig1"]' in repair_prompt
    assert [call["phase"] for call in judgment["calls"]] == [
        "single_context",
        "single_context",
    ]
    assert [call["attempt"] for call in judgment["calls"]] == [
        "initial",
        "evidence_repair",
    ]
    assert "invented or cross-cited" in judgment["calls"][0]["parse_error"]
    assert judgment["calls"][1]["parse_error"] is None
    assert all("usage" in call for call in judgment["calls"])
    assert judgment["calls"][0]["raw_response"] == invented_response
    assert judgment["calls"][1]["raw_response"] == repaired_response
    assert judgment["judgment_call_count"] == 2
    assert judgment["judgment"]["evidence_chunk_ids"] == ["p1#fig1"]


def _write_compaction_corpus(path) -> None:
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#abstract",
            "chunk_type": "title_abstract",
            "text": "Paper One studies several evaluation settings.",
            "metadata": {"page": 1},
        }
    ]
    records.extend(
        {
            "paper_id": "p1",
            "chunk_id": f"p1#noise{index:02d}",
            "chunk_type": "text_span",
            "text": (f"Background discussion {index}. " + "filler " * 400),
            "metadata": {"page": index + 1, "section": "Background"},
        }
        for index in range(1, 13)
    )
    records.append(
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": "Table 7: Method X reports exactly 42 on Dataset Y.",
            "metadata": {"page": 20, "table_id": "Table 7"},
        }
    )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_oversized_paper_is_compacted_and_judged_with_one_initial_call(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_compaction_corpus(corpus)
    llm = FakeLLM(responses=[_judgment("direct_answer", "p1#table")])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        max_paper_context_chars=8_000,
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    all_chunk_ids = {
        record["chunk_id"] for record in reader.chunk_store.load_paper("p1")
    }
    assert len(llm.calls) == 1
    assert judgment["base_judgment_call_count"] == 1
    assert judgment["judgment_call_count"] == 1
    assert judgment["paper_context_compacted"] is True
    assert judgment["context_chunk_count"] < len(all_chunk_ids)
    assert judgment["omitted_chunk_count"] > 0
    assert set(judgment["context_chunk_ids"]).isdisjoint(
        judgment["omitted_chunk_ids"]
    )
    assert set(judgment["context_chunk_ids"]) | set(
        judgment["omitted_chunk_ids"]
    ) == all_chunk_ids
    assert "p1#table" in judgment["context_chunk_ids"]
    assert [item["chunk_id"] for item in judgment["paper_context_selection"]] == (
        judgment["context_chunk_ids"]
    )
    assert judgment["calls"][0]["context_chunk_ids"] == judgment[
        "context_chunk_ids"
    ]
    assert judgment["paper_id"] == "p1"
    assert judgment["evidence_chunk_ids"] == ["p1#table"]
    assert judgment["judgment"]["evidence_chunk_ids"] == ["p1#table"]
    assert "batch_count" not in judgment
    assert "batch_judgments" not in judgment


def test_single_paper_context_is_deterministic_and_preserves_original_records(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_compaction_corpus(corpus)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus), FakeLLM(), max_paper_context_chars=8_000
    )
    query = _query()
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")

    first = reader._paper_context(query, candidate, records)
    second = reader._paper_context(query, candidate, records)

    assert first == second
    assert first["compacted"] is True
    assert len(first["text"]) <= reader.max_paper_context_chars
    assert first["total_chunk_count"] == len(records)
    assert set(first["selected_chunk_ids"]).isdisjoint(first["omitted_chunk_ids"])
    assert set(first["selected_chunk_ids"]) | set(first["omitted_chunk_ids"]) == {
        record["chunk_id"] for record in records
    }
    assert list(first["records_by_id"]) == first["selected_chunk_ids"]
    assert first["records_by_id"]["p1#table"]["text"] == (
        "Table 7: Method X reports exactly 42 on Dataset Y."
    )
    assert '"segment"' not in first["text"]


def test_real_but_omitted_chunk_id_is_rejected_by_stage_one(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_compaction_corpus(corpus)
    store = ChunkStore(corpus)
    query = _query()
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    probe = PairwiseAOAIReader(
        store, FakeLLM(), max_paper_context_chars=8_000
    )
    context = probe._paper_context(query, candidate, store.load_paper("p1"))
    omitted_chunk_id = context["omitted_chunk_ids"][0]
    llm = FakeLLM(
        responses=[_judgment("direct_answer", omitted_chunk_id)]
    )
    reader = PairwiseAOAIReader(
        store, llm, max_paper_context_chars=8_000
    )

    with pytest.raises(ReadingResponseError, match="invented or cross-cited"):
        reader.judge_candidate(query, candidate)

    # The normal judgment is one call. A validator failure gets exactly one
    # bounded evidence-repair attempt, which is rejected by the same validator.
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_family", "multi_paper"),
        ("primary_evidence_type", "table"),
    ],
)
def test_development_metadata_is_rejected_before_any_llm_call(
    tmp_path, field, value
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(responses=[_judgment("irrelevant")])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)
    query = _query()
    setattr(query, field, value)

    with pytest.raises(ValueError, match="forbidden values are present"):
        reader.judge_candidate(
            query, CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
        )
    assert llm.calls == []


def test_stage_one_rejects_relevant_distractor_role(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    payload = json.loads(_judgment("direct_answer", "p1#table"))
    payload["paper_role"] = "distractor"
    payload["blocking_mismatches"] = ["wrong owner"]
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    records = {
        record["chunk_id"]: record for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="incompatible with relevant label"):
        reader._parse_judgment(
            query=_query(),
            candidate=CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
            payload_text=json.dumps(payload),
            allowed_records=records,
        )


def test_stage_one_rejects_claimed_visual_inspection_without_attachment(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=1)
    payload = json.loads(_judgment("direct_answer", "p1#fig1"))
    payload["visual"] = {"required": True, "status": "inspected"}
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), FakeLLM()
    )
    records = {
        record["chunk_id"]: record
        for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(ReadingResponseError, match="actually attached image"):
        reader._parse_judgment(
            query=_query(),
            candidate=CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
            payload_text=json.dumps(payload),
            allowed_records=records,
            attached_image_paths=[],
        )

    parsed = reader._parse_judgment(
        query=_query(),
        candidate=CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
        payload_text=json.dumps(payload),
        allowed_records=records,
        attached_image_paths=[readable_image_path(records["p1#fig1"])],
    )
    assert parsed["visual"] == {"required": True, "status": "inspected"}


def test_stage_one_rejects_visual_evidence_when_a_different_image_was_attached(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=2)
    payload = json.loads(_judgment("direct_answer", "p1#fig2"))
    payload["visual"] = {"required": True, "status": "inspected"}
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), FakeLLM()
    )
    records = {
        record["chunk_id"]: record
        for record in reader.chunk_store.load_paper("p1")
    }

    with pytest.raises(
        ReadingResponseError, match="actually attached source image"
    ):
        reader._parse_judgment(
            query=_query(),
            candidate=CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
            payload_text=json.dumps(payload),
            allowed_records=records,
            attached_image_paths=[image_paths[0]],
        )


def test_answer_evidence_cap_is_round_robin_across_papers(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": f"p1#{index}",
            "chunk_type": "text_span",
            "text": f"P1 evidence {index}",
            "metadata": {"page": index},
        }
        for index in range(1, 4)
    ] + [
        {
            "paper_id": "p2",
            "chunk_id": "p2#1",
            "chunk_type": "text_span",
            "text": "Tail paper evidence",
            "metadata": {"page": 1},
        }
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        max_evidence=2,
        answer_neighbor_chunks=0,
    )
    judgments = [
        {
            "paper_id": "p1",
            "rank": 1,
            "relevant": True,
            "label": "direct_answer",
            "evidence": [
                {"chunk_id": f"p1#{index}", "quote_or_value": str(index)}
                for index in range(1, 4)
            ],
        },
        {
            "paper_id": "p2",
            "rank": 44,
            "relevant": True,
            "label": "partial_answer",
            "evidence": [{"chunk_id": "p2#1", "quote_or_value": "tail"}],
        },
    ]

    context = reader._answer_context(_query(), judgments)

    assert list(context["records_by_id"]) == ["p1#1", "p2#1"]


def test_answer_prompt_preserves_q029_constraint_conflict_and_cache_inputs(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "paper_id": "ecm",
                "chunk_id": "ecm#table",
                "chunk_type": "table",
                "text": (
                    "ECM-XL 102.4M reports FID 2.49 on ImageNet. "
                    "No CIFAR-10 value is reported for this variant."
                ),
                "metadata": {"page": 4, "table_id": "Table 2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    query = Query(
        query_id="q_029",
        question="Report the CIFAR-10 FID for ECM-XL 102.4M.",
        answer_types=["table"],
        table_schema=[
            {"name": "model", "type": "string", "is_row_key": True},
            {"name": "fid", "type": "number", "is_row_key": False},
        ],
    )
    missing = "ECM-XL 102.4M has only an ImageNet result; CIFAR-10 is unreported"
    reason = "2.49 belongs to ImageNet, not the requested CIFAR-10 dataset"
    judgment = {
        "paper_id": "ecm",
        "rank": 1,
        "cache_key": "stage-one-cache",
        "label": "partial_answer",
        "relevant": True,
        "satisfied_constraints": ["ECM-XL 102.4M is listed"],
        "missing_constraints": [missing],
        "evidence": [
            {"chunk_id": "ecm#table", "quote_or_value": "ImageNet: 2.49"}
        ],
        "candidate_answer": {"dataset": "ImageNet", "fid": 2.49},
        "reason": reason,
        "visual": {"required": False, "status": "not_needed"},
    }
    reader = PairwiseAOAIReader(
        ChunkStore(corpus), FakeLLM(), answer_neighbor_chunks=0
    )
    context = reader._answer_context(query, [judgment])

    prompt = reader._answer_prompt(query, [judgment], context)

    summary_text = prompt.split(
        "Accepted paper summary (fallible hints, not evidence):\n", 1
    )[1].split("\n\n", 1)[0]
    summary = json.loads(summary_text)
    assert summary == [
        {
            "paper_id": "ecm",
            "title": "",
            "rank": 1,
            "label": "partial_answer",
            "paper_role": "uncertain",
            "satisfied_constraints": ["ECM-XL 102.4M is listed"],
            "missing_constraints": [missing],
            "blocking_mismatches": [],
            "candidate_answer": {"dataset": "ImageNet", "fid": 2.49},
            "reason": reason,
        }
    ]
    assert "dataset, split, model variant/size" in prompt
    assert "Never borrow a nearby value from another setting" in prompt
    assert "completeness" in prompt
    assert missing in prompt

    base_key = reader.answer_cache_key(query, [judgment])
    changed_keys = {
        reader.answer_cache_key(
            query, [{**judgment, "missing_constraints": ["different missing"]}]
        ),
        reader.answer_cache_key(
            query, [{**judgment, "reason": "different reason"}]
        ),
        reader.answer_cache_key(
            query,
            [
                {
                    **judgment,
                    "visual": {"required": True, "status": "inspected"},
                }
            ],
        ),
        reader.answer_cache_key(
            query, [{**judgment, "paper_role": "comparison_source"}]
        ),
        reader.answer_cache_key(
            query, [{**judgment, "blocking_mismatches": ["wrong setting"]}]
        ),
    }
    assert base_key not in changed_keys
    assert len(changed_keys) == 5


def test_table_prompt_requires_exact_schema_keys_and_source_cells(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        query_id="synthetic_lexical_cells",
        question="Copy the displayed results into the requested table.",
        answer_types=["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "string", "is_row_key": False},
            {"name": "Aux", "type": "string", "is_row_key": False},
        ],
    )
    relevant = [
        {
            "paper_id": "p1",
            "rank": 1,
            "label": "direct_answer",
            "satisfied_constraints": ["table cells"],
            "missing_constraints": [],
            "candidate_answer": {
                "Method": "Cedar",
                "Score": "007.25",
                "Aux": "not measured",
            },
            "reason": "Table 1 displays the requested row",
            "visual_conflict": False,
        }
    ]
    context = {
        "text": "Method | Score | Aux\nCedar | 007.25 | not measured",
        "records_by_id": {},
        "image_paths": [],
    }

    prompt = reader._answer_prompt(query, relevant, context)

    assert "Use every table_schema name verbatim" in prompt
    assert "exact string displayed in the cited" in prompt
    assert "Do not append %, units, or explanatory prose" in prompt
    assert "unless they literally appear in" in prompt
    assert "Preserve punctuation and typography byte-for-byte as displayed" in prompt
    assert "Do not numerically normalize a string-valued cell" in prompt
    assert "Preserve\n  a visibly printed missing-value mark as a string" in prompt
    assert "only a genuinely" in prompt
    assert "Never replace a mark or blank" in prompt
    assert "attached table image conflicts with lossy OCR or extracted Markdown" in prompt
    assert "use the cell visibly printed in the image" in prompt
    assert "Every emitted cell must be directly" in prompt
    assert '"Method":"source string"' in prompt
    assert '"Score":"source string"' in prompt


def test_table_prompt_uses_canonical_row_keys_and_splits_named_settings(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        query_id="synthetic_canonical_row_key",
        question=(
            "Report LinrNet under the warm-start and cold-start settings."
        ),
        answer_types=["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Setting", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "string", "is_row_key": False},
        ],
    )
    relevant = [
        {
            "paper_id": "p1",
            "rank": 1,
            "label": "direct_answer",
            "satisfied_constraints": ["both settings"],
            "missing_constraints": [],
            "candidate_answer": {},
            "reason": "The source table spells the method LinearNet.",
            "visual_conflict": False,
        }
    ]
    context = {
        "text": "Method | Setting | Score\nLinearNet | warm-start | 4.8 ± 0.6",
        "records_by_id": {},
        "image_paths": [],
    }

    prompt = reader._answer_prompt(query, relevant, context)

    assert "row-key entity or method name" in prompt
    assert "canonical spelling visibly" in prompt
    assert "question contains an obvious typo" in prompt
    assert "copy its spacing" in prompt
    assert "typography exactly from the source" in prompt
    assert "two\n  separately requested rows" in prompt
    assert "not one impossible combined setting" in prompt
    assert "Never invent" in prompt
    assert "Prefer the owning paper" in prompt
    assert "one direct object chunk per answer unit" in prompt


def test_answer_images_prioritize_direct_primary_evidence_without_starvation(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    records = []
    direct_image_paths = []
    direct_evidence = []
    for index in range(1, 8):
        image_path = _write_trusted_image(
            tmp_path, "direct", f"direct-{index}.png"
        )
        direct_image_paths.append(str(image_path))
        chunk_id = f"direct#fig{index}"
        direct_evidence.append(
            {"chunk_id": chunk_id, "quote_or_value": f"direct figure {index}"}
        )
        records.append(
            {
                "paper_id": "direct",
                "chunk_id": chunk_id,
                "chunk_type": "figure",
                "text": f"Direct evidence figure {index}",
                "metadata": {
                    "page": index,
                    "figure_id": f"Figure {index}",
                    "image_path": str(image_path),
                },
            }
        )

    judgments = []
    partial_primary_paths = []
    for rank in range(1, 13):
        paper_id = f"partial-{rank:02d}"
        neighbor_path = _write_trusted_image(
            tmp_path, paper_id, f"{paper_id}-neighbor.png"
        )
        primary_path = _write_trusted_image(
            tmp_path, paper_id, f"{paper_id}-primary.png"
        )
        partial_primary_paths.append(str(primary_path))
        records.extend(
            [
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}#neighbor",
                    "chunk_type": "figure",
                    "text": "Neighbouring figure",
                    "metadata": {
                        "page": 1,
                        "figure_id": "Figure N",
                        "image_path": str(neighbor_path),
                    },
                },
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}#primary",
                    "chunk_type": "figure",
                    "text": "Explicitly cited partial evidence",
                    "metadata": {
                        "page": 2,
                        "figure_id": "Figure P",
                        "image_path": str(primary_path),
                    },
                },
            ]
        )
        judgments.append(
            {
                "paper_id": paper_id,
                "rank": rank,
                "relevant": True,
                "label": "partial_answer",
                "evidence": [
                    {
                        "chunk_id": f"{paper_id}#primary",
                        "quote_or_value": "partial",
                    }
                ],
            }
        )

    # Put the direct paper last and at a worse rank to prove label priority is
    # stronger than input/rank order. Its seventh cited figure must still fit.
    judgments.append(
        {
            "paper_id": "direct",
            "rank": 50,
            "relevant": True,
            "label": "direct_answer",
            "evidence": direct_evidence,
        }
    )
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)),
        FakeLLM(),
        answer_neighbor_chunks=1,
        max_answer_images=10,
    )

    context = reader._answer_context(_query(), judgments)

    assert context["image_paths"][:7] == direct_image_paths
    assert context["image_paths"][7:] == partial_primary_paths[:3]
    assert len(context["image_paths"]) == 10
    assert direct_image_paths[-1] in context["image_paths"]
    assert all("neighbor" not in path for path in context["image_paths"])


def test_answer_images_round_robin_papers_within_the_same_label(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    records = []
    paths_by_paper = {}
    for paper_id, rank, image_count in (
        ("partial-rank1", 1, 5),
        ("partial-rank5", 5, 1),
        ("partial-rank9", 9, 1),
    ):
        if paper_id == "partial-rank1":
            # A non-image primary citation before this paper's figures must
            # not let a lower-ranked paper's first image jump ahead.
            records.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}#text",
                    "chunk_type": "text_span",
                    "text": "Relevant prose before the figures.",
                    "metadata": {"page": 1},
                }
            )
        paper_paths = []
        for image_index in range(1, image_count + 1):
            image_path = _write_trusted_image(
                tmp_path, paper_id, f"{paper_id}-{image_index}.png"
            )
            paper_paths.append(str(image_path))
            records.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}#fig{image_index}",
                    "chunk_type": "figure",
                    "text": f"Partial evidence {image_index}",
                    "metadata": {
                        "page": image_index,
                        "figure_id": f"Figure {image_index}",
                        "image_path": str(image_path),
                    },
                }
            )
        paths_by_paper[paper_id] = paper_paths
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    rank_by_paper = {
        "partial-rank1": 1,
        "partial-rank5": 5,
        "partial-rank9": 9,
    }
    # Deliberately use non-rank input order; image ordering must use rank.
    judgments = [
        {
            "paper_id": paper_id,
            "rank": rank_by_paper[paper_id],
            "relevant": True,
            "label": "partial_answer",
            "evidence": (
                [
                    {
                        "chunk_id": f"{paper_id}#text",
                        "quote_or_value": "prose",
                    }
                ]
                if paper_id == "partial-rank1"
                else []
            )
            + [
                {
                    "chunk_id": f"{paper_id}#fig{image_index}",
                    "quote_or_value": "partial",
                }
                for image_index in range(
                    1, len(paths_by_paper[paper_id]) + 1
                )
            ],
        }
        for paper_id in ("partial-rank9", "partial-rank1", "partial-rank5")
    ]
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)),
        FakeLLM(),
        answer_neighbor_chunks=0,
        max_answer_images=5,
    )

    context = reader._answer_context(_query(), judgments)

    assert context["image_paths"] == [
        paths_by_paper["partial-rank1"][0],
        paths_by_paper["partial-rank5"][0],
        paths_by_paper["partial-rank9"][0],
        paths_by_paper["partial-rank1"][1],
        paths_by_paper["partial-rank1"][2],
    ]


def test_answer_rejects_chunk_without_official_locator(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#bad",
                "chunk_type": "text_span",
                "text": "The value is 42.",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llm = FakeLLM(
        responses=[
            _judgment("direct_answer", "p1#bad"),
            json.dumps(_structured_answer_payload({"p1": ["p1#bad"]})),
            json.dumps(_structured_answer_payload({"p1": ["p1#bad"]})),
        ]
    )
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    judgment = reader.judge_candidate(_query(), candidate)

    with pytest.raises(ReadingResponseError, match="valid official"):
        reader.answer_from_judgments(_query(), (candidate,), [judgment])


def test_answer_repairs_invalid_locator_once_and_keeps_fail_closed_evidence(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": "Table 1: Method X reports 42.",
            "metadata": {"page": 3, "table_id": "Table 1"},
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#bad-table",
            "chunk_type": "table",
            "text": "A comparison table with missing caption metadata.",
            "metadata": {"page": 4, "table_id": None},
        },
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    first_answer = json.dumps(
        _structured_answer_payload(
            {"p1": ["p1#table"], "p2": ["p2#bad-table"]}
        )
    )
    repaired_answer = json.dumps(
        _structured_answer_payload({"p1": ["p1#table"]})
    )
    llm = FakeLLM(responses=[first_answer, repaired_answer])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm, answer_neighbor_chunks=0)
    candidates = (
        CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
        CandidatePaper("p2", 2, "Paper Two", "ACL", 2025),
    )
    judgments = [
        {
            "paper_id": "p1",
            "rank": 1,
            "relevant": True,
            "label": "direct_answer",
            "evidence": [{"chunk_id": "p1#table", "quote_or_value": "42"}],
        },
        {
            "paper_id": "p2",
            "rank": 2,
            "relevant": True,
            "label": "partial_answer",
            "evidence": [
                {"chunk_id": "p2#bad-table", "quote_or_value": "comparison"}
            ],
        },
    ]

    prediction, answer_record = reader.answer_from_judgments(
        _query(), candidates, judgments
    )

    assert prediction.gold_papers == [{"paper_id": "p1"}]
    assert len(llm.calls) == 2
    assert '\"submission_eligible\":false' in llm.calls[0]
    eligible_line = llm.calls[1].split("Eligible evidence chunk_ids:", 1)[1].split(
        "\n", 1
    )[0]
    assert "p1#table" in eligible_line
    assert "p2#bad-table" not in eligible_line
    assert len(answer_record["attempts"]) == 2
    assert "valid official" in answer_record["attempts"][0]["parse_error"]
    assert answer_record["attempts"][1]["parse_error"] is None


def test_answer_repairs_comparison_polarity_once(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    query = Query(
        query_id="q-compare",
        question="Does the first value exceed the second value?",
        answer_types=["freeform"],
    )
    first = _structured_answer_payload({"p1": ["p1#table"]})
    first["derivation"]["facts"] = [
        {
            "id": "f_left",
            "name": "first value",
            "value": 30,
            "value_kind": "reported",
            "paper_id": "p1",
            "chunk_ids": ["p1#table"],
        },
        {
            "id": "f_right",
            "name": "second value",
            "value": 21,
            "value_kind": "reported",
            "paper_id": "p1",
            "chunk_ids": ["p1#table"],
        },
    ]
    first["derivation"]["operations"] = [
        {
            "id": "compare_values",
            "kind": "compare",
            "fact_ids": ["f_left", "f_right"],
            "left": 30,
            "operator": ">",
            "right": 21,
            "result": False,
            "answer_binding": {
                "answer_path": "answer.freeform.text",
                "expected": False,
                "answer_fragment": "No",
            },
        }
    ]
    first["derivation"]["answer_bindings"] = [
        {
            "answer_path": "answer.freeform.text",
            "source_type": "operation",
            "source_id": "compare_values",
            "answer_fragment": "No",
        }
    ]
    first["derivation"]["final_semantic_answer"] = "No"
    first["answer"] = {"freeform": {"text": "No"}}
    repaired = _structured_answer_payload({"p1": ["p1#table"]})
    repaired["derivation"]["facts"] = first["derivation"]["facts"]
    repaired["derivation"]["operations"] = [
        {
            "id": "compare_values",
            "kind": "compare",
            "fact_ids": ["f_left", "f_right"],
            "left": 30,
            "operator": ">",
            "right": 21,
            "result": True,
            "answer_binding": {
                "answer_path": "answer.freeform.text",
                "expected": True,
                "answer_fragment": "Yes",
            },
        }
    ]
    repaired["derivation"]["answer_bindings"] = [
        {
            "answer_path": "answer.freeform.text",
            "source_type": "operation",
            "source_id": "compare_values",
            "answer_fragment": "Yes",
        }
    ]
    repaired["derivation"]["final_semantic_answer"] = "Yes"
    repaired["answer"] = {"freeform": {"text": "Yes"}}
    llm = FakeLLM(responses=[json.dumps(first), json.dumps(repaired)])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm, answer_neighbor_chunks=0)
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    judgment = {
        "paper_id": "p1",
        "rank": 1,
        "relevant": True,
        "label": "direct_answer",
        "evidence": [{"chunk_id": "p1#table", "quote_or_value": "30 and 21"}],
    }

    prediction, answer_record = reader.answer_from_judgments(
        query, (candidate,), [judgment]
    )

    assert prediction.answer.freeform == {"text": "Yes"}
    assert len(llm.calls) == 2
    assert "30 > 21 is True" in answer_record["attempts"][0]["parse_error"]
    assert "Correct the JSON once and recompute" in llm.calls[1]


def test_candidate_cache_key_changes_when_image_bytes_change(tmp_path):
    image = _write_trusted_image(tmp_path, "p1", "table.png")
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#table",
                "chunk_type": "table",
                "text": "Table 1",
                "metadata": {
                    "page": 1,
                    "table_id": "Table 1",
                    "image_path": str(image),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), FakeLLM()
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")
    before = reader.judgment_cache_key(_query(), candidate, records)

    image.write_bytes(VALID_PNG_ALTERNATE)
    after = reader.judgment_cache_key(_query(), candidate, records)

    assert before != after


def test_candidate_cache_key_hashes_chunks_omitted_from_single_context(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_compaction_corpus(corpus)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus), FakeLLM(), max_paper_context_chars=8_000
    )
    query = _query()
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")
    context = reader._paper_context(query, candidate, records)
    omitted_chunk_id = context["omitted_chunk_ids"][0]
    omitted_record = next(
        record for record in records if record["chunk_id"] == omitted_chunk_id
    )
    before = reader.judgment_cache_key(query, candidate, records)

    omitted_record["text"] += " changed outside the selected context"
    after = reader.judgment_cache_key(query, candidate, records)

    assert omitted_chunk_id not in context["selected_chunk_ids"]
    assert before != after


def test_candidate_cache_key_hashes_images_not_selected_for_attachment(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=12)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)),
        FakeLLM(),
        max_paper_images=1,
    )
    query = Query(
        "q_visual_12",
        "What value is visible in Figure 12?",
        ["freeform"],
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")
    context = reader._paper_context(query, candidate, records)
    unselected_path = next(
        path for path in image_paths if path not in context["image_paths"]
    )
    before = reader.judgment_cache_key(query, candidate, records)

    Path(unselected_path).write_bytes(VALID_PNG_ALTERNATE)
    after = reader.judgment_cache_key(query, candidate, records)

    assert len(context["image_paths"]) == 1
    assert before != after


def test_image_policy_rejection_falls_back_to_text_and_is_recorded(tmp_path):
    class ImagePolicyError(Exception):
        status_code = 400
        body: ClassVar[dict[str, str]] = {"code": "content_policy_violation"}

    class PolicyRejectingLLM:
        def __init__(self):
            self.calls = []

        def complete_with_metadata(self, prompt, image_paths=None):
            self.calls.append({"prompt": prompt, "image_paths": image_paths})
            if image_paths:
                raise ImagePolicyError("content_policy_violation")
            return {"text": "{}", "usage": None}

    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = PolicyRejectingLLM()
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)

    result = reader._complete("prompt", ["blocked.jpg"])

    assert [item["image_paths"] for item in llm.calls] == [["blocked.jpg"], None]
    assert "No image is attached" in llm.calls[1]["prompt"]
    assert "Ignore every earlier image mapping" in llm.calls[1]["prompt"]
    assert "do not claim visual inspection" in llm.calls[1]["prompt"]
    assert result["image_fallback_reason"] == "content_policy_violation"
    assert result["requested_image_count"] == 1
    assert result["attached_image_count"] == 0
    assert result["provider_invocation_count"] == 2


def test_many_images_are_ranked_and_attached_in_one_paper_call(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=12)
    raw = json.loads(_judgment("direct_answer", "p1#fig12", answer_meaning="12"))
    raw["visual"] = {"required": True, "status": "inspected"}
    llm = _RecordingMultimodalLLM([json.dumps(raw)])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)),
        llm,
        max_paper_images=10,
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    query = Query(
        "q_visual_12",
        "How many panels are visible in Figure 12?",
        ["freeform"],
    )

    judgment = reader.judge_candidate(query, candidate)

    assert len(llm.calls) == 1
    attached = llm.calls[0]["image_paths"]
    assert len(attached) == 2
    assert len(set(attached)) == 2
    assert image_paths[11] in attached
    assert image_paths[11] == attached[0]
    prompt = str(llm.calls[0]["prompt"])
    assert "p1#fig12" in prompt
    assert "figure-12.png" in prompt
    assert judgment["base_judgment_call_count"] == 1
    assert judgment["judgment_call_count"] == 1
    assert judgment["label"] == "direct_answer"
    assert judgment["evidence_chunk_ids"] == ["p1#fig12"]
    assert judgment["judgment"]["visual"] == {
        "required": True,
        "status": "inspected",
    }
    assert judgment["paper_readable_image_count"] == 12
    assert judgment["attached_image_count"] == 2
    assert judgment["paper_image_compacted"] is True
    assert len(judgment["omitted_image_chunk_ids"]) == 10
    assert "batch_count" not in judgment
    assert "judgment_image_mode" not in judgment


def test_paper_image_selection_is_deterministic(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=12)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)),
        FakeLLM(),
        max_paper_images=4,
    )
    query = Query(
        "q_visual_12",
        "What value is visible in Figure 12?",
        ["freeform"],
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")

    first = reader._paper_context(query, candidate, records)
    second = reader._paper_context(query, candidate, records)

    assert first == second
    assert len(first["image_paths"]) == 2
    assert first["image_paths"][0].endswith("figure-12.png")
    assert len(set(first["image_paths"])) == 2
    assert first["total_readable_image_count"] == 12
    assert len(first["omitted_image_chunk_ids"]) == 10


def test_implicit_numeric_comparison_keeps_early_figure_fallbacks(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=12)
    reader = PairwiseAOAIReader(
        ChunkStore(corpus, image_root=_image_root(tmp_path)), FakeLLM()
    )
    query = Query(
        "q_implicit_chart",
        "Does Category Cedar have more prompts than Category Flint?",
        ["freeform"],
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)

    context = reader._paper_context(
        query, candidate, reader.chunk_store.load_paper("p1")
    )

    assert 2 <= len(context["image_paths"]) <= 7
    assert image_paths[0] in context["image_paths"]
    assert image_paths[1] in context["image_paths"]


def test_judgment_cache_key_includes_single_context_limits(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=2)
    store = ChunkStore(corpus, image_root=_image_root(tmp_path))
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = store.load_paper("p1")
    readers = (
        PairwiseAOAIReader(
            store,
            FakeLLM(),
            max_paper_context_chars=8_000,
            max_paper_images=1,
        ),
        PairwiseAOAIReader(
            store,
            FakeLLM(),
            max_paper_context_chars=9_000,
            max_paper_images=1,
        ),
        PairwiseAOAIReader(
            store,
            FakeLLM(),
            max_paper_context_chars=8_000,
            max_paper_images=2,
        ),
        PairwiseAOAIReader(
            store,
            FakeLLM(),
            max_paper_context_chars=8_000,
            max_judgment_prompt_chars=20_000,
            max_paper_images=1,
        ),
    )

    keys = {
        reader.judgment_cache_key(_query(), candidate, records)
        for reader in readers
    }

    assert len(keys) == 4


@pytest.mark.parametrize(
    "image_limits",
    [
        {"max_paper_images": 11},
        {"max_answer_images": 11},
    ],
)
def test_reader_rejects_more_images_than_aoai_accepts(tmp_path, image_limits):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)

    with pytest.raises(ValueError, match="between 0 and 10"):
        PairwiseAOAIReader(ChunkStore(corpus), FakeLLM(), **image_limits)
