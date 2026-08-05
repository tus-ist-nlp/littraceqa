from __future__ import annotations

import json
from typing import ClassVar

import pytest

from littraceqa.aoai_pairwise_reader import (
    PairwiseAOAIReader,
    ReadingResponseError,
    merge_batch_judgments,
)
from littraceqa.candidate_handoff import CandidatePaper
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.submission import prediction_to_submission


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
        image_path = tmp_path / f"figure-{index}.png"
        image_path.write_bytes(f"image-{index}".encode())
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
    raw["derivation"]["final_semantic_answer"] = "Epsilon"
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
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
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
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)

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
    assert 'Allowed batch chunk_ids: ["p1#fig1"]' in repair_prompt
    assert [call["phase"] for call in judgment["calls"]] == ["full", "full"]
    assert [call["attempt"] for call in judgment["calls"]] == [
        "initial",
        "evidence_repair",
    ]
    assert "invented or cross-cited" in judgment["calls"][0]["parse_error"]
    assert judgment["calls"][1]["parse_error"] is None
    assert all("usage" in call for call in judgment["calls"])
    assert judgment["calls"][0]["raw_response"] == invented_response
    assert judgment["calls"][1]["raw_response"] == repaired_response


def test_oversized_paper_is_batched_but_merged_as_one_pair_judgment(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus, long_text="Method X value 42. " * 1800)
    llm = FakeLLM(responses=[_judgment("direct_answer", "p1#table")])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        max_batch_chars=8_000,
        batch_overlap_chars=200,
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert judgment["batch_count"] > 1
    assert len(llm.calls) == judgment["batch_count"]
    assert judgment["paper_id"] == "p1"
    assert judgment["evidence_chunk_ids"] == ["p1#table"]


def test_development_metadata_is_rejected_before_any_llm_call(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = FakeLLM(responses=[_judgment("irrelevant")])
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)
    query = _query()
    query.task_family = "multi_paper"

    with pytest.raises(ValueError, match="four official input fields"):
        reader.judge_candidate(
            query, CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
        )
    assert llm.calls == []


def test_irrelevant_batch_evidence_is_not_merged_into_relevant_paper():
    merged = merge_batch_judgments(
        [
            {
                "label": "partial_answer",
                "answerable_from_this_paper": False,
                "satisfied_constraints": ["needed value"],
                "missing_constraints": [],
                "evidence": [{"chunk_id": "p1#table"}],
                "candidate_answer": {"meaning": "42"},
                "confidence": 0.9,
                "reason": "useful",
            },
            {
                "label": "mention_only",
                "answerable_from_this_paper": False,
                "satisfied_constraints": [],
                "missing_constraints": ["not here"],
                "evidence": [{"chunk_id": "p1#noise"}],
                "candidate_answer": {},
                "confidence": 0.8,
                "reason": "mere mention",
            },
        ]
    )

    assert merged["relevant"] is True
    assert merged["evidence_chunk_ids"] == ["p1#table"]


def test_owner_mismatch_overrides_same_numbered_figure_positive_batch():
    positive = {
        "paper_role": "answer_source",
        "label": "direct_answer",
        "answerable_from_this_paper": True,
        "satisfied_constraints": ["Figure 4 panel count"],
        "missing_constraints": [],
        "blocking_mismatches": [],
        "visual": {"required": True, "status": "inspected"},
        "evidence": [{"chunk_id": "fastmoe#fig4"}],
        "candidate_answer": {"units": [{"name": "panels", "value": 2}]},
        "confidence": 0.95,
        "reason": "this batch contains a Figure 4",
    }
    owner_conflict = {
        "paper_role": "distractor",
        "label": "irrelevant",
        "answerable_from_this_paper": False,
        "satisfied_constraints": [],
        "missing_constraints": ["WavePipe Figure 4"],
        "blocking_mismatches": ["candidate is FastMoE, not WavePipe"],
        "visual": {"required": True, "status": "inspected"},
        "evidence": [],
        "candidate_answer": {},
        "confidence": 0.99,
        "reason": "wrong owning paper",
    }

    merged = merge_batch_judgments([positive, owner_conflict])

    assert merged["label"] == "irrelevant"
    assert merged["relevant"] is False
    assert merged["identity_conflict"] is True
    assert merged["paper_role"] == "distractor"


def test_owner_mismatch_veto_does_not_depend_on_english_phrase():
    positive = {
        "paper_role": "answer_source",
        "label": "direct_answer",
        "answerable_from_this_paper": True,
        "satisfied_constraints": ["Figure 4 panel count"],
        "missing_constraints": [],
        "blocking_mismatches": [],
        "visual": {"required": True, "status": "inspected"},
        "evidence": [{"chunk_id": "fastmoe#fig4"}],
        "candidate_answer": {"units": [{"name": "panels", "value": 2}]},
        "confidence": 0.95,
        "reason": "this batch contains a Figure 4",
    }
    owner_conflict = {
        **positive,
        "paper_role": "distractor",
        "label": "irrelevant",
        "answerable_from_this_paper": False,
        "blocking_mismatches": ["著者と題名が指定対象に一致しない"],
        "evidence": [],
        "candidate_answer": {},
    }

    merged = merge_batch_judgments([positive, owner_conflict])

    assert merged["label"] == "irrelevant"
    assert merged["identity_conflict"] is True


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
            batch_index=1,
        )


def test_stage_one_rejects_claimed_visual_inspection_without_attachment(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=1)
    payload = json.loads(_judgment("direct_answer", "p1#fig1"))
    payload["visual"] = {"required": True, "status": "inspected"}
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
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
            batch_index=1,
            attached_image_count=0,
        )

    parsed = reader._parse_judgment(
        query=_query(),
        candidate=CandidatePaper("p1", 1, "Paper One", "ACL", 2025),
        payload_text=json.dumps(payload),
        allowed_records=records,
        batch_index=1,
        attached_image_count=1,
    )
    assert parsed["visual"] == {"required": True, "status": "inspected"}


def test_later_satisfied_constraint_resolves_same_missing_constraint():
    base = {
        "paper_role": "answer_source",
        "label": "partial_answer",
        "answerable_from_this_paper": False,
        "blocking_mismatches": [],
        "visual": {"required": False, "status": "not_needed"},
        "evidence": [{"chunk_id": "p1#c1"}],
        "candidate_answer": {"units": [{"name": "value", "value": 42}]},
        "confidence": 0.9,
        "reason": "partial batch",
    }
    merged = merge_batch_judgments(
        [
            {
                **base,
                "satisfied_constraints": [],
                "missing_constraints": [" Dataset Y score "],
            },
            {
                **base,
                "satisfied_constraints": ["dataset y score"],
                "missing_constraints": [],
                "evidence": [{"chunk_id": "p1#c2"}],
            },
        ]
    )

    assert merged["satisfied_constraints"] == ["dataset y score"]
    assert merged["missing_constraints"] == []


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
        "visual_conflict": True,
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
            "candidate_answers_by_batch": [],
            "reason": reason,
            "visual_conflict": True,
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
            query, [{**judgment, "visual_conflict": False}]
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


def test_q052_table_prompt_requires_exact_schema_keys_and_source_cells(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        query_id="q_052",
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
            "candidate_answer": {"Method": "X", "Score": ".9", "Aux": "-"},
            "reason": "Table 1 displays the requested row",
            "visual_conflict": False,
        }
    ]
    context = {
        "text": "Method | Score | Aux\nX | .9 | -",
        "records_by_id": {},
        "image_paths": [],
    }

    prompt = reader._answer_prompt(query, relevant, context)

    assert "Use every table_schema name verbatim" in prompt
    assert "exact string displayed in the cited" in prompt
    assert "Do not append %, units, or explanatory prose" in prompt
    assert "unless they literally appear in" in prompt
    assert "Preserve punctuation and typography byte-for-byte as displayed" in prompt
    assert "A displayed `.9` remains `.9`, not `0.9`" in prompt
    assert "printed dash or" in prompt
    assert "minus-like missing-value mark" in prompt
    assert "ASCII string `-`" in prompt
    assert "only a genuinely" in prompt
    assert "Never replace a dash or blank" in prompt
    assert "attached table image conflicts with lossy OCR or extracted Markdown" in prompt
    assert "use the cell visibly printed in the image" in prompt
    assert "Every emitted cell must be directly" in prompt
    assert '"Method":"source string"' in prompt
    assert '"Score":"source string"' in prompt


def test_q055_prompt_preserves_conflicting_batch_answers_for_source_reconciliation(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        query_id="q_055",
        question="Does the first method outperform the second method?",
        answer_types=["freeform"],
        table_schema=[],
    )
    batch_answers = [
        {"source": "text/table", "answer": "Yes", "values": ["~30", "~21"]},
        {
            "source": "visual",
            "answer": "No",
            "reason": "axis labels were shifted",
        },
        {"source": "visual/table", "answer": "Yes", "values": ["~30", "~21"]},
    ]
    judgment = {
        "paper_id": "p1",
        "rank": 1,
        "label": "direct_answer",
        "satisfied_constraints": ["requested comparison"],
        "missing_constraints": [],
        "candidate_answer": batch_answers[0],
        "candidate_answers_by_batch": batch_answers,
        "reason": "The accepted evidence includes text, a table, and a chart.",
        "visual_conflict": True,
    }
    context = {
        "text": "The table reports about 30 for the first method and 21 for the second.",
        "records_by_id": {},
        "image_paths": [],
    }

    prompt = reader._answer_prompt(query, [judgment], context)
    summary_payload = prompt.split(
        "Accepted paper summary (fallible hints, not evidence):\n", 1
    )[1].split("\n\n", 1)[0]
    summary = json.loads(summary_payload)

    assert summary[0]["candidate_answers_by_batch"] == batch_answers
    assert summary[0]["batch_answer_conflict"] is True
    assert summary[0]["candidate_answer"] == {}
    assert "Stage-1 summaries are fallible hints, never evidence" in prompt
    assert "A2_yes_no_polarity" in prompt
    assert '"kind":"compare"' in prompt
    assert "final polarity and selected option must agree" in prompt

    changed_judgment = {
        **judgment,
        "candidate_answers_by_batch": [
            *batch_answers[:-1],
            {"source": "visual/table", "answer": "No"},
        ],
    }
    assert reader.answer_cache_key(query, [judgment]) != reader.answer_cache_key(
        query, [changed_judgment]
    )


def test_q056_table_prompt_uses_canonical_row_keys_and_splits_named_settings(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = Query(
        query_id="q_056",
        question=(
            "Report AP-BPTT under the short-context and long-context settings."
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
            "candidate_answers_by_batch": [],
            "reason": "The source table spells the method AT-BPTT.",
            "visual_conflict": False,
        }
    ]
    context = {
        "text": "Method | Setting | Score\nAT-BPTT | short-context | 1.2 ± 0.3",
        "records_by_id": {},
        "image_paths": [],
    }

    prompt = reader._answer_prompt(query, relevant, context)

    assert "row-key entity or method name" in prompt
    assert "canonical spelling visibly" in prompt
    assert "question contains an obvious typo" in prompt
    assert "numeric uncertainty compactly as `x±y` with no spaces around `±`" in prompt
    assert "two\n  separately requested rows" in prompt
    assert "not one impossible combined setting" in prompt
    assert "Never invent" in prompt
    assert "Prefer the owning paper" in prompt
    assert "one direct object chunk per answer unit" in prompt


def test_stage_two_summary_bounds_batch_answer_hints(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    query = _query()
    batch_answers = [{"batch": index, "answer": "x"} for index in range(100)]
    judgment = {
        "paper_id": "p1",
        "rank": 1,
        "label": "direct_answer",
        "satisfied_constraints": [],
        "missing_constraints": [],
        "candidate_answer": {},
        "candidate_answers_by_batch": batch_answers,
        "reason": "bounded diagnostic hints",
        "visual_conflict": False,
    }
    context = {"text": "accepted evidence", "records_by_id": {}, "image_paths": []}

    prompt = reader._answer_prompt(query, [judgment], context)
    summary_payload = prompt.split(
        "Accepted paper summary (fallible hints, not evidence):\n", 1
    )[1].split("\n\n", 1)[0]
    summary = json.loads(summary_payload)

    assert summary[0]["candidate_answers_by_batch"] == batch_answers[:64]


def test_answer_images_prioritize_direct_primary_evidence_without_starvation(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    records = []
    direct_image_paths = []
    direct_evidence = []
    for index in range(1, 8):
        image_path = tmp_path / f"direct-{index}.png"
        image_path.write_bytes(f"direct-{index}".encode())
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
        neighbor_path = tmp_path / f"{paper_id}-neighbor.png"
        primary_path = tmp_path / f"{paper_id}-primary.png"
        neighbor_path.write_bytes(b"neighbor")
        primary_path.write_bytes(b"primary")
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
        ChunkStore(corpus),
        FakeLLM(),
        answer_neighbor_chunks=1,
        max_answer_images=12,
    )

    context = reader._answer_context(_query(), judgments)

    assert context["image_paths"][:7] == direct_image_paths
    assert context["image_paths"][7:] == partial_primary_paths[:5]
    assert len(context["image_paths"]) == 12
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
            image_path = tmp_path / f"{paper_id}-{image_index}.png"
            image_path.write_bytes(f"{paper_id}-{image_index}".encode())
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
        ChunkStore(corpus),
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
    image = tmp_path / "table.png"
    image.write_bytes(b"first-image")
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
    reader = PairwiseAOAIReader(ChunkStore(corpus), FakeLLM())
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = reader.chunk_store.load_paper("p1")
    before = reader.judgment_cache_key(_query(), candidate, records)

    image.write_bytes(b"second-image")
    after = reader.judgment_cache_key(_query(), candidate, records)

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


def test_full_image_mode_remains_default_and_uses_image_limited_batches(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=7)
    llm = _RecordingMultimodalLLM([_judgment("irrelevant")])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus), llm, max_images_per_batch=6
    )
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)

    judgment = reader.judge_candidate(_query(), candidate)

    assert reader.judgment_image_mode == "full"
    assert judgment["judgment_image_mode"] == "full"
    assert judgment["visual_refinement_status"] == "not_applicable_full"
    assert [call["image_paths"] for call in llm.calls] == [
        image_paths[:6],
        image_paths[6:],
    ]
    assert [item["phase"] for item in judgment["batch_judgments"]] == [
        "full",
        "full",
    ]
    assert [item["phase"] for item in judgment["calls"]] == ["full", "full"]

    explicit_full = PairwiseAOAIReader(
        ChunkStore(corpus),
        FakeLLM(),
        max_images_per_batch=6,
        judgment_image_mode="full",
    )
    records = reader.chunk_store.load_paper("p1")
    assert reader.judgment_cache_key(_query(), candidate, records) == (
        explicit_full.judgment_cache_key(_query(), candidate, records)
    )


def test_hybrid_text_screen_ignores_image_cap_and_skips_irrelevant_images(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=7)
    llm = _RecordingMultimodalLLM([_judgment("irrelevant")])
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        max_images_per_batch=6,
        judgment_image_mode="text_then_relevant_images",
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    # Seven images would make two full-mode batches, but the text screen is
    # governed only by the character limit and must remain a single call.
    assert len(llm.calls) == 1
    assert llm.calls[0]["image_paths"] == []
    assert judgment["batch_count"] == 1
    assert judgment["calls"][0]["phase"] == "text_screen"
    assert judgment["visual_refinement_status"] == "skipped_label"
    assert judgment["visual_conflict"] is False
    assert "visual_judgment" not in judgment


def test_hybrid_visual_refine_upgrades_label_and_unions_evidence(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=2)
    llm = _RecordingMultimodalLLM(
        [
            _judgment("partial_answer", "p1#fig1"),
            _judgment("direct_answer", "p1#fig2"),
        ]
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        judgment_image_mode="text_then_relevant_images",
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert [call["image_paths"] for call in llm.calls] == [[], image_paths]
    assert [item["phase"] for item in judgment["calls"]] == [
        "text_screen",
        "visual_refine",
    ]
    assert judgment["visual_refinement_status"] == "complete"
    assert judgment["text_judgment"]["label"] == "partial_answer"
    assert judgment["visual_judgment"]["label"] == "direct_answer"
    assert judgment["label"] == "direct_answer"
    assert judgment["evidence_chunk_ids"] == ["p1#fig1", "p1#fig2"]
    assert judgment["visual_conflict"] is False


def test_hybrid_refines_text_unreadable_paper_with_available_image(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=1)
    unreadable = json.loads(_judgment("unreadable"))
    unreadable.update(
        {
            "paper_role": "target_owner",
            "missing_constraints": ["visible Figure 1 panels"],
            "visual": {"required": True, "status": "missing"},
        }
    )
    direct = json.loads(_judgment("direct_answer", "p1#fig1"))
    direct["visual"] = {"required": True, "status": "inspected"}
    llm = _RecordingMultimodalLLM(
        [json.dumps(unreadable), json.dumps(direct)]
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        judgment_image_mode="text_then_relevant_images",
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert [call["image_paths"] for call in llm.calls] == [[], image_paths]
    assert judgment["text_judgment"]["label"] == "unreadable"
    assert judgment["visual_judgment"]["label"] == "direct_answer"
    assert judgment["label"] == "direct_answer"
    assert judgment["visual_refinement_status"] == "complete"


def test_hybrid_same_label_prefers_visual_candidate_answer_and_unions_evidence(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=2)
    llm = _RecordingMultimodalLLM(
        [
            _judgment("partial_answer", "p1#fig1", answer_meaning="2.60"),
            _judgment("partial_answer", "p1#fig2", answer_meaning="2.06"),
        ]
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        judgment_image_mode="text_then_relevant_images",
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert [call["image_paths"] for call in llm.calls] == [[], image_paths]
    assert judgment["text_judgment"]["label"] == "partial_answer"
    assert judgment["visual_judgment"]["label"] == "partial_answer"
    assert judgment["label"] == "partial_answer"
    assert judgment["candidate_answer"] == {"meaning": "2.06"}
    assert judgment["evidence_chunk_ids"] == ["p1#fig1", "p1#fig2"]
    assert judgment["visual_conflict"] is False


def test_hybrid_irrelevant_visual_preserves_text_evidence_and_flags_conflict(
    tmp_path,
):
    corpus = tmp_path / "chunks.jsonl"
    image_paths = _write_image_corpus(corpus, tmp_path, image_count=1)
    llm = _RecordingMultimodalLLM(
        [
            _judgment("direct_answer", "p1#fig1"),
            _judgment("irrelevant"),
        ]
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        judgment_image_mode="text_then_relevant_images",
    )

    judgment = reader.judge_candidate(
        _query(), CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    )

    assert [call["image_paths"] for call in llm.calls] == [[], image_paths]
    assert judgment["label"] == "direct_answer"
    assert judgment["relevant"] is True
    assert judgment["evidence_chunk_ids"] == ["p1#fig1"]
    assert judgment["visual_judgment"]["label"] == "irrelevant"
    assert judgment["visual_conflict"] is True


def test_judgment_cache_key_includes_hybrid_mode_and_refine_labels(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_image_corpus(corpus, tmp_path, image_count=1)
    store = ChunkStore(corpus)
    candidate = CandidatePaper("p1", 1, "Paper One", "ACL", 2025)
    records = store.load_paper("p1")
    full = PairwiseAOAIReader(store, FakeLLM())
    hybrid = PairwiseAOAIReader(
        store, FakeLLM(), judgment_image_mode="text_then_relevant_images"
    )
    direct_only = PairwiseAOAIReader(
        store,
        FakeLLM(),
        judgment_image_mode="text_then_relevant_images",
        image_refine_labels=["direct_answer"],
    )

    keys = {
        reader.judgment_cache_key(_query(), candidate, records)
        for reader in (full, hybrid, direct_only)
    }

    assert len(keys) == 3


def test_hybrid_refine_labels_cannot_be_empty(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)

    with pytest.raises(ValueError, match="non-empty subset"):
        PairwiseAOAIReader(
            ChunkStore(corpus),
            FakeLLM(),
            judgment_image_mode="text_then_relevant_images",
            image_refine_labels=[],
        )
