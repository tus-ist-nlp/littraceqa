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
            "label": label,
            "answerable_from_this_paper": label == "direct_answer",
            "satisfied_constraints": ["Method X value"] if chunk_id else [],
            "missing_constraints": [] if chunk_id else ["Method X value"],
            "evidence": evidence,
            "candidate_answer": {"meaning": answer_meaning} if chunk_id else {},
            "confidence": 0.98,
            "reason": "the table contains the value" if chunk_id else "no match",
        }
    )


def _answer() -> str:
    return json.dumps(
        {
            "papers": [
                {"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]}
            ],
            "answer": {"freeform": {"text": "42"}},
            "completeness": {"answered_parts": ["value"], "missing": []},
        }
    )


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

    summary_text = prompt.split("Accepted paper summary:\n", 1)[1].split(
        "\n\n", 1
    )[0]
    summary = json.loads(summary_text)
    assert summary == [
            {
                "paper_id": "ecm",
                "title": "",
                "rank": 1,
            "label": "partial_answer",
            "satisfied_constraints": ["ECM-XL 102.4M is listed"],
            "missing_constraints": [missing],
            "candidate_answer": {"dataset": "ImageNet", "fid": 2.49},
            "candidate_answers_by_batch": [],
            "reason": reason,
            "visual_conflict": True,
        }
    ]
    assert "dataset, evaluation split, model variant or size" in prompt
    assert (
        "training budget, NFE, and step or checkpoint as hard constraints" in prompt
    )
    assert "different constraint setting" in prompt
    assert "record it in completeness.missing" in prompt
    assert "do not fabricate the corresponding row" in prompt

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
    }
    assert base_key not in changed_keys
    assert len(changed_keys) == 3


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

    assert "use every table_schema name verbatim as its JSON key" in prompt
    assert "do not rename keys or add columns" in prompt
    assert "exact string displayed in the cited source cell" in prompt
    assert "Do not append %, units, explanatory prose" in prompt
    assert "unless they literally appear in that source cell" in prompt
    assert "Preserve punctuation and typography byte-for-byte as displayed" in prompt
    assert "a decimal displayed as `.9` must be returned as `.9`, never `0.9`" in prompt
    assert "printed dash or minus-like missing-value mark" in prompt
    assert "ASCII string `-`, never as an empty string" in prompt
    assert "only a genuinely blank source cell may be empty" in prompt
    assert "Never replace a dash or blank with 'unreported', 'N/A', null" in prompt
    assert "attached table image conflicts with lossy OCR or extracted Markdown" in prompt
    assert "use the cell visibly printed in the image" in prompt
    assert "Every emitted cell must be directly grounded in the cited evidence" in prompt
    assert "silently self-check every row and cell" in prompt
    assert "(1) exact schema keys, (2) exact source typography" in prompt
    assert "(3) leading-dot decimals, (4) printed dash versus genuine blank" in prompt
    assert "(5) no added characters, and (6) cited evidence support" in prompt
    assert '"each exact table_schema name": "exact displayed source cell string"' in prompt


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
    summary_payload = prompt.split("Accepted paper summary:\n", 1)[1].split(
        "\n\n", 1
    )[0]
    summary = json.loads(summary_payload)

    assert summary[0]["candidate_answers_by_batch"] == batch_answers
    assert summary[0]["batch_answer_conflict"] is True
    assert summary[0]["candidate_answer"] == {}
    assert "fallible hints that may conflict" in prompt
    assert "Do not let the first candidate_answer dominate" in prompt
    assert "candidate_answer is intentionally empty" in prompt
    assert "never reconstruct or privilege the old merged shortcut" in prompt
    assert "Resolve all conflicts only from the original chunks and attached images" in prompt
    assert "map rotated axis labels to their bars carefully" in prompt
    assert (
        "cross-check chart comparisons against accepted list, table, and text evidence"
        in prompt
    )
    assert "For every yes/no comparison" in prompt
    assert "final Yes/No polarity agrees with both the numbers and its explanation" in prompt
    assert "if A is greater than B" in prompt
    assert "the answer must be Yes" in prompt

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
    assert "canonical spelling visibly supported by the source" in prompt
    assert "question contains an obvious typo" in prompt
    assert "deliberate exception to byte-for-byte source formatting" in prompt
    assert "numeric uncertainty compactly as `x±y` with no spaces around `±`" in prompt
    assert "question joins two named settings with 'and'" in prompt
    assert "two separately requested rows" in prompt
    assert "not as one impossible combined setting" in prompt
    assert "never invent a missing value" in prompt
    assert "prefer each method's owning/original paper" in prompt
    assert "that paper's own reported result" in prompt
    assert "later paper's comparison or reproduction value" in prompt
    assert "one direct table, figure, or text chunk per requested item" in prompt


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
    summary_payload = prompt.split("Accepted paper summary:\n", 1)[1].split(
        "\n\n", 1
    )[0]
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
            json.dumps(
                {
                    "papers": [
                        {"paper_id": "p1", "evidence_chunk_ids": ["p1#bad"]}
                    ],
                    "answer": {"freeform": {"text": "42"}},
                }
            ),
            json.dumps(
                {
                    "papers": [
                        {"paper_id": "p1", "evidence_chunk_ids": ["p1#bad"]}
                    ],
                    "answer": {"freeform": {"text": "42"}},
                }
            ),
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
        {
            "papers": [
                {"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]},
                {"paper_id": "p2", "evidence_chunk_ids": ["p2#bad-table"]},
            ],
            "answer": {"freeform": {"text": "42"}},
        }
    )
    repaired_answer = json.dumps(
        {
            "papers": [
                {"paper_id": "p1", "evidence_chunk_ids": ["p1#table"]}
            ],
            "answer": {"freeform": {"text": "42"}},
        }
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
            self.calls.append(image_paths)
            if image_paths:
                raise ImagePolicyError("content_policy_violation")
            return {"text": "{}", "usage": None}

    corpus = tmp_path / "chunks.jsonl"
    _write_corpus(corpus)
    llm = PolicyRejectingLLM()
    reader = PairwiseAOAIReader(ChunkStore(corpus), llm)

    result = reader._complete("prompt", ["blocked.jpg"])

    assert llm.calls == [["blocked.jpg"], None]
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
