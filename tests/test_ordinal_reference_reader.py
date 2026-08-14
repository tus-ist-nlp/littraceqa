from __future__ import annotations

import json

from littraceqa.aoai_pairwise_reader import (
    FIXED_SELECTED_CHECKPOINT_KIND,
    FIXED_SELECTED_PAPER_POLICY,
    PairwiseAOAIReader,
)
from littraceqa.candidate_handoff import CandidatePaper
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _record(
    chunk_id: str,
    text: str,
    *,
    page: int,
    section: str,
) -> dict[str, object]:
    return {
        "paper_id": "p1",
        "chunk_id": chunk_id,
        "chunk_type": "text_span",
        "text": text,
        "metadata": {"page": page, "section": section},
    }


def _answer_payload(*, author: str, chunk_id: str) -> dict[str, object]:
    return {
        "status": "ready",
        "paper_relevance": [
            {
                "paper_id": "p1",
                "role": "target_owner",
                "reason": "The selected paper owns the requested bibliography.",
            }
        ],
        "papers": [{"paper_id": "p1", "evidence_chunk_ids": [chunk_id]}],
        "derivation": {
            "facts": [
                {
                    "id": "f_author",
                    "name": "first author of the requested reference",
                    "value": author,
                    "value_kind": "reported",
                    "paper_id": "p1",
                    "chunk_ids": [chunk_id],
                }
            ],
            "operations": [],
            "answer_bindings": [
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "f_author",
                    "answer_fragment": author,
                },
                {
                    "answer_path": "answer.table.rows[0].Author",
                    "source_type": "fact",
                    "source_id": "f_author",
                    "answer_fragment": author,
                },
            ],
            "final_semantic_answer": author,
        },
        "answer": {
            "freeform": {"text": author},
            "table": {"rows": [{"Author": author}]},
        },
        "support": [
            {
                "answer_path": "answer.freeform.text",
                "paper_id": "p1",
                "chunk_ids": [chunk_id],
            },
            {
                "answer_path": "answer.table.rows[0].Author",
                "paper_id": "p1",
                "chunk_ids": [chunk_id],
            },
        ],
        "completeness": {
            "answered_parts": ["first author"],
            "missing": [],
        },
    }


def test_q027_shaped_reader_supplies_label_three_and_repairs_body_order_answer(
    tmp_path,
):
    body = _record(
        "p1#body",
        "The introduction first cites [4], then diffusion models [24, 54, 57].",
        page=1,
        section="Introduction",
    )
    requested = _record(
        "p1#refs-early",
        "[CVPR 2025] EnergyMoGen\n"
        "[1] First Author. First work, 2021.\n"
        "[2] Second Author. Second work, 2022.\n"
        "[3] Nikos Athanasiou, Mathis Petrovich, Michael J. Black, and "
        "Gul Varol. SINC, 2023.\n"
        "[4] Fourth Author. Fourth work, 2024.",
        page=9,
        section="References",
    )
    wrong = _record(
        "p1#refs-late",
        "[54] Robin Rombach, Andreas Blattmann, et al. Latent diffusion "
        "models, 2022.",
        page=10,
        section="References",
    )
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in [body, requested, wrong]),
        encoding="utf-8",
    )

    query = Query(
        query_id="q027-shaped",
        question=(
            "Who is the first author of the third reference cited in a CVPR "
            "2025 paper on compositional human motion generation that uses an "
            "energy-based diffusion model in latent space?"
        ),
        answer_types=["freeform", "table"],
        table_schema=[
            {"name": "Author", "type": "string", "is_row_key": True}
        ],
    )
    wrong_payload = _answer_payload(
        author="Robin Rombach", chunk_id="p1#refs-late"
    )
    corrected_payload = _answer_payload(
        author="Nikos Athanasiou", chunk_id="p1#refs-early"
    )
    llm = FakeLLM(
        responses=[json.dumps(wrong_payload), json.dumps(corrected_payload)]
    )
    reader = PairwiseAOAIReader(
        ChunkStore(corpus),
        llm,
        answer_neighbor_chunks=0,
        paper_set_policy=FIXED_SELECTED_PAPER_POLICY,
    )
    candidate = CandidatePaper(
        "p1",
        1,
        "EnergyMoGen: Compositional Human Motion Generation with Energy-Based "
        "Diffusion Model in Latent Space",
        "CVPR",
        2025,
    )
    judgment = {
        "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
        "paper_id": "p1",
        "rank": 1,
        "title": candidate.title,
        "is_relevant_to_answer": True,
        "has_usable_answer_evidence": True,
        "send_to_answer_agent": True,
        "evidence_chunk_ids": ["p1#body", "p1#refs-late"],
        "evidence": [
            {
                "chunk_id": "p1#body",
                "purpose": "citation_fact",
                "quote_or_value": str(body["text"]),
            },
            {
                "chunk_id": "p1#refs-late",
                "purpose": "citation_fact",
                "quote_or_value": str(wrong["text"]),
            },
        ],
        "extracted_facts": [
            {
                "chunk_id": "p1#body",
                "purpose": "citation_fact",
                "fact": "The third citation occurrence is [54].",
                "source_excerpt": str(body["text"]),
            },
            {
                "chunk_id": "p1#refs-late",
                "purpose": "citation_fact",
                "fact": "Reference [54] starts with Robin Rombach.",
                "source_excerpt": str(wrong["text"]),
            },
        ],
    }

    answer_pool = reader._fixed_selected_answer_pool(  # noqa: SLF001
        query, [candidate], [judgment]
    )
    context = reader._answer_context(query, answer_pool)  # noqa: SLF001
    assert "p1#refs-early" in context["records_by_id"]
    assert "p1#refs-early" in context["python_supplemental_chunk_ids"]
    assert "Nikos Athanasiou" in context["text"]

    prediction, answer_record = reader.answer_from_judgments(
        query, [candidate], [judgment]
    )

    assert prediction.answer.freeform == {"text": "Nikos Athanasiou"}
    assert prediction.answer.table == {
        "schema": query.table_schema,
        "rows": [{"Author": "Nikos Athanasiou"}],
    }
    assert [
        (item.locator.page, item.locator.citation_id)
        for item in prediction.evidence
    ] == [(9, "3")]
    assert len(llm.calls) == 2
    assert "bibliography label [3]" in answer_record["attempts"][0]["parse_error"]
    assert "never means the same-numbered citation occurrence" in llm.calls[1]
    assert '"p1#refs-early"' in llm.calls[1]
