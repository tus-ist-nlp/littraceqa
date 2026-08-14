from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from littraceqa.candidate_handoff import production_query_from_record
from littraceqa.pairwise_prompts import (
    answer_response_shape,
    render_answer_prompt,
    render_selected_evidence_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REVISION = "bd35dc14cf0483e0ffa51fa2a54d2689c13f9845"
OFFICIAL_TEST = (
    ROOT
    / "artifacts"
    / "official_release"
    / OFFICIAL_REVISION
    / "data"
    / "test.jsonl"
)
pytestmark = pytest.mark.skipif(
    not OFFICIAL_TEST.is_file(),
    reason="requires the external official-release test artifact",
)
FIXED_SELECTED_CONFIG = (
    ROOT / "configs" / "agent_style" / "aoai_selected_paper_reader.yaml"
)

COMMON_QUERY_KEYS = {"query_id", "benchmark", "question", "answer_types"}
DEVELOPMENT_ONLY_QUERY_KEYS = {
    "_gold",
    "answer",
    "evidence",
    "gold_evidence",
    "gold_papers",
    "primary_evidence_type",
    "task_family",
}

AUDIT_PAPER_ID = "synthetic_static_prompt_audit_paper"
AUDIT_CHUNK_ID = f"{AUDIT_PAPER_ID}#text_1"
AUDIT_CANDIDATE = {
    "paper_id": AUDIT_PAPER_ID,
    "rank": 1,
    "title": "Synthetic Static Prompt Audit Paper",
    "venue": "SYNTHETIC",
    "year": 2025,
}
AUDIT_LOCATOR = {"page": 1, "section": "Synthetic static audit"}
AUDIT_CONTEXT_COVERAGE = {
    "paper_context_complete": False,
    "selected_chunk_count": 1,
    "total_chunk_count": 1,
    "omitted_chunk_count": 0,
}
AUDIT_PAPER_TEXT = (
    "[chunk "
    + json.dumps(
        {
            "paper_id": AUDIT_PAPER_ID,
            "chunk_id": AUDIT_CHUNK_ID,
            "source_type": "text_span",
            "locator": AUDIT_LOCATOR,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    + "]\nSYNTHETIC STATIC AUDIT SOURCE WITH NO ANSWER VALUE."
)
AUDIT_SELECTED_SUMMARY = [
    {
        "paper_id": AUDIT_PAPER_ID,
        "title": AUDIT_CANDIDATE["title"],
        "rank": 1,
        "checkpoint_kind": "fixed_selected_evidence",
        "fixed_selected_extraction_was_empty": True,
        "evidence": [
            {
                "chunk_id": AUDIT_CHUNK_ID,
                "source_type": "text_span",
                "locator": AUDIT_LOCATOR,
                "purpose": "answer",
            }
        ],
        "extracted_facts": [],
    }
]
AUDIT_ANSWER_EVIDENCE = (
    "[chunk "
    + json.dumps(
        {
            "paper_id": AUDIT_PAPER_ID,
            "chunk_id": AUDIT_CHUNK_ID,
            "source_type": "text_span",
            "locator": AUDIT_LOCATOR,
            "stage1_selected": True,
            "submission_eligible": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    + "]\nSYNTHETIC STATIC AUDIT EVIDENCE WITH NO ANSWER VALUE."
)


def _load_official_records() -> list[dict]:
    return [
        json.loads(line)
        for line in OFFICIAL_TEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _delimited_json(text: str, opening: str, closing: str) -> dict:
    encoded = text.split(opening, 1)[1].split(closing, 1)[0]
    payload = json.loads(encoded)
    assert isinstance(payload, dict)
    return payload


def _stage2_live_query(text: str) -> tuple[dict, str]:
    live = text.split(
        "LIVE TASK (the synthetic examples above are format demonstrations only)",
        1,
    )[1]
    encoded = live.split("Official query JSON:\n", 1)[1].split(
        "\n\nRequired answer object shape:", 1
    )[0]
    payload = json.loads(encoded)
    assert isinstance(payload, dict)
    return payload, live


def test_all_official_test_fixed_selected_prompts_render_gold_free() -> None:
    records = _load_official_records()
    config = yaml.safe_load(FIXED_SELECTED_CONFIG.read_text(encoding="utf-8"))
    params = config["params"]

    assert config["params"]["paper_set_policy"] == "fixed_selected"
    assert len(records) == 71
    assert len({record["query_id"] for record in records}) == 71
    assert Counter(tuple(record["answer_types"]) for record in records) == {
        ("multiple_choice",): 50,
        ("table",): 21,
    }
    assert len(AUDIT_PAPER_TEXT) <= params["max_paper_context_chars"]
    assert len(AUDIT_ANSWER_EVIDENCE) <= params["answer_context_chars"]

    rendered_query_ids: set[str] = set()
    stage1_sizes: dict[str, int] = {}
    stage2_sizes: dict[str, int] = {}

    for record in records:
        answer_types = record["answer_types"]
        conditional_keys = (
            {"multiple_choice_options"}
            if answer_types == ["multiple_choice"]
            else {"table_schema"}
        )
        assert set(record) == COMMON_QUERY_KEYS | conditional_keys
        assert not (set(record) & DEVELOPMENT_ONLY_QUERY_KEYS)

        # This is the same fail-closed organizer-record projection used by the
        # production reader. It validates the conditional MC/table JSON shapes.
        query = production_query_from_record(record)
        query_payload = query.to_dict()
        assert query_payload == record
        assert not (set(query_payload) & DEVELOPMENT_ONLY_QUERY_KEYS)

        stage1_prompt = render_selected_evidence_prompt(
            query=query,
            query_payload=query_payload,
            candidate_payload=AUDIT_CANDIDATE,
            context_coverage=AUDIT_CONTEXT_COVERAGE,
            paper_text=AUDIT_PAPER_TEXT,
            image_legend="",
        )
        stage2_prompt = render_answer_prompt(
            query=query,
            query_payload=query_payload,
            accepted_summary=AUDIT_SELECTED_SUMMARY,
            evidence_text=AUDIT_ANSWER_EVIDENCE,
            image_legend="",
            answer_shape=answer_response_shape(query),
            max_evidence=params["max_evidence"],
            max_evidence_per_paper=params["max_evidence_per_paper"],
            paper_set_policy=params["paper_set_policy"],
        )

        assert _delimited_json(stage1_prompt, "<query>\n", "\n</query>") == record
        stage2_query_payload, stage2_live = _stage2_live_query(stage2_prompt)
        assert stage2_query_payload == record
        assert "<selected_paper>" in stage1_prompt
        assert "Return only the required one-field JSON object." in stage1_prompt
        assert "Selected-paper extraction ledger" in stage2_live

        # Stage 1 has a hard whole-prompt guard. For this static dummy context,
        # also keep Stage 2 below the smaller configured answer-context budget so
        # template/example growth cannot silently consume the production budget.
        assert len(stage1_prompt) <= params["max_judgment_prompt_chars"]
        assert len(stage2_prompt) <= params["answer_context_chars"]

        shape = answer_response_shape(query)
        assert "answer.freeform" not in stage2_live
        if answer_types == ["multiple_choice"]:
            assert set(shape["answer"]) == {"multiple_choice"}
            assert shape["support"][0]["answer_path"] == "answer.multiple_choice"
            assert "Allowed support answer_path forms for this live query:\n" in stage2_live
            assert '["answer.multiple_choice"]' in stage2_live
            assert "answer.table.rows[" not in stage2_live
        else:
            assert set(shape["answer"]) == {"table"}
            assert shape["support"][0]["answer_path"] == "answer.table.rows[0]"
            expected_columns = [column["name"] for column in record["table_schema"]]
            assert list(shape["answer"]["table"]["rows"][0]) == expected_columns
            assert "answer.table.rows[i] for every emitted row index i" in stage2_live
            assert "answer.multiple_choice" not in stage2_live

        rendered_query_ids.add(query.query_id)
        stage1_sizes[query.query_id] = len(stage1_prompt)
        stage2_sizes[query.query_id] = len(stage2_prompt)

    assert rendered_query_ids == {record["query_id"] for record in records}
    assert max(stage1_sizes.values()) <= params["max_judgment_prompt_chars"]
    assert max(stage2_sizes.values()) <= params["answer_context_chars"]
