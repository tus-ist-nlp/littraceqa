from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "render_aoai_prompts", ROOT / "scripts/render_aoai_prompts.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RENDERER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RENDERER)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _multiple_choice_query() -> dict:
    return {
        "query_id": "test_001",
        "benchmark": "LitTraceQA",
        "question": "Which result is supported by Figure 4?",
        "answer_types": ["freeform", "multiple_choice"],
        "multiple_choice_options": [
            {"label": "A", "text": "Alpha"},
            {"label": "B", "text": "Beta"},
            {"label": "E", "text": "Epsilon"},
        ],
        "task_family": "DEV_FIELD_MUST_NOT_LEAK",
        "primary_evidence_type": "DEV_FIELD_MUST_NOT_LEAK",
        "_gold": {"answer": "GOLD_SENTINEL_MUST_NOT_LEAK"},
    }


def test_json_preview_uses_official_projection_and_provided_context(tmp_path):
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    paper_text = tmp_path / "paper.txt"
    summary = tmp_path / "summary.json"
    evidence = tmp_path / "evidence.txt"
    output = tmp_path / "preview.json"
    _write_jsonl(queries, [_multiple_choice_query()])
    _write_jsonl(
        candidates,
        [
            {
                "query_id": "test_001",
                "candidate_papers": [
                    {
                        "paper_id": "p1",
                        "rank": 1,
                        "title": "Paper One",
                        "venue": "TEST",
                        "year": 2025,
                    },
                    {
                        "paper_id": "p2",
                        "rank": 2,
                        "title": "Paper Two",
                        "venue": "TEST",
                        "year": 2025,
                    },
                ],
            }
        ],
    )
    paper_text.write_text("[chunk p2#fig4]\nREAL_PAPER_TEXT", encoding="utf-8")
    summary.write_text(
        json.dumps([{"paper_id": "p2", "rank": 2, "label": "direct_answer"}]),
        encoding="utf-8",
    )
    evidence.write_text("[chunk p2#fig4]\nREAL_ANSWER_EVIDENCE", encoding="utf-8")

    result = _RENDERER.main(
        [
            "--queries",
            str(queries),
            "--query-id",
            "test_001",
            "--candidates",
            str(candidates),
            "--paper-id",
            "p2",
            "--paper-text-file",
            str(paper_text),
            "--accepted-summary-file",
            str(summary),
            "--evidence-file",
            str(evidence),
            "--stage",
            "all",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    preview = json.loads(output.read_text(encoding="utf-8"))
    assert preview["query_id"] == "test_001"
    assert preview["candidate_source"] == "sidecar"
    assert preview["candidate_payload"]["paper_id"] == "p2"
    assert preview["synthetic_paper_text"] is False
    assert preview["synthetic_answer_context"] is False
    assert "task_family" not in preview["query_payload"]
    assert "primary_evidence_type" not in preview["query_payload"]
    assert "GOLD_SENTINEL_MUST_NOT_LEAK" not in output.read_text(encoding="utf-8")
    assert preview["query_payload"]["multiple_choice_options"][-1] == {
        "label": "E",
        "text": "Epsilon",
    }
    assert [item["stage"] for item in preview["prompts"]] == [
        "judgment",
        "answer",
    ]
    judgment, answer = preview["prompts"]
    assert "REAL_PAPER_TEXT" in judgment["text"]
    assert "REAL_ANSWER_EVIDENCE" in answer["text"]
    assert '"label":"<one of: A, B, E>"' in answer["text"]
    for prompt in preview["prompts"]:
        assert prompt["sha256"] == hashlib.sha256(
            prompt["text"].encode("utf-8")
        ).hexdigest()
        assert prompt["messages"] == [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["text"]},
        ]
        assert prompt["system_sha256"] == hashlib.sha256(
            prompt["system"].encode("utf-8")
        ).hexdigest()


def test_markdown_preview_without_candidate_uses_conspicuous_samples(tmp_path):
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "preview.md"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "table_001",
                "benchmark": "LitTraceQA",
                "question": "What score and pass status does each method report?",
                "answer_types": ["table"],
                "table_schema": [
                    {"name": "Method", "type": "string", "is_row_key": True},
                    {"name": "Score", "type": "number", "is_row_key": False},
                    {"name": "Passed", "type": "boolean", "is_row_key": False},
                ],
            }
        ],
    )

    _RENDERER.main(
        [
            "--queries",
            str(queries),
            "--stage",
            "all",
            "--output",
            str(output),
        ]
    )

    markdown = output.read_text(encoding="utf-8")
    assert "# Pairwise AOAI prompt preview" in markdown
    assert "## Stage 1: candidate judgment" in markdown
    assert "## Stage 2: final answer" in markdown
    assert "Candidate source: `synthetic`" in markdown
    assert "Synthetic paper text: `true`" in markdown
    assert "Synthetic preview paper text" in markdown
    assert "Synthetic preview evidence" in markdown
    assert "A9_native_table_types" in markdown
    assert "Prompt version:" in markdown
    assert "### System message" in markdown
    assert "### User message" in markdown
    assert "SHA-256:" in markdown


def test_preview_cli_has_no_batch_options(tmp_path):
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(queries, [_multiple_choice_query()])
    parser = _RENDERER.build_parser()

    assert "batch_index" not in {action.dest for action in parser._actions}
    assert "batch_count" not in {action.dest for action in parser._actions}
    with pytest.raises(SystemExit):
        parser.parse_args(["--queries", str(queries), "--batch-index", "1"])


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--paper-text-file", "paper text"),
        ("--evidence-file", "answer evidence"),
    ],
)
def test_preview_rejects_empty_supplied_context(tmp_path, flag, message):
    queries = tmp_path / "queries.jsonl"
    empty = tmp_path / "empty.txt"
    _write_jsonl(queries, [_multiple_choice_query()])
    empty.write_text("\n", encoding="utf-8")
    args = _RENDERER.build_parser().parse_args(
        ["--queries", str(queries), flag, str(empty)]
    )

    with pytest.raises(ValueError, match=message):
        _RENDERER.build_preview(args)


def test_markdown_fence_cannot_be_closed_by_mineru_markdown():
    fenced = _RENDERER._markdown_fence("before\n```python\nvalue = 1\n```\nafter")

    assert fenced[0] == "````text"
    assert fenced[-1] == "````"
    assert "```python" in fenced[1]
