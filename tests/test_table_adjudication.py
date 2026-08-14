from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from littraceqa.table_adjudication import (
    AdjudicationError,
    compose_submission,
    create_review,
)


def _write_jsonl(path: Path, records: list[dict], *, compact: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            if compact:
                encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            else:
                encoded = json.dumps(record, ensure_ascii=False)
            handle.write(encoded + "\n")


def _inputs() -> list[dict]:
    return [
        {
            "query_id": "q_mc",
            "benchmark": "LitTraceQA",
            "question": "Choose one.",
            "answer_types": ["multiple_choice"],
            "multiple_choice_options": [
                {"label": "A", "text": "Alpha"},
                {"label": "B", "text": "Beta"},
            ],
        },
        {
            "query_id": "q_table",
            "benchmark": "LitTraceQA",
            "question": "Report the scores.",
            "answer_types": ["table"],
            "table_schema": [
                {"name": "method", "type": "string", "is_row_key": True},
                {"name": "score", "type": "number", "is_row_key": False},
            ],
        },
        {
            "query_id": "q_both",
            "benchmark": "LitTraceQA",
            "question": "Choose and tabulate.",
            "answer_types": ["multiple_choice", "table"],
            "multiple_choice_options": [
                {"label": "A", "text": "Alpha"},
                {"label": "B", "text": "Beta"},
            ],
            "table_schema": [
                {"name": "setting", "type": "string", "is_row_key": True},
                {"name": "value", "type": "string", "is_row_key": False},
            ],
        },
    ]


def _paper(paper_id: str) -> list[dict]:
    return [{"paper_id": paper_id}]


def _evidence(paper_id: str) -> list[dict]:
    return [
        {
            "paper_id": paper_id,
            "source_type": "table",
            "locator": {"page": 2, "table_id": "Table 1"},
        }
    ]


def _base() -> list[dict]:
    return [
        {
            "query_id": "q_mc",
            "gold_papers": _paper("p_mc"),
            "evidence": _evidence("p_mc"),
            "answer": {"multiple_choice": {"gold": "A"}},
        },
        {
            "query_id": "q_table",
            "gold_papers": _paper("p_table"),
            "evidence": _evidence("p_table"),
            "answer": {
                "table": {"rows": [{"method": "Base", "score": 1.0}]}
            },
        },
        {
            "query_id": "q_both",
            "gold_papers": _paper("p_both"),
            "evidence": _evidence("p_both"),
            "answer": {
                "multiple_choice": {"gold": "B"},
                "table": {"rows": [{"setting": "Base", "value": "old"}]},
            },
        },
    ]


def _full_candidate() -> list[dict]:
    candidate = copy.deepcopy(_base())
    candidate[0]["answer"]["multiple_choice"]["gold"] = "B"
    candidate[1]["gold_papers"] = _paper("candidate_paper")
    candidate[1]["evidence"] = _evidence("candidate_paper")
    candidate[1]["answer"]["table"] = {
        "rows": [{"method": "Candidate", "score": 2.0}]
    }
    candidate[2]["answer"]["multiple_choice"]["gold"] = "A"
    candidate[2]["answer"]["table"] = {
        "rows": [{"setting": "Candidate", "value": "new"}]
    }
    return candidate


def _table_only_candidate() -> list[dict]:
    return [
        {
            "query_id": "q_table",
            "answer": {
                "table": {"rows": [{"method": "Alt", "score": 3.0}]}
            },
        },
        {
            "query_id": "q_both",
            "answer": {
                "table": {"rows": [{"setting": "Alt", "value": "third"}]}
            },
        },
    ]


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    inputs = tmp_path / "inputs.jsonl"
    base = tmp_path / "base.jsonl"
    full = tmp_path / "full.jsonl"
    table_only = tmp_path / "table_only.jsonl"
    _write_jsonl(inputs, _inputs(), compact=True)
    _write_jsonl(base, _base(), compact=False)
    _write_jsonl(full, _full_candidate(), compact=True)
    _write_jsonl(table_only, _table_only_candidate(), compact=True)
    return inputs, base, full, table_only


def _complete_decisions(template_path: Path) -> Path:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    selections = {"q_table": "full", "q_both": "table_only"}
    papers = {"q_table": "p_table", "q_both": "p_both"}
    for decision in payload["decisions"]:
        query_id = decision["query_id"]
        decision["selected_candidate"] = selections[query_id]
        decision["source_checked"] = True
        decision["notes"] = "Checked against the cited source table."
        decision["locator"] = [
            {
                "paper_id": papers[query_id],
                "source_type": "table",
                "locator": {"page": 2, "table_id": "Table 1"},
            }
        ]
    path = template_path.with_name("decisions.json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_review_supports_full_and_table_only_candidates(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)

    result = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )

    review = json.loads(Path(result["review_json"]).read_text(encoding="utf-8"))
    template = json.loads(
        Path(result["decision_template"]).read_text(encoding="utf-8")
    )
    assert review["table_query_count"] == 2
    assert [item["coverage"] for item in review["candidates"]] == [
        "full",
        "table_only",
    ]
    assert [item["query_id"] for item in review["queries"]] == [
        "q_table",
        "q_both",
    ]
    assert review["queries"][0]["frozen_gold_papers"] == _paper("p_table")
    assert review["queries"][0]["frozen_evidence"] == _evidence("p_table")
    assert all(not item["source_checked"] for item in template["decisions"])
    markdown = Path(result["review_markdown"]).read_text(encoding="utf-8")
    assert "q_table" in markdown
    assert "Frozen papers and evidence" in markdown
    assert '"table_id": "Table 1"' in markdown


def test_review_refuses_to_overwrite_generated_artifacts(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    kwargs = {
        "inputs_path": inputs,
        "base_path": base,
        "candidate_specs": [("full", full), ("table_only", table_only)],
        "output_dir": tmp_path / "review",
    }
    create_review(**kwargs)

    with pytest.raises(AdjudicationError, match="already exists"):
        create_review(**kwargs)


def test_review_markdown_contains_untrusted_cells_without_breaking_markup(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    records = _table_only_candidate()
    records[0]["answer"]["table"]["rows"][0]["method"] = "Alt ``` | </code>"
    _write_jsonl(table_only, records)

    result = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "safe_markdown",
    )

    markdown = Path(result["review_markdown"]).read_text(encoding="utf-8")
    assert "````json" in markdown
    assert "&#124;" in markdown
    assert "&lt;/code&gt;" in markdown


def test_compose_changes_only_selected_tables_and_preserves_non_table_bytes(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "composed.jsonl"

    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    base_lines = base.read_bytes().splitlines(keepends=True)
    output_lines = output.read_bytes().splitlines(keepends=True)
    assert output_lines[0] == base_lines[0]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    base_records = _base()
    assert records[1]["answer"]["table"]["rows"][0]["method"] == "Candidate"
    assert records[2]["answer"]["table"]["rows"][0]["setting"] == "Alt"
    assert records[2]["answer"]["multiple_choice"] == {"gold": "B"}
    for before, after in zip(base_records, records, strict=True):
        assert after["gold_papers"] == before["gold_papers"]
        assert after["evidence"] == before["evidence"]
    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    assert audit["changed_query_ids"] == ["q_table", "q_both"]
    assert all(audit["freeze_checks"].values())


@pytest.mark.parametrize("missing", ["source_checked", "notes", "locator"])
def test_compose_requires_source_check_notes_and_locator(
    tmp_path: Path, missing: str
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    if missing == "source_checked":
        payload["decisions"][0][missing] = False
    elif missing == "notes":
        payload["decisions"][0][missing] = ""
    else:
        payload["decisions"][0][missing] = []
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdjudicationError, match=missing):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_compose_rejects_candidate_changed_after_review(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    changed = _table_only_candidate()
    changed[0]["answer"]["table"]["rows"][0]["score"] = 99.0
    _write_jsonl(table_only, changed, compact=True)

    with pytest.raises(AdjudicationError, match="sealed field mismatch: candidates"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_review_rejects_invalid_base_evidence_locator(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    records = _base()
    del records[1]["evidence"][0]["locator"]["table_id"]
    _write_jsonl(base, records)

    with pytest.raises(AdjudicationError, match="incomplete for table"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "invalid_base_review",
        )


def test_compose_requires_review_locator_from_frozen_evidence(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    payload["decisions"][0]["locator"][0]["locator"] = {
        "page": 3,
        "table_id": "Table 2",
    }
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdjudicationError, match="present in frozen evidence"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_compose_rejects_tampered_schema_hash_in_decision(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    payload["table_schema_sha256"] = "0" * 64
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AdjudicationError, match="sealed field mismatch: table_schema_sha256"
    ):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_review_rejects_candidate_order_and_schema_errors(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    reversed_records = list(reversed(_table_only_candidate()))
    _write_jsonl(table_only, reversed_records, compact=True)

    with pytest.raises(AdjudicationError, match="query order"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "review_order",
        )

    invalid = _table_only_candidate()
    del invalid[0]["answer"]["table"]["rows"][0]["score"]
    _write_jsonl(table_only, invalid, compact=True)
    with pytest.raises(AdjudicationError, match="must contain exactly"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "review_schema",
        )
