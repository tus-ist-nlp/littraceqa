"""Tests for rebuilding the retrieval gold set from evidence-backed papers."""

from __future__ import annotations

import json

import pytest

from scripts.build_evidence_gold import (
    build,
    evidence_paper_ids,
    evidence_without_text,
    filter_record,
    load_jsonl,
)


def _record(
    query_id: str,
    gold: list[str],
    evidence: list[dict],
    *,
    task_family: str = "multi_paper",
) -> dict:
    return {
        "query_id": query_id,
        "task_family": task_family,
        "primary_evidence_type": "table",
        "question": f"question for {query_id}",
        "gold_papers": [{"paper_id": paper_id} for paper_id in gold],
        "evidence": evidence,
    }


def _evidence(paper_id: str, *, text: str = "value", evidence_id: str = "ev_1") -> dict:
    return {
        "evidence_id": evidence_id,
        "paper_id": paper_id,
        "source_type": "table",
        "evidence_text_or_value": text,
    }


def test_keeps_only_gold_papers_that_evidence_points_at():
    record = _record("q1", ["a", "b", "c"], [_evidence("a"), _evidence("c")])

    filtered, audit = filter_record(record)

    assert [p["paper_id"] for p in filtered["gold_papers"]] == ["a", "c"]
    assert audit["dropped_paper_ids"] == ["b"]
    assert audit["gold_before"] == 3
    assert audit["gold_after"] == 2


def test_preserves_gold_paper_order_and_payload():
    record = _record("q1", ["b", "a"], [_evidence("a"), _evidence("b")])
    record["gold_papers"][0]["title"] = "B paper"

    filtered, _ = filter_record(record)

    assert [p["paper_id"] for p in filtered["gold_papers"]] == ["b", "a"]
    assert filtered["gold_papers"][0]["title"] == "B paper"


def test_leaves_every_other_field_untouched():
    record = _record("q1", ["a"], [_evidence("a")])
    record["answer"] = {"freeform": {"text": "42"}}

    filtered, _ = filter_record(record)

    assert filtered["answer"] == {"freeform": {"text": "42"}}
    assert filtered["question"] == "question for q1"
    assert record["gold_papers"] == [{"paper_id": "a"}]  # source not mutated


def test_cited_paper_titles_do_not_justify_a_gold_paper():
    """locator.cited_paper holds a free-text title, never a corpus paper_id."""

    evidence = [
        {
            "evidence_id": "ev_1",
            "paper_id": "a",
            "source_type": "citation_context",
            "evidence_text_or_value": "cites something",
            "locator": {"cited_paper": "b"},
        }
    ]
    record = _record("q1", ["a", "b"], evidence)

    filtered, audit = filter_record(record)

    assert [p["paper_id"] for p in filtered["gold_papers"]] == ["a"]
    assert audit["dropped_paper_ids"] == ["b"]


def test_evidence_with_empty_text_still_keeps_the_paper():
    record = _record("q1", ["a"], [_evidence("a", text="  ")])

    filtered, audit = filter_record(record)

    assert [p["paper_id"] for p in filtered["gold_papers"]] == ["a"]
    assert audit["evidence_without_text"] == ["ev_1"]


def test_record_without_evidence_empties_its_gold_papers():
    record = _record("q1", ["a"], [])

    filtered, audit = filter_record(record)

    assert filtered["gold_papers"] == []
    assert audit["dropped_paper_ids"] == ["a"]


def test_evidence_paper_ids_ignores_malformed_items():
    record = _record("q1", ["a"], ["not a dict", {"paper_id": ""}, _evidence("a")])

    assert evidence_paper_ids(record) == {"a"}


def test_evidence_without_text_reports_positional_id_when_missing():
    record = _record("q1", ["a"], [{"paper_id": "a", "evidence_text_or_value": ""}])

    assert evidence_without_text(record) == ["evidence[0]"]


def test_build_summarizes_drops_by_task_family():
    records = [
        _record("q1", ["a", "b"], [_evidence("a")]),
        _record("q2", ["c"], [_evidence("c")], task_family="single"),
        _record("q3", ["d"], []),
    ]

    _, report = build(records)

    assert report["gold_before"] == 4
    assert report["gold_after"] == 2
    assert report["queries_with_drops"] == 2
    assert report["queries_emptied"] == ["q3"]
    assert report["by_task_family"]["single"] == {
        "queries": 1,
        "gold_before": 1,
        "gold_after": 1,
    }
    assert report["by_task_family"]["multi_paper"]["gold_after"] == 1


def test_rejects_gold_papers_without_a_paper_id():
    record = _record("q1", [], [_evidence("a")])
    record["gold_papers"] = [{"title": "no id"}]

    with pytest.raises(ValueError, match="paper_id"):
        filter_record(record)


def test_rejects_non_list_gold_papers():
    record = _record("q1", ["a"], [_evidence("a")])
    record["gold_papers"] = "a"

    with pytest.raises(ValueError, match="gold_papers"):
        filter_record(record)


def test_load_jsonl_skips_blank_lines_and_reports_bad_json(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text('{"query_id": "q1"}\n\n', encoding="utf-8")

    assert load_jsonl(path) == [{"query_id": "q1"}]

    path.write_text('{"query_id": "q1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl(path)


def test_real_validation_file_keeps_every_query_non_empty():
    """The shipped development set must not lose a whole query to this rule."""

    records = load_jsonl(__import__("pathlib").Path("data/validation.jsonl"))
    _, report = build(records)

    assert report["queries"] == 55
    assert report["gold_before"] == 146
    assert report["gold_after"] == 117
    assert report["queries_emptied"] == []


def test_written_gold_file_matches_the_filtering_rule():
    """The checked-in artifact stays in sync with the script."""

    from pathlib import Path

    source = Path("data/validation.jsonl")
    built = Path("data/validation_evidence_gold.jsonl")
    if not built.exists():
        pytest.skip("run scripts/build_evidence_gold.py first")

    expected, _ = build(load_jsonl(source))
    actual = load_jsonl(built)

    assert [json.dumps(r, sort_keys=True) for r in actual] == [
        json.dumps(r, sort_keys=True) for r in expected
    ]
