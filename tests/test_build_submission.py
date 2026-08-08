"""Building a paper-only submission from a finished retrieval run."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.build_submission import empty_answer, load_queries, load_rankings


def _retrieval(tmp_path, queries):
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
    return path


def _queries(tmp_path, records):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    ("answer_types", "expected"),
    [
        (["freeform"], {"freeform": {"text": ""}}),
        (["multiple_choice"], {"multiple_choice": {"gold": ""}}),
        (["table"], {"table": {"rows": []}}),
        (
            ["freeform", "table"],
            {"freeform": {"text": ""}, "table": {"rows": []}},
        ),
        (None, {}),
        (["unknown"], {}),
    ],
)
def test_answer_skeleton_matches_the_sample_submission(answer_types, expected):
    assert empty_answer(answer_types) == expected


def test_answer_skeletons_are_not_shared_between_records():
    first = empty_answer(["table"])
    second = empty_answer(["table"])
    first["table"]["rows"].append("x")

    assert second == {"table": {"rows": []}}


def test_load_rankings_rejects_an_empty_run(tmp_path):
    with pytest.raises(ValueError, match="no queries"):
        load_rankings(_retrieval(tmp_path, []))


def test_load_queries_rejects_an_empty_file(tmp_path):
    with pytest.raises(ValueError, match="no queries"):
        load_queries(_queries(tmp_path, []))


def _run(tmp_path, *, style="f1_balanced", queries=None, ranked=None):
    records = queries or [
        {
            "query_id": "q1",
            "question": "For the two ICCV 2025 papers, compare their speed.",
            "answer_types": ["table"],
        },
        {
            "query_id": "q2",
            "question": "What batch size does TCM use?",
            "answer_types": ["multiple_choice"],
        },
    ]
    runs = ranked or [
        {"query_id": "q1", "ranked_papers": ["a", "b", "c"]},
        {"query_id": "q2", "ranked_papers": ["d", "e"]},
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "submission.jsonl"
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_submission.py",
            "--retrieval", str(_retrieval(tmp_path, runs)),
            "--queries", str(_queries(tmp_path, records)),
            "--select", f"configs/select_style/{style}.yaml",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    ), output


def test_writes_one_record_per_query_in_input_order(tmp_path):
    result, output = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["gold_papers"] == [{"paper_id": "a"}, {"paper_id": "b"}]
    assert rows[1]["gold_papers"] == [{"paper_id": "d"}]
    assert all(row["evidence"] == [] for row in rows)
    assert rows[0]["answer"] == {"table": {"rows": []}}


def test_the_selected_style_changes_how_many_papers_are_submitted(tmp_path):
    _, balanced = _run(tmp_path / "a", style="f1_balanced")
    _, recall = _run(tmp_path / "b", style="high_recall")

    balanced_rows = [json.loads(line) for line in balanced.read_text().splitlines()]
    recall_rows = [json.loads(line) for line in recall.read_text().splitlines()]

    assert len(balanced_rows[0]["gold_papers"]) == 2
    assert len(recall_rows[0]["gold_papers"]) == 3


def test_a_query_without_a_retrieval_result_is_an_error(tmp_path):
    result, _ = _run(
        tmp_path,
        ranked=[{"query_id": "q1", "ranked_papers": ["a", "b"]}],
    )

    # Scoring walks the gold file, so a silently missing record would score
    # zero on papers rather than being reported.
    assert result.returncode != 0
    assert "no retrieval result" in result.stderr
