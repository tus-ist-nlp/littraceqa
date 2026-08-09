"""Building a paper-only submission from a finished retrieval run."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.build_submission import empty_answer
from littraceqa.di_pipeline.evaluation.selection_input import (
    load_queries,
    load_rankings,
    load_retrieval_run,
)


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


def _method_owner_index(tmp_path, owners=None):
    path = tmp_path / "method_alias_graph.json"
    path.write_text(
        json.dumps({"schema_version": 3, "owners": owners or {}}),
        encoding="utf-8",
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


def test_retrieval_checkpoint_locates_the_method_owner_sidecar(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps(
            {
                "queries": [{"query_id": "q1", "ranked_papers": ["a"]}],
                "_checkpoint": {
                    "run_spec": {
                        "config": {
                            "retriever": {
                                "indexers": [
                                    {
                                        "name": "paper_bm25",
                                        "params": {"index_dir": "/indexes/papers"},
                                    }
                                ]
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    run = load_retrieval_run(path)

    assert run.method_owner_index_path is not None
    assert str(run.method_owner_index_path) == (
        "/indexes/papers/method_alias_graph.json"
    )


def test_ambiguous_method_owner_sidecar_requires_an_override(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps(
            {
                "queries": [{"query_id": "q1", "ranked_papers": ["a"]}],
                "_checkpoint": {
                    "run_spec": {
                        "config": {
                            "retriever": {
                                "indexers": [
                                    {
                                        "name": "paper_bm25",
                                        "params": {"index_dir": "/indexes/a"},
                                    },
                                    {
                                        "name": "paper_bm25",
                                        "params": {"index_dir": "/indexes/b"},
                                    },
                                ]
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_retrieval_run(path).method_owner_index_path is None


def _run(
    tmp_path,
    *,
    style="f1_method_owner",
    queries=None,
    ranked=None,
    owners=None,
):
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
            "--method-owner-index", str(_method_owner_index(tmp_path, owners)),
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
    _, f1_output = _run(tmp_path / "a", style="f1_method_owner")
    _, recall = _run(tmp_path / "b", style="high_recall")

    f1_rows = [json.loads(line) for line in f1_output.read_text().splitlines()]
    recall_rows = [json.loads(line) for line in recall.read_text().splitlines()]

    assert len(f1_rows[0]["gold_papers"]) == 2
    assert len(recall_rows[0]["gold_papers"]) == 3


def test_f1_style_prefers_an_explicit_method_owner(tmp_path):
    result, output = _run(
        tmp_path,
        queries=[
            {
                "query_id": "q1",
                "question": "What is reported in the EasySpec paper?",
                "answer_types": ["table"],
            }
        ],
        ranked=[{"query_id": "q1", "ranked_papers": ["wrong", "easy"]}],
        owners={"EasySpec": "easy"},
    )

    assert result.returncode == 0, result.stderr
    row = json.loads(output.read_text().splitlines()[0])
    assert row["gold_papers"] == [{"paper_id": "easy"}]


def test_a_query_without_a_retrieval_result_is_an_error(tmp_path):
    result, _ = _run(
        tmp_path,
        ranked=[{"query_id": "q1", "ranked_papers": ["a", "b"]}],
    )

    # Scoring walks the gold file, so a silently missing record would score
    # zero on papers rather than being reported.
    assert result.returncode != 0
    assert "no retrieval result" in result.stderr
