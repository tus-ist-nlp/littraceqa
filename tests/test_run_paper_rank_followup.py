"""Safety and protocol tests for the fixed PaperRank follow-up runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_paper_rank_followup as runner


def _write_index_chunks(path: Path, paper_ids: list[str], *, duplicates: int = 1):
    path.mkdir()
    with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for paper_id in paper_ids:
            for index in range(duplicates):
                handle.write(
                    json.dumps(
                        {
                            "chunk_id": f"{paper_id}#c{index:04d}",
                            "paper_id": paper_id,
                            "text": "text",
                            "chunk_type": "text_span",
                            "metadata": {},
                        }
                    )
                    + "\n"
                )


def test_parser_has_no_gold_or_candidate_depth_override():
    destinations = {
        action.dest for action in runner.build_parser()._actions  # noqa: SLF001
    }

    assert "gold" not in destinations
    assert "candidate_depth" not in destinations
    assert "task_family" not in destinations
    assert "primary_evidence_type" not in destinations
    assert {
        "chunk_index_dir",
        "paper_index_dir",
        "queries",
        "output_dir",
        "paper_count",
    }.issubset(destinations)


def test_three_variants_and_depths_are_fixed():
    assert tuple(runner.METHOD_CONFIGS) == runner.METHOD_ORDER
    assert len(runner.METHOD_ORDER) == 3
    assert {
        config["candidate_depth"] for config in runner.METHOD_CONFIGS.values()
    } == {100, 1000}
    assert runner.ALLOWED_CANDIDATE_DEPTHS == {100, 1000}
    assert runner.ALLOWED_PAPER_COUNTS == {100, 200}
    assert runner.DEFAULT_PAPER_COUNT == 100
    assert runner.MAX_PAPERS == 200
    assert runner.MAX_QUERIES == 55


def test_effective_paper_depth_is_bounded_by_the_fixed_corpus():
    assert min(
        runner.METHOD_CONFIGS[
            "mineru_v1_paper_rank_rrf_fill20_d1000"
        ]["candidate_depth"],
        runner.DEFAULT_PAPER_COUNT,
    ) == 100


def test_query_limit_stops_before_later_invalid_or_label_records(tmp_path: Path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        """{"query_id":"q1","question":"First?","answer_types":[],"table_schema":[],"task_family":"hidden"}
{"query_id":"q2","question":"Second?","answer_types":[],"table_schema":[],"primary_evidence_type":"hidden"}
not valid JSON
""",
        encoding="utf-8",
    )

    loaded = runner.load_bounded_queries(queries, limit=2)

    assert [query.query_id for query in loaded] == ["q1", "q2"]
    assert all(query.task_family is None for query in loaded)
    assert all(query.primary_evidence_type is None for query in loaded)


def test_index_pair_requires_the_same_exact_100_papers(tmp_path: Path):
    expected = [f"paper-{index:03d}" for index in range(100)]
    chunk_index = tmp_path / "chunk"
    paper_index = tmp_path / "paper"
    _write_index_chunks(chunk_index, expected, duplicates=2)
    _write_index_chunks(paper_index, expected)

    paper_ids, counts = runner.validate_index_pair(chunk_index, paper_index)

    assert paper_ids == expected
    assert counts == {
        "chunk_index_record_count": 200,
        "paper_index_record_count": 100,
    }


def test_index_pair_accepts_the_predeclared_200_paper_corpus(tmp_path: Path):
    expected = [f"paper-{index:03d}" for index in range(200)]
    chunk_index = tmp_path / "chunk"
    paper_index = tmp_path / "paper"
    _write_index_chunks(chunk_index, expected, duplicates=2)
    _write_index_chunks(paper_index, expected)

    paper_ids, counts = runner.validate_index_pair(
        chunk_index,
        paper_index,
        expected_papers=200,
    )

    assert paper_ids == expected
    assert counts == {
        "chunk_index_record_count": 400,
        "paper_index_record_count": 200,
    }


def test_index_inspection_stops_at_paper_101(tmp_path: Path):
    index = tmp_path / "oversized"
    _write_index_chunks(index, [f"paper-{number}" for number in range(101)])

    with pytest.raises(ValueError, match="100-paper safety cap"):
        runner.inspect_index_papers(index, require_one_record_per_paper=False)


def test_output_validation_rejects_protected_and_nonempty_paths(tmp_path: Path):
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlaps protected input"):
        runner.validate_output_directory(protected / "result", [protected])

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        runner.validate_output_directory(nonempty, [protected])

    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner.validate_output_directory(empty, [protected]) == empty.resolve()


def test_index_inspection_rejects_too_many_repeated_records(
    tmp_path: Path, monkeypatch
):
    index = tmp_path / "too-many-records"
    _write_index_chunks(index, ["paper-000"], duplicates=3)
    monkeypatch.setattr(runner, "MAX_INDEX_RECORDS", 2)

    with pytest.raises(ValueError, match="record safety cap"):
        runner.inspect_index_papers(index, require_one_record_per_paper=False)


def test_directory_snapshot_rejects_oversized_index(tmp_path: Path, monkeypatch):
    index = tmp_path / "oversized-index"
    index.mkdir()
    (index / "data").write_bytes(b"1234")
    monkeypatch.setattr(runner, "MAX_TOTAL_INDEX_BYTES", 3)

    with pytest.raises(ValueError, match="byte limit"):
        runner._directory_snapshot(index)


def test_search_variant_uses_only_the_predeclared_depth_and_fusion_budget(monkeypatch):
    calls = []

    class FakeIndex:
        def __init__(self, source: str):
            self.source = source

        def search(self, question: str, top_k: int):
            calls.append((self.source, question, top_k))
            return []

    class FakeFuser:
        def __init__(self, **kwargs):
            calls.append(("fuser_init", kwargs))

        def fuse(self, runs, top_k: int):
            calls.append(("fuse", len(runs), top_k))
            return []

    monkeypatch.setattr(runner, "PaperRankRRFFuser", FakeFuser)
    config = runner.METHOD_CONFIGS[
        "mineru_v1_paper_rank_rrf_fill20_d1000"
    ]

    runner._search_variant(
        "question", FakeIndex("chunk"), FakeIndex("paper"), config
    )

    assert ("chunk", "question", 1000) in calls
    assert ("paper", "question", 1000) in calls
    assert ("fuse", 2, 20) in calls
    assert any(
        call[0] == "fuser_init" and call[1]["fill_to_top_k"] is True
        for call in calls
    )
