"""Safety and fixed-protocol tests for the BGE-M3 comparison runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_bge_m3_retrieval as runner


def _record(paper_id: str, chunk_type: str = "title_abstract") -> dict:
    return {
        "chunk_id": f"{paper_id}#{chunk_type}",
        "paper_id": paper_id,
        "text": f"{paper_id} observed text",
        "chunk_type": chunk_type,
        "metadata": {},
    }


def test_parser_exposes_no_gold_or_label_routing_inputs():
    destinations = {
        action.dest for action in runner.build_parser()._actions  # noqa: SLF001
    }

    assert "gold" not in destinations
    assert "task_family" not in destinations
    assert "primary_evidence_type" not in destinations
    assert "candidate_depth" not in destinations
    assert "max_length" not in destinations
    assert {
        "chunks",
        "chunk_index_dir",
        "paper_index_dir",
        "model_path",
        "queries",
        "output_dir",
        "dense_work_dir",
        "resume",
        "query_limit",
        "paper_count",
    }.issubset(destinations)


def test_parser_only_allows_the_fixed_two_query_smoke_limit():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--query-limit", "1"])


def test_protocol_has_exactly_three_predeclared_methods():
    assert runner.METHOD_ORDER == (
        "mineru_v1_paper_rank_rrf_fill20_d100",
        "bge_m3_title_abstract_dense",
        "bm25_bge_m3_title_abstract_rrf",
    )
    assert runner.CANDIDATE_DEPTH == 100
    assert runner.PAPER_TOP_K == 20
    assert runner.BGE_MAX_LENGTH == 512
    assert runner.BGE_BATCH_SIZE == 1
    assert runner.ALLOWED_PAPER_COUNTS == {100, 200}
    assert runner.DEFAULT_PAPER_COUNT == 100
    assert runner.MAX_PAPERS == 200
    configs = runner._method_config()
    assert tuple(configs) == runner.METHOD_ORDER
    assert configs[runner.METHOD_ORDER[2]]["weights"] == {
        "bm25s": 1.0,
        "paper_bm25": 1.0,
        "bge_m3_numpy": 1.0,
    }


def test_title_loader_streams_exactly_one_observed_view_per_paper(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(100):
            paper_id = f"paper-{index:03d}"
            handle.write(json.dumps(_record(paper_id)) + "\n")
            handle.write(json.dumps(_record(paper_id, "figure")) + "\n")

    chunks, papers, counts = runner.load_title_abstract_chunks(path)

    assert len(chunks) == len(papers) == 100
    assert all(chunk.chunk_type == "title_abstract" for chunk in chunks)
    assert counts["chunk_record_count"] == 200
    assert counts["chunk_type_figure"] == 100
    assert counts["chunk_type_title_abstract"] == 100


def test_title_loader_accepts_the_predeclared_200_paper_corpus(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(200):
            handle.write(json.dumps(_record(f"paper-{index:03d}")) + "\n")

    chunks, papers, counts = runner.load_title_abstract_chunks(
        path,
        expected_papers=200,
    )

    assert len(chunks) == len(papers) == 200
    assert counts["chunk_record_count"] == 200


def test_title_loader_rejects_paper_101_before_processing_more(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "chunks.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    consumed = []

    def records(_path):
        for index in range(101):
            consumed.append(index)
            yield _record(f"paper-{index:03d}")
        raise AssertionError("loader consumed beyond paper 101")

    monkeypatch.setattr(runner, "_iter_jsonl", records)
    with pytest.raises(ValueError, match="capped at 100 papers"):
        runner.load_title_abstract_chunks(path)
    assert consumed == list(range(101))


def test_title_loader_rejects_missing_or_duplicate_views(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    with missing.open("w", encoding="utf-8") as handle:
        for index in range(100):
            kind = "figure" if index == 99 else "title_abstract"
            handle.write(json.dumps(_record(f"paper-{index:03d}", kind)) + "\n")
    with pytest.raises(ValueError, match="missing title_abstract"):
        runner.load_title_abstract_chunks(missing)

    duplicate = tmp_path / "duplicate.jsonl"
    with duplicate.open("w", encoding="utf-8") as handle:
        for index in range(100):
            record = _record(f"paper-{index:03d}")
            handle.write(json.dumps(record) + "\n")
            if index == 0:
                second = dict(record)
                second["chunk_id"] = "duplicate"
                handle.write(json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="more than one"):
        runner.load_title_abstract_chunks(duplicate)


def test_model_snapshot_records_blob_identity_without_hashing_weights(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")

    snapshot = runner._model_snapshot(model)

    assert snapshot["file_count"] == 2
    assert snapshot["bytes"] == 9
    assert len(snapshot["tree_sha256"]) == 64
    assert {item["path"] for item in snapshot["files"]} == {
        "config.json",
        "weights.bin",
    }


def test_model_snapshot_digest_changes_with_regular_file_content(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"first")
    first = runner._model_snapshot(model)
    weights.write_bytes(b"other")
    second = runner._model_snapshot(model)

    assert first["tree_sha256"] != second["tree_sha256"]


def test_build_fingerprint_records_fixed_common_chunk_protocol(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(_record("paper-001")) + "\n", encoding="utf-8")
    chunks = [runner.Chunk(**_record("paper-001"))]
    snapshot = {"tree_sha256": "a" * 64, "bytes": 1}
    fingerprint = runner._build_fingerprint(
        chunks_path,
        chunks,
        tmp_path / "model",
        snapshot,
        {"runner": "hash"},
    )
    assert "document_format" not in fingerprint
    assert fingerprint["selected_chunk_count"] == 1


def test_new_dense_index_uses_fixed_common_chunk_input(tmp_path: Path, monkeypatch):
    observed: dict = {}

    class FakeIndex:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(runner, "BGEM3NumpyIndex", FakeIndex)

    index = runner._new_dense_index(
        tmp_path / "index",
        tmp_path / "model",
    )

    assert isinstance(index, FakeIndex)
    assert "document_format" not in observed


def test_resume_work_validation_rejects_protected_or_missing_paths(tmp_path: Path):
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlaps protected"):
        runner._validate_resume_work_directory(protected, [protected])
    with pytest.raises(ValueError, match="existing dense work"):
        runner._validate_resume_work_directory(tmp_path / "missing", [protected])


def test_resume_manifest_is_checked_before_index_constructor(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()

    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("index constructor must not run before manifest validation")

    monkeypatch.setattr(runner, "_new_dense_index", unexpected_constructor)
    with pytest.raises(ValueError, match="build_manifest"):
        runner._prepare_dense_index(
            work,
            [],
            {"fixed": True},
            tmp_path / "model",
            resume=True,
        )
    assert not (work / "index").exists()


def test_interrupted_dense_build_is_recorded(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"

    class InterruptedIndex:
        def build(self, chunks):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        runner,
        "_new_dense_index",
        lambda index_dir, model_path: InterruptedIndex(),
    )
    with pytest.raises(KeyboardInterrupt):
        runner._prepare_dense_index(
            work,
            [],
            {"fixed": True},
            tmp_path / "model",
            resume=False,
        )

    manifest = json.loads((work / "build_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
