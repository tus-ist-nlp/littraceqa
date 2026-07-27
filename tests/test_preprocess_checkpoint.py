"""Focused tests for resumable per-paper preprocessing checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.preprocess.checkpoint import PreprocessCache


def _paper(paper_id: str, *, title: str | None = None) -> dict:
    return {
        "paper_id": paper_id,
        "title": title or f"Title {paper_id}",
        "venue": "ACL",
        "year": 2025,
        "authors": ["A. Author"],
        "abstract": "Abstract",
    }


def _chunk(paper_id: str, index: int, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{paper_id}#c{index:04d}",
        paper_id=paper_id,
        text=text or f"text {paper_id} {index}",
        chunk_type="text_span",
        metadata={"page": index + 1},
    )


def _cache(tmp_path: Path, *, process_config: dict | None = None) -> PreprocessCache:
    module = tmp_path / "chunker.py"
    if not module.exists():
        module.write_text("IMPLEMENTATION_VERSION = 1\n", encoding="utf-8")
    return PreprocessCache(
        tmp_path / "cache",
        process_config=process_config or {"name": "mineru", "params": {"size": 2000}},
        source_module_path=module,
    )


def _source(tmp_path: Path, paper_id: str) -> Path:
    path = tmp_path / "sources" / f"{paper_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n", encoding="utf-8")
    return path


def test_store_and_reload_uses_path_safe_atomic_files(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    paper = _paper("../../unsafe paper")
    source = _source(tmp_path, "source")

    record = cache.store_success(
        paper,
        source,
        [_chunk(paper["paper_id"], 0), _chunk(paper["paper_id"], 1)],
    )

    path = cache.cache_path(paper["paper_id"])
    assert path.is_file()
    assert path.parent.parent == cache.papers_dir
    assert ".." not in path.relative_to(cache.root).parts
    assert record["chunk_count"] == 2
    assert cache.load_valid_chunks(paper, source) == [
        _chunk(paper["paper_id"], 0),
        _chunk(paper["paper_id"], 1),
    ]
    assert not list(path.parent.glob("*.tmp"))


def test_signatures_cover_config_metadata_source_stat_and_module(tmp_path: Path) -> None:
    source = _source(tmp_path, "p1")
    paper = _paper("p1")
    first = _cache(
        tmp_path,
        process_config={"params": {"size": 2000}, "name": "mineru"},
    )
    same = _cache(
        tmp_path,
        process_config={"name": "mineru", "params": {"size": 2000}},
    )

    assert first.process_signature == same.process_signature
    assert first.input_signature(paper, source)[0] == same.input_signature(
        paper, source
    )[0]
    assert first.input_signature(_paper("p1", title="Changed"), source)[0] != (
        first.input_signature(paper, source)[0]
    )

    source_signature_before = first.input_signature(paper, source)[0]
    source.write_text("[{}]\n", encoding="utf-8")
    os.utime(source, None)
    assert first.input_signature(paper, source)[0] != source_signature_before

    (tmp_path / "chunker.py").write_text(
        "IMPLEMENTATION_VERSION = 2\n", encoding="utf-8"
    )
    changed_module = _cache(tmp_path)
    assert changed_module.process_signature != first.process_signature


def test_process_signature_covers_declared_source_dependencies(
    tmp_path: Path,
) -> None:
    module = tmp_path / "chunker.py"
    dependency = tmp_path / "helper.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")
    dependency.write_text("HELPER_VERSION = 1\n", encoding="utf-8")
    first = PreprocessCache(
        tmp_path / "cache",
        process_config={"name": "mineru", "params": {}},
        source_module_path=module,
        source_dependency_paths=[dependency],
    )

    dependency.write_text("HELPER_VERSION = 2\n", encoding="utf-8")
    changed = PreprocessCache(
        tmp_path / "cache",
        process_config={"name": "mineru", "params": {}},
        source_module_path=module,
        source_dependency_paths=[dependency],
    )

    assert changed.process_signature != first.process_signature


@pytest.mark.parametrize("internal_name", ["papers", "manifest.jsonl"])
def test_cache_rejects_internal_symlinks(
    tmp_path: Path,
    internal_name: str,
) -> None:
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    root.mkdir()
    if internal_name == "papers":
        outside.mkdir()
    else:
        outside.write_text("", encoding="utf-8")
    (root / internal_name).symlink_to(
        outside,
        target_is_directory=internal_name == "papers",
    )
    module = tmp_path / "chunker.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        PreprocessCache(
            root,
            process_config={"name": "mineru", "params": {}},
            source_module_path=module,
        )


def test_stale_or_corrupted_success_is_not_resumed(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    paper = _paper("p1")
    source = _source(tmp_path, "p1")
    cache.store_success(paper, source, [_chunk("p1", 0)])

    assert cache.load_valid_chunks(paper, source) is not None
    cache.cache_path("p1").write_text("{broken\n", encoding="utf-8")
    assert cache.load_valid_chunks(paper, source) is None

    cache.store_success(paper, source, [_chunk("p1", 0)])
    source.write_text("[{\"changed\": true}]\n", encoding="utf-8")
    assert cache.load_valid_chunks(paper, source) is None


def test_failure_is_recorded_and_retried_as_the_next_attempt(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    paper = _paper("p1")
    source = _source(tmp_path, "p1")

    failure = cache.record_failure(paper, source, RuntimeError("bad input"))

    assert failure["attempt"] == 1
    assert failure["error_type"] == "RuntimeError"
    assert cache.load_valid_chunks(paper, source) is None

    success = cache.store_success(paper, source, [_chunk("p1", 0)])
    assert success["attempt"] == 2
    assert cache.load_valid_chunks(paper, source) is not None


def test_truncated_manifest_tail_is_repaired_before_new_records(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    source1 = _source(tmp_path, "p1")
    cache.store_success(_paper("p1"), source1, [_chunk("p1", 0)])
    with cache.manifest_path.open("ab") as handle:
        handle.write(b'{"schema_version":1,"paper_id":"partial')

    resumed = _cache(tmp_path)
    assert resumed.load_valid_chunks(_paper("p1"), source1) is not None
    resumed.record_failure(
        _paper("p2"),
        _source(tmp_path, "p2"),
        ValueError("expected"),
    )

    records = [
        json.loads(line)
        for line in resumed.manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["paper_id"] for record in records] == ["p1", "p2"]


def test_manifest_rejects_corruption_before_the_final_line(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text(
        '{"schema_version":1,"paper_id":"p1","status":"failed"}\n'
        "{broken}\n"
        '{"schema_version":1,"paper_id":"p2","status":"failed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a valid manifest record"):
        _cache(tmp_path)


def test_merge_is_selected_order_only_and_atomic_on_failure(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    papers = [_paper("p1"), _paper("p2"), _paper("unused")]
    sources = {
        paper["paper_id"]: _source(tmp_path, paper["paper_id"]) for paper in papers
    }
    for paper in papers:
        paper_id = paper["paper_id"]
        cache.store_success(
            paper,
            sources[paper_id],
            [_chunk(paper_id, 0), _chunk(paper_id, 1)],
        )

    destination = tmp_path / "merged" / "chunks.jsonl"
    result = cache.merge_selected(
        [
            (papers[1], sources["p2"]),
            (papers[0], sources["p1"]),
        ],
        destination,
    )

    merged = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["paper_id"] for record in merged] == [
        "p2",
        "p2",
        "p1",
        "p1",
    ]
    assert result.paper_count == 2
    assert result.chunk_count == 4
    assert result.byte_count == destination.stat().st_size

    original = destination.read_bytes()
    with pytest.raises(ValueError, match="no valid preprocessing cache"):
        cache.merge_selected(
            [
                (papers[0], sources["p1"]),
                (_paper("missing"), tmp_path / "missing.json"),
            ],
            destination,
        )
    assert destination.read_bytes() == original


def test_store_rejects_empty_wrong_paper_and_duplicate_chunks(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    paper = _paper("p1")
    source = _source(tmp_path, "p1")

    with pytest.raises(ValueError, match="no chunks"):
        cache.store_success(paper, source, [])
    with pytest.raises(ValueError, match="belongs to"):
        cache.store_success(paper, source, [_chunk("p2", 0)])
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        cache.store_success(
            paper,
            source,
            [_chunk("p1", 0), _chunk("p1", 0, text="different")],
        )
