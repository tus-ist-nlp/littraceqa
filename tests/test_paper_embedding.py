"""Tests for the model-free paper embedding sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from littraceqa.di_pipeline.index.paper_embedding import PaperEmbeddingStore


def _paper(paper_id: str) -> dict:
    return {
        "chunk_id": f"{paper_id}#c0000",
        "paper_id": paper_id,
        "text": f"title and abstract for {paper_id}",
        "chunk_type": "title_abstract",
        "metadata": {"title": f"Title {paper_id}"},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(
    directory: Path,
    *,
    papers: list[dict] | None = None,
    embeddings: np.ndarray | None = None,
    source: dict | None = None,
) -> None:
    records = papers or [_paper("query"), _paper("beta"), _paper("alpha")]
    matrix = (
        embeddings
        if embeddings is not None
        else np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.8, -0.6],
            ],
            dtype=np.float32,
        )
    )
    embeddings_path = directory / "embeddings.npy"
    papers_path = directory / "papers.jsonl"
    np.save(embeddings_path, matrix)
    papers_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (directory / "index_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paper_count": len(records),
                "dimension": int(matrix.shape[1]),
                "files": {
                    "embeddings.npy": {
                        "sha256": _sha256(embeddings_path),
                        "size": embeddings_path.stat().st_size,
                    },
                    "papers.jsonl": {
                        "sha256": _sha256(papers_path),
                        "size": papers_path.stat().st_size,
                    },
                },
                "source": source or {"model": "fixture"},
            }
        ),
        encoding="utf-8",
    )


def test_loads_mmap_and_searches_with_deterministic_ties(tmp_path):
    _write_sidecar(tmp_path, source={"model": "fixture", "revision": "one"})
    store = PaperEmbeddingStore(tmp_path)

    store.load()
    results = store.search_by_paper_id("query", top_k=10)

    assert isinstance(store._embeddings, np.memmap)
    assert not store._embeddings.flags.writeable
    assert store.paper_count == 3
    assert store.dimension == 2
    assert store.source == {"model": "fixture", "revision": "one"}
    assert [result.paper_id for result in results] == ["alpha", "beta"]
    assert [result.score for result in results] == pytest.approx([0.8, 0.8])
    assert all(result.paper_id != "query" for result in results)
    assert all(result.source == "paper_embedding" for result in results)
    assert results[0].metadata == {"title": "Title alpha"}


def test_search_handles_limits_missing_ids_and_invalid_inputs(tmp_path):
    _write_sidecar(tmp_path)
    store = PaperEmbeddingStore(tmp_path)
    with pytest.raises(RuntimeError, match="not loaded"):
        store.search_by_paper_id("query", 1)

    store.load()

    assert store.search_by_paper_id("query", 0) == []
    assert store.search_by_paper_id("missing", 2) == []
    assert len(store.search_by_paper_id("query", 1)) == 1
    with pytest.raises(TypeError, match="paper_id"):
        store.search_by_paper_id(3, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        store.search_by_paper_id("", 1)
    with pytest.raises(TypeError, match="top_k"):
        store.search_by_paper_id("query", True)


def test_rejects_checksum_mismatch(tmp_path):
    _write_sidecar(tmp_path)
    content = (tmp_path / "papers.jsonl").read_bytes()
    (tmp_path / "papers.jsonl").write_bytes(
        content.replace(b"query", b"other", 1)
    )

    with pytest.raises(ValueError, match="checksum"):
        PaperEmbeddingStore(tmp_path).load()


def test_rejects_size_mismatch_before_hashing(monkeypatch, tmp_path):
    _write_sidecar(tmp_path)
    config_path = tmp_path / "index_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["files"]["embeddings.npy"]["size"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fail_if_hashed(path):
        raise AssertionError(f"unexpected hash call for {path}")

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.paper_embedding._sha256_file",
        fail_if_hashed,
    )

    with pytest.raises(ValueError, match="size"):
        PaperEmbeddingStore(tmp_path).load()


@pytest.mark.parametrize(
    ("embeddings", "error"),
    [
        (
            np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float64),
            "dtype float32",
        ),
        (
            np.asarray([[2, 0], [0, 1], [1, 0]], dtype=np.float32),
            "L2-normalized",
        ),
        (
            np.asarray(
                [[np.nan, 0], [0, 1], [1, 0]],
                dtype=np.float32,
            ),
            "finite",
        ),
    ],
)
def test_rejects_invalid_embedding_values(tmp_path, embeddings, error):
    _write_sidecar(tmp_path, embeddings=embeddings)

    with pytest.raises(ValueError, match=error):
        PaperEmbeddingStore(tmp_path).load()


def test_rejects_shape_and_record_count_mismatches(tmp_path):
    _write_sidecar(
        tmp_path,
        embeddings=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="shape"):
        PaperEmbeddingStore(tmp_path).load()


def test_rejects_duplicate_paper_ids(tmp_path):
    duplicate = _paper("duplicate")
    _write_sidecar(
        tmp_path,
        papers=[duplicate, {**duplicate, "chunk_id": "other#c0000"}],
        embeddings=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="duplicate paper_id"):
        PaperEmbeddingStore(tmp_path).load()
