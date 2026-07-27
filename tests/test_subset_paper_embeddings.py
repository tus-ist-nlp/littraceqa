"""Tests for the bounded shared-FAISS paper subset exporter."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scripts.subset_paper_embeddings import ExportError, export_subset


def _record(paper_id: str, *, chunk_type: str = "title_abstract") -> dict:
    return {
        "chunk_id": f"{paper_id}#title_abstract",
        "paper_id": paper_id,
        "text": f"[Venue 2025] Title {paper_id}\nAbstract for {paper_id}.",
        "chunk_type": chunk_type,
        "metadata": {"title": f"Title {paper_id}", "venue": "Venue"},
    }


def _write_source(source: Path, records: list[dict]) -> tuple[bytes, bytes]:
    source.mkdir(parents=True)
    index_content = b"read-only fake FAISS index"
    records_content = b"".join(
        json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        for record in records
    )
    (source / "index.faiss").write_bytes(index_content)
    (source / "chunks.jsonl").write_bytes(records_content)
    return index_content, records_content


def _install_fake_faiss(monkeypatch, vectors: np.ndarray, *, flat_ip=True):
    module = types.ModuleType("faiss")

    class IndexFlatIP:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)
            self.ntotal, self.d = self.values.shape

        def reconstruct(self, row):
            return self.values[row].copy()

    class OtherIndex:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)
            self.ntotal, self.d = self.values.shape

        def reconstruct(self, row):
            return self.values[row].copy()

    index = IndexFlatIP(vectors) if flat_ip else OtherIndex(vectors)
    module.IndexFlatIP = IndexFlatIP
    module.read_index = lambda path: index
    monkeypatch.setitem(sys.modules, "faiss", module)
    return index


def _export(
    tmp_path: Path,
    monkeypatch,
    *,
    records: list[dict],
    vectors: np.ndarray,
    ids: str,
    expected_count: int,
    max_papers: int,
    flat_ip: bool = True,
):
    shared = tmp_path / "shared"
    source = shared / "paper_index"
    source_bytes = _write_source(source, records)
    ids_path = tmp_path / "paper_ids.txt"
    ids_path.write_text(ids, encoding="utf-8")
    output = tmp_path / "output"
    index = _install_fake_faiss(
        monkeypatch,
        vectors,
        flat_ip=flat_ip,
    )
    kwargs = {
        "source_index_dir": source,
        "paper_ids_file": ids_path,
        "output_dir": output,
        "shared_read_only_root": shared,
        "expected_count": expected_count,
        "max_papers": max_papers,
    }
    return output, source, source_bytes, index, kwargs


def test_export_preserves_source_order_and_writes_compatible_sidecar(
    tmp_path, monkeypatch
):
    records = [_record("p1"), _record("p2"), _record("p3"), _record("p4")]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    output, source, source_bytes, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=records,
        vectors=vectors,
        ids="p3\np1\n",
        expected_count=2,
        max_papers=2,
    )

    result = export_subset(**kwargs)

    assert result.output_dir == output.resolve()
    assert result.paper_count == 2
    assert result.dimension == 3
    assert result.source_paper_count == 4
    np.testing.assert_array_equal(
        np.load(output / "embeddings.npy", allow_pickle=False),
        vectors[[0, 2]],
    )
    exported_records = [
        json.loads(line)
        for line in (output / "papers.jsonl").read_text().splitlines()
    ]
    assert [record["paper_id"] for record in exported_records] == ["p1", "p3"]

    config = json.loads((output / "index_config.json").read_text())
    assert set(config) == {
        "schema_version",
        "paper_count",
        "dimension",
        "files",
        "source",
    }
    assert config["schema_version"] == 1
    assert config["paper_count"] == 2
    assert config["dimension"] == 3
    assert set(config["files"]) == {"embeddings.npy", "papers.jsonl"}
    for filename, record in config["files"].items():
        content = (output / filename).read_bytes()
        assert record == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    assert config["source"]["index_type"] == "IndexFlatIP"
    assert config["source"]["source_paper_count"] == 4
    assert config["source"]["selection_order"] == "source_row"
    assert config["source"]["expected_count"] == 2
    assert config["source"]["max_papers"] == 2

    from littraceqa.di_pipeline.index.paper_embedding import (
        PaperEmbeddingStore,
    )

    store = PaperEmbeddingStore(output)
    store.load()
    assert store.paper_count == 2
    assert store.dimension == 3
    assert [
        result.paper_id
        for result in store.search_by_paper_id("p1", top_k=1)
    ] == ["p3"]

    assert (source / "index.faiss").read_bytes() == source_bytes[0]
    assert (source / "chunks.jsonl").read_bytes() == source_bytes[1]


def test_expected_count_and_maximum_are_required_safety_bounds(
    tmp_path, monkeypatch
):
    _, _, _, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1"), _record("p2")],
        vectors=np.eye(2, dtype=np.float32),
        ids="p1\np2\n",
        expected_count=2,
        max_papers=1,
    )

    with pytest.raises(ValueError, match="must not exceed"):
        export_subset(**kwargs)


def test_duplicate_target_id_is_rejected(tmp_path, monkeypatch):
    output, _, _, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1")],
        vectors=np.ones((1, 2), dtype=np.float32),
        ids="p1\np1\n",
        expected_count=2,
        max_papers=2,
    )

    with pytest.raises(ExportError, match="duplicate paper ID"):
        export_subset(**kwargs)
    assert not output.exists()


def test_output_within_shared_read_only_root_is_rejected(
    tmp_path, monkeypatch
):
    _, source, source_bytes, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1")],
        vectors=np.ones((1, 2), dtype=np.float32),
        ids="p1\n",
        expected_count=1,
        max_papers=1,
    )
    kwargs["output_dir"] = source.parent / "forbidden-output"

    with pytest.raises(ExportError, match="outside"):
        export_subset(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert (source / "index.faiss").read_bytes() == source_bytes[0]
    assert (source / "chunks.jsonl").read_bytes() == source_bytes[1]


def test_non_index_flat_ip_is_rejected(tmp_path, monkeypatch):
    output, _, _, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1")],
        vectors=np.ones((1, 2), dtype=np.float32),
        ids="p1\n",
        expected_count=1,
        max_papers=1,
        flat_ip=False,
    )

    with pytest.raises(ExportError, match="IndexFlatIP"):
        export_subset(**kwargs)
    assert not output.exists()


@pytest.mark.parametrize(
    ("records", "vectors", "message"),
    [
        (
            [_record("p1"), _record("p2")],
            np.ones((1, 2), dtype=np.float32),
            "ntotal",
        ),
        (
            [_record("p1"), _record("p1")],
            np.ones((2, 2), dtype=np.float32),
            "not unique",
        ),
        (
            [_record("p1", chunk_type="text_span")],
            np.ones((1, 2), dtype=np.float32),
            "title_abstract",
        ),
    ],
)
def test_source_structure_is_validated(
    tmp_path, monkeypatch, records, vectors, message
):
    output, _, _, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=records,
        vectors=vectors,
        ids="p1\n",
        expected_count=1,
        max_papers=1,
    )

    with pytest.raises(ExportError, match=message):
        export_subset(**kwargs)
    assert not output.exists()


def test_source_chunk_ids_must_be_unique(tmp_path, monkeypatch):
    records = [_record("p1"), _record("p2")]
    records[1]["chunk_id"] = records[0]["chunk_id"]
    output, _, _, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=records,
        vectors=np.eye(2, dtype=np.float32),
        ids="p1\n",
        expected_count=1,
        max_papers=1,
    )

    with pytest.raises(ExportError, match="chunk_id is not unique"):
        export_subset(**kwargs)
    assert not output.exists()


def test_all_target_ids_must_exist_and_failure_is_atomic(
    tmp_path, monkeypatch
):
    output, source, source_bytes, _, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1"), _record("p2")],
        vectors=np.eye(2, dtype=np.float32),
        ids="p1\nmissing\n",
        expected_count=2,
        max_papers=2,
    )

    with pytest.raises(ExportError, match="missing"):
        export_subset(**kwargs)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*.building"))
    assert (source / "index.faiss").read_bytes() == source_bytes[0]
    assert (source / "chunks.jsonl").read_bytes() == source_bytes[1]


def test_invalid_reconstructed_vector_leaves_no_partial_output(
    tmp_path, monkeypatch
):
    output, _, _, index, kwargs = _export(
        tmp_path,
        monkeypatch,
        records=[_record("p1"), _record("p2")],
        vectors=np.eye(2, dtype=np.float32),
        ids="p1\np2\n",
        expected_count=2,
        max_papers=2,
    )
    original = index.reconstruct

    def reconstruct(row):
        if row == 1:
            return np.asarray([np.nan, 0.0], dtype=np.float32)
        return original(row)

    index.reconstruct = reconstruct

    with pytest.raises(ExportError, match="invalid vector"):
        export_subset(**kwargs)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*.building"))
