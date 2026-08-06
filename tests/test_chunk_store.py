from __future__ import annotations

import base64
import json

import pytest

from littraceqa.chunk_store import IMAGE_PATH_ERROR_KEY, ChunkStore


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _record(paper_id: str, chunk_id: str, text: str = "本文", **metadata) -> dict:
    return {
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "chunk_type": "text_span",
        "text": text,
        "metadata": metadata,
    }


def _write_chunks(path, records: list[dict], final_newline: bool = True) -> None:
    text = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(text + ("\n" if final_newline else ""), encoding="utf-8")


def test_random_access_uses_utf8_byte_offsets_and_no_final_newline(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks,
        [_record("p1", "p1#1", "日本語"), _record("p2", "p2#1", "answer")],
        final_newline=False,
    )
    store = ChunkStore(chunks)

    assert [item["chunk_id"] for item in store.load_paper("p2")] == ["p2#1"]
    assert store.load_paper("missing") == []
    assert len(store) == 2


def test_noncontiguous_paper_blocks_are_rejected(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks,
        [_record("p1", "1"), _record("p2", "2"), _record("p1", "3")],
    )
    with pytest.raises(ValueError, match="連続していない"):
        len(ChunkStore(chunks))


def test_corrupt_offset_index_is_rebuilt(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    index = tmp_path / "custom.offsets.json"
    _write_chunks(chunks, [_record("p1", "1", page=1)])
    stat = chunks.stat()
    index.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "offsets": {"p1": [-1, 999999]},
            }
        ),
        encoding="utf-8",
    )
    store = ChunkStore(chunks, index_path=index)

    assert store.load_paper("p1")[0]["chunk_id"] == "1"
    rebuilt = json.loads(index.read_text(encoding="utf-8"))
    assert rebuilt["offsets"]["p1"][0] == 0


def test_stale_index_is_rebuilt(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    store = ChunkStore(chunks) if chunks.exists() else None
    assert store is None
    _write_chunks(chunks, [_record("p1", "1")])
    first = ChunkStore(chunks)
    assert len(first) == 1
    _write_chunks(chunks, [_record("p1", "1"), _record("p2", "2")])

    assert len(ChunkStore(chunks)) == 2


def test_live_store_reloads_offsets_when_corpus_changes(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [_record("p1", "1")])
    store = ChunkStore(chunks)
    assert store.paper_ids() == ["p1"]

    _write_chunks(chunks, [_record("p1", "1"), _record("p2", "2")])

    assert store.paper_ids() == ["p1", "p2"]
    assert store.load_paper("p2")[0]["chunk_id"] == "2"


def test_structurally_valid_but_swapped_index_is_rebuilt(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    index = tmp_path / "chunks.offsets.json"
    _write_chunks(chunks, [_record("p1", "1"), _record("p2", "2")])
    clean = ChunkStore(chunks, index_path=index)
    assert len(clean) == 2

    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["offsets"]["p1"], payload["offsets"]["p2"] = (
        payload["offsets"]["p2"],
        payload["offsets"]["p1"],
    )
    index.write_text(json.dumps(payload), encoding="utf-8")

    repaired = ChunkStore(chunks, index_path=index)

    assert repaired.load_paper("p1")[0]["chunk_id"] == "1"
    rebuilt = json.loads(index.read_text(encoding="utf-8"))
    assert rebuilt["offsets"]["p1"][0] == 0


def test_image_paths_can_be_rebased(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    image_root = tmp_path / "images"
    target = image_root / "p1" / "auto" / "images" / "figure.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(VALID_PNG)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["figure_id"] = "Figure 1"
    record["metadata"]["image_path"] = "/old/root/p1/auto/images/figure.jpg"
    _write_chunks(chunks, [record])

    figures = ChunkStore(chunks, image_root=image_root).figures("p1")

    assert len(figures) == 1
    assert figures[0]["metadata"]["image_path"] == str(target)


def test_rebase_rejects_external_path_without_paper_id(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    image_root = tmp_path / "images"
    image_root.mkdir()
    external = tmp_path / "secret.png"
    external.write_bytes(VALID_PNG)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = str(external)
    _write_chunks(chunks, [record])

    loaded = ChunkStore(chunks, image_root=image_root).load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == ""
    assert "paper_id/auto/images" in loaded["metadata"][IMAGE_PATH_ERROR_KEY]


def test_image_path_is_disabled_without_explicit_image_root(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    external = tmp_path / "secret.png"
    external.write_bytes(VALID_PNG)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = str(external)
    _write_chunks(chunks, [record])

    store = ChunkStore(chunks)
    loaded = store.load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == ""
    assert "image_root is required" in loaded["metadata"][IMAGE_PATH_ERROR_KEY]
    assert store.figures("p1") == []


def test_rebase_rejects_dotdot_traversal(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    image_root = tmp_path / "images"
    image_root.mkdir()
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = "/old/p1/../../secret.png"
    _write_chunks(chunks, [record])

    loaded = ChunkStore(chunks, image_root=image_root).load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == ""
    assert "traversal" in loaded["metadata"][IMAGE_PATH_ERROR_KEY]


def test_rebase_rejects_image_symlink_outside_root(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    image_root = tmp_path / "images"
    image_dir = image_root / "p1" / "auto" / "images"
    image_dir.mkdir(parents=True)
    external = tmp_path / "secret.png"
    external.write_bytes(VALID_PNG)
    (image_dir / "figure.png").symlink_to(external)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = "/old/p1/auto/images/figure.png"
    _write_chunks(chunks, [record])

    loaded = ChunkStore(chunks, image_root=image_root).load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == ""
    assert "outside configured image_root" in loaded["metadata"][IMAGE_PATH_ERROR_KEY]


def test_rebase_stores_canonical_path_when_image_root_is_a_symlink(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    real_root = tmp_path / "real-images"
    target = real_root / "p1" / "auto" / "images" / "figure.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(VALID_PNG)
    linked_root = tmp_path / "linked-images"
    linked_root.symlink_to(real_root, target_is_directory=True)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = "/old/p1/auto/images/figure.png"
    _write_chunks(chunks, [record])

    loaded = ChunkStore(chunks, image_root=linked_root).load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == str(target.resolve())


def test_image_root_symlink_is_frozen_when_store_is_constructed(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    first_root = tmp_path / "first-images"
    first_target = first_root / "p1" / "auto" / "images" / "figure.png"
    first_target.parent.mkdir(parents=True)
    first_target.write_bytes(VALID_PNG)
    second_root = tmp_path / "second-images"
    second_target = second_root / "p1" / "auto" / "images" / "figure.png"
    second_target.parent.mkdir(parents=True)
    second_target.write_bytes(VALID_PNG)
    linked_root = tmp_path / "linked-images"
    linked_root.symlink_to(first_root, target_is_directory=True)
    record = _record("p1", "1", page=2)
    record["chunk_type"] = "figure"
    record["metadata"]["image_path"] = "/old/p1/auto/images/figure.png"
    _write_chunks(chunks, [record])

    store = ChunkStore(chunks, image_root=linked_root)
    linked_root.unlink()
    linked_root.symlink_to(second_root, target_is_directory=True)
    loaded = store.load_paper("p1")[0]

    assert loaded["metadata"]["image_path"] == str(first_target.resolve())
