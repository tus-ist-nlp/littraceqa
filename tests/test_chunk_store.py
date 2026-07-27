"""Tests for lazy, offset-based Chunk JSONL access."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.chunk_store import ChunkJsonlStore


def _records() -> list[dict]:
    return [
        {
            "chunk_id": "paper-1#c0000",
            "paper_id": "paper-1",
            "text": "first α",
            "chunk_type": "text_span",
            "metadata": {"page": 1},
        },
        {
            "chunk_id": "paper-2#c0000",
            "paper_id": "paper-2",
            "text": "second",
            "chunk_type": "figure",
            "metadata": {"visible_id": "Figure 1"},
        },
        {
            "chunk_id": "paper-3#c0000",
            "paper_id": "paper-3",
            "text": "third",
            "chunk_type": "table",
            "metadata": {},
        },
    ]


def _write_store(
    directory: Path,
    records: list[object] | None = None,
) -> tuple[Path, Path]:
    jsonl_path = directory / "chunks.jsonl"
    offsets_path = directory / "chunks.offsets.npy"
    payloads = [
        json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        for record in (records if records is not None else _records())
    ]
    jsonl_path.write_bytes(b"".join(payloads))
    offsets = [0]
    for payload in payloads:
        offsets.append(offsets[-1] + len(payload))
    np.save(offsets_path, np.asarray(offsets, dtype=np.uint64))
    return jsonl_path, offsets_path


def test_integer_negative_slice_and_iteration(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(
        jsonl_path,
        offsets_path,
        expected_documents=3,
    )

    assert len(store) == 3
    assert isinstance(store._offsets, np.memmap)
    assert not store._offsets.flags.writeable
    assert store[0] == Chunk(**_records()[0])
    assert store[-1].paper_id == "paper-3"
    assert [chunk.paper_id for chunk in store[::2]] == [
        "paper-1",
        "paper-3",
    ]
    assert [chunk.paper_id for chunk in store[::-1]] == [
        "paper-3",
        "paper-2",
        "paper-1",
    ]
    assert [chunk.paper_id for chunk in store] == [
        "paper-1",
        "paper-2",
        "paper-3",
    ]


def test_index_errors_match_sequence_behavior(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(jsonl_path, offsets_path)

    with pytest.raises(IndexError, match="out of range"):
        _ = store[3]
    with pytest.raises(IndexError, match="out of range"):
        _ = store[-4]
    with pytest.raises(TypeError, match="integers or slices"):
        _ = store["0"]  # type: ignore[index]


def test_integer_access_reads_and_decodes_only_the_selected_record(
    monkeypatch,
    tmp_path,
):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(jsonl_path, offsets_path)
    pread_calls: list[tuple[int, int]] = []
    decode_calls: list[str] = []
    real_pread = os.pread
    real_loads = json.loads

    def recording_pread(descriptor: int, count: int, offset: int) -> bytes:
        pread_calls.append((count, offset))
        return real_pread(descriptor, count, offset)

    def recording_loads(value: str):
        decode_calls.append(value)
        return real_loads(value)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.chunk_store.os.pread",
        recording_pread,
    )
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.chunk_store.json.loads",
        recording_loads,
    )

    selected = store[1]

    raw_lines = jsonl_path.read_bytes().splitlines(keepends=True)
    assert selected.paper_id == "paper-2"
    assert pread_calls == [(len(raw_lines[1]), len(raw_lines[0]))]
    assert len(decode_calls) == 1
    assert decode_calls[0].encode("utf-8") == raw_lines[1]


def test_integer_access_uses_no_follow_and_no_persistent_descriptor(
    monkeypatch,
    tmp_path,
):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(jsonl_path, offsets_path)
    flags_seen: list[int] = []
    descriptors_opened: list[int] = []
    descriptors_closed: list[int] = []
    real_open = os.open
    real_close = os.close

    def recording_open(path, flags, mode=0o777):
        flags_seen.append(flags)
        descriptor = real_open(path, flags, mode)
        descriptors_opened.append(descriptor)
        return descriptor

    def recording_close(descriptor: int):
        descriptors_closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.chunk_store.os.open",
        recording_open,
    )
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.chunk_store.os.close",
        recording_close,
    )

    assert store[0].paper_id == "paper-1"
    assert store[1].paper_id == "paper-2"

    assert len(flags_seen) == 2
    if hasattr(os, "O_NOFOLLOW"):
        assert all(flags & os.O_NOFOLLOW for flags in flags_seen)
    assert descriptors_closed == descriptors_opened


def test_concurrent_reads_do_not_share_a_file_position(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(jsonl_path, offsets_path)
    positions = [2, 0, 1, 2, 1, 0] * 20

    with ThreadPoolExecutor(max_workers=8) as executor:
        paper_ids = list(
            executor.map(lambda index: store[index].paper_id, positions)
        )

    assert paper_ids == [f"paper-{position + 1}" for position in positions]


@pytest.mark.parametrize(
    ("offsets", "error"),
    [
        (np.asarray([0, 1], dtype=np.int64), "dtype uint64"),
        (np.asarray([[0, 1]], dtype=np.uint64), "one-dimensional"),
        (np.asarray([], dtype=np.uint64), "initial zero"),
        (np.asarray([1, 2], dtype=np.uint64), "start at zero"),
        (
            np.asarray([0, 2, 1], dtype=np.uint64),
            "monotonically non-decreasing",
        ),
        (
            np.asarray([0, 1], dtype=np.uint64),
            "JSONL byte size",
        ),
    ],
)
def test_rejects_invalid_offset_arrays(tmp_path, offsets, error):
    jsonl_path, offsets_path = _write_store(tmp_path)
    np.save(offsets_path, offsets)

    with pytest.raises(ValueError, match=error):
        ChunkJsonlStore(jsonl_path, offsets_path)


def test_rejects_expected_document_count_mismatch(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)

    with pytest.raises(ValueError, match="expected_documents"):
        ChunkJsonlStore(
            jsonl_path,
            offsets_path,
            expected_documents=2,
        )


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_rejects_invalid_expected_document_count_type(tmp_path, value):
    jsonl_path, offsets_path = _write_store(tmp_path)

    with pytest.raises(TypeError, match="expected_documents"):
        ChunkJsonlStore(
            jsonl_path,
            offsets_path,
            expected_documents=value,  # type: ignore[arg-type]
        )


def test_rejects_negative_expected_document_count(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)

    with pytest.raises(ValueError, match="must not be negative"):
        ChunkJsonlStore(
            jsonl_path,
            offsets_path,
            expected_documents=-1,
        )


def test_rejects_short_read(monkeypatch, tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path)
    store = ChunkJsonlStore(jsonl_path, offsets_path)
    real_pread = os.pread

    def short_pread(descriptor: int, count: int, offset: int) -> bytes:
        return real_pread(descriptor, max(0, count - 1), offset)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.chunk_store.os.pread",
        short_pread,
    )

    with pytest.raises(ValueError, match="short read"):
        _ = store[0]


def test_rejects_record_without_final_lf(tmp_path):
    jsonl_path = tmp_path / "chunks.jsonl"
    offsets_path = tmp_path / "chunks.offsets.npy"
    payload = json.dumps(_records()[0]).encode("utf-8")
    jsonl_path.write_bytes(payload)
    np.save(offsets_path, np.asarray([0, len(payload)], dtype=np.uint64))

    store = ChunkJsonlStore(jsonl_path, offsets_path)
    with pytest.raises(ValueError, match="does not end with LF"):
        _ = store[0]


def test_rejects_invalid_utf8(tmp_path):
    jsonl_path = tmp_path / "chunks.jsonl"
    offsets_path = tmp_path / "chunks.offsets.npy"
    payload = b"\xff\n"
    jsonl_path.write_bytes(payload)
    np.save(offsets_path, np.asarray([0, len(payload)], dtype=np.uint64))

    store = ChunkJsonlStore(jsonl_path, offsets_path)
    with pytest.raises(ValueError, match="not valid UTF-8"):
        _ = store[0]


def test_rejects_invalid_json(tmp_path):
    jsonl_path = tmp_path / "chunks.jsonl"
    offsets_path = tmp_path / "chunks.offsets.npy"
    payload = b"{not-json}\n"
    jsonl_path.write_bytes(payload)
    np.save(offsets_path, np.asarray([0, len(payload)], dtype=np.uint64))

    store = ChunkJsonlStore(jsonl_path, offsets_path)
    with pytest.raises(ValueError, match="not valid JSON"):
        _ = store[0]


def test_rejects_non_object_json(tmp_path):
    jsonl_path, offsets_path = _write_store(tmp_path, records=[["not", "object"]])
    store = ChunkJsonlStore(jsonl_path, offsets_path)

    with pytest.raises(ValueError, match="must be a JSON object"):
        _ = store[0]


@pytest.mark.parametrize(
    "record",
    [
        {
            "paper_id": "paper-1",
            "text": "missing chunk id",
            "chunk_type": "text_span",
            "metadata": {},
        },
        {
            "chunk_id": "paper-1#c0000",
            "paper_id": "paper-1",
            "text": "wrong metadata type",
            "chunk_type": "text_span",
            "metadata": [],
        },
    ],
)
def test_rejects_records_that_violate_chunk_contract(tmp_path, record):
    jsonl_path, offsets_path = _write_store(tmp_path, records=[record])
    store = ChunkJsonlStore(jsonl_path, offsets_path)

    with pytest.raises(ValueError, match="Chunk contract"):
        _ = store[0]
