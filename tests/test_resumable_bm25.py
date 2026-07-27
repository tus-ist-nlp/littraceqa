"""Tests for the memory-bounded resumable bm25s construction core."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import bm25s
import numpy as np
import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.resumable_bm25 import (
    ConfigurationChangedError,
    CorruptCheckpointError,
    InputChangedError,
    ResumableBM25Builder,
)


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"paper-{index}#c0000",
            paper_id=f"paper-{index}",
            text=text,
            chunk_type="text_span",
            metadata={"order": index},
        )
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    "method", ["lucene", "robertson", "atire", "bm25l", "bm25+"]
)
def test_scores_and_rankings_match_bm25s_exactly(tmp_path, method):
    texts = [
        "neural retrieval retrieval for scientific papers",
        "scientific document question answering",
        "graph retrieval with tables and figures",
        "unrelated language modeling baseline",
    ]
    params = {
        "method": method,
        "idf_method": method,
        "k1": 1.3,
        "b": 0.65,
        "delta": 0.7,
    }
    reference_tokens = bm25s.tokenize(
        texts, stopwords="en", show_progress=False
    )
    reference = bm25s.BM25(**params)
    reference.index(reference_tokens, show_progress=False)

    generation = tmp_path / method
    result = ResumableBM25Builder(
        generation, batch_size=2, **params
    ).build(_chunks(texts))
    loaded = bm25s.BM25.load(
        str(generation), load_corpus=False, mmap=True
    )

    assert isinstance(loaded.scores["data"], np.memmap)
    assert loaded.vocab_dict == reference.vocab_dict
    np.testing.assert_array_equal(loaded.scores["data"], reference.scores["data"])
    np.testing.assert_array_equal(
        loaded.scores["indices"], reference.scores["indices"]
    )
    np.testing.assert_array_equal(
        loaded.scores["indptr"], reference.scores["indptr"]
    )
    if method in ("bm25l", "bm25+"):
        np.testing.assert_array_equal(
            loaded.nonoccurrence_array,
            reference.nonoccurrence_array,
        )
    query = bm25s.tokenize(
        ["scientific neural retrieval"], stopwords="en", show_progress=False
    )
    actual_indices, actual_scores = loaded.retrieve(
        query, k=4, show_progress=False
    )
    expected_indices, expected_scores = reference.retrieve(
        query, k=4, show_progress=False
    )
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_scores, expected_scores)
    assert result.num_documents == 4
    assert result.scored_batches == 2
    assert result.rebuilt_score_batches == 2
    assert result.reused_score_batches == 0


def test_chunk_offsets_are_exact_utf8_jsonl_boundaries(tmp_path):
    chunks = _chunks(
        [
            "naïve 日本語 alpha\nsecond line",
            "emoji 📄 beta and a literal \\\\n sequence",
        ]
    )
    chunks[0].metadata["note"] = "escaped\nmetadata"
    generation = tmp_path / "index"

    ResumableBM25Builder(generation, batch_size=1).build(chunks)

    expected_lines = [
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for chunk in chunks
    ]
    expected_offsets = [0]
    for line in expected_lines:
        expected_offsets.append(expected_offsets[-1] + len(line))

    assert (generation / "chunks.jsonl").read_bytes() == b"".join(
        expected_lines
    )
    offsets = np.load(
        generation / "chunks.offsets.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert offsets.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(
        offsets, np.asarray(expected_offsets, dtype=np.uint64)
    )
    bm25s.BM25.load(str(generation), mmap=True, load_corpus=False)


def test_completed_resume_reuses_token_and_score_batches(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta", "epsilon zeta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    first = builder.build(chunks)

    def unexpected_tokenize(*args, **kwargs):
        raise AssertionError("tokenization should have been reused")

    def unexpected_scoring(*args, **kwargs):
        raise AssertionError("score calculation should have been reused")

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.resumable_bm25.bm25s.tokenize",
        unexpected_tokenize,
    )
    monkeypatch.setattr(builder, "_score_arrays", unexpected_scoring)
    second = builder.build(iter(chunks))

    assert second.input_sha256 == first.input_sha256
    assert second.reused_batches == 3
    assert second.rebuilt_batches == 0
    assert second.reused_score_batches == 3
    assert second.rebuilt_score_batches == 0


def test_resume_after_interrupted_input_reuses_checkpoint(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)

    def interrupted():
        yield chunks[0]
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        builder.build(interrupted())

    calls = 0
    original = bm25s.tokenize

    def counting_tokenize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.resumable_bm25.bm25s.tokenize",
        counting_tokenize,
    )
    result = builder.build(chunks)

    assert calls == 1
    assert result.reused_batches == 1
    assert result.rebuilt_batches == 1
    assert result.num_documents == 2


def test_resume_after_score_stage_failure_reuses_completed_score_shard(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    original = builder._write_score_shard
    calls = 0

    def fail_second(index, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated score interruption")
        return original(index, **kwargs)

    monkeypatch.setattr(builder, "_write_score_shard", fail_second)
    with pytest.raises(RuntimeError, match="simulated score interruption"):
        builder.build(chunks)

    monkeypatch.setattr(builder, "_write_score_shard", original)
    result = builder.build(chunks)

    assert result.reused_score_batches == 1
    assert result.rebuilt_score_batches == 1
    bm25s.BM25.load(str(generation), mmap=True, load_corpus=False)


def test_changed_input_invalidates_existing_generation(tmp_path):
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(_chunks(["alpha beta", "gamma delta"]))
    manifest_before = (generation / "resumable-build.json").read_bytes()

    with pytest.raises(InputChangedError, match="batch 1"):
        builder.build(_chunks(["alpha beta", "changed document"]))

    assert (generation / "resumable-build.json").read_bytes() == manifest_before


def test_corrupt_token_part_is_rebuilt_from_matching_input(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(chunks)
    part = generation / ".resumable-bm25-parts" / "00000001.json"
    part.write_bytes(b"broken")
    calls = 0
    original = bm25s.tokenize

    def counting_tokenize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.resumable_bm25.bm25s.tokenize",
        counting_tokenize,
    )
    result = builder.build(chunks)

    assert calls == 1
    assert result.rebuilt_batches == 1
    assert result.reused_batches == 1
    assert part.read_bytes() != b"broken"
    bm25s.BM25.load(str(generation), mmap=True, load_corpus=False)


def test_corrupt_score_shard_alone_is_rebuilt(monkeypatch, tmp_path):
    chunks = _chunks(["alpha beta", "gamma delta", "epsilon zeta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(chunks)
    shard = generation / ".resumable-bm25-scores" / "00000001.npz"
    shard.write_bytes(b"broken")
    calls = 0
    original = builder._score_arrays

    def counting_score(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(builder, "_score_arrays", counting_score)
    result = builder.build(chunks)

    assert calls == 1
    assert result.reused_score_batches == 2
    assert result.rebuilt_score_batches == 1
    assert shard.read_bytes() != b"broken"


def test_wrong_score_statistics_signature_rebuilds_only_that_shard(tmp_path):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(chunks)
    meta_path = (
        generation / ".resumable-bm25-scores" / "00000001.meta.json"
    )
    meta = json.loads(meta_path.read_text())
    meta["statistics_signature"] = "0" * 64
    meta_path.write_text(json.dumps(meta))

    result = builder.build(chunks)

    assert result.reused_score_batches == 1
    assert result.rebuilt_score_batches == 1


def test_corrupt_completed_output_is_repaired_from_score_shards(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(chunks)
    data_path = generation / "data.csc.index.npy"
    data_path.write_bytes(b"broken")

    def unexpected_scoring(*args, **kwargs):
        raise AssertionError("valid score shards should be reused")

    monkeypatch.setattr(builder, "_score_arrays", unexpected_scoring)
    result = builder.build(chunks)

    assert result.reused_score_batches == 2
    assert result.rebuilt_score_batches == 0
    assert data_path.read_bytes() != b"broken"
    bm25s.BM25.load(str(generation), mmap=True, load_corpus=False)


def test_corrupt_chunk_offsets_are_repaired_from_score_shards(
    monkeypatch, tmp_path
):
    chunks = _chunks(["alpha beta", "gamma delta"])
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation, batch_size=1)
    builder.build(chunks)
    offsets_path = generation / "chunks.offsets.npy"
    chunks_size = (generation / "chunks.jsonl").stat().st_size
    with offsets_path.open("wb") as output:
        np.save(
            output,
            np.asarray([0, 1, chunks_size], dtype=np.uint64),
            allow_pickle=False,
        )
    manifest_path = generation / "resumable-build.json"
    manifest = json.loads(manifest_path.read_text())
    content = offsets_path.read_bytes()
    manifest["files"]["chunks.offsets.npy"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    manifest_path.write_text(json.dumps(manifest))

    def unexpected_scoring(*args, **kwargs):
        raise AssertionError("valid score shards should be reused")

    monkeypatch.setattr(builder, "_score_arrays", unexpected_scoring)
    result = builder.build(chunks)

    assert result.reused_score_batches == 2
    assert result.rebuilt_score_batches == 0
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    assert offsets.dtype == np.dtype("uint64")
    assert offsets.shape == (3,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == (generation / "chunks.jsonl").stat().st_size
    bm25s.BM25.load(str(generation), mmap=True, load_corpus=False)


@pytest.mark.parametrize("texts", [[], ["and the or"]])
def test_empty_or_empty_token_corpus_is_rejected(tmp_path, texts):
    builder = ResumableBM25Builder(tmp_path / "index")
    with pytest.raises(ValueError, match="empty corpus|no searchable tokens"):
        builder.build(_chunks(texts))


def test_corrupt_root_manifest_is_rejected(tmp_path):
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation)
    builder.build(_chunks(["alpha beta"]))
    manifest_path = generation / "resumable-build.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CorruptCheckpointError, match="manifest"):
        builder.build(_chunks(["alpha beta"]))


def test_corrupt_token_metadata_is_rejected(tmp_path):
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation)
    chunks = _chunks(["alpha beta"])
    builder.build(chunks)
    meta_path = (
        generation / ".resumable-bm25-parts" / "00000000.meta.json"
    )
    meta = json.loads(meta_path.read_text())
    meta["part_size"] = builder.max_part_bytes + 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(CorruptCheckpointError, match="metadata"):
        builder.build(chunks)


def test_malformed_npz_with_valid_checksum_is_rebuilt(tmp_path):
    generation = tmp_path / "index"
    builder = ResumableBM25Builder(generation)
    chunks = _chunks(["alpha beta"])
    builder.build(chunks)
    score_path = generation / ".resumable-bm25-scores" / "00000000.npz"
    with score_path.open("wb") as output:
        np.savez(
            output,
            data=np.asarray([1.0], dtype=np.float32),
            wrong=np.asarray([0], dtype=np.int32),
        )
    meta_path = (
        generation / ".resumable-bm25-scores" / "00000000.meta.json"
    )
    meta = json.loads(meta_path.read_text())
    content = score_path.read_bytes()
    meta["score_sha256"] = hashlib.sha256(content).hexdigest()
    meta["score_size"] = len(content)
    meta_path.write_text(json.dumps(meta))

    result = builder.build(chunks)

    assert result.rebuilt_score_batches == 1
    assert result.reused_score_batches == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 1.5}, TypeError),
        ({"max_batch_characters": 0}, ValueError),
        ({"max_part_bytes": 0}, ValueError),
        ({"method": "unknown"}, ValueError),
        ({"idf_method": "unknown"}, ValueError),
        ({"k1": -1}, ValueError),
        ({"b": 1.1}, ValueError),
        ({"delta": -1}, ValueError),
        ({"dtype": "int32"}, ValueError),
        ({"int_dtype": "float32"}, ValueError),
        ({"lower": 1}, TypeError),
        ({"token_pattern": "["}, ValueError),
    ],
)
def test_build_parameter_validation(tmp_path, kwargs, error):
    with pytest.raises(error):
        ResumableBM25Builder(tmp_path / "index", **kwargs)


def test_changed_configuration_is_rejected(tmp_path):
    generation = tmp_path / "index"
    chunks = _chunks(["alpha beta"])
    ResumableBM25Builder(generation, batch_size=1).build(chunks)

    with pytest.raises(ConfigurationChangedError):
        ResumableBM25Builder(generation, batch_size=2).build(chunks)


def test_changed_implementation_signature_is_rejected(tmp_path):
    generation = tmp_path / "index"
    chunks = _chunks(["alpha beta"])
    ResumableBM25Builder(generation).build(chunks)
    changed = ResumableBM25Builder(generation)
    changed.implementation_signature = "0" * 64

    with pytest.raises(ConfigurationChangedError):
        changed.build(chunks)


def test_insufficient_disk_space_stops_before_scoring(
    monkeypatch,
    tmp_path,
):
    builder = ResumableBM25Builder(tmp_path / "index")
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.resumable_bm25.shutil.disk_usage",
        lambda path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="insufficient free space"):
        builder.build(_chunks(["alpha beta"]))

    assert not (builder.generation_dir / ".resumable-bm25-scores").exists()
