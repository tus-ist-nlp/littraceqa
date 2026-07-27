"""Focused tests for configurable bm25s index construction."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.bm25_index import (
    BM25BuildLockError,
    BM25Index,
)
from littraceqa.di_pipeline.index.chunk_store import ChunkJsonlStore


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="paper-1#c0000",
        paper_id="paper-1",
        text="retrieval text",
        chunk_type="text_span",
        metadata={},
    )


def test_build_passes_explicit_bm25_parameters(monkeypatch, tmp_path):
    created: list[dict] = []

    class FakeBM25:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def index(self, corpus_tokens):
            self.corpus_tokens = corpus_tokens

        def save(self, index_dir):
            self.index_dir = index_dir

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.tokenize",
        lambda texts, stopwords: [texts, stopwords],
    )
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25", FakeBM25
    )

    index = BM25Index(
        str(tmp_path),
        method="bm25+",
        idf_method="robertson",
        k1=1.2,
        b=0.6,
        delta=0.8,
    )
    index.build([_chunk()])

    assert created == [
        {
            "method": "bm25+",
            "k1": 1.2,
            "b": 0.6,
            "delta": 0.8,
            "idf_method": "robertson",
        }
    ]
    assert index.build_params == created[0]


def test_default_idf_method_matches_scoring_method(tmp_path):
    index = BM25Index(str(tmp_path), method="atire")

    assert index.build_params["idf_method"] == "atire"


@pytest.mark.parametrize(
    ("argument", "value", "error", "message"),
    [
        ("resumable_build", 1, TypeError, "must be a bool"),
        ("build_batch_size", True, TypeError, "must be an integer"),
        ("build_batch_size", 1.5, TypeError, "must be an integer"),
        ("build_batch_size", 0, ValueError, "must be positive"),
        ("max_batch_characters", True, TypeError, "must be an integer"),
        ("max_batch_characters", 1.5, TypeError, "must be an integer"),
        ("max_batch_characters", 0, ValueError, "must be positive"),
    ],
)
def test_resumable_build_options_are_validated(
    tmp_path, argument, value, error, message
):
    with pytest.raises(error, match=message):
        BM25Index(str(tmp_path), **{argument: value})


def test_resumable_build_rejects_custom_records_filename(tmp_path):
    with pytest.raises(ValueError, match="requires records_filename"):
        BM25Index(
            str(tmp_path),
            records_filename="records.jsonl",
            resumable_build=True,
        )


def test_resumable_build_delegates_and_loads_completed_index(
    monkeypatch, tmp_path
):
    builder_calls: list[dict] = []
    built_inputs: list[list[Chunk]] = []
    load_calls: list[bool] = []

    class FakeBuilder:
        def __init__(self, generation_dir, **kwargs):
            builder_calls.append(
                {"generation_dir": generation_dir, **kwargs}
            )

        def build(self, chunks):
            built_inputs.append(list(chunks))
            Path(self.generation_dir).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(input_sha256="a" * 64)

        @property
        def generation_dir(self):
            return builder_calls[-1]["generation_dir"]

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        FakeBuilder,
    )
    index = BM25Index(
        str(tmp_path),
        method="bm25+",
        idf_method="robertson",
        k1=1.2,
        b=0.6,
        delta=0.8,
        resumable_build=True,
        build_batch_size=3,
        max_batch_characters=1234,
    )
    monkeypatch.setattr(index, "load", lambda: load_calls.append(True))

    index.build(iter([_chunk()]))

    assert built_inputs == [[_chunk()]]
    assert load_calls == [True]
    assert builder_calls == [
        {
            "generation_dir": tmp_path / ".resumable-staging",
            "batch_size": 3,
            "max_batch_characters": 1234,
            "method": "bm25+",
            "k1": 1.2,
            "b": 0.6,
            "delta": 0.8,
            "idf_method": "robertson",
        }
    ]


def test_resumable_build_with_signature_uses_isolated_staging(
    monkeypatch,
    tmp_path,
):
    signature = "1" * 64
    builder_directories: list[Path] = []

    class FakeBuilder:
        def __init__(self, generation_dir, **kwargs):
            self.generation_dir = Path(generation_dir)
            builder_directories.append(self.generation_dir)

        def build(self, chunks):
            list(chunks)
            self.generation_dir.mkdir(parents=True)
            return SimpleNamespace(input_sha256="a" * 64)

    index = BM25Index(str(tmp_path), resumable_build=True)
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        FakeBuilder,
    )
    monkeypatch.setattr(index, "load", lambda: None)

    index.build_with_signature(iter([_chunk()]), signature)

    assert builder_directories == [
        tmp_path / f".resumable-staging-{signature}"
    ]


@pytest.mark.parametrize(
    "signature",
    [
        None,
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "../" + "a" * 61,
    ],
)
def test_build_with_signature_rejects_invalid_sha256(tmp_path, signature):
    index = BM25Index(str(tmp_path), resumable_build=True)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        index.build_with_signature([_chunk()], signature)

    assert list(tmp_path.iterdir()) == []


def test_changed_build_signature_keeps_old_staging_untouched(
    monkeypatch,
    tmp_path,
):
    old_signature = "1" * 64
    new_signature = "2" * 64
    old_staging = tmp_path / f".resumable-staging-{old_signature}"
    old_staging.mkdir(parents=True)
    old_checkpoint = old_staging / "checkpoint"
    old_checkpoint.write_text("old state", encoding="utf-8")
    builder_directories: list[Path] = []

    class InterruptedBuilder:
        def __init__(self, generation_dir, **kwargs):
            self.generation_dir = Path(generation_dir)
            builder_directories.append(self.generation_dir)

        def build(self, chunks):
            list(chunks)
            self.generation_dir.mkdir(parents=True)
            (self.generation_dir / "checkpoint").write_text(
                "new state",
                encoding="utf-8",
            )
            raise RuntimeError("simulated interruption")

    index = BM25Index(str(tmp_path), resumable_build=True)
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        InterruptedBuilder,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index.build_with_signature([_chunk()], new_signature)

    assert builder_directories == [
        tmp_path / f".resumable-staging-{new_signature}"
    ]
    assert old_checkpoint.read_text(encoding="utf-8") == "old state"
    assert (
        tmp_path / f".resumable-staging-{new_signature}" / "checkpoint"
    ).read_text(encoding="utf-8") == "new state"


def test_same_build_signature_resumes_its_staging(
    monkeypatch,
    tmp_path,
):
    signature = "3" * 64
    staging = tmp_path / f".resumable-staging-{signature}"
    staging.mkdir(parents=True)
    (staging / "checkpoint").write_text("partial", encoding="utf-8")
    saw_checkpoint: list[bool] = []

    class FakeBuilder:
        def __init__(self, generation_dir, **kwargs):
            self.generation_dir = Path(generation_dir)

        def build(self, chunks):
            list(chunks)
            saw_checkpoint.append(
                (self.generation_dir / "checkpoint").is_file()
            )
            return SimpleNamespace(input_sha256="c" * 64)

    index = BM25Index(str(tmp_path), resumable_build=True)
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        FakeBuilder,
    )
    monkeypatch.setattr(index, "load", lambda: None)

    index.build_with_signature([_chunk()], signature)

    assert saw_checkpoint == [True]
    pointer = json.loads((tmp_path / "CURRENT.json").read_text())
    assert (
        tmp_path / pointer["generation"] / "checkpoint"
    ).read_text(encoding="utf-8") == "partial"


def test_resumable_build_can_resume_and_search(tmp_path):
    chunks = [
        _chunk(),
        Chunk(
            chunk_id="paper-2#c0000",
            paper_id="paper-2",
            text="quantum banana banana",
            chunk_type="text_span",
            metadata={"kind": "target"},
        ),
        Chunk(
            chunk_id="paper-3#c0000",
            paper_id="paper-3",
            text="ocean coral ecology",
            chunk_type="text_span",
            metadata={},
        ),
    ]
    index_dir = tmp_path / "resumable-index"
    first = BM25Index(
        str(index_dir),
        resumable_build=True,
        build_batch_size=1,
    )
    first.build(iter(chunks))

    pointer = json.loads((index_dir / "CURRENT.json").read_text())
    generation_dir = index_dir / pointer["generation"]
    assert (generation_dir / "resumable-build.json").is_file()
    assert not (index_dir / ".resumable-staging").exists()
    assert first.search("quantum banana", top_k=1)[0].paper_id == "paper-2"

    resumed = BM25Index(
        str(index_dir),
        resumable_build=True,
        build_batch_size=1,
    )
    resumed.build(iter(chunks))
    result = resumed.search("quantum banana", top_k=1)[0]

    assert result.paper_id == "paper-2"
    assert result.metadata == {"kind": "target"}


def test_default_build_keeps_legacy_path(monkeypatch, tmp_path):
    class UnexpectedBuilder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("resumable builder must remain opt-in")

    class FakeBM25:
        def __init__(self, **kwargs):
            pass

        def index(self, corpus_tokens):
            pass

        def save(self, index_dir):
            pass

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        UnexpectedBuilder,
    )
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.tokenize",
        lambda texts, stopwords: [],
    )
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25", FakeBM25,
    )

    index = BM25Index(str(tmp_path))
    index.build([_chunk()])

    assert (tmp_path / "chunks.jsonl").is_file()


def test_constructor_does_not_create_index_directory(tmp_path):
    index_dir = tmp_path / "missing" / "index"

    BM25Index(str(index_dir))

    assert not index_dir.exists()


def test_load_missing_index_does_not_create_directory(tmp_path):
    index_dir = tmp_path / "shared" / "missing"
    index = BM25Index(str(index_dir))

    with pytest.raises(FileNotFoundError, match="index directory is missing"):
        index.load()

    assert not index_dir.exists()


@pytest.mark.parametrize("filename", ["", "../chunks.jsonl", "nested/chunks.jsonl"])
def test_records_filename_rejects_empty_or_nested_paths(tmp_path, filename):
    with pytest.raises(ValueError, match="non-empty file name"):
        BM25Index(str(tmp_path), records_filename=filename)


def test_load_uses_persisted_parameters_not_constructor_values(monkeypatch, tmp_path):
    persisted = SimpleNamespace(
        method="robertson",
        idf_method="lucene",
        k1=1.7,
        b=0.4,
        delta=0.3,
    )
    load_calls: list[tuple[str, bool, bool]] = []

    class FakeBM25:
        @classmethod
        def load(cls, index_dir, load_corpus, mmap):
            load_calls.append((index_dir, load_corpus, mmap))
            return persisted

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25", FakeBM25
    )
    index = BM25Index(str(tmp_path), method="bm25+", k1=9.0)
    monkeypatch.setattr(index, "_load_chunks", lambda: [])

    index.load()

    assert load_calls == [(str(tmp_path), False, True)]
    assert index.build_params == {
        "method": "robertson",
        "idf_method": "lucene",
        "k1": 1.7,
        "b": 0.4,
        "delta": 0.3,
    }


def _write_lazy_chunk_artifacts(directory: Path, chunks: list[Chunk]) -> None:
    lines = [
        json.dumps(chunk.to_dict(), ensure_ascii=False).encode("utf-8") + b"\n"
        for chunk in chunks
    ]
    (directory / "chunks.jsonl").write_bytes(b"".join(lines))
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    np.save(
        directory / "chunks.offsets.npy",
        np.asarray(offsets, dtype=np.uint64),
    )


def _persisted_bm25(num_docs: int) -> SimpleNamespace:
    return SimpleNamespace(
        method="lucene",
        idf_method="lucene",
        k1=1.5,
        b=0.75,
        delta=0.5,
        scores={"num_docs": num_docs},
    )


def test_load_uses_lazy_chunks_when_offset_sidecar_exists(
    monkeypatch,
    tmp_path,
):
    target = Chunk(
        chunk_id="paper-α#c0000",
        paper_id="paper-α",
        text="Unicode retrieval text",
        chunk_type="text_span",
        metadata={"page": 2},
    )
    _write_lazy_chunk_artifacts(tmp_path, [target])

    class FakeBM25:
        @classmethod
        def load(cls, index_dir, load_corpus, mmap):
            return _persisted_bm25(1)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25",
        FakeBM25,
    )
    index = BM25Index(str(tmp_path))

    index.load()

    assert isinstance(index._chunks, ChunkJsonlStore)
    assert index._chunks[0] == target


def test_load_fails_closed_for_corrupt_offset_sidecar(monkeypatch, tmp_path):
    _write_lazy_chunk_artifacts(tmp_path, [_chunk()])
    np.save(
        tmp_path / "chunks.offsets.npy",
        np.asarray([0, (tmp_path / "chunks.jsonl").stat().st_size], dtype=np.int64),
    )

    class FakeBM25:
        @classmethod
        def load(cls, index_dir, load_corpus, mmap):
            return _persisted_bm25(1)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25",
        FakeBM25,
    )

    with pytest.raises(ValueError, match="dtype uint64"):
        BM25Index(str(tmp_path)).load()


def test_load_rejects_offset_document_count_mismatch(monkeypatch, tmp_path):
    _write_lazy_chunk_artifacts(tmp_path, [_chunk()])

    class FakeBM25:
        @classmethod
        def load(cls, index_dir, load_corpus, mmap):
            return _persisted_bm25(2)

    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25",
        FakeBM25,
    )

    with pytest.raises(ValueError, match="expected_documents"):
        BM25Index(str(tmp_path)).load()


def test_load_uses_legacy_root_when_current_pointer_is_absent(
    monkeypatch, tmp_path
):
    persisted = SimpleNamespace(
        method="lucene",
        idf_method="lucene",
        k1=1.5,
        b=0.75,
        delta=0.5,
    )
    load_paths: list[str] = []

    class FakeBM25:
        @classmethod
        def load(cls, index_dir, load_corpus, mmap):
            load_paths.append(index_dir)
            return persisted

    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.bm25s.BM25", FakeBM25
    )

    BM25Index(str(tmp_path)).load()

    assert load_paths == [str(tmp_path)]


def test_resumable_build_publishes_generation_only_after_builder_completes(
    monkeypatch, tmp_path
):
    old_generation = tmp_path / "generations" / "old"
    old_generation.mkdir(parents=True)
    current = tmp_path / "CURRENT.json"
    current.write_text(
        json.dumps({"generation": "generations/old"}),
        encoding="utf-8",
    )
    observed_pointers: list[dict] = []

    class FakeBuilder:
        def __init__(self, generation_dir, **kwargs):
            self.generation_dir = Path(generation_dir)

        def build(self, chunks):
            observed_pointers.append(json.loads(current.read_text()))
            self.generation_dir.mkdir(parents=True)
            (self.generation_dir / "marker").write_text("complete")
            observed_pointers.append(json.loads(current.read_text()))
            return SimpleNamespace(input_sha256="b" * 64)

    index = BM25Index(str(tmp_path), resumable_build=True)
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        FakeBuilder,
    )
    monkeypatch.setattr(index, "load", lambda: None)

    index.build([_chunk()])

    assert observed_pointers == [
        {"generation": "generations/old"},
        {"generation": "generations/old"},
    ]
    published = json.loads(current.read_text())
    assert published["generation"].startswith(f"generations/{'b' * 64}-")
    assert (tmp_path / published["generation"] / "marker").read_text() == "complete"
    assert old_generation.is_dir()


@pytest.mark.parametrize(
    "generation",
    [
        "../outside",
        "generations/../../outside",
        "/tmp/outside",
    ],
)
def test_load_rejects_pointer_path_traversal(tmp_path, generation):
    (tmp_path / "CURRENT.json").write_text(
        json.dumps({"generation": generation}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes index root"):
        BM25Index(str(tmp_path)).load()


def test_load_rejects_generation_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    generations = tmp_path / "generations"
    generations.mkdir()
    (generations / "escaped").symlink_to(outside, target_is_directory=True)
    (tmp_path / "CURRENT.json").write_text(
        json.dumps({"generation": "generations/escaped"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes index root"):
        BM25Index(str(tmp_path)).load()


def test_load_rejects_current_pointer_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-current"
    outside.write_text(
        json.dumps({"generation": "generations/example"}),
        encoding="utf-8",
    )
    (tmp_path / "CURRENT.json").symlink_to(outside)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        BM25Index(str(tmp_path)).load()


def test_resumable_build_rejects_concurrent_builder(monkeypatch, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    lock_path = tmp_path / ".build.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = BM25Index(str(tmp_path), resumable_build=True)
        with pytest.raises(BM25BuildLockError, match="already running"):
            index.build([_chunk()])


def test_resumable_build_reuses_staging_after_interrupted_build(
    monkeypatch, tmp_path
):
    staging = tmp_path / ".resumable-staging"
    staging.mkdir(parents=True)
    checkpoint = staging / "checkpoint"
    checkpoint.write_text("partial", encoding="utf-8")
    saw_checkpoint: list[bool] = []

    class FakeBuilder:
        def __init__(self, generation_dir, **kwargs):
            self.generation_dir = Path(generation_dir)

        def build(self, chunks):
            saw_checkpoint.append((self.generation_dir / "checkpoint").is_file())
            return SimpleNamespace(input_sha256="c" * 64)

    index = BM25Index(str(tmp_path), resumable_build=True)
    monkeypatch.setattr(
        "littraceqa.di_pipeline.index.bm25_index.ResumableBM25Builder",
        FakeBuilder,
    )
    monkeypatch.setattr(index, "load", lambda: None)

    index.build([_chunk()])

    assert saw_checkpoint == [True]
    pointer = json.loads((tmp_path / "CURRENT.json").read_text())
    assert (tmp_path / pointer["generation"] / "checkpoint").is_file()
