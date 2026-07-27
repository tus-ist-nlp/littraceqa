"""BM25 による疎検索インデックス。

chunks.jsonl 由来の Chunk 列から bm25s (https://github.com/xhluca/bm25s) の
BM25 インデックスを構築し、クエリに対して RetrievalResult を返す。
索引本体は bm25s.BM25.save/load で index_dir に永続化するが、bm25s は
Chunk のメタデータ（paper_id, chunk_type, metadata など）までは持たないため、
検索結果を Chunk に戻すための chunks.jsonl も同じディレクトリに保存する。
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import bm25s

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.index.chunk_store import ChunkJsonlStore
from littraceqa.di_pipeline.index.resumable_bm25 import ResumableBM25Builder
from littraceqa.di_pipeline.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_CHUNK_OFFSETS_FILENAME = "chunks.offsets.npy"
_CURRENT_FILENAME = "CURRENT.json"
_GENERATIONS_DIRNAME = "generations"
_STAGING_DIRNAME = ".resumable-staging"
_BUILD_LOCK_FILENAME = ".build.lock"


class BM25BuildLockError(RuntimeError):
    """Raised when another process already owns an index build lock."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@register("indexer", "bm25s")
class BM25Index:
    name = "bm25s"
    checkpoint_dependencies = (
        bm25s.BM25,
        bm25s.tokenize,
        ChunkJsonlStore,
        ResumableBM25Builder,
    )

    def __init__(
        self,
        index_dir: str,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        idf_method: str | None = None,
        records_filename: str = _CHUNKS_FILENAME,
        resumable_build: bool = False,
        build_batch_size: int = 1,
        max_batch_characters: int = 2_000_000,
    ):
        if not records_filename or Path(records_filename).name != records_filename:
            raise ValueError("records_filename must be a non-empty file name")
        if not isinstance(resumable_build, bool):
            raise TypeError("resumable_build must be a bool")
        if isinstance(build_batch_size, bool) or not isinstance(
            build_batch_size, int
        ):
            raise TypeError("build_batch_size must be an integer")
        if build_batch_size <= 0:
            raise ValueError("build_batch_size must be positive")
        if isinstance(max_batch_characters, bool) or not isinstance(
            max_batch_characters, int
        ):
            raise TypeError("max_batch_characters must be an integer")
        if max_batch_characters <= 0:
            raise ValueError("max_batch_characters must be positive")
        if resumable_build and records_filename != _CHUNKS_FILENAME:
            raise ValueError(
                "resumable_build requires records_filename='chunks.jsonl'"
            )
        self.index_dir = Path(index_dir)
        self.records_filename = records_filename
        self.resumable_build = resumable_build
        self.build_batch_size = build_batch_size
        self.max_batch_characters = max_batch_characters
        # Resolve bm25s' implicit IDF default so experiment metadata is explicit.
        self.build_params: dict[str, str | float] = {
            "method": method,
            "k1": k1,
            "b": b,
            "delta": delta,
            "idf_method": idf_method if idf_method is not None else method,
        }
        self._chunks: Sequence[Chunk] = []
        self._retriever: bm25s.BM25 | None = None
        self._active_index_dir = self.index_dir

    def build(self, chunks: Iterable[Chunk]) -> None:
        if self.resumable_build:
            self._build_resumable(
                chunks,
                staging_dir=self.index_dir / _STAGING_DIRNAME,
            )
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks = list(chunks)
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in self._chunks], stopwords="en"
        )
        retriever = bm25s.BM25(**self.build_params)
        retriever.index(corpus_tokens)
        retriever.save(str(self.index_dir))
        self._save_chunks()
        self._retriever = retriever

    def build_with_signature(
        self,
        chunks: Iterable[Chunk],
        build_signature: str,
    ) -> None:
        """Build with resumable state isolated by an external build signature."""
        self._validate_build_signature(build_signature)
        if not self.resumable_build:
            self.build(chunks)
            return
        self._build_resumable(
            chunks,
            staging_dir=(
                self.index_dir
                / f"{_STAGING_DIRNAME}-{build_signature}"
            ),
        )

    def load(self) -> None:
        if not self.index_dir.is_dir():
            raise FileNotFoundError(
                f"BM25 index directory is missing: {self.index_dir}"
            )
        active_index_dir = self._resolve_active_index_dir()
        self._retriever = bm25s.BM25.load(
            str(active_index_dir),
            load_corpus=False,
            mmap=True,
        )
        self._active_index_dir = active_index_dir
        # Loading uses parameters persisted by bm25s, never constructor overrides.
        self.build_params = {
            "method": self._retriever.method,
            "k1": self._retriever.k1,
            "b": self._retriever.b,
            "delta": self._retriever.delta,
            "idf_method": self._retriever.idf_method,
        }
        self._chunks = self._load_chunks()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._retriever is None:
            raise RuntimeError(
                "index is not built or loaded; call build() or load() first"
            )
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="en")
        doc_indices, scores = self._retriever.retrieve(query_tokens, k=k)
        results: list[RetrievalResult] = []
        for doc_index, score in zip(doc_indices[0], scores[0]):
            chunk = self._chunks[int(doc_index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(score),
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata,
                    source=self.name,
                )
            )
        return results

    def _save_chunks(self) -> None:
        path = self.index_dir / self.records_filename
        with path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _load_chunks(self) -> Sequence[Chunk]:
        path = self._active_index_dir / self.records_filename
        offsets_path = self._active_index_dir / _CHUNK_OFFSETS_FILENAME
        if (
            self.records_filename == _CHUNKS_FILENAME
            and (offsets_path.exists() or offsets_path.is_symlink())
        ):
            if self._retriever is None:
                raise RuntimeError("BM25 retriever must be loaded before Chunk records")
            num_documents = self._retriever.scores.get("num_docs")
            if (
                isinstance(num_documents, bool)
                or not isinstance(num_documents, int)
                or num_documents < 0
            ):
                raise ValueError("BM25 index has an invalid document count")
            return ChunkJsonlStore(
                path,
                offsets_path,
                expected_documents=num_documents,
            )

        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                chunks.append(Chunk(**record))
        return chunks

    @contextmanager
    def _exclusive_build_lock(self) -> Iterator[BinaryIO]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.index_dir / _BUILD_LOCK_FILENAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise BM25BuildLockError(
                    f"BM25 build lock must not be a symbolic link: {lock_path}"
                ) from exc
            raise
        lock_file = os.fdopen(descriptor, "a+b")
        try:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise BM25BuildLockError(
                    f"another BM25 build is already running for {self.index_dir}"
                ) from exc
            yield lock_file
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def _validate_build_directory(self, staging_dir: Path) -> None:
        if staging_dir.is_symlink():
            raise ValueError(
                "BM25 resumable staging directory must not be a symbolic link"
            )

    def _validate_build_signature(self, build_signature: str) -> None:
        if (
            not isinstance(build_signature, str)
            or len(build_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in build_signature
            )
        ):
            raise ValueError(
                "build_signature must be a lowercase SHA-256 hexadecimal digest"
            )

    def _build_resumable(
        self,
        chunks: Iterable[Chunk],
        *,
        staging_dir: Path,
    ) -> None:
        with self._exclusive_build_lock():
            generations_dir = self.index_dir / _GENERATIONS_DIRNAME
            self._validate_build_directory(staging_dir)
            generations_dir.mkdir(parents=True, exist_ok=True)
            if generations_dir.is_symlink():
                raise ValueError(
                    "BM25 generations directory must not be a symbolic link"
                )

            builder = ResumableBM25Builder(
                staging_dir,
                batch_size=self.build_batch_size,
                max_batch_characters=self.max_batch_characters,
                **self.build_params,
            )
            result = builder.build(chunks)
            input_sha256 = result.input_sha256
            if (
                not isinstance(input_sha256, str)
                or len(input_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in input_sha256
                )
            ):
                raise ValueError(
                    "resumable BM25 builder returned an invalid input checksum"
                )
            generation_name = f"{input_sha256}-{uuid4().hex}"
            generation_dir = generations_dir / generation_name
            os.rename(staging_dir, generation_dir)
            _fsync_directory(generations_dir)
            _atomic_write_json(
                self.index_dir / _CURRENT_FILENAME,
                {
                    "generation": (
                        f"{_GENERATIONS_DIRNAME}/{generation_name}"
                    ),
                    "input_sha256": input_sha256,
                },
            )
            # Load while holding the writer lock so this instance opens the
            # generation it just published, not a subsequent writer's one.
            self.load()

    def _resolve_active_index_dir(self) -> Path:
        current_path = self.index_dir / _CURRENT_FILENAME
        if current_path.is_symlink():
            raise ValueError(
                f"BM25 generation pointer must not be a symbolic link: {current_path}"
            )
        if not current_path.exists():
            return self.index_dir
        try:
            content = current_path.read_bytes()
            pointer = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid BM25 generation pointer: {current_path}"
            ) from exc
        if not isinstance(pointer, dict):
            raise ValueError(f"invalid BM25 generation pointer: {current_path}")
        relative = pointer.get("generation")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"invalid BM25 generation pointer: {current_path}")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 2
            or relative_path.parts[0] != _GENERATIONS_DIRNAME
            or relative_path.parts[1] in ("", ".", "..")
        ):
            raise ValueError(
                f"BM25 generation pointer escapes index root: {relative}"
            )
        try:
            root = self.index_dir.resolve(strict=True)
            generation = (self.index_dir / relative_path).resolve(strict=True)
            generation.relative_to(root)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"BM25 generation pointer escapes index root or is missing: {relative}"
            ) from exc
        if not generation.is_dir():
            raise ValueError(
                f"BM25 generation pointer is not a directory: {relative}"
            )
        return generation
