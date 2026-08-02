"""On-disk format of a resumable bm25s build, and the primitives that write it.

Every checkpoint file is written atomically and re-read through a size bound,
so a build interrupted mid-write is detected as corrupt rather than resumed
from a truncated file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from littraceqa.di_pipeline.contracts import Chunk


EXPECTED_BM25S_VERSION = "0.3.9"
SCHEMA_VERSION = 2
MANIFEST_NAME = "resumable-build.json"
PARTS_DIR_NAME = ".resumable-bm25-parts"
SCORES_DIR_NAME = ".resumable-bm25-scores"
DATA_NAME = "data.csc.index.npy"
INDICES_NAME = "indices.csc.index.npy"
INDPTR_NAME = "indptr.csc.index.npy"
VOCAB_NAME = "vocab.index.json"
PARAMS_NAME = "params.index.json"
NONOCCURRENCE_NAME = "nonoccurrence_array.index.npy"
CHUNKS_NAME = "chunks.jsonl"
CHUNK_OFFSETS_NAME = "chunks.offsets.npy"
METHODS = frozenset({"lucene", "robertson", "atire", "bm25l", "bm25+"})
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "configuration",
        "phase",
        "batch_count",
        "input_sha256",
        "global_statistics",
        "files",
    }
)
PART_META_KEYS = frozenset(
    {
        "schema_version",
        "index",
        "documents",
        "input_sha256",
        "part_sha256",
        "part_size",
    }
)
PART_PAYLOAD_KEYS = frozenset(
    {"schema_version", "index", "chunks", "tokens"}
)
SCORE_META_KEYS = frozenset(
    {
        "schema_version",
        "index",
        "documents",
        "document_start",
        "entries",
        "part_sha256",
        "statistics_signature",
        "score_sha256",
        "score_size",
    }
)
STATISTICS_KEYS = frozenset(
    {
        "signature",
        "num_documents",
        "vocabulary_size",
        "nonzero_scores",
        "total_document_length",
        "average_document_length",
        "vocabulary_sha256",
        "document_frequencies_sha256",
    }
)
FILE_RECORD_KEYS = frozenset({"sha256", "size"})
CHUNK_KEYS = frozenset(
    {"chunk_id", "paper_id", "text", "chunk_type", "metadata"}
)
HEX_DIGITS = frozenset("0123456789abcdef")
METADATA_MAX_BYTES = 64 * 1024
SCORE_CONTAINER_OVERHEAD = 64 * 1024
MIN_FREE_SPACE_BYTES = 512 * 1024 * 1024
VALIDATION_BLOCK_ITEMS = 1_000_000


class ResumableBM25Error(RuntimeError):
    """Base class for resumable-index construction errors."""


class InputChangedError(ResumableBM25Error):
    """Raised when a generation is resumed with different ordered input."""


class ConfigurationChangedError(ResumableBM25Error):
    """Raised when a generation is resumed with different build parameters."""


class CorruptCheckpointError(ResumableBM25Error):
    """Raised when checkpoint metadata itself cannot be trusted."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_int64_array(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype=np.dtype("<i8"))
    digest = hashlib.sha256()
    digest.update(memoryview(normalized).cast("B"))
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json(value) + b"\n")


def atomic_vocab(path: Path, vocab: dict[str, int]) -> None:
    """Write a large vocabulary without copying it into another dict or bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("{")
            first = True
            for token, token_id in vocab.items():
                if not first:
                    output.write(",")
                json.dump(token, output, ensure_ascii=False, allow_nan=False)
                output.write(":")
                output.write(str(token_id))
                first = False
            if "" not in vocab:
                if not first:
                    output.write(",")
                output.write('"":')
                output.write(str(len(vocab)))
            output.write("}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_bounded(path: Path, expected_size: int, maximum_size: int) -> bytes:
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or expected_size > maximum_size
    ):
        raise CorruptCheckpointError(f"invalid stored size for {path}")
    try:
        if path.is_symlink():
            raise CorruptCheckpointError(
                f"checkpoint must not be a symbolic link: {path}"
            )
        actual_size = path.stat().st_size
    except OSError as exc:
        raise CorruptCheckpointError(f"checkpoint is unreadable: {path}") from exc
    if actual_size != expected_size or actual_size > maximum_size:
        raise CorruptCheckpointError(f"checkpoint size mismatch: {path}")
    try:
        with path.open("rb") as file:
            content = file.read(expected_size + 1)
    except OSError as exc:
        raise CorruptCheckpointError(f"checkpoint is unreadable: {path}") from exc
    if len(content) != expected_size:
        raise CorruptCheckpointError(f"checkpoint changed while reading: {path}")
    return content


def read_small_json(path: Path) -> Any:
    try:
        size = path.stat().st_size
        content = read_bounded(path, size, METADATA_MAX_BYTES)
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise CorruptCheckpointError(f"checkpoint is not valid JSON: {path}") from exc
    except OSError as exc:
        raise CorruptCheckpointError(f"checkpoint is unreadable: {path}") from exc


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def batched(
    chunks: Iterable[Chunk],
    batch_size: int,
    max_batch_characters: int,
) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    characters = 0
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise TypeError("chunks must contain Chunk instances")
        chunk_characters = len(chunk.text)
        if chunk_characters > max_batch_characters:
            raise ValueError(
                "one chunk exceeds max_batch_characters: "
                f"{chunk.chunk_id} has {chunk_characters} characters"
            )
        if batch and (
            len(batch) >= batch_size
            or characters + chunk_characters > max_batch_characters
        ):
            yield batch
            batch = []
            characters = 0
        batch.append(chunk)
        characters += chunk_characters
        if len(batch) >= batch_size:
            yield batch
            batch = []
            characters = 0
    if batch:
        yield batch
