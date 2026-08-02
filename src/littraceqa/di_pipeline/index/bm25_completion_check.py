"""Prove a finished generation directory is a loadable, self-consistent index.

The build only reports success after re-reading every file it wrote, because a
truncated or mismatched array would otherwise surface much later as a wrong
search result rather than as a build failure.
"""

from __future__ import annotations

from typing import Any

import bm25s
import numpy as np

from littraceqa.di_pipeline.index.bm25_checkpoint_format import (
    CHUNK_OFFSETS_NAME,
    CHUNKS_NAME,
    CorruptCheckpointError,
    DATA_NAME,
    FILE_RECORD_KEYS,
    INDICES_NAME,
    INDPTR_NAME,
    NONOCCURRENCE_NAME,
    PARAMS_NAME,
    VALIDATION_BLOCK_ITEMS,
    VOCAB_NAME,
    is_sha256,
    sha256_file,
)
from littraceqa.di_pipeline.index.bm25_checkpoint_layout import (
    BM25Parameters,
    CheckpointLayout,
    GlobalStatistics,
)


def output_names(parameters: BM25Parameters) -> list[str]:
    names = [
        DATA_NAME,
        INDICES_NAME,
        INDPTR_NAME,
        VOCAB_NAME,
        PARAMS_NAME,
        CHUNKS_NAME,
        CHUNK_OFFSETS_NAME,
    ]
    if parameters.method in ("bm25l", "bm25+"):
        names.append(NONOCCURRENCE_NAME)
    return names


def validate_completed(
    layout: CheckpointLayout,
    parameters: BM25Parameters,
    manifest: dict[str, Any],
    statistics: GlobalStatistics,
    *,
    verify_checksums: bool = True,
) -> None:
    files = manifest["files"]
    expected_names = set(output_names(parameters))
    if not isinstance(files, dict) or set(files) != expected_names:
        raise CorruptCheckpointError(
            "completed manifest has an invalid final file set"
        )
    for filename, record in files.items():
        path = layout.generation_dir / filename
        if (
            not isinstance(record, dict)
            or set(record) != FILE_RECORD_KEYS
            or not is_sha256(record["sha256"])
            or isinstance(record["size"], bool)
            or not isinstance(record["size"], int)
            or record["size"] <= 0
        ):
            raise CorruptCheckpointError(
                f"completed file metadata is invalid: {path}"
            )
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["size"]
                or (
                    verify_checksums
                    and sha256_file(path) != record["sha256"]
                )
            ):
                raise CorruptCheckpointError(
                    f"completed generation file is corrupt: {path}"
                )
        except OSError as exc:
            raise CorruptCheckpointError(
                f"completed generation file is unreadable: {path}"
            ) from exc
    try:
        data = np.load(
            layout.generation_dir / DATA_NAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        indices = np.load(
            layout.generation_dir / INDICES_NAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        indptr = np.load(
            layout.generation_dir / INDPTR_NAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        chunk_offsets = np.load(
            layout.generation_dir / CHUNK_OFFSETS_NAME,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise CorruptCheckpointError(
            "completed index arrays are unreadable"
        ) from exc
    if (
        data.dtype != parameters.dtype
        or indices.dtype != parameters.int_dtype
        or indptr.dtype != np.dtype("int64")
        or data.shape != (statistics.nonzero_scores,)
        or indices.shape != (statistics.nonzero_scores,)
        or indptr.shape != (len(statistics.vocab) + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != statistics.nonzero_scores
    ):
        raise CorruptCheckpointError("completed CSC arrays are invalid")
    for start in range(0, data.size, VALIDATION_BLOCK_ITEMS):
        end = min(start + VALIDATION_BLOCK_ITEMS, data.size)
        if (
            not np.all(np.isfinite(data[start:end]))
            or np.any(indices[start:end] < 0)
            or np.any(indices[start:end] >= statistics.num_documents)
        ):
            raise CorruptCheckpointError(
                "completed CSC arrays are invalid"
            )
    previous = int(indptr[0])
    for start in range(1, indptr.size, VALIDATION_BLOCK_ITEMS):
        end = min(start + VALIDATION_BLOCK_ITEMS, indptr.size)
        block = indptr[start:end]
        if (
            block.size
            and (
                int(block[0]) < previous
                or np.any(block[1:] < block[:-1])
            )
        ):
            raise CorruptCheckpointError(
                "completed CSC arrays are invalid"
            )
        if block.size:
            previous = int(block[-1])
    chunks_path = layout.generation_dir / CHUNKS_NAME
    chunks_size = chunks_path.stat().st_size
    if (
        chunk_offsets.dtype != np.dtype("uint64")
        or chunk_offsets.ndim != 1
        or chunk_offsets.shape != (statistics.num_documents + 1,)
        or int(chunk_offsets[0]) != 0
        or int(chunk_offsets[-1]) != chunks_size
    ):
        raise CorruptCheckpointError(
            "completed chunk metadata offsets are invalid"
        )
    previous = int(chunk_offsets[0])
    for start in range(1, chunk_offsets.size, VALIDATION_BLOCK_ITEMS):
        end = min(
            start + VALIDATION_BLOCK_ITEMS, chunk_offsets.size
        )
        block = chunk_offsets[start:end]
        if (
            block.size
            and (
                int(block[0]) <= previous
                or np.any(block[1:] <= block[:-1])
            )
        ):
            raise CorruptCheckpointError(
                "completed chunk metadata offsets are invalid"
            )
        if block.size:
            previous = int(block[-1])
    boundary_index = 1
    absolute_position = 0
    try:
        with chunks_path.open("rb") as chunks_file:
            while content := chunks_file.read(1024 * 1024):
                search_start = 0
                while True:
                    newline_index = content.find(b"\n", search_start)
                    if newline_index < 0:
                        break
                    boundary = absolute_position + newline_index + 1
                    if (
                        boundary_index >= chunk_offsets.size
                        or int(chunk_offsets[boundary_index]) != boundary
                    ):
                        raise CorruptCheckpointError(
                            "completed chunk metadata offsets are invalid"
                        )
                    boundary_index += 1
                    search_start = newline_index + 1
                absolute_position += len(content)
    except OSError as exc:
        raise CorruptCheckpointError(
            "completed chunk metadata is unreadable"
        ) from exc
    if (
        boundary_index != statistics.num_documents + 1
        or absolute_position != chunks_size
    ):
        raise CorruptCheckpointError(
            "completed chunk metadata count is invalid"
        )
    try:
        bm25s.BM25.load(
            str(layout.generation_dir),
            load_corpus=False,
            mmap=True,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorruptCheckpointError(
            "completed generation cannot be loaded by bm25s"
        ) from exc
