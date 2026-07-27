"""Read-only random access to Chunk records stored as JSON Lines."""

from __future__ import annotations

import errno
import json
import operator
import os
import stat
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload

import numpy as np

from littraceqa.di_pipeline.contracts import Chunk

_OFFSET_VALIDATION_BLOCK_SIZE = 1_000_000


class ChunkJsonlStore(Sequence[Chunk]):
    """Expose an offset-indexed JSONL file as a lazy sequence of chunks.

    ``offsets_path`` must contain a one-dimensional ``uint64`` NumPy array.
    Entry ``i`` is the byte offset of JSONL record ``i`` and the final entry is
    the JSONL byte size. The JSONL file is opened separately for every integer
    access, so concurrent readers never share a file position or a persistent
    descriptor.
    """

    def __init__(
        self,
        jsonl_path: str | os.PathLike[str],
        offsets_path: str | os.PathLike[str],
        expected_documents: int | None = None,
    ) -> None:
        if expected_documents is not None:
            if isinstance(expected_documents, bool) or not isinstance(
                expected_documents, int
            ):
                raise TypeError("expected_documents must be an integer or None")
            if expected_documents < 0:
                raise ValueError("expected_documents must not be negative")

        self.jsonl_path = Path(jsonl_path)
        self.offsets_path = Path(offsets_path)
        jsonl_size = self._jsonl_size()
        self._offsets = self._load_offsets()
        self._validate_offsets(jsonl_size, expected_documents)

    def __len__(self) -> int:
        return len(self._offsets) - 1

    @overload
    def __getitem__(self, index: int) -> Chunk: ...

    @overload
    def __getitem__(self, index: slice) -> list[Chunk]: ...

    def __getitem__(self, index: int | slice) -> Chunk | list[Chunk]:
        if isinstance(index, slice):
            return [
                self[position]
                for position in range(*index.indices(len(self)))
            ]

        try:
            position = operator.index(index)
        except TypeError:
            raise TypeError(
                "ChunkJsonlStore indices must be integers or slices"
            ) from None
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("ChunkJsonlStore index out of range")
        return self._read_chunk(position)

    def __iter__(self) -> Iterator[Chunk]:
        for position in range(len(self)):
            yield self._read_chunk(position)

    def _jsonl_size(self) -> int:
        try:
            file_stat = os.stat(self.jsonl_path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"cannot inspect Chunk JSONL file: {self.jsonl_path}"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode):
            raise ValueError(
                f"Chunk JSONL file must not be a symbolic link: "
                f"{self.jsonl_path}"
            )
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(
                f"Chunk JSONL path must be a regular file: {self.jsonl_path}"
            )
        return file_stat.st_size

    def _load_offsets(self) -> np.ndarray:
        if self.offsets_path.is_symlink():
            raise ValueError(
                f"Chunk offsets file must not be a symbolic link: "
                f"{self.offsets_path}"
            )
        try:
            offsets = np.load(
                self.offsets_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                f"cannot load Chunk offsets file: {self.offsets_path}"
            ) from exc
        if not isinstance(offsets, np.ndarray):
            raise ValueError(
                f"Chunk offsets must be a NumPy array: {self.offsets_path}"
            )
        return offsets

    def _validate_offsets(
        self,
        jsonl_size: int,
        expected_documents: int | None,
    ) -> None:
        offsets = self._offsets
        if offsets.dtype != np.dtype(np.uint64):
            raise ValueError(
                "Chunk offsets must have dtype uint64, "
                f"not {offsets.dtype}"
            )
        if offsets.ndim != 1:
            raise ValueError(
                f"Chunk offsets must be one-dimensional, not {offsets.ndim}D"
            )
        if len(offsets) == 0:
            raise ValueError(
                "Chunk offsets must contain at least the initial zero"
            )
        if int(offsets[0]) != 0:
            raise ValueError("Chunk offsets must start at zero")

        for start in range(0, len(offsets) - 1, _OFFSET_VALIDATION_BLOCK_SIZE):
            stop = min(
                start + _OFFSET_VALIDATION_BLOCK_SIZE,
                len(offsets) - 1,
            )
            if np.any(offsets[start + 1 : stop + 1] < offsets[start:stop]):
                raise ValueError(
                    "Chunk offsets must be monotonically non-decreasing"
                )

        final_offset = int(offsets[-1])
        if final_offset != jsonl_size:
            raise ValueError(
                "final Chunk offset must equal the JSONL byte size "
                f"({final_offset} != {jsonl_size})"
            )

        document_count = len(offsets) - 1
        if (
            expected_documents is not None
            and document_count != expected_documents
        ):
            raise ValueError(
                "Chunk offsets document count does not match "
                f"expected_documents ({document_count} != "
                f"{expected_documents})"
            )

    def _read_chunk(self, position: int) -> Chunk:
        start = int(self._offsets[position])
        stop = int(self._offsets[position + 1])
        byte_count = stop - start
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.jsonl_path, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                detail = "the path is a symbolic link"
            else:
                detail = str(exc)
            raise ValueError(
                f"cannot open Chunk JSONL record {position}: {detail}"
            ) from exc
        try:
            try:
                payload = os.pread(descriptor, byte_count, start)
            except OSError as exc:
                raise ValueError(
                    f"cannot read Chunk JSONL record {position}: {exc}"
                ) from exc
        finally:
            os.close(descriptor)

        if len(payload) != byte_count:
            raise ValueError(
                f"Chunk JSONL record {position} was a short read "
                f"({len(payload)} of {byte_count} bytes)"
            )
        if not payload.endswith(b"\n"):
            raise ValueError(
                f"Chunk JSONL record {position} does not end with LF"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Chunk JSONL record {position} is not valid UTF-8"
            ) from exc
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Chunk JSONL record {position} is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Chunk JSONL record {position} must be a JSON object"
            )
        try:
            chunk = Chunk(**value)
        except TypeError as exc:
            raise ValueError(
                f"Chunk JSONL record {position} does not satisfy "
                f"the Chunk contract: {exc}"
            ) from exc
        if (
            not isinstance(chunk.chunk_id, str)
            or not isinstance(chunk.paper_id, str)
            or not isinstance(chunk.text, str)
            or not isinstance(chunk.chunk_type, str)
            or not isinstance(chunk.metadata, dict)
        ):
            raise ValueError(
                f"Chunk JSONL record {position} does not satisfy "
                "the Chunk contract: invalid field type"
            )
        return chunk
