"""Memory-bounded, resumable construction of a bm25s 0.3.9 index.

Tokenization and scoring are checkpointed independently for each small input
batch.  Score checkpoints use corpus-wide statistics, so merging them produces
the same single global BM25 index as ``bm25s.BM25.index``; they are not
independent shard-local indexes with different IDF values.

This module constructs one generation directory.  Selecting and publishing a
generation remains the caller's responsibility.
"""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
from bm25s.scoring import (
    _build_idf_array,
    _build_nonoccurrence_array,
    _select_idf_scorer,
    _select_tfc_scorer,
)

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index import (
    bm25_checkpoint_format,
    bm25_checkpoint_layout,
    bm25_completion_check,
)
from littraceqa.di_pipeline.index.bm25_completion_check import (
    output_names,
    validate_completed,
)
from littraceqa.di_pipeline.index.bm25_checkpoint_layout import (
    BM25Parameters,
    CheckpointLayout,
    GlobalStatistics,
)
from littraceqa.di_pipeline.index.bm25_checkpoint_format import (
    CHUNK_KEYS,
    CHUNK_OFFSETS_NAME,
    CHUNKS_NAME,
    ConfigurationChangedError,
    CorruptCheckpointError,
    DATA_NAME,
    EXPECTED_BM25S_VERSION,
    INDICES_NAME,
    INDPTR_NAME,
    InputChangedError,
    METHODS,
    MIN_FREE_SPACE_BYTES,
    NONOCCURRENCE_NAME,
    PARAMS_NAME,
    PART_META_KEYS,
    PART_PAYLOAD_KEYS,
    ResumableBM25Error,
    ROOT_KEYS,
    SCHEMA_VERSION,
    SCORE_CONTAINER_OVERHEAD,
    SCORE_META_KEYS,
    STATISTICS_KEYS,
    VOCAB_NAME,
    atomic_json,
    atomic_vocab,
    atomic_write,
    batched,
    canonical_json,
    finite_number,
    is_sha256,
    positive_integer,
    read_bounded,
    read_small_json,
    sha256_bytes,
    sha256_file,
    sha256_int64_array,
    sha256_ordered_strings,
)


__all__ = [
    "BuildResult",
    "ConfigurationChangedError",
    "CorruptCheckpointError",
    "InputChangedError",
    "ResumableBM25Builder",
    "ResumableBM25Error",
]


@dataclass(frozen=True)
class BuildResult:
    """Summary of a completed, bm25s-loadable generation."""

    generation_dir: Path
    input_sha256: str
    num_documents: int
    vocabulary_size: int
    nonzero_scores: int
    average_document_length: float
    tokenized_batches: int
    reused_batches: int
    rebuilt_batches: int
    scored_batches: int
    reused_score_batches: int
    rebuilt_score_batches: int


def _implementation_signature() -> str:
    """Fingerprint code and runtime components that affect persisted scores."""

    # Every module that holds build logic must be covered: a change in an
    # extracted stage has to invalidate a resumable checkpoint just as a change
    # in the builder itself does.
    source_objects = (
        ("builder", ResumableBM25Builder),
        ("checkpoint_format", bm25_checkpoint_format),
        ("checkpoint_layout", bm25_checkpoint_layout),
        ("completion_check", bm25_completion_check),
        ("bm25", bm25s.BM25),
        ("tokenize", bm25s.tokenize),
        ("idf_scorer", _select_idf_scorer),
        ("tfc_scorer", _select_tfc_scorer),
    )
    records: list[dict[str, str]] = []
    for label, source_object in source_objects:
        source_path = Path(inspect.getfile(source_object)).resolve()
        records.append(
            {
                "label": label,
                "sha256": sha256_file(source_path),
            }
        )
    return sha256_bytes(
        canonical_json(
            {
                "numpy_version": np.__version__,
                "sources": records,
            }
        )
    )


class ResumableBM25Builder:
    """Build one immutable bm25s generation from an ordered Chunk iterable."""

    def __init__(
        self,
        generation_dir: str | Path,
        *,
        batch_size: int = 1,
        max_batch_characters: int = 2_000_000,
        max_part_bytes: int = 64 * 1024 * 1024,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        idf_method: str | None = None,
        dtype: str = "float32",
        int_dtype: str = "int32",
        stopwords: str | Sequence[str] = "en",
        lower: bool = True,
        token_pattern: str = r"(?u)\b\w\w+\b",
    ) -> None:
        self.batch_size = positive_integer(batch_size, "batch_size")
        self.max_batch_characters = positive_integer(
            max_batch_characters, "max_batch_characters"
        )
        self.max_part_bytes = positive_integer(
            max_part_bytes, "max_part_bytes"
        )
        if bm25s.__version__ != EXPECTED_BM25S_VERSION:
            raise RuntimeError(
                "resumable construction requires bm25s "
                f"{EXPECTED_BM25S_VERSION}, found {bm25s.__version__}"
            )
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        resolved_idf = method if idf_method is None else idf_method
        if not isinstance(resolved_idf, str) or resolved_idf not in METHODS:
            raise ValueError(f"idf_method must be one of {sorted(METHODS)}")
        k1_value = finite_number(k1, "k1")
        b_value = finite_number(b, "b")
        delta_value = finite_number(delta, "delta")
        if k1_value < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b_value <= 1:
            raise ValueError("b must be between 0 and 1")
        if delta_value < 0:
            raise ValueError("delta must be non-negative")
        try:
            score_dtype = np.dtype(dtype)
            row_dtype = np.dtype(int_dtype)
        except TypeError as exc:
            raise ValueError("dtype and int_dtype must be valid NumPy dtypes") from exc
        if score_dtype.kind != "f":
            raise ValueError("dtype must be a floating-point dtype")
        if row_dtype.kind not in "iu":
            raise ValueError("int_dtype must be an integer dtype")
        if not isinstance(lower, bool):
            raise TypeError("lower must be a bool")
        if not isinstance(token_pattern, str) or not token_pattern:
            raise ValueError("token_pattern must be a non-empty string")
        try:
            re.compile(token_pattern)
        except re.error as exc:
            raise ValueError(
                "token_pattern must be a valid regular expression"
            ) from exc
        if isinstance(stopwords, str):
            normalized_stopwords: str | list[str] = stopwords
        elif isinstance(stopwords, Sequence) and all(
            isinstance(word, str) for word in stopwords
        ):
            normalized_stopwords = list(stopwords)
        else:
            raise TypeError("stopwords must be a string or a sequence of strings")

        # Keep bm25s' own canonical scalar/string representation.
        reference = bm25s.BM25(
            method=method,
            k1=k1_value,
            b=b_value,
            delta=delta_value,
            idf_method=resolved_idf,
            dtype=score_dtype.name,
            int_dtype=row_dtype.name,
            backend="numpy",
            csc_backend="numpy",
            auto_compile=False,
        )
        self.generation_dir = Path(generation_dir)
        self._layout = CheckpointLayout(self.generation_dir)
        self.method = reference.method
        self.k1 = reference.k1
        self.b = reference.b
        self.delta = reference.delta
        self.idf_method = reference.idf_method
        self.dtype = np.dtype(reference.dtype)
        self.int_dtype = np.dtype(reference.int_dtype)
        self.stopwords = normalized_stopwords
        self.lower = lower
        self.token_pattern = token_pattern
        self.implementation_signature = _implementation_signature()
        # Built last: bm25s normalises the scalars above, so the grouped
        # parameters must be taken from the normalised values.
        self._parameters = BM25Parameters(
            method=self.method,
            idf_method=self.idf_method,
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
        )

    def _configuration(self) -> dict[str, Any]:
        return {
            "bm25s_version": EXPECTED_BM25S_VERSION,
            "implementation_signature": self.implementation_signature,
            "batch_size": self.batch_size,
            "max_batch_characters": self.max_batch_characters,
            "max_part_bytes": self.max_part_bytes,
            "method": self.method,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "idf_method": self.idf_method,
            "dtype": self.dtype.name,
            "int_dtype": self.int_dtype.name,
            "stopwords": self.stopwords,
            "lower": self.lower,
            "token_pattern": self.token_pattern,
        }

    def _new_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "configuration": self._configuration(),
            "phase": "tokenizing",
            "batch_count": None,
            "input_sha256": None,
            "global_statistics": None,
            "files": None,
        }

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if not self._layout.manifest_path.exists():
            if self.generation_dir.exists() and any(self.generation_dir.iterdir()):
                raise CorruptCheckpointError(
                    "generation directory is non-empty but has no build manifest"
                )
            self.generation_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._new_manifest()
            atomic_json(self._layout.manifest_path, manifest)
            return manifest
        manifest = read_small_json(self._layout.manifest_path)
        if not isinstance(manifest, dict) or set(manifest) != ROOT_KEYS:
            raise CorruptCheckpointError("build manifest has an invalid schema")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise CorruptCheckpointError("unsupported build manifest schema")
        if manifest["configuration"] != self._configuration():
            raise ConfigurationChangedError(
                "build parameters differ from the existing generation"
            )
        phase = manifest["phase"]
        if phase not in {"tokenizing", "tokenized", "scoring", "complete"}:
            raise CorruptCheckpointError("build manifest has an invalid phase")
        batch_count = manifest["batch_count"]
        input_sha256 = manifest["input_sha256"]
        statistics = manifest["global_statistics"]
        files = manifest["files"]
        if phase == "tokenizing":
            if any(
                value is not None
                for value in (batch_count, input_sha256, statistics, files)
            ):
                raise CorruptCheckpointError(
                    "tokenizing manifest contains finalized fields"
                )
        else:
            if (
                isinstance(batch_count, bool)
                or not isinstance(batch_count, int)
                or batch_count <= 0
                or not is_sha256(input_sha256)
            ):
                raise CorruptCheckpointError(
                    "build manifest has invalid input metadata"
                )
            if phase == "tokenized":
                if statistics is not None or files is not None:
                    raise CorruptCheckpointError(
                        "tokenized manifest contains later-phase fields"
                    )
            elif not self._valid_statistics_record(statistics):
                raise CorruptCheckpointError(
                    "build manifest has invalid global statistics"
                )
            if phase == "scoring" and files is not None:
                raise CorruptCheckpointError(
                    "scoring manifest contains final file metadata"
                )
            if phase == "complete" and not isinstance(files, dict):
                raise CorruptCheckpointError(
                    "completed manifest has no file metadata"
                )
        return manifest

    @staticmethod
    def _valid_statistics_record(value: object) -> bool:
        if not isinstance(value, dict) or set(value) != STATISTICS_KEYS:
            return False
        integer_fields = (
            "num_documents",
            "vocabulary_size",
            "nonzero_scores",
            "total_document_length",
        )
        if any(
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] < 0
            for field in integer_fields
        ):
            return False
        return (
            value["num_documents"] > 0
            and value["vocabulary_size"] > 0
            and is_sha256(value["signature"])
            and is_sha256(value["vocabulary_sha256"])
            and is_sha256(value["document_frequencies_sha256"])
            and not isinstance(value["average_document_length"], bool)
            and isinstance(value["average_document_length"], (int, float))
            and math.isfinite(float(value["average_document_length"]))
            and value["average_document_length"] >= 0
        )

    def _tokenize(self, chunks: list[Chunk]) -> list[list[str]]:
        tokenized = bm25s.tokenize(
            [chunk.text for chunk in chunks],
            lower=self.lower,
            token_pattern=self.token_pattern,
            stopwords=self.stopwords,
            return_ids=False,
            show_progress=False,
        )
        if (
            not isinstance(tokenized, list)
            or len(tokenized) != len(chunks)
            or any(
                not isinstance(document, list)
                or any(not isinstance(token, str) or not token for token in document)
                for document in tokenized
            )
        ):
            raise RuntimeError("bm25s.tokenize returned an unexpected value")
        return tokenized

    @staticmethod
    def _batch_content(chunks: list[Chunk] | list[dict[str, Any]]) -> bytes:
        serialized = [
            canonical_json(
                chunk.to_dict() if isinstance(chunk, Chunk) else chunk
            )
            for chunk in chunks
        ]
        return b"".join(
            len(item).to_bytes(8, "big") + item for item in serialized
        )

    def _part_payload(
        self,
        index: int,
        chunks: list[Chunk],
        tokens: list[list[str]],
    ) -> bytes:
        return (
            canonical_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "index": index,
                    "chunks": [chunk.to_dict() for chunk in chunks],
                    "tokens": tokens,
                }
            )
            + b"\n"
        )

    @staticmethod
    def _validate_chunk_record(chunk: object, path: Path) -> None:
        if not isinstance(chunk, dict) or set(chunk) != CHUNK_KEYS:
            raise CorruptCheckpointError(f"token part has invalid chunk: {path}")
        if any(
            not isinstance(chunk[field], str)
            for field in ("chunk_id", "paper_id", "text", "chunk_type")
        ) or not isinstance(chunk["metadata"], dict):
            raise CorruptCheckpointError(f"token part has invalid chunk: {path}")

    def _read_part_meta(self, index: int) -> dict[str, Any]:
        path = self._layout.part_meta_path(index)
        meta = read_small_json(path)
        if not isinstance(meta, dict) or set(meta) != PART_META_KEYS:
            raise CorruptCheckpointError(f"token metadata is malformed: {path}")
        if (
            meta["schema_version"] != SCHEMA_VERSION
            or meta["index"] != index
            or isinstance(meta["documents"], bool)
            or not isinstance(meta["documents"], int)
            or meta["documents"] <= 0
            or not is_sha256(meta["input_sha256"])
            or not is_sha256(meta["part_sha256"])
            or isinstance(meta["part_size"], bool)
            or not isinstance(meta["part_size"], int)
            or not 0 < meta["part_size"] <= self.max_part_bytes
        ):
            raise CorruptCheckpointError(f"token metadata is malformed: {path}")
        return meta

    def _read_part(
        self,
        index: int,
        meta: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[list[str]]]:
        path = self._layout.part_path(index)
        content = read_bounded(
            path, meta["part_size"], self.max_part_bytes
        )
        if sha256_bytes(content) != meta["part_sha256"]:
            raise CorruptCheckpointError(f"token part checksum mismatch: {path}")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise CorruptCheckpointError(f"token part is malformed: {path}") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != PART_PAYLOAD_KEYS
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["index"] != index
        ):
            raise CorruptCheckpointError(f"token part is malformed: {path}")
        chunks = payload["chunks"]
        tokens = payload["tokens"]
        if (
            not isinstance(chunks, list)
            or not isinstance(tokens, list)
            or len(chunks) != meta["documents"]
            or len(tokens) != meta["documents"]
        ):
            raise CorruptCheckpointError(f"token part length mismatch: {path}")
        for chunk in chunks:
            self._validate_chunk_record(chunk, path)
        if any(
            not isinstance(document, list)
            or any(not isinstance(token, str) or not token for token in document)
            for document in tokens
        ):
            raise CorruptCheckpointError(f"token part has invalid tokens: {path}")
        if sha256_bytes(self._batch_content(chunks)) != meta["input_sha256"]:
            raise CorruptCheckpointError(
                f"token part input checksum mismatch: {path}"
            )
        return chunks, tokens

    def _write_part(
        self,
        index: int,
        chunks: list[Chunk],
        input_sha256: str,
    ) -> dict[str, Any]:
        tokens = self._tokenize(chunks)
        content = self._part_payload(index, chunks, tokens)
        if len(content) > self.max_part_bytes:
            raise ValueError(
                f"token checkpoint batch {index} exceeds max_part_bytes"
            )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "documents": len(chunks),
            "input_sha256": input_sha256,
            "part_sha256": sha256_bytes(content),
            "part_size": len(content),
        }
        atomic_write(self._layout.part_path(index), content)
        atomic_json(self._layout.part_meta_path(index), meta)
        return meta

    @staticmethod
    def _indexed_metadata_count(directory: Path, suffix: str) -> int:
        if not directory.exists():
            return 0
        if directory.is_symlink() or not directory.is_dir():
            raise CorruptCheckpointError(
                f"checkpoint path is not a normal directory: {directory}"
            )
        indices: list[int] = []
        for path in directory.glob(f"*{suffix}"):
            prefix = path.name[: -len(suffix)]
            if len(prefix) != 8 or not prefix.isdigit():
                raise CorruptCheckpointError(
                    f"checkpoint metadata has an invalid name: {path}"
                )
            indices.append(int(prefix))
        if sorted(indices) != list(range(len(indices))):
            raise CorruptCheckpointError(
                f"checkpoint metadata is not contiguous: {directory}"
            )
        return len(indices)

    def _checkpoint_tokens(
        self,
        chunks: Iterable[Chunk],
        manifest: dict[str, Any],
    ) -> tuple[str, int, int, int]:
        corpus_digest = hashlib.sha256()
        reused = 0
        rebuilt = 0
        seen_batches = 0
        existing_count = self._indexed_metadata_count(
            self._layout.parts_dir, ".meta.json"
        )
        expected_count = manifest["batch_count"]
        if expected_count is not None and existing_count != expected_count:
            raise CorruptCheckpointError(
                "token metadata count differs from the build manifest"
            )

        for index, batch in enumerate(
            batched(chunks, self.batch_size, self.max_batch_characters)
        ):
            seen_batches += 1
            batch_content = self._batch_content(batch)
            batch_sha256 = sha256_bytes(batch_content)
            corpus_digest.update(len(batch_content).to_bytes(8, "big"))
            corpus_digest.update(batch_content)
            if index < existing_count:
                meta = self._read_part_meta(index)
                if (
                    meta["input_sha256"] != batch_sha256
                    or meta["documents"] != len(batch)
                ):
                    raise InputChangedError(
                        f"input differs at batch {index}; "
                        "use a new generation directory"
                    )
                try:
                    self._read_part(index, meta)
                except CorruptCheckpointError:
                    self._write_part(index, batch, batch_sha256)
                    rebuilt += 1
                else:
                    reused += 1
                continue
            if manifest["phase"] != "tokenizing":
                raise InputChangedError(
                    "input has additional batches; use a new generation directory"
                )
            self._write_part(index, batch, batch_sha256)
            rebuilt += 1

        if seen_batches == 0:
            raise ValueError("cannot build a BM25 index from an empty corpus")
        if seen_batches < existing_count:
            raise InputChangedError(
                "input has fewer batches; use a new generation directory"
            )
        input_sha256 = corpus_digest.hexdigest()
        if manifest["input_sha256"] is not None:
            if manifest["input_sha256"] != input_sha256:
                raise InputChangedError(
                    "input checksum changed; use a new generation directory"
                )
        else:
            manifest.update(
                {
                    "phase": "tokenized",
                    "batch_count": seen_batches,
                    "input_sha256": input_sha256,
                }
            )
            atomic_json(self._layout.manifest_path, manifest)
        return input_sha256, seen_batches, reused, rebuilt

    def _iter_parts(
        self, manifest: dict[str, Any]
    ) -> Iterator[
        tuple[int, dict[str, Any], list[dict[str, Any]], list[list[str]]]
    ]:
        for index in range(manifest["batch_count"]):
            meta = self._read_part_meta(index)
            chunks, tokens = self._read_part(index, meta)
            yield index, meta, chunks, tokens

    def _global_statistics(
        self, manifest: dict[str, Any]
    ) -> GlobalStatistics:
        vocab: dict[str, int] = {}
        document_frequencies: list[int] = []
        num_documents = 0
        total_document_length = 0
        for _, _, chunks, documents in self._iter_parts(manifest):
            num_documents += len(chunks)
            for tokens in documents:
                total_document_length += len(tokens)
                seen: set[int] = set()
                for token in tokens:
                    token_id = vocab.get(token)
                    if token_id is None:
                        token_id = len(vocab)
                        vocab[token] = token_id
                        document_frequencies.append(0)
                    seen.add(token_id)
                for token_id in seen:
                    document_frequencies[token_id] += 1
        if not vocab:
            raise ValueError(
                "cannot build a BM25 index because the corpus has no "
                "searchable tokens after tokenization"
            )
        if num_documents > np.iinfo(self.int_dtype).max:
            raise OverflowError("document count does not fit int_dtype")
        if len(vocab) - 1 > np.iinfo(self.int_dtype).max:
            raise OverflowError("vocabulary IDs do not fit int_dtype")
        frequencies = np.asarray(document_frequencies, dtype=np.int64)
        average = np.float64(total_document_length) / np.float64(num_documents)
        nonzero_scores = int(frequencies.sum(dtype=np.int64))
        vocabulary_sha256 = sha256_ordered_strings(vocab)
        frequency_sha256 = sha256_int64_array(frequencies)
        signature_payload = {
            "input_sha256": manifest["input_sha256"],
            "implementation_signature": self.implementation_signature,
            "method": self.method,
            "idf_method": self.idf_method,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "dtype": self.dtype.name,
            "int_dtype": self.int_dtype.name,
            "num_documents": num_documents,
            "total_document_length": total_document_length,
            "vocabulary_sha256": vocabulary_sha256,
            "document_frequencies_sha256": frequency_sha256,
        }
        signature = sha256_bytes(canonical_json(signature_payload))
        record = {
            "signature": signature,
            "num_documents": num_documents,
            "vocabulary_size": len(vocab),
            "nonzero_scores": nonzero_scores,
            "total_document_length": total_document_length,
            "average_document_length": float(average),
            "vocabulary_sha256": vocabulary_sha256,
            "document_frequencies_sha256": frequency_sha256,
        }
        return GlobalStatistics(
            vocab=vocab,
            document_frequencies=frequencies,
            num_documents=num_documents,
            total_document_length=total_document_length,
            average_document_length=average,
            nonzero_scores=nonzero_scores,
            signature=signature,
            record=record,
        )

    def _prepare_statistics(
        self, manifest: dict[str, Any], statistics: GlobalStatistics
    ) -> None:
        if manifest["global_statistics"] is None:
            if manifest["phase"] != "tokenized":
                raise CorruptCheckpointError(
                    "global statistics are missing in a later build phase"
                )
            manifest.update(
                {
                    "phase": "scoring",
                    "global_statistics": statistics.record,
                    "files": None,
                }
            )
            atomic_json(self._layout.manifest_path, manifest)
        elif manifest["global_statistics"] != statistics.record:
            raise CorruptCheckpointError(
                "global statistics do not match token checkpoints"
            )

    def _validate_disk_capacity(
        self,
        manifest: dict[str, Any],
        statistics: GlobalStatistics,
    ) -> None:
        """Refuse a build that cannot retain checkpoints and final arrays."""

        score_entry_bytes = (
            self.dtype.itemsize + 2 * self.int_dtype.itemsize
        )
        estimated_score_bytes = (
            statistics.nonzero_scores * score_entry_bytes
            + manifest["batch_count"] * SCORE_CONTAINER_OVERHEAD
        )
        existing_score_bytes = 0
        if self._layout.scores_dir.is_dir() and not self._layout.scores_dir.is_symlink():
            for path in self._layout.scores_dir.glob("*.npz"):
                try:
                    if path.is_file() and not path.is_symlink():
                        existing_score_bytes += path.stat().st_size
                except OSError:
                    continue
        remaining_score_bytes = max(
            0,
            estimated_score_bytes - existing_score_bytes,
        )
        final_array_bytes = (
            statistics.nonzero_scores
            * (self.dtype.itemsize + self.int_dtype.itemsize)
            + (len(statistics.vocab) + 1) * np.dtype("int64").itemsize
            + (statistics.num_documents + 1) * np.dtype("uint64").itemsize
        )
        if self.method in ("bm25l", "bm25+"):
            final_array_bytes += len(statistics.vocab) * self.dtype.itemsize
        chunks_upper_bound = sum(
            self._read_part_meta(index)["part_size"]
            for index in range(manifest["batch_count"])
        )
        required = (
            remaining_score_bytes
            + final_array_bytes
            + chunks_upper_bound
            + MIN_FREE_SPACE_BYTES
        )
        free = shutil.disk_usage(self.generation_dir).free
        if free < required:
            raise OSError(
                "insufficient free space for resumable BM25 build: "
                f"requires approximately {required} bytes including reserve, "
                f"but only {free} bytes are free"
            )

    def _scoring_context(
        self, statistics: GlobalStatistics
    ) -> tuple[np.ndarray, np.ndarray | None]:
        frequency_map = {
            token_id: int(frequency)
            for token_id, frequency in enumerate(
                statistics.document_frequencies
            )
        }
        idf = _build_idf_array(
            frequency_map,
            n_docs=statistics.num_documents,
            compute_idf_fn=_select_idf_scorer(self.idf_method),
            dtype=self.dtype.name,
        )
        if self.method in ("bm25l", "bm25+"):
            nonoccurrence = _build_nonoccurrence_array(
                frequency_map,
                n_docs=statistics.num_documents,
                compute_idf_fn=_select_idf_scorer(self.idf_method),
                calculate_tfc_fn=_select_tfc_scorer(self.method),
                l_d=statistics.average_document_length,
                l_avg=statistics.average_document_length,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
                dtype=self.dtype.name,
            )
        else:
            nonoccurrence = None
        return idf, nonoccurrence

    def _score_arrays(
        self,
        documents: list[list[str]],
        *,
        document_start: int,
        statistics: GlobalStatistics,
        idf: np.ndarray,
        nonoccurrence: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        scorer = _select_tfc_scorer(self.method)
        for offset, tokens in enumerate(documents):
            token_ids = [statistics.vocab[token] for token in tokens]
            counts = Counter(token_ids)
            columns = np.asarray(list(counts), dtype=self.int_dtype)
            term_frequencies = np.asarray(
                list(counts.values()), dtype=self.dtype
            )
            term_component = scorer(
                tf_array=term_frequencies,
                l_d=len(token_ids),
                l_avg=statistics.average_document_length,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
            scores = idf[columns] * term_component
            if nonoccurrence is not None:
                scores -= nonoccurrence[columns]
            data_parts.append(np.asarray(scores, dtype=self.dtype))
            row_parts.append(
                np.full(
                    columns.shape,
                    document_start + offset,
                    dtype=self.int_dtype,
                )
            )
            column_parts.append(columns)
        if not data_parts:
            return (
                np.empty(0, dtype=self.dtype),
                np.empty(0, dtype=self.int_dtype),
                np.empty(0, dtype=self.int_dtype),
            )
        return (
            np.concatenate(data_parts),
            np.concatenate(row_parts),
            np.concatenate(column_parts),
        )

    def _expected_rows_columns(
        self,
        documents: list[list[str]],
        *,
        document_start: int,
        vocab: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[int] = []
        columns: list[int] = []
        for offset, tokens in enumerate(documents):
            unique_ids = list(Counter(vocab[token] for token in tokens))
            rows.extend([document_start + offset] * len(unique_ids))
            columns.extend(unique_ids)
        return (
            np.asarray(rows, dtype=self.int_dtype),
            np.asarray(columns, dtype=self.int_dtype),
        )

    def _write_score_shard(
        self,
        index: int,
        *,
        data: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
        documents: int,
        document_start: int,
        part_sha256: str,
        statistics_signature: str,
    ) -> dict[str, Any]:
        self._layout.scores_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._layout.scores_dir,
            prefix=f".{index:08d}.",
            suffix=".npz.tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                np.savez(
                    output,
                    data=data,
                    rows=rows,
                    columns=columns,
                )
                output.flush()
                os.fsync(output.fileno())
            content_size = temporary.stat().st_size
            score_sha256 = sha256_file(temporary)
            os.replace(temporary, self._layout.score_path(index))
        finally:
            temporary.unlink(missing_ok=True)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "documents": documents,
            "document_start": document_start,
            "entries": int(data.size),
            "part_sha256": part_sha256,
            "statistics_signature": statistics_signature,
            "score_sha256": score_sha256,
            "score_size": content_size,
        }
        atomic_json(self._layout.score_meta_path(index), meta)
        return meta

    def _read_score_meta(self, index: int) -> dict[str, Any]:
        path = self._layout.score_meta_path(index)
        meta = read_small_json(path)
        if not isinstance(meta, dict) or set(meta) != SCORE_META_KEYS:
            raise CorruptCheckpointError(f"score metadata is malformed: {path}")
        integer_fields = (
            "index",
            "documents",
            "document_start",
            "entries",
            "score_size",
        )
        if (
            meta["schema_version"] != SCHEMA_VERSION
            or any(
                isinstance(meta[field], bool)
                or not isinstance(meta[field], int)
                or meta[field] < 0
                for field in integer_fields
            )
            or meta["index"] != index
            or meta["documents"] <= 0
            or meta["score_size"] <= 0
            or not is_sha256(meta["part_sha256"])
            or not is_sha256(meta["statistics_signature"])
            or not is_sha256(meta["score_sha256"])
        ):
            raise CorruptCheckpointError(f"score metadata is malformed: {path}")
        return meta

    def _read_score_shard(
        self,
        index: int,
        *,
        part_meta: dict[str, Any],
        documents: list[list[str]],
        document_start: int,
        statistics: GlobalStatistics,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        meta = self._read_score_meta(index)
        expected_rows, expected_columns = self._expected_rows_columns(
            documents,
            document_start=document_start,
            vocab=statistics.vocab,
        )
        expected_entries = int(expected_rows.size)
        maximum_size = max(
            SCORE_CONTAINER_OVERHEAD,
            expected_entries
            * (self.dtype.itemsize + 2 * self.int_dtype.itemsize)
            * 2
            + SCORE_CONTAINER_OVERHEAD,
        )
        if (
            meta["documents"] != len(documents)
            or meta["document_start"] != document_start
            or meta["entries"] != expected_entries
            or meta["part_sha256"] != part_meta["part_sha256"]
            or meta["statistics_signature"] != statistics.signature
        ):
            raise CorruptCheckpointError(
                f"score metadata does not match batch {index}"
            )
        path = self._layout.score_path(index)
        content = read_bounded(path, meta["score_size"], maximum_size)
        if sha256_bytes(content) != meta["score_sha256"]:
            raise CorruptCheckpointError(f"score checksum mismatch: {path}")
        try:
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                if (
                    len(archive.files) != 3
                    or set(archive.files) != {"data", "rows", "columns"}
                ):
                    raise CorruptCheckpointError(
                        f"score archive has unexpected arrays: {path}"
                    )
                data = archive["data"]
                rows = archive["rows"]
                columns = archive["columns"]
        except (OSError, ValueError, KeyError) as exc:
            raise CorruptCheckpointError(
                f"score archive is malformed: {path}"
            ) from exc
        if (
            data.dtype != self.dtype
            or rows.dtype != self.int_dtype
            or columns.dtype != self.int_dtype
            or data.ndim != 1
            or rows.ndim != 1
            or columns.ndim != 1
            or data.size != expected_entries
            or rows.size != expected_entries
            or columns.size != expected_entries
            or not np.all(np.isfinite(data))
            or not np.array_equal(rows, expected_rows)
            or not np.array_equal(columns, expected_columns)
        ):
            raise CorruptCheckpointError(
                f"score archive arrays are invalid: {path}"
            )
        return data, rows, columns

    def _checkpoint_scores(
        self,
        manifest: dict[str, Any],
        statistics: GlobalStatistics,
    ) -> tuple[int, int]:
        idf, nonoccurrence = self._scoring_context(statistics)
        reused = 0
        rebuilt = 0
        document_start = 0
        for index, part_meta, chunks, documents in self._iter_parts(manifest):
            del chunks
            try:
                self._read_score_shard(
                    index,
                    part_meta=part_meta,
                    documents=documents,
                    document_start=document_start,
                    statistics=statistics,
                )
            except CorruptCheckpointError:
                data, rows, columns = self._score_arrays(
                    documents,
                    document_start=document_start,
                    statistics=statistics,
                    idf=idf,
                    nonoccurrence=nonoccurrence,
                )
                self._write_score_shard(
                    index,
                    data=data,
                    rows=rows,
                    columns=columns,
                    documents=len(documents),
                    document_start=document_start,
                    part_sha256=part_meta["part_sha256"],
                    statistics_signature=statistics.signature,
                )
                rebuilt += 1
            else:
                reused += 1
            document_start += len(documents)
        if document_start != statistics.num_documents:
            raise CorruptCheckpointError(
                "score checkpoints do not cover the complete corpus"
            )
        score_meta_count = self._indexed_metadata_count(
            self._layout.scores_dir, ".meta.json"
        )
        if score_meta_count != manifest["batch_count"]:
            raise CorruptCheckpointError(
                "score metadata count differs from the build manifest"
            )
        return reused, rebuilt

    def _prepare_memmap(
        self,
        filename: str,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
    ) -> np.memmap:
        path = self.generation_dir / f".{filename}.building"
        return np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=shape
        )

    def _merge_scores(
        self,
        manifest: dict[str, Any],
        statistics: GlobalStatistics,
    ) -> np.ndarray | None:
        data = self._prepare_memmap(
            DATA_NAME, self.dtype, (statistics.nonzero_scores,)
        )
        indices = self._prepare_memmap(
            INDICES_NAME, self.int_dtype, (statistics.nonzero_scores,)
        )
        indptr = self._prepare_memmap(
            INDPTR_NAME, np.dtype("int64"), (len(statistics.vocab) + 1,)
        )
        indptr[0] = 0
        np.cumsum(statistics.document_frequencies, out=indptr[1:])
        cursor = np.asarray(indptr[:-1], dtype=np.int64).copy()
        document_start = 0
        for index, part_meta, _, documents in self._iter_parts(manifest):
            shard_data, rows, columns = self._read_score_shard(
                index,
                part_meta=part_meta,
                documents=documents,
                document_start=document_start,
                statistics=statistics,
            )
            if columns.size:
                order = np.argsort(columns, kind="stable")
                sorted_columns = np.asarray(columns[order], dtype=np.int64)
                unique_columns, first_indices, counts = np.unique(
                    sorted_columns,
                    return_index=True,
                    return_counts=True,
                )
                repeated_first = np.repeat(first_indices, counts)
                positions = (
                    np.repeat(cursor[unique_columns], counts)
                    + np.arange(columns.size, dtype=np.int64)
                    - repeated_first
                )
                data[positions] = shard_data[order]
                indices[positions] = rows[order]
                cursor[unique_columns] += counts
            document_start += len(documents)
        if not np.array_equal(cursor, np.asarray(indptr[1:])):
            raise CorruptCheckpointError(
                "score shards do not match document frequencies"
            )
        for array in (data, indices, indptr):
            array.flush()
        del data, indices, indptr
        for filename in (DATA_NAME, INDICES_NAME, INDPTR_NAME):
            os.replace(
                self.generation_dir / f".{filename}.building",
                self.generation_dir / filename,
            )
        _, nonoccurrence = self._scoring_context(statistics)
        return nonoccurrence

    def _save_metadata(
        self,
        manifest: dict[str, Any],
        statistics: GlobalStatistics,
        nonoccurrence: np.ndarray | None,
    ) -> None:
        atomic_vocab(self.generation_dir / VOCAB_NAME, statistics.vocab)
        atomic_json(
            self.generation_dir / PARAMS_NAME,
            {
                "k1": self.k1,
                "b": self.b,
                "delta": self.delta,
                "method": self.method,
                "idf_method": self.idf_method,
                "dtype": self.dtype.name,
                "int_dtype": self.int_dtype.name,
                "num_docs": statistics.num_documents,
                "version": EXPECTED_BM25S_VERSION,
                "backend": "numpy",
            },
        )
        chunks_path = self.generation_dir / CHUNKS_NAME
        offsets_path = self.generation_dir / CHUNK_OFFSETS_NAME
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.generation_dir,
            prefix=f".{CHUNKS_NAME}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        temporary_offsets = (
            self.generation_dir / f".{CHUNK_OFFSETS_NAME}.building"
        )
        offsets: np.memmap | None = None
        try:
            with os.fdopen(descriptor, "wb") as output:
                offsets = np.lib.format.open_memmap(
                    temporary_offsets,
                    mode="w+",
                    dtype=np.dtype("uint64"),
                    shape=(statistics.num_documents + 1,),
                )
                offsets[0] = 0
                document_count = 0
                for _, _, chunks, _ in self._iter_parts(manifest):
                    for chunk in chunks:
                        line = canonical_json(chunk) + b"\n"
                        output.write(line)
                        document_count += 1
                        offsets[document_count] = output.tell()
                output.flush()
                os.fsync(output.fileno())
                offsets.flush()
            if document_count != statistics.num_documents:
                raise CorruptCheckpointError(
                    "chunk metadata does not cover the complete corpus"
                )
            chunks_size = temporary.stat().st_size
            if chunks_size <= 0 or int(offsets[-1]) != chunks_size:
                raise CorruptCheckpointError(
                    "chunk metadata offsets do not match the JSONL size"
                )
            with temporary.open("rb") as input_file:
                input_file.seek(-1, os.SEEK_END)
                if input_file.read(1) != b"\n":
                    raise CorruptCheckpointError(
                        "chunk metadata JSONL has no final newline"
                    )
            del offsets
            offsets = None
            with temporary_offsets.open("r+b") as offsets_file:
                os.fsync(offsets_file.fileno())
            os.replace(temporary, chunks_path)
            os.replace(temporary_offsets, offsets_path)
        finally:
            if offsets is not None:
                offsets.flush()
                del offsets
            temporary.unlink(missing_ok=True)
            temporary_offsets.unlink(missing_ok=True)
        if nonoccurrence is None:
            (self.generation_dir / NONOCCURRENCE_NAME).unlink(missing_ok=True)
        else:
            temporary_array = (
                self.generation_dir / f".{NONOCCURRENCE_NAME}.building"
            )
            output = np.lib.format.open_memmap(
                temporary_array,
                mode="w+",
                dtype=self.dtype,
                shape=nonoccurrence.shape,
            )
            output[:] = nonoccurrence
            output.flush()
            del output
            os.replace(
                temporary_array,
                self.generation_dir / NONOCCURRENCE_NAME,
            )


    def _file_records(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "sha256": sha256_file(self.generation_dir / name),
                "size": (self.generation_dir / name).stat().st_size,
            }
            for name in output_names(self._parameters)
        }


    def _result(
        self,
        *,
        input_sha256: str,
        batch_count: int,
        reused: int,
        rebuilt: int,
        reused_scores: int,
        rebuilt_scores: int,
        statistics: GlobalStatistics,
    ) -> BuildResult:
        return BuildResult(
            generation_dir=self.generation_dir,
            input_sha256=input_sha256,
            num_documents=statistics.num_documents,
            vocabulary_size=len(statistics.vocab),
            nonzero_scores=statistics.nonzero_scores,
            average_document_length=float(
                statistics.average_document_length
            ),
            tokenized_batches=batch_count,
            reused_batches=reused,
            rebuilt_batches=rebuilt,
            scored_batches=batch_count,
            reused_score_batches=reused_scores,
            rebuilt_score_batches=rebuilt_scores,
        )

    def build(self, chunks: Iterable[Chunk]) -> BuildResult:
        """Build or resume one generation without materializing all chunks."""
        manifest = self._load_or_create_manifest()
        input_sha256, batch_count, reused, rebuilt = self._checkpoint_tokens(
            chunks, manifest
        )
        statistics = self._global_statistics(manifest)
        self._prepare_statistics(manifest, statistics)
        if manifest["phase"] != "complete":
            self._validate_disk_capacity(manifest, statistics)
        reused_scores, rebuilt_scores = self._checkpoint_scores(
            manifest, statistics
        )
        if manifest["phase"] == "complete":
            try:
                validate_completed(self._layout, self._parameters, manifest, statistics)
            except CorruptCheckpointError:
                manifest.update({"phase": "scoring", "files": None})
                atomic_json(self._layout.manifest_path, manifest)
                self._validate_disk_capacity(manifest, statistics)
            else:
                return self._result(
                    input_sha256=input_sha256,
                    batch_count=batch_count,
                    reused=reused,
                    rebuilt=rebuilt,
                    reused_scores=reused_scores,
                    rebuilt_scores=rebuilt_scores,
                    statistics=statistics,
                )

        nonoccurrence = self._merge_scores(manifest, statistics)
        self._save_metadata(manifest, statistics, nonoccurrence)
        manifest.update(
            {
                "phase": "complete",
                "files": self._file_records(),
            }
        )
        atomic_json(self._layout.manifest_path, manifest)
        validate_completed(
            self._layout,
            self._parameters,
            manifest,
            statistics,
            verify_checksums=False,
        )
        return self._result(
            input_sha256=input_sha256,
            batch_count=batch_count,
            reused=reused,
            rebuilt=rebuilt,
            reused_scores=reused_scores,
            rebuilt_scores=rebuilt_scores,
            statistics=statistics,
        )
