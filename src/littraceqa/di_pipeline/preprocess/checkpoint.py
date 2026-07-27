"""Resumable, per-paper storage for preprocessing output.

The cache is intentionally independent of a concrete preprocessor. Callers
provide the process configuration, the implementation module, the paper
metadata, and the source artifact used for that paper. A cached paper is reused
only when all of those inputs still match and its JSONL payload passes checksum
and contract validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline.contracts import Chunk


_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = "manifest.jsonl"
_PAPERS_DIRECTORY = "papers"


@dataclass(frozen=True)
class MergeResult:
    """Summary of an atomically published merged chunk file."""

    paper_count: int
    chunk_count: int
    byte_count: int
    sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bundle_sha256(paths: Sequence[Path]) -> str:
    """Fingerprint every implementation file that can affect cached output."""
    records: list[dict[str, str]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"preprocessor implementation module is missing: {path}"
            )
        seen.add(path)
        records.append({"path": str(path), "sha256": _sha256_file(path)})
    return _sha256_bytes(_canonical_json_bytes(records))


def _paper_id(paper: Mapping[str, Any]) -> str:
    paper_id = str(paper.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("paper metadata must contain a non-empty paper_id")
    return paper_id


def _source_state(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "exists": False}
    if not resolved.is_file():
        raise ValueError(f"preprocessing source is not a file: {resolved}")
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_manifest_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("manifest records must be JSON objects")
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("manifest record has an unsupported schema_version")
    if not isinstance(record.get("paper_id"), str) or not record["paper_id"].strip():
        raise ValueError("manifest record has no paper_id")
    if record.get("status") not in {"ok", "failed"}:
        raise ValueError("manifest record has an invalid status")
    return record


def _read_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], bytes | None]:
    """Read a last-write-wins manifest and identify a truncated final record."""
    if not path.exists():
        return {}, None

    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    nonempty_indices = [
        index for index, line in enumerate(lines) if line.strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else None
    records: dict[str, dict[str, Any]] = {}
    valid_prefix_size = 0

    for index, raw_line in enumerate(lines):
        line_start = valid_prefix_size
        valid_prefix_size += len(raw_line)
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            record = _validate_manifest_record(json.loads(stripped))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            is_truncated_tail = (
                index == last_nonempty and not payload.endswith(b"\n")
            )
            if is_truncated_tail:
                return records, payload[:line_start]
            raise ValueError(f"{path}:{index + 1} is not a valid manifest record") from exc
        records[record["paper_id"]] = record

    return records, None


def _validate_chunks(chunks: Iterable[Chunk], paper_id: str) -> list[Chunk]:
    materialized = list(chunks)
    if not materialized:
        raise ValueError(f"preprocessing produced no chunks for {paper_id}")

    seen_chunk_ids: set[str] = set()
    for chunk in materialized:
        if not isinstance(chunk, Chunk):
            raise TypeError("preprocessing output must contain Chunk instances")
        if chunk.paper_id != paper_id:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} belongs to {chunk.paper_id!r}, "
                f"expected {paper_id!r}"
            )
        if not chunk.chunk_id:
            raise ValueError(f"a chunk for {paper_id} has an empty chunk_id")
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"duplicate chunk_id for {paper_id}: {chunk.chunk_id}")
        if not isinstance(chunk.metadata, dict):
            raise ValueError(f"chunk metadata must be a dict: {chunk.chunk_id}")
        seen_chunk_ids.add(chunk.chunk_id)
    return materialized


def _serialize_chunks(chunks: Sequence[Chunk]) -> bytes:
    return b"".join(
        _canonical_json_bytes(chunk.to_dict()) + b"\n" for chunk in chunks
    )


class PreprocessCache:
    """Store and validate preprocessing output one paper at a time."""

    def __init__(
        self,
        root: Path,
        *,
        process_config: Mapping[str, Any],
        source_module_path: Path,
        source_dependency_paths: Sequence[Path] = (),
    ) -> None:
        self.root = root.expanduser().resolve()
        self.papers_dir = self.root / _PAPERS_DIRECTORY
        self.manifest_path = self.root / _MANIFEST_FILENAME
        self._ensure_internal_path(self.papers_dir)
        self._ensure_internal_path(self.manifest_path)
        source_paths = (source_module_path, *source_dependency_paths)
        self.process_signature = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "process_config": process_config,
                    "source_bundle_sha256": _source_bundle_sha256(source_paths),
                }
            )
        )
        self._manifest_lock = threading.Lock()
        self._records, truncated_prefix = _read_manifest(self.manifest_path)
        if truncated_prefix is not None:
            _write_bytes_atomic(self.manifest_path, truncated_prefix)

    def cache_path(self, paper_id: str) -> Path:
        """Return a path-safe, sharded cache file name for a paper ID."""
        normalized = paper_id.strip()
        if not normalized:
            raise ValueError("paper_id must be non-empty")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        path = self.papers_dir / digest[:2] / f"{digest}.jsonl"
        self._ensure_internal_path(path)
        return path

    def input_signature(
        self,
        paper: Mapping[str, Any],
        source_path: Path,
    ) -> tuple[str, str, dict[str, Any]]:
        """Return the full input, metadata, and source-state fingerprints."""
        metadata_bytes = _canonical_json_bytes(paper)
        metadata_sha256 = _sha256_bytes(metadata_bytes)
        source = _source_state(source_path)
        signature = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "process_signature": self.process_signature,
                    "metadata_sha256": metadata_sha256,
                    "source": source,
                }
            )
        )
        return signature, metadata_sha256, source

    def load_valid_chunks(
        self,
        paper: Mapping[str, Any],
        source_path: Path,
    ) -> list[Chunk] | None:
        """Load a reusable cache entry, returning ``None`` when it is stale."""
        paper_id = _paper_id(paper)
        record = self._records.get(paper_id)
        if record is None or record.get("status") != "ok":
            return None

        signature, _, _ = self.input_signature(paper, source_path)
        expected_path = self.cache_path(paper_id)
        expected_relative_path = expected_path.relative_to(self.root).as_posix()
        if (
            record.get("input_signature") != signature
            or record.get("chunk_file") != expected_relative_path
            or not expected_path.is_file()
        ):
            return None

        try:
            payload = expected_path.read_bytes()
            if (
                len(payload) != record.get("chunk_bytes")
                or _sha256_bytes(payload) != record.get("chunk_sha256")
            ):
                return None
            chunks = self._parse_chunk_payload(payload, paper_id, expected_path)
            if len(chunks) != record.get("chunk_count"):
                return None
            return chunks
        except (OSError, TypeError, ValueError):
            return None

    def store_success(
        self,
        paper: Mapping[str, Any],
        source_path: Path,
        chunks: Iterable[Chunk],
    ) -> dict[str, Any]:
        """Atomically publish one paper's chunks, then checkpoint its success."""
        paper_id = _paper_id(paper)
        validated = _validate_chunks(chunks, paper_id)
        payload = _serialize_chunks(validated)
        destination = self.cache_path(paper_id)
        _write_bytes_atomic(destination, payload)

        signature, metadata_sha256, source = self.input_signature(
            paper, source_path
        )
        record = {
            "schema_version": _SCHEMA_VERSION,
            "paper_id": paper_id,
            "status": "ok",
            "attempt": self._next_attempt(paper_id),
            "process_signature": self.process_signature,
            "input_signature": signature,
            "metadata_sha256": metadata_sha256,
            "source": source,
            "chunk_file": destination.relative_to(self.root).as_posix(),
            "chunk_count": len(validated),
            "chunk_bytes": len(payload),
            "chunk_sha256": _sha256_bytes(payload),
            "finished_at": _utc_timestamp(),
        }
        self._append_record(record)
        return record

    def record_failure(
        self,
        paper: Mapping[str, Any],
        source_path: Path,
        error: BaseException,
    ) -> dict[str, Any]:
        """Checkpoint one failed paper without invalidating other papers."""
        paper_id = _paper_id(paper)
        signature, metadata_sha256, source = self.input_signature(
            paper, source_path
        )
        record = {
            "schema_version": _SCHEMA_VERSION,
            "paper_id": paper_id,
            "status": "failed",
            "attempt": self._next_attempt(paper_id),
            "process_signature": self.process_signature,
            "input_signature": signature,
            "metadata_sha256": metadata_sha256,
            "source": source,
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": _utc_timestamp(),
        }
        self._append_record(record)
        return record

    def merge_selected(
        self,
        selected: Iterable[tuple[Mapping[str, Any], Path]],
        destination: Path,
    ) -> MergeResult:
        """Merge valid papers in caller-supplied order and publish atomically."""
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_sibling(destination)
        digest = hashlib.sha256()
        paper_count = 0
        chunk_count = 0
        byte_count = 0
        try:
            with temporary.open("wb") as handle:
                for paper, source_path in selected:
                    paper_id = _paper_id(paper)
                    chunks = self.load_valid_chunks(paper, source_path)
                    if chunks is None:
                        raise ValueError(
                            f"no valid preprocessing cache entry for {paper_id}"
                        )
                    payload = _serialize_chunks(chunks)
                    handle.write(payload)
                    digest.update(payload)
                    paper_count += 1
                    chunk_count += len(chunks)
                    byte_count += len(payload)
                if paper_count == 0:
                    raise ValueError("cannot merge an empty paper selection")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        return MergeResult(
            paper_count=paper_count,
            chunk_count=chunk_count,
            byte_count=byte_count,
            sha256=digest.hexdigest(),
        )

    def latest_records(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the latest per-paper manifest records."""
        return dict(self._records)

    def _next_attempt(self, paper_id: str) -> int:
        previous = self._records.get(paper_id, {}).get("attempt", 0)
        if not isinstance(previous, int) or isinstance(previous, bool):
            previous = 0
        return previous + 1

    def _append_record(self, record: dict[str, Any]) -> None:
        encoded = _canonical_json_bytes(record) + b"\n"
        with self._manifest_lock:
            self._ensure_internal_path(self.manifest_path)
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            needs_separator = False
            if self.manifest_path.exists() and self.manifest_path.stat().st_size:
                with self.manifest_path.open("rb") as existing:
                    existing.seek(-1, os.SEEK_END)
                    needs_separator = existing.read(1) != b"\n"
            with self.manifest_path.open("ab") as handle:
                if needs_separator:
                    handle.write(b"\n")
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._records[record["paper_id"]] = record

    def _ensure_internal_path(self, path: Path) -> None:
        """Reject cache paths that escape through pre-existing symlinks."""
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"cache path escapes cache root: {path}") from exc

        current = path
        while current != self.root:
            if current.is_symlink():
                raise ValueError(f"cache path must not contain symlinks: {current}")
            current = current.parent

        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"cache path escapes cache root: {resolved}") from exc

    @staticmethod
    def _parse_chunk_payload(
        payload: bytes,
        paper_id: str,
        path: Path,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError("chunk record must be a JSON object")
                chunks.append(Chunk(**value))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number} is not a valid Chunk record"
                ) from exc
        return _validate_chunks(chunks, paper_id)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
