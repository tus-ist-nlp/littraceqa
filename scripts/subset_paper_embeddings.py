#!/usr/bin/env python3
"""Export a bounded paper subset from a shared FAISS IndexFlatIP.

The exporter reconstructs existing vectors and never loads an embedding model.
Source records are streamed in their original order, and the shared source is
opened read-only.  Both ``--expected-count`` and ``--max-papers`` are required
so an unexpectedly broad ID file cannot silently produce a large export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_INDEX_FILENAME = "index.faiss"
_SOURCE_RECORDS_FILENAME = "chunks.jsonl"
_EMBEDDINGS_FILENAME = "embeddings.npy"
_PAPERS_FILENAME = "papers.jsonl"
_CONFIG_FILENAME = "index_config.json"
_CHUNK_KEYS = frozenset(
    {"chunk_id", "paper_id", "text", "chunk_type", "metadata"}
)
_COPY_BLOCK_BYTES = 1024 * 1024
_MAX_ID_LINE_BYTES = 4096
_MAX_RECORD_LINE_BYTES = 16 * 1024 * 1024
_MAX_DIMENSION = 65_536


class ExportError(RuntimeError):
    """Raised when a source or requested export fails validation."""


@dataclass(frozen=True)
class ExportResult:
    """Summary of one completed atomic export."""

    output_dir: Path
    paper_count: int
    dimension: int
    source_paper_count: int


@dataclass(frozen=True)
class _FileSnapshot:
    size: int
    mtime_ns: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while content := input_file.read(_COPY_BLOCK_BYTES):
            digest.update(content)
    return digest.hexdigest()


def _snapshot(path: Path) -> _FileSnapshot:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ExportError(f"source file is unreadable: {path}") from exc
    if not path.is_file():
        raise ExportError(f"source path is not a regular file: {path}")
    return _FileSnapshot(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _same_snapshot(path: Path, snapshot: _FileSnapshot) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        path.is_file()
        and stat.st_size == snapshot.size
        and stat.st_mtime_ns == snapshot.mtime_ns
    )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _read_target_ids(
    path: Path,
    *,
    expected_count: int,
    max_papers: int,
) -> tuple[list[str], str]:
    snapshot = _snapshot(path)
    digest = hashlib.sha256()
    paper_ids: list[str] = []
    seen: set[str] = set()
    try:
        with path.open("rb") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                digest.update(raw_line)
                if len(raw_line) > _MAX_ID_LINE_BYTES:
                    raise ExportError(
                        f"paper ID line is too long at {path}:{line_number}"
                    )
                try:
                    paper_id = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise ExportError(
                        "paper ID file must be readable UTF-8 text: "
                        f"{path}:{line_number}"
                    ) from exc
                if not paper_id or paper_id.startswith("#"):
                    continue
                if any(character.isspace() for character in paper_id):
                    raise ExportError(
                        f"paper ID contains whitespace at "
                        f"{path}:{line_number}"
                    )
                if paper_id in seen:
                    raise ExportError(
                        f"duplicate paper ID at "
                        f"{path}:{line_number}: {paper_id}"
                    )
                seen.add(paper_id)
                paper_ids.append(paper_id)
                if len(paper_ids) > max_papers:
                    raise ExportError(
                        f"paper ID count exceeds --max-papers: "
                        f"{len(paper_ids)} > {max_papers}"
                    )
    except OSError as exc:
        raise ExportError(
            f"paper ID file must be readable UTF-8 text: {path}"
        ) from exc
    if not _same_snapshot(path, snapshot):
        raise ExportError(f"paper ID file changed while reading: {path}")

    if len(paper_ids) != expected_count:
        raise ExportError(
            "paper ID count does not match --expected-count: "
            f"{len(paper_ids)} != {expected_count}"
        )
    return paper_ids, digest.hexdigest()


def _contains_gold_field(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                normalized = key.casefold().replace("-", "_")
                if normalized.startswith("gold") or "_gold" in normalized:
                    return True
            if _contains_gold_field(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_gold_field(child) for child in value)
    return False


def _validate_source_record(
    record: object,
    *,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _CHUNK_KEYS:
        raise ExportError(
            f"source record has an invalid schema at {path}:{line_number}"
        )
    string_fields = ("chunk_id", "paper_id", "text", "chunk_type")
    if any(
        not isinstance(record[field], str) or not record[field].strip()
        for field in string_fields
    ):
        raise ExportError(
            f"source record has invalid string fields at "
            f"{path}:{line_number}"
        )
    if any(character.isspace() for character in record["paper_id"]):
        raise ExportError(
            f"source paper_id contains whitespace at {path}:{line_number}"
        )
    if record["chunk_type"] != "title_abstract":
        raise ExportError(
            "paper embedding source must contain only title_abstract rows; "
            f"found {record['chunk_type']!r} at {path}:{line_number}"
        )
    if not isinstance(record["metadata"], dict):
        raise ExportError(
            f"source metadata is not an object at {path}:{line_number}"
        )
    if _contains_gold_field(record):
        raise ExportError(
            f"source record contains a gold field at {path}:{line_number}"
        )
    try:
        _canonical_json(record)
    except (TypeError, ValueError) as exc:
        raise ExportError(
            f"source record is not deterministically serializable at "
            f"{path}:{line_number}"
        ) from exc
    return record


def _scan_source_records(
    path: Path,
    target_ids: set[str],
) -> tuple[list[tuple[int, dict[str, Any]]], int, str]:
    digest = hashlib.sha256()
    selected: list[tuple[int, dict[str, Any]]] = []
    seen_paper_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    source_count = 0
    try:
        with path.open("rb") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                digest.update(raw_line)
                if len(raw_line) > _MAX_RECORD_LINE_BYTES:
                    raise ExportError(
                        f"source row is too large at {path}:{line_number}"
                    )
                if not raw_line.strip():
                    raise ExportError(
                        f"source contains a blank row at {path}:{line_number}"
                    )
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ExportError(
                        f"source row is not valid UTF-8 JSON at "
                        f"{path}:{line_number}"
                    ) from exc
                record = _validate_source_record(
                    record,
                    path=path,
                    line_number=line_number,
                )
                paper_id = record["paper_id"]
                if paper_id in seen_paper_ids:
                    raise ExportError(
                        f"source paper_id is not unique at "
                        f"{path}:{line_number}: {paper_id}"
                    )
                chunk_id = record["chunk_id"]
                if chunk_id in seen_chunk_ids:
                    raise ExportError(
                        f"source chunk_id is not unique at "
                        f"{path}:{line_number}: {chunk_id}"
                    )
                seen_paper_ids.add(paper_id)
                seen_chunk_ids.add(chunk_id)
                if paper_id in target_ids:
                    selected.append((source_count, record))
                source_count += 1
    except OSError as exc:
        raise ExportError(f"source records are unreadable: {path}") from exc
    return selected, source_count, digest.hexdigest()


def _validate_index(faiss_module: Any, index: Any) -> tuple[int, int]:
    index_type = getattr(faiss_module, "IndexFlatIP", None)
    if index_type is None or not isinstance(index, index_type):
        raise ExportError("source FAISS index must be an IndexFlatIP")
    ntotal = getattr(index, "ntotal", None)
    dimension = getattr(index, "d", None)
    if (
        isinstance(ntotal, bool)
        or not isinstance(ntotal, (int, np.integer))
        or int(ntotal) <= 0
    ):
        raise ExportError("source FAISS index has an invalid ntotal")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or not 0 < int(dimension) <= _MAX_DIMENSION
    ):
        raise ExportError("source FAISS index has an invalid dimension")
    return int(ntotal), int(dimension)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as output:
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_output(
    *,
    output_dir: Path,
    selected: list[tuple[int, dict[str, Any]]],
    index: Any,
    dimension: int,
    source: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise ExportError(f"output directory already exists: {output_dir}")
    temporary = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
            suffix=".building",
        )
    )
    try:
        embeddings_path = temporary / _EMBEDDINGS_FILENAME
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.dtype("float32"),
            shape=(len(selected), dimension),
        )
        try:
            for output_row, (source_row, _) in enumerate(selected):
                try:
                    vector = np.asarray(
                        index.reconstruct(source_row),
                        dtype=np.float32,
                    )
                except Exception as exc:
                    raise ExportError(
                        f"cannot reconstruct source FAISS row {source_row}"
                    ) from exc
                if (
                    vector.shape != (dimension,)
                    or not np.all(np.isfinite(vector))
                ):
                    raise ExportError(
                        f"source FAISS row {source_row} has an invalid vector"
                    )
                embeddings[output_row] = vector
            embeddings.flush()
        finally:
            del embeddings
        _fsync_file(embeddings_path)

        papers_path = temporary / _PAPERS_FILENAME
        with papers_path.open("wb") as output:
            for _, record in selected:
                output.write(_canonical_json(record) + b"\n")
            output.flush()
            os.fsync(output.fileno())

        files = {
            filename: {
                "sha256": _sha256_file(temporary / filename),
                "size": (temporary / filename).stat().st_size,
            }
            for filename in (_EMBEDDINGS_FILENAME, _PAPERS_FILENAME)
        }
        config = {
            "schema_version": 1,
            "paper_count": len(selected),
            "dimension": dimension,
            "files": files,
            "source": source,
        }
        config_path = temporary / _CONFIG_FILENAME
        with config_path.open("wb") as output:
            output.write(_canonical_json(config) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export_subset(
    *,
    source_index_dir: str | Path,
    paper_ids_file: str | Path,
    output_dir: str | Path,
    shared_read_only_root: str | Path,
    expected_count: int,
    max_papers: int,
) -> ExportResult:
    """Export selected source rows without loading an embedding model."""

    expected_count = _positive_integer(expected_count, "expected_count")
    max_papers = _positive_integer(max_papers, "max_papers")
    if expected_count > max_papers:
        raise ValueError("expected_count must not exceed max_papers")

    try:
        shared_root = Path(shared_read_only_root).expanduser().resolve(
            strict=True
        )
        source_dir = Path(source_index_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExportError(
            "shared root and source index directory must already exist"
        ) from exc
    if not shared_root.is_dir() or not source_dir.is_dir():
        raise ExportError(
            "shared root and source index path must be directories"
        )
    if not _is_within(source_dir, shared_root):
        raise ExportError(
            "source index directory must be within the shared read-only root"
        )

    output = Path(output_dir).expanduser().resolve(strict=False)
    if _is_within(output, shared_root):
        raise ExportError(
            "output directory must be outside the shared read-only root"
        )
    if _is_within(output, source_dir):
        raise ExportError("output directory must not be inside the source")
    if output.exists():
        raise ExportError(f"output directory already exists: {output}")

    ids_path = Path(paper_ids_file).expanduser().resolve(strict=True)
    target_ids, ids_sha256 = _read_target_ids(
        ids_path,
        expected_count=expected_count,
        max_papers=max_papers,
    )
    target_id_set = set(target_ids)

    index_path = source_dir / _INDEX_FILENAME
    records_path = source_dir / _SOURCE_RECORDS_FILENAME
    index_snapshot = _snapshot(index_path)
    records_snapshot = _snapshot(records_path)

    try:
        import faiss  # Imported only when an export is explicitly executed.
    except ImportError as exc:
        raise ExportError(
            "FAISS is required to export existing paper embeddings"
        ) from exc
    index_sha256 = _sha256_file(index_path)
    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise ExportError(f"cannot read source FAISS index: {index_path}") from exc
    ntotal, dimension = _validate_index(faiss, index)

    selected, source_count, records_sha256 = _scan_source_records(
        records_path,
        target_id_set,
    )
    if not _same_snapshot(index_path, index_snapshot) or not _same_snapshot(
        records_path, records_snapshot
    ):
        raise ExportError("shared source changed while it was being read")
    if ntotal != source_count:
        raise ExportError(
            "FAISS ntotal does not match chunks.jsonl row count: "
            f"{ntotal} != {source_count}"
        )

    selected_ids = {record["paper_id"] for _, record in selected}
    missing = sorted(target_id_set - selected_ids)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ExportError(f"target paper IDs are missing: {preview}{suffix}")
    if len(selected) != expected_count:
        raise ExportError(
            "selected paper count does not match --expected-count: "
            f"{len(selected)} != {expected_count}"
        )

    source = {
        "index_type": "IndexFlatIP",
        "index_path": str(index_path),
        "index_sha256": index_sha256,
        "index_size": index_snapshot.size,
        "chunks_path": str(records_path),
        "chunks_sha256": records_sha256,
        "chunks_size": records_snapshot.size,
        "source_paper_count": source_count,
        "source_dimension": dimension,
        "target_ids_path": str(ids_path),
        "target_ids_sha256": ids_sha256,
        "expected_count": expected_count,
        "max_papers": max_papers,
        "selection_order": "source_row",
    }
    _write_atomic_output(
        output_dir=output,
        selected=selected,
        index=index,
        dimension=dimension,
        source=source,
    )
    return ExportResult(
        output_dir=output,
        paper_count=len(selected),
        dimension=dimension,
        source_paper_count=source_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a bounded paper subset from an existing shared "
            "FAISS IndexFlatIP without loading an embedding model."
        )
    )
    parser.add_argument(
        "--source-index-dir",
        required=True,
        help="Read-only directory containing index.faiss and chunks.jsonl.",
    )
    parser.add_argument(
        "--paper-ids-file",
        required=True,
        help="UTF-8 text file with one target paper_id per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory outside the shared read-only root.",
    )
    parser.add_argument(
        "--shared-read-only-root",
        required=True,
        help="Shared root that must never contain the output directory.",
    )
    parser.add_argument(
        "--expected-count",
        required=True,
        type=int,
        help="Exact number of unique target paper IDs expected.",
    )
    parser.add_argument(
        "--max-papers",
        required=True,
        type=int,
        help="Explicit hard upper bound for this export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded exporter CLI."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = export_subset(
            source_index_dir=args.source_index_dir,
            paper_ids_file=args.paper_ids_file,
            output_dir=args.output_dir,
            shared_read_only_root=args.shared_read_only_root,
            expected_count=args.expected_count,
            max_papers=args.max_papers,
        )
    except (ExportError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "paper_count": result.paper_count,
                "dimension": result.dimension,
                "source_paper_count": result.source_paper_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
