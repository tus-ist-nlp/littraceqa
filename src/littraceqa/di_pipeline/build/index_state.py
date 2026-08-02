"""Fingerprint index builds so an interrupted run can resume safely.

An index is only reused when the merged corpus, the configuration and the
indexer implementation all hash to the values recorded when it was written.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from littraceqa.di_pipeline.index.chunk_store import iter_chunks
from littraceqa.di_pipeline.preprocess.checkpoint import MergeResult


_INDEX_BUILD_STATE_SCHEMA = 1


@dataclass(frozen=True)
class IndexBuildRun:
    """Result of a resumable, ordered index build pass."""

    built_count: int
    loaded_count: int


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize state inputs deterministically for signatures."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_source_paths(component: Any) -> tuple[Path, ...]:
    """Return the component module and declared implementation dependencies."""
    paths: list[Path] = []
    pending: list[Any] = [type(component)]
    instance_dependencies = getattr(component, "checkpoint_dependencies", ())
    if isinstance(instance_dependencies, (str, bytes)) or not isinstance(
        instance_dependencies, Sequence
    ):
        raise TypeError("checkpoint_dependencies must be a sequence")
    pending.extend(instance_dependencies)
    seen_objects: set[int] = set()

    while pending:
        dependency = pending.pop(0)
        if isinstance(dependency, (str, Path)):
            path = Path(dependency)
        else:
            identity = id(dependency)
            if identity in seen_objects:
                continue
            seen_objects.add(identity)
            path = Path(inspect.getfile(dependency))
            nested = getattr(dependency, "checkpoint_dependencies", ())
            if isinstance(nested, (str, bytes)) or not isinstance(nested, Sequence):
                raise TypeError("checkpoint_dependencies must be a sequence")
            pending.extend(nested)
        paths.append(path.expanduser().resolve())

    return tuple(dict.fromkeys(paths))


def _source_bundle_sha256(paths: Sequence[Path]) -> str:
    """Hash implementation files without relying on filesystem timestamps."""
    records = []
    for path in sorted(set(paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(f"implementation source is missing: {path}")
        records.append({"path": str(path), "sha256": _file_sha256(path)})
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def fingerprint_chunk_file(path: Path) -> MergeResult:
    """Describe an existing merged JSONL without loading its records."""
    digest = hashlib.sha256()
    byte_count = 0
    chunk_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if line.strip():
                chunk_count += 1
    if chunk_count == 0:
        raise ValueError(f"merged chunk file is empty: {path}")
    return MergeResult(
        paper_count=0,
        chunk_count=chunk_count,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one JSON state snapshot using a same-directory atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_index_build_state(path: Path) -> dict[str, Any]:
    """Return an empty state when a prior snapshot is missing or malformed."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "schema_version": _INDEX_BUILD_STATE_SCHEMA,
            "indexers": {},
        }
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _INDEX_BUILD_STATE_SCHEMA
        or not isinstance(value.get("indexers"), dict)
    ):
        return {
            "schema_version": _INDEX_BUILD_STATE_SCHEMA,
            "indexers": {},
        }
    return value


def _indexer_state_key(position: int, config: Mapping[str, Any]) -> str:
    return f"{position:03d}:{config.get('name', 'indexer')}"


def _indexer_build_signature(
    *,
    indexer: Any,
    config: Mapping[str, Any],
    chunks: MergeResult,
) -> tuple[str, str]:
    """Sign the corpus, config, and all declared implementation sources."""
    implementation_sha256 = _source_bundle_sha256(
        implementation_source_paths(indexer)
    )
    signature = hashlib.sha256(
        _canonical_json_bytes(
            {
                "chunks": {
                    "sha256": chunks.sha256,
                    "byte_count": chunks.byte_count,
                    "chunk_count": chunks.chunk_count,
                },
                "effective_config": config,
                "implementation_sha256": implementation_sha256,
            }
        )
    ).hexdigest()
    return signature, implementation_sha256


def build_indexers_with_resume(
    *,
    indexers: Sequence[Any],
    indexer_configs: Sequence[Mapping[str, Any]],
    chunks_path: Path,
    chunks: MergeResult,
    state_path: Path,
    resume: bool,
) -> IndexBuildRun:
    """Build indexes in order, loading only verified completed checkpoints."""
    if len(indexers) != len(indexer_configs):
        raise ValueError("indexer objects and configs must have the same length")

    state = _read_index_build_state(state_path) if resume else {
        "schema_version": _INDEX_BUILD_STATE_SCHEMA,
        "indexers": {},
    }
    state["chunks"] = {
        "sha256": chunks.sha256,
        "byte_count": chunks.byte_count,
        "chunk_count": chunks.chunk_count,
    }
    records = state["indexers"]
    keys = [
        _indexer_state_key(position, config)
        for position, config in enumerate(indexer_configs)
    ]
    if not resume:
        _write_json_atomic(state_path, state)

    built_count = 0
    loaded_count = 0
    rebuild_remaining = not resume

    for position, (indexer, config, key) in enumerate(
        zip(indexers, indexer_configs, keys, strict=True)
    ):
        signature, implementation_sha256 = _indexer_build_signature(
            indexer=indexer,
            config=config,
            chunks=chunks,
        )
        record = records.get(key)
        can_load = (
            not rebuild_remaining
            and isinstance(record, dict)
            and record.get("status") == "complete"
            and record.get("signature") == signature
        )
        if can_load:
            try:
                indexer.load()
            except Exception:
                rebuild_remaining = True
            else:
                loaded_count += 1
                print(f"  {indexer.name} loaded from a verified checkpoint")
                continue
        else:
            rebuild_remaining = True

        # A changed or unreadable index invalidates itself and every later
        # indexer because the ordered build may have been interrupted.
        for invalid_key in keys[position:]:
            records.pop(invalid_key, None)
        records[key] = {
            "status": "building",
            "signature": signature,
            "implementation_sha256": implementation_sha256,
            "effective_config": config,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json_atomic(state_path, state)

        print(f"  Building {indexer.name}...")
        try:
            build_with_signature = getattr(
                indexer,
                "build_with_signature",
                None,
            )
            if callable(build_with_signature):
                build_with_signature(iter_chunks(chunks_path), signature)
            else:
                indexer.build(iter_chunks(chunks_path))
        except Exception as exc:
            records[key] = {
                **records[key],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json_atomic(state_path, state)
            raise

        built_count += 1
        records[key] = {
            **records[key],
            "status": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json_atomic(state_path, state)

    return IndexBuildRun(
        built_count=built_count,
        loaded_count=loaded_count,
    )
