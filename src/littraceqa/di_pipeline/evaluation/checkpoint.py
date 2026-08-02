"""Describe an evaluation run precisely enough to resume or reject it.

A checkpoint is only reusable when the gold file, the composed configuration,
the retrieval code and the loaded indexes all still hash to what it recorded.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path


_CHECKPOINT_SCHEMA_VERSION = 1
_MAX_INDEX_STATE_ENTRIES = 2048


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    """Fingerprint retrieval source and dependency declarations without data files."""
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src" / "littraceqa" / "di_pipeline"
    paths = [
        Path(__file__).resolve(),
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        *sorted(source_root.rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        try:
            relative_path = path.relative_to(project_root)
        except ValueError:
            relative_path = path
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_state() -> dict:
    """Record the interpreter and retrieval package versions used by a run."""
    package_versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "faiss-cpu", "numpy", "torch", "transformers"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": package_versions,
    }


def _index_state(cfg: dict) -> list[dict]:
    """Snapshot shallow index metadata without hashing large index contents."""
    snapshots: list[dict] = []
    indexers = cfg.get("retriever", {}).get("indexers", [])
    for position, indexer in enumerate(indexers):
        raw_path = (indexer.get("params") or {}).get("index_dir")
        snapshot: dict = {
            "position": position,
            "name": str(indexer.get("name") or ""),
            "path": str(Path(raw_path).expanduser().resolve()) if raw_path else None,
        }
        if not raw_path:
            snapshot["state"] = "unspecified"
            snapshots.append(snapshot)
            continue

        path = Path(raw_path).expanduser().resolve()
        try:
            root_stat = path.stat()
        except FileNotFoundError:
            snapshot["state"] = "missing"
            snapshots.append(snapshot)
            continue
        except OSError as exc:
            snapshot.update({"state": "unreadable", "error_type": type(exc).__name__})
            snapshots.append(snapshot)
            continue

        snapshot.update(
            {
                "state": "directory" if path.is_dir() else "file",
                "size": root_stat.st_size,
                "mtime_ns": root_stat.st_mtime_ns,
            }
        )
        if path.is_dir():
            entries = []
            truncated = False
            try:
                for entry in path.iterdir():
                    if len(entries) >= _MAX_INDEX_STATE_ENTRIES:
                        truncated = True
                        break
                    entries.append(entry)
                children = []
                for entry in sorted(entries, key=lambda item: item.name):
                    entry_stat = entry.stat()
                    children.append(
                        {
                            "name": entry.name,
                            "kind": "directory" if entry.is_dir() else "file",
                            "size": entry_stat.st_size,
                            "mtime_ns": entry_stat.st_mtime_ns,
                        }
                    )
            except OSError as exc:
                snapshot.update(
                    {"state": "unreadable", "error_type": type(exc).__name__}
                )
            else:
                snapshot["children"] = children
                snapshot["children_truncated"] = truncated
        snapshots.append(snapshot)
    return snapshots


def build_checkpoint(
    cfg: dict,
    ks: Sequence[int],
    query_path: Path,
    records: Sequence[dict],
) -> dict:
    """Build a deterministic signature that prevents incompatible resume runs."""
    resolved_query_path = query_path.expanduser().resolve()
    run_spec = {
        "config": cfg,
        "ks": list(ks),
        "queries_path": str(resolved_query_path),
        "queries_sha256": _file_sha256(resolved_query_path),
        "requested_query_ids": [str(record["query_id"]) for record in records],
        "source_sha256": _source_sha256(),
        "runtime": _runtime_state(),
        "index_state": _index_state(cfg),
    }
    encoded = json.dumps(
        run_spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "run_signature": hashlib.sha256(encoded).hexdigest(),
        "run_spec": run_spec,
    }


def load_resume_state(
    output: Path,
    expected_checkpoint: dict,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load successful diagnostics and retryable failures from a checkpoint."""
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read resume checkpoint: {output}") from exc
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a JSON object")

    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("resume output does not contain checkpoint metadata")
    if checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema version is incompatible")
    if checkpoint.get("run_signature") != expected_checkpoint["run_signature"]:
        raise ValueError("resume checkpoint was created with different inputs or settings")

    requested_ids = set(expected_checkpoint["run_spec"]["requested_query_ids"])
    diagnostics: dict[str, dict] = {}
    raw_diagnostics = payload.get("queries", [])
    if not isinstance(raw_diagnostics, list):
        raise ValueError("resume checkpoint queries must be a list")
    for diagnostic in raw_diagnostics:
        if not isinstance(diagnostic, dict):
            raise ValueError("resume checkpoint contains an invalid query diagnostic")
        query_id = str(diagnostic.get("query_id") or "")
        if not query_id or query_id not in requested_ids:
            raise ValueError(f"resume checkpoint contains unexpected query_id: {query_id}")
        if query_id in diagnostics:
            raise ValueError(f"resume checkpoint contains duplicate query_id: {query_id}")
        if not isinstance(diagnostic.get("gold_papers"), list) or not isinstance(
            diagnostic.get("ranked_papers"), list
        ):
            raise ValueError(
                f"resume checkpoint diagnostic is incomplete for query_id: {query_id}"
            )
        diagnostics[query_id] = diagnostic

    failures: dict[str, dict] = {}
    raw_failures = payload.get("failures", [])
    if not isinstance(raw_failures, list):
        raise ValueError("resume checkpoint failures must be a list")
    for failure in raw_failures:
        if not isinstance(failure, dict):
            raise ValueError("resume checkpoint contains an invalid failure")
        query_id = str(failure.get("query_id") or "")
        if not query_id or query_id not in requested_ids:
            raise ValueError(
                f"resume checkpoint contains unexpected failed query_id: {query_id}"
            )
        if query_id in failures:
            raise ValueError(
                f"resume checkpoint contains duplicate failed query_id: {query_id}"
            )
        if query_id in diagnostics:
            continue
        attempts = failure.get("attempts", 1)
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts <= 0
        ):
            raise ValueError(
                f"resume checkpoint has invalid attempts for query_id: {query_id}"
            )
        failures[query_id] = {
            "query_id": query_id,
            "error_type": str(failure.get("error_type") or "UnknownError"),
            "error": str(failure.get("error") or "")[:2000],
            "attempts": attempts,
        }
    return diagnostics, failures
