#!/usr/bin/env python3
"""Seed a fresh answer-only run with immutable Stage-1 checkpoints.

Only ``candidate_judgments.jsonl`` files are copied.  Answers, submissions,
provider ledgers, and aggregate files are intentionally excluded.  The normal
pairwise runner must subsequently recompute and validate every judgment cache
key against the current query, candidates, corpus, images, Stage-1 prompt, and
reader limits before it makes any Stage-2 call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise TypeError(f"checkpoint is not an object at {path}:{line_number}")
        records.append(record)
    return records


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _direct_query_directories(run: Path) -> list[Path]:
    """Return real, immediate child directories in deterministic order."""

    if not run.exists():
        return []
    if not run.is_dir():
        raise NotADirectoryError(f"run path is not a directory: {run}")
    query_directories: list[Path] = []
    for child in run.iterdir():
        if child.is_symlink():
            raise ValueError(f"run directory contains a symlinked child: {child}")
        if child.is_dir():
            query_directories.append(child)
    return sorted(query_directories, key=lambda path: path.name)


def seed(source_run: Path, destination_run: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    destination_run = destination_run.resolve()
    if source_run == destination_run:
        raise ValueError("source and destination run directories must differ")
    source_manifest = source_run / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest is missing: {source_manifest}")
    if (destination_run / "manifest.json").exists():
        raise FileExistsError(
            "destination already has a manifest; seed only a new run directory"
        )
    destination_query_directories = _direct_query_directories(destination_run)
    existing_payloads = sorted(
        (
            path
            for query_directory in destination_query_directories
            for path in query_directory.iterdir()
            if path.name != "candidate_judgments.jsonl"
        ),
        key=lambda path: str(path),
    )
    if existing_payloads:
        raise FileExistsError(
            f"destination contains non-judgment run state: {existing_payloads[0]}"
        )

    sources = [
        checkpoint
        for query_directory in _direct_query_directories(source_run)
        if (checkpoint := query_directory / "candidate_judgments.jsonl").is_file()
        and not checkpoint.is_symlink()
    ]
    if not sources:
        raise FileNotFoundError(f"no Stage-1 checkpoints found under {source_run}")
    copied: list[dict[str, Any]] = []
    total_pairs = 0
    for source in sources:
        query_id = source.parent.name
        records = _read_jsonl(source)
        if not records:
            raise ValueError(f"empty Stage-1 checkpoint: {source}")
        paper_ids: set[str] = set()
        for line_number, record in enumerate(records, 1):
            paper_id = str(record.get("paper_id") or "")
            if (
                record.get("query_id") != query_id
                or record.get("status") != "complete"
                or not paper_id
                or paper_id in paper_ids
                or not isinstance(record.get("cache_key"), str)
            ):
                raise ValueError(
                    f"invalid complete Stage-1 checkpoint at {source}:{line_number}"
                )
            paper_ids.add(paper_id)
        destination = destination_run / query_id / "candidate_judgments.jsonl"
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise FileExistsError(f"different judgment seed already exists: {destination}")
        if not destination.exists():
            _atomic_copy(source, destination)
        digest = _sha256(source)
        copied.append(
            {
                "query_id": query_id,
                "pairs": len(records),
                "sha256": digest,
                "source": str(source),
                "destination": str(destination),
            }
        )
        total_pairs += len(records)

    provenance = {
        "schema_version": 1,
        "purpose": "stage1_judgment_seed_for_fresh_answer_only_run",
        "source_run": str(source_run),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "destination_run": str(destination_run),
        "queries": len(copied),
        "candidate_judgments": total_pairs,
        "files": copied,
        "required_next_step": (
            "run scripts/run_aoai_pairwise_reader.py --stage answer; its "
            "validate_judgment_checkpoint path must accept every copied cache key"
        ),
    }
    provenance_path = destination_run / "seeded_judgments.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance_path.with_name(f".{provenance_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, provenance_path)
    finally:
        temporary.unlink(missing_ok=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    args = parser.parse_args()
    result = seed(args.source_run, args.destination_run)
    print(
        "seeded "
        f"{result['candidate_judgments']} Stage-1 judgments across "
        f"{result['queries']} queries; the answer runner must now validate "
        "their cache keys"
    )


if __name__ == "__main__":
    main()
