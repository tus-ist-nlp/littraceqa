"""Persist the derived method graph beside a paper-level BM25 index.

The sidecar caches work already done during ``build``. It is trusted only when
it still describes the same corpus and the same degree bound, so a stale or
hand-edited file is rebuilt rather than believed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.method_matching import method_corpus_signature

METHOD_GRAPH_FILENAME = "method_alias_graph.json"
METHOD_GRAPH_SCHEMA_VERSION = 3


def method_sidecar_path(index_dir: Path) -> Path:
    return Path(index_dir) / METHOD_GRAPH_FILENAME


def save_method_sidecar(
    index_dir: Path,
    documents: Sequence[Chunk],
    method_owner_by_alias: dict[str, str] | None,
    method_neighbors_by_paper_id: (
        dict[str, tuple[dict[str, Any], ...]] | None
    ),
    method_max_degree: int,
) -> None:
    if method_owner_by_alias is None or method_neighbors_by_paper_id is None:
        return
    index_dir = Path(index_dir)
    payload = {
        "schema_version": METHOD_GRAPH_SCHEMA_VERSION,
        "method_max_degree": method_max_degree,
        "corpus_signature": method_corpus_signature(documents),
        "owners": method_owner_by_alias,
        "neighbors": {
            paper_id: list(items)
            for paper_id, items in method_neighbors_by_paper_id.items()
        },
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    path = method_sidecar_path(index_dir)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=index_dir,
            prefix=f".{METHOD_GRAPH_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def validate_method_sidecar(
    payload: object,
    documents: Sequence[Chunk],
    method_max_degree: int,
) -> tuple[
    dict[str, str],
    dict[str, tuple[dict[str, Any], ...]],
]:
    if not isinstance(payload, dict):
        raise ValueError("method graph sidecar must be an object")
    if payload.get("schema_version") != METHOD_GRAPH_SCHEMA_VERSION:
        raise ValueError("method graph sidecar schema is incompatible")
    if payload.get("method_max_degree") != method_max_degree:
        raise ValueError("method graph sidecar degree limit differs")
    if payload.get("corpus_signature") != method_corpus_signature(documents):
        raise ValueError("method graph sidecar corpus differs")

    paper_ids = {chunk.paper_id for chunk in documents}
    raw_owners = payload.get("owners")
    raw_neighbors = payload.get("neighbors")
    if not isinstance(raw_owners, dict) or not isinstance(
        raw_neighbors,
        dict,
    ):
        raise ValueError("method graph sidecar fields are invalid")

    owners: dict[str, str] = {}
    for alias, paper_id in raw_owners.items():
        if (
            not isinstance(alias, str)
            or not alias
            or not isinstance(paper_id, str)
            or paper_id not in paper_ids
        ):
            raise ValueError("method graph owner is invalid")
        owners[alias] = paper_id

    neighbors: dict[str, tuple[dict[str, Any], ...]] = {}
    directed_relations: set[
        tuple[str, str, tuple[str, ...], int]
    ] = set()
    for paper_id, items in raw_neighbors.items():
        if paper_id not in paper_ids or not isinstance(items, list):
            raise ValueError("method graph neighbor list is invalid")
        checked_items: list[dict[str, Any]] = []
        seen_neighbors: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("method graph neighbor is invalid")
            neighbor_id = item.get("paper_id")
            aliases = item.get("aliases")
            strength = item.get("strength")
            if (
                not isinstance(neighbor_id, str)
                or neighbor_id not in paper_ids
                or neighbor_id == paper_id
                or not isinstance(aliases, list)
                or not aliases
                or aliases != sorted(set(aliases))
                or any(
                    not isinstance(alias, str)
                    or alias not in owners
                    or owners[alias] not in {paper_id, neighbor_id}
                    for alias in aliases
                )
                or isinstance(strength, bool)
                or not isinstance(strength, int)
                or strength != len(aliases)
                or neighbor_id in seen_neighbors
            ):
                raise ValueError("method graph neighbor is invalid")
            seen_neighbors.add(neighbor_id)
            checked_items.append(
                {
                    "paper_id": neighbor_id,
                    "aliases": list(aliases),
                    "strength": strength,
                }
            )
            directed_relations.add(
                (
                    paper_id,
                    neighbor_id,
                    tuple(aliases),
                    strength,
                )
            )
        if checked_items != sorted(
            checked_items,
            key=lambda item: (
                -item["strength"],
                item["paper_id"],
                item["aliases"],
            ),
        ):
            raise ValueError("method graph neighbors are not deterministic")
        neighbors[paper_id] = tuple(checked_items)
    for paper_id, neighbor_id, aliases, strength in directed_relations:
        if (
            neighbor_id,
            paper_id,
            aliases,
            strength,
        ) not in directed_relations:
            raise ValueError("method graph relation is not undirected")
    return dict(sorted(owners.items())), neighbors
