"""Load the inputs shared by paper-selection scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from littraceqa.common import read_json, read_jsonl
from littraceqa.di_pipeline.index.method_sidecar import METHOD_GRAPH_FILENAME


@dataclass(frozen=True)
class RetrievalRun:
    rankings: dict[str, list[str]]
    method_owner_index_path: Path | None


def _method_owner_index_path(payload: dict[str, Any]) -> Path | None:
    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    run_spec = checkpoint.get("run_spec")
    config = run_spec.get("config") if isinstance(run_spec, dict) else None
    retriever = config.get("retriever") if isinstance(config, dict) else None
    indexers = retriever.get("indexers") if isinstance(retriever, dict) else None
    if not isinstance(indexers, list):
        return None
    paths: list[Path] = []
    for indexer in indexers:
        if not isinstance(indexer, dict) or indexer.get("name") != "paper_bm25":
            continue
        params = indexer.get("params")
        index_dir = params.get("index_dir") if isinstance(params, dict) else None
        if isinstance(index_dir, str) and index_dir:
            paths.append(Path(index_dir) / METHOD_GRAPH_FILENAME)
    return paths[0] if len(paths) == 1 else None


def load_retrieval_run(path: Path) -> RetrievalRun:
    """Read rankings and the optional method-owner sidecar location."""

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    queries = payload.get("queries") or []
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path} contains no queries")
    rankings = {
        str(entry["query_id"]): list(entry.get("ranked_papers") or [])
        for entry in queries
    }
    return RetrievalRun(
        rankings=rankings,
        method_owner_index_path=_method_owner_index_path(payload),
    )


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Read ``query_id -> ranked paper ids`` from a retrieval output."""

    return load_retrieval_run(path).rankings


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Read non-empty query JSONL in file order."""

    records = read_jsonl(path)
    if not records:
        raise ValueError(f"{path} contains no queries")
    return records
