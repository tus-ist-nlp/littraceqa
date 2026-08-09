"""Load the inputs shared by paper-selection scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from littraceqa.common import compact_text, read_json, read_jsonl
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


def load_paper_metadata(
    path: Path,
    wanted_ids: set[str],
    *,
    abstract_chars: int,
) -> dict[str, dict[str, Any]]:
    """Read only requested records from a paper metadata JSONL file."""

    remaining = set(wanted_ids)
    if not remaining:
        return {}
    papers: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            paper_id = record.get("paper_id")
            if paper_id not in remaining:
                continue
            paper = {
                "title": record.get("title"),
                "authors": record.get("authors") or [],
                "venue": record.get("venue"),
                "year": record.get("year"),
            }
            if abstract_chars:
                paper["abstract"] = compact_text(
                    record.get("abstract"), max_chars=abstract_chars
                )
            papers[str(paper_id)] = paper
            remaining.remove(paper_id)
            if not remaining:
                break
    return papers
