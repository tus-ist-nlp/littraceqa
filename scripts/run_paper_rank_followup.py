#!/usr/bin/env python3
"""Run a bounded, gold-blind PaperRank-RRF candidate-depth follow-up.

This runner reuses existing BM25 indexes and writes rankings for exactly three
predeclared variants. It cannot accept gold labels or build indexes, and it
accepts only the predeclared 100- or 200-paper controlled corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from litqa.contracts import Query, RetrievalResult
from litqa.index.bm25_index import BM25Index
from litqa.index.paper_bm25 import PaperBM25Index
from litqa.retrieve.paper_rank_rrf import PaperRankRRFFuser


ALLOWED_PAPER_COUNTS = frozenset({100, 200})
DEFAULT_PAPER_COUNT = 100
MAX_PAPERS = max(ALLOWED_PAPER_COUNTS)
MAX_QUERIES = 55
MAX_INDEX_RECORDS = 100_000
MAX_INDEX_FILES = 1_000
MAX_QUERY_FILE_BYTES = 16 * 1024 * 1024
MAX_INDEX_SIDECAR_BYTES = 256 * 1024 * 1024
MAX_TOTAL_INDEX_BYTES = 512 * 1024 * 1024
PAPER_TOP_K = 20
RRF_K = 60
ALLOWED_CANDIDATE_DEPTHS = frozenset({100, 1000})
INDEX_CHUNKS_FILENAME = "chunks.jsonl"

METHOD_ORDER = (
    "mineru_v1_paper_rank_rrf",
    "mineru_v1_paper_rank_rrf_fill20_d100",
    "mineru_v1_paper_rank_rrf_fill20_d1000",
)
METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "mineru_v1_paper_rank_rrf": {
        "candidate_depth": 100,
        "fill_to_top_k": False,
        "fusion_top_k": 100,
        "final_top_k": PAPER_TOP_K,
        "purpose": "replicate the existing depth-100 PaperRank-RRF behavior",
    },
    "mineru_v1_paper_rank_rrf_fill20_d100": {
        "candidate_depth": 100,
        "fill_to_top_k": True,
        "fusion_top_k": PAPER_TOP_K,
        "final_top_k": PAPER_TOP_K,
        "purpose": "isolate fixed top-20 backfill at the existing depth",
    },
    "mineru_v1_paper_rank_rrf_fill20_d1000": {
        "candidate_depth": 1000,
        "fill_to_top_k": True,
        "fusion_top_k": PAPER_TOP_K,
        "final_top_k": PAPER_TOP_K,
        "purpose": "test deeper chunk candidates with fixed top-20 backfill",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects without retaining the complete input file."""
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
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield record


def _check_file_size(path: Path, maximum: int, description: str) -> None:
    size = path.stat().st_size
    if size > maximum:
        raise ValueError(
            f"{description} exceeds the bounded size limit "
            f"({size} > {maximum} bytes): {path}"
        )


def load_bounded_queries(path: Path, limit: int | None = None) -> list[Query]:
    """Load production query fields only and enforce the 55-query cap."""
    if limit is not None and not 1 <= limit <= MAX_QUERIES:
        raise ValueError(f"query limit must be from 1 to {MAX_QUERIES}")
    _check_file_size(path, MAX_QUERY_FILE_BYTES, "query JSONL")

    queries: list[Query] = []
    seen: set[str] = set()
    for record in _iter_jsonl(path):
        query_id = str(record.get("query_id") or "").strip()
        question = str(record.get("question") or "").strip()
        answer_types = record.get("answer_types")
        table_schema = record.get("table_schema", [])
        if (
            not query_id
            or not question
            or not isinstance(answer_types, list)
            or not isinstance(table_schema, list)
        ):
            raise ValueError(
                "each query needs query_id, question, list answer_types, and "
                f"list table_schema: {path}"
            )
        if query_id in seen:
            raise ValueError(f"duplicate query_id in {path}: {query_id}")
        seen.add(query_id)
        queries.append(
            Query(
                query_id=query_id,
                question=question,
                answer_types=[str(item) for item in answer_types],
                table_schema=table_schema,
                multiple_choice_options=record.get("multiple_choice_options"),
            )
        )
        if limit is not None and len(queries) >= limit:
            break
        if len(queries) > MAX_QUERIES:
            raise ValueError(
                f"the PaperRank follow-up is capped at {MAX_QUERIES} queries"
            )

    if not queries:
        raise ValueError(f"no queries found in {path}")
    return queries


def inspect_index_papers(
    index_dir: Path,
    *,
    require_one_record_per_paper: bool,
    expected_papers: int = DEFAULT_PAPER_COUNT,
) -> tuple[list[str], int]:
    """Inspect saved metadata and require one predeclared corpus size."""
    if expected_papers not in ALLOWED_PAPER_COUNTS:
        raise ValueError(
            f"expected_papers must be one of {sorted(ALLOWED_PAPER_COUNTS)}"
        )
    chunks_path = index_dir / INDEX_CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"saved index metadata is missing: {chunks_path}")
    _check_file_size(
        chunks_path,
        MAX_INDEX_SIDECAR_BYTES,
        "saved index Chunk sidecar",
    )

    paper_ids: list[str] = []
    seen: set[str] = set()
    record_count = 0
    for record in _iter_jsonl(chunks_path):
        record_count += 1
        if record_count > MAX_INDEX_RECORDS:
            raise ValueError(
                f"saved index exceeds the {MAX_INDEX_RECORDS}-record safety "
                f"cap: {index_dir}"
            )
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError(f"saved index record has no paper_id: {chunks_path}")
        if paper_id not in seen:
            seen.add(paper_id)
            paper_ids.append(paper_id)
            if len(paper_ids) > expected_papers:
                raise ValueError(
                    f"saved index exceeds the {expected_papers}-paper safety cap: "
                    f"{index_dir}"
                )

    if len(paper_ids) != expected_papers:
        raise ValueError(
            f"saved index must contain exactly {expected_papers} papers; "
            f"found {len(paper_ids)} in {index_dir}"
        )
    if require_one_record_per_paper and record_count != len(paper_ids):
        raise ValueError(
            "paper-level index must contain exactly one record per paper: "
            f"{index_dir}"
        )
    return paper_ids, record_count


def validate_index_pair(
    chunk_index_dir: Path,
    paper_index_dir: Path,
    expected_papers: int = DEFAULT_PAPER_COUNT,
) -> tuple[list[str], dict[str, int]]:
    """Require matching predeclared corpora in both saved indexes."""
    chunk_papers, chunk_count = inspect_index_papers(
        chunk_index_dir,
        require_one_record_per_paper=False,
        expected_papers=expected_papers,
    )
    paper_papers, paper_count = inspect_index_papers(
        paper_index_dir,
        require_one_record_per_paper=True,
        expected_papers=expected_papers,
    )
    if set(chunk_papers) != set(paper_papers):
        only_chunk = sorted(set(chunk_papers) - set(paper_papers))
        only_paper = sorted(set(paper_papers) - set(chunk_papers))
        raise ValueError(
            "chunk and paper indexes contain different paper IDs; "
            f"chunk_only={only_chunk[:5]}, paper_only={only_paper[:5]}"
        )
    return chunk_papers, {
        "chunk_index_record_count": chunk_count,
        "paper_index_record_count": paper_count,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_directory(path: Path, protected_roots: Iterable[Path]) -> Path:
    """Require a new or empty output tree outside every protected input."""
    resolved = path.expanduser().resolve()
    for root in protected_roots:
        protected = root.expanduser().resolve()
        if _is_within(resolved, protected) or _is_within(protected, resolved):
            raise ValueError(
                f"output directory overlaps protected input {protected}: {resolved}"
            )
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {resolved}")
    return resolved


def _directory_snapshot(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Return deterministic per-file hashes for one bounded saved index."""
    byte_limit = MAX_TOTAL_INDEX_BYTES if max_bytes is None else max_bytes
    if byte_limit < 0:
        raise ValueError("saved index byte limit must be non-negative")
    bounded_files: list[tuple[Path, int]] = []
    total_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if len(bounded_files) >= MAX_INDEX_FILES:
            raise ValueError(
                f"saved index exceeds the {MAX_INDEX_FILES}-file safety cap: {path}"
            )
        size = item.stat().st_size
        total_bytes += size
        if total_bytes > byte_limit:
            raise ValueError(
                "saved index exceeds the bounded byte limit "
                f"({total_bytes} > {byte_limit}): {path}"
            )
        bounded_files.append((item, size))

    files = []
    for item, size in sorted(bounded_files):
        files.append(
            {
                "path": str(item.relative_to(path)),
                "bytes": size,
                "sha256": _sha256_file(item),
            }
        )
    if not files:
        raise ValueError(f"saved index contains no files: {path}")
    digest = hashlib.sha256()
    for record in files:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "file_count": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile for a non-empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status_porcelain": run("status", "--porcelain"),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "numpy", "scipy", "PyStemmer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _code_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        repo_root / "litqa/contracts.py",
        repo_root / "litqa/index/bm25_index.py",
        repo_root / "litqa/index/paper_bm25.py",
        repo_root / "litqa/retrieve/paper_rank_rrf.py",
        Path(__file__).resolve(),
    )
    return {
        str(path.relative_to(repo_root)): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _ranking_record(
    query: Query,
    results: list[RetrievalResult],
    search_seconds: float,
) -> dict[str, Any]:
    papers = []
    seen: set[str] = set()
    for result in results:
        if result.paper_id in seen:
            continue
        seen.add(result.paper_id)
        papers.append(
            {
                "rank": len(papers) + 1,
                "paper_id": result.paper_id,
                "score": result.score,
                "representative_chunk_id": result.chunk_id,
                "source": result.source,
                "chunk_type": result.chunk_type,
                "page": result.metadata.get("page"),
                "figure_id": result.metadata.get("figure_id"),
                "table_id": result.metadata.get("table_id"),
            }
        )
        if len(papers) >= PAPER_TOP_K:
            break
    return {
        "query_id": query.query_id,
        "papers": papers,
        "search_seconds": search_seconds,
    }


def _search_variant(
    question: str,
    chunk_index: BM25Index,
    paper_index: PaperBM25Index,
    config: dict[str, Any],
) -> list[RetrievalResult]:
    depth = config["candidate_depth"]
    if depth not in ALLOWED_CANDIDATE_DEPTHS:
        raise ValueError(f"unsupported fixed candidate depth: {depth}")
    chunk_run = chunk_index.search(question, depth)
    paper_run = paper_index.search(question, depth)
    fuser = PaperRankRRFFuser(
        k=RRF_K,
        weights={"bm25s": 1.0, "paper_bm25": 1.0},
        budget_source="bm25s",
        fill_to_top_k=config["fill_to_top_k"],
    )
    return fuser.fuse(
        [chunk_run, paper_run], top_k=config["fusion_top_k"]
    )[: config["final_top_k"]]


def _run_method(
    method: str,
    config: dict[str, Any],
    queries: list[Query],
    chunk_index: BM25Index,
    paper_index: PaperBM25Index,
    index_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    ranked_paper_counts: list[int] = []

    for query in queries:
        query_started = time.perf_counter()
        results = _search_variant(
            query.question, chunk_index, paper_index, config
        )
        elapsed = time.perf_counter() - query_started
        row = _ranking_record(query, results, elapsed)
        rows.append(row)
        latencies.append(elapsed)
        ranked_paper_counts.append(len(row["papers"]))

    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    usage = resource.getrusage(resource.RUSAGE_SELF)
    resources = {
        "method": method,
        "index_build_seconds": 0.0,
        "indexes_reused": True,
        "index_bytes": index_bytes,
        "query_count": len(queries),
        "search_total_seconds": sum(latencies),
        "query_seconds_mean": statistics.fmean(latencies),
        "query_seconds_p50": percentile(latencies, 0.50),
        "query_seconds_p95": percentile(latencies, 0.95),
        "method_wall_seconds": wall_seconds,
        "method_cpu_seconds": cpu_seconds,
        "average_process_cpu_percent": (
            100.0 * cpu_seconds / wall_seconds if wall_seconds else 0.0
        ),
        "process_peak_rss_kib_so_far": usage.ru_maxrss,
        "ranked_papers_min": min(ranked_paper_counts),
        "ranked_papers_mean": statistics.fmean(ranked_paper_counts),
        "ranked_papers_max": max(ranked_paper_counts),
    }
    return rows, resources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-index-dir", type=Path, required=True)
    parser.add_argument("--paper-index-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-count",
        type=int,
        choices=sorted(ALLOWED_PAPER_COUNTS),
        default=DEFAULT_PAPER_COUNT,
        help="Expected controlled corpus size.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        help=f"Use only the first N queries for a smoke test (maximum {MAX_QUERIES}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    chunk_index_dir = args.chunk_index_dir.expanduser().resolve()
    paper_index_dir = args.paper_index_dir.expanduser().resolve()
    queries_path = args.queries.expanduser().resolve()
    if not chunk_index_dir.is_dir() or not paper_index_dir.is_dir():
        raise FileNotFoundError("both saved index directories must exist")
    if not queries_path.is_file():
        raise FileNotFoundError("--queries must be an existing file")

    output_dir = validate_output_directory(
        args.output_dir,
        [Path("/data2/iseakira"), chunk_index_dir, paper_index_dir, queries_path],
    )
    paper_ids, index_record_counts = validate_index_pair(
        chunk_index_dir,
        paper_index_dir,
        expected_papers=args.paper_count,
    )
    queries = load_bounded_queries(queries_path, args.query_limit)
    if args.query_limit is None and len(queries) != MAX_QUERIES:
        raise ValueError(
            f"the full follow-up requires exactly {MAX_QUERIES} queries; "
            f"received {len(queries)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir = output_dir / "rankings"
    rankings_dir.mkdir()
    chunk_snapshot = _directory_snapshot(chunk_index_dir)
    paper_snapshot = _directory_snapshot(
        paper_index_dir,
        max_bytes=MAX_TOTAL_INDEX_BYTES - chunk_snapshot["bytes"],
    )
    if chunk_snapshot["bytes"] + paper_snapshot["bytes"] > MAX_TOTAL_INDEX_BYTES:
        raise ValueError(
            "combined saved indexes exceed the bounded byte limit "
            f"({MAX_TOTAL_INDEX_BYTES} bytes)"
        )

    load_started = time.perf_counter()
    chunk_load_started = time.perf_counter()
    chunk_index = BM25Index(index_dir=str(chunk_index_dir))
    chunk_index.load()
    chunk_load_seconds = time.perf_counter() - chunk_load_started
    paper_load_started = time.perf_counter()
    paper_index = PaperBM25Index(
        index_dir=str(paper_index_dir),
        exclude_references=False,
    )
    paper_index.load()
    paper_load_seconds = time.perf_counter() - paper_load_started
    total_load_seconds = time.perf_counter() - load_started

    started_at = time.time()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "gold_blind_search": True,
        "started_unix": started_at,
        "inputs": {
            "queries": str(queries_path),
            "queries_sha256": _sha256_file(queries_path),
            "query_count": len(queries),
            "query_ids": [query.query_id for query in queries],
            "query_fields_loaded": [
                "query_id",
                "question",
                "answer_types",
                "table_schema",
                "multiple_choice_options",
            ],
            "retrieval_text_field": "question",
            "paper_count": len(paper_ids),
            "paper_ids": paper_ids,
            **index_record_counts,
            "chunk_index": chunk_snapshot,
            "paper_index": paper_snapshot,
        },
        "fixed_protocol": {
            "methods": list(METHOD_ORDER),
            "candidate_depths": sorted(ALLOWED_CANDIDATE_DEPTHS),
            "ranked_papers": PAPER_TOP_K,
            "rrf_k": RRF_K,
            "paper_count": args.paper_count,
            "max_papers": MAX_PAPERS,
            "max_queries": MAX_QUERIES,
            "selection": "three predeclared gold-blind follow-up variants",
            "query_mode": (
                "bounded_smoke_subset"
                if args.query_limit is not None
                else "full_55_query_comparison"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "pid": os.getpid(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "git": _git_snapshot(repo_root),
            "code_sha256": _code_hashes(repo_root),
        },
        "index_loading": {
            "chunk_index_seconds": chunk_load_seconds,
            "paper_index_seconds": paper_load_seconds,
            "total_seconds": total_load_seconds,
        },
        "methods": {
            method: {
                "config": {
                    **METHOD_CONFIGS[method],
                    "chunk_candidate_depth_effective": min(
                        METHOD_CONFIGS[method]["candidate_depth"],
                        index_record_counts["chunk_index_record_count"],
                    ),
                    "paper_candidate_depth_effective": min(
                        METHOD_CONFIGS[method]["candidate_depth"],
                        index_record_counts["paper_index_record_count"],
                    ),
                }
            }
            for method in METHOD_ORDER
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    index_bytes = chunk_snapshot["bytes"] + paper_snapshot["bytes"]
    failures = 0
    for method in METHOD_ORDER:
        method_started = time.perf_counter()
        try:
            rows, resources = _run_method(
                method,
                METHOD_CONFIGS[method],
                queries,
                chunk_index,
                paper_index,
                index_bytes,
            )
            ranking_path = rankings_dir / f"{method}.jsonl"
            _atomic_jsonl(ranking_path, rows)
            manifest["methods"][method].update(
                {
                    "status": "success",
                    "ranking_file": str(ranking_path),
                    "ranking_sha256": _sha256_file(ranking_path),
                    "resources": resources,
                }
            )
        except Exception as exc:  # Keep every predeclared method accountable.
            failures += 1
            manifest["methods"][method].update(
                {
                    "status": "failed",
                    "wall_seconds_before_failure": time.perf_counter()
                    - method_started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _atomic_json(output_dir / "manifest.json", manifest)

    manifest.update(
        {
            "status": "success" if failures == 0 else "partial",
            "finished_unix": time.time(),
            "total_wall_seconds": time.time() - started_at,
            "successful_method_count": len(METHOD_ORDER) - failures,
            "failed_method_count": failures,
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
