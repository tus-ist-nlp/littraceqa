#!/usr/bin/env python3
"""Run a bounded, gold-blind BGE-M3 paper-retrieval comparison.

The dense path embeds exactly one observed ``title_abstract`` Chunk per paper.
Image-only Figure Chunks remain available to the frozen BM25 index but are not
invented as text inputs for BGE-M3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from litqa.contracts import Chunk, Query, RetrievalResult
from litqa.index.bge_m3_index import BGEM3NumpyIndex
from litqa.index.bm25_index import BM25Index
from litqa.index.paper_bm25 import PaperBM25Index
from litqa.retrieve.paper_rank_rrf import PaperRankRRFFuser

import run_paper_rank_followup as sparse_followup


ALLOWED_PAPER_COUNTS = sparse_followup.ALLOWED_PAPER_COUNTS
DEFAULT_PAPER_COUNT = sparse_followup.DEFAULT_PAPER_COUNT
MAX_PAPERS = sparse_followup.MAX_PAPERS
MAX_QUERIES = 55
MAX_CHUNK_RECORDS = 100_000
MAX_CHUNK_FILE_BYTES = 256 * 1024 * 1024
MAX_MODEL_FILES = 100
MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024
CANDIDATE_DEPTH = 100
PAPER_TOP_K = 20
RRF_K = 60
BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_MAX_LENGTH = 512
BGE_BATCH_SIZE = 1
BGE_DEVICE = "cpu"

METHOD_ORDER = (
    "mineru_v1_paper_rank_rrf_fill20_d100",
    "bge_m3_title_abstract_dense",
    "bm25_bge_m3_title_abstract_rrf",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield record


def load_title_abstract_chunks(
    path: Path,
    expected_papers: int = DEFAULT_PAPER_COUNT,
) -> tuple[list[Chunk], list[str], dict[str, int]]:
    """Stream a bounded common-Chunk file and select one observed paper view."""
    if expected_papers not in ALLOWED_PAPER_COUNTS:
        raise ValueError(
            f"expected_papers must be one of {sorted(ALLOWED_PAPER_COUNTS)}"
        )
    if path.stat().st_size > MAX_CHUNK_FILE_BYTES:
        raise ValueError(f"Chunk JSONL exceeds the bounded byte limit: {path}")
    paper_ids: list[str] = []
    seen_papers: set[str] = set()
    title_chunks: dict[str, Chunk] = {}
    modalities: Counter[str] = Counter()
    record_count = 0

    for record in _iter_jsonl(path):
        record_count += 1
        if record_count > MAX_CHUNK_RECORDS:
            raise ValueError(
                f"Chunk JSONL exceeds the {MAX_CHUNK_RECORDS}-record safety cap"
            )
        paper_id = str(record.get("paper_id") or "").strip()
        chunk_type = str(record.get("chunk_type") or "").strip()
        if not paper_id or not chunk_type:
            raise ValueError(f"Chunk record has no paper_id or chunk_type: {path}")
        if paper_id not in seen_papers:
            seen_papers.add(paper_id)
            paper_ids.append(paper_id)
            if len(paper_ids) > expected_papers:
                raise ValueError(
                    f"dense comparison is capped at {expected_papers} papers"
                )
        modalities[chunk_type] += 1
        if chunk_type != "title_abstract":
            continue
        if paper_id in title_chunks:
            raise ValueError(f"paper has more than one title_abstract Chunk: {paper_id}")
        chunk = Chunk(**record)
        if not chunk.chunk_id or not chunk.text.strip():
            raise ValueError(f"empty title_abstract Chunk for paper: {paper_id}")
        title_chunks[paper_id] = chunk

    if len(paper_ids) != expected_papers:
        raise ValueError(
            f"dense comparison requires exactly {expected_papers} papers; "
            f"found {len(paper_ids)}"
        )
    missing = [paper_id for paper_id in paper_ids if paper_id not in title_chunks]
    if missing:
        raise ValueError(f"papers are missing title_abstract Chunks: {missing[:5]}")
    selected = [title_chunks[paper_id] for paper_id in paper_ids]
    return selected, paper_ids, {
        "chunk_record_count": record_count,
        **{f"chunk_type_{name}": count for name, count in sorted(modalities.items())},
    }


def _selected_chunks_sha256(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        payload = json.dumps(
            chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _model_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"local BGE-M3 snapshot does not exist: {path}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if len(files) >= MAX_MODEL_FILES:
            raise ValueError(f"model snapshot exceeds {MAX_MODEL_FILES} files")
        resolved = item.resolve(strict=True)
        size = resolved.stat().st_size
        total_bytes += size
        if total_bytes > MAX_MODEL_BYTES:
            raise ValueError("model snapshot exceeds the bounded byte limit")
        record: dict[str, Any] = {
            "path": str(item.relative_to(path)),
            "bytes": size,
            "resolved_blob": resolved.name,
        }
        if item.is_symlink():
            record["symlink_target"] = os.readlink(item)
        else:
            record["sha256"] = _sha256_file(item)
        files.append(record)
    if not files:
        raise ValueError(f"model snapshot contains no files: {path}")
    files.sort(key=lambda value: value["path"])
    digest = hashlib.sha256()
    for record in files:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "path": str(path),
        "file_count": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy",
        "bm25s",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _code_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        repo_root / "litqa/contracts.py",
        repo_root / "litqa/index/bge_m3_index.py",
        repo_root / "litqa/index/bm25_index.py",
        repo_root / "litqa/index/paper_bm25.py",
        repo_root / "litqa/retrieve/paper_rank_rrf.py",
        repo_root / "scripts/run_paper_rank_followup.py",
        Path(__file__).resolve(),
    )
    return {
        str(path.relative_to(repo_root)): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _build_fingerprint(
    chunks_path: Path,
    chunks: list[Chunk],
    model_path: Path,
    model_snapshot: dict[str, Any],
    code_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "chunks": str(chunks_path),
        "chunks_sha256": _sha256_file(chunks_path),
        "selected_chunks_sha256": _selected_chunks_sha256(chunks),
        "selected_chunk_count": len(chunks),
        "model": BGE_MODEL,
        "revision": BGE_REVISION,
        "model_path": str(model_path),
        "model_snapshot_tree_sha256": model_snapshot["tree_sha256"],
        "model_snapshot_bytes": model_snapshot["bytes"],
        "batch_size": BGE_BATCH_SIZE,
        "device": BGE_DEVICE,
        "max_length": BGE_MAX_LENGTH,
        "normalize_embeddings": True,
        "pooling": "model-provided CLS pooling",
        "similarity": "exact NumPy inner product",
        "code_sha256": code_hashes,
    }


def _prepare_dense_index(
    work_dir: Path,
    chunks: list[Chunk],
    fingerprint: dict[str, Any],
    model_path: Path,
    resume: bool,
) -> tuple[BGEM3NumpyIndex, dict[str, Any]]:
    manifest_path = work_dir / "build_manifest.json"
    index_dir = work_dir / "index"
    started = time.perf_counter()

    if resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires an existing build_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success" or manifest.get("fingerprint") != fingerprint:
            raise ValueError("saved dense build fingerprint does not match this run")
        if not index_dir.is_dir():
            raise ValueError("saved dense index directory is missing")
        observed = sparse_followup._directory_snapshot(index_dir)
        expected = manifest.get("index_snapshot", {})
        if (
            observed.get("tree_sha256") != expected.get("tree_sha256")
            or observed.get("bytes") != expected.get("bytes")
        ):
            raise ValueError("saved dense index checksum does not match its manifest")
        index = _new_dense_index(index_dir, model_path)
        index.load()
        return index, {
            "action": "resumed",
            "seconds": time.perf_counter() - started,
            "index_snapshot": observed,
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        manifest_path,
        {"schema_version": 1, "status": "running", "fingerprint": fingerprint},
    )
    try:
        index = _new_dense_index(index_dir, model_path)
        index.build(chunks)
        snapshot = sparse_followup._directory_snapshot(index_dir)
        manifest = {
            "schema_version": 1,
            "status": "success",
            "fingerprint": fingerprint,
            "index_snapshot": snapshot,
        }
        _atomic_json(manifest_path, manifest)
    except KeyboardInterrupt:
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "interrupted",
                "fingerprint": fingerprint,
                "error": "KeyboardInterrupt: dense index build was interrupted",
            },
        )
        raise
    except Exception as exc:
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "failed",
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    return index, {
        "action": "built",
        "seconds": time.perf_counter() - started,
        "index_snapshot": snapshot,
    }


def _new_dense_index(
    index_dir: Path,
    model_path: Path,
) -> BGEM3NumpyIndex:
    return BGEM3NumpyIndex(
        index_dir=str(index_dir),
        model=BGE_MODEL,
        revision=BGE_REVISION,
        model_path=str(model_path),
        batch_size=BGE_BATCH_SIZE,
        device=BGE_DEVICE,
        max_length=BGE_MAX_LENGTH,
        local_files_only=True,
        include_chunk_types=["title_abstract"],
    )


def _validate_resume_work_directory(path: Path, protected_roots: list[Path]) -> Path:
    """Require an existing resume tree outside every protected input."""
    resolved = path.expanduser().resolve()
    for root in protected_roots:
        protected = root.expanduser().resolve()
        if sparse_followup._is_within(  # noqa: SLF001
            resolved, protected
        ) or sparse_followup._is_within(protected, resolved):  # noqa: SLF001
            raise ValueError(
                f"dense work directory overlaps protected input {protected}: "
                f"{resolved}"
            )
    if not resolved.is_dir():
        raise ValueError("--resume requires an existing dense work directory")
    return resolved


def _method_config() -> dict[str, dict[str, Any]]:
    return {
        METHOD_ORDER[0]: {
            "runs": ["bm25s", "paper_bm25"],
            "candidate_depth": CANDIDATE_DEPTH,
            "paper_top_k": PAPER_TOP_K,
            "rrf_k": RRF_K,
            "fill_to_top_k": True,
        },
        METHOD_ORDER[1]: {
            "runs": ["bge_m3_numpy"],
            "paper_view": "observed title_abstract Chunk",
            "paper_top_k": PAPER_TOP_K,
        },
        METHOD_ORDER[2]: {
            "runs": ["bm25s", "paper_bm25", "bge_m3_numpy"],
            "candidate_depth": CANDIDATE_DEPTH,
            "paper_top_k": PAPER_TOP_K,
            "rrf_k": RRF_K,
            "weights": {
                "bm25s": 1.0,
                "paper_bm25": 1.0,
                "bge_m3_numpy": 1.0,
            },
            "fill_to_top_k": True,
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    return sparse_followup.percentile(values, fraction)


def _smoke_query_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("query limit must be 2") from exc
    if parsed != 2:
        raise argparse.ArgumentTypeError("query limit must be 2")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--chunk-index-dir", type=Path, required=True)
    parser.add_argument("--paper-index-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-work-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-count",
        type=int,
        choices=sorted(ALLOWED_PAPER_COUNTS),
        default=DEFAULT_PAPER_COUNT,
        help="Expected controlled corpus size.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--query-limit",
        type=_smoke_query_limit,
        help="Use exactly the first 2 queries for the bounded smoke test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    chunks_path = args.chunks.expanduser().resolve()
    chunk_index_dir = args.chunk_index_dir.expanduser().resolve()
    paper_index_dir = args.paper_index_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    queries_path = args.queries.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = args.dense_work_dir.expanduser().resolve()

    for path in (chunks_path, queries_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input file does not exist: {path}")
    for path in (chunk_index_dir, paper_index_dir, model_path):
        if not path.is_dir():
            raise FileNotFoundError(f"required input directory does not exist: {path}")
    if model_path.name != BGE_REVISION:
        raise ValueError(
            "local BGE-M3 snapshot directory must use the pinned revision name"
        )
    if args.query_limit not in (None, 2):
        raise ValueError("--query-limit is fixed at 2 for the bounded smoke test")

    protected = [
        Path("/data2/iseakira"),
        chunks_path,
        chunk_index_dir,
        paper_index_dir,
        model_path,
        queries_path,
    ]
    if args.resume:
        work_dir = _validate_resume_work_directory(work_dir, protected)
        output_dir = sparse_followup.validate_output_directory(
            output_dir, [*protected, work_dir]
        )
    else:
        output_dir = sparse_followup.validate_output_directory(output_dir, protected)
        work_dir = sparse_followup.validate_output_directory(
            work_dir, [*protected, output_dir]
        )
    if output_dir == work_dir:
        raise ValueError("output and dense work directories must differ")

    title_chunks, paper_ids, modality_counts = load_title_abstract_chunks(
        chunks_path,
        expected_papers=args.paper_count,
    )
    bm25_papers, bm25_counts = sparse_followup.validate_index_pair(
        chunk_index_dir,
        paper_index_dir,
        expected_papers=args.paper_count,
    )
    if set(paper_ids) != set(bm25_papers):
        raise ValueError("dense and BM25 inputs do not contain the same paper IDs")
    queries = sparse_followup.load_bounded_queries(queries_path, args.query_limit)
    if args.query_limit is None and len(queries) != MAX_QUERIES:
        raise ValueError(
            f"full dense comparison requires exactly {MAX_QUERIES} queries"
        )

    code_hashes = _code_hashes(repo_root)
    model_snapshot = _model_snapshot(model_path)
    chunk_index_snapshot = sparse_followup._directory_snapshot(chunk_index_dir)
    paper_index_snapshot = sparse_followup._directory_snapshot(paper_index_dir)
    chunk_sidecar = chunk_index_dir / "chunks.jsonl"
    if _sha256_file(chunk_sidecar) != _sha256_file(chunks_path):
        raise ValueError("Chunk JSONL does not exactly match the saved BM25 index input")
    fingerprint = _build_fingerprint(
        chunks_path,
        title_chunks,
        model_path,
        model_snapshot,
        code_hashes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir = output_dir / "rankings"
    rankings_dir.mkdir()
    started_unix = time.time()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "gold_blind_search": True,
        "started_unix": started_unix,
        "inputs": {
            "chunks": str(chunks_path),
            "chunks_sha256": fingerprint["chunks_sha256"],
            "selected_chunks_sha256": fingerprint["selected_chunks_sha256"],
            "paper_count": len(paper_ids),
            "paper_ids": paper_ids,
            "selected_dense_chunk_count": len(title_chunks),
            "selected_dense_chunk_type": "title_abstract",
            "selected_dense_char_count": sum(len(chunk.text) for chunk in title_chunks),
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
            "modality_counts": modality_counts,
            "chunk_index": chunk_index_snapshot,
            "paper_index": paper_index_snapshot,
            **bm25_counts,
        },
        "model": {
            **model_snapshot,
            "model_id": BGE_MODEL,
            "revision": BGE_REVISION,
            "license": "MIT",
            "dimension": 1024,
            "max_length": BGE_MAX_LENGTH,
            "batch_size": BGE_BATCH_SIZE,
            "device": BGE_DEVICE,
            "dtype": "float32",
            "normalize_embeddings": True,
            "query_prefix": "",
            "document_prefix": "",
            "document_format": "common_chunk",
            "pooling": "model-provided CLS pooling",
        },
        "fixed_protocol": {
            "methods": list(METHOD_ORDER),
            "candidate_depth": CANDIDATE_DEPTH,
            "ranked_papers": PAPER_TOP_K,
            "rrf_k": RRF_K,
            "paper_count": args.paper_count,
            "max_papers": MAX_PAPERS,
            "max_queries": MAX_QUERIES,
            "dense_scope": "one observed title_abstract Chunk per paper",
            "full_chunk_dense_executed": False,
            "selection": "three predeclared zero-shot development ablations",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "dependencies": _dependency_versions(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "git": sparse_followup._git_snapshot(repo_root),
            "code_sha256": code_hashes,
        },
        "methods": {
            method: {"config": config}
            for method, config in _method_config().items()
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    preparation_started = time.perf_counter()
    try:
        dense_index, dense_preparation = _prepare_dense_index(
            work_dir,
            title_chunks,
            fingerprint,
            model_path,
            args.resume,
        )
    except KeyboardInterrupt:
        manifest.update(
            {
                "status": "interrupted",
                "finished_unix": time.time(),
                "total_wall_seconds": time.time() - started_unix,
                "error": "KeyboardInterrupt: retrieval run was interrupted",
            }
        )
        _atomic_json(output_dir / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_unix": time.time(),
                "total_wall_seconds": time.time() - started_unix,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(output_dir / "manifest.json", manifest)
        raise
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
    preparation_seconds = time.perf_counter() - preparation_started

    baseline_fuser = PaperRankRRFFuser(
        k=RRF_K,
        weights={"bm25s": 1.0, "paper_bm25": 1.0},
        budget_source="bm25s",
        fill_to_top_k=True,
    )
    hybrid_fuser = PaperRankRRFFuser(
        k=RRF_K,
        weights={
            "bm25s": 1.0,
            "paper_bm25": 1.0,
            "bge_m3_numpy": 1.0,
        },
        budget_source="bm25s",
        fill_to_top_k=True,
    )

    rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    latencies: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    component_latencies: dict[str, list[float]] = {
        "chunk_bm25": [],
        "paper_bm25": [],
        "bge_m3_query": [],
        "baseline_fusion": [],
        "hybrid_fusion": [],
    }
    failures: list[dict[str, str]] = []
    method_failed_query_ids: dict[str, set[str]] = {
        method: set() for method in METHOD_ORDER
    }
    search_started = time.perf_counter()
    for query in queries:
        component_errors: dict[str, str] = {}
        chunk_run: list[RetrievalResult] | None = None
        paper_run: list[RetrievalResult] | None = None
        dense_run: list[RetrievalResult] | None = None
        chunk_seconds = 0.0
        paper_seconds = 0.0
        dense_seconds = 0.0

        try:
            started = time.perf_counter()
            chunk_run = chunk_index.search(query.question, CANDIDATE_DEPTH)
            chunk_seconds = time.perf_counter() - started
            component_latencies["chunk_bm25"].append(chunk_seconds)
        except Exception as exc:
            component_errors["chunk_bm25"] = f"{type(exc).__name__}: {exc}"

        try:
            started = time.perf_counter()
            paper_run = paper_index.search(query.question, CANDIDATE_DEPTH)
            paper_seconds = time.perf_counter() - started
            component_latencies["paper_bm25"].append(paper_seconds)
        except Exception as exc:
            component_errors["paper_bm25"] = f"{type(exc).__name__}: {exc}"

        try:
            started = time.perf_counter()
            dense_run = dense_index.search(query.question, CANDIDATE_DEPTH)
            dense_seconds = time.perf_counter() - started
            component_latencies["bge_m3_query"].append(dense_seconds)
        except Exception as exc:
            component_errors["bge_m3_query"] = f"{type(exc).__name__}: {exc}"

        outputs: dict[str, tuple[list[RetrievalResult], float]] = {}
        if chunk_run is not None and paper_run is not None:
            try:
                started = time.perf_counter()
                baseline = baseline_fuser.fuse(
                    [chunk_run, paper_run], top_k=PAPER_TOP_K
                )
                baseline_fusion_seconds = time.perf_counter() - started
                component_latencies["baseline_fusion"].append(
                    baseline_fusion_seconds
                )
                outputs[METHOD_ORDER[0]] = (
                    baseline,
                    chunk_seconds + paper_seconds + baseline_fusion_seconds,
                )
            except Exception as exc:
                component_errors["baseline_fusion"] = f"{type(exc).__name__}: {exc}"

        if dense_run is not None:
            outputs[METHOD_ORDER[1]] = (dense_run[:PAPER_TOP_K], dense_seconds)

        if chunk_run is not None and paper_run is not None and dense_run is not None:
            try:
                started = time.perf_counter()
                hybrid = hybrid_fuser.fuse(
                    [chunk_run, paper_run, dense_run], top_k=PAPER_TOP_K
                )
                hybrid_fusion_seconds = time.perf_counter() - started
                component_latencies["hybrid_fusion"].append(hybrid_fusion_seconds)
                outputs[METHOD_ORDER[2]] = (
                    hybrid,
                    chunk_seconds
                    + paper_seconds
                    + dense_seconds
                    + hybrid_fusion_seconds,
                )
            except Exception as exc:
                component_errors["hybrid_fusion"] = f"{type(exc).__name__}: {exc}"

        method_dependencies = {
            METHOD_ORDER[0]: ("chunk_bm25", "paper_bm25", "baseline_fusion"),
            METHOD_ORDER[1]: ("bge_m3_query",),
            METHOD_ORDER[2]: (
                "chunk_bm25",
                "paper_bm25",
                "bge_m3_query",
                "hybrid_fusion",
            ),
        }
        for method in METHOD_ORDER:
            if method in outputs:
                results, elapsed = outputs[method]
                rows[method].append(
                    sparse_followup._ranking_record(query, results, elapsed)
                )
                latencies[method].append(elapsed)
                continue

            relevant_errors = {
                stage: component_errors[stage]
                for stage in method_dependencies[method]
                if stage in component_errors
            }
            failures.append(
                {
                    "query_id": query.query_id,
                    "method": method,
                    "stage": ",".join(relevant_errors) or "unknown",
                    "error": "; ".join(
                        f"{stage}: {error}"
                        for stage, error in relevant_errors.items()
                    )
                    or "method produced no ranking",
                }
            )
            method_failed_query_ids[method].add(query.query_id)
            rows[method].append(
                {"query_id": query.query_id, "papers": [], "search_seconds": 0.0}
            )
    search_seconds = time.perf_counter() - search_started

    for method in METHOD_ORDER:
        ranking_path = rankings_dir / f"{method}.jsonl"
        _atomic_jsonl(ranking_path, rows[method])
        counts = [len(row["papers"]) for row in rows[method]]
        values = latencies[method]
        manifest["methods"][method].update(
            {
                "status": (
                    "success" if not method_failed_query_ids[method] else "partial"
                ),
                "successful_query_count": len(queries)
                - len(method_failed_query_ids[method]),
                "failed_query_count": len(method_failed_query_ids[method]),
                "ranking_file": str(ranking_path),
                "ranking_sha256": _sha256_file(ranking_path),
                "resources": {
                    "query_seconds_mean": statistics.fmean(values) if values else 0.0,
                    "query_seconds_p50": _percentile(values, 0.50),
                    "query_seconds_p95": _percentile(values, 0.95),
                    "search_total_seconds": sum(values),
                    "ranked_papers_min": min(counts),
                    "ranked_papers_mean": statistics.fmean(counts),
                    "ranked_papers_max": max(counts),
                },
            }
        )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    failed_query_ids = {
        query_id
        for query_ids in method_failed_query_ids.values()
        for query_id in query_ids
    }
    manifest.update(
        {
            "status": "success" if not failures else "partial",
            "finished_unix": time.time(),
            "total_wall_seconds": time.time() - started_unix,
            "successful_query_count": len(queries) - len(failed_query_ids),
            "failed_query_count": len(failed_query_ids),
            "method_failure_count": len(failures),
            "failures": failures,
            "resources": {
                "preparation_seconds": preparation_seconds,
                "dense_index": dense_preparation,
                "chunk_index_load_seconds": chunk_load_seconds,
                "paper_index_load_seconds": paper_load_seconds,
                "search_wall_seconds": search_seconds,
                "process_peak_rss_kib": usage.ru_maxrss,
                "component_query_seconds": {
                    name: {
                        "mean": statistics.fmean(values) if values else 0.0,
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                        "total": sum(values),
                    }
                    for name, values in component_latencies.items()
                },
            },
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
