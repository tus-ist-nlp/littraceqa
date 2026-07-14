#!/usr/bin/env python3
"""Run a configured preprocessing, indexing, retrieval, and evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


from litqa.config import build_pipeline, compose_config, load_config
from litqa.contracts import Chunk, Query


_SAFE_PAPER_LIMIT = 200
_MAX_WORKERS = 8
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PREPROCESS_DEPENDENCIES = (
    "marker-pdf",
    "pypdf",
)


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def process_worker_limit(process_cfg: dict[str, Any]) -> int:
    """Return the global or process-specific safety cap for worker threads."""
    configured = process_cfg.get("max_workers", _MAX_WORKERS)
    if (
        not isinstance(configured, int)
        or isinstance(configured, bool)
        or configured <= 0
    ):
        raise ValueError("process max_workers must be a positive integer")
    return min(_MAX_WORKERS, configured)


def build_process_fingerprint(preprocessor_cfg: dict[str, Any]) -> str:
    """Hash resolved preprocessing config, code, Python, and dependencies."""
    preprocess_root = Path(__file__).resolve().parents[1] / "litqa" / "preprocess"
    code_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(preprocess_root.glob("*.py"))
    }
    dependency_versions = {}
    for distribution in _PREPROCESS_DEPENDENCIES:
        try:
            dependency_versions[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    payload = {
        "preprocessor_config": preprocessor_cfg,
        "python": platform.python_version(),
        "dependencies": dependency_versions,
        "preprocess_code": code_hashes,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_paper_input_fingerprint(paper: dict[str, Any], preprocessor: Any) -> str:
    """Hash paper metadata and source artifacts without loading them into memory."""
    digest = hashlib.sha256()
    metadata = json.dumps(
        paper,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest.update(b"metadata\0")
    digest.update(metadata)

    input_paths = getattr(preprocessor, "input_paths", None)
    if input_paths is None:
        return digest.hexdigest()
    if not callable(input_paths):
        raise TypeError("preprocessor input_paths must be callable")

    paths = input_paths(paper)
    if not isinstance(paths, list) or not paths:
        raise ValueError("preprocessor input_paths must return a non-empty list")
    for position, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"preprocessor input is missing or empty: {path}")
        digest.update(f"input:{position}\0{path}\0".encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one artifact without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file without loading the whole file."""
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


def load_papers(path: Path) -> list[dict]:
    """Load paper metadata for the legacy unbounded preprocessing path."""
    return list(iter_jsonl(path))


def load_chunks(path: Path) -> list[Chunk]:
    """Load common chunks from JSONL."""
    return [Chunk(**record) for record in iter_jsonl(path)]


def load_queries(path: Path) -> list[Query]:
    """Load only fields available in production query inputs."""
    queries: list[Query] = []
    seen_query_ids: set[str] = set()
    for record in iter_jsonl(path):
        query_id = record.get("query_id")
        question = record.get("question")
        answer_types = record.get("answer_types")
        table_schema = record.get("table_schema")
        multiple_choice_options = record.get("multiple_choice_options")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"each query needs a non-empty query_id: {path}")
        if query_id != query_id.strip():
            raise ValueError(f"query_id must not have surrounding whitespace: {path}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"each query needs a non-empty question: {path}")
        if question != question.strip():
            raise ValueError(f"question must not have surrounding whitespace: {path}")
        if not isinstance(answer_types, list):
            raise ValueError(f"each query needs list answer_types: {path}")
        if not answer_types:
            raise ValueError(f"answer_types must not be empty: {path}")
        for index, answer_type in enumerate(answer_types):
            if not isinstance(answer_type, str) or not answer_type.strip():
                raise ValueError(
                    f"answer_types[{index}] must be a non-empty string: {path}"
                )
            if answer_type != answer_type.strip():
                raise ValueError(
                    f"answer_types[{index}] must not have surrounding whitespace: "
                    f"{path}"
                )
        if table_schema is not None and not isinstance(table_schema, list):
            raise ValueError(f"table_schema must be a list when present: {path}")
        for index, column in enumerate(table_schema or []):
            if not isinstance(column, dict):
                raise ValueError(
                    f"table_schema[{index}] must be an object: {path}"
                )
        if multiple_choice_options is not None and not isinstance(
            multiple_choice_options, (dict, list)
        ):
            raise ValueError(
                "multiple_choice_options must be an object or list when present: "
                f"{path}"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id in {path}: {query_id}")
        seen_query_ids.add(query_id)
        queries.append(
            Query(
                query_id=query_id,
                question=question,
                answer_types=answer_types,
                table_schema=table_schema or [],
                multiple_choice_options=multiple_choice_options,
            )
        )
    return queries


def validate_paper_id(value: object, context: str) -> str:
    """Return a safe paper ID or reject invalid metadata before processing."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} needs a non-empty string paper_id")
    if value != value.strip():
        raise ValueError(f"{context} paper_id must not have surrounding whitespace")
    if not _SAFE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{context} has unsafe paper_id: {value!r}")
    return value


def read_requested_paper_ids(
    command_line_ids: Iterable[str], paper_ids_file: Path | None
) -> list[str]:
    """Read and validate requested IDs, rejecting duplicates."""
    values = [value.strip() for value in command_line_ids if value.strip()]
    if paper_ids_file is not None:
        with paper_ids_file.open(encoding="utf-8") as handle:
            values.extend(line.strip() for line in handle if line.strip())
    validated: list[str] = []
    seen: set[str] = set()
    for value in values:
        paper_id = validate_paper_id(value, "requested paper")
        if paper_id in seen:
            raise ValueError(f"duplicate requested paper_id: {paper_id}")
        seen.add(paper_id)
        validated.append(paper_id)
    return validated


def select_papers(
    metadata_path: Path,
    requested_ids: list[str],
    limit: int | None,
) -> tuple[list[dict], list[str], list[dict[str, Any]]]:
    """Select a bounded paper list while streaming the metadata file once."""
    if requested_ids and limit is not None and len(requested_ids) > limit:
        raise ValueError(
            f"{len(requested_ids)} paper IDs were requested but --limit is {limit}"
        )

    if requested_ids:
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("duplicate requested paper_id")
        for paper_id in requested_ids:
            validate_paper_id(paper_id, "requested paper")
        wanted = set(requested_ids)
        found: dict[str, dict] = {}
        for paper in iter_jsonl(metadata_path):
            raw_paper_id = paper.get("paper_id")
            if not isinstance(raw_paper_id, str):
                continue
            paper_id = raw_paper_id
            if paper_id in wanted and paper_id not in found:
                found[paper_id] = paper
                if len(found) == len(wanted):
                    break
        selected = [found[paper_id] for paper_id in requested_ids if paper_id in found]
        missing = [paper_id for paper_id in requested_ids if paper_id not in found]
        return selected, missing, []

    selected: list[dict] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    records: Iterable[dict[str, Any]] = iter_jsonl(metadata_path)
    if limit is not None:
        records = islice(records, limit)
    for record_number, paper in enumerate(records, start=1):
        raw_paper_id = paper.get("paper_id")
        try:
            paper_id = validate_paper_id(
                raw_paper_id, f"{metadata_path} record {record_number}"
            )
        except ValueError as exc:
            rejected.append(
                {
                    "paper_id": "" if raw_paper_id is None else str(raw_paper_id),
                    "status": "failed",
                    "error_type": "InvalidPaperMetadata",
                    "error": str(exc),
                }
            )
            continue
        if paper_id in seen:
            rejected.append(
                {
                    "paper_id": paper_id,
                    "status": "failed",
                    "error_type": "DuplicatePaperMetadata",
                    "error": "duplicate paper_id in bounded metadata selection",
                }
            )
            continue
        seen.add(paper_id)
        selected.append(paper)
    return selected, [], rejected


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_artifact_root(
    artifact_root: Path,
    source_root: Path,
    read_only_roots: Iterable[Path],
) -> Path:
    """Reject output locations that overlap an input or another read-only tree."""
    artifact = artifact_root.expanduser().resolve()
    source = source_root.expanduser().resolve()
    protected = [source, *(root.expanduser().resolve() for root in read_only_roots)]
    for root in protected:
        if _is_within(artifact, root) or _is_within(root, artifact):
            raise ValueError(f"artifact root overlaps read-only input {root}: {artifact}")
    return artifact


def validate_artifact_path(path: Path, artifact_root: Path) -> Path:
    """Reject an artifact path whose existing components use symlink escapes."""
    root = artifact_root.expanduser().resolve()
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside artifact root: {candidate}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {current}")
        if current.exists() and not _is_within(current.resolve(), root):
            raise ValueError(f"artifact path escapes artifact root: {current}")
    return candidate


def require_bounded_artifact_root(
    process_name: str, artifact_root: Path | None
) -> Path:
    """Require an isolated artifact root before constructing bounded backends."""
    if artifact_root is None:
        raise ValueError(
            f"{process_name} requires --artifact-root for both build and search"
        )
    return artifact_root


def validate_bounded_index_dirs(cfg: dict[str, Any], artifact_root: Path) -> None:
    """Require every bounded-build index to stay under its artifact root."""
    allowed_root = (artifact_root / "index").resolve()
    for indexer in cfg["retriever"]["indexers"]:
        index_dir = Path(indexer["params"]["index_dir"]).expanduser().resolve()
        if not _is_within(index_dir, allowed_root):
            raise ValueError(
                f"index_dir for {indexer['name']} is outside artifact root: "
                f"{index_dir}"
            )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_chunk_shard_atomic(path: Path, chunks: list[Chunk]) -> None:
    """Write one paper shard and publish it with an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def valid_chunk_shard(path: Path, paper_id: str) -> bool:
    """Return whether an existing shard is non-empty and belongs to one paper."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        chunks = load_chunks(path)
    except (OSError, TypeError, ValueError):
        return False
    return bool(chunks) and all(
        chunk.paper_id == paper_id
        and isinstance(chunk.chunk_id, str)
        and bool(chunk.chunk_id)
        and isinstance(chunk.text, str)
        and isinstance(chunk.chunk_type, str)
        and bool(chunk.chunk_type)
        and isinstance(chunk.metadata, dict)
        for chunk in chunks
    )


def state_matches_fingerprint(
    path: Path,
    process_fingerprint: str | None,
    shard_path: Path | None = None,
    input_fingerprint: str | None = None,
) -> bool:
    """Return whether resume state and shard match the preprocessing build."""
    if process_fingerprint is None and input_fingerprint is None:
        return True
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    matches = isinstance(state, dict) and state.get("status") in {
        "success",
        "partial",
        "skipped",
    }
    if process_fingerprint is not None:
        matches = matches and state.get("process_fingerprint") == process_fingerprint
    if input_fingerprint is not None:
        matches = matches and state.get("input_fingerprint") == input_fingerprint
    if not matches or shard_path is None:
        return matches
    expected_hash = state.get("shard_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        return False
    try:
        actual_hash = _sha256_file(shard_path)
    except OSError:
        return False
    return hmac.compare_digest(expected_hash, actual_hash)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _paper_paths(artifact_root: Path, process_name: str, paper_id: str) -> tuple[Path, Path]:
    if (
        not _SAFE_ID_RE.fullmatch(paper_id)
        or paper_id in {".", ".."}
        or not _SAFE_ID_RE.fullmatch(process_name)
    ):
        raise ValueError(f"unsafe paper or process identifier: {paper_id!r}, {process_name!r}")
    shard = validate_artifact_path(
        artifact_root / "chunks" / process_name / "papers" / f"{paper_id}.jsonl",
        artifact_root,
    )
    state = validate_artifact_path(
        artifact_root / "state" / process_name / f"{paper_id}.json",
        artifact_root,
    )
    return shard, state


def process_one_paper(
    paper: dict,
    preprocessor: Any,
    artifact_root: Path,
    process_name: str,
    resume: bool,
    process_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Process one paper and return a serializable status record."""
    started = time.perf_counter()
    paper_id = str(paper.get("paper_id") or "")
    try:
        shard_path, state_path = _paper_paths(artifact_root, process_name, paper_id)
    except ValueError as exc:
        result = {
            "paper_id": paper_id,
            "status": "failed",
            "chunk_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if process_fingerprint is not None:
            result["process_fingerprint"] = process_fingerprint
        return result

    try:
        input_fingerprint = build_paper_input_fingerprint(paper, preprocessor)
    except Exception as exc:
        result = {
            "paper_id": paper_id,
            "status": "failed",
            "chunk_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if process_fingerprint is not None:
            result["process_fingerprint"] = process_fingerprint
        _write_json_atomic(state_path, result)
        return result

    if (
        resume
        and valid_chunk_shard(shard_path, paper_id)
        and state_matches_fingerprint(
            state_path,
            process_fingerprint,
            shard_path,
            input_fingerprint,
        )
    ):
        existing_chunks = load_chunks(shard_path)
        chunk_count = len(existing_chunks)
        status = (
            "partial"
            if any(
                chunk.metadata.get("preprocess_status") == "partial"
                for chunk in existing_chunks
            )
            else "skipped"
        )
        result = {
            "paper_id": paper_id,
            "status": status,
            "resumed": True,
            "chunk_count": chunk_count,
            "elapsed_seconds": time.perf_counter() - started,
            "shard": str(shard_path),
            "shard_sha256": _sha256_file(shard_path),
            "input_fingerprint": input_fingerprint,
        }
        if process_fingerprint is not None:
            result["process_fingerprint"] = process_fingerprint
        _write_json_atomic(state_path, result)
        return result

    try:
        chunks = preprocessor.process(paper)
        if not chunks:
            raise ValueError("preprocessor returned no chunks")
        if any(chunk.paper_id != paper_id for chunk in chunks):
            raise ValueError("preprocessor returned a chunk with the wrong paper_id")
        write_chunk_shard_atomic(shard_path, chunks)
        status = (
            "partial"
            if any(
                chunk.metadata.get("preprocess_status") == "partial"
                for chunk in chunks
            )
            else "success"
        )
        result = {
            "paper_id": paper_id,
            "status": status,
            "resumed": False,
            "chunk_count": len(chunks),
            "elapsed_seconds": time.perf_counter() - started,
            "shard": str(shard_path),
            "shard_sha256": _sha256_file(shard_path),
            "input_fingerprint": input_fingerprint,
        }
    except Exception as exc:
        result = {
            "paper_id": paper_id,
            "status": "failed",
            "chunk_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "input_fingerprint": input_fingerprint,
        }
    if process_fingerprint is not None:
        result["process_fingerprint"] = process_fingerprint
    _write_json_atomic(state_path, result)
    return result


def _batches(values: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def preprocess_selected_papers(
    papers: list[dict],
    preprocessor: Any,
    artifact_root: Path,
    process_name: str,
    workers: int,
    batch_size: int,
    resume: bool,
    failures_path: Path,
    process_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Process a bounded list in small batches while isolating paper failures."""
    results: list[dict[str, Any]] = []
    for batch in _batches(papers, batch_size):
        if workers == 1:
            batch_results = [
                process_one_paper(
                    paper,
                    preprocessor,
                    artifact_root,
                    process_name,
                    resume,
                    process_fingerprint,
                )
                for paper in batch
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch_results = list(
                    executor.map(
                        lambda paper: process_one_paper(
                            paper,
                            preprocessor,
                            artifact_root,
                            process_name,
                            resume,
                            process_fingerprint,
                        ),
                        batch,
                    )
                )
        for result in batch_results:
            results.append(result)
            if result["status"] == "failed":
                _append_jsonl(failures_path, result)
    return results


def merge_selected_shards(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Merge successful shards in selection order without loading their contents."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for result in results:
            if result["status"] not in {"success", "partial", "skipped"}:
                continue
            shard = Path(result["shard"])
            with shard.open(encoding="utf-8") as source:
                for line in source:
                    destination.write(line)
    temporary.replace(output_path)


def write_predictions_atomic(
    output_path: Path, queries: list[Query], agent: Any
) -> None:
    """Publish predictions only after every query completes successfully."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, query in enumerate(queries, start=1):
                prediction = agent.run(query)
                handle.write(
                    json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n"
                )
                if index % 10 == 0:
                    print(f"completed {index}/{len(queries)} queries")
        temporary.replace(output_path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _default_read_only_roots(source_root: Path) -> list[Path]:
    if source_root.name == "mineru" and source_root.parent.name == "pdfs":
        return [source_root.parent.parent]
    if source_root.name == "pdfs" and source_root.parent.name == "pdfs":
        return [source_root.parent.parent]
    if source_root.name == "pdfs":
        return [source_root.parent]
    return [source_root]


def combine_read_only_roots(
    source_root: Path, additional_roots: Iterable[Path]
) -> list[Path]:
    """Keep automatic owner protection and add explicit read-only roots."""
    combined = [
        *(_default_read_only_roots(source_root)),
        *(root.expanduser().resolve() for root in additional_roots),
    ]
    return list(dict.fromkeys(combined))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold", help="Optional gold JSONL. No evaluation runs when omitted.")
    parser.add_argument("--build", action="store_true", help="Build chunks and indexes")
    parser.add_argument("--mineru-root", help="Read-only MinerU root; overrides the environment")
    parser.add_argument(
        "--pdf-root",
        help="Read-only PDF root; overrides LITTRACEQA_PDF_ROOT and paths.pdf_dir",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Repeatable paper ID")
    parser.add_argument("--paper-ids-file", type=Path, help="One paper ID per line")
    parser.add_argument("--limit", type=positive_int, help="Maximum number of papers")
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse valid per-paper shards",
    )
    parser.add_argument("--artifact-root", type=Path, help="Isolated chunks/index/state root")
    parser.add_argument(
        "--read-only-root",
        action="append",
        type=Path,
        default=[],
        help="Reject artifact output under this path",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    paths_cfg = load_config(args.paths)
    process_cfg = load_config(args.process)
    search_cfg = load_config(args.search)
    agent_cfg = load_config(args.agent)
    process_name = process_cfg["name"]
    process_source = process_cfg.get("source", process_name)
    is_mineru_process = process_source == "mineru"
    is_bounded_process = is_mineru_process or bool(
        process_cfg.get("bounded_build", False)
    )

    artifact_root: Path | None = None
    if is_mineru_process:
        raw_mineru_root = (
            args.mineru_root
            or os.getenv("LITTRACEQA_MINERU_ROOT")
            or paths_cfg.get("mineru_root")
        )
        if not raw_mineru_root:
            parser.error(
                "mineru requires --mineru-root, LITTRACEQA_MINERU_ROOT, "
                "or paths.mineru_root"
            )
        paths_cfg["mineru_root"] = str(
            Path(raw_mineru_root).expanduser().resolve()
        )
    elif process_cfg.get("path_key", "pdf_dir") == "pdf_dir":
        raw_pdf_root = (
            args.pdf_root
            or os.getenv("LITTRACEQA_PDF_ROOT")
            or paths_cfg.get("pdf_dir")
        )
        if not raw_pdf_root:
            parser.error(
                f"{process_name} requires --pdf-root, LITTRACEQA_PDF_ROOT, "
                "or paths.pdf_dir"
            )
        paths_cfg["pdf_dir"] = str(Path(raw_pdf_root).expanduser().resolve())

    if is_bounded_process:
        if args.build and args.limit is None:
            parser.error(f"{process_name} --build requires a positive --limit")
        if args.limit is not None and args.limit > _SAFE_PAPER_LIMIT:
            parser.error(
                f"this bounded runner refuses --limit above {_SAFE_PAPER_LIMIT}; "
                "incremental indexers are required first"
            )
        try:
            max_workers = process_worker_limit(process_cfg)
        except ValueError as exc:
            parser.error(str(exc))
        if args.workers > max_workers:
            parser.error(
                f"{process_name} allows at most {max_workers} worker(s)"
            )
        try:
            requested_artifact_root = require_bounded_artifact_root(
                process_name, args.artifact_root
            )
        except ValueError as exc:
            parser.error(str(exc))

        path_key = process_cfg.get("path_key", "pdf_dir")
        raw_source_root = paths_cfg.get(path_key) or process_cfg.get(
            "params", {}
        ).get(path_key)
        if not raw_source_root:
            parser.error(
                f"{process_name} requires source path {path_key!r} in paths or params"
            )
        source_root = Path(raw_source_root).expanduser().resolve()
        if not source_root.is_dir():
            parser.error(f"source root is not a directory: {source_root}")
        process_cfg.setdefault("params", {})[path_key] = str(source_root)
        read_only_roots = combine_read_only_roots(
            source_root, args.read_only_root
        )
        try:
            artifact_root = validate_artifact_root(
                requested_artifact_root, source_root, read_only_roots
            )
        except ValueError as exc:
            parser.error(str(exc))
        paths_cfg["chunks_dir"] = str(artifact_root / "chunks")
        paths_cfg["index_dir"] = str(artifact_root / "index")
        try:
            validate_artifact_path(artifact_root / "chunks", artifact_root)
            validate_artifact_path(artifact_root / "index", artifact_root)
            validate_artifact_path(artifact_root / "state", artifact_root)
        except ValueError as exc:
            parser.error(str(exc))
        image_param = process_cfg.get("artifact_image_param")
        if image_param:
            image_path = artifact_root / "images" / process_name
            try:
                validate_artifact_path(image_path, artifact_root)
            except ValueError as exc:
                parser.error(str(exc))
            process_cfg.setdefault("params", {})[image_param] = str(image_path)

        output_path_for_check = Path(args.output).expanduser().resolve()
        protected_roots = [source_root, *read_only_roots]
        for protected_root in protected_roots:
            protected_root = protected_root.expanduser().resolve()
            if _is_within(output_path_for_check, protected_root):
                parser.error(
                    f"prediction output is under read-only root {protected_root}: "
                    f"{output_path_for_check}"
                )
    elif args.artifact_root is not None:
        artifact_root = args.artifact_root.expanduser().resolve()
        paths_cfg["chunks_dir"] = str(artifact_root / "chunks")
        paths_cfg["index_dir"] = str(artifact_root / "index")

    cfg = compose_config(
        paths=paths_cfg,
        process=process_cfg,
        search=search_cfg,
        agent=agent_cfg,
    )
    if is_bounded_process:
        assert artifact_root is not None
        try:
            validate_bounded_index_dirs(cfg, artifact_root)
        except ValueError as exc:
            parser.error(str(exc))
    preprocessor, retriever, agent = build_pipeline(cfg)

    if args.build:
        chunks_path = Path(cfg["paths"]["chunks"])
        if is_bounded_process:
            assert artifact_root is not None
            metadata_path = Path(
                cfg.get("paths", {}).get("paper_metadata", "data/paper_metadata.jsonl")
            )
            try:
                requested_ids = read_requested_paper_ids(
                    args.paper_id, args.paper_ids_file
                )
                papers, missing_ids, rejected_papers = select_papers(
                    metadata_path, requested_ids, args.limit
                )
            except ValueError as exc:
                parser.error(str(exc))
            if not papers and not missing_ids and not rejected_papers:
                parser.error("no papers were selected")

            failures_path = artifact_root / "failures.jsonl"
            for paper_id in missing_ids:
                _append_jsonl(
                    failures_path,
                    {
                        "paper_id": paper_id,
                        "status": "failed",
                        "error_type": "MissingPaperMetadata",
                        "error": "paper_id was not found in paper metadata",
                    },
                )
            for rejected in rejected_papers:
                _append_jsonl(failures_path, rejected)
            started = time.perf_counter()
            process_fingerprint = build_process_fingerprint(cfg["preprocessor"])
            results = preprocess_selected_papers(
                papers=papers,
                preprocessor=preprocessor,
                artifact_root=artifact_root,
                process_name=process_name,
                workers=args.workers,
                batch_size=args.batch_size,
                resume=args.resume,
                failures_path=failures_path,
                process_fingerprint=process_fingerprint,
            )
            merge_selected_shards(chunks_path, results)
            chunks = load_chunks(chunks_path)
            summary = {
                "selected": len(papers) + len(missing_ids) + len(rejected_papers),
                "success": sum(result["status"] == "success" for result in results),
                "skipped": sum(result["status"] == "skipped" for result in results),
                "partial": sum(result["status"] == "partial" for result in results),
                "failed": sum(result["status"] == "failed" for result in results)
                + len(missing_ids)
                + len(rejected_papers),
                "chunks": len(chunks),
                "elapsed_seconds": time.perf_counter() - started,
                "process_fingerprint": process_fingerprint,
            }
            _write_json_atomic(artifact_root / "run_summary.json", summary)
            print(json.dumps({"preprocessing": summary}, ensure_ascii=False))
            if not chunks:
                parser.error("no chunks were produced; see failures.jsonl")
        elif preprocessor is not None:
            metadata_path = Path(
                cfg.get("paths", {}).get("paper_metadata", "data/paper_metadata.jsonl")
            )
            papers = load_papers(metadata_path)
            chunks = []
            for paper in tqdm(papers, desc="preprocessing"):
                chunks.extend(preprocessor.process(paper))
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with chunks_path.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        else:
            if not chunks_path.exists():
                parser.error(f"chunk file does not exist: {chunks_path}")
            chunks = load_chunks(chunks_path)

        for indexer in retriever.indexers:
            print(f"building {indexer.name}...")
            indexer.build(chunks)
        print("index build complete")
    else:
        print("loading existing indexes...")
        for indexer in retriever.indexers:
            try:
                indexer.load()
            except Exception as exc:
                print(
                    f"failed to load {indexer.name}: {exc}\n"
                    "run once with --build before searching",
                    file=sys.stderr,
                )
                raise SystemExit(1) from exc
        print("index load complete")

    queries = load_queries(Path(args.queries))
    output_path = Path(args.output).expanduser()
    write_predictions_atomic(output_path, queries, agent)
    print(f"wrote predictions to {output_path}")

    if args.gold:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--gold",
                args.gold,
                "--pred",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
