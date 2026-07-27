#!/usr/bin/env python3
"""configs/{paths,process_style,search_style,agent_style}/*.yaml を組み合わせて

前処理・索引構築・検索・評価を一気通貫で実行する e2e スクリプト。

Usage:
    # Build at most three papers without constructing or calling an LLM agent.
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output ~/littraceqa_data/mineru_eval/unused.jsonl \\
      --artifact-root ~/littraceqa_data/mineru_eval/smoke \\
      --limit 3 --build --build-only

    # Load a prebuilt index and run retrieval plus the configured agent.
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/abstract_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.contracts import Chunk, Query
from littraceqa.di_pipeline.preprocess.checkpoint import MergeResult, PreprocessCache

_DEFAULT_MAX_BOUNDED_BUILD_PAPERS = 5_000
_ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS = 10_000
_LARGE_BUILD_THRESHOLD = 200
_MAX_CHARS_PER_CHUNK = 100_000
_INDEX_BUILD_STATE_SCHEMA = 1


@dataclass(frozen=True)
class PreprocessingRun:
    """Result of one bounded, paper-by-paper preprocessing pass."""

    selected_count: int
    processed_count: int
    reused_count: int
    failures: tuple[dict[str, Any], ...]
    merge_result: MergeResult | None


@dataclass(frozen=True)
class IndexBuildRun:
    """Result of a resumable, ordered index build pass."""

    built_count: int
    loaded_count: int


def normalize_paper_ids(*groups: list[str]) -> list[str]:
    """Return unique, non-empty paper IDs in first-seen order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_paper_id in group:
            paper_id = raw_paper_id.strip()
            if paper_id and paper_id not in seen:
                normalized.append(paper_id)
                seen.add(paper_id)
    return normalized


def validate_build_ceiling(max_build_papers: int) -> None:
    """Validate the explicit per-run ceiling before reading build inputs."""
    if (
        isinstance(max_build_papers, bool)
        or not isinstance(max_build_papers, int)
        or not 1
        <= max_build_papers
        <= _ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS
    ):
        raise ValueError(
            "--max-build-papers must be between 1 and "
            f"{_ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS}"
        )


def load_paper_ids_file(
    path: Path,
    *,
    max_papers: int = _DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
) -> list[str]:
    """Load unique paper IDs while stopping immediately at the safety ceiling."""
    validate_build_ceiling(max_papers)
    paper_ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw_paper_id in handle:
            paper_id = raw_paper_id.strip()
            if not paper_id or paper_id in seen:
                continue
            paper_ids.append(paper_id)
            seen.add(paper_id)
            if len(paper_ids) > max_papers:
                raise ValueError(
                    f"--paper-ids-file contains more than {max_papers} "
                    "distinct paper IDs"
                )
    if not paper_ids:
        raise ValueError(f"--paper-ids-file contains no paper IDs: {path}")
    return paper_ids


def validate_large_build_selection(
    selected_count: int,
    *,
    paper_ids_file: Path | None,
    confirm_paper_count: int | None,
    limit: int | None = None,
    max_build_papers: int = _DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
) -> None:
    """Require explicit, redundant confirmation for large bounded builds."""
    validate_build_ceiling(max_build_papers)
    if selected_count > max_build_papers:
        raise ValueError(
            f"selected paper count ({selected_count}) exceeds "
            f"--max-build-papers ({max_build_papers})"
        )
    if max_build_papers > _DEFAULT_MAX_BOUNDED_BUILD_PAPERS:
        if selected_count != max_build_papers:
            raise ValueError(
                "the selected paper count must equal --max-build-papers "
                f"({max_build_papers}) above "
                f"{_DEFAULT_MAX_BOUNDED_BUILD_PAPERS}"
            )
        if limit != selected_count:
            raise ValueError(
                "--limit must equal the selected paper count "
                f"({selected_count}) above "
                f"{_DEFAULT_MAX_BOUNDED_BUILD_PAPERS}"
            )
    if selected_count <= _LARGE_BUILD_THRESHOLD:
        return
    if paper_ids_file is None:
        raise ValueError(
            f"selecting more than {_LARGE_BUILD_THRESHOLD} papers requires "
            "--paper-ids-file"
        )
    if confirm_paper_count != selected_count:
        raise ValueError(
            "--confirm-paper-count must equal the selected paper count "
            f"({selected_count})"
        )


def load_papers(path: Path) -> list[dict]:
    papers = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def select_papers_for_bounded_build(
    path: Path,
    paper_ids: list[str],
    limit: int | None,
    *,
    max_build_papers: int = _DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
) -> list[dict]:
    """Select a deterministic, bounded paper subset in metadata order."""
    validate_build_ceiling(max_build_papers)
    requested = set(normalize_paper_ids(paper_ids))
    if not requested and limit is None:
        raise ValueError("--build requires --paper-id and/or --limit")
    if limit is not None and not 1 <= limit <= max_build_papers:
        raise ValueError(
            f"--limit must be between 1 and {max_build_papers}"
        )
    if len(requested) > max_build_papers:
        raise ValueError(
            f"at most {max_build_papers} distinct --paper-id values are allowed"
        )
    if requested and limit is not None and limit < len(requested):
        raise ValueError("--limit cannot be smaller than the number of --paper-id values")

    selected: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            paper_id = str(paper.get("paper_id") or "").strip()
            if not paper_id or paper_id in seen:
                continue
            if requested and paper_id not in requested:
                continue
            selected.append(paper)
            seen.add(paper_id)
            if limit is not None and len(selected) >= limit:
                break
            if requested and seen == requested:
                break

    missing = requested - seen
    if missing:
        raise ValueError(f"requested paper IDs were not found: {sorted(missing)}")
    if not selected:
        raise ValueError("paper selection produced no records")
    return selected


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def validate_write_paths(
    output_paths: list[Path],
    artifact_root: Path,
    read_only_root: Path,
) -> None:
    """Reject writes outside the requested artifact root or into shared input."""
    artifact_root = artifact_root.expanduser().resolve()
    read_only_root = read_only_root.expanduser().resolve()
    if _paths_overlap(artifact_root, read_only_root):
        raise ValueError("--artifact-root must not overlap --read-only-root")
    for output_path in output_paths:
        resolved = output_path.expanduser().resolve()
        if _paths_overlap(resolved, read_only_root):
            raise ValueError(f"refusing to write inside read-only input: {resolved}")
        try:
            resolved.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"build output must stay under --artifact-root: {resolved}"
            ) from exc


def resolve_preprocess_cache_root(
    explicit_root: Path | None,
    artifact_root: Path,
) -> Path:
    """Resolve an optional shared cache while preserving the prior default."""
    root = explicit_root if explicit_root is not None else artifact_root / "preprocess"
    return root.expanduser().resolve()


def preprocessing_source_roots(preprocessor: Any) -> list[Path]:
    """Return roots containing the artifacts read by a preprocessor."""
    mineru_dir = getattr(preprocessor, "mineru_dir", None)
    if mineru_dir is not None:
        return [Path(mineru_dir).expanduser().resolve()]
    pdf_dir = getattr(preprocessor, "pdf_dir", None)
    if pdf_dir is not None:
        return [Path(pdf_dir).expanduser().resolve()]
    return []


def validate_preprocess_cache_root(
    cache_root: Path,
    *,
    read_only_root: Path,
    source_roots: Sequence[Path],
) -> None:
    """Reject cache locations that could overwrite or contain input data."""
    cache_root = cache_root.expanduser().resolve()
    protected_roots = [
        read_only_root.expanduser().resolve(),
        *(root.expanduser().resolve() for root in source_roots),
    ]
    for protected_root in protected_roots:
        if _paths_overlap(cache_root, protected_root):
            raise ValueError(
                "preprocess cache must not overlap read-only or source input: "
                f"{protected_root}"
            )

    for internal_path in (
        cache_root / "papers",
        cache_root / "manifest.jsonl",
    ):
        if internal_path.is_symlink():
            raise ValueError(
                "preprocess cache internal paths must not be symlinks: "
                f"{internal_path}"
            )
        resolved_internal = internal_path.resolve()
        try:
            resolved_internal.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(
                "preprocess cache internal path escapes its root: "
                f"{resolved_internal}"
            ) from exc
        for protected_root in protected_roots:
            if _paths_overlap(resolved_internal, protected_root):
                raise ValueError(
                    "preprocess cache internal path overlaps protected input: "
                    f"{resolved_internal}"
                )

    if cache_root.exists():
        if not cache_root.is_dir():
            raise ValueError(
                f"preprocess cache root is not a directory: {cache_root}"
            )
        if cache_root.stat().st_uid != os.getuid():
            raise ValueError(
                f"preprocess cache root is not owned by the current user: {cache_root}"
            )
        if not os.access(cache_root, os.W_OK | os.X_OK):
            raise ValueError(
                f"preprocess cache root is not writable: {cache_root}"
            )
        return

    writable_parent = cache_root
    while not writable_parent.exists() and writable_parent != writable_parent.parent:
        writable_parent = writable_parent.parent
    if not writable_parent.is_dir() or not os.access(
        writable_parent,
        os.W_OK | os.X_OK,
    ):
        raise ValueError(
            "preprocess cache root has no writable parent owned or accessible "
            f"to the current user: {cache_root}"
        )


def override_max_chars_per_chunk(process_cfg: dict, value: int | None) -> dict:
    """Return a copied MinerU config with a validated chunk-size override."""
    if value is None:
        return process_cfg
    if process_cfg.get("name") != "mineru":
        raise ValueError("--max-chars-per-chunk currently supports MinerU only")
    if not 1 <= value <= _MAX_CHARS_PER_CHUNK:
        raise ValueError(
            f"--max-chars-per-chunk must be between 1 and {_MAX_CHARS_PER_CHUNK}"
        )
    updated = dict(process_cfg)
    params = dict(updated.get("params", {}))
    params["max_chars_per_chunk"] = value
    updated["params"] = params
    return updated


def validate_build_mode(
    *,
    build: bool,
    build_only: bool,
    resume: bool,
    max_chars_per_chunk: int | None,
    preprocess_cache_root: Path | None = None,
) -> None:
    """Validate options whose meaning depends on index build mode."""
    if build_only and not build:
        raise ValueError("--build-only requires --build")
    if resume and not build:
        raise ValueError("--resume requires --build")
    if max_chars_per_chunk is not None and not build:
        raise ValueError("--max-chars-per-chunk requires --build")
    if preprocess_cache_root is not None and not build:
        raise ValueError("--preprocess-cache-root requires --build")


def iter_chunks(path: Path) -> Iterator[Chunk]:
    """Stream chunk records from JSONL without retaining the whole corpus."""
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("chunk record must be a JSON object")
                yield Chunk(**record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number} is not a valid Chunk record"
                ) from exc


def load_chunks(path: Path) -> list[Chunk]:
    """Load all chunks for callers that explicitly need a materialized list."""
    return list(iter_chunks(path))


def preprocessing_source_path(preprocessor: Any, paper: Mapping[str, Any]) -> Path:
    """Return the source artifact whose state invalidates one paper cache."""
    paper_id = str(paper.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("paper metadata must contain a non-empty paper_id")

    content_list_path = getattr(preprocessor, "content_list_path", None)
    if callable(content_list_path):
        return Path(content_list_path(paper_id))

    pdf_dir = getattr(preprocessor, "pdf_dir", None)
    if pdf_dir is not None:
        return Path(pdf_dir) / f"{paper_id}.pdf"

    raise TypeError(
        f"{type(preprocessor).__name__} exposes neither content_list_path() "
        "nor pdf_dir"
    )


def _write_failure_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace the current-run preprocessing failure report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def preprocess_selected_papers(
    *,
    preprocessor: Any,
    selected_papers: Sequence[Mapping[str, Any]],
    cache: PreprocessCache,
    chunks_path: Path,
    failures_path: Path,
    resume: bool,
) -> PreprocessingRun:
    """Checkpoint each paper and publish a merged JSONL only after full success."""
    selected_sources = [
        (paper, preprocessing_source_path(preprocessor, paper))
        for paper in selected_papers
    ]
    failures: list[dict[str, Any]] = []
    processed_count = 0
    reused_count = 0

    for paper, source_path in tqdm(selected_sources, desc="Preprocessing papers"):
        if resume and cache.load_valid_chunks(paper, source_path) is not None:
            reused_count += 1
            continue

        try:
            paper_chunks = preprocessor.process(dict(paper))
            cache.store_success(paper, source_path, paper_chunks)
            processed_count += 1
        except Exception as exc:  # One malformed paper must not discard prior work.
            cache.record_failure(paper, source_path, exc)
            failures.append(
                {
                    "paper_id": paper.get("paper_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    _write_failure_records(failures_path, failures)
    merge_result = None
    if not failures:
        merge_result = cache.merge_selected(selected_sources, chunks_path)

    return PreprocessingRun(
        selected_count=len(selected_sources),
        processed_count=processed_count,
        reused_count=reused_count,
        failures=tuple(failures),
        merge_result=merge_result,
    )


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


def _implementation_source_paths(component: Any) -> tuple[Path, ...]:
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


def _fingerprint_chunk_file(path: Path) -> MergeResult:
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
        _implementation_source_paths(indexer)
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


# 本番の入力に実際に入っているフィールドはこの4つだけ（確定仕様）。
# multiple_choice の options は**本番では与えられない**ので、ここには入れない。
_PRODUCTION_FIELDS = ("query_id", "question", "answer_types", "table_schema")


def load_mc_options(path: Path) -> dict[str, dict]:
    """query_id -> multiple_choice の options だけを読む（gold は絶対に読まない）。

    本番入力に options は無い（上記 _PRODUCTION_FIELDS）。よってこれを結合した実行は
    「選択肢を教えてもらえたら何点取れるか」を見る **oracle 設定** であり、本番の点数
    ではない。gold（正解の選択肢）は読まないので答えそのものの漏洩ではないが、
    41/55 問が multiple_choice で、うち21問は freeform すら無い（選択肢が無いと
    そもそも文字を決められない）ため、点数への影響は大きい。
    """
    options_map: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mc = (record.get("answer") or {}).get("multiple_choice") or {}
            options = mc.get("options") if isinstance(mc, dict) else None
            if options:
                options_map[record["query_id"]] = options
    return options_map


def load_queries(
    path: Path, production_input: bool = True, options_path: Path | None = None
) -> list[Query]:
    """Load queries using production fields by default.

    Validation-only labels are discarded unless an explicit oracle run passes
    ``production_input=False``. Multiple-choice options are also oracle-only
    because the production input does not provide them.
    """
    options_map = (
        load_mc_options(options_path)
        if options_path is not None and not production_input
        else {}
    )
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if production_input:
                record = {k: v for k, v in record.items() if k in _PRODUCTION_FIELDS}
            if not record.get("options") and record["query_id"] in options_map:
                record["options"] = options_map[record["query_id"]]
            queries.append(Query.from_dict(record))
    return queries


def git_sha() -> str | None:
    """実行時のコミットハッシュ。git 管理外なら None（記録は best effort）。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:  # noqa: BLE001 - 由来情報が取れなくても実験は続行する。
        pass
    return None


# reranker の実効設定として記録する属性（インスタンス側の名前 -> 記録名）。
# yaml に書かれていない既定値（instruction / compile など）も残さないと、
# 「この数字がどの設定で出たのか」が後から再現できない。
_RERANKER_EFFECTIVE_ATTRS = {
    "model_name": "model",
    "revision": "revision",
    "device": "device",
    "fp16": "fp16",
    "batch_size": "batch_size",
    "max_tokens": "max_tokens",
    "instruction": "instruction",
    "compile": "compile",
    "base_rank_weight": "base_rank_weight",
    "rank_fusion_k": "rank_fusion_k",
}


def _flatten(prefix: str, params: dict | None) -> dict:
    """{"k": 60} -> {"fuser_k": 60}。ネストしたままだと差分比較で読めないため平らにする。"""
    if not isinstance(params, dict):
        return {}
    return {f"{prefix}_{key}": value for key, value in params.items()}


def tuned_params(cfg: dict, retriever_obj: Any = None) -> dict:
    """チューニング対象のパラメータだけを平らな dict にまとめる。

    experiments.jsonl には解決済みの cfg 全体も残すが、そちらは index_dir などの
    環境依存の値も含んで長い。実験を横並び比較するときに見たいのは
    「振ったつまみ」なので、それだけを抜き出す。

    ネストした *_params は平らにする（reranker_max_tokens のように1つずつ列になり、
    1つだけ変えた実験の差分が読めるようにするため）。reranker は yaml に書かない
    既定値も効いてしまうので、組み立て済みインスタンスから実効値を拾う。
    """
    retriever = cfg.get("retriever", {})
    preprocessor = cfg.get("preprocessor", {})
    preprocessor_params = {
        key: value
        for key, value in (preprocessor.get("params") or {}).items()
        if key not in {"pdf_dir", "mineru_dir"}
    }
    agent = cfg.get("agent", {})
    agent_params = agent.get("params", {})
    fuser = retriever.get("fuser", {})
    reranker = retriever.get("reranker", {})

    # reranker: 実インスタンスの属性を優先し、取れなければ yaml 宣言値にフォールバック。
    reranker_effective = _flatten("reranker", reranker.get("params"))
    obj = getattr(retriever_obj, "reranker", None)
    if obj is not None:
        for attr, label in _RERANKER_EFFECTIVE_ATTRS.items():
            if hasattr(obj, attr):
                reranker_effective[f"reranker_{label}"] = getattr(obj, attr)

    return {
        # Preprocessing
        "preprocessor": preprocessor.get("name"),
        **_flatten("preprocessor", preprocessor_params),
        # 検索側
        "per_index_k": retriever.get("per_index_k"),
        "pool_k": retriever.get("pool_k"),
        "indexers": [ix.get("index_name", ix["name"]) for ix in retriever.get("indexers", [])],
        "fuser": fuser.get("name"),
        **_flatten("fuser", fuser.get("params")),
        "reranker": reranker.get("name"),
        **reranker_effective,
        # エージェント側
        "agent": agent.get("name"),
        "agent_llm": (agent.get("llm") or {}).get("name"),
        **{f"agent_{k}": v for k, v in agent_params.items()},
    }


def log_experiment(
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    cfg: dict,
    retriever_obj: Any = None,
    options_joined: bool = False,
) -> None:
    """どの組み合わせで何点だったかを results/experiments.jsonl に追記する。

    設定ファイルの「パス」だけだと、同じ yaml を書き換えて振った実験が
    全部同じ行に見えてしまい、後から「この数字はどのパラメータで出たのか」が
    追えない。compose_config() が解決した実際の値ごと残す。

    options_joined は multiple_choice の選択肢を与えた oracle 実行かどうか。
    本番では options が来ないので、True の行の multiple_choice_accuracy は
    本番の点数として読んではいけない。後から見分けられるように残す。
    """
    path = Path("results/experiments.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": args.paths,
        "process": args.process,
        "search": args.search,
        "agent": args.agent,
        "queries": args.queries,
        "production_input": args.production_input,
        "options_joined": options_joined,
        "n_queries": n_queries,
        "output": args.output,
        "git_sha": git_sha(),
        "tuned_params": tuned_params(cfg, retriever_obj),
        "config": cfg,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"実験結果を {path} に追記しました")


def _load_matching_experiments(
    process: str, search: str, agent: str, limit: int = 3
) -> list[dict]:
    """results/experiments.jsonl から同じ組み合わせの過去記録を、直近 limit 件取り出す。"""
    path = Path("results/experiments.jsonl")
    if not path.exists():
        return []
    matches = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("process"), record.get("search"), record.get("agent")) == (
                process,
                search,
                agent,
            ):
                matches.append(record)
    return matches[-limit:]


def generate_comment(llm, args: argparse.Namespace, metrics: dict, n_queries: int) -> str:
    """指標を読んで、LLM に簡潔な所感を書かせる。llm が無ければ固定文言を返す。

    log_experiment() が今回の記録を results/experiments.jsonl に追記した後に
    呼ばれる前提なので、直近の一致レコードの末尾(=今回分)を除いて過去分だけを渡す。
    """
    if llm is None:
        return "(LLMコメントなし: このagent_styleはLLMを使用しない設定です)"

    history = _load_matching_experiments(args.process, args.search, args.agent, limit=4)[:-1]
    history_text = "\n".join(
        f"- {record['timestamp']}: {json.dumps(record['metrics'], ensure_ascii=False)}"
        for record in history
    ) or "(同じ組み合わせの過去記録なし)"

    prompt = (
        "あなたは検索システムの実験結果を確認する研究者です。次の実験結果を読み、"
        "指標の良し悪しや気になる点、次に試すとよさそうなことを日本語で簡潔にコメントしてください。\n\n"
        f"設定: process={args.process}, search={args.search}, agent={args.agent}\n"
        f"クエリ数: {n_queries} (production_input={args.production_input})\n"
        f"今回の指標: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        f"同じ組み合わせの過去の実行記録(古い順):\n{history_text}\n\n"
        '出力は JSON のみとし、{"comment": "..."} の形式で3〜5文程度にまとめてください。'
    )
    try:
        parsed = parse_json_object(llm(prompt))
    except Exception as exc:
        return f"(LLMコメントの生成に失敗しました: {exc})"
    if not parsed or not isinstance(parsed.get("comment"), str):
        return "(LLMコメントの生成に失敗しました: 応答をパースできませんでした)"
    return parsed["comment"]


def write_report(
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    comment: str,
    cfg: dict,
    retriever_obj: Any = None,
    options_joined: bool = False,
) -> None:
    """1回の実行につき、設定・指標・LLMコメントをまとめた Markdown を report/ に1枚書く。"""
    process_name = Path(args.process).stem
    search_name = Path(args.search).stem
    agent_name = Path(args.agent).stem
    now = datetime.now()

    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{process_name}_{search_name}_{agent_name}.md"
    path = report_dir / filename

    lines = [
        f"# {process_name} + {search_name} + {agent_name}",
        "",
        f"- 実行日時: {now.isoformat(timespec='seconds')}",
        f"- paths: `{args.paths}`",
        f"- process: `{args.process}`",
        f"- search: `{args.search}`",
        f"- agent: `{args.agent}`",
        f"- queries: `{args.queries}` ({n_queries}件, production_input={args.production_input})",
        f"- output: `{args.output}`",
    ]
    sha = git_sha()
    if sha:
        lines.append(f"- git: `{sha[:12]}`")
    if options_joined:
        lines.append(
            "- **[oracle] multiple_choice の選択肢を与えて実行**（本番入力に options は"
            "無いため、multiple_choice_accuracy は本番の点数ではない）"
        )
    # yaml は後から書き換わるので、レポート単体で「どの値で回したか」が分かるように
    # 解決済みのパラメータをここに焼き込む。
    lines.extend(
        [
            "",
            "## 設定（この実行時の実際の値）",
            "",
            "| パラメータ | 値 |",
            "|---|---|",
        ]
    )
    for key, value in tuned_params(cfg, retriever_obj).items():
        if value is None:
            continue
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)}` |")
    lines.extend(
        [
            "",
            "## 指標",
            "",
            "| 指標 | 値 |",
            "|---|---|",
        ]
    )
    for key, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")
    lines.extend(["", "## コメント", "", comment, ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"レポートを {path} に書き出しました")


def main() -> None:
    # Import optional retrieval dependencies only when the CLI is executed.
    # Query-loading and path-safety helpers remain testable with the base extra.
    from littraceqa.di_pipeline.config import (
        build_pipeline,
        compose_config,
        load_config,
        override_rerank_pool,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--build", action="store_true", help="前処理 + 索引構築をする（初回のみ）"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build a bounded index without constructing or calling an agent.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse valid per-paper preprocessing and completed indexer "
            "checkpoints during --build."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="User-owned output root required for bounded index builds.",
    )
    parser.add_argument(
        "--preprocess-cache-root",
        type=Path,
        help=(
            "Optional user-owned per-paper cache shared by multiple artifact "
            "roots; defaults to <artifact-root>/preprocess."
        ),
    )
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="Shared input root that must never receive writes.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Paper ID to include in a bounded build; repeat as needed.",
    )
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        help="File containing one paper ID per line for a bounded build.",
    )
    parser.add_argument(
        "--confirm-paper-count",
        type=int,
        help=(
            "Exact selected paper count required when building more than "
            f"{_LARGE_BUILD_THRESHOLD} papers."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Exact maximum papers for a bounded build. The default safety "
            f"ceiling is {_DEFAULT_MAX_BOUNDED_BUILD_PAPERS}."
        ),
    )
    parser.add_argument(
        "--max-build-papers",
        type=int,
        default=_DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
        help=(
            "Safety ceiling for this build. Values above "
            f"{_DEFAULT_MAX_BOUNDED_BUILD_PAPERS} require redundant exact-count "
            f"confirmation and cannot exceed {_ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS}."
        ),
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        help=(
            "Override MinerU text chunk size for a bounded build; "
            f"maximum {_MAX_CHARS_PER_CHUNK}."
        ),
    )
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Override the enabled reranker's candidate pool (1-1000).",
    )
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--production-input",
        dest="production_input",
        action="store_true",
        default=True,
        help="Use only query_id, question, answer_types, and table_schema (default).",
    )
    input_mode.add_argument(
        "--allow-validation-labels",
        dest="production_input",
        action="store_false",
        help="[oracle] Allow validation-only labels such as task_family.",
    )
    parser.add_argument(
        "--options-file",
        default=None,
        help="[oracle] multiple_choice の options を結合する jsonl（gold は読まない）。"
        "本番入力に options は無いので、これを付けた実行は「選択肢を教えてもらえたら"
        "何点取れるか」を見る ablation であり本番の点数ではない。"
        "--production-input との併用時は無視される。",
    )
    args = parser.parse_args()

    try:
        validate_build_mode(
            build=args.build,
            build_only=args.build_only,
            resume=args.resume,
            max_chars_per_chunk=args.max_chars_per_chunk,
            preprocess_cache_root=args.preprocess_cache_root,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        validate_build_ceiling(args.max_build_papers)
        if (
            args.build
            and args.max_build_papers
            > _DEFAULT_MAX_BOUNDED_BUILD_PAPERS
        ):
            validate_large_build_selection(
                args.max_build_papers,
                paper_ids_file=args.paper_ids_file,
                confirm_paper_count=args.confirm_paper_count,
                limit=args.limit,
                max_build_papers=args.max_build_papers,
            )
    except ValueError as exc:
        parser.error(str(exc))

    if _paths_overlap(Path(args.output), args.read_only_root):
        parser.error("--output must not overlap --read-only-root")

    requested_paper_ids = normalize_paper_ids(args.paper_id)
    if args.build and args.paper_ids_file is not None:
        try:
            file_paper_ids = load_paper_ids_file(
                args.paper_ids_file,
                max_papers=args.max_build_papers,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        requested_paper_ids = normalize_paper_ids(
            requested_paper_ids,
            file_paper_ids,
        )

    preprocess_cache_root: Path | None = None
    paths_cfg = load_config(args.paths)
    if args.build:
        if args.artifact_root is None:
            parser.error("--build requires --artifact-root")
        artifact_root = args.artifact_root.expanduser().resolve()
        preprocess_cache_root = resolve_preprocess_cache_root(
            args.preprocess_cache_root,
            artifact_root,
        )
        paths_cfg = dict(paths_cfg)
        paths_cfg["chunks_dir"] = str(artifact_root / "chunks")
        paths_cfg["index_dir"] = str(artifact_root / "index")

    process_cfg = load_config(args.process)
    try:
        process_cfg = override_max_chars_per_chunk(
            process_cfg,
            args.max_chars_per_chunk,
        )
    except ValueError as exc:
        parser.error(str(exc))

    search_cfg = load_config(args.search)
    try:
        search_cfg = override_rerank_pool(search_cfg, args.rerank_pool_k)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = compose_config(
        paths=paths_cfg,
        process=process_cfg,
        search=search_cfg,
        agent=load_config(args.agent),
    )
    if args.build:
        index_paths = [
            Path(indexer["params"]["index_dir"])
            for indexer in cfg["retriever"]["indexers"]
        ]
        validate_write_paths(
            [Path(cfg["paths"]["chunks"]), *index_paths],
            artifact_root,
            args.read_only_root,
        )

    selected_papers: list[dict] | None = None
    if args.build:
        metadata_path = Path(
            cfg.get("paths", {}).get("paper_metadata", "data/paper_metadata.jsonl")
        )
        try:
            selected_papers = select_papers_for_bounded_build(
                metadata_path,
                requested_paper_ids,
                args.limit,
                max_build_papers=args.max_build_papers,
            )
            validate_large_build_selection(
                len(selected_papers),
                paper_ids_file=args.paper_ids_file,
                confirm_paper_count=args.confirm_paper_count,
                limit=args.limit,
                max_build_papers=args.max_build_papers,
            )
        except ValueError as exc:
            parser.error(str(exc))

    preprocessor, retriever, agent = build_pipeline(
        cfg,
        build_agent=not args.build_only,
        build_preprocessor=args.build,
    )
    if args.build:
        if preprocessor is None or preprocess_cache_root is None:
            raise RuntimeError("build preprocessing cache was not initialized")
        try:
            validate_preprocess_cache_root(
                preprocess_cache_root,
                read_only_root=args.read_only_root,
                source_roots=[
                    *preprocessing_source_roots(preprocessor),
                    Path(cfg["paths"]["paper_metadata"]),
                ],
            )
        except ValueError as exc:
            parser.error(str(exc))

    if args.build:
        chunks_path = Path(cfg["paths"]["chunks"])
        merged_chunks: MergeResult | None = None

        if preprocessor is not None:
            if selected_papers is None:
                raise RuntimeError("bounded paper selection was not initialized")

            failures_path = artifact_root / "failures.jsonl"
            implementation_paths = _implementation_source_paths(preprocessor)
            cache = PreprocessCache(
                preprocess_cache_root,
                process_config=cfg["preprocessor"],
                source_module_path=implementation_paths[0],
                source_dependency_paths=implementation_paths[1:],
            )
            preprocessing = preprocess_selected_papers(
                preprocessor=preprocessor,
                selected_papers=selected_papers,
                cache=cache,
                chunks_path=chunks_path,
                failures_path=failures_path,
                resume=args.resume,
            )
            print(
                "Bounded preprocessing: "
                f"{preprocessing.processed_count} processed, "
                f"{preprocessing.reused_count} reused, "
                f"{len(preprocessing.failures)} failed; "
                f"failures: {failures_path}"
            )
            if preprocessing.failures:
                print(
                    "Preprocessing stopped before global index construction. "
                    "Completed papers remain checkpointed; rerun the same build "
                    "with --resume to retry only failed or stale papers.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if preprocessing.merge_result is None:
                raise RuntimeError("preprocessing did not publish merged chunks")
            merged_chunks = preprocessing.merge_result
            print(
                f"{preprocessing.merge_result.chunk_count} chunks from "
                f"{preprocessing.merge_result.paper_count} papers were "
                f"atomically saved to {chunks_path}"
            )

        else:
            if not chunks_path.exists():
                print(f"エラー: {chunks_path} が存在しません", file=sys.stderr)
                sys.exit(1)
            merged_chunks = _fingerprint_chunk_file(chunks_path)

        if merged_chunks is None:
            raise RuntimeError("merged chunk fingerprint was not initialized")
        index_build = build_indexers_with_resume(
            indexers=retriever.indexers,
            indexer_configs=cfg["retriever"]["indexers"],
            chunks_path=chunks_path,
            chunks=merged_chunks,
            state_path=artifact_root / "index_build_state.json",
            resume=args.resume,
        )
        print(
            f"Index checkpoints: {index_build.built_count} built, "
            f"{index_build.loaded_count} loaded"
        )
        print("索引構築完了")
        if args.build_only:
            print("Build-only mode completed without constructing or calling an agent.")
            return

    else:
        print("既存の索引を読み込み中...")
        for indexer in retriever.indexers:
            try:
                indexer.load()
            except Exception as exc:
                print(
                    f"エラー: {indexer.name} の索引読み込みに失敗しました: {exc}\n"
                    f"先に --build を付けて索引を構築してください。",
                    file=sys.stderr,
                )
                sys.exit(1)
        print("読み込み完了")

    # multiple_choice の options の入手先を決める。本番入力に options は無いので、
    # 結合するのは常に oracle 設定（--options-file を明示したときだけ）。
    # --production-input との併用は矛盾（本番と揃えるフラグなのに本番に無い情報を足す）
    # なので、その場合は結合しない。
    options_path = Path(args.options_file) if args.options_file else None
    if options_path is not None and args.production_input:
        print(
            "警告: --production-input と --options-file は併用できません"
            "（本番入力に options は無い）。options の結合をスキップします。",
            file=sys.stderr,
        )
        options_path = None
    queries = load_queries(
        Path(args.queries),
        production_input=args.production_input,
        options_path=options_path,
    )
    if args.production_input:
        print("本番と同じ4フィールド（query_id/question/answer_types/table_schema）で走らせます")
    if options_path is not None:
        n_opt = sum(1 for q in queries if q.options)
        print(
            f"[oracle] multiple_choice options を {options_path} から結合しました"
            f"（{n_opt}件）。本番では与えられないので、この点数は本番の点数ではありません。"
        )
    print(f"{len(queries)} 件の質問に対して検索中...")

    if agent is None:
        raise RuntimeError("agent was not built")

    predictions = []
    for i, query in enumerate(queries):
        pred = agent.run(query)
        predictions.append(pred.to_dict())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} 完了")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"予測結果を {output_path} に書き出しました")

    print("\n採点中...")
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate.py",
            "--gold", "data/validation.jsonl",
            "--pred", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    try:
        metrics = json.loads(result.stdout)["metrics"]
    except (json.JSONDecodeError, KeyError):
        print("採点結果を解釈できなかったので実験ログには残しません", file=sys.stderr)
        return
    options_joined = options_path is not None
    log_experiment(args, metrics, len(queries), cfg, retriever, options_joined)
    comment = generate_comment(getattr(agent, "llm", None), args, metrics, len(queries))
    write_report(args, metrics, len(queries), comment, cfg, retriever, options_joined)


if __name__ == "__main__":
    main()
