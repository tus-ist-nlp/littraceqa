"""Run a preprocessor over selected papers with per-paper checkpoints.

One unreadable paper must not stop a 27,487-paper conversion, so failures are
recorded per paper and the run continues.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.di_pipeline.preprocess.checkpoint import (
    MergeResult,
    PreprocessCache,
)


MAX_CHARS_PER_CHUNK = 100_000


@dataclass(frozen=True)
class PreprocessingRun:
    """Result of one bounded, paper-by-paper preprocessing pass."""

    selected_count: int
    processed_count: int
    reused_count: int
    failures: tuple[dict[str, Any], ...]
    merge_result: MergeResult | None


def override_max_chars_per_chunk(process_cfg: dict, value: int | None) -> dict:
    """Return a copied MinerU config with a validated chunk-size override."""
    if value is None:
        return process_cfg
    if process_cfg.get("name") != "mineru":
        raise ValueError("--max-chars-per-chunk currently supports MinerU only")
    if not 1 <= value <= MAX_CHARS_PER_CHUNK:
        raise ValueError(
            f"--max-chars-per-chunk must be between 1 and {MAX_CHARS_PER_CHUNK}"
        )
    updated = dict(process_cfg)
    params = dict(updated.get("params", {}))
    params["max_chars_per_chunk"] = value
    updated["params"] = params
    return updated


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
