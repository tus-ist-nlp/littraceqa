"""Choose which papers a bounded build processes, and gate large runs.

Every ceiling here exists so an accidental flag cannot start a 27,487-paper
build: the caller must state the count three times before crossing 5,000.
"""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MAX_BOUNDED_BUILD_PAPERS = 5_000
ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS = 27_487
LARGE_BUILD_THRESHOLD = 200


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
        <= ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS
    ):
        raise ValueError(
            "--max-build-papers must be between 1 and "
            f"{ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS}"
        )


def load_paper_ids_file(
    path: Path,
    *,
    max_papers: int = DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
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
    max_build_papers: int = DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
) -> None:
    """Require explicit, redundant confirmation for large bounded builds."""
    validate_build_ceiling(max_build_papers)
    if selected_count > max_build_papers:
        raise ValueError(
            f"selected paper count ({selected_count}) exceeds "
            f"--max-build-papers ({max_build_papers})"
        )
    if max_build_papers > DEFAULT_MAX_BOUNDED_BUILD_PAPERS:
        if selected_count != max_build_papers:
            raise ValueError(
                "the selected paper count must equal --max-build-papers "
                f"({max_build_papers}) above "
                f"{DEFAULT_MAX_BOUNDED_BUILD_PAPERS}"
            )
        if limit != selected_count:
            raise ValueError(
                "--limit must equal the selected paper count "
                f"({selected_count}) above "
                f"{DEFAULT_MAX_BOUNDED_BUILD_PAPERS}"
            )
    if selected_count <= LARGE_BUILD_THRESHOLD:
        return
    if paper_ids_file is None:
        raise ValueError(
            f"selecting more than {LARGE_BUILD_THRESHOLD} papers requires "
            "--paper-ids-file"
        )
    if confirm_paper_count != selected_count:
        raise ValueError(
            "--confirm-paper-count must equal the selected paper count "
            f"({selected_count})"
        )


def select_papers_for_bounded_build(
    path: Path,
    paper_ids: list[str],
    limit: int | None,
    *,
    max_build_papers: int = DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
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
