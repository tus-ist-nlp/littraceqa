"""Gold-free integrity checks for a MinerU corpus and candidate handoff."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from littraceqa.candidate_handoff import CandidateHandoff
from littraceqa.chunk_store import (
    IMAGE_PATH_ERROR_KEY,
    IMAGE_PATH_ORIGINAL_KEY,
    ChunkStore,
)
from littraceqa.mineru_record import record_source_type, submission_evidence_eligible
from littraceqa.mineru_record import validate_image_file
from littraceqa.submission import OFFICIAL_SOURCE_TYPES

_TABLE_RE = re.compile(r"\b(table|row|column)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(
    r"\b(figure|fig\.?|plot|chart|graph|diagram|panel|subplot)\b",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(
    r"\b(equation|objective|loss function|formula|algorithm)\b", re.IGNORECASE
)
_CITATION_RE = re.compile(
    r"\b(citation|cited|reference|bibliography)\b", re.IGNORECASE
)

# Keep the image-availability gate deliberately conservative.  In particular,
# a question about "image generation" does not necessarily require inspecting
# an image. Figure numbers, panels, charts/plots/graphs, and wording that
# explicitly asks the reader to inspect an image do.
_VISUAL_IMAGE_REQUIRED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bfig(?:ure)?\.?\s*(?:\d+[a-z]?|[ivx]+)(?:\s*\([a-z0-9]+\))?\b",
        r"\b(?:panel|subplot)s?(?:\s*\([a-z0-9]+\)|\s+[a-z0-9]+)?\b",
        r"\b(?:chart|plot)s?\b",
        r"\b(?:according to|shown in|visible in|depicted in|displayed in|"
        r"read from|inspect|from)\s+(?:an?\s+|the\s+|this\s+|that\s+)?"
        r"(?:image|figure|graph|diagram)\b",
        r"\b(?:image|figure|graph|diagram)\s+"
        r"(?:shows|depicts|displays|illustrates|contains)\b",
        r"\b(?:plotted|graphed)\s+(?:value|ratio|curve|point|result)s?\b",
        # Explicit ownership/location wording without a numbered Figure, e.g.
        # q_020: "in their primary method/framework figure". Requiring a
        # demonstrative/possessive avoids matching topical phrases such as
        # "performance in figure generation".
        r"\b(?:in|within)\s+(?:the|this|that|their|its)\s+"
        r"(?:(?:primary|main|proposed)\s+)?"
        r"(?:(?:method|framework|architecture)"
        r"(?:\s*/\s*(?:method|framework|architecture))?\s+)?"
        r"(?:figure|diagram)\b",
    )
)


def inspect_corpus(
    handoffs: Iterable[CandidateHandoff],
    store: ChunkStore,
    canonical_paper_ids: set[str] | None = None,
    *,
    allow_missing_figure_images: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Inspect only observable queries plus their candidate papers.

    When canonical paper IDs are supplied, the byte-offset index is also
    checked for complete 27k-paper coverage. No validation answers, evidence
    labels, task families or primary evidence types are read.
    """

    handoffs = list(handoffs)
    candidate_ids = {
        paper.paper_id for handoff in handoffs for paper in handoff.candidate_papers
    }
    chunk_types: Counter[str] = Counter()
    invalid_pages: Counter[str] = Counter()
    invalid_object_ids: Counter[str] = Counter()
    valid_types_by_paper: dict[str, set[str]] = {}
    image_types_by_paper: dict[str, set[str]] = {}
    missing_papers: list[str] = []
    papers_without_valid_evidence: list[str] = []

    image_eligible = image_declared = image_existing = image_unreadable = 0
    image_paths_seen: set[str] = set()
    readable_image_paths: set[str] = set()
    missing_image_examples: list[str] = []
    unreadable_image_examples: list[str] = []
    unsafe_image_examples: list[dict[str, str]] = []
    image_digest = hashlib.sha256()
    unsafe_image_paths: set[str] = set()

    for paper_id in sorted(candidate_ids):
        chunks = store.load_paper(paper_id)
        if not chunks:
            missing_papers.append(paper_id)
            valid_types_by_paper[paper_id] = set()
            image_types_by_paper[paper_id] = set()
            continue
        valid_types: set[str] = set()
        image_types: set[str] = set()
        for chunk in chunks:
            source_type = record_source_type(chunk)
            chunk_types[source_type] += 1
            metadata = chunk.get("metadata") or {}
            page = metadata.get("page")
            page_valid = (
                not isinstance(page, bool) and isinstance(page, int) and page >= 1
            )
            if source_type in OFFICIAL_SOURCE_TYPES and not page_valid:
                invalid_pages[source_type] += 1
            if source_type == "table" and not _nonempty_string(
                metadata.get("table_id")
            ):
                invalid_object_ids[source_type] += 1
            if source_type == "figure" and not _nonempty_string(
                metadata.get("figure_id")
            ):
                invalid_object_ids[source_type] += 1
            if submission_evidence_eligible(chunk):
                valid_types.add(source_type)

            if source_type in {"table", "figure"}:
                image_eligible += 1
                path_error = metadata.get(IMAGE_PATH_ERROR_KEY)
                if path_error:
                    original = str(metadata.get(IMAGE_PATH_ORIGINAL_KEY) or "")
                    image_declared += 1
                    unsafe_key = original or f"{paper_id}:{chunk.get('chunk_id')}"
                    if unsafe_key not in unsafe_image_paths:
                        unsafe_image_paths.add(unsafe_key)
                        if len(unsafe_image_examples) < 20:
                            unsafe_image_examples.append(
                                {
                                    "path": original,
                                    "reason": str(path_error),
                                }
                            )
                    continue
                raw_path = metadata.get("image_path")
                if not _nonempty_string(raw_path):
                    continue
                image_declared += 1
                image_path = Path(str(raw_path))
                resolved = str(image_path.resolve())
                if resolved in image_paths_seen:
                    if resolved in readable_image_paths:
                        image_types.add(source_type)
                    continue
                image_paths_seen.add(resolved)
                if not image_path.is_file():
                    if len(missing_image_examples) < 20:
                        missing_image_examples.append(resolved)
                    continue
                try:
                    validate_image_file(image_path)
                    file_digest = _sha256_image(image_path)
                    stat = image_path.stat()
                except (OSError, ValueError):
                    image_unreadable += 1
                    if len(unreadable_image_examples) < 20:
                        unreadable_image_examples.append(resolved)
                    continue
                image_existing += 1
                readable_image_paths.add(resolved)
                image_types.add(source_type)
                image_digest.update(resolved.encode("utf-8"))
                image_digest.update(b"\0")
                image_digest.update(str(stat.st_size).encode("ascii"))
                image_digest.update(b"\0")
                image_digest.update(file_digest.encode("ascii"))
                image_digest.update(b"\n")
        valid_types_by_paper[paper_id] = valid_types
        image_types_by_paper[paper_id] = image_types
        if not valid_types:
            papers_without_valid_evidence.append(paper_id)

    queries_without_valid_evidence: list[str] = []
    missing_source_hints: list[dict[str, str]] = []
    visual_image_required_queries: list[str] = []
    queries_without_required_visual_images: list[str] = []
    for handoff in handoffs:
        paper_ids = [paper.paper_id for paper in handoff.candidate_papers]
        union_types: set[str] = set()
        union_image_types: set[str] = set()
        for paper_id in paper_ids:
            union_types.update(valid_types_by_paper.get(paper_id, set()))
            union_image_types.update(image_types_by_paper.get(paper_id, set()))
        if not union_types:
            queries_without_valid_evidence.append(handoff.query.query_id)
        for source_type in sorted(_source_modality_hints(handoff)):
            if source_type not in union_types:
                missing_source_hints.append(
                    {
                        "query_id": handoff.query.query_id,
                        "source_type": source_type,
                    }
                )
        if requires_visual_image(handoff.query.question):
            visual_image_required_queries.append(handoff.query.query_id)
            if "figure" not in union_image_types:
                queries_without_required_visual_images.append(
                    handoff.query.query_id
                )

    corpus_ids = set(store.paper_ids())
    canonical_missing: list[str] = []
    corpus_extra: list[str] = []
    if canonical_paper_ids is not None:
        canonical_missing = sorted(canonical_paper_ids - corpus_ids)
        corpus_extra = sorted(corpus_ids - canonical_paper_ids)

    errors: list[str] = []
    warnings: list[str] = []
    if canonical_missing:
        errors.append(
            f"{len(canonical_missing)} canonical papers are missing from corpus"
        )
    if missing_papers:
        errors.append(f"{len(missing_papers)} candidate papers are missing from corpus")
    if queries_without_valid_evidence:
        errors.append(
            f"{len(queries_without_valid_evidence)} queries have no valid evidence locator"
        )
    if missing_source_hints:
        warnings.append(
            f"{len(missing_source_hints)} query/source hints have no matching "
            "submission-eligible chunk; source hints are diagnostic only"
        )
    if queries_without_required_visual_images:
        message = (
            f"{len(queries_without_required_visual_images)} explicit visual-reading "
            "queries have no readable candidate figure/chart image"
        )
        if allow_missing_figure_images:
            warnings.append(
                message
                + " (allowed by --allow-missing-required-visual-images after "
                "global image-path validation)"
            )
        else:
            errors.append(message)
    if unsafe_image_paths:
        errors.append(
            f"{len(unsafe_image_paths)} table/figure image paths are unsafe; "
            "every declared image must be rebased inside an explicit image_root "
            "using paper_id/auto/images/filename"
        )
    image_missing = len(image_paths_seen) - image_existing - image_unreadable
    image_without_path = image_eligible - image_declared
    if image_paths_seen and image_existing == 0:
        unique_declared = len(image_paths_seen) + len(unsafe_image_paths)
        errors.append(
            "all declared table/figure images are unavailable "
            f"(0 of {unique_declared} unique paths readable); check "
            "--image-root. This global configuration failure cannot be "
            "overridden by --allow-missing-required-visual-images"
        )
    if image_without_path:
        warnings.append(f"{image_without_path} table/figure chunks have no image_path")
    if image_unreadable:
        warnings.append(
            f"{image_unreadable} declared image files are unreadable or corrupt"
        )
    if image_missing:
        warnings.append(f"{image_missing} declared image files do not exist")
    if chunk_types.get("citation_context", 0) == 0:
        warnings.append(
            "citation_context is absent; reference questions need a References parser"
        )
    if corpus_extra:
        warnings.append(f"{len(corpus_extra)} corpus paper IDs are absent from metadata")

    report = {
        "allow_missing_figure_images": allow_missing_figure_images,
        "queries": len(handoffs),
        "candidate_entries": sum(
            len(handoff.candidate_papers) for handoff in handoffs
        ),
        "unique_candidate_papers": len(candidate_ids),
        "corpus_papers": len(corpus_ids),
        "canonical_papers": (
            len(canonical_paper_ids) if canonical_paper_ids is not None else None
        ),
        "canonical_papers_missing_from_corpus": canonical_missing,
        "corpus_papers_absent_from_metadata": corpus_extra,
        "missing_candidate_papers": missing_papers,
        "papers_without_valid_evidence": papers_without_valid_evidence,
        "queries_without_valid_evidence": queries_without_valid_evidence,
        "missing_source_hints": missing_source_hints,
        # Backwards-compatible report aliases. These are warnings/diagnostics,
        # not hard source requirements under the current input contract.
        "missing_required_modalities": missing_source_hints,
        "visual_image_required_queries": visual_image_required_queries,
        "queries_without_required_visual_images": (
            queries_without_required_visual_images
        ),
        "queries_without_figure_images": queries_without_required_visual_images,
        "chunk_types": dict(sorted(chunk_types.items())),
        "invalid_page_by_type": dict(sorted(invalid_pages.items())),
        "invalid_object_id_by_type": dict(sorted(invalid_object_ids.items())),
        "image_paths": {
            "eligible_chunks": image_eligible,
            "declared": image_declared,
            "without_path": image_without_path,
            "unique_declared": len(image_paths_seen) + len(unsafe_image_paths),
            "existing": image_existing,
            "missing": image_missing,
            "unreadable": image_unreadable,
            "unsafe": len(unsafe_image_paths),
            "content_sha256": image_digest.hexdigest() if image_existing else None,
            "missing_examples": missing_image_examples,
            "unreadable_examples": unreadable_image_examples,
            "unsafe_examples": unsafe_image_examples,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return report, errors


def _source_modality_hints(handoff: CandidateHandoff) -> set[str]:
    """Return non-binding source hints inferred from observable query text.

    ``answer_types`` describes the output representation, not where evidence
    must be found. A table-shaped answer may be assembled from prose,
    equations, citations, or several papers, so it must never imply that a
    table chunk is required.
    """

    query = handoff.query
    text = query.question
    hints: set[str] = set()
    if _TABLE_RE.search(text):
        hints.add("table")
    if _FIGURE_RE.search(text):
        hints.add("figure")
    if _EQUATION_RE.search(text):
        hints.add("equation_algorithm")
    if _CITATION_RE.search(text):
        hints.add("citation_context")
    return hints


def requires_visual_image(question: str) -> bool:
    """Return whether the observable wording explicitly requires visual reading."""

    return any(pattern.search(question) for pattern in _VISUAL_IMAGE_REQUIRED_PATTERNS)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
