"""Gold-free integrity checks for a MinerU corpus and candidate handoff."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from littraceqa.candidate_handoff import CandidateHandoff
from littraceqa.chunk_store import ChunkStore
from littraceqa.mineru_record import record_source_type, submission_evidence_eligible
from littraceqa.submission import OFFICIAL_SOURCE_TYPES

_TABLE_RE = re.compile(r"\b(table|row|column)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\b(figure|fig\.?|plot|chart|diagram)\b", re.IGNORECASE)
_EQUATION_RE = re.compile(
    r"\b(equation|objective|loss function|formula|algorithm)\b", re.IGNORECASE
)
_CITATION_RE = re.compile(
    r"\b(citation|cited|reference|bibliography)\b", re.IGNORECASE
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
    image_digest = hashlib.sha256()

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
    missing_required_modalities: list[dict[str, str]] = []
    queries_without_figure_images: list[str] = []
    for handoff in handoffs:
        paper_ids = [paper.paper_id for paper in handoff.candidate_papers]
        union_types: set[str] = set()
        union_image_types: set[str] = set()
        for paper_id in paper_ids:
            union_types.update(valid_types_by_paper.get(paper_id, set()))
            union_image_types.update(image_types_by_paper.get(paper_id, set()))
        if not union_types:
            queries_without_valid_evidence.append(handoff.query.query_id)
        for source_type in sorted(_required_modalities(handoff)):
            if source_type not in union_types:
                missing_required_modalities.append(
                    {
                        "query_id": handoff.query.query_id,
                        "source_type": source_type,
                    }
                )
        if "figure" in _required_modalities(handoff) and "figure" not in union_image_types:
            queries_without_figure_images.append(handoff.query.query_id)

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
    if missing_required_modalities:
        errors.append(
            f"{len(missing_required_modalities)} query/modality requirements have no valid locator"
        )
    if queries_without_figure_images:
        message = (
            f"{len(queries_without_figure_images)} figure queries have no readable "
            "candidate image"
        )
        if allow_missing_figure_images:
            warnings.append(
                message + " (allowed by --allow-missing-figure-images)"
            )
        else:
            errors.append(message)
    if image_unreadable:
        errors.append(
            f"{image_unreadable} declared image files are unreadable or corrupt"
        )

    image_missing = len(image_paths_seen) - image_existing - image_unreadable
    image_without_path = image_eligible - image_declared
    if image_without_path:
        warnings.append(f"{image_without_path} table/figure chunks have no image_path")
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
        "missing_required_modalities": missing_required_modalities,
        "queries_without_figure_images": queries_without_figure_images,
        "chunk_types": dict(sorted(chunk_types.items())),
        "invalid_page_by_type": dict(sorted(invalid_pages.items())),
        "invalid_object_id_by_type": dict(sorted(invalid_object_ids.items())),
        "image_paths": {
            "eligible_chunks": image_eligible,
            "declared": image_declared,
            "without_path": image_without_path,
            "unique_declared": len(image_paths_seen),
            "existing": image_existing,
            "missing": image_missing,
            "unreadable": image_unreadable,
            "content_sha256": image_digest.hexdigest() if image_existing else None,
            "missing_examples": missing_image_examples,
            "unreadable_examples": unreadable_image_examples,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return report, errors


def _required_modalities(handoff: CandidateHandoff) -> set[str]:
    query = handoff.query
    text = query.question
    required: set[str] = set()
    if "table" in query.answer_types or _TABLE_RE.search(text):
        required.add("table")
    if _FIGURE_RE.search(text):
        required.add("figure")
    if _EQUATION_RE.search(text):
        required.add("equation_algorithm")
    if _CITATION_RE.search(text):
        required.add("citation_context")
    return required or {"text_span"}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header = handle.read(16)
        if not _supported_image_header(header):
            raise ValueError(f"unsupported/corrupt image: {path}")
        digest.update(header)
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _supported_image_header(header: bytes) -> bool:
    return bool(
        header.startswith(
            (
                b"\xff\xd8\xff",
                b"\x89PNG\r\n\x1a\n",
                b"GIF87a",
                b"GIF89a",
                b"BM",
                b"II*\x00",
                b"MM\x00*",
            )
        )
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
    )
