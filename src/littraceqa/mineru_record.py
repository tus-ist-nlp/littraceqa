"""Canonical interpretation of one MinerU chunk for LitTraceQA output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from littraceqa.chunk_store import Record
from littraceqa.submission import OFFICIAL_SOURCE_TYPES


def record_source_type(record: Record) -> str:
    """Map MinerU chunk vocabulary to an official evidence source type."""
    chunk_type = str(record.get("chunk_type") or "")
    if chunk_type == "title_abstract":
        chunk_type = "text_span"
    metadata = record.get("metadata") or {}
    section = str(metadata.get("section") or "").strip().lower()
    if chunk_type == "text_span" and (
        metadata.get("citation_id")
        or section in {"references", "bibliography"}
        or section.startswith("references ")
    ):
        return "citation_context"
    return chunk_type


def coarse_locator(record: Record) -> dict[str, Any]:
    """Build the locator fields used by the official coarse evidence metric."""
    metadata = record.get("metadata") or {}
    locator: dict[str, Any] = {"page": metadata.get("page")}
    source_type = record_source_type(record)
    if source_type == "table":
        locator["table_id"] = metadata.get("table_id")
    elif source_type == "figure":
        locator["figure_id"] = metadata.get("figure_id")
    return locator


def submission_evidence_eligible(record: Record) -> bool:
    """Return whether a chunk has every field required for submission."""
    source_type = record_source_type(record)
    if source_type not in OFFICIAL_SOURCE_TYPES:
        return False
    metadata = record.get("metadata") or {}
    page = metadata.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return False
    if source_type == "table" and not _nonempty_string(metadata.get("table_id")):
        return False
    return source_type != "figure" or _nonempty_string(metadata.get("figure_id"))


def readable_image_path(record: Record) -> str:
    """Return a readable table/figure image path, otherwise an empty string."""
    if str(record.get("chunk_type") or "") not in {"table", "figure"}:
        return ""
    path = str((record.get("metadata") or {}).get("image_path") or "")
    return path if path and Path(path).is_file() else ""


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
