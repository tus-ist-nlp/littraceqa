"""Canonical interpretation of one MinerU chunk for LitTraceQA output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from littraceqa.chunk_store import Record
from littraceqa.submission import OFFICIAL_SOURCE_TYPES, normalize_visible_id


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
    locator: dict[str, Any] = {}
    page = _valid_page(metadata.get("page"))
    section = _clean_string(metadata.get("section"))
    if page is not None:
        locator["page"] = page
    elif section:
        locator["section"] = section

    source_type = record_source_type(record)
    if source_type == "table":
        _put_nonempty(locator, "table_id", metadata.get("table_id"))
    elif source_type == "figure":
        _put_nonempty(locator, "figure_id", metadata.get("figure_id"))
    elif source_type == "equation_algorithm":
        equation_id = _clean_string(metadata.get("equation_id"))
        algorithm_id = _clean_string(metadata.get("algorithm_id"))
        if equation_id:
            locator["equation_id"] = equation_id
        elif algorithm_id:
            locator["algorithm_id"] = algorithm_id
    elif source_type == "citation_context":
        _put_nonempty(locator, "citation_id", metadata.get("citation_id"))
    return locator


def submission_evidence_eligible(record: Record) -> bool:
    """Return whether a chunk has every field required for submission."""
    source_type = record_source_type(record)
    if source_type not in OFFICIAL_SOURCE_TYPES:
        return False
    locator = coarse_locator(record)
    has_page = "page" in locator
    has_location = has_page or bool(locator.get("section"))
    if source_type == "table":
        return has_location and bool(
            normalize_visible_id(locator.get("table_id"), "table")
        )
    if source_type == "figure":
        return has_location and bool(
            normalize_visible_id(locator.get("figure_id"), "figure")
        )
    if source_type == "equation_algorithm":
        return has_location or bool(
            normalize_visible_id(
                locator.get("equation_id") or locator.get("algorithm_id"),
                "equation",
            )
        )
    if source_type == "citation_context":
        return has_location or bool(
            normalize_visible_id(locator.get("citation_id"), "citation")
        )
    return has_location


def readable_image_path(record: Record) -> str:
    """Return a readable table/figure image path, otherwise an empty string."""
    if str(record.get("chunk_type") or "") not in {"table", "figure"}:
        return ""
    path = str((record.get("metadata") or {}).get("image_path") or "")
    return path if path and Path(path).is_file() else ""


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _put_nonempty(locator: dict[str, Any], key: str, value: Any) -> None:
    cleaned = _clean_string(value)
    if cleaned:
        locator[key] = cleaned


def _valid_page(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value
