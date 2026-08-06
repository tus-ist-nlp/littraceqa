"""Canonical interpretation of one MinerU chunk for LitTraceQA output."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from littraceqa.chunk_store import Record
from littraceqa.submission import OFFICIAL_SOURCE_TYPES, normalize_visible_id


# Azure/OpenAI image inputs have a finite request-size budget.  Refuse a single
# unexpectedly huge corpus file before ``read_bytes`` can allocate it or upload
# it.  The real student bundle is well below one MiB per extracted image.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
# Azure OpenAI Chat Completions currently accepts at most ten images in one
# request. Keep both reader batching and the final adapter guard on one value.
MAX_AOAI_IMAGES_PER_REQUEST = 10

_IMAGE_MIME_BY_FORMAT = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidationError(ValueError):
    """A local file exists but is not a supported, structurally valid image."""


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
    if not path:
        return ""
    try:
        resolved, identity = _image_file_identity(path)
    except (OSError, RuntimeError):
        return ""
    return path if _cached_image_is_valid(resolved, identity) else ""


def _image_file_identity(
    image_path: str | Path,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    """Return a cache key that changes when a local image is replaced.

    This cache is only a performance aid for repeated local readability checks;
    it is not an upload security boundary.  Device/inode identify the file,
    size detects growth or truncation, and nanosecond mtime/ctime detect normal
    in-place content or metadata changes.  The AOAI adapter deliberately calls
    the uncached ``validate_image_file`` and then validates the bytes it actually
    read immediately before encoding them.
    """

    path = Path(image_path).resolve(strict=True)
    stat = path.stat()
    identity = (
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    return str(path), identity


@lru_cache(maxsize=8192)
def _cached_image_is_valid(
    resolved_path: str,
    identity: tuple[int, int, int, int, int, int],
) -> bool:
    """Fully decode an unchanged local image at most once per file identity."""

    # ``identity`` participates in the cache key even though validation itself
    # only needs the path.  Keep the explicit binding to make that safety
    # property apparent and prevent a future simplification from dropping it.
    del identity
    try:
        validate_image_file(resolved_path)
    except (OSError, ImageValidationError):
        return False
    return True


def validate_image_file(
    image_path: str | Path, *, max_bytes: int = MAX_IMAGE_BYTES
) -> str:
    """Validate one local image and return its content-derived MIME type.

    This deliberately does not trust a filename extension.  It performs the
    same bounded structural checks used immediately before AOAI upload, so an
    image rejected by preflight cannot later be sent merely because it exists.
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    stat = path.stat()
    if stat.st_size <= 0:
        raise ImageValidationError(f"image is empty: {path}")
    if stat.st_size > max_bytes:
        raise ImageValidationError(
            f"image exceeds {max_bytes} byte limit: {path} ({stat.st_size} bytes)"
        )
    # Bound the read independently of ``stat`` in case the file is replaced or
    # grows between the two operations.
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    return validate_image_bytes(payload, source=str(path), max_bytes=max_bytes)


def validate_image_bytes(
    payload: bytes, *, source: str = "image", max_bytes: int = MAX_IMAGE_BYTES
) -> str:
    """Validate already-read image bytes and return a safe MIME type."""

    size = len(payload)
    if size <= 0:
        raise ImageValidationError(f"image is empty: {source}")
    if size > max_bytes:
        raise ImageValidationError(
            f"image exceeds {max_bytes} byte limit: {source} ({size} bytes)"
        )
    return _decode_and_validate_image(payload, source=source)


def _decode_and_validate_image(payload: bytes, *, source: str) -> str:
    """Fully decode a supported image and return its content-derived MIME."""

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - packaging regression guard
        raise RuntimeError(
            "Pillow is required for image validation; install the "
            "pairwise_reader extra"
        ) from exc

    try:
        # ``verify`` checks container integrity without decoding pixels. Reopen
        # and ``load`` as well so files with a valid-looking header/trailer but
        # corrupt compressed data can never reach AOAI.
        with Image.open(BytesIO(payload)) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            if image_format not in _IMAGE_MIME_BY_FORMAT:
                raise ImageValidationError(
                    f"unsupported image format {image_format or 'unknown'}: {source}"
                )
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    f"image dimensions exceed safety limit: {source} "
                    f"({width}x{height}, max {MAX_IMAGE_PIXELS} pixels)"
                )
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ImageValidationError(f"unsupported/corrupt image: {source}") from exc
    return _IMAGE_MIME_BY_FORMAT[image_format]


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
