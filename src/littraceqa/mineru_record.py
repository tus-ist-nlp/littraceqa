"""Canonical interpretation of one MinerU chunk for LitTraceQA output."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re
from typing import Any
import unicodedata

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

# ``coarse_locator`` deliberately remains a corpus-only interpretation.  This
# separate versioned helper is used at the final prediction boundary, where an
# observable query can disambiguate a known MinerU failure mode: one table or
# figure chunk occasionally contains two consecutive captions while metadata
# retains the first object's ID for the second object's body.
QUERY_AWARE_OBJECT_LOCATOR_VERSION = "merged-caption-query-v1"

# Some MinerU table crops are emitted one record late: the first record has the
# table body but no ID, while the next same-page record starts with that table's
# caption and then contains the following table's caption and body. Recovery
# needs paper-order adjacency, so it is deliberately separate from the
# single-record ``coarse_locator`` API.
SPLIT_CAPTION_TABLE_LOCATOR_VERSION = "split-caption-adjacent-v1"

_MERGED_CAPTION_RE = {
    "table": re.compile(
        r"(?im)(?:^|\n)\s*table\s*(?:no\.?\s*)?(\d+[a-z]?)\s*(?=[:.]|\n)"
    ),
    "figure": re.compile(
        r"(?im)(?:^|\n)\s*(?:figure|fig\.?)\s*"
        r"(?:no\.?\s*)?(\d+[a-z]?)\s*(?=[:.]|\n)"
    ),
}

_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"(?m)^\s*\|(?:\s*:?-{3,}:?\s*\|){2,}\s*$"
)
_MINERU_DOCUMENT_HEADER_RE = re.compile(r"^\[[^\]\n]{2,40}\]\s+\S.*$")

# These words describe the QA request or scientific results in general.  A
# match on one of them is not enough to decide which of two merged objects the
# question targets.  Recovery instead requires a discriminating token such as
# a dataset, method, metric variant, or other named condition.
_GENERIC_LOCATOR_QUERY_TERMS = frozenset(
    {
        "about",
        "according",
        "answer",
        "benchmark",
        "between",
        "column",
        "compare",
        "comparison",
        "dataset",
        "datasets",
        "difference",
        "evaluation",
        "figure",
        "following",
        "higher",
        "lower",
        "method",
        "methods",
        "metric",
        "metrics",
        "model",
        "models",
        "paper",
        "performance",
        "reported",
        "result",
        "results",
        "score",
        "scores",
        "setting",
        "shown",
        "table",
        "under",
        "value",
        "values",
        "which",
        "what",
        "where",
        "when",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "does",
        "have",
        "into",
        "than",
        "terms",
        "f1",
        "accuracy",
    }
)

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


def query_aware_prediction_locator(
    record: Record,
    question: str,
) -> dict[str, Any]:
    """Return a final-output locator with conservative merged-object recovery.

    MinerU can merge adjacent table/figure blocks into one chunk.  In the
    affected records the text contains multiple real caption starts, but the
    metadata object ID can describe the earlier caption while the requested
    body belongs to a later one.  We change the object ID only when all of the
    following are observable without labels:

    * the record is a table or figure and has at least two distinct caption IDs;
    * its metadata ID is one of those captions (the signature of a merge, not
      an arbitrary missing/corrupt ID); and
    * either the question explicitly names exactly one present object, or one
      caption segment has a unique, non-generic query-token match and no tie.

    Clean single-caption records and ambiguous merged records retain the
    canonical metadata locator unchanged.  This function is intentionally not
    used for Stage-1 routing, prompt construction, or corpus eligibility.
    """

    locator = coarse_locator(record)
    source_type = record_source_type(record)
    if source_type not in _MERGED_CAPTION_RE:
        return locator

    object_field = "table_id" if source_type == "table" else "figure_id"
    prefix = "Table" if source_type == "table" else "Figure"
    metadata_key = normalize_visible_id(locator.get(object_field), source_type)
    if not metadata_key:
        return locator

    text = str(record.get("text") or "")
    matches = list(_MERGED_CAPTION_RE[source_type].finditer(text))
    if len(matches) < 2:
        return locator

    segments_by_key: dict[str, list[str]] = {}
    display_id_by_key: dict[str, str] = {}
    for index, match in enumerate(matches):
        display_id = f"{prefix} {match.group(1)}"
        key = normalize_visible_id(display_id, source_type)
        if not key:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments_by_key.setdefault(key, []).append(text[match.start() : end])
        display_id_by_key.setdefault(key, display_id)

    if len(segments_by_key) < 2 or metadata_key not in segments_by_key:
        return locator

    explicit_ids = {
        normalize_visible_id(f"{prefix} {match.group(1)}", source_type)
        for match in _explicit_query_object_re(source_type).finditer(question)
    }
    present_explicit_ids = explicit_ids.intersection(segments_by_key)
    if len(explicit_ids) == 1 and len(present_explicit_ids) == 1:
        selected_key = next(iter(present_explicit_ids))
    else:
        query_terms = _discriminating_locator_terms(question, record)
        if not query_terms:
            return locator
        segment_terms = {
            key: _locator_terms("\n".join(segments))
            for key, segments in segments_by_key.items()
        }
        document_frequency = {
            term: sum(term in terms for terms in segment_terms.values())
            for term in query_terms
        }
        unique_matches = {
            key: {
                term
                for term in query_terms.intersection(terms)
                if document_frequency[term] == 1
            }
            for key, terms in segment_terms.items()
        }
        best_score = max((len(terms) for terms in unique_matches.values()), default=0)
        if best_score < 1:
            return locator
        winners = [
            key for key, terms in unique_matches.items() if len(terms) == best_score
        ]
        if len(winners) != 1:
            return locator
        selected_key = winners[0]

    if selected_key == metadata_key:
        return locator
    recovered = dict(locator)
    recovered[object_field] = display_id_by_key[selected_key]
    return recovered


def split_caption_table_locator_overrides(
    records: Iterable[Record],
) -> dict[str, str]:
    """Recover table IDs from a conservative adjacent split-caption pattern.

    The rule uses only original corpus structure. Two records must be adjacent
    tables from the same paper and page. The earlier record must contain a
    Markdown table body but no caption or metadata ID. The later record must:

    * begin, after the standard repeated MinerU document header, with at least
      two distinct table captions before its own Markdown table body; and
    * have no metadata ID, or have an ID matching the first caption.

    In that narrow layout, the first caption belongs to the preceding body and
    the body-nearest final caption belongs to the later body. The returned map
    is keyed by exact chunk ID and never mutates the input records.
    """

    ordered = list(records)
    overrides: dict[str, str] = {}
    for earlier, later in zip(ordered, ordered[1:]):
        if (
            record_source_type(earlier) != "table"
            or record_source_type(later) != "table"
            or str(earlier.get("paper_id") or "")
            != str(later.get("paper_id") or "")
        ):
            continue

        earlier_metadata = earlier.get("metadata") or {}
        later_metadata = later.get("metadata") or {}
        earlier_page = _valid_page(earlier_metadata.get("page"))
        later_page = _valid_page(later_metadata.get("page"))
        if earlier_page is None or earlier_page != later_page:
            continue
        if normalize_visible_id(earlier_metadata.get("table_id"), "table"):
            continue

        earlier_text = str(earlier.get("text") or "")
        if (
            not _MARKDOWN_TABLE_SEPARATOR_RE.search(earlier_text)
            or _MERGED_CAPTION_RE["table"].search(earlier_text)
        ):
            continue

        later_text = str(later.get("text") or "")
        body_match = _MARKDOWN_TABLE_SEPARATOR_RE.search(later_text)
        if body_match is None:
            continue
        caption_matches = [
            match
            for match in _MERGED_CAPTION_RE["table"].finditer(later_text)
            if match.start() < body_match.start()
        ]
        if len(caption_matches) < 2:
            continue
        if not _caption_starts_after_mineru_header(
            later_text[: caption_matches[0].start()]
        ):
            continue

        first_key = normalize_visible_id(
            f"Table {caption_matches[0].group(1)}", "table"
        )
        last_key = normalize_visible_id(
            f"Table {caption_matches[-1].group(1)}", "table"
        )
        distinct_keys = {
            normalize_visible_id(f"Table {match.group(1)}", "table")
            for match in caption_matches
        }
        if not first_key or not last_key or len(distinct_keys) < 2:
            continue
        later_metadata_key = normalize_visible_id(
            later_metadata.get("table_id"), "table"
        )
        if later_metadata_key and later_metadata_key != first_key:
            continue

        earlier_chunk_id = str(earlier.get("chunk_id") or "")
        later_chunk_id = str(later.get("chunk_id") or "")
        if not earlier_chunk_id or not later_chunk_id:
            continue
        overrides[earlier_chunk_id] = f"Table {caption_matches[0].group(1)}"
        overrides[later_chunk_id] = f"Table {caption_matches[-1].group(1)}"
    return overrides


def recover_split_caption_table_records(
    records: Iterable[Record],
) -> list[Record]:
    """Return shallow record copies with safe adjacent table IDs recovered."""

    ordered = list(records)
    overrides = split_caption_table_locator_overrides(ordered)
    if not overrides:
        return ordered
    recovered: list[Record] = []
    for record in ordered:
        table_id = overrides.get(str(record.get("chunk_id") or ""))
        if table_id is None:
            recovered.append(record)
            continue
        copied = dict(record)
        metadata = dict(record.get("metadata") or {})
        metadata["table_id"] = table_id
        copied["metadata"] = metadata
        recovered.append(copied)
    return recovered


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


@lru_cache(maxsize=2)
def _explicit_query_object_re(source_type: str) -> re.Pattern[str]:
    label = r"table" if source_type == "table" else r"(?:figure|fig\.?)"
    return re.compile(
        rf"(?i)\b{label}\s*(?:no\.?\s*)?(\d+[a-z]?)\b"
    )


def _locator_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        variants = {token}
        # Table cells commonly concatenate a method and setting (``ICAE500``).
        # Retain the full token but also expose the alphabetic method prefix so
        # it can be recognized as shared across caption segments.
        if match := re.fullmatch(r"([a-z]{4,})\d+", token):
            variants.add(match.group(1))
        terms.update(
            variant
            for variant in variants
            if (
                len(variant) >= 4
                and not variant.isdigit()
                and variant not in _GENERIC_LOCATOR_QUERY_TERMS
            )
        )
    return terms


def _discriminating_locator_terms(question: str, record: Record) -> set[str]:
    terms = _locator_terms(question)
    # A paper title identifies the owner, not one object inside that paper.
    # Excluding title terms also handles MinerU spacing noise such as
    # ``500x C ompressor`` versus the query's ``500xCompressor``.
    title = str((record.get("metadata") or {}).get("title") or "")
    compact_title = re.sub(
        r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", title).lower()
    )
    return {
        term
        for term in terms
        if not compact_title or term not in compact_title
    }


def _caption_starts_after_mineru_header(prefix: str) -> bool:
    stripped = prefix.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return len(lines) == 1 and bool(_MINERU_DOCUMENT_HEADER_RE.fullmatch(lines[0]))


def _put_nonempty(locator: dict[str, Any], key: str, value: Any) -> None:
    cleaned = _clean_string(value)
    if cleaned:
        locator[key] = cleaned


def _valid_page(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value
