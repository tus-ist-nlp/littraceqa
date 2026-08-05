"""Strict serializer for the official LitTraceQA submission shape."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from littraceqa.di_pipeline.contracts import Prediction, Query

OFFICIAL_SOURCE_TYPES = frozenset(
    {"text_span", "table", "figure", "citation_context", "equation_algorithm"}
)
TOP_LEVEL_KEYS = frozenset({"query_id", "gold_papers", "evidence", "answer"})
TOP_LEVEL_KEYS_WITHOUT_EVIDENCE = frozenset(
    {"query_id", "gold_papers", "answer"}
)
# Legacy inputs in this repository omitted option text and assumed four labels.
# Current official inputs provide the valid labels per query and may include E.
MULTIPLE_CHOICE_CHOICES = "ABCD"
MULTIPLE_CHOICE_KEYS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_visible_id(value: Any, prefix: str) -> str:
    """Normalize a locator object ID exactly as the official evaluator does."""

    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    normalized_prefix = prefix.lower()
    match = re.fullmatch(rf"{normalized_prefix}\s*(\d+[a-z]?)", text)
    if match:
        return f"{normalized_prefix} {match.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{normalized_prefix} {text}"
    return text


def coarse_evidence_key(
    paper_id: str,
    source_type: str,
    locator: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Return the official evaluator's deduplication/comparison key."""

    location = str(locator.get("page") or locator.get("section") or "").strip()
    object_id = ""
    if source_type == "table":
        object_id = normalize_visible_id(locator.get("table_id"), "table")
    elif source_type == "figure":
        object_id = normalize_visible_id(locator.get("figure_id"), "figure")
    elif source_type == "equation_algorithm":
        object_id = normalize_visible_id(
            locator.get("equation_id") or locator.get("algorithm_id"),
            "equation",
        )
    elif source_type == "citation_context":
        object_id = normalize_visible_id(locator.get("citation_id"), "citation")
    return (paper_id.strip(), source_type.strip(), location, object_id)


def deterministic_mc_letter(
    query_id: str, choices: str = MULTIPLE_CHOICE_CHOICES
) -> str:
    """Return a reproducible compatibility fallback when options are unavailable.

    Current official inputs always include options for multiple-choice questions,
    so production serialization never uses this fallback. It remains for old
    local validation snapshots only and is not a grounded answer.
    """

    if not choices:
        raise ValueError("choices must not be empty")
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def prediction_to_submission(
    query: Query,
    prediction: Prediction,
    *,
    require_evidence: bool = True,
) -> dict[str, Any]:
    """Drop analysis fields and normalize one prediction for submission.

    Evidence is required for the scored ``test`` split. ``test_extra`` does not
    score it, so callers may set ``require_evidence=False`` and omit an empty
    evidence list rather than manufacturing a locator.
    """

    if prediction.query_id != query.query_id:
        raise ValueError(
            f"prediction/query id mismatch: {prediction.query_id} != {query.query_id}"
        )

    papers = _normalize_papers(prediction.gold_papers)
    if not papers:
        raise ValueError(f"{query.query_id}: prediction has no papers")
    paper_ids = {item["paper_id"] for item in papers}

    evidence = _normalize_evidence(prediction.evidence, paper_ids)
    if require_evidence and not evidence:
        raise ValueError(f"{query.query_id}: prediction has no valid evidence")

    answer = _normalize_answer(query, prediction)
    output = {
        "query_id": query.query_id,
        "gold_papers": papers,
        "answer": answer,
    }
    if evidence or require_evidence:
        output["evidence"] = evidence
    return output


def _normalize_papers(raw_papers: list[dict[str, str]]) -> list[dict[str, str]]:
    papers: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_papers:
        paper_id = str(item.get("paper_id") or "") if isinstance(item, dict) else ""
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            papers.append({"paper_id": paper_id})
    return papers


def _normalize_evidence(evidence_items: list, paper_ids: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for evidence in evidence_items:
        paper_id = str(evidence.paper_id or "")
        source_type = str(evidence.source_type or "")
        if paper_id not in paper_ids or source_type not in OFFICIAL_SOURCE_TYPES:
            continue
        raw_locator = evidence.locator.to_dict()
        page = _normalize_page(raw_locator.get("page"))
        locator: dict[str, Any] = {}
        if page is not None:
            locator["page"] = page
        else:
            section = str(raw_locator.get("section") or "").strip()
            if section:
                locator["section"] = section
        if source_type == "table":
            table_id = str(raw_locator.get("table_id") or "").strip()
            if not locator or not table_id:
                continue
            locator["table_id"] = table_id
        elif source_type == "figure":
            figure_id = str(raw_locator.get("figure_id") or "").strip()
            if not locator or not figure_id:
                continue
            locator["figure_id"] = figure_id
        elif source_type == "text_span":
            if not locator:
                continue
        elif source_type == "equation_algorithm":
            equation_id = str(raw_locator.get("equation_id") or "").strip()
            algorithm_id = str(raw_locator.get("algorithm_id") or "").strip()
            if equation_id:
                locator["equation_id"] = equation_id
            elif algorithm_id:
                locator["algorithm_id"] = algorithm_id
            if not locator:
                continue
        elif source_type == "citation_context":
            citation_id = str(raw_locator.get("citation_id") or "").strip()
            if citation_id:
                locator["citation_id"] = citation_id
            if not locator:
                continue
        key = coarse_evidence_key(paper_id, source_type, locator)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "paper_id": paper_id,
                "source_type": source_type,
                "locator": locator,
            }
        )
    return output


def _normalize_answer(query: Query, prediction: Prediction) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    requested = list(dict.fromkeys(query.answer_types))

    if "freeform" in requested:
        raw = prediction.answer.freeform
        text = str(raw.get("text") or "").strip() if isinstance(raw, dict) else ""
        if not text:
            raise ValueError(f"{query.query_id}: freeform answer is empty")
        answer["freeform"] = {"text": text}

    if "multiple_choice" in requested:
        raw = prediction.answer.multiple_choice
        letter = str(raw.get("gold") or "").strip().upper() if isinstance(raw, dict) else ""
        valid_labels = query.option_labels
        if valid_labels and letter not in valid_labels:
            raise ValueError(
                f"{query.query_id}: multiple_choice label {letter!r} is not one of "
                f"{list(valid_labels)}"
            )
        if not valid_labels and letter not in MULTIPLE_CHOICE_CHOICES:
            letter = deterministic_mc_letter(query.query_id)
        answer["multiple_choice"] = {"gold": letter}

    if "table" in requested:
        raw = prediction.answer.table
        rows = raw.get("rows") if isinstance(raw, dict) else None
        normalized_rows = _normalize_table_rows(rows, query.table_schema or [])
        if not normalized_rows:
            raise ValueError(f"{query.query_id}: table answer has no valid rows")
        answer["table"] = {"rows": normalized_rows}

    if set(answer) != set(requested):
        missing = sorted(set(requested) - set(answer))
        raise ValueError(f"{query.query_id}: unsupported answer types: {missing}")
    return answer


def _normalize_table_rows(rows: Any, schema: list[dict]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    columns = [
        str(item.get("name"))
        for item in schema
        if isinstance(item, dict) and item.get("name")
    ]
    column_types = {
        str(item.get("name")): str(item.get("type") or "string").lower()
        for item in schema
        if isinstance(item, dict) and item.get("name")
    }
    row_key_columns = [
        str(item.get("name"))
        for item in schema
        if isinstance(item, dict) and item.get("name") and item.get("is_row_key")
    ]
    dedupe_columns = row_key_columns or columns
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"table row {row_index} must be an object")
        if set(row) != set(columns):
            raise ValueError(
                f"table row {row_index} must contain exactly {columns}"
            )
        normalized = {
            column: _normalize_cell(row[column], column_types.get(column, "string"))
            for column in columns
        }
        if any(normalized[column] in (None, "") for column in row_key_columns):
            raise ValueError("table row key must not be empty")
        key = tuple(_normalize_row_key(normalized[column]) for column in dedupe_columns)
        if key in seen:
            raise ValueError(f"duplicate table row key: {key}")
        seen.add(key)
        output.append(normalized)
    return output


def _normalize_cell(value: Any, declared_type: str) -> Any:
    if value is None:
        return None
    if declared_type == "string":
        if not isinstance(value, str):
            raise TypeError(f"string table cell must be a JSON string: {value!r}")
        return value
    if declared_type == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"boolean table cell must be JSON true/false: {value!r}")
        return value
    if declared_type != "number":
        raise ValueError(f"unsupported table column type: {declared_type}")
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric table cell")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric table cell: {value!r}")
        return value
    raise TypeError(f"numeric table cell must be a JSON number: {value!r}")


def _normalize_row_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    return re.sub(r"\s+", " ", text)


def _normalize_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip().isdigit():
        page = int(value.strip())
        return page if page >= 1 else None
    return None
