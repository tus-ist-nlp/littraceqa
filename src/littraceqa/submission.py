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


def deterministic_mc_letter(query_id: str, choices: str = "ABCD") -> str:
    """Return an unbiased, reproducible fallback when option text is unavailable.

    The organizer's four-field test contract omits the mapping from semantic
    choices to letters.  This function only keeps the output structurally valid;
    it is not a grounded answer and must be reported separately from QA quality.
    """

    if not choices:
        raise ValueError("choices must not be empty")
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def prediction_to_submission(query: Query, prediction: Prediction) -> dict[str, Any]:
    """Drop all analysis fields and normalize one prediction for submission."""

    if prediction.query_id != query.query_id:
        raise ValueError(
            f"prediction/query id mismatch: {prediction.query_id} != {query.query_id}"
        )

    papers = _normalize_papers(prediction.gold_papers)
    if not papers:
        raise ValueError(f"{query.query_id}: prediction has no papers")
    paper_ids = {item["paper_id"] for item in papers}

    evidence = _normalize_evidence(prediction.evidence, paper_ids)
    if not evidence:
        raise ValueError(f"{query.query_id}: prediction has no valid evidence")

    answer = _normalize_answer(query, prediction)
    return {
        "query_id": query.query_id,
        "gold_papers": papers,
        "evidence": evidence,
        "answer": answer,
    }


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
        if page is None:
            continue
        locator: dict[str, Any] = {"page": page}
        object_id = ""
        if source_type == "table":
            object_id = str(raw_locator.get("table_id") or "").strip()
            if not object_id:
                continue
            locator["table_id"] = object_id
        elif source_type == "figure":
            object_id = str(raw_locator.get("figure_id") or "").strip()
            if not object_id:
                continue
            locator["figure_id"] = object_id
        key = (paper_id, source_type, page, object_id)
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
        if not re.fullmatch(r"[A-D]", letter):
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
    columns = [str(item.get("name")) for item in schema if isinstance(item, dict) and item.get("name")]
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
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = {
            column: _normalize_cell(row.get(column), column_types.get(column, "string"))
            for column in columns
        }
        if any(normalized[column] in (None, "") for column in row_key_columns):
            raise ValueError("table row key must not be empty")
        key = tuple(str(normalized[column]) for column in dedupe_columns)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _normalize_cell(value: Any, declared_type: str) -> Any:
    if value is None:
        return None
    if declared_type == "string":
        return str(value)
    if declared_type == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "false"}:
            return text == "true"
        raise ValueError(f"invalid boolean table cell: {value!r}")
    if declared_type != "number":
        raise ValueError(f"unsupported table column type: {declared_type}")
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric table cell")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric table cell: {value!r}")
        return value
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*%?", text)
        if match is None:
            raise ValueError(f"invalid numeric table cell: {value!r}")
        number = float(match.group(1))
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric table cell: {value!r}")
    return number


def _normalize_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip().isdigit():
        page = int(value.strip())
        return page if page >= 1 else None
    return None
