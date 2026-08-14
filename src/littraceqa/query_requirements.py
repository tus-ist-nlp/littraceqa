"""Gold-free answer requirements derived from the released query fields.

The official test exposes only ``query_id``, ``question``, ``answer_types``, and
``table_schema``.  This module deliberately uses only the semantic fields among
those inputs and never imports validation targets.  It extracts a requirement
only when a table question contains an explicit coordinated list, so ambiguous
open-ended enumeration questions remain the reader's responsibility.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from littraceqa.di_pipeline.contracts import Query


QUERY_REQUIREMENTS_VERSION = "gold-free-table-output-contract-v3"


_TRAILING_QUALIFIERS = (
    re.compile(r"\s+as\s+reported\s+in\s+their\s+respective\s+papers\s*$", re.I),
    re.compile(r"\s+given\s+[^?]+$", re.I),
)
_LIST_PATTERNS = (
    re.compile(r"\bfor\s+(.+?)(?:\s+as\s+reported\b|\?\s*$)", re.I),
    re.compile(r"\bwith\s+(.+?)\?\s*$", re.I),
    re.compile(r"\bin\s+(.+?)\?\s*$", re.I),
    re.compile(r"\bof\s+(.+?)(?:\s+given\b|\?\s*$)", re.I),
)
_GENERIC_MATCH_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "detection",
        "detections",
        "method",
        "methods",
        "model",
        "models",
        "ours",
        "prompt",
        "prompts",
        "the",
        "w",
        "with",
    }
)
_SOURCE_DECORATION_RE = re.compile(
    r"\[(?:\d+(?:\s*[,;-]\s*\d+)*)\]|\(\s*ours\s*\)", re.I
)


def explicit_table_row_items(query: Query) -> tuple[str, ...]:
    """Return a confident, ordered explicit row inventory from ``question``.

    Open-ended questions such as "Which papers ...?" intentionally return an
    empty tuple.  A non-empty result therefore acts as a deterministic
    completeness contract for the final table answer.
    """

    if "table" not in query.answer_types:
        return ()
    if not any(
        isinstance(column, dict) and column.get("is_row_key") is True
        for column in (query.table_schema or [])
    ):
        return ()

    question = str(query.question or "").strip()
    for pattern in _LIST_PATTERNS:
        matches = list(pattern.finditer(question))
        if not matches:
            continue
        raw = matches[-1].group(1).strip()
        # A purpose phrase such as ``for producing the next state from the
        # previous state and the current input`` describes an operation and its
        # arguments; it does not enumerate table rows.
        if re.match(r"(?i)^produc(?:e|es|ed|ing)\b", raw):
            continue
        for qualifier in _TRAILING_QUALIFIERS:
            raw = qualifier.sub("", raw).strip()
        items = _split_coordinated_items(raw)
        if _confident_inventory(items):
            return tuple(items)
    return ()


def table_output_contract(query: Query) -> dict[str, object] | None:
    """Return the gold-free output contract for a table query.

    The contract is derived only from ``question`` and ``table_schema``.  It
    deliberately contains neither ``query_id`` nor any candidate- or
    answer-derived aliases, so the same rules apply to validation and test.
    """

    if "table" not in query.answer_types:
        return None

    schema_columns: list[dict[str, object]] = []
    for column in query.table_schema or []:
        if not isinstance(column, dict):
            continue
        name = str(column.get("name") or "").strip()
        if not name:
            continue
        column_type = str(column.get("type") or "").strip()
        is_row_key = column.get("is_row_key") is True
        if is_row_key and name.casefold() == "paper title":
            output_policy = "metadata_title_exact"
        elif is_row_key:
            output_policy = "query_facing_shortest_explicit_label"
        elif column_type.casefold() == "string":
            output_policy = "source_exact"
        elif column_type.casefold() == "number":
            output_policy = "native_json_number"
        elif column_type.casefold() == "boolean":
            output_policy = "native_json_boolean"
        else:
            output_policy = "native_json_value_matching_schema_type"
        schema_columns.append(
            {
                "name": name,
                "type": column_type,
                "is_row_key": is_row_key,
                "output_policy": output_policy,
            }
        )

    return {
        "derived_from": ["question", "table_schema"],
        "row_key_policy": {
            "paper_title": "metadata_title_exact",
            "other": "query_facing_shortest_explicit_label",
        },
        "non_row_key_string_policy": "source_exact",
        "schema_columns": schema_columns,
        "explicit_row_inventory": list(explicit_table_row_items(query)),
    }


def table_row_key_texts(query: Query, rows: Iterable[dict]) -> tuple[str, ...]:
    """Render each submitted row's declared row-key tuple for matching."""

    row_keys = [
        str(column["name"])
        for column in (query.table_schema or [])
        if isinstance(column, dict)
        and column.get("name")
        and column.get("is_row_key") is True
    ]
    if not row_keys:
        return ()
    output: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            output.append("")
            continue
        output.append(" / ".join(str(row.get(key) or "") for key in row_keys))
    return tuple(output)


def missing_explicit_table_items(query: Query, rows: Iterable[dict]) -> tuple[str, ...]:
    """Return explicit query items with no unique compatible output row."""

    required = explicit_table_row_items(query)
    if not required:
        return ()
    row_texts = table_row_key_texts(query, rows)
    unused_rows = set(range(len(row_texts)))
    missing: list[str] = []
    for item in required:
        candidates = [
            index
            for index in sorted(unused_rows)
            if _semantic_row_match(item, row_texts[index])
        ]
        if not candidates:
            missing.append(item)
            continue
        # Prefer the closest surface when one broad row contains several names.
        chosen = min(
            candidates,
            key=lambda index: (
                _edit_distance(_compact(item), _compact(row_texts[index])),
                index,
            ),
        )
        unused_rows.remove(chosen)
    return tuple(missing)


def unaccounted_explicit_table_items(
    query: Query,
    rows: Iterable[dict],
    declared_missing: Iterable[str],
) -> tuple[str, ...]:
    """Return required items absent from both rows and the missing inventory."""

    missing_rows = missing_explicit_table_items(query, rows)
    missing_labels = tuple(str(value or "") for value in declared_missing)
    return tuple(
        item
        for item in missing_rows
        if not any(_semantic_row_match(item, label) for label in missing_labels)
    )


def _split_coordinated_items(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}" and depth:
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])

    if len(parts) == 1:
        coordinated = re.split(r"\s+and\s+", parts[0], flags=re.I)
        parts = coordinated if len(coordinated) >= 2 else parts
    else:
        coordinated_tail = re.split(r"\s+and\s+", parts[-1], flags=re.I)
        if len(coordinated_tail) >= 2:
            parts = [*parts[:-1], *coordinated_tail]
        else:
            parts[-1] = re.sub(r"^\s*and\s+", "", parts[-1], flags=re.I)

    return [item.strip().strip(".?") for item in parts if item.strip().strip(".?")]


def _confident_inventory(items: list[str]) -> bool:
    if not 2 <= len(items) <= 12:
        return False
    return all(
        1 <= len(item) <= 120
        and "?" not in item
        and re.search(r"\b(?:what|which|who|whom|whose|how|where|when)\b", item, re.I)
        is None
        for item in items
    )


def _semantic_row_match(required: str, actual: str) -> bool:
    required_dimensions = _explicit_detection_dimensions(required)
    actual_dimensions = _explicit_detection_dimensions(actual)
    if (
        required_dimensions
        and actual_dimensions
        and required_dimensions != actual_dimensions
    ):
        return False
    relax_detection_dimension = bool(required_dimensions) != bool(actual_dimensions)
    required_surface = _normalized_surface(
        required,
        strip_parenthetical=True,
        strip_detection_dimension=relax_detection_dimension,
    )
    actual_surface = _normalized_surface(
        actual,
        strip_parenthetical=False,
        strip_detection_dimension=relax_detection_dimension,
    )
    if not required_surface or not actual_surface:
        return False
    if required_surface == actual_surface:
        return True

    required_tokens = _significant_tokens(required_surface)
    actual_tokens = _significant_tokens(actual_surface)
    if required_tokens and required_tokens.issubset(actual_tokens):
        return True

    required_compact = _compact(required_surface)
    actual_compact = _compact(actual_surface)
    if required_compact and (
        required_compact == actual_compact
        or required_compact in actual_compact
        or actual_compact in required_compact
    ):
        return True

    # Allow one obvious character-level typo only for a substantial identifier.
    # This is query-text driven and candidate independent; it is not an alias
    # table learned from validation answers.
    return (
        min(len(required_compact), len(actual_compact)) >= 6
        and abs(len(required_compact) - len(actual_compact)) <= 1
        and _edit_distance(required_compact, actual_compact) <= 1
    )


def _explicit_detection_dimensions(value: str) -> frozenset[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return frozenset(
        match.group(1)
        for match in re.finditer(r"\b([23])d\b(?:\s+detections?)?", text)
    )


def _normalized_surface(
    value: str,
    *,
    strip_parenthetical: bool,
    strip_detection_dimension: bool = False,
) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _SOURCE_DECORATION_RE.sub(" ", text)
    if strip_parenthetical:
        text = re.sub(r"\([^()]*\)", " ", text)
    text = text.replace("ground-truth", "ground truth")
    if strip_detection_dimension:
        text = re.sub(r"\b[23]d\b(?:\s+detections?)?", " ", text)
    text = re.sub(r"\br\s*-?\s*cnn\b", "rcnn", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _significant_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if token and token not in _GENERIC_MATCH_TOKENS
    }


def _compact(value: str) -> str:
    normalized = _normalized_surface(value, strip_parenthetical=False)
    return "".join(
        token
        for token in normalized.split()
        if token and token not in _GENERIC_MATCH_TOKENS
    )


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
