"""Evidence checks that are confined to one paper table."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import PaperTable, PaperTableSource
from littraceqa.di_pipeline.select.selector import (
    PaperSelection,
    ordered_paper_ids,
)

_LIST_PREPOSITION_RE = re.compile(
    r"\b(?:in|on|for|of|across|among|between)\b",
    re.IGNORECASE,
)
_FINAL_CONJUNCTION_RE = re.compile(r",\s*(?:and|or)\s+[^,]+$", re.IGNORECASE)
_LEADING_CONJUNCTION_RE = re.compile(r"^(?:and|or)\s+", re.IGNORECASE)
_UNSAFE_ITEM_RE = re.compile(
    r"\b(?:what|which|how|does|do|is|are|was|were|achieve|report|reported|"
    r"respectively|given|using|when|where|who)\b",
    re.IGNORECASE,
)
_MULTI_SOURCE_RE = re.compile(
    r"\b(?:papers|works|studies|articles|publications|sources)\b|"
    r"\b(?:separately|respective papers)\b",
    re.IGNORECASE,
)
_ROW_KEY_RE = re.compile(
    r"\b(?:benchmark|dataset|category|organ|prompt|variant|setting|task|scene)s?\b",
    re.IGNORECASE,
)
_PAPER_ROW_KEY_RE = re.compile(
    r"\b(?:method|model|system|paper|work|study|article|publication)s?\b",
    re.IGNORECASE,
)
_GENERIC_VALUE_COLUMNS = frozenset({"metric", "result", "score", "value"})
_EXPLICIT_TABLE_RE = re.compile(
    r"\b(?:In|From)\s+the\s+(?P<anchor>[^,;?]{1,50}?)\s+"
    r"paper(?:['’]s)?\s+(?P<table>Table\s+[A-Z]?\d+[A-Z]?)\b",
    re.IGNORECASE,
)
_EQUATION_RHS_RE = re.compile(
    r"=\s*(?P<rhs>[^,?]{1,80}?)(?=\s+"
    r"(?:achieve|compared|versus|on\s+[A-Z]|and\s+what)\b|[,?]|$)",
    re.IGNORECASE,
)
_LATEX_COMMAND_RE = re.compile(
    r"\\(?:mathrm|mathit|mathbf|mathbb|operatorname|displaystyle|text|bar|"
    r"dot|big|left|right)\b",
    re.IGNORECASE,
)


def _normalize(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _formula_key(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold().replace("×", "x")
    normalized = _LATEX_COMMAND_RE.sub("", normalized)
    normalized = normalized.replace(r"\sigma", "sigma").replace(r"\frac", "")
    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in "/^+-.()"
    )


def _explicit_table_requirements(
    question: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    match = _EXPLICIT_TABLE_RE.search(question)
    if match is None:
        return None
    anchor = _normalize(match.group("anchor"))
    if len(anchor) < 2:
        return None
    equations = tuple(
        dict.fromkeys(
            key
            for equation in _EQUATION_RHS_RE.finditer(question)
            if len(key := _formula_key(equation.group("rhs"))) >= 3
        )
    )
    if len(equations) < 2:
        return None
    return (
        anchor,
        match.group("table"),
        equations,
    )


def _contains_formula(table: PaperTable, formula: str) -> bool:
    """Match a normalized formula without accepting a longer expression prefix."""

    pattern = re.compile(
        rf"(?<![a-z0-9/^+().-]){re.escape(formula)}(?![a-z0-9/^+().-])"
    )
    parts = [cell for row in table.rows for cell in row]
    parts.extend(table.text.splitlines())
    return any(pattern.search(_formula_key(part)) for part in parts)


def _contains_anchor(text: str, anchor: str) -> bool:
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(anchor)}s?(?![a-z0-9])"
    )
    return pattern.search(_normalize(text)) is not None


def _enumerated_rows(question: str) -> tuple[str, ...]:
    """Return a short comma-separated list at the end of a question."""

    text = " ".join(unicodedata.normalize("NFKC", question).split()).rstrip(" ?")
    matches: list[tuple[str, ...]] = []
    for preposition in _LIST_PREPOSITION_RE.finditer(text):
        body = text[preposition.end() :].strip()
        if body.count(",") < 2 or not _FINAL_CONJUNCTION_RE.search(body):
            continue
        items = [
            _LEADING_CONJUNCTION_RE.sub("", part.strip(" ,."))
            for part in body.split(",")
            if part.strip(" ,.")
        ]
        if not 4 <= len(items) <= 10:
            continue
        if any(
            not re.search(r"[A-Za-z]", item)
            or len(item) > 50
            or len(item.split()) > 4
            or _UNSAFE_ITEM_RE.search(item)
            for item in items
        ):
            continue
        normalized = [_normalize(item) for item in items]
        if len(set(normalized)) == len(normalized):
            matches.append(tuple(items))
    return matches[-1] if matches else ()


def _value_column(query: Query) -> str | None:
    if "table" not in query.answer_types or not isinstance(query.table_schema, list):
        return None
    if len(query.table_schema) != 2:
        return None
    if not all(isinstance(column, dict) for column in query.table_schema):
        return None
    row_keys = [column for column in query.table_schema if column.get("is_row_key") is True]
    values = [column for column in query.table_schema if column.get("is_row_key") is False]
    if len(row_keys) != 1 or len(values) != 1:
        return None
    row_name = str(row_keys[0].get("name") or "")
    value_name = str(values[0].get("name") or "")
    if (
        not value_name
        or _normalize(value_name) in _GENERIC_VALUE_COLUMNS
        or not _ROW_KEY_RE.search(row_name)
        or _PAPER_ROW_KEY_RE.search(row_name)
    ):
        return None
    return value_name


def _row_matches(row: Sequence[str], item: str) -> bool:
    target = _normalize(item)
    if not target:
        return False
    pattern = re.compile(rf"(?<![a-z0-9_-]){re.escape(target)}(?![a-z0-9_-])")
    return any(pattern.search(_normalize(cell)) for cell in row)


def _has_distinct_rows(table: PaperTable, items: Sequence[str]) -> bool:
    eligible_rows = [row for row in table.rows if sum(bool(cell.strip()) for cell in row) >= 2]
    choices = [
        [index for index, row in enumerate(eligible_rows) if _row_matches(row, item)]
        for item in items
    ]
    if any(not indexes for indexes in choices):
        return False

    ordered = sorted(choices, key=len)

    def assign(position: int, used: set[int]) -> bool:
        if position == len(ordered):
            return True
        for row_index in ordered[position]:
            if row_index not in used and assign(position + 1, used | {row_index}):
                return True
        return False

    return assign(0, set())


def _covers_table(table: PaperTable, items: Sequence[str], value_column: str) -> bool:
    header_has_value = any(
        _row_matches(row, value_column) for row in table.rows[:2]
    )
    return header_has_value and _has_distinct_rows(table, items)


class ExplicitTableAnchorRefiner:
    """Use an explicitly named paper, table, and equations when all agree."""

    def __init__(self, table_source: PaperTableSource, candidate_limit: int = 20) -> None:
        self.table_source = table_source
        self.candidate_limit = _validate_candidate_limit(candidate_limit)

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection:
        requirements = _explicit_table_requirements(query.question)
        if requirements is None:
            return selection
        anchor, table_id, equations = requirements
        matches: list[str] = []
        for paper_id in ordered_paper_ids(candidates)[: self.candidate_limit]:
            for table in self.table_source.tables(paper_id):
                if (
                    table.table_id is not None
                    and table.table_id.casefold() == table_id.casefold()
                    and _contains_anchor(table.text, anchor)
                    and all(
                        _contains_formula(table, equation) for equation in equations
                    )
                ):
                    matches.append(paper_id)
                    break
        if len(matches) != 1 or selection.paper_ids == (matches[0],):
            return selection
        return selection.with_papers((matches[0],), "explicit_table_coverage")


def _validate_candidate_limit(candidate_limit: int) -> int:
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
        raise TypeError("candidate_limit must be an integer")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    return candidate_limit


class SingleTableCoverageRefiner:
    """Collapse a multi-paper result only after unique same-table coverage."""

    def __init__(self, table_source: PaperTableSource, candidate_limit: int = 20) -> None:
        self.table_source = table_source
        self.candidate_limit = _validate_candidate_limit(candidate_limit)

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection:
        if len(selection.paper_ids) <= 1 or _MULTI_SOURCE_RE.search(query.question):
            return selection
        value_column = _value_column(query)
        items = _enumerated_rows(query.question)
        if value_column is None or not items:
            return selection

        matches: list[str] = []
        for paper_id in ordered_paper_ids(candidates)[: self.candidate_limit]:
            if any(
                _covers_table(table, items, value_column)
                for table in self.table_source.tables(paper_id)
            ):
                matches.append(paper_id)
        if len(matches) != 1:
            return selection
        return selection.with_papers((matches[0],), "single_table_coverage")


__all__ = [
    "ExplicitTableAnchorRefiner",
    "SingleTableCoverageRefiner",
]
