"""Verify open-set citation questions with references and comparison tables."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import (
    PaperDocumentSource,
    PaperEvidenceDocument,
    PaperTable,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.structured_filter import (
    detect_constraint,
)
from littraceqa.di_pipeline.select.selector import (
    PaperSelection,
    ordered_paper_ids,
)

_CITATION_TABLE_RE = re.compile(
    r"\bcite\s+(?P<alias>[A-Za-z][A-Za-z0-9.+-]{1,39})\s*"
    r"\((?P<title>[^()\n]{5,200}),\s*"
    r"(?P<cited_venue>[A-Za-z][A-Za-z .&/-]{1,39}?)\s*"
    r"(?P<cited_year>(?:19|20)\d{2})\)\s+and\s+use\s+it\s+as\s+"
    r"(?:a\s+)?baseline\s+in\s+their\s+main\s+comparison\s+table\b",
    re.IGNORECASE,
)
_OUTER_SCOPE_RE = re.compile(
    r"\bwhich\s+(?P<venue>[A-Za-z][A-Za-z0-9.-]{1,19})\s+"
    r"(?P<year>(?:19|20)\d{2})\s+papers?\b",
    re.IGNORECASE,
)
_REFERENCE_NUMBER_RE = re.compile(r"^\s*\[(?P<number>\d+)\]")
_CITATION_AFTER_ALIAS_RE = re.compile(
    r"(?:-[A-Za-z0-9]+)?\s*[†*]?\s*\[(?P<number>\d+)\]"
)
_MEASUREMENT_RE = re.compile(r"(?<![A-Za-z0-9])[-+−]?\d+(?:\.\d+)?%?")
_METHOD_HEADER_RE = re.compile(r"\bmethods?\b", re.IGNORECASE)
_REFERENCE_VENUE_ALIASES: dict[str, tuple[str, ...]] = {
    "acl": ("association for computational linguistics",),
    "cvpr": ("computer vision and pattern recognition",),
    "eccv": ("european conference on computer vision",),
    "emnlp": ("empirical methods in natural language processing",),
    "iccv": ("international conference on computer vision",),
    "iclr": ("international conference on learning representations",),
    "icml": ("international conference on machine learning",),
    "naacl": (
        "north american chapter of the association for computational linguistics",
    ),
    "neurips": ("neural information processing systems",),
}


@dataclass(frozen=True)
class CitationTableCondition:
    """Constraints stated by one citation-and-comparison-table question."""

    venue: str
    year: int
    alias: str
    cited_title: str
    cited_venue: str
    cited_year: int


@dataclass(frozen=True)
class ReferenceMatches:
    """Reference entries matching the cited paper identity."""

    numbers: frozenset[str]
    count: int


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _is_paper_list_schema(query: Query) -> bool:
    if "table" not in query.answer_types or not isinstance(query.table_schema, list):
        return False
    if len(query.table_schema) != 1 or not isinstance(query.table_schema[0], dict):
        return False
    column = query.table_schema[0]
    name = _normalize(column.get("name"))
    return (
        column.get("is_row_key") is True
        and _normalize(column.get("type")) == "string"
        and "paper" in name.split()
        and "title" in name.split()
    )


def parse_citation_table_condition(query: Query) -> CitationTableCondition | None:
    """Parse one narrow, fully specified open-set citation question."""

    if not _is_paper_list_schema(query):
        return None
    match = _CITATION_TABLE_RE.search(query.question)
    outer = _OUTER_SCOPE_RE.search(query.question)
    if match is None or outer is None:
        return None
    constraint = detect_constraint(
        f"Which {outer.group('venue')} {outer.group('year')} papers use a "
        "main comparison table?"
    )
    if constraint is None or constraint.chunk_type != "table":
        return None
    return CitationTableCondition(
        venue=constraint.venue,
        year=constraint.year,
        alias=match.group("alias"),
        cited_title=match.group("title"),
        cited_venue=match.group("cited_venue"),
        cited_year=int(match.group("cited_year")),
    )


def citation_table_candidate_ids(
    queries: Mapping[str, Query],
    rankings: Mapping[str, Sequence[str]],
    candidate_limit: int = 20,
) -> set[str]:
    """Return only candidates needed by supported citation-table questions."""

    paper_ids: set[str] = set()
    for query_id, query in queries.items():
        if parse_citation_table_condition(query) is not None:
            paper_ids.update(rankings.get(query_id, ())[:candidate_limit])
    return paper_ids


def _alias_pattern(alias: str) -> re.Pattern[str]:
    body = r"[^A-Za-z0-9]+".join(
        re.escape(token)
        for token in re.findall(r"[A-Za-z0-9]+", alias)
    )
    letters = [character for character in alias if character.isalpha()]
    mixed_case = any(c.islower() for c in letters) and any(c.isupper() for c in letters)
    flags = 0 if mixed_case else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9_+./-]){body}(?![A-Za-z0-9_])", flags)


def _matching_references(
    entries: Sequence[str],
    condition: CitationTableCondition,
) -> ReferenceMatches:
    title = _normalize(condition.cited_title)
    venue = _normalize(condition.cited_venue)
    venue_aliases = (venue, *_REFERENCE_VENUE_ALIASES.get(venue, ()))
    year = str(condition.cited_year)
    matches: list[str] = []
    for entry in entries:
        normalized = _normalize(entry)
        words = normalized.split()
        if title not in normalized or year not in words:
            continue
        if any(
            _contains_ocr_spaced_phrase(normalized, alias)
            for alias in venue_aliases
        ):
            matches.append(entry)

    numbers: set[str] = set()
    for entry in matches:
        number = _REFERENCE_NUMBER_RE.match(entry)
        if number is not None:
            numbers.add(number.group("number"))
    return ReferenceMatches(frozenset(numbers), len(matches))


def _contains_ocr_spaced_phrase(text: str, phrase: str) -> bool:
    words = []
    for word in phrase.split():
        letters = r"\s*".join(re.escape(character) for character in word)
        words.append(letters)
    pattern = r"\s+".join(words)
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def _row_mentions_cited_alias(
    row: Sequence[str],
    condition: CitationTableCondition,
    reference_numbers: Collection[str],
    reference_matches: int,
) -> bool:
    pattern = _alias_pattern(condition.alias)
    for cell in row:
        normalized = unicodedata.normalize("NFKC", cell).strip()
        for match in pattern.finditer(normalized):
            suffix = normalized[match.end() : match.end() + 24]
            citation = _CITATION_AFTER_ALIAS_RE.match(suffix)
            if citation is not None:
                if citation.group("number") in reference_numbers:
                    return True
                continue
            if match.start() == 0 and reference_matches == 1:
                return True
    return False


def _has_measurements(row: Sequence[str]) -> bool:
    text = _REFERENCE_NUMBER_RE.sub("", " ".join(row))
    text = re.sub(r"\[\d+\]", "", text)
    return len(_MEASUREMENT_RE.findall(text)) >= 2


def _is_comparison_table(table: PaperTable) -> bool:
    if len(table.rows) < 3:
        return False
    return any(
        _METHOD_HEADER_RE.search(cell) is not None
        for row in table.rows[:2]
        for cell in row
    )


def _has_baseline_row(
    document: PaperEvidenceDocument,
    condition: CitationTableCondition,
    reference_numbers: Collection[str],
    reference_matches: int,
) -> bool:
    for table in document.tables[:2]:
        if not _is_comparison_table(table):
            continue
        if any(
            _has_measurements(row)
            and _row_mentions_cited_alias(
                row,
                condition,
                reference_numbers,
                reference_matches,
            )
            for row in table.rows[1:]
        ):
            return True
    return False


def _metadata_matches(
    metadata: Mapping[str, object] | None,
    condition: CitationTableCondition,
) -> bool:
    if metadata is None:
        return False
    try:
        year = int(metadata.get("year"))
    except (TypeError, ValueError):
        return False
    return (
        _normalize(metadata.get("venue")) == _normalize(condition.venue)
        and year == condition.year
    )


def _paper_matches(
    document: PaperEvidenceDocument,
    metadata: Mapping[str, object] | None,
    condition: CitationTableCondition,
) -> bool:
    if not _metadata_matches(metadata, condition):
        return False
    references = _matching_references(document.reference_entries, condition)
    if references.count == 0:
        return False
    return _has_baseline_row(
        document,
        condition,
        references.numbers,
        references.count,
    )


class CitationTableOpenSetRefiner:
    """Expand one safe seed into a verified citation-table result set."""

    def __init__(
        self,
        source: PaperDocumentSource,
        paper_metadata: Mapping[str, Mapping[str, object]],
        *,
        candidate_limit: int = 20,
        max_papers: int = 10,
    ) -> None:
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer")
        if isinstance(max_papers, bool) or not isinstance(max_papers, int):
            raise TypeError("max_papers must be an integer")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if max_papers < 2:
            raise ValueError("max_papers must be at least two")
        self.source = source
        self.paper_metadata = paper_metadata
        self.candidate_limit = candidate_limit
        self.max_papers = max_papers

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection:
        if len(selection.paper_ids) != 1:
            return selection
        condition = parse_citation_table_condition(query)
        if condition is None:
            return selection
        ranked = ordered_paper_ids(candidates)[: self.candidate_limit]
        verified = tuple(
            paper_id
            for paper_id in ranked
            if _paper_matches(
                self.source.document(paper_id),
                self.paper_metadata.get(paper_id),
                condition,
            )
        )
        if (
            not 2 <= len(verified) <= self.max_papers
            or selection.paper_ids[0] not in verified
        ):
            return selection
        return selection.with_papers(verified, "citation_table_coverage")


__all__ = [
    "CitationTableCondition",
    "CitationTableOpenSetRefiner",
    "citation_table_candidate_ids",
    "parse_citation_table_condition",
]
