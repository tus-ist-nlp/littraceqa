"""Add a second paper only when two answer slots have direct local support."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.method_aliases import (
    extract_self_owned_method_aliases,
    method_aliases_equal,
    text_before_references,
)
from littraceqa.di_pipeline.retrieve.paper_tables import (
    PaperDocumentSource,
    PaperEvidenceDocument,
    PaperTable,
)
from littraceqa.di_pipeline.select.selector import (
    PaperSelection,
    ordered_paper_ids,
)

_PAIRED_ACHIEVE_RE = re.compile(
    r"^What\s+(?P<metric1>.+?)\s+does\s+(?P<subject1>.+?)\s+"
    r"achieve\s+(?P<condition1>.+?),\s+and\s+what\s+"
    r"(?P<metric2>.+?)\s+does\s+(?P<subject2>.+?)\s+"
    r"achieve\s+(?P<condition2>.+?)\?\s*$",
    re.IGNORECASE,
)
_COORDINATED_USE_RE = re.compile(
    r"^What\s+(?P<property>.+?)\s+do\s+"
    r"(?P<target1>[A-Za-z][A-Za-z0-9.+-]{1,39})\s+and\s+"
    r"(?P<target2>[A-Za-z][A-Za-z0-9.+-]{1,39})\s+use\s+"
    r"(?P<context>.+?),\s+and\s+do\s+they\s+(?:match|differ)\?\s*$",
    re.IGNORECASE,
)
_WITH_RE = re.compile(r"\s+with\s+", re.IGNORECASE)
_NUMBER_VECTOR_RE = re.compile(r"\[\s*[-+−]?\s*\d")
_ROW_NUMBER_RE = re.compile(
    r"\b(?P<number>\d+)\s+(?:epochs?|steps?|iterations?)\b",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "does",
        "for",
        "in",
        "of",
        "on",
        "the",
        "their",
        "use",
        "using",
        "value",
        "values",
        "what",
        "with",
    }
)
_TERM_ALIASES = {
    "accuracy": "acc",
    "map": "ap",
    "val": "validation",
}
_VARIANT_ACTION = (
    r"(?:build|builds|building|built|develop|develops|developing|developed|"
    r"introduce|introduces|introducing|introduced|present|presents|presenting|"
    r"presented|propose|proposes|proposing|proposed)"
)
_DIRECT_VARIANT_RE = re.compile(
    rf"(?:\bwe\s+{_VARIANT_ACTION}\s+(?:our\s+)?|"
    rf"\b{_VARIANT_ACTION}\s+our\s+)$",
    re.IGNORECASE,
)
_COORDINATED_VARIANT_RE = re.compile(
    rf"\b(?:we\s+)?{_VARIANT_ACTION}\s+our\s+"
    r"(?P<alias>[A-Za-z][A-Za-z0-9+.-]{2,59})\s+and\s*$",
    re.IGNORECASE,
)
_NUMERIC_CELL_RE = re.compile(r"[-+−]?\d+(?:\.\d+)?(?:\s*%)?")


@dataclass(frozen=True)
class EvidenceSlot:
    """One explicitly named target and the local evidence it must satisfy."""

    target: str
    terms: tuple[str, ...]
    kind: Literal["table", "text"]
    local_terms: tuple[str, ...] = ()
    row_terms: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _term(value: str) -> str:
    value = value.casefold()
    if len(value) > 4 and value.endswith("s"):
        value = value[:-1]
    return _TERM_ALIASES.get(value, value)


def _terms(text: str) -> tuple[str, ...]:
    normalized = re.sub(
        r"(?<![A-Za-z])(?:[A-Z]\s+){1,5}[A-Z](?![A-Za-z])",
        lambda match: re.sub(r"\s+", "", match.group()),
        _normalize(text),
    )
    values = (
        _term(token)
        for token in re.findall(r"[A-Za-z]+|[0-9]+", normalized)
    )
    return tuple(dict.fromkeys(value for value in values if value not in _STOP_WORDS))


def _subject(subject: str) -> tuple[str, str]:
    parts = _WITH_RE.split(_normalize(subject), maxsplit=1)
    return parts[0], parts[1] if len(parts) == 2 else ""


def _table_slot(
    target: str,
    metric: str,
    qualifier: str,
    condition: str,
) -> EvidenceSlot:
    return EvidenceSlot(
        target=target,
        terms=_terms(f"{metric} {qualifier} {condition}"),
        kind="table",
        row_terms=tuple(
            match.group("number") for match in _ROW_NUMBER_RE.finditer(condition)
        ),
    )


def _paired_achievement_slots(
    match: re.Match[str],
) -> tuple[EvidenceSlot, EvidenceSlot] | None:
    target1, qualifier1 = _subject(match.group("subject1"))
    target2, qualifier2 = _subject(match.group("subject2"))
    if not target1 or not target2 or method_aliases_equal(target1, target2):
        return None
    metric = max(
        (match.group("metric1"), match.group("metric2")),
        key=lambda value: len(_terms(value)),
    )
    return (
        _table_slot(target1, metric, qualifier1, match.group("condition1")),
        _table_slot(target2, metric, qualifier2, match.group("condition2")),
    )


def _coordinated_use_slots(
    match: re.Match[str],
) -> tuple[EvidenceSlot, EvidenceSlot] | None:
    target1 = match.group("target1")
    target2 = match.group("target2")
    if method_aliases_equal(target1, target2):
        return None
    requirements = _terms(f"{match.group('property')} {match.group('context')}")
    local_terms = tuple(
        term for term in _terms(match.group("property")) if term != "normalization"
    )
    return (
        EvidenceSlot(target1, requirements, "text", local_terms),
        EvidenceSlot(target2, requirements, "text", local_terms),
    )


def parse_two_slot_question(
    question: object,
) -> tuple[EvidenceSlot, EvidenceSlot] | None:
    """Parse only two narrow question forms supported by the verifier."""

    if not isinstance(question, str):
        return None
    text = _normalize(question)
    match = _PAIRED_ACHIEVE_RE.match(text)
    if match is not None:
        return _paired_achievement_slots(match)

    match = _COORDINATED_USE_RE.match(text)
    if match is None:
        return None
    return _coordinated_use_slots(match)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKC", alias))
    if not tokens:
        return re.compile(r"(?!x)x")
    body = r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens)
    letters = [character for character in alias if character.isalpha()]
    mixed_case = any(c.islower() for c in letters) and any(c.isupper() for c in letters)
    flags = 0 if mixed_case else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags)


def _contains_alias(text: str, alias: str) -> bool:
    return bool(_alias_pattern(alias).search(unicodedata.normalize("NFKC", text)))


def _owned_aliases(document: PaperEvidenceDocument) -> tuple[str, ...]:
    text = "\n".join(document.text_blocks)
    return tuple(
        evidence.alias
        for evidence in extract_self_owned_method_aliases(document.title, text)
    )


def _owns_target(
    document: PaperEvidenceDocument,
    target: str,
    owners: tuple[str, ...] | None = None,
) -> bool:
    owners = owners if owners is not None else _owned_aliases(document)
    if any(method_aliases_equal(owner, target) for owner in owners):
        return True

    title_prefix = document.title.partition(":")[0]
    target_key = _terms(target)
    body = "\n".join(document.text_blocks)
    for owner in owners:
        owner_key = _terms(owner)
        if (
            len("".join(owner_key)) >= 4
            and owner_key
            and target_key[: len(owner_key)] == owner_key
            and method_aliases_equal(title_prefix, owner)
            and _has_variant_claim(body, target, owner)
        ):
            return True
    return False


def _has_variant_claim(text: str, target: str, owner: str) -> bool:
    normalized = text_before_references(text)
    for match in _alias_pattern(target).finditer(normalized):
        prefix = normalized[max(0, match.start() - 140) : match.start()]
        sentence = re.split(r"[.!?\n]", prefix)[-1]
        if _DIRECT_VARIANT_RE.search(sentence):
            return True
        coordinated = _COORDINATED_VARIANT_RE.search(sentence)
        if coordinated is not None:
            previous = coordinated.group("alias")
            if _terms(previous)[: len(_terms(owner))] == _terms(owner):
                return True
    return False


def _table_header(table: PaperTable) -> str:
    header_rows = []
    for row in table.rows[:2]:
        numeric_cells = sum(
            bool(_NUMERIC_CELL_RE.fullmatch(cell.strip())) for cell in row
        )
        if numeric_cells:
            break
        header_rows.append(" ".join(row))
    return " ".join(header_rows)


def _row_group(table: PaperTable, row_index: int) -> str:
    row = table.rows[row_index]
    if not row or not row[0]:
        return " ".join(row)
    rows = [row]
    for previous in reversed(table.rows[max(0, row_index - 4) : row_index]):
        if not previous or previous[0] != row[0]:
            break
        rows.append(previous)
    return " ".join(" ".join(item) for item in rows)


def _table_supports(document: PaperEvidenceDocument, slot: EvidenceSlot) -> bool:
    for table in document.tables:
        shared = f"{table.caption} {_table_header(table)}"
        for row_index, row in enumerate(table.rows):
            row_text = " ".join(row)
            if not _contains_alias(row_text, slot.target):
                continue
            row_terms = set(_terms(row_text))
            if not set(slot.row_terms).issubset(row_terms):
                continue
            scope_terms = set(_terms(f"{shared} {_row_group(table, row_index)}"))
            measurements = sum(
                bool(_NUMERIC_CELL_RE.fullmatch(cell.strip())) for cell in row
            )
            if measurements >= 2 and all(
                term in scope_terms for term in slot.terms
            ):
                return True
    return False


def _text_supports(document: PaperEvidenceDocument, slot: EvidenceSlot) -> bool:
    required = set(slot.terms)
    if len(required) < 3:
        return False
    blocks = document.text_blocks
    for index, block in enumerate(blocks):
        if not _NUMBER_VECTOR_RE.search(block):
            continue
        if not set(slot.local_terms).issubset(_terms(block)):
            continue
        window = " ".join(blocks[max(0, index - 4) : index + 5])
        present = set(_terms(window))
        if required.issubset(present):
            return True
    return False


def _supports(
    document: PaperEvidenceDocument,
    slot: EvidenceSlot,
    owners: tuple[str, ...] | None = None,
) -> bool:
    if not _owns_target(document, slot.target, owners):
        return False
    if slot.kind == "table":
        return _table_supports(document, slot)
    return _text_supports(document, slot)


def _unique_reporters(
    ranked: Sequence[str],
    documents: dict[str, PaperEvidenceDocument],
    slots: tuple[EvidenceSlot, EvidenceSlot],
) -> tuple[str, str] | None:
    owners = {
        paper_id: _owned_aliases(document)
        for paper_id, document in documents.items()
    }
    reporters: list[str] = []
    for slot in slots:
        supported = [
            paper_id
            for paper_id in ranked
            if _supports(documents[paper_id], slot, owners[paper_id])
        ]
        if len(supported) != 1:
            return None
        reporters.append(supported[0])
    return reporters[0], reporters[1]


class MultiPaperCoverageRefiner:
    """Add one uniquely supported paper for a verified two-slot question."""

    def __init__(self, source: PaperDocumentSource, candidate_limit: int = 20) -> None:
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.source = source
        self.candidate_limit = candidate_limit

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection:
        if "multiple_choice" not in query.answer_types or len(selection.paper_ids) != 1:
            return selection
        slots = parse_two_slot_question(query.question)
        if slots is None:
            return selection

        ranked = ordered_paper_ids(candidates)[: self.candidate_limit]
        documents = {
            paper_id: self.source.document(paper_id) for paper_id in ranked
        }
        matches = _unique_reporters(ranked, documents, slots)
        if matches is None:
            return selection

        matched_ids = set(matches)
        paper_ids = tuple(paper_id for paper_id in ranked if paper_id in matched_ids)
        if (
            len(paper_ids) != 2
            or not set(selection.paper_ids).issubset(paper_ids)
        ):
            return selection
        return PaperSelection(
            paper_ids=paper_ids,
            expected_count=2,
            reason=f"{selection.reason}+multi_paper_coverage",
            dropped_without_evidence=selection.dropped_without_evidence,
        )


__all__ = ["EvidenceSlot", "MultiPaperCoverageRefiner", "parse_two_slot_question"]
