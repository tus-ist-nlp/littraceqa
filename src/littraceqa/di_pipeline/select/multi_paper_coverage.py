"""Add a second paper only when two answer slots have direct local support."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

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
from littraceqa.di_pipeline.select.two_slot_question import (
    EvidenceSlot,
    evidence_terms,
    parse_two_slot_question,
)

_NUMBER_VECTOR_RE = re.compile(r"\[\s*[-+−]?\s*\d")
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
    target_key = evidence_terms(target)
    body = "\n".join(document.text_blocks)
    for owner in owners:
        owner_key = evidence_terms(owner)
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
            owner_terms = evidence_terms(owner)
            if evidence_terms(previous)[: len(owner_terms)] == owner_terms:
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
            row_terms = set(evidence_terms(row_text))
            if not set(slot.row_terms).issubset(row_terms):
                continue
            scope_terms = set(
                evidence_terms(f"{shared} {_row_group(table, row_index)}")
            )
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
        if not set(slot.local_terms).issubset(evidence_terms(block)):
            continue
        window = " ".join(blocks[max(0, index - 4) : index + 5])
        present = set(evidence_terms(window))
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
        return selection.with_papers(paper_ids, "multi_paper_coverage")


__all__ = ["EvidenceSlot", "MultiPaperCoverageRefiner", "parse_two_slot_question"]
