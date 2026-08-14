"""Conservative citation-locator recovery from MinerU bibliography text.

MinerU preserves bibliography text and pages, but the student chunk bundle does
not assign one ``citation_id`` to every reference entry.  This module recovers
an ID only when the observable question, a validated answer derivation, and the
original bibliography text agree.  It never reads benchmark gold annotations.

The public helper is intentionally all-or-nothing.  Its return value maps an
already-supported chunk ID to one or more citation IDs.  A caller may expand one
chunk-level evidence item into one evidence item per returned ID.  An empty
mapping means that the ordinary page-level locator must be retained unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from littraceqa.answer_derivation import (
    citation_author_filter,
    citation_identity_key,
    is_aggregate_citation_count_query,
)
from littraceqa.chunk_store import Record
from littraceqa.di_pipeline.contracts import Query
from littraceqa.mineru_record import record_source_type

CitationLocatorOverrides = dict[str, tuple[str, ...]]
CITATION_LOCATOR_VERSION = "mineru-bibliography-v2-ordinal-label"

_NUMBERED_ENTRY_RE = re.compile(
    r"(?m)^\s*\[\s*(?P<number>\d{1,6})\s*\]\s*"
)
_CHUNK_HEADER_RE = re.compile(r"^\s*\[[A-Za-z][A-Za-z0-9 -]*\s+\d{4}\]\s*")
_REFERENCE_SECTION_RE = re.compile(r"^(?:references|bibliography)(?:\b|\s)", re.I)
_ENTRY_START_RE = re.compile(
    r"^[A-ZÀ-ÖØ-Þ][^,\n]{0,80},\s+"
    # Deliberately require an initial after the surname.  A looser
    # ``Surname, CapitalizedWord`` shape also matches publisher continuations
    # such as ``Springer, Cham, 2016.`` and silently shifts every later ordinal.
    # Full given names are therefore unsupported rather than guessed.
    r"[A-ZÀ-ÖØ-Þ]\.?\s*(?:[A-ZÀ-ÖØ-Þ]\.?)?(?:\s|,)",
)
_TITLE_AFTER_AUTHORS_RE = re.compile(
    r"\.\s+(?=(?:"
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\-’]{1,}\b|"
    r"[AI]\s+[a-zà-öø-ÿ]"
    r"))"
)
_YEAR_RE = re.compile(r"(?<!\d)(?:17|18|19|20)\d{2}[a-z]?(?!\w)", re.I)
_LAST_REFERENCE_INDEX_RE = re.compile(
    r"\b(?:index|number)\s+of\s+(?:the\s+)?(?:last|final)\s+reference\b|"
    r"\b(?:last|final)\s+reference(?:'s)?\s+(?:index|number)\b",
    re.I,
)
_NUMERIC_ORDINAL_RE = re.compile(
    r"\b(?P<number>\d{1,6})(?:st|nd|rd|th)\s+(?:bibliographic\s+)?reference\b",
    re.I,
)
_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
_WORD_ORDINAL_RE = re.compile(
    r"\b(?P<word>" + "|".join(_ORDINAL_WORDS) + r")\s+"
    r"(?:bibliographic\s+)?reference\b",
    re.I,
)


@dataclass(frozen=True)
class BibliographyEntry:
    """One explicitly delimited bibliography entry from an original chunk."""

    citation_id: int
    chunk_id: str
    paper_id: str
    text: str


def requested_reference_ordinal(question: str) -> int | None:
    """Return the one bibliography label requested by an ordinal phrase.

    In LitTraceQA, ``the third reference`` denotes bibliography entry ``[3]``.
    It never means the third citation marker encountered while scanning the
    body.  Numeric ordinals (``24th``) and English ordinals through
    ``twentieth`` are supported.  Ambiguous questions naming two different
    ordinals fail closed with ``None``.
    """

    values = [
        int(match.group("number"))
        for match in _NUMERIC_ORDINAL_RE.finditer(question)
    ]
    values.extend(
        _ORDINAL_WORDS[match.group("word").casefold()]
        for match in _WORD_ORDINAL_RE.finditer(question)
    )
    unique = set(values)
    if len(unique) != 1:
        return None
    value = next(iter(unique))
    return value if value > 0 else None


def numbered_bibliography_entries(record: Record) -> tuple[BibliographyEntry, ...]:
    """Parse explicit ``[N]`` entries from one bibliography chunk.

    Entry boundaries must begin a physical line.  This deliberately excludes
    in-text citation occurrences such as ``models [24, 54, 57]`` even when a
    body chunk is accidentally passed to the helper.
    """

    return tuple(_numbered_entries_in_record(record))


def requested_ordinal_bibliography_entries(
    question: str,
    records: Iterable[Record],
) -> tuple[BibliographyEntry, ...]:
    """Return exact bibliography-entry spans for the requested label.

    Only records classified as citation context are eligible.  The result may
    contain one entry per paper when a question intentionally spans several
    selected papers.  Conflicting duplicate records fail closed.
    """

    ordinal = requested_reference_ordinal(question)
    unique_records = _unique_records_or_none(records)
    if ordinal is None or unique_records is None:
        return ()
    return tuple(
        entry
        for record in unique_records
        if record_source_type(record) == "citation_context"
        for entry in _numbered_entries_in_record(record)
        if entry.citation_id == ordinal
    )


def bibliography_entry_supports_value(
    entry: BibliographyEntry,
    value: Any,
) -> bool:
    """Return whether every scalar leaf of a fact is visible in one entry.

    This is intentionally lexical.  It is a deterministic guard against
    attaching the answer from a different bibliography entry that happens to
    share the same MinerU chunk; semantic paraphrases remain an LLM task and do
    not pass this citation-identity check.
    """

    leaves = _scalar_leaves(value)
    return bool(leaves) and all(
        _text_contains_value(entry.text, leaf) for leaf in leaves
    )


def infer_citation_locator_overrides(
    query: Query,
    *,
    derivation: Mapping[str, Any],
    answer: Mapping[str, Any],
    support_records: Iterable[Record],
    paper_records: Iterable[Record],
) -> CitationLocatorOverrides:
    """Return auditable citation IDs for already-selected support chunks.

    Recovery is limited to three evidence shapes:

    * an explicitly numbered Nth reference whose answer-bearing fact occurs in
      entry ``[N]``;
    * a last-reference index whose validated scalar equals the complete,
      gap-free bibliography's terminal explicit number; and
    * an author-filtered bibliography count whose stable author/year identities
      all map uniquely to entries in a complete unnumbered bibliography prefix.

    Any ambiguity, unsupported item, cross-paper mixture, or incomplete
    boundary returns ``{}``.  No partial locator set is emitted.
    """

    support = _unique_records_or_none(support_records)
    corpus = _unique_records_or_none(paper_records)
    if not support or not corpus:
        return {}
    corpus_by_id = {str(record["chunk_id"]): record for record in corpus}
    # The answer context and the full-paper reload are independent production
    # reads.  They must describe byte-for-byte-equivalent decoded records;
    # otherwise an inferred locator could be attached to text other than the
    # text the model actually cited.
    if any(corpus_by_id.get(str(record["chunk_id"])) != record for record in support):
        return {}
    support_by_id = {
        str(record.get("chunk_id") or ""): record
        for record in support
        if record_source_type(record) == "citation_context"
    }
    if not support_by_id:
        return {}

    ordinal = requested_reference_ordinal(query.question)
    if ordinal is not None:
        return _infer_explicit_ordinal(
            ordinal=ordinal,
            derivation=derivation,
            answer=answer,
            support_by_id=support_by_id,
        )

    if _LAST_REFERENCE_INDEX_RE.search(query.question):
        return _infer_last_reference_index(
            derivation=derivation,
            answer=answer,
            support_by_id=support_by_id,
            paper_records=corpus,
        )

    if (
        is_aggregate_citation_count_query(query)
        and citation_author_filter(query) is not None
    ):
        return _infer_author_filtered_count(
            query=query,
            derivation=derivation,
            answer=answer,
            support_by_id=support_by_id,
            paper_records=corpus,
        )
    return {}


def _infer_explicit_ordinal(
    *,
    ordinal: int,
    derivation: Mapping[str, Any],
    answer: Mapping[str, Any],
    support_by_id: Mapping[str, Record],
) -> CitationLocatorOverrides:
    candidates: list[tuple[str, Any]] = []
    for fact in _facts(derivation):
        value = _scalar_value(fact.get("value"))
        if value is None or not _answer_contains_value(answer, value):
            continue
        paper_id = str(fact.get("paper_id") or "")
        for chunk_id in _string_list(fact.get("chunk_ids")):
            record = support_by_id.get(chunk_id)
            if record is None or str(record.get("paper_id") or "") != paper_id:
                continue
            entries = _numbered_entries_in_record(record)
            target = [entry for entry in entries if entry.citation_id == ordinal]
            if len(target) == 1 and _text_contains_value(target[0].text, value):
                candidates.append((chunk_id, value))
    unique_chunk_ids = {chunk_id for chunk_id, _ in candidates}
    if len(unique_chunk_ids) != 1:
        return {}
    chunk_id = next(iter(unique_chunk_ids))
    record = support_by_id[chunk_id]
    if not _metadata_id_is_compatible(record, ordinal):
        return {}
    return {chunk_id: (str(ordinal),)}


def _infer_last_reference_index(
    *,
    derivation: Mapping[str, Any],
    answer: Mapping[str, Any],
    support_by_id: Mapping[str, Record],
    paper_records: list[Record],
) -> CitationLocatorOverrides:
    scalar_facts: list[tuple[dict[str, Any], int]] = []
    for fact in _facts(derivation):
        integer = _positive_integer(fact.get("value"))
        if integer is not None and _answer_contains_value(answer, integer):
            scalar_facts.append((fact, integer))
    if len(scalar_facts) != 1:
        return {}
    fact, terminal_id = scalar_facts[0]
    paper_id = str(fact.get("paper_id") or "")
    if not paper_id:
        return {}

    paper = [record for record in paper_records if record.get("paper_id") == paper_id]
    reference_records = _contiguous_reference_records(paper)
    if not reference_records:
        return {}
    numbered: list[BibliographyEntry] = []
    for record in reference_records:
        numbered.extend(_numbered_entries_in_record(record))
    numbers = [entry.citation_id for entry in numbered]
    if numbers != list(range(1, terminal_id + 1)):
        return {}

    fact_chunk_ids = set(_string_list(fact.get("chunk_ids")))
    target_entries = [
        entry
        for entry in numbered
        if entry.citation_id == terminal_id
        and entry.chunk_id in fact_chunk_ids
        and entry.chunk_id in support_by_id
        and _text_contains_value(entry.text, terminal_id)
    ]
    if len(target_entries) != 1:
        return {}
    chunk_id = target_entries[0].chunk_id
    if not _metadata_id_is_compatible(support_by_id[chunk_id], terminal_id):
        return {}
    return {chunk_id: (str(terminal_id),)}


def _infer_author_filtered_count(
    *,
    query: Query,
    derivation: Mapping[str, Any],
    answer: Mapping[str, Any],
    support_by_id: Mapping[str, Record],
    paper_records: list[Record],
) -> CitationLocatorOverrides:
    operations = [
        operation
        for operation in derivation.get("operations") or []
        if isinstance(operation, Mapping) and operation.get("kind") == "count"
    ]
    if len(operations) != 1:
        return {}
    operation = operations[0]
    raw_items = operation.get("items")
    result = operation.get("result")
    if (
        not isinstance(raw_items, list)
        or not raw_items
        or isinstance(result, bool)
        or not isinstance(result, int)
        or result != len(raw_items)
        or not _answer_contains_value(answer, result)
    ):
        return {}

    item_keys: list[str] = []
    for item in raw_items:
        key = citation_identity_key(item)
        if key is None or not key.startswith("author-year:"):
            return {}
        item_keys.append(key)
    if len(set(item_keys)) != len(item_keys):
        return {}

    facts_by_id = {
        str(fact.get("id") or ""): fact
        for fact in _facts(derivation)
        if fact.get("id")
    }
    operation_facts = [
        facts_by_id.get(fact_id)
        for fact_id in _string_list(operation.get("fact_ids"))
    ]
    if not operation_facts or any(fact is None for fact in operation_facts):
        return {}
    paper_ids = {str(fact.get("paper_id") or "") for fact in operation_facts if fact}
    if len(paper_ids) != 1 or "" in paper_ids:
        return {}
    paper_id = next(iter(paper_ids))
    supported_chunk_ids = {
        chunk_id
        for fact in operation_facts
        if fact is not None
        for chunk_id in _string_list(fact.get("chunk_ids"))
        if chunk_id in support_by_id
    }
    if not supported_chunk_ids:
        return {}

    paper = [record for record in paper_records if record.get("paper_id") == paper_id]
    reference_records = _contiguous_reference_records(paper)
    if not reference_records:
        return {}
    entries = _bibliography_entries(reference_records)
    if not entries:
        return {}

    required_author = citation_author_filter(query)
    if not required_author:
        return {}
    matches: list[BibliographyEntry] = []
    for key in item_keys:
        _, author, year = key.split(":", maxsplit=2)
        matching_entries = [
            entry
            for entry in entries
            if _entry_identity(entry.text) == (author, year)
            and _contains_author(entry.text, required_author)
            and entry.chunk_id in supported_chunk_ids
        ]
        if len(matching_entries) != 1:
            return {}
        matches.append(matching_entries[0])
    citation_ids = [entry.citation_id for entry in matches]
    if len(set(citation_ids)) != result:
        return {}

    overrides: dict[str, list[str]] = defaultdict(list)
    for entry in matches:
        record = support_by_id.get(entry.chunk_id)
        if record is None or not _metadata_id_is_compatible(record, entry.citation_id):
            return {}
        value = str(entry.citation_id)
        if value not in overrides[entry.chunk_id]:
            overrides[entry.chunk_id].append(value)
    if sum(len(values) for values in overrides.values()) != result:
        return {}
    return {
        chunk_id: tuple(sorted(values, key=int))
        for chunk_id, values in overrides.items()
    }


def _bibliography_entries(records: list[Record]) -> list[BibliographyEntry]:
    numbered: list[BibliographyEntry] = []
    for record in records:
        numbered.extend(_numbered_entries_in_record(record))
    if numbered:
        numbers = [entry.citation_id for entry in numbered]
        if numbers != list(range(1, len(numbers) + 1)):
            return []
        return numbered
    return _unnumbered_entries(records)


def _unnumbered_entries(records: list[Record]) -> list[BibliographyEntry]:
    lines: list[tuple[str, str, str]] = []
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        paper_id = str(record.get("paper_id") or "")
        raw_lines = str(record.get("text") or "").splitlines()
        if raw_lines and _CHUNK_HEADER_RE.match(raw_lines[0]):
            raw_lines = raw_lines[1:]
        lines.extend((line.strip(), chunk_id, paper_id) for line in raw_lines if line.strip())
    if not lines or not _ENTRY_START_RE.match(lines[0][0]):
        return []

    entries: list[BibliographyEntry] = []
    entry_lines: list[str] = []
    entry_chunk_id = ""
    entry_paper_id = ""
    for line, chunk_id, paper_id in lines:
        if _ENTRY_START_RE.match(line):
            if entry_lines:
                entries.append(
                    BibliographyEntry(
                        citation_id=len(entries) + 1,
                        chunk_id=entry_chunk_id,
                        paper_id=entry_paper_id,
                        text=" ".join(entry_lines),
                    )
                )
            entry_lines = [line]
            entry_chunk_id = chunk_id
            entry_paper_id = paper_id
        elif entry_lines:
            entry_lines.append(line)
        else:
            return []
    if entry_lines:
        entries.append(
            BibliographyEntry(
                citation_id=len(entries) + 1,
                chunk_id=entry_chunk_id,
                paper_id=entry_paper_id,
                text=" ".join(entry_lines),
            )
        )
    return entries


def _numbered_entries_in_record(record: Record) -> list[BibliographyEntry]:
    text = str(record.get("text") or "")
    matches = list(_NUMBERED_ENTRY_RE.finditer(text))
    entries: list[BibliographyEntry] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries.append(
            BibliographyEntry(
                citation_id=int(match.group("number")),
                chunk_id=str(record.get("chunk_id") or ""),
                paper_id=str(record.get("paper_id") or ""),
                text=text[match.start() : end],
            )
        )
    return entries


def _contiguous_reference_records(records: list[Record]) -> list[Record]:
    indices = [
        index
        for index, record in enumerate(records)
        if record_source_type(record) == "citation_context"
        and _REFERENCE_SECTION_RE.match(
            str((record.get("metadata") or {}).get("section") or "").strip()
        )
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        return []
    return records[indices[0] : indices[-1] + 1]


def _entry_identity(text: str) -> tuple[str, str] | None:
    first_line = text.splitlines()[0] if text else ""
    surname_match = re.match(r"^\s*([^,\s]+),", first_line)
    if not surname_match:
        return None
    years = _YEAR_RE.findall(text)
    if not years:
        return None
    surname = _normalize(surname_match.group(1)).replace(" ", "")
    # Bibliography entries can mention a conference year in the title/body and
    # then give the publication year at the end.  The final visible year is the
    # stable author/year identity used by Stage 2.
    year = years[-1].casefold()
    return (surname, year) if surname else None


def _contains_author(text: str, author: str) -> bool:
    # Bibliography titles can contain a person's surname (for example,
    # ``Smith, J. A critique of Bonawitz``).  Only the author-list prefix is
    # admissible evidence for an "as an author" query.  Requiring an explicit
    # title boundary keeps this conservative: unfamiliar citation styles are a
    # no-op rather than a title-to-author false positive.
    boundary = _TITLE_AFTER_AUTHORS_RE.search(text)
    if boundary is None:
        return False
    normalized_text = f" {_normalize(text[: boundary.start() + 1])} "
    normalized_author = _normalize(author)
    return bool(normalized_author and f" {normalized_author} " in normalized_text)


def _facts(derivation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(fact)
        for fact in derivation.get("facts") or []
        if isinstance(fact, Mapping)
    ]


def _scalar_value(value: Any) -> str | int | float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value.strip() if isinstance(value, str) else value


def _scalar_leaves(value: Any) -> list[str | int | float]:
    if isinstance(value, Mapping):
        return [
            leaf
            for nested in value.values()
            for leaf in _scalar_leaves(nested)
        ]
    if isinstance(value, list):
        return [leaf for nested in value for leaf in _scalar_leaves(nested)]
    scalar = _scalar_value(value)
    return [scalar] if scalar is not None else []


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"\s*[1-9]\d*\s*", value):
        return int(value)
    return None


def _answer_contains_value(answer: Mapping[str, Any], value: Any) -> bool:
    needle = _normalize(value)
    if not needle:
        return False
    return any(
        needle == leaf or f" {needle} " in f" {leaf} "
        for leaf in _normalized_answer_leaves(answer)
    )


def _normalized_answer_leaves(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            leaf
            for nested in value.values()
            for leaf in _normalized_answer_leaves(nested)
        ]
    if isinstance(value, list):
        return [
            leaf
            for nested in value
            for leaf in _normalized_answer_leaves(nested)
        ]
    if isinstance(value, bool) or value is None:
        return []
    normalized = _normalize(value)
    return [normalized] if normalized else []


def _text_contains_value(text: str, value: Any) -> bool:
    needle = _normalize(value)
    haystack = _normalize(text)
    return bool(needle and (needle == haystack or f" {needle} " in f" {haystack} "))


def _metadata_id_is_compatible(record: Record, citation_id: int) -> bool:
    raw = (record.get("metadata") or {}).get("citation_id")
    if raw in (None, ""):
        return True
    if isinstance(raw, bool):
        return False
    numbers = re.findall(r"\d+", str(raw))
    return len(numbers) == 1 and int(numbers[0]) == citation_id


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _unique_records_or_none(records: Iterable[Record]) -> list[Record] | None:
    output: list[Record] = []
    seen: dict[str, Record] = {}
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        if not chunk_id:
            return None
        previous = seen.get(chunk_id)
        if previous is None:
            seen[chunk_id] = record
            output.append(record)
        elif previous != record:
            return None
    return output


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))
