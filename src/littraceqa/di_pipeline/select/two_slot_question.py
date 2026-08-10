"""Parse supported two-target questions into evidence requirements."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from littraceqa.di_pipeline.retrieve.method_aliases import method_aliases_equal

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


@dataclass(frozen=True)
class EvidenceSlot:
    """One named target and the local evidence it must satisfy."""

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


def evidence_terms(text: str) -> tuple[str, ...]:
    """Return normalized evidence terms in first-seen order."""

    normalized = re.sub(
        r"(?<![A-Za-z])(?:[A-Z]\s+){1,5}[A-Z](?![A-Za-z])",
        lambda match: re.sub(r"\s+", "", match.group()),
        _normalize(text),
    )
    values = (
        _term(token)
        for token in re.findall(r"[A-Za-z]+|[0-9]+", normalized)
    )
    return tuple(
        dict.fromkeys(value for value in values if value not in _STOP_WORDS)
    )


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
        terms=evidence_terms(f"{metric} {qualifier} {condition}"),
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
        key=lambda value: len(evidence_terms(value)),
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
    requirements = evidence_terms(f"{match.group('property')} {match.group('context')}")
    local_terms = tuple(
        term
        for term in evidence_terms(match.group("property"))
        if term != "normalization"
    )
    return (
        EvidenceSlot(target1, requirements, "text", local_terms),
        EvidenceSlot(target2, requirements, "text", local_terms),
    )


def parse_two_slot_question(
    question: object,
) -> tuple[EvidenceSlot, EvidenceSlot] | None:
    """Parse only the two question forms supported by the verifier."""

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


__all__ = ["EvidenceSlot", "evidence_terms", "parse_two_slot_question"]
