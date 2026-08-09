"""Read an explicitly stated paper count from a question.

The rules use question wording only. They do not depend on validation-only
fields or infer a larger set when the wording is ambiguous.
"""

from __future__ import annotations

import re
import unicodedata

MAX_EXPECTED_PAPERS = 10

_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

# "the two ICCV 2025 papers", "these three works", "two NAACL 2025 studies".
# The noun has to follow within a few words so that a count belonging to
# something else ("two-step FID", "three seeds") cannot be picked up.
#
# Only spelled-out numbers count. A digit in "Figure 4 of the DynaPipe paper"
# belongs to the figure, not to a set of four papers, and no amount of context
# within the window separates the two reliably.
_COUNTED_PAPERS_RE = re.compile(
    r"\b(?:the|these|those|all)?\s*"
    r"(two|three|four|five|six|seven|eight|nine)\s+"
    r"(?:[A-Za-z0-9()/'’\-]+\s+){0,4}?"
    r"(papers?|works?|studies|articles|publications|submissions)\b",
    re.IGNORECASE,
)

# "... paper ... and the ... paper ...": each mention of a paper-like noun is
# one referenced work when the question joins them explicitly.
_PAPER_NOUN_RE = re.compile(
    r"\b(paper|work|study|article|publication)\b",
    re.IGNORECASE,
)

# An enumeration of named systems: "for TCM, sCT, ECM-XL (100k), and IMM".
# Requires the Oxford-style tail so that ordinary comma lists of words do not
# match.
# Decimals inside an item ("ECM-XL (with 102.4M training budget)") must not end
# the enumeration, so the body stops at a sentence break rather than any period.
_ENUMERATION_RE = re.compile(
    r"(?:\bfor\b|\bof\b|\bamong\b|\bbetween\b|\bcompare\b|\bacross\b)\s+"
    r"(?P<body>[^?;:]{6,240}?\b(?:and|versus|vs\.?)\s+[^?;:,]{1,60})",
    re.IGNORECASE,
)
_SENTENCE_BREAK_RE = re.compile(r"\.\s+[A-Z]")

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")


def _normalize(question: object) -> str:
    if not isinstance(question, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", question).split())


def _looks_like_system_name(token: str) -> bool:
    """Accept the shapes a method or model name takes, reject ordinary words."""

    if len(token) < 2 or token.isdigit():
        return False
    letters = [character for character in token if character.isalpha()]
    if not letters:
        return False
    has_internal_capital = bool(re.search(r"[a-z0-9][A-Z]", token))
    all_capitals = all(character.isupper() for character in letters)
    has_digit = any(character.isdigit() for character in token)
    return has_internal_capital or all_capitals or has_digit or "-" in token


def _enumerated_system_count(question: str) -> int:
    """Count distinct named systems in an explicit "A, B and C" enumeration."""

    best = 0
    for match in _ENUMERATION_RE.finditer(question):
        body = match.group("body")
        sentence_break = _SENTENCE_BREAK_RE.search(body)
        if sentence_break is not None:
            body = body[: sentence_break.start()]
        # Split on the separators an enumeration actually uses, so that a
        # descriptive phrase inside one item stays with that item.
        parts = [
            part.strip()
            for part in re.split(r",|\band\b|\bversus\b|\bvs\.?\b", body)
            if part.strip()
        ]
        names: set[str] = set()
        for part in parts:
            tokens = [
                token
                for token in _IDENTIFIER_RE.findall(part)
                if _looks_like_system_name(token)
            ]
            if tokens:
                names.add(tokens[0].casefold())
        if len(names) >= 2:
            best = max(best, len(names))
    return best


def expected_paper_count(question: object, default: int = 1) -> int:
    """Return how many papers the question says it is about.

    ``default`` is returned when the wording states nothing, which is the
    common case and the one where a single paper is the safest submission.
    Pass ``0`` to tell "states nothing" apart from "states one".
    """

    if isinstance(default, bool) or not isinstance(default, int):
        raise TypeError("default must be an integer")
    if default < 0:
        raise ValueError("default must be non-negative")

    text = _normalize(question)
    if not text:
        return default

    # The three signals are tried in order of how directly they state a count,
    # and the first one that fires wins. Combining them by taking the largest
    # over-counts every question that states its size and then goes on to name
    # things: "For these two ICCV 2025 papers, report the speedups for the
    # zero-shot key-pruning method and ..." states two and names three.
    counted = [
        value
        for match in _COUNTED_PAPERS_RE.finditer(text)
        if (value := _NUMBER_WORDS.get(match.group(1).lower())) is not None
        and 2 <= value <= MAX_EXPECTED_PAPERS
    ]
    if counted:
        return max(counted)

    # "the X paper ... and the Y paper ..." names one work per paper noun.
    paper_nouns = len(_PAPER_NOUN_RE.findall(text))
    if paper_nouns >= 2 and re.search(r"\band\b", text, re.IGNORECASE):
        return min(paper_nouns, MAX_EXPECTED_PAPERS)

    enumerated = _enumerated_system_count(text)
    if enumerated >= 2:
        return min(enumerated, MAX_EXPECTED_PAPERS)

    return default


def is_open_ended_enumeration(question: object) -> bool:
    """Detect "which papers ..." questions, whose answer set has no stated size."""

    text = _normalize(question)
    if not text:
        return False
    return bool(
        re.search(r"\bwhich\b[^?]{0,200}\b(papers|works|studies)\b", text, re.IGNORECASE)
        or re.search(
            r"\b(?:among|across)\b[^?]{0,240}\b(?:papers|works|studies)\b[^?]{0,240}"
            r"\beach\b",
            text,
            re.IGNORECASE,
        )
    )
